#!/usr/bin/env python3
"""Container smoke test: run the Prometheus image the way docs/guide/docker.md says, and check what it promises.

    python packaging/docker/smoke_image.py IMAGE [--platform linux/arm64] [--quick] [--expect-version X.Y.Z]

Run by CI on a native amd64 runner and a native arm64 runner (``container`` in ci.yml), and by the release
before anything is published (``image`` in release.yml). It needs a Docker engine and nothing else: stdlib
only, no import from src/prometheus, so it judges the image and not the checkout.

  S1  image config     user 10001, the healthcheck, both volumes, both ports, tini, no secret-shaped ENV
  S2  inside           runs as 10001, cannot write its own code, `oara --version`
  --quick stops here (the release's arm64 leg runs under emulation; CI runs the rest natively on arm64)
  S3  setup mode       fresh volumes, no config: healthy, /api/setup/status says setup mode, the banner's
                       pairing code pairs, and the token it hands out is persisted on the home volume
  S4  CLI setup        `oara setup --noninteractive --probe-url` against the FIRSTLIGHT stub model, in its
                       own container, writes the config onto the home volume
  S5  configured       healthy, /health ok, the minted token from `oara token show` opens /api/status and
                       nothing else does, one REST agent turn completes a model->tool->model round trip, the
                       token survives a restart, `docker stop` is a clean SIGTERM, state is on both volumes
  S6  env token        PROMETHEUS_API_TOKEN replaces the minted token
  S7  secret token     PROMETHEUS_API_TOKEN_FILE (a read-only mounted secret) replaces it too
  S8  broken secret    a PROMETHEUS_API_TOKEN_FILE that names nothing stops the container, with the reason

Every token here is minted for the run and never printed. Containers, volumes and the network are removed
at the end whatever happened, by the names this run gave them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STUB = REPO / "scripts" / "firstlight_stub_model.py"
FINAL_MARKER = "FIRSTLIGHT-COMPLETE"
UID = "10001"
HOME = "/home/prometheus"
DATA = f"{HOME}/.prometheus/data"
TOKEN_FILE_VAR = "PROMETHEUS_API_TOKEN_FILE"
TOKEN_VAR = "PROMETHEUS_API_TOKEN"
SECRET_NAME = re.compile(r"(?:^|_)(?:TOKENS?|KEY|SECRET|PASSWORD|PASSWD|CREDENTIALS?)(?:$|_)", re.I)
#: Secret-shaped names that are not secrets. The python base image sets GPG_KEY to the PUBLIC fingerprint of
#: the key its CPython tarball was verified against; an ENV line cannot be removed by an image built on it.
PUBLIC_BASE_ENV = frozenset({"GPG_KEY"})

# Never through a proxy: every request here is to a port this run published on 127.0.0.1.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class SmokeFailure(Exception):
    pass


class Run:
    def __init__(self, image: str, platform: str | None) -> None:
        self.image = image
        self.platform = platform
        self.tag = f"prom-smoke-{os.getpid()}-{secrets.token_hex(3)}"
        self.containers: list[str] = []
        self.volumes: list[str] = []
        self.network: str | None = None
        self.tmp = Path(tempfile.mkdtemp(prefix="prom-smoke-"))

    # ── docker plumbing ─────────────────────────────────────────────────────

    def docker(self, *args: str, check: bool = True, timeout: float = 300, stdin: str | None = None) -> subprocess.CompletedProcess:
        proc = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, input=stdin)
        if check and proc.returncode != 0:
            raise SmokeFailure(f"docker {' '.join(args[:3])}… exited {proc.returncode}: {proc.stderr.strip()[-600:]}")
        return proc

    def _platform(self) -> list[str]:
        return ["--platform", self.platform] if self.platform else []

    def volume(self, label: str) -> str:
        name = f"{self.tag}-{label}"
        self.docker("volume", "create", name)
        self.volumes.append(name)
        return name

    def net(self) -> str:
        if self.network is None:
            self.network = self.tag
            self.docker("network", "create", self.network)
        return self.network

    def run_once(self, *args: str, timeout: float = 300, stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
        """A throwaway container (``--rm``) running one command."""
        return self.docker("run", "--rm", *(["-i"] if stdin is not None else []), *self._platform(),
                           *args, timeout=timeout, stdin=stdin, check=check)

    def start(self, label: str, *args: str, cmd: tuple[str, ...] = (), healthcheck: bool = True) -> str:
        name = f"{self.tag}-{label}"
        self.containers.append(name)
        self.docker("run", "-d", "--name", name, *self._platform(),
                    *([] if healthcheck else ["--no-healthcheck"]), *args, self.image, *cmd)
        return name

    def host_port(self, container: str, port: int) -> int:
        out = self.docker("port", container, f"{port}/tcp").stdout.strip().splitlines()
        for line in out:
            if line.startswith("127.0.0.1:"):
                return int(line.rsplit(":", 1)[1])
        raise SmokeFailure(f"{container}: {port}/tcp is not published on 127.0.0.1 ({out})")

    def state(self, container: str) -> dict:
        return json.loads(self.docker("inspect", "--format", "{{json .State}}", container).stdout)

    def logs(self, container: str) -> str:
        proc = self.docker("logs", container, check=False)
        return proc.stdout + proc.stderr

    def wait_healthy(self, container: str, timeout: float = 240) -> None:
        deadline = time.time() + timeout
        status = "?"
        while time.time() < deadline:
            st = self.state(container)
            if not st.get("Running"):
                raise SmokeFailure(f"{container} stopped (exit {st.get('ExitCode')}) before it was healthy:\n"
                                   f"{self.logs(container)[-2500:]}")
            status = (st.get("Health") or {}).get("Status", "none")
            if status == "healthy":
                return
            time.sleep(2)
        health = self.state(container).get("Health") or {}
        last = (health.get("Log") or [{}])[-1].get("Output", "").strip()
        raise SmokeFailure(f"{container} not healthy after {timeout:.0f}s (status {status}; last probe: {last}):\n"
                           f"{self.logs(container)[-2500:]}")

    def cleanup(self) -> None:
        for name in self.containers:
            self.docker("rm", "-f", "-v", name, check=False, timeout=60)
        for name in self.volumes:
            self.docker("volume", "rm", "-f", name, check=False, timeout=60)
        if self.network:
            self.docker("network", "rm", self.network, check=False, timeout=60)
        for p in sorted(self.tmp.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        self.tmp.rmdir()


# ── http ─────────────────────────────────────────────────────────────────────


def http(method: str, url: str, token: str | None = None, body: dict | None = None, timeout: float = 10) -> tuple[int, str]:
    headers = {"Content-Type": "application/json"} if body is not None else {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, method=method, headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def expect(cond: bool, message: str) -> None:
    if not cond:
        raise SmokeFailure(message)


def _mounts(home: str, data: str) -> list[str]:
    return ["-v", f"{home}:{HOME}", "-v", f"{data}:{DATA}"]


def _published() -> list[str]:
    return ["-p", "127.0.0.1::8005", "-p", "127.0.0.1::8010"]


def _status_code(base: str, token: str | None) -> int:
    return http("GET", f"{base}/api/status", token=token)[0]


def _wait_status(base: str, token: str, timeout: float = 120) -> int:
    """/api/status with *token*, retried while the daemon is still coming up (connection refused or reset)."""
    deadline = time.time() + timeout
    while True:
        try:
            code = _status_code(base, token)
            if code != 503:
                return code
        except OSError:
            pass
        if time.time() > deadline:
            return -1
        time.sleep(2)


# ── steps ────────────────────────────────────────────────────────────────────


def s1_image_config(run: Run, expect_arch: str | None) -> str:
    image = json.loads(run.docker("image", "inspect", run.image, "--format", "{{json .}}").stdout)
    cfg = image["Config"]
    expect(cfg.get("User") == f"{UID}:{UID}", f"User is {cfg.get('User')!r}, not {UID}:{UID}")
    test = " ".join((cfg.get("Healthcheck") or {}).get("Test") or [])
    expect("prometheus.web.healthcheck" in test, f"no daemon healthcheck: {test!r}")
    expect({HOME, DATA} <= set(cfg.get("Volumes") or {}), f"volumes: {sorted(cfg.get('Volumes') or {})}")
    expect({"8005/tcp", "8010/tcp"} <= set(cfg.get("ExposedPorts") or {}), f"ports: {cfg.get('ExposedPorts')}")
    expect(cfg.get("Entrypoint") == ["/usr/bin/tini", "--"], f"entrypoint: {cfg.get('Entrypoint')}")
    expect(cfg.get("Cmd") == ["oara", "daemon"], f"cmd: {cfg.get('Cmd')}")
    names = [e.split("=", 1)[0] for e in cfg.get("Env") or []]
    leaked = [n for n in names if SECRET_NAME.search(n) and n not in PUBLIC_BASE_ENV]
    expect(not leaked, f"secret-shaped variables baked into the image: {leaked}")
    if expect_arch:
        expect(image.get("Architecture") == expect_arch, f"architecture {image.get('Architecture')}, wanted {expect_arch}")
    return f"user {cfg['User']}, healthcheck, 2 volumes, 8005+8010, tini, no secret ENV, {image.get('Architecture')}"


def s2_inside(run: Run, expect_version: str | None) -> str:
    uid = run.run_once(run.image, "id", "-u").stdout.strip()
    expect(uid == UID, f"runs as uid {uid}")
    probe = run.run_once(run.image, "sh", "-c", "touch /opt/prometheus/.w 2>/dev/null && echo writable || echo readonly")
    expect(probe.stdout.strip() == "readonly", "the agent's user can write the code it runs (/opt/prometheus)")
    version = run.run_once(run.image, "oara", "--version").stdout.strip()
    expect(version.startswith("◉ prometheus "), f"oara --version printed {version!r}")
    if expect_version:
        expect(version == f"◉ prometheus {expect_version} · OAra Labs", f"{version!r}, wanted ◉ prometheus {expect_version} · OAra Labs")
    return f"uid {uid}, /opt/prometheus read-only, {version}"


def s3_setup_mode(run: Run) -> str:
    home, data = run.volume("setup-home"), run.volume("setup-data")
    c = run.start("setup", *_published(), *_mounts(home, data))
    run.wait_healthy(c)
    base = f"http://127.0.0.1:{run.host_port(c, 8005)}"
    code, body = http("GET", f"{base}/api/setup/status")
    expect(code == 200 and json.loads(body).get("setup_mode") is True, f"/api/setup/status -> {code} {body[:200]}")
    code, body = http("GET", f"{base}/health")
    expect(code == 403 and '"setup_mode"' in body, f"setup mode should answer /health from its catch-all (403 setup_mode): {code} {body[:200]}")
    m = re.search(r"Pairing code[^\n]*\n\s*\n\s*(\d{6})\s*$", run.logs(c), re.M)
    expect(m is not None, f"no pairing code in the setup-mode banner:\n{run.logs(c)[-1500:]}")
    code, body = http("POST", f"{base}/api/setup/pair", body={"code": m.group(1)})
    expect(code == 200, f"pairing with the banner's code -> {code} {body[:200]}")
    token = json.loads(body).get("token") or ""
    expect(len(token) >= 32, "pairing returned no token")
    saved = run.docker("exec", "-i", c, "sh", "-c", f'grep -qxF "{TOKEN_VAR}=$(cat)" "$HOME/.config/prometheus/env"',
                       stdin=token, check=False)
    expect(saved.returncode == 0, "the paired token is not in ~/.config/prometheus/env on the home volume")
    run.docker("stop", "-t", "30", c)
    return "healthy while unpaired; the banner's code paired; token persisted on the home volume"


def s4_cli_setup(run: Run) -> tuple[str, str, str]:
    net = run.net()
    stub = run.start("stub", "--network", net, "--network-alias", "stub",
                     "-v", f"{STUB}:/stub/firstlight_stub_model.py:ro", healthcheck=False,
                     cmd=("python", "/stub/firstlight_stub_model.py", "--host", "0.0.0.0", "--port", "8080"))
    deadline = time.time() + 60
    while "listening on" not in run.logs(stub):
        expect(time.time() < deadline, f"the stub model never started:\n{run.logs(stub)[-1500:]}")
        time.sleep(1)
    home, data = run.volume("home"), run.volume("data")
    proc = run.run_once("--network", net, *_mounts(home, data), run.image,
                        "oara", "setup", "--noninteractive", "--timeout", "3", "--probe-url", "http://stub:8080",
                        check=False)
    expect(proc.returncode == 0, f"oara setup exited {proc.returncode}:\n{(proc.stdout + proc.stderr)[-2500:]}")
    cfg = run.run_once(*_mounts(home, data), run.image, "cat", f"{HOME}/.prometheus/prometheus.yaml",
                       check=False)
    expect(cfg.returncode == 0, "setup wrote no ~/.prometheus/prometheus.yaml on the home volume")
    expect("http://stub:8080" in cfg.stdout, "the config does not point at the probed server")
    return home, data, "config written on the home volume, provider aimed at the stub"


def s5_configured(run: Run, home: str, data: str) -> tuple[str, str]:
    c = run.start("daemon", "--network", run.net(), *_published(), *_mounts(home, data))
    run.wait_healthy(c)
    base = f"http://127.0.0.1:{run.host_port(c, 8005)}"
    code, body = http("GET", f"{base}/health")
    expect(code == 200 and json.loads(body).get("status") == "ok", f"/health -> {code} {body[:200]}")
    shown = run.docker("exec", c, "oara", "token", "show").stdout.splitlines()
    token = shown[0].strip() if shown else ""
    expect(len(token) >= 32 and " " not in token, "`oara token show` printed no token")
    code, body = http("GET", f"{base}/api/status")
    # Beacon recognises a Prometheus by this 401 (both words); a container must answer it the same way.
    expect(code == 401 and "unauthorized" in body.lower() and "bearer" in body.lower(),
           f"/api/status without a token -> {code} {body[:200]}")
    expect(_status_code(base, "not-the-token") == 401, "a wrong token opened /api/status")
    code, body = http("GET", f"{base}/api/status", token=token)
    expect(code == 200, "the minted token does not open /api/status")
    # docs/guide/docker.md says neither bash floor is in force inside a default container (no AppArmor
    # profile can be loaded from in here, no bubblewrap) and that the boot log says so. Hold it to that.
    floors = json.loads(body).get("security") or {}
    read_state = (floors.get("bash_read_floor") or {}).get("state")
    write_state = (floors.get("bash_write_floor") or {}).get("state")
    expect(read_state not in (None, "active") and write_state not in (None, "active"),
           f"the guide says no bash floor is in force in a container; /api/status says read={read_state} "
           f"write={write_state}: update docs/guide/docker.md")
    if read_state == "dark":
        expect("READ floor is NOT in force" in run.logs(c), "the read floor is dark and the boot log never said so")

    code, body = http("POST", f"{base}/api/chat/send", token=token,
                      body={"session_id": "container-smoke", "message": "smoke: run your tool round"})
    expect(code == 200, f"POST /api/chat/send -> {code} {body[:300]}")
    deadline = time.time() + 240
    while True:
        code, body = http("GET", f"{base}/api/sessions/container-smoke/messages", token=token)
        if code == 200 and FINAL_MARKER in body:
            break
        expect(time.time() < deadline, f"no {FINAL_MARKER} in the session after 240s:\n{run.logs(c)[-2500:]}")
        time.sleep(2)

    run.docker("restart", "-t", "30", c)
    time.sleep(3)    # let Docker drop the health state of the run before the restart
    run.wait_healthy(c)
    base = f"http://127.0.0.1:{run.host_port(c, 8005)}"
    expect(_wait_status(base, token) == 200, "the minted token did not survive a restart")

    run.docker("stop", "-t", "30", c)
    exit_code = run.state(c).get("ExitCode")
    expect(exit_code in (0, 143), f"docker stop ended with exit {exit_code} (137 means it ignored SIGTERM)")
    listing = run.run_once(*_mounts(home, data), run.image, "sh", "-c",
                           f"ls -A {DATA}; echo ---; ls -A {HOME}/.prometheus").stdout
    in_data, _, in_home = listing.partition("---")
    expect(in_data.split(), f"nothing on the data volume ({DATA})")
    expect("prometheus.yaml" in in_home.split(), "the config is not on the home volume")
    return token, (f"minted token opens the API and only it; bash floors read={read_state} write={write_state}; "
                   f"agent turn reached {FINAL_MARKER}; token survived a restart; stop exit {exit_code}; "
                   f"data volume: {' '.join(in_data.split()[:6])}")


def s6_env_token(run: Run, home: str, data: str, minted: str) -> str:
    env_token = secrets.token_urlsafe(32)
    c = run.start("envtoken", "--network", run.net(), *_published(), *_mounts(home, data), "-e", f"{TOKEN_VAR}={env_token}")
    run.wait_healthy(c)
    base = f"http://127.0.0.1:{run.host_port(c, 8005)}"
    expect(_status_code(base, env_token) == 200, f"{TOKEN_VAR} does not open the API")
    expect(_status_code(base, minted) == 401, f"the env-file token still opens the API under {TOKEN_VAR}")
    run.docker("stop", "-t", "30", c)
    return f"{TOKEN_VAR} is the token; the env file's is refused"


def s7_secret_token(run: Run, home: str, data: str, minted: str) -> str:
    secret_token = secrets.token_urlsafe(32)
    secret = run.tmp / "prometheus_api_token"
    secret.write_text(secret_token + "\n", encoding="utf-8")
    secret.chmod(0o644)          # the container user (10001) is not the file's owner
    run.tmp.chmod(0o755)
    c = run.start("secrettoken", "--network", run.net(), *_published(), *_mounts(home, data),
                  "-v", f"{secret}:/run/secrets/prometheus_api_token:ro",
                  "-e", f"{TOKEN_FILE_VAR}=/run/secrets/prometheus_api_token")
    run.wait_healthy(c)
    base = f"http://127.0.0.1:{run.host_port(c, 8005)}"
    expect(_status_code(base, secret_token) == 200, f"the {TOKEN_FILE_VAR} secret does not open the API")
    expect(_status_code(base, minted) == 401, "the env-file token still opens the API under a secret")
    env = run.docker("exec", c, "sh", "-c", f'grep -c "^{TOKEN_VAR}=" "$HOME/.config/prometheus/env"', check=False)
    expect(env.stdout.strip() == "1", "the secret's token was copied into the env file")
    run.docker("stop", "-t", "30", c)
    return f"{TOKEN_FILE_VAR} is the token, read from a read-only secret and copied nowhere"


def s8_broken_secret(run: Run) -> str:
    home, data = run.volume("broken-home"), run.volume("broken-data")
    proc = run.docker("run", "--rm", *run._platform(), *_mounts(home, data),
                      "-e", f"{TOKEN_FILE_VAR}=/run/secrets/not-there", run.image, check=False, timeout=120)
    expect(proc.returncode not in (0, 137), f"a missing secret did not stop the daemon (exit {proc.returncode})")
    expect(TOKEN_FILE_VAR in proc.stdout + proc.stderr, "the refusal does not name the variable")
    return f"refused to start (exit {proc.returncode}) and said why"


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("--platform", default=None, help="e.g. linux/arm64 (default: the engine's own)")
    parser.add_argument("--quick", action="store_true", help="S1 and S2 only")
    parser.add_argument("--expect-version", default=None)
    args = parser.parse_args(argv)
    arch = args.platform.split("/")[1] if args.platform else None

    run = Run(args.image, args.platform)
    results: list[tuple[str, bool, str]] = []

    def step(name: str, fn, *a):
        try:
            out = fn(*a)
        except Exception as exc:
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
            raise
        detail = out if isinstance(out, str) else out[-1]
        results.append((name, True, detail))
        return out

    try:
        step("S1 image config", s1_image_config, run, arch)
        step("S2 inside", s2_inside, run, args.expect_version)
        if not args.quick:
            step("S3 setup mode", s3_setup_mode, run)
            home, data, _ = step("S4 CLI setup", s4_cli_setup, run)
            minted, _ = step("S5 configured", s5_configured, run, home, data)
            step("S6 env token", s6_env_token, run, home, data, minted)
            step("S7 secret token", s7_secret_token, run, home, data, minted)
            step("S8 broken secret", s8_broken_secret, run)
    except Exception:  # recorded by step(); the summary below says which and why
        pass
    finally:
        run.cleanup()

    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed ({args.image}"
          f"{', ' + args.platform if args.platform else ''}{', quick' if args.quick else ''})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
