#!/usr/bin/env python3
"""Smoke test for the OpenShell sandbox image (packaging/openshell/Dockerfile) and its policy.

    python packaging/openshell/smoke_openshell.py --docker IMAGE
    python packaging/openshell/smoke_openshell.py --openshell IMAGE --gateway NAME [--openshell-bin PATH]

--docker (CI, both architectures): the image under plain Docker. Its config is the sandbox shape (user
10001, HOME and workdir /sandbox, iproute2), and the daemon runs with all its state under /sandbox:
`oara setup` against the FIRSTLIGHT stub model, then a configured daemon that is healthy, answers only
its token, and completes a model->tool->model turn.

--openshell (a Linux host with an OpenShell gateway running; see docs/guide/openshell.md): the same image
in a real sandbox under packaging/openshell/policy.yaml, plus a rule for the stub's port. It checks:
  * the daemon sets itself up and runs inside the sandbox, reaching a model server on the host as
    host.openshell.internal;
  * `openshell forward` brings the API (8005) and the WebSocket bridge (8010) to the host's loopback;
  * the token, the floors, and an agent turn, through the forward;
  * egress: the allowed gateway hosts answer the daemon's interpreter (Telegram and Slack over HTTPS,
    Discord over HTTPS and its gateway WebSocket), a host the policy does not list is refused, and a
    binary the policy does not name (curl) is refused even for an allowed host;
  * the filesystem: /sandbox is writable, the code and the rest of the image are not.

Stdlib only, no import from src/prometheus. Everything this run creates (the sandbox, forwards, the stub
container, the anonymous volumes the image's VOLUME lines produce) is removed at the end by the names it
recorded; nothing else is touched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STUB = REPO / "scripts" / "firstlight_stub_model.py"
POLICY = REPO / "packaging" / "openshell" / "policy.yaml"
FINAL_MARKER = "FIRSTLIGHT-COMPLETE"
UID = "10001"
# Absolute, so nothing depends on the PATH a sandbox exec builds.
OARA = "/opt/prometheus/bin/oara"
PYTHON = "/opt/prometheus/bin/python"
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# Run inside the sandbox by the daemon's own interpreter: GET each URL, report status and the start of the body.
# Redirects are not followed: a redirect to a host the policy does not list would read as the first host refused.
_HTTP_PROBE = r"""
import json, sys, urllib.request, urllib.error
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None
opener = urllib.request.build_opener(NoRedirect)
def get(u):
    # Discord's edge refuses library default user agents (Cloudflare 1010); discord.py sends its own like this.
    ua = "DiscordBot (https://oara.ai, 1.0)" if "discord.com" in u else "prometheus-smoke"
    try:
        with opener.open(urllib.request.Request(u, headers={"User-Agent": ua}), timeout=20) as r:
            return [r.status, r.read(400).decode("utf-8", "replace")]
    except urllib.error.HTTPError as e:
        return [e.code, e.read(400).decode("utf-8", "replace")]
    except Exception as e:
        return [-1, f"{type(e).__name__}: {e}"]
print(json.dumps({u: get(u) for u in sys.argv[1:]}))
"""

# Discord's gateway sends HELLO (op 10) to any client that connects; no token is involved.
_WS_PROBE = r"""
import asyncio, json, websockets
async def main():
    async with websockets.connect("wss://gateway.discord.gg/?v=10&encoding=json", open_timeout=20) as ws:
        print(json.loads(await asyncio.wait_for(ws.recv(), 20)).get("op"))
asyncio.run(main())
"""


class SmokeFailure(Exception):
    pass


def expect(cond: bool, message: str) -> None:
    if not cond:
        raise SmokeFailure(message)


def run(argv: list[str], *, check: bool = True, timeout: float = 300, stdin: str | None = None,
        env: dict | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, input=stdin, env=env)
    if check and proc.returncode != 0:
        raise SmokeFailure(f"{' '.join(argv[:4])}… exited {proc.returncode}: {(proc.stderr or proc.stdout).strip()[-800:]}")
    return proc


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


def wait_http(url: str, want: int = 200, timeout: float = 180) -> tuple[int, str]:
    deadline = time.time() + timeout
    last: tuple[int, str] = (-1, "")
    while time.time() < deadline:
        try:
            last = http("GET", url)
            if last[0] == want:
                return last
        except OSError as exc:
            last = (-1, str(exc))
        time.sleep(2)
    raise SmokeFailure(f"{url} never answered {want} (last: {last[0]} {last[1][:200]})")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def agent_turn(base: str, token: str, session: str) -> None:
    code, body = http("POST", f"{base}/api/chat/send", token=token,
                      body={"session_id": session, "message": "smoke: run your tool round"})
    expect(code == 200, f"POST /api/chat/send -> {code} {body[:300]}")
    deadline = time.time() + 240
    while True:
        code, body = http("GET", f"{base}/api/sessions/{session}/messages", token=token)
        if code == 200 and FINAL_MARKER in body:
            return
        expect(time.time() < deadline, f"no {FINAL_MARKER} in the session after 240s")
        time.sleep(2)


def check_api(base: str, token: str) -> str:
    code, body = http("GET", f"{base}/health")
    expect(code == 200 and json.loads(body).get("status") == "ok", f"/health -> {code} {body[:200]}")
    code, body = http("GET", f"{base}/api/status")
    expect(code == 401, f"/api/status without a token -> {code}")
    code, body = http("GET", f"{base}/api/status", token=token)
    expect(code == 200, f"/api/status with the token -> {code} {body[:200]}")
    floors = json.loads(body).get("security") or {}
    return "read={} write={}".format((floors.get("bash_read_floor") or {}).get("state"),
                                     (floors.get("bash_write_floor") or {}).get("state"))


class Resources:
    """Everything this run creates, removed by name at the end."""

    def __init__(self) -> None:
        self.tag = f"prom-osh-{os.getpid()}-{secrets.token_hex(3)}"
        self.containers: list[str] = []
        self.networks: list[str] = []
        self.cleanups: list[list[str]] = []
        self.volumes_before = set(run(["docker", "volume", "ls", "-q"]).stdout.split())
        self.tmp = Path(tempfile.mkdtemp(prefix="prom-osh-"))

    def cleanup(self) -> list[str]:
        notes = []
        for argv in reversed(self.cleanups):
            run(argv, check=False, timeout=120)
        for name in self.containers:
            run(["docker", "rm", "-f", "-v", name], check=False, timeout=60)
        for name in self.networks:
            run(["docker", "network", "rm", name], check=False, timeout=60)
        # Anonymous volumes this run produced (the image's VOLUME lines; a sandbox delete leaves them).
        # Only names that did not exist before, and only anonymous (64 hex) ones.
        time.sleep(3)
        after = set(run(["docker", "volume", "ls", "-q"], check=False).stdout.split())
        mine = sorted(v for v in after - self.volumes_before if re.fullmatch(r"[0-9a-f]{64}", v))
        for v in mine:
            run(["docker", "volume", "rm", v], check=False, timeout=60)
        if mine:
            notes.append(f"removed {len(mine)} anonymous volume(s) this run created")
        left = sorted((set(run(["docker", "volume", "ls", "-q"], check=False).stdout.split())
                       - self.volumes_before))
        if left:
            notes.append(f"volumes still here that did not exist before: {left}")
        for p in sorted(self.tmp.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        self.tmp.rmdir()
        return notes


# ── --docker ─────────────────────────────────────────────────────────────────


def docker_mode(image: str, res: Resources, results: list) -> None:
    def step(name, fn):
        out = fn()
        results.append((name, True, out))

    def d1():
        cfg = json.loads(run(["docker", "image", "inspect", image, "--format", "{{json .Config}}"]).stdout)
        expect(cfg.get("User") == f"{UID}:{UID}", f"User {cfg.get('User')!r}")
        expect("HOME=/sandbox" in (cfg.get("Env") or []), "HOME is not /sandbox")
        expect(cfg.get("WorkingDir") == "/sandbox", f"WorkingDir {cfg.get('WorkingDir')!r}")
        probe = run(["docker", "run", "--rm", image, "sh", "-c",
                     "ip -V >/dev/null && touch /sandbox/w && rm /sandbox/w && "
                     "(touch /opt/prometheus/w 2>/dev/null && echo code-writable || echo ok)"]).stdout.strip()
        expect(probe == "ok", f"in the container: {probe!r}")
        return "user 10001, HOME and workdir /sandbox, iproute2, /sandbox writable, code read-only"

    def d2():
        net = res.tag
        run(["docker", "network", "create", net])
        res.networks.append(net)
        stub = f"{res.tag}-stub"
        res.containers.append(stub)
        run(["docker", "run", "-d", "--name", stub, "--no-healthcheck", "--network", net, "--network-alias", "stub",
             "-v", f"{STUB}:/stub/stub.py:ro", image, "python", "/stub/stub.py", "--host", "0.0.0.0", "--port", "8080"])
        name = f"{res.tag}-daemon"
        res.containers.append(name)
        # The start command OpenShell is given after `--`, verbatim.
        run(["docker", "run", "-d", "--name", name, "--network", net, "-p", "127.0.0.1::8005", image, "sh", "-c",
             "oara setup --noninteractive --timeout 3 --probe-url http://stub:8080 && exec oara daemon"])
        port = run(["docker", "port", name, "8005/tcp"]).stdout.strip().splitlines()[0].rsplit(":", 1)[1]
        base = f"http://127.0.0.1:{port}"
        wait_http(f"{base}/health")
        token = run(["docker", "exec", name, "oara", "token", "show"]).stdout.splitlines()[0].strip()
        floors = check_api(base, token)
        agent_turn(base, token, "sandbox-docker")
        where = run(["docker", "exec", name, "sh", "-c",
                     "test -f /sandbox/.prometheus/prometheus.yaml && test -f /sandbox/.config/prometheus/env "
                     "&& test -d /sandbox/.prometheus/data && echo ok"], check=False).stdout.strip()
        expect(where == "ok", "the daemon's state is not under /sandbox")
        return f"setup, healthy, token-only API, agent turn reached {FINAL_MARKER}; floors {floors}; state under /sandbox"

    step("D1 sandbox image shape", d1)
    step("D2 daemon with HOME=/sandbox", d2)


# ── --openshell ──────────────────────────────────────────────────────────────


def openshell_mode(image: str, gateway: str, osh: str, res: Resources, results: list) -> None:
    sb = f"prom-{secrets.token_hex(3)}"  # a sandbox name is at most 19 characters

    def o(*args: str, **kw) -> subprocess.CompletedProcess:
        return run([osh, "--gateway", gateway, *args], **kw)

    def sandbox_exec(*argv: str, check: bool = True, timeout: float = 120) -> subprocess.CompletedProcess:
        return o("sandbox", "exec", "-n", sb, "--", *argv, check=check, timeout=timeout)

    def step(name, fn):
        out = fn()
        results.append((name, True, out))

    versions = {}

    def o0():
        versions["cli"] = run([osh, "--version"]).stdout.strip()
        for p in (8005, 8010):
            with socket.socket() as s:
                expect(s.connect_ex(("127.0.0.1", p)) != 0, f"127.0.0.1:{p} is already in use on this host")
        return versions["cli"]

    stub_port = free_port()

    def o1():
        stub = f"{res.tag}-stub"
        res.containers.append(stub)
        run(["docker", "run", "-d", "--name", stub, "--no-healthcheck", "-p", f"127.0.0.1:{stub_port}:8080",
             "-v", f"{STUB}:/stub/stub.py:ro", image, "python", "/stub/stub.py", "--host", "0.0.0.0", "--port", "8080"])
        wait_http(f"http://127.0.0.1:{stub_port}/health", timeout=60)
        # The shipped policy plus one endpoint for the stub, beside the Ollama one it copies.
        text = POLICY.read_text(encoding="utf-8")
        # The endpoint line itself, not the comment above it that names the same host and port.
        line = next(ln for ln in text.splitlines()
                    if "host: host.openshell.internal" in ln and "port: 11434" in ln and not ln.lstrip().startswith("#"))
        text = text.replace(line, line + "\n" + line.replace("11434", str(stub_port)).split("#")[0].rstrip(), 1)
        expect(f"port: {stub_port}," in text, "the stub's port did not make it into the test policy")
        (res.tmp / "policy.yaml").write_text(text, encoding="utf-8")
        return f"stub model on the host at 127.0.0.1:{stub_port}; policy = shipped + that port"

    def o2():
        res.cleanups.append([osh, "--gateway", gateway, "sandbox", "delete", sb])
        # docs/guide/openshell.md's start command, verbatim apart from the absolute paths: set up only when there
        # is no config yet (so a restart keeps it), then run the daemon.
        start = (f'test -f "$HOME/.prometheus/prometheus.yaml" || {OARA} setup --noninteractive --timeout 3 '
                 f'--probe-url http://host.openshell.internal:{stub_port}; exec {OARA} daemon')
        versions["start"] = start
        o("sandbox", "create", "--name", sb, "--from", image, "--policy", str(res.tmp / "policy.yaml"), "--detach",
          "--restart-policy", "on-failure", "--", "/bin/sh", "-c", start, timeout=1800)
        wait_daemon()
        return "`oara setup` reached the host's model server as host.openshell.internal; the daemon is healthy"

    def wait_daemon(timeout: float = 600) -> None:
        deadline = time.time() + timeout
        while True:
            probe = sandbox_exec(PYTHON, "-m", "prometheus.web.healthcheck", check=False, timeout=60)
            if probe.returncode == 0 and "daemon answering" in probe.stdout:
                return
            expect(time.time() < deadline, f"the daemon never became healthy in the sandbox: "
                                           f"{(probe.stdout + probe.stderr)[-600:]}")
            time.sleep(5)

    def o3():
        for port in (8005, 8010):
            res.cleanups.append([osh, "--gateway", gateway, "forward", "stop", str(port), sb])
            # To a file, never a pipe: `-d` leaves an ssh child running that inherits the parent's stdout,
            # so a captured pipe never closes and the call would wait for the forward to end.
            log = res.tmp / f"forward-{port}.log"
            with open(log, "w", encoding="utf-8") as fh:
                proc = subprocess.run([osh, "--gateway", gateway, "forward", "start", str(port), sb, "-d"],
                                      stdout=fh, stderr=subprocess.STDOUT, timeout=120)
            expect(proc.returncode == 0, f"forward {port}: {log.read_text(encoding='utf-8')[-400:]}")
        wait_http("http://127.0.0.1:8005/health", timeout=120)
        with socket.create_connection(("127.0.0.1", 8010), timeout=10):
            pass
        token = sandbox_exec(OARA, "token", "show").stdout.splitlines()[0].strip()
        floors = check_api("http://127.0.0.1:8005", token)
        versions["token"] = token
        return f"forwards 8005 and 8010 on 127.0.0.1; token-only API; floors {floors}"

    def o4():
        agent_turn("http://127.0.0.1:8005", versions["token"], "sandbox-openshell")
        return f"a REST agent turn through the forward reached {FINAL_MARKER} (model on the host)"

    def o5():
        # Each answers directly, with no token: Telegram's Bot API refuses the bot (401/404 JSON), Slack's api.test
        # says ok, Discord names its gateway.
        allowed = ["https://api.telegram.org/bot0/getMe", "https://slack.com/api/api.test",
                   "https://discord.com/api/v10/gateway"]
        denied = ["https://example.com/"]
        res_http = json.loads(sandbox_exec(PYTHON, "-c", _HTTP_PROBE, *allowed, *denied, timeout=180).stdout.strip().splitlines()[-1])

        def refused(code_body) -> bool:
            code, body = code_body
            return code == -1 or (code == 403 and "policy" in body.lower())

        for u in allowed:
            expect(not refused(res_http[u]), f"{u} was refused to the daemon's interpreter: {res_http[u]}")
        expect(refused(res_http["https://example.com/"]), f"a host the policy does not list answered: {res_http['https://example.com/']}")
        expect('"ok":true' in res_http["https://slack.com/api/api.test"][1].replace(" ", ""), "Slack api.test did not answer ok")
        expect('"ok":false' in res_http[allowed[0]][1].replace(" ", ""), f"Telegram's Bot API did not answer: {res_http[allowed[0]]}")
        expect(res_http[allowed[2]][0] == 200 and "gateway.discord.gg" in res_http[allowed[2]][1],
               f"Discord's REST API did not name its gateway: {res_http[allowed[2]]}")
        op = sandbox_exec(PYTHON, "-c", _WS_PROBE, check=False, timeout=120)
        expect(op.stdout.strip().splitlines()[-1:] == ["10"], f"Discord's gateway WebSocket sent no HELLO: {(op.stdout + op.stderr)[-400:]}")
        curl = sandbox_exec("/usr/bin/curl", "-sS", "-m", "20", "-o", "/dev/null", "-w", "%{http_code}",
                            "https://api.telegram.org/bot0/getMe",
                            check=False, timeout=60)
        expect(curl.returncode != 0 or curl.stdout.strip() in ("000", "403"),
               f"curl, which no rule names, reached an allowed host: rc {curl.returncode} {curl.stdout!r}")
        return ("python reached Telegram {}, Slack api.test ok, Discord REST {} and its gateway WebSocket (HELLO); "
                "example.com refused; curl refused".format(res_http[allowed[0]][0], res_http[allowed[2]][0]))

    def o6():
        out = sandbox_exec("/bin/sh", "-c", "touch /sandbox/w && rm /sandbox/w && echo sandbox-ok; "
                                       "touch /opt/prometheus/w 2>/dev/null && echo CODE-WRITABLE; "
                                       "touch /home/prometheus/w 2>/dev/null && echo HOME-WRITABLE; "
                                       "touch /etc/w 2>/dev/null && echo ETC-WRITABLE; true").stdout.split()
        expect(out == ["sandbox-ok"], f"filesystem: {out}")
        return "/sandbox writable; /opt/prometheus, /home/prometheus and /etc are not"

    def o7():
        # Stop and start the sandbox: the start command runs again, finds the config, and does not redo setup;
        # the token and the memory of the earlier turn are still there.
        before = sandbox_exec("/bin/sh", "-c", "ls $HOME/.prometheus/prometheus.yaml* | wc -l").stdout.strip()
        o("sandbox", "stop", sb, timeout=300)
        o("sandbox", "start", sb, timeout=600)
        wait_daemon()
        after = sandbox_exec("/bin/sh", "-c", "ls $HOME/.prometheus/prometheus.yaml* | wc -l").stdout.strip()
        expect(before == after, f"setup ran again on restart ({before} -> {after} config files: a backup was made)")
        token = sandbox_exec(OARA, "token", "show").stdout.splitlines()[0].strip()
        expect(token == versions["token"], "the API token changed across a stop/start")
        return "stop/start keeps the config (setup not rerun) and the token"

    step("O0 host", o0)
    step("O1 model server on the host", o1)
    step("O2 sandbox up", o2)
    step("O3 forwards and the API", o3)
    step("O4 agent turn", o4)
    step("O5 egress policy", o5)
    step("O6 filesystem policy", o6)
    step("O7 restart", o7)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--docker", metavar="IMAGE")
    mode.add_argument("--openshell", metavar="IMAGE")
    parser.add_argument("--gateway", default="openshell", help="--openshell: the gateway name the CLI knows")
    parser.add_argument("--openshell-bin", default="openshell")
    args = parser.parse_args(argv)

    res = Resources()
    results: list[tuple[str, bool, str]] = []
    try:
        if args.docker:
            docker_mode(args.docker, res, results)
        else:
            openshell_mode(args.openshell, args.gateway, args.openshell_bin, res, results)
    except Exception as exc:
        results.append(("(failed step)", False, f"{type(exc).__name__}: {exc}"))
    finally:
        notes = res.cleanup()
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    for note in notes:
        print(f"note: {note}")
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
