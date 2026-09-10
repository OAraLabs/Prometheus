#!/usr/bin/env python3
"""FIRSTLIGHT fresh-install harness — the stranger's first ten minutes, executable.

Drives the exact flow the README promises, in an environment with no host
config, and fails loudly naming WHICH step broke and why:

  S1  git clone (of --source, at its current SHA) into a temp tree
  S2  python -m venv + install                      (--install-mode editable|wheel)
  S2i oara identity --regenerate               (SOUL.md/AGENTS.md from the
      SHIPPED templates — the ONE first-run step `setup --noninteractive`
      never reaches, and the step that stayed green while templates/ did not
      ship, because an editable install keeps the checkout layout live)
  S3  oara setup --noninteractive              (against a stub model server)
      --leg cloud: no local server is offered; one cloud key in the environment
      must pick the provider, and the harness points that provider at the stub
  S4  oara doctor                              (must exit 0)
  S5  oara --once "..."                        (one CLI turn that CALLS A TOOL)
  S6  oara daemon                              (401 bare -> token show -> /api/status; one REST turn)
  S7  teardown                                       (no residue: temp gone, ports closed)

CONTRACT
  * This file never imports from src/prometheus. It drives the CLI and the
    HTTP API the way a user does; assertions are on exit codes, files, and
    wire responses only.
  * Isolation is environmental, not assumed: product processes run with a
    minimal, non-inherited environment (fresh HOME under the temp tree), so
    the operator's real ~/.prometheus, ~/.config/prometheus and env vars
    cannot leak in — that is the "no step requires something only Will has"
    guarantee, made mechanical.
  * The model is scripts/firstlight_stub_model.py (stdlib, OpenAI-compatible).
    Its FINAL marker is only ever emitted after it has seen a tool result,
    so the marker appearing in a transcript proves a model->tool->model
    round trip — the tool-call assertion rides the protocol.
  * Two harness-owned edits are made to the config setup wrote, and they are
    infra, not user steps: api_port/ws_port are moved to free ports so a
    live daemon on the host can never collide with the run. The cloud leg
    makes a third: model.base_url is set to the stub, because the cloud
    provider's real endpoint is exactly what this harness must never reach.

OUT OF SCOPE — documented, not silently skipped:
  * real inference (quality, GBNF, adapter tiers) — the stub is a script
  * messaging gateways (Telegram/Slack/Discord) — need real tokens
  * voice (whisper/TTS), media pipelines, GPU anything
  * Beacon/desktop clients; WebSocket streaming semantics beyond boot
  * real cloud APIs and anything needing a paid key — `--leg cloud` covers
    the cloud CODE PATH (setup lands on a cloud provider with no local server,
    the CLI and daemon build that provider and talk OpenAI wire to the stub);
    it never reaches a real endpoint
  * long-horizon subsystems (LCM compaction cadence, SENTINEL, curator)

SELF-TEST LEVERS (mutation testing THIS harness's reporting):
  --stub-mode models-500   breaks S3 (setup finds no server)
  --stub-mode no-final     breaks S5 (the agent turn can never conclude)
  --self-mutation busy-api breaks S6 (the API port is already taken)
  --self-mutation no-templates  deletes the packaged identity templates from the
                           INSTALLED tree (needs --install-mode wheel) and must
                           break S2i — this is the pre-fix wheel, reproduced
  --self-mutation no-cloud-key  (--leg cloud) breaks S3: no server AND no key,
                           so setup must refuse to write and exit 2
A healthy tree must go red at exactly that step, naming it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

FINAL_MARKER = "FIRSTLIGHT-COMPLETE"
# --leg cloud: which preset the stranger's key selects, and the dummy value.
CLOUD_PROVIDER = "openai"
CLOUD_KEY_ENV = "OPENAI_API_KEY"
CLOUD_KEY_VALUE = "firstlight-dummy-cloud-key-never-real"



class StepFailure(Exception):
    def __init__(self, why: str, log: Path | None = None) -> None:
        super().__init__(why)
        self.log = log


# ---------------------------------------------------------------------------
# Small mechanics
# ---------------------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def http_get(url: str, timeout: float = 5.0,
             headers: dict[str, str] | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def http_post_json(url: str, payload: dict, timeout: float = 10.0,
                   headers: dict[str, str] | None = None) -> tuple[int, str]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def tail(path: Path, lines: int = 40) -> str:
    if not path or not path.exists():
        return "(no log captured)"
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


class Harness:
    def __init__(self, source: Path, keep: bool, stub_mode: str,
                 self_mutation: str, strict_shutdown: bool = False,
                 leg: str = "local", install_mode: str = "editable") -> None:
        self.source = source
        self.leg = leg
        self.install_mode = install_mode
        self.keep = keep
        self.stub_mode = stub_mode
        self.self_mutation = self_mutation
        self.strict_shutdown = strict_shutdown
        self.work = Path(tempfile.mkdtemp(prefix="firstlight-"))
        self.clone = self.work / "clone"
        self.home = self.work / "home"
        self.logs = self.work / "logs"
        self.home.mkdir()
        self.logs.mkdir()
        # Every subprocess runs with HOME set to this isolated tree, so THIS
        # is git's global config for them. A --source mounted read-only under
        # a different uid (running the harness inside a container) otherwise
        # trips git's dubious-ownership guard at S1. safe.directory is
        # deliberately ignored from env/-c scopes — a global config file is
        # the only place it counts. Same fix as the upgrade harness.
        (self.home / ".gitconfig").write_text(
            "[safe]\n\tdirectory = *\n", encoding="utf-8")
        self.venv = self.work / "venv"
        self.stub_port = free_port()
        self.api_port = free_port()
        self.ws_port = free_port()
        self.sha = "?"
        self.stub_proc: subprocess.Popen | None = None
        self.daemon_proc: subprocess.Popen | None = None
        self._mutation_sock: socket.socket | None = None

    # -- environment the product runs in (NOT inherited from the host) -----
    def env(self) -> dict[str, str]:
        env = {
            "HOME": str(self.home),
            "PATH": f"{self.venv}/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "TERM": "dumb",
            "PYTHONUNBUFFERED": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
        if self.leg == "cloud" and self.self_mutation != "no-cloud-key":
            # The stranger's one asset: a cloud key, exported. A fixed dummy —
            # the provider is pointed at the stub, so the value is never
            # checked by anything, and it must not be a real one.
            env[CLOUD_KEY_ENV] = CLOUD_KEY_VALUE
        return env

    def run(self, cmd: list[str], log_name: str, timeout: int,
            cwd: Path | None = None, expect_rc: int | None = 0,
            stdin_text: str | None = None) -> tuple[int, Path]:
        log = self.logs / f"{log_name}.log"
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"$ {' '.join(cmd)}\n")
            fh.flush()
            try:
                proc = subprocess.run(
                    cmd, cwd=cwd or self.clone, env=self.env(),
                    input=stdin_text, text=stdin_text is not None,
                    stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                raise StepFailure(
                    f"`{' '.join(cmd)}` did not finish within {timeout}s", log
                )
        if expect_rc is not None and proc.returncode != expect_rc:
            raise StepFailure(
                f"`{' '.join(cmd)}` exited {proc.returncode} "
                f"(expected {expect_rc})", log,
            )
        return proc.returncode, log

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def s1_clone(self) -> str:
        out = subprocess.run(
            ["git", "-C", str(self.source), "rev-parse", "HEAD"],
            capture_output=True, text=True, env=self.env(),
        )
        if out.returncode != 0:
            raise StepFailure(f"--source {self.source} is not a git repo: "
                              f"{out.stderr.strip()}")
        self.sha = out.stdout.strip()
        self.run(["git", "clone", "--quiet", str(self.source), str(self.clone)],
                 "s1-clone", timeout=300, cwd=self.work)
        self.run(["git", "-C", str(self.clone), "checkout", "--quiet",
                  "--detach", self.sha], "s1-clone", timeout=60, cwd=self.work)
        # The stranger's tree has no live config. If one rode along, every
        # later step tests Will's box, not the product (§3b coupling).
        stray = self.clone / "config" / "prometheus.yaml"
        if stray.exists():
            raise StepFailure(f"clone contains a live config at {stray} — "
                              f"the source tree is leaking operator state")
        return f"SHA {self.sha[:12]}"

    def s2_install(self) -> str:
        """Install the product the way a stranger does.

        TWO SHAPES, because they are not the same artefact and only one of
        them was ever exercised. `pip install -e` keeps the CHECKOUT layout
        live, so anything that lives outside `packages = ["src/prometheus"]`
        is still on disk and still importable — which is precisely why this
        harness ran green for the entire time `templates/` did not ship. In
        `--install-mode wheel` the venv gets a built wheel and nothing else,
        so a file that was not packaged is genuinely absent, the way it is
        absent for someone who typed `pip install oara-prometheus`.
        """
        self.run([sys.executable, "-m", "venv", str(self.venv)],
                 "s2-install", timeout=120, cwd=self.work)
        if self.install_mode == "wheel":
            dist = self.work / "dist"
            self.run([str(self.venv / "bin" / "pip"), "install", "--quiet",
                      "build"], "s2-install", timeout=600)
            self.run([str(self.venv / "bin" / "python"), "-m", "build",
                      "--wheel", "--outdir", str(dist)],
                     "s2-install", timeout=900)
            wheels = sorted(dist.glob("*.whl"))
            if len(wheels) != 1:
                raise StepFailure(
                    f"expected exactly one wheel in {dist}, got "
                    f"{[w.name for w in wheels]}", self.logs / "s2-install.log")
            self.run([str(self.venv / "bin" / "pip"), "install", "--quiet",
                      f"{wheels[0]}[full]"], "s2-install", timeout=1500)
            what = f"pip install {wheels[0].name}[full]"
        else:
            self.run([str(self.venv / "bin" / "pip"), "install", "--quiet",
                      "-e", ".[full]"], "s2-install", timeout=1500)
            what = "pip install -e '.[full]'"
        # The command is `oara`; `prometheus` is the alias kept for the
        # deprecation window — a fresh install must have both on PATH.
        self.run([str(self.venv / "bin" / "prometheus"), "--help"],
                 "s2-alias-help", timeout=60)
        rc, _ = self.run([str(self.venv / "bin" / "oara"), "--help"],
                         "s2-install", timeout=60)
        if self.self_mutation == "no-templates":
            # Self-test lever: delete the packaged identity templates from the
            # INSTALLED tree, reproducing the pre-fix wheel. S2i must go red.
            #
            # ⚠ MEASURED, not assumed: on an EDITABLE install this deletion
            # changes nothing observable. Hatchling does materialise the
            # force-included files under site-packages, so they are there to
            # delete — but the module still imports from the clone, so the
            # resolver's checkout fallback finds <clone>/templates and every
            # step stays green. A lever that cannot go red is worse than no
            # lever, so the combination is refused rather than run.
            if self.install_mode != "wheel":
                raise StepFailure(
                    "--self-mutation no-templates requires --install-mode "
                    "wheel. On an editable install the module imports from "
                    "the clone, so the resolver falls back to "
                    "<clone>/templates and the mutation cannot fail anything "
                    "— which is exactly why this harness stayed green while "
                    "the templates did not ship.", None)
            removed = 0
            for hit in self.venv.glob(
                    "lib/*/site-packages/prometheus/templates/*"):
                hit.unlink()
                removed += 1
            if not removed:
                raise StepFailure(
                    "--self-mutation no-templates found no packaged templates "
                    "to delete in the installed tree — they already do not "
                    "ship, which is the defect this lever exists to model.",
                    None)
        return f"{what} + entrypoint present"

    def s2i_identity(self) -> str:
        """`oara identity --regenerate` — the FIRST thing setup does.

        WHY THIS STEP EXISTS. `oara setup --noninteractive` (S3) never
        generates identity: only the interactive wizard and the setup server
        call `generate_identity_files`. So the templates that SOUL.md and
        AGENTS.md are rendered from were read by nothing this harness ran,
        and when they turned out not to ship in the wheel, every leg stayed
        green while a real first run died with

            FileNotFoundError: .../site-packages/templates/SOUL.md.template

        `oara identity --regenerate` is that code path behind a one-purpose
        CLI entry point, so the harness can drive it without scripting the
        whole wizard — three prompts, and the assertions are on the FILES,
        not on the prompt text, so adding a wizard question cannot make this
        fail for a non-product reason.

        Answers: a name, a description, then blank for the hardware layout
        (default: single machine).
        """
        self.run([str(self.venv / "bin" / "oara"), "identity", "--regenerate"],
                 "s2i-identity", timeout=120,
                 stdin_text="Stranger\nbuilds things\n\n")
        home_cfg = self.home / ".prometheus"
        for name in ("SOUL.md", "AGENTS.md"):
            f = home_cfg / name
            if not f.exists():
                raise StepFailure(
                    f"identity generation exited 0 but wrote no {name} at {f}",
                    self.logs / "s2i-identity.log")
            text = f.read_text(encoding="utf-8")
            if "{{" in text:
                raise StepFailure(
                    f"{name} carries unsubstituted template slots — the "
                    f"template was found but not rendered",
                    self.logs / "s2i-identity.log")
        soul = (home_cfg / "SOUL.md").read_text(encoding="utf-8")
        if "Stranger" not in soul:
            raise StepFailure(
                "SOUL.md does not carry the name that was typed — identity is "
                "generic, so the personalisation step did nothing",
                self.logs / "s2i-identity.log")
        return "SOUL.md + AGENTS.md rendered from the SHIPPED templates"

    def _wait_stub(self) -> None:
        for _ in range(50):
            try:
                code, _ = http_get(f"http://127.0.0.1:{self.stub_port}/health",
                                   timeout=1)
                if code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.1)
        raise StepFailure("stub model server never became ready", None)

    def s3_setup(self) -> str:
        stub_log = (self.logs / "stub.log").open("a", encoding="utf-8")
        self.stub_proc = subprocess.Popen(
            [str(self.venv / "bin" / "python"),
             str(self.clone / "scripts" / "firstlight_stub_model.py"),
             "--port", str(self.stub_port), "--mode", self.stub_mode],
            stdout=stub_log, stderr=subprocess.STDOUT, env=self.env(),
        )
        self._wait_stub()
        if self.leg == "cloud":
            # No local server on offer: --probe-url at a port nothing listens
            # on replaces the four well-known candidates, so detection finds
            # nothing wherever the host happens to run a model. With no
            # prompts allowed, the one exported cloud key has to carry it.
            self.run([str(self.venv / "bin" / "oara"), "setup",
                      "--noninteractive", "--timeout", "1",
                      "--probe-url", "http://127.0.0.1:9"],
                     "s3-setup", timeout=120)
        else:
            self.run([str(self.venv / "bin" / "oara"), "setup",
                      "--noninteractive", "--timeout", "3",
                      "--probe-url", f"http://127.0.0.1:{self.stub_port}"],
                     "s3-setup", timeout=120)
        cfg = self.home / ".prometheus" / "prometheus.yaml"
        if not cfg.exists():
            raise StepFailure(f"setup exited 0 but wrote no config at {cfg}",
                              self.logs / "s3-setup.log")
        text = cfg.read_text(encoding="utf-8")
        if self.leg == "cloud":
            for needle, why in (
                (f"provider: {CLOUD_PROVIDER}", "setup did not land on the cloud provider the key selects"),
                (f"api_key_env: {CLOUD_KEY_ENV}", "config does not name the key's env var"),
            ):
                if needle not in text:
                    raise StepFailure(f"{why} — {cfg} lacks `{needle}`",
                                      self.logs / "s3-setup.log")
            if "base_url:" in text:
                raise StepFailure("a cloud config must not carry a base_url "
                                  "from setup", self.logs / "s3-setup.log")
            env_file = self.home / ".config" / "prometheus" / "env"
            if not env_file.exists() or f"{CLOUD_KEY_ENV}=" not in env_file.read_text(encoding="utf-8"):
                raise StepFailure("the key exported to setup was not copied into "
                                  f"the daemon's env file {env_file} — the daemon "
                                  "would boot without it", self.logs / "s3-setup.log")
            if CLOUD_KEY_VALUE in text:
                raise StepFailure("the key value landed in the yaml", None)
            # Harness infra (third edit, documented in the header): aim the
            # cloud provider at the stub. Explicit base_url wins in the
            # provider registry, so this is a user-settable key, used here
            # for the one purpose of never reaching the real endpoint.
            needle = f"  api_key_env: {CLOUD_KEY_ENV}\n"
            if text.count(needle) != 1:
                raise StepFailure(
                    f"expected exactly one `{needle.strip()}` line in the "
                    f"generated config, found {text.count(needle)} — the "
                    "setup template changed; update the harness's base_url "
                    "insert", None)
            text = text.replace(
                needle, needle + f"  base_url: http://127.0.0.1:{self.stub_port}\n")
        elif f"127.0.0.1:{self.stub_port}" not in text:
            raise StepFailure("config does not point at the probed server — "
                              f"{cfg} lacks the stub URL", self.logs / "s3-setup.log")
        # Harness infra (documented in the module header): move the web ports
        # off the defaults so a daemon already running on the host can never
        # collide with this run. These are ordinary user-settable keys.
        for key, port in (("api_port", self.api_port), ("ws_port", self.ws_port)):
            old = f"{key}: {8005 if key == 'api_port' else 8010}"
            if text.count(old) != 1:
                raise StepFailure(
                    f"expected exactly one `{old}` in the generated config, "
                    f"found {text.count(old)} — the setup template changed; "
                    f"update the harness's port rewrite", None)
            text = text.replace(old, f"{key}: {port}")
        cfg.write_text(text, encoding="utf-8")
        if self.leg == "cloud":
            return (f"no local server; ${CLOUD_KEY_ENV} picked {CLOUD_PROVIDER}; "
                    "key copied to the env file; provider aimed at the stub")
        return f"config written; model={FINAL_MARKER.split('-')[0].lower()}-stub via --probe-url"

    def s4_doctor(self) -> str:
        _, log = self.run([str(self.venv / "bin" / "oara"), "doctor"],
                          "s4-doctor", timeout=120)
        return "doctor exit 0"

    def s5_cli_turn(self) -> str:
        _, log = self.run(
            [str(self.venv / "bin" / "oara"), "--once",
             "firstlight: enumerate the repository files"],
            "s5-cli-turn", timeout=300,
        )
        if FINAL_MARKER not in log.read_text(encoding="utf-8", errors="replace"):
            raise StepFailure(
                f"CLI turn finished without the stub's final marker "
                f"({FINAL_MARKER}) — the model->tool->model round trip did "
                f"not complete", log)
        return "one --once turn, tool round trip proven by marker"

    def s6_daemon_rest(self) -> str:
        if self.self_mutation == "busy-api":
            self._mutation_sock = socket.socket()
            self._mutation_sock.bind(("127.0.0.1", self.api_port))
            self._mutation_sock.listen(1)
        daemon_log_path = self.logs / "s6-daemon.log"
        daemon_log = daemon_log_path.open("a", encoding="utf-8")
        self.daemon_proc = subprocess.Popen(
            [str(self.venv / "bin" / "oara"), "daemon"],
            cwd=self.clone, env=self.env(),
            stdout=daemon_log, stderr=subprocess.STDOUT,
        )
        base = f"http://127.0.0.1:{self.api_port}"
        deadline = time.time() + 120
        first_code = None
        while time.time() < deadline:
            if self.daemon_proc.poll() is not None:
                raise StepFailure(
                    f"daemon exited rc={self.daemon_proc.returncode} before "
                    f"/api/status ever answered", daemon_log_path)
            try:
                first_code, _ = http_get(f"{base}/api/status", timeout=2)
                break
            except Exception:
                pass
            time.sleep(1)
        if first_code is None:
            raise StepFailure(f"/api/status on {base} never answered "
                              f"within 120s", daemon_log_path)

        # Fresh-install security default, pinned in BOTH directions: the
        # first daemon start MINTS a web API token (printed once; saved to
        # the env file; `oara token show` re-prints it), so a bare
        # request must be refused...
        if first_code != 401:
            raise StepFailure(
                f"unauthenticated /api/status answered {first_code} — a "
                f"fresh install must mint a token and refuse bare requests "
                f"(expected 401)", daemon_log_path)
        # ...and a client that reads the product's own message gets in. The
        # harness does what that message says: `oara token show`.
        rc, tok_log = self.run([str(self.venv / "bin" / "oara"),
                                "token", "show"], "s6-token-show", timeout=60)
        token_lines = [
            ln.strip() for ln in
            tok_log.read_text(encoding="utf-8", errors="replace").splitlines()
            if ln.strip() and not ln.startswith("$")
        ]
        if not token_lines or " " in token_lines[0]:
            raise StepFailure(
                "`oara token show` did not print a token on its first "
                "line — the fresh install minted one, so show must re-print "
                "it", tok_log)
        auth = {"Authorization": f"Bearer {token_lines[0]}"}

        deadline = time.time() + 60
        status = None
        while time.time() < deadline:
            code, body = http_get(f"{base}/api/status", timeout=2, headers=auth)
            if code == 200 and '"state"' in body:
                status = body
                break
            time.sleep(1)
        if status is None:
            raise StepFailure("authenticated /api/status never answered 200 "
                              "— the token `oara token show` printed "
                              "does not open the API it minted",
                              daemon_log_path)

        code, body = http_post_json(
            f"{base}/api/chat/send",
            {"session_id": "firstlight-e2e",
             "message": "firstlight: run your tool round"},
            headers=auth,
        )
        if code != 200:
            raise StepFailure(f"POST /api/chat/send -> {code}: {body[:200]}",
                              daemon_log_path)
        deadline = time.time() + 240
        while time.time() < deadline:
            code, body = http_get(
                f"{base}/api/sessions/firstlight-e2e/messages", timeout=5,
                headers=auth)
            if code == 200 and FINAL_MARKER in body:
                break
            time.sleep(2)
        else:
            raise StepFailure(
                f"REST turn never produced the stub's final marker in "
                f"/api/sessions/firstlight-e2e/messages within 240s",
                daemon_log_path)

        # FL-1 (FIXED): uvicorn's capture_signals used to replace the
        # daemon's SIGTERM/SIGINT handlers, so the daemon never saw the
        # signal and hung until SIGKILL. Strict shutdown is now the DEFAULT
        # — a hang here is a regression, not a known defect. The
        # --lenient-shutdown escape keeps the old tolerate-and-warn behavior
        # for bisecting old SHAs.
        self.daemon_proc.send_signal(signal.SIGTERM)
        shutdown_note = "clean SIGTERM shutdown"
        try:
            rc = self.daemon_proc.wait(timeout=15)
            if rc not in (0, -signal.SIGTERM):
                raise StepFailure(f"daemon exited rc={rc} on SIGTERM "
                                  f"(expected clean shutdown)", daemon_log_path)
        except subprocess.TimeoutExpired:
            self.daemon_proc.kill()
            self.daemon_proc.wait(timeout=15)
            if self.strict_shutdown:
                raise StepFailure(
                    "daemon ignored SIGTERM for 15s — FL-1-class regression "
                    "(something re-captured the daemon's signal handlers or "
                    "a task/thread outlives shutdown; see start_web's "
                    "_EmbeddedServer note)", daemon_log_path)
            shutdown_note = ("SIGKILL required after 15s — FL-1-class hang, "
                             "tolerated only because --lenient-shutdown is set")
            print(f"[FIRSTLIGHT] WARNING: {shutdown_note}")
        self.daemon_proc = None
        return f"status + one REST turn; shutdown: {shutdown_note}"

    def s7_teardown(self) -> str:
        self._stop_procs()
        for name, port in (("stub", self.stub_port), ("api", self.api_port),
                           ("ws", self.ws_port)):
            deadline = time.time() + 10
            while port_open(port):
                if time.time() > deadline:
                    raise StepFailure(f"{name} port {port} still open after "
                                      f"teardown — a process leaked")
                time.sleep(0.5)
        if self.keep:
            return f"ports closed; --keep set, tree retained at {self.work}"
        shutil.rmtree(self.work, ignore_errors=False)
        if self.work.exists():
            raise StepFailure(f"work tree {self.work} survived removal")
        return "ports closed, temp tree removed, no residue"

    # ------------------------------------------------------------------

    def _stop_procs(self) -> None:
        for proc in (self.daemon_proc, self.stub_proc):
            if proc and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
        if self._mutation_sock is not None:
            self._mutation_sock.close()
            self._mutation_sock = None
        self.daemon_proc = self.stub_proc = None

    def main(self) -> int:
        steps = [
            ("S1", "git clone at source SHA", self.s1_clone),
            ("S2", ("pip install <wheel>[full]" if self.install_mode == "wheel"
                    else "pip install -e '.[full]'"), self.s2_install),
            ("S2i", "oara identity --regenerate (renders the SHIPPED templates)",
             self.s2i_identity),
            ("S3", ("oara setup --noninteractive (no server, one cloud key)"
                    if self.leg == "cloud" else
                    "oara setup --noninteractive (stub model)"), self.s3_setup),
            ("S4", "oara doctor exits 0", self.s4_doctor),
            ("S5", "one CLI turn that calls a tool (--once)", self.s5_cli_turn),
            ("S6", "daemon boots; /api/status; one REST turn", self.s6_daemon_rest),
            ("S7", "teardown, no residue", self.s7_teardown),
        ]
        t0 = time.time()
        print(f"[FIRSTLIGHT] leg={self.leg} install={self.install_mode} "
              f"source={self.source} work={self.work}")
        for sid, name, fn in steps:
            started = time.time()
            try:
                detail = fn()
            except StepFailure as exc:
                self._stop_procs()
                print(f"\n[FIRSTLIGHT] FAILED at {sid} — {name}")
                print(f"[FIRSTLIGHT] why: {exc}")
                if exc.log:
                    print(f"[FIRSTLIGHT] last lines of {exc.log.name}:")
                    print(tail(exc.log))
                print(f"[FIRSTLIGHT] full logs kept at: {self.logs}")
                return 1
            print(f"[FIRSTLIGHT] {sid} ok ({time.time() - started:5.1f}s) — "
                  f"{name}: {detail}")
        print(f"\n[FIRSTLIGHT] PASS ({self.leg} leg, {self.install_mode} "
              f"install) — all {len(steps)} steps, "
              f"{time.time() - t0:.1f}s, SHA {self.sha[:12]}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=".",
                        help="repo to test (cloned at its current HEAD)")
    parser.add_argument("--keep", action="store_true",
                        help="keep the temp tree on success too")
    parser.add_argument("--leg", default="local", choices=["local", "cloud"],
                        help="local: setup detects the stub as a local server "
                             "(default). cloud: no local server, one cloud key "
                             "in the environment, provider aimed at the stub")
    parser.add_argument("--install-mode", default="editable",
                        choices=["editable", "wheel"],
                        help="editable: `pip install -e .[full]`, the README's "
                             "contributor line (default). wheel: build a wheel "
                             "and install THAT — the shape a `pip install "
                             "oara-prometheus` user gets, and the only one in "
                             "which an unpackaged file is genuinely absent")
    parser.add_argument("--stub-mode", default="normal",
                        choices=["normal", "models-500", "no-final"],
                        help="stub model mutation (harness self-test)")
    parser.add_argument("--self-mutation", default="none",
                        choices=["none", "busy-api", "no-cloud-key",
                                 "no-templates"],
                        help="harness-side mutation (harness self-test)")
    parser.add_argument("--strict-shutdown", dest="strict_shutdown",
                        action="store_true", default=True,
                        help="treat a SIGTERM hang as a failure (DEFAULT "
                             "since the FL-1 fix — the ratchet is a gate)")
    parser.add_argument("--lenient-shutdown", dest="strict_shutdown",
                        action="store_false",
                        help="tolerate a shutdown hang with a loud warning "
                             "(pre-FL-1 behavior; for bisecting only)")
    args = parser.parse_args()
    return Harness(Path(args.source).resolve(), args.keep, args.stub_mode,
                   args.self_mutation, args.strict_shutdown, leg=args.leg,
                   install_mode=args.install_mode).main()


if __name__ == "__main__":
    sys.exit(main())
