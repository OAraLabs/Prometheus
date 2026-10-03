"""scripts/deploy.sh — the static half of its contract, and B6 dry runs.

The script's phases need uv, a network, systemd and /proc, so they are
exercised by running it (phase A end to end, and B0/B6 against a live
daemon, in the PR that added it), not here. What CAN rot silently is
checked here: the script parses, refuses bad usage, and the drop-in it
writes still sets PROMETHEUS_VENV — the variable that makes
scripts/deploy_guard.sh compare the venv with the checkout's uv.lock. Drop
that line and the guard's lock check quietly stops applying.

B6 is the exception: it only talks to the daemon's REST API, so its block is
cut out of the script and run as-is, by bash, against a stub daemon. Since
WP-X.7 the daemon keeps model choices across a restart, so B6 CHECKS them
against B0's record and stops the deploy on a difference; the old re-apply
runs only under --reapply-model-choices.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
DEPLOY = SCRIPTS / "deploy.sh"
GUARD = SCRIPTS / "deploy_guard.sh"


def test_the_script_is_present_executable_and_parses():
    assert os.access(DEPLOY, os.X_OK), f"{DEPLOY} is not executable"
    r = subprocess.run(["bash", "-n", str(DEPLOY)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_bad_usage_exits_2_and_help_exits_0():
    for args, want in (([], 2), (["--bogus"], 2), (["a", "b"], 2), (["--help"], 0)):
        r = subprocess.run(["bash", str(DEPLOY), *args], capture_output=True,
                           text=True, timeout=30)
        assert r.returncode == want, (args, r.returncode, r.stderr)
        assert "USAGE" in r.stderr


def test_the_drop_in_arms_the_guards_lock_check():
    text = DEPLOY.read_text(encoding="utf-8")
    dropin = text[text.index('cat > "$DROPIN" <<EOF'):]
    dropin = dropin[:dropin.index("\nEOF\n")]
    assert "Environment=PROMETHEUS_VENV=$ROOT/current" in dropin
    assert "ExecStart=\n" in dropin, "the base ExecStart must be cleared first"
    assert "PROMETHEUS_VENV" in GUARD.read_text(encoding="utf-8")


def test_the_venv_records_the_lock_the_guard_reads():
    assert 'BUILT_FROM_UV_LOCK"' in DEPLOY.read_text(encoding="utf-8")
    assert "/BUILT_FROM_UV_LOCK" in GUARD.read_text(encoding="utf-8")


def test_the_embedded_python_compiles():
    """B0/B6 and G1 are heredocs; a syntax error there would only show up
    mid-deploy, after the daemon was already being switched."""
    text = DEPLOY.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'EOF'[^\n]*\n(.*?)\nEOF\n", text, re.S)
    assert len(blocks) == 4, (
        f"expected G1, B0, B6 re-apply and B6 check heredocs, found {len(blocks)}")
    for i, src in enumerate(blocks):
        compile(src, f"deploy.sh heredoc #{i}", "exec")


def test_reapply_flag_is_accepted_not_bad_usage():
    """The fallback flag parses; the run then stops on the missing interpreter."""
    env = {**os.environ, "PROMETHEUS_DEPLOY_PYTHON": "/nonexistent/python3"}
    r = subprocess.run(["bash", str(DEPLOY), "abc1234", "--reapply-model-choices"],
                       capture_output=True, text=True, timeout=30, env=env)
    assert r.returncode == 1, (r.returncode, r.stderr)
    assert "USAGE" not in r.stderr
    assert "interpreter /nonexistent/python3 not found" in r.stderr


# ---------------------------------------------------------------------------
# B6 dry runs: the script's own B6 block, by bash, against a stub daemon
# ---------------------------------------------------------------------------

# Letters only, built by concatenation (the pre-commit scanner reads this file).
TOKEN = "stub" + "daemon" + "token"


def _row(key, model, *, provider="alibaba", backend=None, is_default=False):
    return {"key": key, "label": key, "provider": provider, "model": model,
            "backend": backend, "is_default": is_default, "available": True}


DEFAULT = _row("local", "Qwen3.8-27B", provider="llama_cpp", backend="local", is_default=True)
CLOUD = _row("qwen", "qwen3.8-max")
FLASH = _row("qwen:qwen3.8-flash", "qwen3.8-flash")
BACKEND = _row("4090", "Qwen3.8-27B", provider="llama_cpp", backend="4090")
FIELDS = ("key", "provider", "model", "backend", "is_default")


class _StubDaemon:
    """GET/POST /api/sessions/<id>/model, bearer-checked, every request logged."""

    def __init__(self, live: dict[str, dict]):
        self.live = live
        self.requests: list[tuple[str, str, dict | None]] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: ANN002
                pass

            def _sid(self):
                m = re.fullmatch(r"/api/sessions/(.+)/model", self.path)
                return unquote(m.group(1)) if m else None

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _authed(self):
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    self._send(401, {"error": "unauthorized"})
                    return False
                return True

            def do_GET(self):  # noqa: N802
                stub.requests.append(("GET", self.path, None))
                sid = self._sid()
                if not self._authed():
                    return
                if sid in stub.live:
                    self._send(200, stub.live[sid])
                else:
                    self._send(404, {"error": f"no session {sid}"})

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append(("POST", self.path, body))
                if self._authed():
                    self._send(200, {"ok": True})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.api = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def writes(self):
        return [r for r in self.requests if r[0] != "GET"]


@pytest.fixture
def stub_daemon():
    made: list[_StubDaemon] = []

    def make(live):
        made.append(_StubDaemon(live))
        return made[-1]

    yield make
    for d in made:
        d.close()


def _b6_block() -> str:
    text = DEPLOY.read_text(encoding="utf-8")
    start = text.index('if [ ! -f "$CHOICES" ]; then')
    end = text.index('say "done:', start)
    helpers = [line for line in text.splitlines()
               if line.startswith(("say() {", "die() {"))]
    assert len(helpers) == 2, "say/die helpers moved; update the dry run"
    return "set -euo pipefail\n" + "\n".join(helpers) + "\n" + text[start:end]


def _record(tmp_path: Path, saved: dict[str, dict]) -> Path:
    """B0's file format: one row per session, the five fields B0 keeps."""
    rows = [{"session_id": sid, **{k: r.get(k) for k in FIELDS}} for sid, r in saved.items()]
    path = tmp_path / "model-choices-dryrun.json"
    path.write_text(json.dumps(rows, indent=1))
    return path


def _run_b6(api: str, choices: Path, *, reapply: bool = False):
    env = {**os.environ, "API": api, "TOKEN": TOKEN, "CHOICES": str(choices), "REF": "v-dryrun",
           "REAPPLY_CHOICES": "1" if reapply else "0"}
    return subprocess.run(["bash", "-c", _b6_block()], capture_output=True,
                          text=True, timeout=60, env=env)


SAVED = {"telegram:alpha": CLOUD, "web:bravo": FLASH, "web:charlie": BACKEND,
         "beacon:delta": DEFAULT}


def test_b6_check_prints_ok_when_live_matches_the_record(tmp_path, stub_daemon):
    daemon = stub_daemon(dict(SAVED))
    r = _run_b6(daemon.api, _record(tmp_path, SAVED))
    assert r.returncode == 0, r.stderr
    assert "B6 OK" in r.stderr and "4 of 4" in r.stderr
    assert daemon.writes() == [], "the check must only read"
    assert len(daemon.requests) == 4


def test_b6_check_stops_the_deploy_listing_each_difference(tmp_path, stub_daemon):
    live = dict(SAVED)
    live["telegram:alpha"] = DEFAULT        # a cloud choice fell back to the default
    live["web:charlie"] = DEFAULT           # a backend choice came back on the primary
    del live["web:bravo"]                   # the session is gone
    daemon = stub_daemon(live)

    r = _run_b6(daemon.api, _record(tmp_path, SAVED))

    assert r.returncode == 1, r.stderr
    assert "STOP" in r.stderr and "3 of 4" in r.stderr
    lines = r.stderr.splitlines()
    alpha = [ln for ln in lines if "telegram:alpha" in ln]
    charlie = [ln for ln in lines if "web:charlie" in ln]
    bravo = [ln for ln in lines if "web:bravo" in ln]
    assert alpha and "qwen3.8-max" in alpha[0] and "Qwen3.8-27B" in alpha[0]
    assert charlie and "4090" in charlie[0] and "local" in charlie[0]
    assert bravo and "HTTP 404" in bravo[0]
    # Session ids and model names only: no provider, no error body, no token.
    assert "alibaba" not in r.stderr and "llama_cpp" not in r.stderr
    assert "no session" not in r.stderr and TOKEN not in r.stderr
    assert not [ln for ln in lines if "beacon:delta" in ln], "a match is not listed"
    assert daemon.writes() == [], "the check must not re-apply anything"


def test_b6_check_compares_the_backend_and_default_flag_too(tmp_path, stub_daemon):
    """Same key and model, different backend: still a difference."""
    moved = {**BACKEND, "backend": "mac"}
    daemon = stub_daemon({**SAVED, "web:charlie": moved})
    r = _run_b6(daemon.api, _record(tmp_path, SAVED))
    assert r.returncode == 1
    assert any("web:charlie" in ln for ln in r.stderr.splitlines())


def test_b6_reapply_flag_runs_the_old_reapply(tmp_path, stub_daemon):
    custom = _row("custom", "claude-opus-5-5", provider="anthropic")
    saved = {**SAVED, "telegram:echo": custom}
    daemon = stub_daemon({sid: DEFAULT for sid in saved})

    r = _run_b6(daemon.api, _record(tmp_path, saved), reapply=True)

    assert r.returncode == 0, r.stderr
    posts = sorted((path, body["key"]) for _, path, body in daemon.writes())
    assert posts == [("/api/sessions/telegram%3Aalpha/model", "qwen"),
                     ("/api/sessions/web%3Abravo/model", "qwen:qwen3.8-flash")]
    assert "re-run its slash command by hand" in r.stderr
    assert not any(m == "GET" for m, _, _ in daemon.requests), "re-apply does not check"


def test_b6_without_a_record_says_so_and_touches_nothing(tmp_path, stub_daemon):
    daemon = stub_daemon(dict(SAVED))
    r = _run_b6(daemon.api, tmp_path / "missing.json")
    assert r.returncode == 0
    assert "B6 skipped" in r.stderr
    assert daemon.requests == []
