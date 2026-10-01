"""WP-X.7 — a session's model choice survives a daemon restart.

On 2026-09-28 a plain `systemctl --user stop/start` dropped five sessions'
cloud choices: they lived only in the router's in-memory table, and only
deploy.sh knew to record and re-apply them. The sessions ran on the local
model for ~16 hours and nothing said so.

The pin is the whole path, against the real artifact: a REAL `oara daemon`
subprocess with an isolated HOME and a stub local model. A choice is set
through the same REST call the Beacon picker makes, the daemon is stopped with
SIGTERM, a second daemon boots on the same HOME, and the choice is read back
through REST. No doubles anywhere on the path.

The second boot also carries the restore rulings (Will, 2026-09-30):
* a stored model that no longer exists → the default model, a WARNING naming
  the session and key, a ``silent_failures`` row, and the stored row deleted;
* a missing credential → the default model, a WARNING and a row, and the
  stored row KEPT, so fixing the key and restarting brings the choice back.
"""

from __future__ import annotations

import json
import os
import signal
import site
import socket
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

# Fake credentials, letters only (the pre-commit scanner reads staged files).
_TOKEN = "restart" + "test" + "bearer" + "token"
_FAKE_KEY = "not" + "a" + "real" + "key"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ModelsHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: ANN002
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "wpx7-model"}]}).encode()
        self.send_response(200 if "/v1/models" in self.path else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _config(stub_port: int, api_port: int, ws_port: int, qwen_models: list[str]) -> str:
    return textwrap.dedent(f"""\
        model:
          provider: llama_cpp
          base_url: http://127.0.0.1:{stub_port}
          model: wpx7-model
        gateway:
          telegram_enabled: false
        web:
          enabled: true
          api_port: {api_port}
          ws_port: {ws_port}
        tools:
          deferred_loading:
            enabled: auto
            always_loaded: [bash, read_file]
        slash_commands:
          qwen:
            models: {json.dumps(qwen_models)}
        """)


class _Daemon:
    """One real daemon process on *home*. Booted = the REST port answers."""

    def __init__(self, tmp_path: Path, home: Path, api_port: int, env_extra: dict[str, str]):
        self.api_port = api_port
        self.log_path = tmp_path / f"daemon-{time.monotonic_ns()}.log"
        env = {
            "HOME": str(home),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONUNBUFFERED": "1",
            "LANG": "C.UTF-8",
            "PROMETHEUS_API_TOKEN": _TOKEN,
            "PYTHONUSERBASE": site.getuserbase(),
            **env_extra,
        }
        if "VIRTUAL_ENV" in os.environ:
            env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]
        if os.environ.get("PYTHONPATH"):
            env["PYTHONPATH"] = os.environ["PYTHONPATH"]
        self._log = self.log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "prometheus", "daemon"],
            cwd=tmp_path, env=env, stdout=self._log, stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 90
        while time.time() < deadline:
            if self.proc.poll() is not None:
                pytest.fail(f"daemon exited rc={self.proc.returncode} before booting:\n"
                            + self.log_path.read_text()[-3000:])
            try:
                if self.call("GET", "/api/status")[0] == 200:
                    return
            except OSError:
                pass
            time.sleep(0.5)
        self.stop()
        pytest.fail("daemon REST never answered within 90s:\n" + self.log_path.read_text()[-3000:])

    def call(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.api_port}{path}", method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": f"Bearer {_TOKEN}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self._log.close()


def _stored(home: Path) -> dict[str, str]:
    db = home / ".prometheus" / "data" / "lcm.db"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return dict(con.execute("SELECT session_id, key FROM session_backends").fetchall())
    finally:
        con.close()


def _silent_failures(home: Path) -> list[dict]:
    con = sqlite3.connect(f"file:{home / '.prometheus' / 'telemetry.db'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(
            "SELECT subsystem, operation, exception_msg, context FROM silent_failures "
            "WHERE subsystem = 'model_choice'")]
    finally:
        con.close()


def test_model_choices_survive_a_real_daemon_restart(tmp_path):
    stub = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    home = tmp_path / "home"
    (home / ".prometheus").mkdir(parents=True)
    cfg = home / ".prometheus" / "prometheus.yaml"
    api_port, ws_port = _free_port(), _free_port()
    kept, removed, keyless, untouched = (
        "beacon:wpx7-kept", "beacon:wpx7-removed", "beacon:wpx7-keyless", "beacon:wpx7-untouched")
    try:
        # ── boot 1: choose ────────────────────────────────────────────────
        cfg.write_text(_config(stub.server_port, api_port, ws_port,
                               ["qwen3.8-flash", "qwen-wpx7-retired"]), encoding="utf-8")
        d1 = _Daemon(tmp_path, home, api_port,
                     {"ANTHROPIC_API_KEY": _FAKE_KEY, "QWEN_API_KEY": _FAKE_KEY})
        try:
            for sid, key in ((kept, "claude"), (removed, "qwen:qwen-wpx7-retired"), (keyless, "gpt")):
                code, body = d1.call("POST", f"/api/sessions/{sid}/model", {"key": key})
                assert code == 200 and body["key"] == key, (sid, code, body)
        finally:
            d1.stop()

        # ── boot 2: the retired model is gone from config; gpt has no key ──
        cfg.write_text(_config(stub.server_port, api_port, ws_port, ["qwen3.8-flash"]),
                       encoding="utf-8")
        d2 = _Daemon(tmp_path, home, api_port,
                     {"ANTHROPIC_API_KEY": _FAKE_KEY, "QWEN_API_KEY": _FAKE_KEY})
        try:
            seen = {sid: d2.call("GET", f"/api/sessions/{sid}/model")[1]["key"]
                    for sid in (kept, removed, keyless, untouched)}
        finally:
            d2.stop()
        log = d2.log_path.read_text()

        # The defect: on main every one of these came back "local".
        assert seen[kept] == "claude", f"the cloud choice did not survive the restart: {seen}"
        assert seen[untouched] == "local"

        # A retired model: default, WARNING, a row, the stored choice deleted.
        assert seen[removed] == "local"
        assert any("WARNING" in line and removed in line and "qwen:qwen-wpx7-retired" in line
                   for line in log.splitlines()), "no WARNING names the retired choice"
        # A missing credential: default, WARNING, a row, the stored choice KEPT.
        assert seen[keyless] == "local"
        assert any("WARNING" in line and keyless in line and "gpt" in line
                   for line in log.splitlines()), "no WARNING names the keyless choice"

        stored = _stored(home)
        assert stored.get(kept) == "claude"
        assert removed not in stored, "a retired choice must be deleted, not kept"
        assert stored.get(keyless) == "gpt", "a choice missing only its key must be kept"

        rows = _silent_failures(home)
        contexts = [json.loads(r["context"] or "{}") for r in rows]
        assert {c.get("session_id") for c in contexts} >= {removed, keyless}, rows
        assert _FAKE_KEY not in json.dumps(rows), "a credential value reached telemetry"
    finally:
        stub.shutdown()
