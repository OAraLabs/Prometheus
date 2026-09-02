"""FIRSTLIGHT FL-1 — the daemon must exit on SIGTERM and SIGINT, promptly.

The stranger walk ended on this: "Press Ctrl+C to stop" hung the process
(25s SIGINT probe; SIGTERM identical), on every fresh install, forever.
Production never felt it because systemd escalates to SIGKILL at its 90s
stop timeout.

Root cause, corrected by mutation testing: the embedded cron scheduler's
``run_scheduler_loop`` registered its own SIGTERM/SIGINT handlers via
``loop.add_signal_handler`` — asyncio keeps exactly ONE callback per
signal per loop, last registration wins, and cron registered after the
daemon. uvicorn was the obvious suspect (its ``capture_signals`` also
grabs both signals) and was exonerated: on >=0.29 it restores the previous
handlers and RE-RAISES the captured signal on its way out, so with cron
fixed, a mutation restoring plain ``uvicorn.Server`` changed nothing.

The pin is deliberately behavioral and total: a REAL ``oara daemon``
subprocess, booted against a loopback stub model with an isolated HOME,
must exit within seconds of SIGTERM and of SIGINT. Whatever component of
the signal chain regresses — cron, uvicorn semantics on some future
version, a new subsystem grabbing handlers — this goes red. No mocks
anywhere on the path the defect lived in.
"""

from __future__ import annotations

import json
import os
import signal
import site
import socket
import subprocess
import sys
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ModelsHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: ANN002
        pass

    def do_GET(self):
        body = json.dumps(
            {"object": "list", "data": [{"id": "fl1-model"}]}
        ).encode()
        self.send_response(200 if "/v1/models" in self.path else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT],
                         ids=["SIGTERM", "SIGINT-ctrl-c"])
def test_daemon_exits_promptly_on_sigterm(tmp_path, sig):
    """The FL-1 claim, against the real artifact: boot `oara daemon`
    with an isolated HOME and a stub model server, signal it, and require
    exit within seconds — not systemd's 90s SIGKILL. Both signals: the
    walk's literal complaint was Ctrl+C (SIGINT); systemd stop is SIGTERM."""
    stub = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    try:
        api_port, ws_port = _free_port(), _free_port()
        home = tmp_path / "home"
        home.mkdir()
        cfg_dir = home / ".prometheus"
        cfg_dir.mkdir()
        (cfg_dir / "prometheus.yaml").write_text(textwrap.dedent(f"""\
            model:
              provider: llama_cpp
              base_url: http://127.0.0.1:{stub.server_port}
              model: fl1-model
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
            """), encoding="utf-8")

        env = {
            "HOME": str(home),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONUNBUFFERED": "1",
            "LANG": "C.UTF-8",
        }
        if "VIRTUAL_ENV" in os.environ:
            env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]
        # Same reason as test_cli_logging: a foreign HOME moves the user
        # site-packages, and a box whose deps live in ~/.local then loses
        # pydantic in the child. Isolation is about config, not packages.
        env["PYTHONUSERBASE"] = site.getuserbase()
        pythonpath = os.environ.get("PYTHONPATH")
        if pythonpath:
            env["PYTHONPATH"] = pythonpath

        log = (tmp_path / "daemon.log").open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-m", "prometheus", "daemon"],
            cwd=tmp_path, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            # Booted = the web API answers at all (401 or 200 both count).
            deadline = time.time() + 60
            up = False
            while time.time() < deadline:
                if proc.poll() is not None:
                    pytest.fail(
                        "daemon exited rc=%s before booting — see %s"
                        % (proc.returncode,
                           (tmp_path / "daemon.log").read_text()[-2000:]))
                try:
                    with socket.create_connection(("127.0.0.1", api_port),
                                                  timeout=1):
                        up = True
                        break
                except OSError:
                    time.sleep(0.5)
            assert up, "daemon web port never opened within 60s"

            proc.send_signal(sig)
            try:
                rc = proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                pytest.fail(
                    f"FL-1 regression: daemon still alive 15s after "
                    f"{signal.Signals(sig).name} (pre-fix behavior was an "
                    f"indefinite hang masked by systemd's 90s SIGKILL)")
            assert rc in (0, -sig), (
                f"daemon exited rc={rc} on {signal.Signals(sig).name} — "
                f"expected a clean exit")
        finally:
            if proc.poll() is None:
                proc.kill()
            log.close()
    finally:
        stub.shutdown()
