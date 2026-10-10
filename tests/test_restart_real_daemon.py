"""The restart route against the real artifact: a real ``oara daemon``, restarted over HTTP, comes back.

A REAL daemon subprocess (isolated HOME, a loopback stub model, bound to 127.0.0.1) is asked to restart through
``POST /api/restart``. It must answer 202, drop its sockets, boot a SECOND time in the SAME process (re-exec, not
an exit a supervisor has to notice), and answer ``GET /api/hello`` again with the same ``fp`` within Beacon's 60
seconds. Stopped by its recorded PID at the end, like tests/test_daemon_shutdown.py.
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
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

TOKEN = "restart-real-daemon-token-0123456789"
BOOTED = "Prometheus daemon running"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ModelsHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: ANN002
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "restart-model"}]}).encode()
        self.send_response(200 if "/v1/models" in self.path else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _hello(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/hello", timeout=2) as response:
            return json.loads(response.read())
    except (OSError, urllib.error.URLError, ValueError):
        return None


def test_a_real_daemon_restarted_over_http_comes_back_with_the_same_fp(tmp_path):
    stub = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    api_port, ws_port = _free_port(), _free_port()
    home = tmp_path / "home"
    (home / ".prometheus").mkdir(parents=True)
    (home / ".prometheus" / "prometheus.yaml").write_text(textwrap.dedent(f"""\
        model:
          provider: llama_cpp
          base_url: http://127.0.0.1:{stub.server_port}
          model: restart-model
        gateway:
          telegram_enabled: false
        web:
          enabled: true
          bind: 127.0.0.1
          api_port: {api_port}
          ws_port: {ws_port}
          api_token: {TOKEN}
        tools:
          deferred_loading:
            enabled: auto
            always_loaded: [bash, read_file]
        """), encoding="utf-8")
    env = {"HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONUNBUFFERED": "1",
           "LANG": "C.UTF-8", "PYTHONUSERBASE": site.getuserbase()}
    if "VIRTUAL_ENV" in os.environ:
        env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]
    # THIS tree's code, first and absolute: the child runs in tmp_path, where a relative PYTHONPATH (src) resolves
    # to nothing and an editable install may point at another checkout. That ran stale code once, whose hello
    # answered 401.
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)
    log_path = tmp_path / "daemon.log"
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "prometheus", "daemon"], cwd=tmp_path, env=env,
                            stdout=log, stderr=subprocess.STDOUT)

    def wait_for(predicate, what: str, seconds: float = 60):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"daemon exited rc={proc.returncode} while waiting for {what}: "
                            f"{log_path.read_text()[-2000:]}")
            value = predicate()
            if value:
                return value
            time.sleep(0.5)
        pytest.fail(f"timed out waiting for {what}: {log_path.read_text()[-2000:]}")

    try:
        first = wait_for(lambda: _hello(api_port), "the first boot")
        assert first["fp"], first
        request = urllib.request.Request(f"http://127.0.0.1:{api_port}/api/restart", method="POST", data=b"",
                                         headers={"Authorization": f"Bearer {TOKEN}"})
        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.status == 202 and json.loads(response.read()) == {"restarting": True}
        wait_for(lambda: log_path.read_text().count(BOOTED) >= 2, "the second boot")
        second = wait_for(lambda: _hello(api_port), "hello after the restart")
        assert second["fp"] == first["fp"], "the restarted daemon is not the same instance"
        assert proc.poll() is None, "it exited instead of re-exec'ing in place"
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.close()
        stub.shutdown()
