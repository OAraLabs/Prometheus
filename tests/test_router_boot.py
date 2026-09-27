"""WP-X.17 — a router entry the loader refuses must never cost the boot.

An upgrade that starts refusing a rule or fallback entry (a key its provider
never reads, a missing provider) skips THAT entry with one WARNING naming it
and the key, and the daemon comes up with everything else. A config still
carrying the removed ``router.auxiliary`` block boots with the deprecation
WARNING and nothing else from the router.

Against the real artifact, the same way tests/test_daemon_shutdown.py pins
FL-1: a real ``oara daemon`` subprocess on a loopback stub model with an
isolated HOME. ``--config`` is explicit so a checkout's own
config/prometheus.yaml can never be the one that boots.
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
from pathlib import Path

import pytest

ROUTER_LOGGER = "prometheus.router.model_router"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ModelsHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: ANN002
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "boot-model"}]}).encode()
        self.send_response(200 if "/v1/models" in self.path else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _boot(tmp_path: Path, router_yaml: str) -> str:
    """Boot `oara daemon` with this router: section; return its log once the
    web API answers. Fails if the process exits or never comes up."""
    stub = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    try:
        api_port, ws_port = _free_port(), _free_port()
        home = tmp_path / "home"
        (home / ".prometheus").mkdir(parents=True)
        config_path = home / ".prometheus" / "prometheus.yaml"
        config_path.write_text(textwrap.dedent(f"""\
            model:
              provider: llama_cpp
              base_url: http://127.0.0.1:{stub.server_port}
              model: boot-model
            gateway:
              telegram_enabled: false
            web:
              enabled: true
              api_port: {api_port}
              ws_port: {ws_port}
            """) + textwrap.dedent(router_yaml), encoding="utf-8")

        env = {
            "HOME": str(home),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONUNBUFFERED": "1",
            "LANG": "C.UTF-8",
            # A foreign HOME moves user site-packages; see test_daemon_shutdown.
            "PYTHONUSERBASE": site.getuserbase(),
        }
        for key in ("VIRTUAL_ENV", "PYTHONPATH"):
            if key in os.environ:
                env[key] = os.environ[key]

        log_path = tmp_path / "daemon.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "prometheus", "--config", str(config_path), "daemon"],
                cwd=tmp_path, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        try:
            deadline = time.time() + 90
            while time.time() < deadline:
                if proc.poll() is not None:
                    pytest.fail(f"daemon exited rc={proc.returncode} during boot:\n"
                                f"{log_path.read_text()[-3000:]}")
                try:
                    with socket.create_connection(("127.0.0.1", api_port), timeout=1):
                        break
                except OSError:
                    time.sleep(0.5)
            else:
                pytest.fail("daemon API port never opened within 90s:\n"
                            + log_path.read_text()[-3000:])
            assert proc.poll() is None, "daemon died right after opening its port"
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=20)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
        return log_path.read_text(encoding="utf-8")
    finally:
        stub.shutdown()


def _router_lines(log: str, level: str) -> list[str]:
    """This logger's lines about the router: section itself. The same logger
    also warns once per unconfigured slash_commands entry, which every config
    without that section gets and which is not about router:."""
    return [ln for ln in log.splitlines() if f"{ROUTER_LOGGER} {level} router" in ln]


def test_bad_entries_are_skipped_and_the_daemon_boots(tmp_path):
    log = _boot(tmp_path, """\
        router:
          rules:
            - task_type: code_generation
              provider: llama_cpp
              model: qwen3.8-27b
            - task_type: reasoning
              provider: openai
              model: gpt-5.6-luna
              api_key_evn: MY_KEY
          fallback:
            - provider: ollama
              base_url: http://ollama-box:11434
              model: qwen3.5:9b
            - base_url: http://other-box:8080
              model: qwen3.8-27b
        """)

    # One WARNING per refused entry, naming the entry and the key.
    warnings = _router_lines(log, "WARNING")
    assert len(warnings) == 2, "\n".join(warnings) or log[-3000:]
    assert "router.fallback[1] ?/qwen3.8-27b refused: missing provider" in warnings[0]
    assert "router.rules[1] openai/gpt-5.6-luna refused: api_key_evn is not read" in warnings[1]
    # And what did load: the good entries, without the refused ones.
    [loaded] = [ln for ln in _router_lines(log, "INFO") if "router: loaded" in ln]
    assert ("router: loaded rules [code_generation → llama_cpp/qwen3.8-27b], "
            "fallback [ollama/qwen3.5:9b]; 2 refused") in loaded


@pytest.mark.parametrize("auxiliary", [
    # What a copy of the template shipped from #221 until now.
    """\
        router:
          auxiliary:
            compression:
            summarization:
            vision:
        """,
    # And a block someone actually filled in.
    """\
        router:
          auxiliary:
            vision:
              provider: openai
              model: gpt-4o
        """,
], ids=["template-placeholders", "configured"])
def test_a_config_with_router_auxiliary_boots_with_the_deprecation_only(tmp_path, auxiliary):
    log = _boot(tmp_path, auxiliary)

    [warning] = _router_lines(log, "WARNING")
    assert "router.auxiliary: config key is deprecated" in warning
