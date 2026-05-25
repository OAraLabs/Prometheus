"""Tests for ``prometheus init`` (Polish & Platform sprint, WS4)."""

from __future__ import annotations

import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from prometheus.cli.init import (
    DetectedServer,
    detect_local_servers,
    run_init,
    write_config,
    write_env_template,
)


# ---------------------------------------------------------------------------
# detect_local_servers — fake HTTP server fixture
# ---------------------------------------------------------------------------


class _FakeLlamaCppHandler(BaseHTTPRequestHandler):
    """Responds with an OpenAI-shaped /v1/models payload."""

    def do_GET(self):  # noqa: N802 — required name
        if self.path == "/v1/models":
            body = json.dumps({
                "object": "list",
                "data": [
                    {"id": "gemma4-26b", "object": "model"},
                    {"id": "qwen3.5-32b", "object": "model"},
                ],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args, **kwargs):  # silence test output
        pass


class _FakeOllamaHandler(BaseHTTPRequestHandler):
    """Responds with the Ollama /api/tags shape."""

    def do_GET(self):  # noqa: N802
        if self.path == "/api/tags":
            body = json.dumps({
                "models": [
                    {"name": "llama3:8b", "size": 1234567890},
                    {"name": "mistral:7b", "size": 9876543210},
                ],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args, **kwargs):
        pass


def _free_port() -> int:
    """Grab an ephemeral free TCP port."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextmanager
def _serve(handler_cls):
    """Start a tiny HTTPServer on an ephemeral port; yield the URL."""
    port = _free_port()
    srv = HTTPServer(("127.0.0.1", port), handler_cls)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()
        srv.server_close()


class TestDetectLocalServers:
    """detect_local_servers should find anything reachable on probe."""

    def test_no_servers_returns_empty_list(self):
        # All defaults point at standard ports; with a tight timeout and
        # custom candidate list pointed at a closed port, we should see nothing.
        candidates = [{
            "name": "ghost",
            "url": "http://127.0.0.1:1",  # port 1 is reserved/closed
            "models_path": "/v1/models",
            "provider": "test",
        }]
        result = detect_local_servers(timeout=0.1, candidates=candidates)
        assert result == []

    def test_detects_openai_compat_server(self):
        with _serve(_FakeLlamaCppHandler) as url:
            candidates = [{
                "name": "fake-llamacpp",
                "url": url,
                "models_path": "/v1/models",
                "provider": "llama_cpp",
            }]
            result = detect_local_servers(timeout=2.0, candidates=candidates)
        assert len(result) == 1
        assert result[0].name == "fake-llamacpp"
        assert result[0].provider == "llama_cpp"
        assert "gemma4-26b" in result[0].models
        assert "qwen3.5-32b" in result[0].models
        assert result[0].response_ms > 0

    def test_detects_ollama_server(self):
        with _serve(_FakeOllamaHandler) as url:
            candidates = [{
                "name": "fake-ollama",
                "url": url,
                "models_path": "/api/tags",
                "provider": "ollama",
            }]
            result = detect_local_servers(timeout=2.0, candidates=candidates)
        assert len(result) == 1
        assert result[0].models == ["llama3:8b", "mistral:7b"]


# ---------------------------------------------------------------------------
# write_config / write_env_template
# ---------------------------------------------------------------------------


class TestWriteConfig:
    def test_writes_yaml(self, tmp_path):
        path = tmp_path / "prometheus.yaml"
        write_config({"system": {"name": "x"}}, path, backup_existing=True)
        import yaml
        loaded = yaml.safe_load(path.read_text())
        assert loaded == {"system": {"name": "x"}}

    def test_backs_up_existing(self, tmp_path):
        path = tmp_path / "prometheus.yaml"
        path.write_text("old: true\n")
        backup = write_config({"new": True}, path, backup_existing=True)
        assert backup is not None
        assert backup.exists()
        assert "old: true" in backup.read_text()
        assert "new: true" in path.read_text()

    def test_no_backup_when_missing(self, tmp_path):
        path = tmp_path / "prometheus.yaml"
        backup = write_config({"new": True}, path, backup_existing=True)
        assert backup is None
        assert path.exists()


class TestWriteEnvTemplate:
    def test_writes_template_when_missing(self, tmp_path):
        path = tmp_path / "env"
        written = write_env_template(path)
        assert written is True
        body = path.read_text()
        assert "PROMETHEUS_TELEGRAM_TOKEN" in body
        assert "ANTHROPIC_API_KEY" in body

    def test_preserves_existing(self, tmp_path):
        path = tmp_path / "env"
        path.write_text("CUSTOM=1\n")
        written = write_env_template(path)
        assert written is False
        assert path.read_text() == "CUSTOM=1\n"


# ---------------------------------------------------------------------------
# run_init — end-to-end with no servers
# ---------------------------------------------------------------------------


class TestRunInit:
    def test_noninteractive_writes_config(self, tmp_path):
        config = run_init(
            noninteractive=True,
            target_dir=tmp_path,
            timeout=0.05,  # short timeout — pretend no servers respond
        )
        cfg_path = tmp_path / "prometheus.yaml"
        env_path = tmp_path / "env"
        assert cfg_path.exists()
        assert env_path.exists()

        import yaml
        loaded = yaml.safe_load(cfg_path.read_text())
        # Shape sanity — minimal required sections present
        assert "model" in loaded
        assert "gateway" in loaded
        assert "web" in loaded
        assert loaded["web"]["enabled"] is True
        # Returned config matches what was written
        assert loaded == config

    def test_existing_config_is_backed_up(self, tmp_path):
        cfg_path = tmp_path / "prometheus.yaml"
        cfg_path.write_text("previous: real\n")

        run_init(noninteractive=True, target_dir=tmp_path, timeout=0.05)

        # Original is preserved as a backup
        backups = list(tmp_path.glob("prometheus.yaml.backup-*"))
        assert len(backups) == 1
        assert "previous: real" in backups[0].read_text()
