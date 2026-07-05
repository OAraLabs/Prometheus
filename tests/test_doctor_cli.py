"""``prometheus doctor`` extended checks (Phase 0, item 4).

Server up/down is exercised against a real ephemeral HTTP server (same
pattern as test_cli_init); everything filesystem-shaped is confined to
tmp dirs via PROMETHEUS_CONFIG_DIR / PROMETHEUS_ENV_FILE.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import yaml

from prometheus.cli.doctor import (
    check_config,
    check_dirs_writable,
    check_inference,
    check_token,
    check_web_port,
    check_whisper,
    render_report,
    run_doctor_command,
)


class _ModelsHandler(BaseHTTPRequestHandler):
    payload: dict = {"data": [{"id": "test-model-7b"}]}

    def do_GET(self):  # noqa: N802
        body = json.dumps(self.payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs):
        pass


class _EmptyModelsHandler(_ModelsHandler):
    payload = {"data": []}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextmanager
def _serve(handler_cls):
    port = _free_port()
    srv = HTTPServer(("127.0.0.1", port), handler_cls)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", port
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(tmp_path / "envfile"))
    monkeypatch.delenv("PROMETHEUS_API_TOKEN", raising=False)
    monkeypatch.delenv("PROMETHEUS_DATA_DIR", raising=False)
    monkeypatch.delenv("PROMETHEUS_LOGS_DIR", raising=False)
    monkeypatch.delenv("PROMETHEUS_WORKSPACE_DIR", raising=False)
    return tmp_path


class TestCheckConfig:
    def test_missing_config_is_error(self, isolated_dirs):
        check, config = check_config(str(isolated_dirs / "nope.yaml"))
        assert check.status == "error"
        assert config == {}
        assert "prometheus setup" in (check.fix or "")

    def test_valid_config_parses(self, isolated_dirs):
        path = isolated_dirs / "ok.yaml"
        path.write_text("model:\n  provider: ollama\n", encoding="utf-8")
        check, config = check_config(str(path))
        assert check.status == "ok"
        assert config["model"]["provider"] == "ollama"

    def test_broken_yaml_is_error(self, isolated_dirs):
        path = isolated_dirs / "broken.yaml"
        path.write_text("model: [unclosed\n", encoding="utf-8")
        check, config = check_config(str(path))
        assert check.status == "error"
        assert config == {}


class TestCheckInference:
    def test_server_up_model_detected(self):
        with _serve(_ModelsHandler) as (url, _port):
            reach, model = check_inference(
                {"model": {"provider": "llama_cpp", "base_url": url}},
                timeout=3.0,
            )
        assert reach.status == "ok"
        assert model.status == "ok"
        assert "test-model-7b" in model.message

    def test_server_up_no_model_is_error(self):
        with _serve(_EmptyModelsHandler) as (url, _port):
            reach, model = check_inference(
                {"model": {"provider": "llama_cpp", "base_url": url}},
                timeout=3.0,
            )
        assert reach.status == "ok"
        assert model.status == "error"

    def test_server_down_is_error(self):
        url = f"http://127.0.0.1:{_free_port()}"
        reach, model = check_inference(
            {"model": {"provider": "llama_cpp", "base_url": url}},
            timeout=0.3,
        )
        assert reach.status == "error"
        assert model.status == "error"

    def test_cloud_provider_checks_key(self, monkeypatch):
        monkeypatch.delenv("TEST_DOCTOR_KEY", raising=False)
        cfg = {"model": {"provider": "anthropic",
                         "api_key_env": "TEST_DOCTOR_KEY", "model": "m"}}
        reach, _model = check_inference(cfg)
        assert reach.status == "error"
        monkeypatch.setenv("TEST_DOCTOR_KEY", "k")
        reach, model = check_inference(cfg)
        assert reach.status == "ok"
        assert model.status == "ok"


class TestCheckWebPort:
    def test_disabled_is_warning(self):
        assert check_web_port({"web": {"enabled": False}}).status == "warning"

    def test_free_port_is_ok(self):
        cfg = {"web": {"enabled": True, "api_port": _free_port()}}
        check = check_web_port(cfg)
        assert check.status == "ok"
        assert "free" in check.message

    def test_foreign_listener_is_error(self):
        # A plain TCP listener that never speaks HTTP — occupied port,
        # not Prometheus.
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            check = check_web_port(
                {"web": {"enabled": True, "api_port": port}}, timeout=0.5,
            )
        finally:
            srv.close()
        assert check.status == "error"


class TestCheckTokenAndDirs:
    def test_token_unset_warns(self, isolated_dirs):
        assert check_token({}).status == "warning"

    def test_token_set_ok(self, isolated_dirs, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_API_TOKEN", "s3cr3t-value-xyz")
        check = check_token({})
        assert check.status == "ok"
        assert "s3cr3t-value-xyz" not in check.message  # never leak the token

    def test_dirs_writable(self, isolated_dirs):
        assert check_dirs_writable().status == "ok"


class TestCheckWhisper:
    def test_voice_disabled_skips(self):
        assert check_whisper({}).status == "info"

    def test_voice_enabled_engine_missing_is_error(self, monkeypatch):
        monkeypatch.setattr(
            "prometheus.tools.builtin.whisper_stt._detect_whisper_engine",
            lambda: None,
        )
        check = check_whisper({"whisper": {"enabled": True}})
        assert check.status == "error"
        assert "oara-prometheus[voice]" in (check.fix or "")

    def test_voice_enabled_engine_present_is_ok(self, monkeypatch):
        monkeypatch.setattr(
            "prometheus.tools.builtin.whisper_stt._detect_whisper_engine",
            lambda: "faster-whisper",
        )
        assert check_whisper({"whisper": {"enabled": True}}).status == "ok"


class TestDoctorCommand:
    def test_healthy_config_exits_zero(self, isolated_dirs, capsys):
        with _serve(_ModelsHandler) as (url, _port):
            cfg_path = isolated_dirs / "prometheus.yaml"
            cfg_path.write_text(yaml.safe_dump({
                "model": {"provider": "llama_cpp", "base_url": url},
                "web": {"enabled": True, "api_port": _free_port()},
            }), encoding="utf-8")
            rc = run_doctor_command(argparse.Namespace(
                config=str(cfg_path), no_scan=True, timeout=3.0,
            ))
        out = capsys.readouterr().out
        assert rc == 0
        assert "✓" in out
        assert "RESULT" in out

    def test_unreachable_server_exits_nonzero(self, isolated_dirs, capsys):
        cfg_path = isolated_dirs / "prometheus.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "model": {"provider": "llama_cpp",
                      "base_url": f"http://127.0.0.1:{_free_port()}"},
            "web": {"enabled": True, "api_port": _free_port()},
        }), encoding="utf-8")
        rc = run_doctor_command(argparse.Namespace(
            config=str(cfg_path), no_scan=True, timeout=0.3,
        ))
        out = capsys.readouterr().out
        assert rc == 1
        assert "✗" in out

    def test_missing_config_exits_nonzero(self, isolated_dirs):
        rc = run_doctor_command(argparse.Namespace(
            config=str(isolated_dirs / "absent.yaml"), no_scan=True, timeout=0.3,
        ))
        assert rc == 1

    def test_report_includes_fix_lines(self, isolated_dirs, capsys):
        from prometheus.infra.doctor import DiagnosticCheck
        report = render_report([
            DiagnosticCheck(name="Thing", category="platform",
                            status="error", message="broken", fix="do the fix"),
        ])
        assert "✗ Thing: broken" in report
        assert "fix: do the fix" in report
