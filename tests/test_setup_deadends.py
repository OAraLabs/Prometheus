"""Dead-end fixes (Phase 0, item 5) + wizard papercuts (item 6).

Contract under test: NO setup path may write a config that is known to
be broken. Every no-server path either establishes a working target
(remote URL, cloud provider with a real key) or exits cleanly having
written nothing.
"""

from __future__ import annotations

import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import yaml

from prometheus.cli import init as init_mod
from prometheus.cli.init import run_init


class _OllamaHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/api/tags":
            body = json.dumps(
                {"models": [{"name": "llama3:8b"}]}
            ).encode()
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


@contextmanager
def _serve(handler_cls):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = HTTPServer(("127.0.0.1", port), handler_cls)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()
        srv.server_close()


_NO_SERVERS = [{
    "name": "ghost", "url": "http://127.0.0.1:9",  # discard port — closed
    "models_path": "/v1/models", "provider": "test",
}]

_SECRET_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PROMETHEUS_API_TOKEN")


@pytest.fixture(autouse=True)
def _isolate_secret_env():
    """Save/restore provider keys around every test.

    The cloud dead-end path exports the pasted key into ``os.environ``
    (for the smoke test) by design; ``monkeypatch.delenv(raising=False)``
    on an absent var records nothing and would NOT undo that export.
    """
    import os
    saved = {k: os.environ.pop(k, None) for k in _SECRET_KEYS}
    yield
    for k, v in saved.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v


@pytest.fixture
def scripted_input(monkeypatch):
    """Feed a scripted sequence of answers to init's _prompt()."""
    answers: list[str] = []

    def fake_prompt(label: str, default: str = "") -> str:
        assert answers, f"unexpected prompt: {label!r}"
        value = answers.pop(0)
        return value if value != "" else default

    monkeypatch.setattr(init_mod, "_prompt", fake_prompt)
    return answers


class TestFastPathDeadEnds:
    def test_noninteractive_no_server_writes_nothing(self, tmp_path, capsys):
        result = run_init(
            noninteractive=True, target_dir=tmp_path,
            timeout=0.1, candidates=_NO_SERVERS,
        )
        assert result is None
        assert not (tmp_path / "prometheus.yaml").exists()
        out = capsys.readouterr().out
        assert "No config was written" in out
        assert "ollama.com/install.sh" in out

    def test_interactive_instructions_path_writes_nothing(
        self, tmp_path, scripted_input, capsys,
    ):
        scripted_input.extend(["3"])  # dead-end menu: instructions + exit
        result = run_init(
            noninteractive=False, target_dir=tmp_path,
            timeout=0.1, candidates=_NO_SERVERS,
        )
        assert result is None
        assert not (tmp_path / "prometheus.yaml").exists()
        assert "ollama.com/install.sh" in capsys.readouterr().out

    def test_interactive_remote_url_path_writes_valid_config(
        self, tmp_path, scripted_input,
    ):
        with _serve(_OllamaHandler) as url:
            scripted_input.extend([
                "1",   # dead-end menu: remote URL
                url,   # the URL
                "1",   # gateway: CLI only
            ])
            result = run_init(
                noninteractive=False, target_dir=tmp_path,
                timeout=0.1, candidates=_NO_SERVERS,
            )
        assert result is not None
        written = yaml.safe_load((tmp_path / "prometheus.yaml").read_text())
        assert written["model"]["provider"] == "ollama"
        assert written["model"]["base_url"] == url
        assert written["model"]["model"] == "llama3:8b"
        assert written["web"]["enabled"] is True

    def test_interactive_remote_url_unreachable_never_written(
        self, tmp_path, scripted_input,
    ):
        scripted_input.extend([
            "1",                      # dead-end menu: remote URL
            "http://127.0.0.1:9",     # dead URL
            "",                       # retry? default N
            "3",                      # back at menu: instructions + exit
        ])
        result = run_init(
            noninteractive=False, target_dir=tmp_path,
            timeout=0.1, candidates=_NO_SERVERS,
        )
        assert result is None
        assert not (tmp_path / "prometheus.yaml").exists()

    def test_interactive_cloud_path_writes_key_to_env_file(
        self, tmp_path, scripted_input, monkeypatch,
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        scripted_input.extend([
            "2",                        # dead-end menu: cloud provider
            "1",                        # anthropic
            "sk-ant-test-1234567890",   # pasted key
            "",                         # model: default
            "1",                        # gateway: CLI only
        ])
        result = run_init(
            noninteractive=False, target_dir=tmp_path,
            timeout=0.1, candidates=_NO_SERVERS,
        )
        assert result is not None
        written = yaml.safe_load((tmp_path / "prometheus.yaml").read_text())
        assert written["model"]["provider"] == "anthropic"
        assert written["model"]["api_key_env"] == "ANTHROPIC_API_KEY"
        assert "base_url" not in written["model"]
        # Key persisted to the (target-dir-confined) env file, not the yaml.
        env_body = (tmp_path / "env").read_text()
        assert "ANTHROPIC_API_KEY=sk-ant-test-1234567890" in env_body
        assert "sk-ant-test" not in (tmp_path / "prometheus.yaml").read_text()

    def test_cloud_path_without_key_refuses_to_write(
        self, tmp_path, scripted_input, monkeypatch,
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        scripted_input.extend([
            "2",   # cloud provider
            "1",   # anthropic
            "",    # no key pasted, env unset
            "3",   # back at the menu: instructions + exit
        ])
        result = run_init(
            noninteractive=False, target_dir=tmp_path,
            timeout=0.1, candidates=_NO_SERVERS,
        )
        assert result is None
        assert not (tmp_path / "prometheus.yaml").exists()

    def test_detected_server_still_writes_web_enabled(self, tmp_path):
        with _serve(_OllamaHandler) as url:
            candidates = [{
                "name": "Ollama", "url": url,
                "models_path": "/api/tags", "provider": "ollama",
            }]
            result = run_init(
                noninteractive=True, target_dir=tmp_path,
                timeout=2.0, candidates=candidates,
            )
        assert result is not None
        assert result["web"]["enabled"] is True
        assert result["model"]["base_url"] == url


class TestWizardPapercuts:
    def test_apply_wizard_fields_enables_web(self):
        from prometheus.setup_wizard import SetupWizard
        wizard = SetupWizard()
        cfg: dict = {}
        wizard._apply_wizard_fields(cfg)
        assert cfg["web"]["enabled"] is True
        assert cfg["web"]["api_port"] == 8005

    def test_apply_wizard_fields_respects_explicit_off(self):
        from prometheus.setup_wizard import SetupWizard
        wizard = SetupWizard()
        cfg: dict = {"web": {"enabled": False}}
        wizard._apply_wizard_fields(cfg)
        assert cfg["web"]["enabled"] is False

    def test_migration_offer_skipped_when_nothing_detected(self, monkeypatch):
        """No Hermes/OpenClaw on disk → the wizard must not prompt at all."""
        from prometheus.setup_wizard import SetupWizard

        monkeypatch.setattr(
            "prometheus.cli.migrate.detect_sources", lambda: {},
        )

        def explode(*args, **kwargs):  # any prompt = the offer fired
            raise AssertionError("migration offer prompted with no sources")

        monkeypatch.setattr("prometheus.setup_wizard._input", explode)
        SetupWizard()._offer_migration()  # must return silently

    def test_wizard_cloud_key_paste_writes_env_file(
        self, tmp_path, monkeypatch,
    ):
        from prometheus import setup_wizard as wizard_mod

        env_path = tmp_path / "env"
        monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(env_path))
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        answers = iter(["sk-ant-wizard-test-000000", "1"])  # key, model choice
        monkeypatch.setattr(
            wizard_mod, "_input", lambda label, default="": next(answers),
        )
        wizard = wizard_mod.SetupWizard()
        wizard._step_cloud_provider("anthropic")
        assert wizard._api_key_env == "ANTHROPIC_API_KEY"
        assert "ANTHROPIC_API_KEY=sk-ant-wizard-test-000000" in env_path.read_text()
