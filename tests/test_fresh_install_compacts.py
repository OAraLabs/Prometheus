"""A config Prometheus writes has the context compactor on, and a config already written without it is repaired at boot.

Every writer left ``compaction`` out: setup mode's ``configure`` (Beacon's first run, local and cloud),
``oara setup --fast`` / ``--noninteractive`` (local and cloud), and the interactive wizard (it starts a new install from
``{}``, not the template). ``ContextCompactor.from_config`` returns None without ``compaction.enabled``, so every such
install has run without the assembly-time compactor since it shipped (2026-06-11), while the template ships ``true``.
#325 kept the absence failing closed on the premise that fresh installs carry the key from the template; they never
did.

Red first. Two fixes, pinned separately:
1. the writers write ``compaction: {enabled: true}``, the template's own value;
2. a config that has no ``compaction.enabled`` follows the template at boot (with a WARNING naming the line to write),
   so existing installs are repaired on upgrade without editing their files. An explicit ``false`` stays off.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.api_token import TOKEN_ENV_VAR  # noqa: E402
from prometheus.context.compactor import ContextCompactor  # noqa: E402
from prometheus.web.setup_server import PairingState, SetupModeState, create_setup_app  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "config" / "prometheus.yaml.default"
CODE = "042999"


def _shipped() -> dict:
    return yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))["compaction"]


@pytest.fixture
def home(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(tmp_path / "env"))
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "confdir"))
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    os.environ.pop(TOKEN_ENV_VAR, None)


class _Models(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = json.dumps({"object": "list", "data": [{"id": "qwen3-8b", "object": "model"}]}).encode()
        self.send_response(200 if self.path == "/v1/models" else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.path == "/v1/models":
            self.wfile.write(body)

    def log_message(self, *args, **kwargs):
        pass


@contextmanager
def _server():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = HTTPServer(("127.0.0.1", port), _Models)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()
        srv.server_close()


def _configure(home: Path, body: dict) -> dict:
    client = TestClient(create_setup_app(PairingState(code=CODE), api_port=8123, ws_port=8124, state=SetupModeState()))
    token = client.post("/api/setup/pair", json={"code": CODE}).json()["token"]
    resp = client.post("/api/setup/configure", json=body, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text
    return yaml.safe_load((home / "confdir" / "prometheus.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. every writer writes it
# ---------------------------------------------------------------------------


class TestEveryWriterWritesIt:
    def test_the_template_ships_it_on(self):
        assert _shipped().get("enabled") is True

    def test_setup_mode_configure_local(self, home):
        with _server() as url:
            cfg = _configure(home, {"provider": "llama_cpp", "base_url": url, "model": "qwen3-8b"})
        assert cfg.get("compaction") == {"enabled": True}

    def test_setup_mode_configure_cloud(self, home):
        cfg = _configure(home, {"provider": "openrouter", "api_key": "sk-or-v1-" + "ab12" * 8})
        assert cfg.get("compaction") == {"enabled": True}

    def test_setup_fast_local(self, home):
        from prometheus.cli.init import run_init

        with _server() as url:
            cfg = run_init(noninteractive=True, target_dir=home / "fast", candidates=[
                {"name": "llama.cpp", "url": url, "models_path": "/v1/models", "provider": "llama_cpp"}])
        assert cfg is not None and cfg.get("compaction") == {"enabled": True}

    def test_setup_fast_cloud(self, home, monkeypatch):
        from prometheus.cli.init import run_init

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + "cd34" * 8)
        cfg = run_init(noninteractive=True, target_dir=home / "fast-cloud", provider="openrouter")
        assert cfg is not None and cfg.get("compaction") == {"enabled": True}

    def test_the_interactive_wizard_on_a_new_install(self, home, monkeypatch):
        import prometheus.setup_wizard as wizard_mod

        path = home / "wizard" / "prometheus.yaml"
        path.parent.mkdir()
        monkeypatch.setattr(wizard_mod, "_config_target", lambda: path)
        wizard_mod.SetupWizard()._write_config()
        assert yaml.safe_load(path.read_text(encoding="utf-8")).get("compaction") == {"enabled": True}

    def test_the_wizard_keeps_an_existing_choice(self, home, monkeypatch):
        """Running the wizard again over a config that says false must not switch it on."""
        import prometheus.setup_wizard as wizard_mod

        path = home / "wizard2" / "prometheus.yaml"
        path.parent.mkdir()
        path.write_text(yaml.safe_dump({"compaction": {"enabled": False}, "web": {"enabled": True}}), encoding="utf-8")
        monkeypatch.setattr(wizard_mod, "_config_target", lambda: path)
        wizard_mod.SetupWizard()._merge_and_write(yaml.safe_load(path.read_text(encoding="utf-8")))
        assert yaml.safe_load(path.read_text(encoding="utf-8"))["compaction"] == {"enabled": False}

    def test_what_a_writer_writes_builds_a_compactor(self, home):
        with _server() as url:
            cfg = _configure(home, {"provider": "llama_cpp", "base_url": url, "model": "qwen3-8b"})
        assert ContextCompactor.from_config(cfg, provider=MagicMock(), model="qwen3-8b") is not None


# ---------------------------------------------------------------------------
# 2. a config already written without it is repaired at boot
# ---------------------------------------------------------------------------


class TestExistingInstallsAreRepairedAtBoot:
    def _follow(self, config: dict) -> dict:
        from prometheus.daemon import _follow_shipped_compaction

        _follow_shipped_compaction(config)
        return config

    def test_absent_follows_the_template(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = self._follow({"model": {"provider": "openrouter"}})
        assert cfg["compaction"] == {"enabled": True}
        assert "compaction.enabled" in caplog.text and "no compactor is constructed" not in caplog.text

    def test_a_section_without_the_key_follows_the_template_and_keeps_its_other_keys(self):
        cfg = self._follow({"compaction": {"threshold_pct": 0.6}})
        assert cfg["compaction"] == {"threshold_pct": 0.6, "enabled": True}

    def test_an_explicit_false_is_a_decision(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = self._follow({"compaction": {"enabled": False}})
        assert cfg["compaction"] == {"enabled": False}
        assert "compaction" not in caplog.text

    def test_the_boot_warning_no_longer_says_it_is_off(self, caplog):
        from prometheus.daemon import _warn_absent_gating_keys

        with caplog.at_level(logging.WARNING):
            _warn_absent_gating_keys({"web": {"enabled": True}, "trajectory_export": {"enabled": False}})
        assert "no compactor is constructed" not in caplog.text

    def test_the_daemon_repairs_after_the_pins_and_before_the_compactor_is_built(self):
        import prometheus.daemon as daemon_mod

        source = Path(daemon_mod.__file__).read_text(encoding="utf-8")
        pins = source.index('apply_config_pins(config, get_config_dir()')
        repair = source.index("_follow_shipped_compaction(config)")
        build = source.index("ContextCompactor.from_config(")
        assert pins < repair < build
