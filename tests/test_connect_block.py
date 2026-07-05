"""Wizard "Connect a client (Beacon)" end screen (Onboarding Phase 1, item 1).

The block itself (address from the machine's hostname — never
hardcoded — plus the token-or-mint-note) and its presence at the end of
BOTH wizard paths: the fast/noninteractive path (cli.init.run_init) and
the rich wizard's summary (SetupWizard._print_summary).
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from prometheus.config.api_token import (
    TOKEN_ENV_VAR,
    format_connect_client_block,
)
from prometheus.config.env_file import set_env_value


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / "env"
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(path))
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    return path


# ---------------------------------------------------------------------------
# The block itself
# ---------------------------------------------------------------------------


class TestFormatConnectClientBlock:
    def test_address_uses_hostname_never_hardcoded(self, env_file):
        block = format_connect_client_block({})
        assert f"{socket.gethostname()}:8005" in block
        assert "Tailscale / LAN address" in block
        assert "CONNECT A CLIENT" in block

    def test_port_read_from_config(self, env_file):
        block = format_connect_client_block({"web": {"api_port": 9123}})
        assert f"{socket.gethostname()}:9123" in block
        assert ":8005" not in block

    def test_existing_env_file_token_is_printed(self, env_file):
        set_env_value(TOKEN_ENV_VAR, "tok-abc-123", env_file)
        block = format_connect_client_block({})
        assert "tok-abc-123" in block
        assert str(env_file) in block

    def test_no_token_prints_mint_note_not_a_value(self, env_file):
        block = format_connect_client_block({})
        assert "minted on first daemon start" in block
        assert "prometheus token show" in block

    def test_never_writes_the_env_file(self, env_file):
        format_connect_client_block({})
        assert not env_file.exists()  # print-only — no side effects


# ---------------------------------------------------------------------------
# Fast / noninteractive path (prometheus setup --noninteractive)
# ---------------------------------------------------------------------------


class _FakeLlamaCpp(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = json.dumps({"data": [{"id": "test-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs):  # silence
        pass


class TestFastPathEndScreen:
    def test_noninteractive_prints_connect_block(self, tmp_path, env_file, capsys):
        from prometheus.cli.init import run_init

        srv = HTTPServer(("127.0.0.1", 0), _FakeLlamaCpp)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}"
            config = run_init(
                noninteractive=True, target_dir=tmp_path, timeout=2.0,
                candidates=[{
                    "name": "fake", "url": url,
                    "models_path": "/v1/models", "provider": "llama_cpp",
                }],
            )
        finally:
            srv.shutdown()
            srv.server_close()
        assert config is not None
        out = capsys.readouterr().out
        assert "CONNECT A CLIENT" in out
        assert f"{socket.gethostname()}:8005" in out
        assert "minted on first daemon start" in out  # no token yet


# ---------------------------------------------------------------------------
# Rich path (SetupWizard summary)
# ---------------------------------------------------------------------------


class TestRichPathEndScreen:
    def test_print_summary_ends_with_connect_block(
        self, tmp_path, env_file, monkeypatch, capsys,
    ):
        import prometheus.setup_wizard as wizard_mod

        cfg_path = tmp_path / "prometheus.yaml"
        cfg_path.write_text("web:\n  enabled: true\n  api_port: 9321\n")
        monkeypatch.setattr(wizard_mod, "_config_target", lambda: cfg_path)
        set_env_value(TOKEN_ENV_VAR, "tok-rich-path", tmp_path / "env")

        wizard = wizard_mod.SetupWizard()
        wizard._print_summary(True)

        out = capsys.readouterr().out
        assert "CONNECT A CLIENT" in out
        assert f"{socket.gethostname()}:9321" in out  # port from the written config
        assert "tok-rich-path" in out
        # The block is the LAST thing on screen (spec: end screen).
        assert out.rstrip().endswith("=" * 68)

    def test_print_summary_without_config_still_prints_block(
        self, tmp_path, env_file, monkeypatch, capsys,
    ):
        import prometheus.setup_wizard as wizard_mod

        monkeypatch.setattr(
            wizard_mod, "_config_target", lambda: tmp_path / "missing.yaml"
        )
        wizard = wizard_mod.SetupWizard()
        wizard._print_summary(False)
        out = capsys.readouterr().out
        assert "CONNECT A CLIENT" in out
        assert "minted on first daemon start" in out


class TestBeaconDownloadPointer:
    """A newcomer must be told WHERE to get Beacon, not just that it exists."""

    def test_connect_block_has_download_url(self):
        from prometheus.config.api_token import (
            BEACON_DOWNLOAD_URL,
            format_connect_client_block,
        )
        assert BEACON_DOWNLOAD_URL in format_connect_client_block({})

    def test_pairing_banner_has_download_url(self):
        from prometheus.config.api_token import BEACON_DOWNLOAD_URL
        from prometheus.web.setup_server import format_pairing_banner
        assert BEACON_DOWNLOAD_URL in format_pairing_banner("123456", 8005)
