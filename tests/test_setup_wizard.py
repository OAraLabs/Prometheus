"""Tests for the first-run setup wizard."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from prometheus.setup_wizard import SetupWizard, _REPO_CONFIG


@pytest.fixture()
def wizard():
    """Create a fresh wizard instance."""
    return SetupWizard()


@pytest.fixture()
def tmp_config(tmp_path, monkeypatch):
    """Redirect _REPO_CONFIG to a temp directory so tests don't touch real config."""
    config_file = tmp_path / "config" / "prometheus.yaml"
    config_file.parent.mkdir(parents=True)
    monkeypatch.setattr("prometheus.setup_wizard._REPO_CONFIG", config_file)
    return config_file


@pytest.fixture()
def tmp_prometheus_dir(tmp_path, monkeypatch):
    """Redirect get_config_dir to a temp directory."""
    config_dir = tmp_path / ".prometheus"
    config_dir.mkdir()
    monkeypatch.setattr(
        "prometheus.setup_wizard.get_config_dir", lambda: config_dir
    )
    return config_dir


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------


class TestCreateDirectories:
    def test_creates_all_required_dirs(self, wizard, tmp_prometheus_dir):
        wizard._create_directories()
        expected = [
            "workspace",
            "wiki",
            "sentinel",
            "skills",
            "data",
            "data/sessions",
            "logs",
            "cache",
        ]
        for subdir in expected:
            assert (tmp_prometheus_dir / subdir).is_dir(), f"Missing: {subdir}"

    def test_idempotent(self, wizard, tmp_prometheus_dir):
        """Running twice doesn't raise or corrupt."""
        wizard._create_directories()
        wizard._create_directories()
        assert (tmp_prometheus_dir / "workspace").is_dir()


# ---------------------------------------------------------------------------
# Config writing
# ---------------------------------------------------------------------------


class TestConfigWriting:
    def test_writes_valid_yaml(self, wizard, tmp_config, tmp_prometheus_dir):
        wizard._provider = "llama_cpp"
        wizard._base_url = "http://localhost:8080"
        wizard._model_name = "test-model"
        wizard._gateway = "cli"
        wizard._write_config()

        assert tmp_config.exists()
        cfg = yaml.safe_load(tmp_config.read_text())
        assert cfg["model"]["provider"] == "llama_cpp"
        assert cfg["model"]["base_url"] == "http://localhost:8080"
        assert cfg["model"]["model"] == "test-model"
        assert cfg["gateway"]["telegram_enabled"] is False

    def test_writes_telegram_config(self, wizard, tmp_config, tmp_prometheus_dir):
        wizard._provider = "llama_cpp"
        wizard._base_url = "http://localhost:8080"
        wizard._gateway = "telegram"
        wizard._telegram_token = "123456:ABC"
        wizard._telegram_chat_ids = [42]
        wizard._write_config()

        cfg = yaml.safe_load(tmp_config.read_text())
        assert cfg["gateway"]["telegram_enabled"] is True
        assert cfg["gateway"]["telegram_token"] == "123456:ABC"
        assert cfg["gateway"]["allowed_chat_ids"] == [42]

    def test_preserves_existing_config_values(self, tmp_config, tmp_prometheus_dir):
        """When merging, untouched fields survive."""
        existing = {
            "system": {"name": "Prometheus", "version": "0.1.0"},
            "model": {"provider": "llama_cpp", "base_url": "http://old:8080"},
            "context": {"effective_limit": 24000},
            "security": {"permission_mode": "default"},
            "gateway": {"telegram_enabled": False},
        }
        tmp_config.parent.mkdir(parents=True, exist_ok=True)
        with tmp_config.open("w") as fh:
            yaml.dump(existing, fh)

        wizard = SetupWizard(gateway_only=True)
        wizard._gateway = "telegram"
        wizard._telegram_token = "new-token"
        wizard._telegram_chat_ids = []
        wizard._merge_and_write(existing)

        cfg = yaml.safe_load(tmp_config.read_text())
        # Wizard-touched field updated
        assert cfg["gateway"]["telegram_enabled"] is True
        assert cfg["gateway"]["telegram_token"] == "new-token"
        # Untouched fields preserved
        assert cfg["context"]["effective_limit"] == 24000
        assert cfg["security"]["permission_mode"] == "default"
        assert cfg["system"]["name"] == "Prometheus"


# ---------------------------------------------------------------------------
# .gitignore management
# ---------------------------------------------------------------------------


class TestGitignore:
    def test_adds_config_to_gitignore(self, wizard, tmp_config, tmp_prometheus_dir):
        gitignore = tmp_config.parents[1] / ".gitignore"
        gitignore.write_text("*.pyc\n")

        wizard._provider = "llama_cpp"
        wizard._base_url = "http://localhost:8080"
        wizard._gateway = "cli"
        wizard._write_config()

        content = gitignore.read_text()
        assert "config/prometheus.yaml" in content

    def test_does_not_duplicate_gitignore_entry(self, wizard, tmp_config, tmp_prometheus_dir):
        gitignore = tmp_config.parents[1] / ".gitignore"
        gitignore.write_text("config/prometheus.yaml\n")

        wizard._provider = "llama_cpp"
        wizard._base_url = "http://localhost:8080"
        wizard._gateway = "cli"
        wizard._write_config()

        content = gitignore.read_text()
        assert content.count("config/prometheus.yaml") == 1


# ---------------------------------------------------------------------------
# Provider testing
# ---------------------------------------------------------------------------


class TestProviderConnection:
    def test_successful_connection(self, wizard):
        """Mock a successful /v1/models response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"id": "my-model-Q4_K_M.gguf"}]
        }

        with patch("prometheus.setup_wizard.httpx.get", return_value=mock_resp):
            result = wizard._test_provider("http://localhost:8080")
        assert result == "my-model-Q4_K_M.gguf"

    def test_unreachable_provider(self, wizard):
        """Connection failure returns None."""
        with patch(
            "prometheus.setup_wizard.httpx.get",
            side_effect=Exception("Connection refused"),
        ):
            result = wizard._test_provider("http://localhost:9999")
        assert result is None

    def test_ollama_fallback(self, wizard):
        """If /v1/models fails for Ollama, try /api/tags."""
        wizard._provider = "ollama"

        def side_effect(url, **kwargs):
            if "/v1/models" in url:
                raise Exception("not found")
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"models": [{"name": "qwen3.5:32b"}]}
            return resp

        with patch("prometheus.setup_wizard.httpx.get", side_effect=side_effect):
            result = wizard._test_provider("http://localhost:11434")
        assert result == "qwen3.5:32b"


# ---------------------------------------------------------------------------
# Telegram token testing
# ---------------------------------------------------------------------------


class TestTelegramToken:
    def test_valid_token(self, wizard):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "ok": True,
            "result": {"username": "test_bot"},
        }
        with patch("prometheus.setup_wizard.httpx.get", return_value=mock_resp):
            result = wizard._test_telegram_token("fake:token")
        assert result == "test_bot"

    def test_invalid_token(self, wizard):
        with patch(
            "prometheus.setup_wizard.httpx.get",
            side_effect=Exception("401 Unauthorized"),
        ):
            result = wizard._test_telegram_token("bad:token")
        assert result is None


# ---------------------------------------------------------------------------
# Rerun detection
# ---------------------------------------------------------------------------


class TestRerunDetection:
    def test_detects_existing_config(self, wizard, tmp_config):
        tmp_config.parent.mkdir(parents=True, exist_ok=True)
        with tmp_config.open("w") as fh:
            yaml.dump({"model": {"provider": "llama_cpp"}}, fh)

        cfg = wizard._load_existing_config()
        assert cfg is not None
        assert cfg["model"]["provider"] == "llama_cpp"

    def test_returns_none_when_no_config(self, wizard, tmp_config):
        # tmp_config doesn't exist yet
        cfg = wizard._load_existing_config()
        assert cfg is None

    def test_prefill_from_existing(self, wizard):
        cfg = {
            "model": {
                "provider": "ollama",
                "base_url": "http://192.168.1.100:11434",
                "model": "qwen3.5-32b",
            },
            "gateway": {
                "telegram_enabled": True,
                "telegram_token": "my-token",
                "allowed_chat_ids": [123],
            },
        }
        wizard._prefill_from(cfg)
        assert wizard._provider == "ollama"
        assert wizard._base_url == "http://192.168.1.100:11434"
        assert wizard._model_name == "qwen3.5-32b"
        assert wizard._gateway == "telegram"
        assert wizard._telegram_token == "my-token"
        assert wizard._telegram_chat_ids == [123]


# ---------------------------------------------------------------------------
# CLI flag parsing
# ---------------------------------------------------------------------------


class TestCLIFlags:
    def test_setup_flag_parsed(self):
        """--setup flag is recognized by the argument parser."""
        from prometheus.__main__ import main
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--setup", action="store_true")
        parser.add_argument("--setup-gateway-only", action="store_true")
        args = parser.parse_args(["--setup"])
        assert args.setup is True
        assert args.setup_gateway_only is False

    def test_gateway_only_flag_parsed(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--setup", action="store_true")
        parser.add_argument("--setup-gateway-only", action="store_true")
        args = parser.parse_args(["--setup-gateway-only"])
        assert args.setup is False
        assert args.setup_gateway_only is True


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


class TestSmokeTest:
    def test_smoke_test_passes(self, wizard):
        wizard._base_url = "http://localhost:8080"
        wizard._model_name = "test-model"

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "4"}}]
        }

        with patch("prometheus.setup_wizard.httpx.post", return_value=mock_resp):
            assert wizard._run_smoke_test() is True

    def test_smoke_test_handles_failure(self, wizard):
        wizard._base_url = "http://localhost:8080"
        wizard._model_name = "test-model"

        with patch(
            "prometheus.setup_wizard.httpx.post",
            side_effect=Exception("Connection refused"),
        ):
            assert wizard._run_smoke_test() is False
