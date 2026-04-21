"""Tests for Sprint 11: env var overrides + secret file loading."""

from __future__ import annotations

import os
import textwrap

import pytest

from prometheus.config.env_override import apply_env_overrides, read_secret_file


class TestEnvOverrides:
    def test_env_var_overrides_config(self, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_TELEGRAM_TOKEN", "from_env")
        config = {"gateway": {"telegram_token": "from_yaml"}}
        apply_env_overrides(config)
        assert config["gateway"]["telegram_token"] == "from_env"

    def test_env_var_creates_missing_keys(self, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_TELEGRAM_TOKEN", "new_token")
        config = {}
        apply_env_overrides(config)
        assert config["gateway"]["telegram_token"] == "new_token"

    def test_trust_level_coercion(self, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_TRUST_LEVEL", "3")
        config = {"security": {"trust_level": 1}}
        apply_env_overrides(config)
        assert config["security"]["trust_level"] == 3
        assert isinstance(config["security"]["trust_level"], int)

    def test_model_override(self, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_MODEL", "gemma4-26b")
        config = {"model": {"model": "qwen3.5-32b"}}
        apply_env_overrides(config)
        assert config["model"]["model"] == "gemma4-26b"

    def test_provider_url_override(self, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_LLAMA_CPP_URL", "http://gpu:8080")
        config = {}
        apply_env_overrides(config)
        assert config["providers"]["llama_cpp"]["base_url"] == "http://gpu:8080"

    def test_no_env_vars_is_noop(self):
        config = {"gateway": {"telegram_token": "original"}}
        apply_env_overrides(config)
        assert config["gateway"]["telegram_token"] == "original"


class TestSecretFile:
    def test_reads_secret(self, tmp_path):
        secret_file = tmp_path / "token.txt"
        secret_file.write_text("  my_secret_token  \n")
        secret = read_secret_file(str(secret_file), "test")
        assert secret == "my_secret_token"

    def test_rejects_symlink(self, tmp_path):
        real_file = tmp_path / "real.txt"
        real_file.write_text("secret")
        link = tmp_path / "link.txt"
        link.symlink_to(real_file)
        secret = read_secret_file(str(link), "test")
        assert secret is None

    def test_rejects_empty(self, tmp_path):
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("   \n")
        secret = read_secret_file(str(empty_file), "test")
        assert secret is None

    def test_rejects_missing(self, tmp_path):
        secret = read_secret_file(str(tmp_path / "nope.txt"), "test")
        assert secret is None

    def test_rejects_oversized(self, tmp_path):
        big_file = tmp_path / "big.txt"
        big_file.write_text("x" * 20000)
        secret = read_secret_file(str(big_file), "test", max_bytes=16384)
        assert secret is None

    def test_secret_file_env_var_override(self, tmp_path, monkeypatch):
        secret_file = tmp_path / "tg_token.txt"
        secret_file.write_text("secret_from_file\n")
        monkeypatch.setenv("PROMETHEUS_TELEGRAM_TOKEN_FILE", str(secret_file))
        config = {}
        apply_env_overrides(config)
        assert config["gateway"]["telegram_token"] == "secret_from_file"

    def test_direct_env_overrides_secret_file(self, tmp_path, monkeypatch):
        """Direct env var takes precedence over secret file."""
        secret_file = tmp_path / "tg_token.txt"
        secret_file.write_text("from_file\n")
        monkeypatch.setenv("PROMETHEUS_TELEGRAM_TOKEN_FILE", str(secret_file))
        monkeypatch.setenv("PROMETHEUS_TELEGRAM_TOKEN", "from_env_direct")
        config = {}
        apply_env_overrides(config)
        # Direct env var wins (applied second)
        assert config["gateway"]["telegram_token"] == "from_env_direct"
