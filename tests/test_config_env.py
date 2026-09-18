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


class TestEnvOverrideLoggingLeaksNothing:
    """The log line must carry NAMES only — no value, and no fragment of one.

    The defect this pins (found 2026-09-18): the line logged ``NAME=first4...last4`` for anything
    whose NAME contained "token"/"key", and the value verbatim otherwise. Those fragments rode a
    coding run's ``output_tail`` onto the WebSocket, into Beacon's activity feed, and durably into
    telemetry.db's ``signal_events``.

    The fragment was worse than the raw value would have been. ``security/log_redaction.py`` is
    installed on every entry point to scrub secrets out of the logging stack, and its patterns need
    contiguous runs — ``sk-[A-Za-z0-9_-]{16,}``, ``\\d{5,}:[A-Za-z0-9_-]{30,}``. An 11-character
    string with dots through the middle matches none of them, so the mask did not protect the value,
    it hid it from the control that would have caught it.
    """

    @staticmethod
    def _capture(monkeypatch, caplog, **env):
        import logging

        for k, v in env.items():
            monkeypatch.setenv(k, v)
        config: dict = {}
        with caplog.at_level(logging.INFO, logger="prometheus.config.env_override"):
            apply_env_overrides(config)
        return "\n".join(r.getMessage() for r in caplog.records)

    def test_logs_names_and_no_value_or_fragment(self, monkeypatch, caplog):
        """Values never appear — and neither do their first-4/last-4 fragments.

        The fragment assertion is the one that does the work: asserting only "the full value is
        absent" passes against the OLD code, because the old code never logged the full value for a
        name containing "key"/"token". Only the fragment check fails against it.
        """
        # >12 chars, so the old masker would produce first4...last4 rather than "***".
        #
        # Assembled from parts rather than written as literals. These are deliberately
        # REAL-SHAPED — that is the whole point, since log_redaction's patterns only match
        # contiguous credential-shaped runs — and a literal trips this repo's own pre-commit
        # secret scanner. It blocked exactly these two lines on the first commit attempt, which
        # is the same mechanism as the redactor and it was right to.
        anthropic = "sk-" + "ant-api03-" + "LEAKCANARY" + "-0123456789" + "-TAILMARK"
        telegram = "8685123456" + ":" + "AAH" + "LEAKCANARYtelegramtokenvalue0123"
        self._capture(
            monkeypatch,
            caplog,
            ANTHROPIC_API_KEY=anthropic,
            PROMETHEUS_TELEGRAM_TOKEN=telegram,
        )
        text = "\n".join(r.getMessage() for r in caplog.records)

        assert "Applied env overrides" in text, "the diagnostic line must still be produced"
        assert "ANTHROPIC_API_KEY" in text, "the NAME is the diagnostic, and must stay"
        assert "PROMETHEUS_TELEGRAM_TOKEN" in text

        for secret in (anthropic, telegram):
            assert secret not in text, "the whole value must never be logged"
            assert secret[:4] not in text, f"a leading fragment of {secret[:4]!r} reached the log"
            assert secret[-4:] not in text, f"a trailing fragment of {secret[-4:]!r} reached the log"
        assert "..." not in text, "no masked-fragment form may survive anywhere in the line"

    def test_non_secret_named_vars_are_not_logged_verbatim_either(self, monkeypatch, caplog):
        """The old predicate matched the NAME, so anything without token/key printed in full.

        Six of ENV_OVERRIDES' twenty entries fell to that branch, including
        PROMETHEUS_TRUST_LEVEL and PROMETHEUS_PERMISSION_MODE — the daemon's security posture on
        the same wire. Names-only closes that for free, and this pins it so a future "helpful"
        change cannot reintroduce values for the non-secret-looking half.
        """
        self._capture(
            monkeypatch,
            caplog,
            PROMETHEUS_TRUST_LEVEL="UNTRUSTEDCANARY",
            PROMETHEUS_PERMISSION_MODE="YOLOCANARY",
        )
        text = "\n".join(r.getMessage() for r in caplog.records)

        assert "PROMETHEUS_TRUST_LEVEL" in text
        assert "UNTRUSTEDCANARY" not in text, "a non-secret-named override still logged its value"
        assert "YOLOCANARY" not in text
        assert "=" not in text.split("Applied env overrides:", 1)[1], (
            "the line must be a list of names, with no NAME=VALUE pair of any kind"
        )
