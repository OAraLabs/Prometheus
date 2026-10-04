"""A credential is shown by its last 4 characters, never its head.

``CredentialPool`` logged ``key[:8]...key[-4:]``: twelve characters of a live
API key in every rotation, dead-key and revive line. The setup wizard printed
``token[:4]...token[-4:]`` for the Telegram, Slack and Discord tokens; for a
Telegram token the head is the bot id. Both now show ``...`` plus the last 4,
and a short value (8 characters or fewer) shows nothing of itself.
"""

from __future__ import annotations

import logging

import pytest

from prometheus.providers.credential_pool import CredentialPool

# Fake credentials, letters only, built by concatenation (the pre-commit
# scanner reads staged files).
HEAD = "sk" + "-ant-" + "abcdefgh"
KEY = HEAD + "ijklmnopqrstuv" + "wxyz"
OTHER = "sk" + "-ant-" + "zyxwvutsrqponmlkjihgfe" + "dcba"


def _assert_tail_only(text: str, key: str) -> None:
    assert f"...{key[-4:]}" in text, text
    # No 4-character run of the key other than its tail shows up (the
    # wizard's old head was exactly 4 characters).
    for i in range(len(key) - 4):
        assert key[i:i + 4] not in text, f"{key[i:i + 4]!r} leaked: {text}"


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_report_error_logs_only_the_last_four(status, caplog):
    pool = CredentialPool([KEY, OTHER])
    with caplog.at_level(logging.INFO, logger="prometheus.providers.credential_pool"):
        pool.report_error(KEY, status)
    [record] = caplog.records
    _assert_tail_only(record.getMessage(), KEY)


def test_revive_logs_only_the_last_four(caplog):
    pool = CredentialPool([KEY, OTHER], dead_key_cooldown_seconds=0)
    pool.report_error(KEY, 401)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="prometheus.providers.credential_pool"):
        assert pool.active_count == 2  # revives the dead key
    [record] = [r for r in caplog.records if "Revived" in r.getMessage()]
    _assert_tail_only(record.getMessage(), KEY)


def test_a_short_key_shows_nothing_of_itself(caplog):
    short = "abc" + "defg"
    pool = CredentialPool([short, OTHER])
    with caplog.at_level(logging.INFO, logger="prometheus.providers.credential_pool"):
        pool.report_error(short, 401)
    [record] = caplog.records
    assert "defg" not in record.getMessage() and "abc" not in record.getMessage()


def test_setup_wizard_summary_shows_only_the_last_four(tmp_path, monkeypatch, capsys):
    from prometheus import setup_wizard

    monkeypatch.setattr(setup_wizard, "_config_target", lambda: tmp_path / "prometheus.yaml")
    telegram = "botid" + "abcdefghij" + "klmnopqrst" + "TGzz"
    slack = "xoxb" + "-abcdefghijklmnop" + "SLyy"
    discord = "disc" + "abcdefghijklmnopqrst" + "DSxx"
    cfg = {
        "model": {"provider": "llama_cpp", "base_url": "http://localhost:8080"},
        "gateway": {
            "telegram_enabled": True, "telegram_token": telegram,
            "slack_enabled": True, "slack_bot_token": slack,
            "discord": {"enabled": True, "token": discord},
        },
    }

    setup_wizard.SetupWizard()._save_config(cfg)

    out = capsys.readouterr().out
    for token in (telegram, slack, discord):
        _assert_tail_only(out, token)
