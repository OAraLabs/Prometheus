"""Tests for the env-file helpers and the API-token bootstrap (Phase 0).

Hard rule honored throughout: everything targets a tmp env file via the
``PROMETHEUS_ENV_FILE`` override — the real ``~/.config/prometheus/env``
is never touched.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from prometheus.config import api_token as api_token_mod
from prometheus.config.api_token import (
    TOKEN_ENV_VAR,
    describe_web_auth,
    ensure_api_token,
    format_minted_banner,
    mint_api_token,
    resolve_api_token,
    rotate_api_token,
)
from prometheus.config.env_file import (
    get_env_file_path,
    load_env_file,
    parse_env_file,
    set_env_value,
)


@pytest.fixture(autouse=True)
def _isolate_token_env():
    """Save/restore PROMETHEUS_API_TOKEN around every test.

    ensure/rotate export the token into ``os.environ`` by design;
    ``monkeypatch.delenv(raising=False)`` on an ABSENT var records
    nothing, so it would NOT undo those exports — this fixture does.
    """
    saved = os.environ.pop(TOKEN_ENV_VAR, None)
    yield
    os.environ.pop(TOKEN_ENV_VAR, None)
    if saved is not None:
        os.environ[TOKEN_ENV_VAR] = saved


@pytest.fixture
def env_file(tmp_path, monkeypatch) -> Path:
    """Point the env-file machinery at a tmp file."""
    path = tmp_path / "env"
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(path))
    return path


# ---------------------------------------------------------------------------
# env_file primitives
# ---------------------------------------------------------------------------


class TestEnvFilePath:
    def test_override_wins(self, env_file):
        assert get_env_file_path() == env_file

    def test_default_is_config_prometheus_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PROMETHEUS_ENV_FILE", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        assert get_env_file_path() == tmp_path / "xdg" / "prometheus" / "env"


class TestParseEnvFile:
    def test_missing_file_is_empty(self, env_file):
        assert parse_env_file() == {}

    def test_parses_assignments_skips_comments(self, env_file):
        env_file.write_text(
            "# comment\n"
            "\n"
            "FOO=bar\n"
            "export QUOTED=\"with spaces\"\n"
            "#   PROMETHEUS_API_TOKEN=\n"
            "EMPTY=\n",
            encoding="utf-8",
        )
        values = parse_env_file()
        assert values == {"FOO": "bar", "QUOTED": "with spaces", "EMPTY": ""}
        # The commented token line must NOT count as an explicit empty value.
        assert TOKEN_ENV_VAR not in values


class TestSetEnvValue:
    def test_creates_file_with_0600(self, env_file):
        set_env_value("FOO", "bar")
        assert parse_env_file() == {"FOO": "bar"}
        assert (env_file.stat().st_mode & 0o777) == 0o600

    def test_updates_in_place_preserving_comments(self, env_file):
        env_file.write_text("# keep me\nFOO=old\nBAR=1\n", encoding="utf-8")
        set_env_value("FOO", "new")
        body = env_file.read_text(encoding="utf-8")
        assert "# keep me" in body
        assert body.index("FOO=new") < body.index("BAR=1")
        assert parse_env_file() == {"FOO": "new", "BAR": "1"}

    def test_appends_when_absent(self, env_file):
        env_file.write_text("FOO=1\n", encoding="utf-8")
        set_env_value("NEW", "x")
        assert parse_env_file() == {"FOO": "1", "NEW": "x"}


class TestLoadEnvFile:
    def test_setdefault_semantics(self, env_file, monkeypatch):
        env_file.write_text("A_TEST_VAR=fromfile\nB_TEST_VAR=fromfile\n")
        monkeypatch.setenv("A_TEST_VAR", "fromenv")
        monkeypatch.delenv("B_TEST_VAR", raising=False)
        loaded = load_env_file()
        assert loaded == 1
        assert os.environ["A_TEST_VAR"] == "fromenv"   # real env wins
        assert os.environ["B_TEST_VAR"] == "fromfile"
        monkeypatch.delenv("B_TEST_VAR", raising=False)


# ---------------------------------------------------------------------------
# API token bootstrap
# ---------------------------------------------------------------------------


class TestResolveApiToken:
    def test_config_wins(self, env_file, monkeypatch):
        monkeypatch.setenv(TOKEN_ENV_VAR, "envtoken")
        token, source = resolve_api_token({"web": {"api_token": "cfgtoken"}})
        assert (token, source) == ("cfgtoken", "config")

    def test_env_then_file(self, env_file, monkeypatch):
        set_env_value(TOKEN_ENV_VAR, "filetoken")
        assert resolve_api_token({}) == ("filetoken", "env-file")
        monkeypatch.setenv(TOKEN_ENV_VAR, "envtoken")
        assert resolve_api_token({}) == ("envtoken", "env")

    def test_nothing_set(self, env_file):
        assert resolve_api_token({}) == ("", "")


class TestEnsureApiToken:
    def test_mints_and_persists_when_unset(self, env_file):
        token, minted = ensure_api_token({"web": {"enabled": True}})
        assert minted is True
        assert len(token) >= 32
        # Persisted to the env file the daemon/systemd unit read.
        assert parse_env_file()[TOKEN_ENV_VAR] == token
        # Exported so the web launcher picks it up in-process.
        assert os.environ[TOKEN_ENV_VAR] == token
        os.environ.pop(TOKEN_ENV_VAR, None)

    def test_idempotent_second_start(self, env_file):
        token1, minted1 = ensure_api_token({})
        os.environ.pop(TOKEN_ENV_VAR, None)  # simulate a fresh process
        token2, minted2 = ensure_api_token({})
        assert minted1 is True and minted2 is False
        assert token1 == token2

    def test_explicit_blank_in_file_stays_open(self, env_file):
        env_file.write_text(f"{TOKEN_ENV_VAR}=\n", encoding="utf-8")
        token, minted = ensure_api_token({})
        assert (token, minted) == ("", False)
        # Nothing was minted into the file either.
        assert parse_env_file()[TOKEN_ENV_VAR] == ""

    def test_explicit_blank_in_environ_stays_open(self, env_file, monkeypatch):
        monkeypatch.setenv(TOKEN_ENV_VAR, "")
        token, minted = ensure_api_token({})
        assert (token, minted) == ("", False)

    def test_existing_token_not_replaced(self, env_file):
        set_env_value(TOKEN_ENV_VAR, "keepme")
        token, minted = ensure_api_token({})
        assert (token, minted) == ("keepme", False)
        os.environ.pop(TOKEN_ENV_VAR, None)


class TestRotateAndDescribe:
    def test_rotate_replaces_persisted_token(self, env_file):
        set_env_value(TOKEN_ENV_VAR, "old")
        new = rotate_api_token()
        assert new != "old"
        assert parse_env_file()[TOKEN_ENV_VAR] == new
        os.environ.pop(TOKEN_ENV_VAR, None)

    def test_describe_states(self, env_file, monkeypatch):
        assert "OPEN" in describe_web_auth({})
        monkeypatch.setenv(TOKEN_ENV_VAR, "tok")
        assert "ENABLED" in describe_web_auth({})

    def test_minted_banner_contains_token_once(self, env_file):
        token = mint_api_token()
        banner = format_minted_banner(token)
        assert banner.count(token) == 1
        assert "prometheus token show" in banner


# ---------------------------------------------------------------------------
# token CLI command
# ---------------------------------------------------------------------------


class TestTokenCli:
    def _args(self, action: str):
        import argparse
        return argparse.Namespace(token_action=action)

    def test_show_when_unset_exits_nonzero(self, env_file, capsys):
        from prometheus.cli.token import run_token_command
        rc = run_token_command(self._args("show"), {})
        assert rc == 1
        assert "OPEN" in capsys.readouterr().out

    def test_rotate_then_show(self, env_file, capsys):
        from prometheus.cli.token import run_token_command
        assert run_token_command(self._args("rotate"), {}) == 0
        rotated = parse_env_file()[TOKEN_ENV_VAR]
        os.environ.pop(TOKEN_ENV_VAR, None)
        capsys.readouterr()
        assert run_token_command(self._args("show"), {}) == 0
        out = capsys.readouterr().out
        assert rotated in out
