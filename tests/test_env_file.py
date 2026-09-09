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
        assert "oara token show" in banner


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


class TestTemplateBootsAuthenticated:
    """The audit's CRITICAL finding: a verbatim template copy booted an
    UNAUTHENTICATED control plane.

    The shipped template carries ``web.api_token:`` which YAML parses as a
    present-with-NULL value. The old ``_deliberately_open`` read present-but-
    falsy as "the operator chose an open API" and skipped minting — so anyone
    who copied the template exactly as instructed got REST on :8005, WS on
    :8010 and the OpenAI-compatible surface with no token, on 0.0.0.0, with
    bash reachable.

    These tests drive the REAL shipped template (not a hand-written dict) so a
    future template edit that reintroduces the hole fails here, and pin the
    distinction: NULL mints, an explicit empty STRING stays open.
    """

    def test_verbatim_template_mints_a_token(self, env_file):
        """The whole finding: the template as shipped must NOT boot open."""
        from prometheus.config.template import load_template

        tmpl = load_template()
        assert tmpl.get("web", {}).get("api_token", "MISSING") is None, (
            "precondition: the shipped template carries api_token as NULL; if "
            "that changed, re-derive this test from the new shape"
        )
        token, minted = ensure_api_token(tmpl)
        assert minted is True, (
            "a verbatim template copy booted an unauthenticated control plane"
        )
        assert len(token) >= 32
        assert parse_env_file()[TOKEN_ENV_VAR] == token
        os.environ.pop(TOKEN_ENV_VAR, None)

    def test_null_api_token_is_not_deliberate_open(self):
        """NULL == 'not configured' (mint). It is not a choice to be open."""
        from prometheus.config.api_token import _deliberately_open

        assert _deliberately_open({"web": {"api_token": None}}) is False
        assert _deliberately_open({"web": {}}) is False
        assert _deliberately_open({}) is False

    def test_explicit_empty_string_config_stays_open(self):
        """The deliberate-open channel still works via an explicit empty
        string in config — symmetric with the env-file/env-var channels."""
        from prometheus.config.api_token import _deliberately_open

        assert _deliberately_open({"web": {"api_token": ""}}) is True

    def test_explicit_empty_string_config_mints_nothing(self, env_file):
        """End-to-end: ``api_token: ""`` is honored as auth-OFF (no mint)."""
        token, minted = ensure_api_token({"web": {"api_token": ""}})
        assert (token, minted) == ("", False)
        assert TOKEN_ENV_VAR not in parse_env_file()

    def test_a_real_token_in_config_is_never_overridden(self, env_file):
        token, minted = ensure_api_token({"web": {"api_token": "pinned-secret"}})
        assert (token, minted) == ("pinned-secret", False)
        os.environ.pop(TOKEN_ENV_VAR, None)


class TestRefuseToServeOnBootstrapFailure:
    """The audit's second open door: when ensure_api_token RAISES (unwritable
    env-file dir → a minted token cannot persist), the daemon logged one ERROR
    and launched the web bridge OPEN anyway — unauthenticated, bash reachable.

    The decision is now a pure helper so it can be tested without a daemon
    boot. Refusing to serve beats serving open. The key distinction: a RAISE
    (could not establish auth at all) refuses; deliberate-open (an explicit
    empty token, no raise) still serves.
    """

    def test_bootstrap_raise_with_no_token_configured_refuses(self):
        from prometheus.config.api_token import web_refused_on_bootstrap_failure

        assert web_refused_on_bootstrap_failure(
            {"web": {"enabled": True}},
            bootstrap_raised=True,
            environ={},
        ) is True, "a bootstrap raise with nothing configured must refuse to serve"

    def test_no_raise_never_refuses(self):
        from prometheus.config.api_token import web_refused_on_bootstrap_failure

        # Normal mint path (no raise) must never trip the refusal.
        assert web_refused_on_bootstrap_failure(
            {"web": {"enabled": True}}, bootstrap_raised=False, environ={}
        ) is False

    def test_raise_but_token_in_config_still_serves(self):
        from prometheus.config.api_token import web_refused_on_bootstrap_failure

        # The env-file write failed, but a token is pinned in config — auth is
        # established, so serving is correct (not open).
        assert web_refused_on_bootstrap_failure(
            {"web": {"api_token": "pinned"}}, bootstrap_raised=True, environ={}
        ) is False

    def test_raise_but_token_in_environ_still_serves(self):
        from prometheus.config.api_token import web_refused_on_bootstrap_failure
        from prometheus.config.api_token import TOKEN_ENV_VAR

        assert web_refused_on_bootstrap_failure(
            {"web": {}}, bootstrap_raised=True, environ={TOKEN_ENV_VAR: "envtok"}
        ) is False

    def test_deliberate_open_is_not_a_raise_so_serves(self):
        """The distinction the fix must preserve: an explicit empty token
        resolves to ("", False) with NO exception, so it is not a refusal —
        the operator chose open and open is what they get."""
        from prometheus.config.api_token import (
            _deliberately_open,
            web_refused_on_bootstrap_failure,
        )

        cfg = {"web": {"api_token": ""}}
        # deliberate-open: no raise, so the refusal helper stays False
        assert _deliberately_open(cfg) is True
        assert web_refused_on_bootstrap_failure(
            cfg, bootstrap_raised=False, environ={}
        ) is False
