"""``PROMETHEUS_API_TOKEN_FILE`` — the web API token read from a file (a Docker or compose secret).

The container image must take its token from the environment or from a secret, never from the image
itself. ``PROMETHEUS_API_TOKEN`` already covers the environment. A secret arrives as a FILE
(``/run/secrets/<name>``), so the daemon learns to read the token from the file a variable names, the
way ``PROMETHEUS_TELEGRAM_TOKEN_FILE`` already works for the Telegram token.

What is pinned here:

* precedence: ``web.api_token`` > ``PROMETHEUS_API_TOKEN`` > ``PROMETHEUS_API_TOKEN_FILE`` > the env file;
* a file that gives no token (missing, empty, a symlink) FAILS CLOSED. It never falls back to the env
  file's token and never mints a fresh one: an operator who named a secret meant that secret, and a
  daemon that quietly served under a different token would leave every client they configured locked
  out while printing a token they never asked for;
* ``oara daemon`` refuses to start on such a file, before setup mode or the real boot, so a container
  restarts with the reason in its log instead of running with web auth in an unknown state;
* the token is never written anywhere: no env-file line, no rotation into a file the daemon does not own;
* the variable is secret-shaped, so model-written shells never inherit it.
"""

from __future__ import annotations

import argparse
import os
import sys

import pytest

from prometheus.config import api_token as api_token_mod
from prometheus.config.api_token import (
    TOKEN_ENV_VAR,
    ApiTokenFileError,
    TokenRotationBlocked,
    ensure_api_token,
    resolve_api_token,
    rotate_api_token,
)
from prometheus.config.env_file import get_env_file_path, parse_env_file

TOKEN_FILE_VAR = "PROMETHEUS_API_TOKEN_FILE"
FILE_TOKEN = "file-token-0123456789abcdefghijklmnop"


@pytest.fixture(autouse=True)
def _no_ambient_token_file(monkeypatch):
    monkeypatch.delenv(TOKEN_FILE_VAR, raising=False)
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)


@pytest.fixture
def secret_file(tmp_path, monkeypatch):
    path = tmp_path / "run-secrets" / "prometheus_api_token"
    path.parent.mkdir()
    # A trailing newline is what `echo token > file` and most secret stores write.
    path.write_text(FILE_TOKEN + "\n", encoding="utf-8")
    monkeypatch.setenv(TOKEN_FILE_VAR, str(path))
    return path


def test_the_variable_name_is_exported():
    assert api_token_mod.TOKEN_FILE_ENV_VAR == TOKEN_FILE_VAR


def test_a_token_file_supplies_the_token(secret_file):
    assert resolve_api_token(None) == (FILE_TOKEN, "file")


def test_the_environment_wins_over_the_file(secret_file, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV_VAR, "env-token")
    assert resolve_api_token(None) == ("env-token", "env")


def test_the_config_wins_over_the_file(secret_file):
    assert resolve_api_token({"web": {"api_token": "yaml-token"}}) == ("yaml-token", "config")


def test_the_file_wins_over_the_env_file(secret_file):
    get_env_file_path().write_text(f"{TOKEN_ENV_VAR}=env-file-token\n", encoding="utf-8")
    assert resolve_api_token(None) == (FILE_TOKEN, "file")


def test_an_explicit_empty_environment_value_does_not_hide_a_named_file(secret_file, monkeypatch):
    # PROMETHEUS_API_TOKEN= is the deliberate-open channel. Naming a secret file as well is a contradiction,
    # and the safe reading of a contradiction is the one with auth on.
    monkeypatch.setenv(TOKEN_ENV_VAR, "")
    token, minted = ensure_api_token(None)
    assert (token, minted) == (FILE_TOKEN, False)
    # The web launcher and the WS bridge read ONLY the variable. Left empty, they would serve open while
    # ensure_api_token and the startup log both reported the file's token in force.
    assert os.environ[TOKEN_ENV_VAR] == FILE_TOKEN


def test_an_explicit_empty_environment_value_does_not_hide_an_env_file_token(monkeypatch):
    # The same hole without a secret file: ensure_api_token reported the env file's token while the variable the
    # web plane reads stayed empty (setdefault never overwrites an existing key, empty or not).
    monkeypatch.setenv(TOKEN_ENV_VAR, "")
    get_env_file_path().write_text(f"{TOKEN_ENV_VAR}=env-file-token\n", encoding="utf-8")
    assert ensure_api_token(None) == ("env-file-token", False)
    assert os.environ[TOKEN_ENV_VAR] == "env-file-token"


@pytest.mark.parametrize("problem", ["missing", "empty", "symlink"])
def test_a_file_that_gives_no_token_fails_closed(problem, tmp_path, monkeypatch):
    target = tmp_path / "token"
    if problem == "empty":
        target.write_text("\n", encoding="utf-8")
    elif problem == "symlink":
        real = tmp_path / "real-token"
        real.write_text(FILE_TOKEN, encoding="utf-8")
        target.symlink_to(real)
    monkeypatch.setenv(TOKEN_FILE_VAR, str(target))
    # Even with a perfectly good token sitting in the env file: the operator named THIS file.
    get_env_file_path().write_text(f"{TOKEN_ENV_VAR}=env-file-token\n", encoding="utf-8")
    with pytest.raises(ApiTokenFileError) as excinfo:
        resolve_api_token(None)
    assert TOKEN_FILE_VAR in str(excinfo.value)
    assert str(target) in str(excinfo.value)


def test_ensure_never_mints_over_a_broken_file(tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_FILE_VAR, str(tmp_path / "absent"))
    with pytest.raises(ApiTokenFileError):
        ensure_api_token(None)
    assert TOKEN_ENV_VAR not in parse_env_file()
    assert TOKEN_ENV_VAR not in os.environ


def test_the_daemon_refuses_the_web_plane_when_the_file_is_broken(tmp_path, monkeypatch):
    # daemon.run_daemon catches the bootstrap raise and asks this gate whether to serve.
    monkeypatch.setenv(TOKEN_FILE_VAR, str(tmp_path / "absent"))
    raised = False
    try:
        ensure_api_token(None)
    except Exception:
        raised = True
    assert raised
    assert api_token_mod.web_refused_on_bootstrap_failure(None, bootstrap_raised=True)


def test_ensure_uses_the_file_token_and_writes_nothing(secret_file, monkeypatch):
    token, minted = ensure_api_token(None)
    assert (token, minted) == (FILE_TOKEN, False)
    # The web launcher and the WS bridge read the token from the process environment.
    assert os.environ[TOKEN_ENV_VAR] == FILE_TOKEN
    assert not get_env_file_path().exists(), "a token from a secret must not be copied into the env file"


def test_setup_mode_pairing_hands_out_the_file_token(secret_file):
    issued = api_token_mod.issue_setup_credential(None)
    assert issued.token == FILE_TOKEN


def test_rotation_refuses_a_token_that_lives_in_a_file(secret_file):
    with pytest.raises(TokenRotationBlocked) as excinfo:
        rotate_api_token(None)
    message = str(excinfo.value)
    assert TOKEN_FILE_VAR in message
    assert str(secret_file) in message
    assert "web.api_token" not in message, "the file case must not be described as a YAML pin"
    assert TOKEN_ENV_VAR not in parse_env_file()


def test_describe_web_auth_names_the_file_source(secret_file):
    line = api_token_mod.describe_web_auth(None)
    assert "ENABLED" in line and "file" in line
    assert FILE_TOKEN not in line


def test_the_variable_is_scrubbed_from_model_shells():
    from prometheus.security.env_scrub import is_secret_name

    assert is_secret_name(TOKEN_FILE_VAR)


def test_the_suite_strips_an_ambient_token_file_variable():
    # tests/conftest.py::_hermetic_prometheus_env must drop it the way it drops PROMETHEUS_API_TOKEN:
    # a developer's own secret file would otherwise turn bearer auth on under every API test.
    import inspect

    from tests import conftest

    assert TOKEN_FILE_VAR in inspect.getsource(conftest._hermetic_prometheus_env)


# ── the CLI surfaces ─────────────────────────────────────────────────────────


def test_token_show_names_the_file(secret_file, capsys):
    from prometheus.cli.token import run_token_command

    rc = run_token_command(argparse.Namespace(token_action="show"), None)
    out = capsys.readouterr().out
    assert rc == 0
    assert out.splitlines()[0] == FILE_TOKEN
    assert TOKEN_FILE_VAR in out and str(secret_file) in out


def test_token_show_reports_a_broken_file_without_a_traceback(tmp_path, monkeypatch, capsys):
    from prometheus.cli.token import run_token_command

    monkeypatch.setenv(TOKEN_FILE_VAR, str(tmp_path / "absent"))
    rc = run_token_command(argparse.Namespace(token_action="show"), None)
    out = capsys.readouterr().out
    assert rc == 1
    assert TOKEN_FILE_VAR in out


def test_doctor_reports_a_broken_file_as_an_error(tmp_path, monkeypatch):
    from prometheus.cli.doctor import check_token

    monkeypatch.setenv(TOKEN_FILE_VAR, str(tmp_path / "absent"))
    check = check_token({})
    assert check.status == "error"
    assert TOKEN_FILE_VAR in check.message


def test_doctor_names_the_file_source(secret_file):
    from prometheus.cli.doctor import check_token

    check = check_token({})
    assert check.status == "ok"
    assert "file" in check.message


def test_oara_daemon_refuses_to_start_on_a_broken_file(tmp_path, monkeypatch, capsys):
    """Before setup mode and before the real boot: a container restarts with the reason in its log."""
    from prometheus import daemon
    from prometheus.web import setup_server

    def _must_not_run(*_a, **_k):
        raise AssertionError("the daemon went on to serve with a broken token file")

    monkeypatch.setattr(setup_server, "run_setup_mode", _must_not_run)
    monkeypatch.setattr(setup_server, "find_config_file", lambda *_a, **_k: None)
    monkeypatch.setattr(daemon, "run_daemon", _must_not_run)
    monkeypatch.setenv(TOKEN_FILE_VAR, str(tmp_path / "absent"))
    monkeypatch.setattr(sys, "argv", ["prometheus-daemon"])
    with pytest.raises(SystemExit) as excinfo:
        daemon.main()
    assert excinfo.value.code not in (0, None)
    assert TOKEN_FILE_VAR in capsys.readouterr().err
