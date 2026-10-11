"""An EMPTY ``PROMETHEUS_API_TOKEN=`` beside a token in the env file — asserted at the AUTH PATH.

The bug this file pins: ``ensure_api_token`` resolved the env file's token (an empty variable is falsy, so
resolution moves on), reported it, and exported it with ``os.environ.setdefault``. ``setdefault`` never
replaces a key that already exists, empty or not, so the variable stayed ``""``. ``create_app`` and the
WebSocket bridge read only ``config.web.api_token or PROMETHEUS_API_TOKEN``, so the control plane — every
tool, bash included — served OPEN while the startup log said ``web auth: ENABLED (token from env-file)``.

How the empty variable gets there is ordinary: ``PROMETHEUS_API_TOKEN=`` in a unit file, a compose file or a
shell profile, with the real token in ``~/.config/prometheus/env``. ``load_env_file`` has setdefault semantics
too, so the file's value never replaces it either.

Every assertion goes through the real bearer middleware (``create_app``), modelling the daemon's own order:
``load_env_file()`` at boot, then ``ensure_api_token``, then the app.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from prometheus.config.api_token import TOKEN_ENV_VAR, describe_web_auth, ensure_api_token
from prometheus.config.env_file import load_env_file
from prometheus.web.server import create_app

ENV_FILE_TOKEN = "env-file-token-for-the-empty-variable-test"


@pytest.fixture
def boot(tmp_path, monkeypatch):
    """The daemon's token bootstrap with PROMETHEUS_API_TOKEN= exported empty and a token in the env file."""
    env_file = tmp_path / "env"
    env_file.write_text(f"{TOKEN_ENV_VAR}={ENV_FILE_TOKEN}\n", encoding="utf-8")
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(env_file))
    monkeypatch.setenv(TOKEN_ENV_VAR, "")
    load_env_file()                       # run_daemon's first step
    token, minted = ensure_api_token({})  # the web bootstrap
    yield token, minted
    os.environ.pop(TOKEN_ENV_VAR, None)


def _status(client: TestClient, token: str | None) -> int:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.get("/api/status", headers=headers).status_code


def test_the_bootstrap_reports_the_env_file_token(boot):
    # The premise: resolution finds the env file's token and the log line the daemon writes next says auth
    # is on. (Its "from …" label reads env once the token is exported, as it always has when the variable
    # was unset.)
    assert boot == (ENV_FILE_TOKEN, False)
    assert describe_web_auth({}).startswith("web auth: ENABLED")


def test_the_api_does_not_serve_open_when_the_log_says_auth_is_on(boot):
    client = TestClient(create_app({}, session_mgr=None, model_router=None))
    assert _status(client, None) == 401, "no token at all opened the API while the log said ENABLED"
    assert _status(client, "a-wrong-token") == 401
    assert _status(client, ENV_FILE_TOKEN) == 200


def test_the_variable_the_web_plane_reads_carries_the_resolved_token(boot):
    assert os.environ[TOKEN_ENV_VAR] == ENV_FILE_TOKEN
