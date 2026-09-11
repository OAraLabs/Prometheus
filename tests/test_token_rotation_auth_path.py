"""``oara token rotate`` — asserted at the AUTH PATH, not at its own report (#320).

The bug this file pins: ``rotate_api_token`` wrote the new token to the env
file and nowhere else, while ``create_app`` resolves
``config.web.api_token or PROMETHEUS_API_TOKEN`` — config FIRST. With a token
pinned in ``prometheus.yaml`` the command printed "New web API token generated
and saved", handed back a token the daemon would never accept, and left the
old (in the incident that opened #320, leaked) token valid across restarts.

Every assertion here goes through the REAL bearer middleware — ``create_app``
+ ``verify_token`` — because the report and the outcome disagreed, and only
the outcome matters. A test that asserted the exit code or the printed string
passed against the broken version; so did the pre-existing
``test_rotate_replaces_persisted_token``, which checked the env file the
daemon was not reading.

"Restarting the daemon" is modelled the way systemd does it: the process
environment is dropped and rebuilt from the env file
(``EnvironmentFile=-%h/.config/prometheus/env``, packaging/prometheus.service),
then the app is constructed from the unchanged YAML.
"""

from __future__ import annotations

import argparse
import os

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from prometheus.cli.token import run_token_command
from prometheus.config.api_token import (
    TOKEN_ENV_VAR,
    TokenRotationBlocked,
    rotate_api_token,
)
from prometheus.config.env_file import load_env_file, parse_env_file, set_env_value
from prometheus.web.server import create_app


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """Point the env-file helpers at a scratch file (never the real one)."""
    path = tmp_path / "env"
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(path))
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    return path


def restart_daemon(config: dict) -> TestClient:
    """Rebuild the app as a systemd restart would: env from the env file only."""
    os.environ.pop(TOKEN_ENV_VAR, None)
    load_env_file()
    return TestClient(create_app(config, session_mgr=None, model_router=None))


def status_code_for(client: TestClient, token: str) -> int:
    return client.get(
        "/api/status", headers={"Authorization": f"Bearer {token}"}
    ).status_code


class TestYamlResidentTokenCannotSilentlyNoOp:
    """The #320 case: the effective token is pinned in prometheus.yaml."""

    def test_rotate_refuses_and_says_where_the_token_lives(self, env_file, capsys):
        config = {"web": {"api_token": "LEAKED-YAML-TOKEN"}}

        # Baseline: the YAML token is what the real middleware accepts.
        client = restart_daemon(config)
        assert status_code_for(client, "LEAKED-YAML-TOKEN") == 200
        assert status_code_for(client, "some-other-token") == 401

        args = argparse.Namespace(token_action="rotate")
        exit_code = run_token_command(args, config)

        # BEFORE THE FIX this was 0 with a "generated and saved" banner.
        assert exit_code != 0, "rotation must not report success it did not achieve"

        # Nothing was written: a refusal leaves no half-rotated state behind.
        assert TOKEN_ENV_VAR not in parse_env_file()

        # The refusal names the location and the key, so the operator can act.
        out = capsys.readouterr().out
        assert "web.api_token" in out
        assert "prometheus.yaml" in out

        # THE POINT: the old token's fate is unchanged and honestly reported —
        # it still works, and the command said so instead of pretending.
        client = restart_daemon(config)
        assert status_code_for(client, "LEAKED-YAML-TOKEN") == 200

    def test_library_call_raises_rather_than_no_op(self, env_file):
        config = {"web": {"api_token": "LEAKED-YAML-TOKEN"}}
        with pytest.raises(TokenRotationBlocked) as excinfo:
            rotate_api_token(config)
        assert "web.api_token" in str(excinfo.value)
        assert TOKEN_ENV_VAR not in parse_env_file()

    def test_the_old_silent_success_shape_is_gone(self, env_file):
        """Regression pin for the exact defect: a rotate that returns a token
        the daemon will not accept. Whatever rotate hands back must be live."""
        config = {"web": {"api_token": "LEAKED-YAML-TOKEN"}}
        try:
            returned = rotate_api_token(config)
        except TokenRotationBlocked:
            return  # refused loudly — nothing was promised, nothing to check
        client = restart_daemon(config)
        assert status_code_for(client, returned) == 200, (
            "rotate returned a token the daemon refuses — the #320 silent no-op"
        )


class TestEnvFileResidentTokenActuallyRotates:
    """The rotatable case: no YAML pin, so the env file governs."""

    def test_old_token_refused_and_new_token_accepted(self, env_file):
        config: dict = {"web": {}}
        set_env_value(TOKEN_ENV_VAR, "OLD-ENV-TOKEN")

        client = restart_daemon(config)
        assert status_code_for(client, "OLD-ENV-TOKEN") == 200

        args = argparse.Namespace(token_action="rotate")
        assert run_token_command(args, config) == 0

        new_token = parse_env_file()[TOKEN_ENV_VAR]
        assert new_token != "OLD-ENV-TOKEN"

        # Both halves asserted at the middleware: refusal is the whole point.
        client = restart_daemon(config)
        assert status_code_for(client, "OLD-ENV-TOKEN") == 401
        assert status_code_for(client, new_token) == 200


class TestShowReportsTheEffectiveLocation:
    """``token show`` had the same blind spot in its FOOTER: it printed the
    env-file path unconditionally, so a YAML-resident token was reported as
    living in a file that did not contain it."""

    def test_show_prints_yaml_token_and_names_yaml_not_the_env_file(
        self, env_file, capsys
    ):
        config = {"web": {"api_token": "YAML-TOKEN"}}
        set_env_value(TOKEN_ENV_VAR, "DECOY-ENV-TOKEN")

        args = argparse.Namespace(token_action="show")
        assert run_token_command(args, config) == 0

        out = capsys.readouterr().out
        # The token the DAEMON uses, not the one the env file happens to hold.
        assert "YAML-TOKEN" in out
        assert "DECOY-ENV-TOKEN" not in out
        assert "web.api_token" in out
        # It must not point at the env file for a token that is not there.
        assert str(env_file) not in out

    def test_show_still_names_the_env_file_when_that_is_the_source(
        self, env_file, capsys
    ):
        config: dict = {"web": {}}
        set_env_value(TOKEN_ENV_VAR, "ENV-TOKEN")
        os.environ.pop(TOKEN_ENV_VAR, None)

        args = argparse.Namespace(token_action="show")
        assert run_token_command(args, config) == 0

        out = capsys.readouterr().out
        assert "ENV-TOKEN" in out
        assert str(env_file) in out
