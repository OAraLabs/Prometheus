"""OpenRouter as a cloud provider the setup path can choose and the daemon can then RUN.

Beacon's O9 makes "Sign in with OpenRouter" the Recommended way to connect a cloud AI: one account, no key to copy.
Beacon runs the sign-in and hands the resulting key to ``POST /api/setup/configure`` as
``{"provider": "openrouter", "api_key": "<key>"}`` — no ``base_url``, ``model`` optional.

Pinned here, because a config that is written but cannot boot is the dead end every setup path is held against:

* the provider is registered end to end — setup's table, the registry that builds it at boot, the env-var surfaces;
* the default model is a real, PAID, tool-capable OpenRouter model, never a free one (free models are capped at 50
  requests a day and their hosts often train on what they are sent). The fixture is the live catalog entry;
* what setup WRITES, the daemon's own registry then READS (the consumed layer, not just the changed one);
* the key lands in the env file, never the yaml, the response or the logs.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from prometheus.cli.init import _CLOUD_FAST_PROVIDERS, _cloud_default_config
from prometheus.config.env_file import parse_env_file
from prometheus.providers.registry import CLOUD_DEFAULTS, ProviderRegistry

pytest.importorskip("fastapi")

from tests.test_setup_api_phase2 import (  # noqa: E402
    config_dir,  # noqa: F401 — fixtures re-exported
    env_file,  # noqa: F401
    make_client,
    pair,
)

KEY = "sk-or-v1-" + "d4c3b2a1" * 8
BASE = "https://openrouter.ai/api/v1"
CATALOG = json.loads(
    (Path(__file__).parent / "fixtures" / "provider_key_checks" / "openrouter_default_model.json")
    .read_text(encoding="utf-8")
)["model"]


@pytest.fixture(autouse=True)
def no_cloud_keys_in_environment(monkeypatch):
    for key_env, _model, _limit in _CLOUD_FAST_PROVIDERS.values():
        monkeypatch.delenv(key_env, raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)


# --------------------------------------------------------------------------- #
# The default model
# --------------------------------------------------------------------------- #


class TestTheDefaultModel:
    def test_it_is_the_real_paid_tool_capable_model_in_the_recorded_catalog(self):
        model = CLOUD_DEFAULTS["openrouter"]["model"]
        assert model == CATALOG["id"], "the default must be a model OpenRouter actually lists"
        assert float(CATALOG["pricing"]["prompt"]) > 0 and float(CATALOG["pricing"]["completion"]) > 0, (
            "mid-tier PAID, never free"
        )
        assert "tools" in CATALOG["supported_parameters"], "the agent loop needs tool calls"

    def test_it_is_never_a_free_variant(self):
        for model in (CLOUD_DEFAULTS["openrouter"]["model"], _CLOUD_FAST_PROVIDERS["openrouter"][1]):
            assert not model.endswith(":free") and "free" not in model.split("/")[-1].split(":")

    def test_setup_and_the_registry_name_the_same_default(self):
        assert _CLOUD_FAST_PROVIDERS["openrouter"][1] == CLOUD_DEFAULTS["openrouter"]["model"]

    def test_the_context_budget_is_the_one_the_same_model_gets_on_the_direct_route(self):
        assert _CLOUD_FAST_PROVIDERS["openrouter"][2] == _CLOUD_FAST_PROVIDERS["anthropic"][2]


# --------------------------------------------------------------------------- #
# Registered end to end
# --------------------------------------------------------------------------- #


class TestRegistered:
    def test_the_registry_lists_it_and_calls_it_a_cloud_service(self):
        assert "openrouter" in ProviderRegistry.list_providers()
        assert ProviderRegistry.is_cloud("openrouter") is True

    def test_the_registry_builds_a_provider_that_points_at_openrouter_with_the_key_from_the_env(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": KEY}):
            provider = ProviderRegistry.create({"provider": "openrouter", "api_key_env": "OPENROUTER_API_KEY"})
        assert provider._base_url == BASE
        assert provider._model == CLOUD_DEFAULTS["openrouter"]["model"]
        assert provider.provider_name == "openrouter"

    def test_a_base_url_env_var_overrides_it_like_every_other_provider(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": KEY, "OPENROUTER_BASE_URL": "http://localhost:9999/v1"}):
            provider = ProviderRegistry.create({"provider": "openrouter", "api_key_env": "OPENROUTER_API_KEY"})
        assert provider._base_url == "http://localhost:9999/v1"

    def test_its_key_is_a_secret_the_bash_tool_never_sees(self):
        from prometheus.security.env_scrub import is_secret_name

        assert is_secret_name("OPENROUTER_API_KEY") is True

    def test_the_env_override_surface_knows_the_key(self):
        from prometheus.config.env_override import ENV_OVERRIDES

        assert ENV_OVERRIDES["OPENROUTER_API_KEY"] == ("providers", "openrouter", "api_key")

    def test_doctor_reports_whether_the_key_is_set_and_never_its_value(self):
        from prometheus.infra.doctor import Doctor

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": KEY}, clear=True):
            message = Doctor({})._check_cloud_keys().message
        assert "OpenRouter set" in message and KEY not in message and KEY[:8] not in message
        with patch.dict(os.environ, {}, clear=True):
            assert "OpenRouter not set" in Doctor({})._check_cloud_keys().message

    def test_the_cli_fast_path_writes_a_config_the_registry_accepts(self):
        env_name, model, _limit = _CLOUD_FAST_PROVIDERS["openrouter"]
        section = _cloud_default_config("openrouter", env_name, model)["model"]
        assert section["provider"] == "openrouter" and section["api_key_env"] == "OPENROUTER_API_KEY"
        assert "base_url" not in section
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": KEY}):
            assert ProviderRegistry.create(section)._base_url == BASE


# --------------------------------------------------------------------------- #
# POST /api/setup/configure and /api/setup/detect
# --------------------------------------------------------------------------- #


class TestSetupConfigure:
    def test_configure_writes_openrouter_with_the_default_model_and_the_key_in_the_env_file(
            self, env_file, config_dir):  # noqa: F811
        client, state = make_client()
        resp = client.post("/api/setup/configure", json={"provider": "openrouter", "api_key": KEY}, headers=pair(client))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["configured"] is True and body["provider"] == "openrouter"
        assert body["model"] == CLOUD_DEFAULTS["openrouter"]["model"]
        assert body["api_key_env"] == "OPENROUTER_API_KEY" and body["api_key_saved"] is True
        assert body["base_url"] is None
        assert KEY not in resp.text
        assert state.configured is True
        assert parse_env_file(env_file)["OPENROUTER_API_KEY"] == KEY
        assert KEY not in (config_dir / "prometheus.yaml").read_text(encoding="utf-8")

    def test_what_setup_writes_the_daemons_registry_can_boot(self, env_file, config_dir):  # noqa: F811
        client, _ = make_client()
        client.post("/api/setup/configure", json={"provider": "openrouter", "api_key": KEY}, headers=pair(client))
        written = yaml.safe_load((config_dir / "prometheus.yaml").read_text(encoding="utf-8"))["model"]
        assert written["provider"] == "openrouter"
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": parse_env_file(env_file)["OPENROUTER_API_KEY"]}):
            provider = ProviderRegistry.create(written)
        assert provider._base_url == BASE and provider._model == CLOUD_DEFAULTS["openrouter"]["model"]

    def test_a_model_may_be_chosen_and_a_base_url_may_not(self, env_file, config_dir):  # noqa: F811
        client, _ = make_client()
        auth = pair(client)
        chosen = client.post("/api/setup/configure", headers=auth,
                             json={"provider": "openrouter", "api_key": KEY, "model": "openai/gpt-5.6"})
        assert chosen.status_code == 200 and chosen.json()["model"] == "openai/gpt-5.6"
        refused = client.post("/api/setup/configure", headers=auth,
                              json={"provider": "openrouter", "api_key": KEY, "base_url": "https://evil.example/v1"})
        assert refused.status_code == 400 and refused.json()["error"] == "unexpected_base_url"

    def test_without_a_key_anywhere_it_is_refused_like_every_cloud_provider(self, env_file, config_dir):  # noqa: F811
        client, state = make_client()
        resp = client.post("/api/setup/configure", json={"provider": "openrouter"}, headers=pair(client))
        assert resp.status_code == 400 and resp.json()["error"] == "cloud_key_missing"
        assert state.configured is False

    def test_the_key_is_not_logged(self, env_file, config_dir, caplog):  # noqa: F811
        caplog.set_level(logging.DEBUG)
        client, _ = make_client()
        client.post("/api/setup/configure", json={"provider": "openrouter", "api_key": KEY}, headers=pair(client))
        assert KEY not in "\n".join(r.getMessage() for r in caplog.records)

    def test_detect_offers_it_as_a_cloud_choice(self, env_file, config_dir):  # noqa: F811
        client, _ = make_client()
        cloud = client.get("/api/setup/detect", headers=pair(client)).json()["cloud_providers"]
        entry = next(c for c in cloud if c["provider"] == "openrouter")
        assert entry["api_key_env"] == "OPENROUTER_API_KEY"
        assert entry["default_model"] == CLOUD_DEFAULTS["openrouter"]["model"]
        assert entry["key_present"] is False
