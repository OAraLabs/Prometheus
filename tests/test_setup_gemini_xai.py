"""Gemini and xAI can be chosen in setup, the same way every other cloud provider can (Beacon v0.4, P8 item 7).

The daemon could always RUN them: ``providers/registry.py`` builds both, ``model_router`` has their presets, the cost
table prices their models, the wizard offers them. What it could not do was be SET UP with them over the API or
without a prompt, because the table those paths read (``cli/init._CLOUD_FAST_PROVIDERS``) did not list them. So
``POST /api/setup/configure {"provider": "gemini", "api_key": ...}`` was an ``unsupported_provider``, and a person who
wanted Gemini from Beacon's setup had no way to say so. The same table feeds ``oara setup --provider``, the cloud
presets ``GET /api/setup/detect`` lists, and (since #737) the providers ``PUT /api/model`` can switch to, so one
entry each makes all four work.

Its sibling was an env-file template that told people to set ``GOOGLE_API_KEY`` for Gemini. The daemon reads
``GEMINI_API_KEY`` everywhere else (registry, presets, wizard, docs), so a key set as the template said was never
found. The template and its guard are pinned here.

Expected values are literals, not read from the table they are meant to hold.
"""

from __future__ import annotations

import re

import pytest
import yaml

pytest.importorskip("fastapi")

from prometheus.cli import init as init_mod  # noqa: E402
from prometheus.cli.init import _ENV_TEMPLATE, run_init  # noqa: E402
from prometheus.config.env_file import parse_env_file  # noqa: E402
from prometheus.providers.registry import ProviderRegistry  # noqa: E402
from tests.test_setup_api_phase2 import config_dir, env_file, make_client, pair  # noqa: E402,F401

#: provider → (key variable, default model, the host its requests go to)
EXPECTED = {
    "gemini": ("GEMINI_API_KEY", "gemini-2.5-flash", "generativelanguage.googleapis.com"),
    "xai": ("XAI_API_KEY", "grok-4.5", "api.x.ai"),
}
KEY = "dummy-key-for-the-setup-tests-0123456789"


@pytest.fixture(autouse=True)
def no_cloud_keys(monkeypatch):
    for key_env, _model, _limit in init_mod._CLOUD_FAST_PROVIDERS.values():
        monkeypatch.delenv(key_env, raising=False)
    for env, _model, _host in EXPECTED.values():
        monkeypatch.delenv(env, raising=False)


def written(config_dir):
    return yaml.safe_load((config_dir / "prometheus.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("provider", sorted(EXPECTED))
class TestConfigure:
    def test_a_key_and_a_provider_are_enough(self, provider, env_file, config_dir):
        env, model, _host = EXPECTED[provider]
        client, _state = make_client()
        resp = client.post("/api/setup/configure", json={"provider": provider, "api_key": KEY}, headers=pair(client))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert (body["provider"], body["model"], body["api_key_env"], body["api_key_saved"]) == (
            provider, model, env, True)
        assert body["base_url"] is None and KEY not in resp.text
        cfg = written(config_dir)
        assert (cfg["model"]["provider"], cfg["model"]["model"], cfg["model"]["api_key_env"]) == (provider, model, env)
        assert "base_url" not in cfg["model"] and KEY not in (config_dir / "prometheus.yaml").read_text()
        assert cfg["context"]["effective_limit"] == 64000
        assert parse_env_file(env_file)[env] == KEY, "the key is in the env file under the name the daemon reads"

    def test_what_is_written_is_a_provider_the_daemon_builds(self, provider, env_file, config_dir, monkeypatch):
        """The consumed layer: the registry builds a provider from the saved section and the saved key."""
        env, model, host = EXPECTED[provider]
        client, _state = make_client()
        client.post("/api/setup/configure", json={"provider": provider, "api_key": KEY}, headers=pair(client))
        monkeypatch.setenv(env, parse_env_file(env_file)[env])
        built = ProviderRegistry.create(written(config_dir)["model"])
        assert host in built._base_url and built._model == model

    def test_a_key_already_in_the_env_file_is_enough(self, provider, env_file, config_dir):
        env, _model, _host = EXPECTED[provider]
        env_file.write_text(f"{env}={KEY}\n", encoding="utf-8")
        client, _state = make_client()
        resp = client.post("/api/setup/configure", json={"provider": provider}, headers=pair(client))
        assert resp.status_code == 200 and resp.json()["api_key_saved"] is False

    def test_with_no_key_anywhere_it_says_so_and_names_the_variable(self, provider, env_file, config_dir):
        env, _model, _host = EXPECTED[provider]
        client, _state = make_client()
        resp = client.post("/api/setup/configure", json={"provider": provider}, headers=pair(client))
        assert resp.status_code == 400 and resp.json()["error"] == "cloud_key_missing"
        assert env in resp.json()["detail"]
        assert not (config_dir / "prometheus.yaml").exists()

    def test_a_model_may_be_chosen(self, provider, env_file, config_dir):
        client, _state = make_client()
        resp = client.post("/api/setup/configure", json={"provider": provider, "api_key": KEY, "model": "other-model"},
                           headers=pair(client))
        assert resp.status_code == 200 and written(config_dir)["model"]["model"] == "other-model"

    def test_a_cloud_provider_takes_no_address(self, provider, env_file, config_dir):
        client, _state = make_client()
        resp = client.post("/api/setup/configure", headers=pair(client),
                           json={"provider": provider, "api_key": KEY, "base_url": "http://localhost:8080"})
        assert resp.status_code == 400 and resp.json()["error"] == "unexpected_base_url"


class TestDetectListsThem:
    def test_the_cloud_presets_include_both_with_key_presence_not_values(self, env_file, config_dir):
        env_file.write_text(f"XAI_API_KEY={KEY}\n", encoding="utf-8")
        client, _state = make_client()
        rows = {r["provider"]: r for r in
                client.get("/api/setup/detect", headers=pair(client)).json()["cloud_providers"]}
        assert rows["gemini"] == {"provider": "gemini", "api_key_env": "GEMINI_API_KEY",
                                  "default_model": "gemini-2.5-flash", "key_present": False}
        assert rows["xai"] == {"provider": "xai", "api_key_env": "XAI_API_KEY", "default_model": "grok-4.5",
                               "key_present": True}


@pytest.mark.parametrize("provider", sorted(EXPECTED))
class TestWithoutAPrompt:
    def test_oara_setup_provider_writes_the_cloud_config(self, provider, tmp_path, monkeypatch):
        env, model, _host = EXPECTED[provider]
        monkeypatch.setenv(env, KEY)
        result = run_init(noninteractive=True, target_dir=tmp_path, timeout=0.1, provider=provider, candidates=[{
            "name": "ghost", "url": "http://127.0.0.1:9", "models_path": "/v1/models", "provider": "test"}])
        assert result is not None
        cfg = yaml.safe_load((tmp_path / "prometheus.yaml").read_text(encoding="utf-8"))
        assert (cfg["model"]["provider"], cfg["model"]["model"], cfg["model"]["api_key_env"]) == (provider, model, env)

    def test_the_command_line_offers_it(self, provider):
        import argparse

        from prometheus.cli.setup import add_setup_arguments

        parser = argparse.ArgumentParser()
        add_setup_arguments(parser)
        assert parser.parse_args(["--provider", provider]).setup_provider == provider


class TestTheEnvTemplate:
    def test_it_names_the_variable_the_daemon_reads(self):
        assert "GEMINI_API_KEY" in _ENV_TEMPLATE and "XAI_API_KEY" in _ENV_TEMPLATE
        assert "GOOGLE_API_KEY" not in _ENV_TEMPLATE, "the daemon never reads it: a key set that way is not found"

    def test_every_cloud_provider_setup_offers_has_its_variable_in_the_template(self):
        """The next provider added to the table cannot be forgotten here: OpenRouter already had been."""
        listed = set(re.findall(r"^#\s+([A-Z][A-Z0-9_]+_API_KEY)=", _ENV_TEMPLATE, flags=re.MULTILINE))
        missing = sorted({env for env, _model, _limit in init_mod._CLOUD_FAST_PROVIDERS.values()} - listed)
        assert missing == [], f"{missing} are offered by setup but absent from the env file template"

    def test_the_dead_end_menu_names_every_provider_it_can_take(self, monkeypatch, tmp_path):
        """It listed six by hand while the table had more; a person with a Gemini key was not told it was an option."""
        shown: list[str] = []

        def pick_exit(_label, options, default_index=0):
            shown.extend(options)
            return 2                                   # "show install instructions and exit"

        monkeypatch.setattr(init_mod, "_prompt_choice", pick_exit)
        monkeypatch.setattr(init_mod, "_print_install_instructions", lambda: None)
        assert init_mod._handle_no_server(tmp_path / "env", 0.1) == (None, None)
        cloud_line = next(line for line in shown if "cloud provider" in line).lower()
        for name in init_mod._CLOUD_FAST_PROVIDERS:
            assert name in cloud_line, f"{name} is offered by setup but not named in the menu"
