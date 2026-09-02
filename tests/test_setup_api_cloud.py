"""Remote (couch-mode) setup can land on a cloud provider.

Installability survey, 2026-09-01: ``POST /api/setup/configure`` required
a ``base_url`` and re-probed it with an unauthenticated ``GET /v1/models``
— a cloud endpoint 401s that probe, and the body had no field for a key
at all, so Beacon's wizard could not produce a cloud-only install.
Contract under test: a cloud provider takes a key instead of a URL, the
key lands in the env file (never the yaml, never the response), and
without a key anywhere the call is refused — the same dead-end rule the
local path enforces with the probe.
"""

from __future__ import annotations

import pytest
import yaml

pytest.importorskip("fastapi")

from prometheus.cli.init import _CLOUD_FAST_PROVIDERS, _cloud_default_config  # noqa: E402
from prometheus.config.env_file import parse_env_file  # noqa: E402
from tests.test_setup_api_phase2 import (  # noqa: E402
    _FakeLlamaCppHandler,
    _serve,
    config_dir,  # noqa: F401 — fixtures re-exported
    env_file,  # noqa: F401
    make_client,
    pair,
)

KEY = "dummy-anthropic-key-for-the-api-tests"


@pytest.fixture(autouse=True)
def no_cloud_keys_in_environment(monkeypatch):
    for key_env, _model, _limit in _CLOUD_FAST_PROVIDERS.values():
        monkeypatch.delenv(key_env, raising=False)


def _cfg(config_dir):
    return yaml.safe_load((config_dir / "prometheus.yaml").read_text(encoding="utf-8"))


class TestConfigureCloud:
    def test_cloud_provider_with_api_key_writes_the_cli_writer_config(self, env_file, config_dir):
        client, state = make_client(api_port=8005, ws_port=8010)
        headers = pair(client)
        resp = client.post(
            "/api/setup/configure",
            json={"provider": "anthropic", "api_key": KEY},
            headers=headers,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["configured"] is True
        assert body["provider"] == "anthropic"
        assert body["base_url"] is None
        assert body["model"] == _CLOUD_FAST_PROVIDERS["anthropic"][1]
        assert body["api_key_env"] == "ANTHROPIC_API_KEY"
        assert body["api_key_saved"] is True
        assert KEY not in resp.text
        assert state.configured is True

        cfg = _cfg(config_dir)
        # ONE writer with the CLI: the model section is what
        # _cloud_default_config produces, byte for byte.
        expected = _cloud_default_config("anthropic", "ANTHROPIC_API_KEY", body["model"])
        assert yaml.safe_dump(cfg["model"], sort_keys=False) == \
            yaml.safe_dump(expected["model"], sort_keys=False)
        assert "base_url" not in cfg["model"]
        assert cfg["web"] == {"enabled": True, "api_port": 8005, "ws_port": 8010}
        # The key: in the env file, not the yaml.
        assert parse_env_file(env_file)["ANTHROPIC_API_KEY"] == KEY
        assert KEY not in (config_dir / "prometheus.yaml").read_text(encoding="utf-8")

    def test_model_override_is_honoured(self, env_file, config_dir):
        client, _ = make_client()
        headers = pair(client)
        resp = client.post(
            "/api/setup/configure",
            json={"provider": "openai", "api_key": "dummy-openai-key", "model": "gpt-4.1-mini"},
            headers=headers,
        )
        assert resp.status_code == 200, resp.text
        assert _cfg(config_dir)["model"]["model"] == "gpt-4.1-mini"

    def test_key_already_in_the_env_file_is_enough(self, env_file, config_dir):
        env_file.write_text("DEEPSEEK_API_KEY=dummy-deepseek-key-from-file\n")
        client, _ = make_client()
        headers = pair(client)
        resp = client.post(
            "/api/setup/configure", json={"provider": "deepseek"}, headers=headers,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["api_key_saved"] is False
        assert _cfg(config_dir)["model"]["provider"] == "deepseek"
        # Untouched: the file still holds the key it had, nothing was rewritten over it.
        assert parse_env_file(env_file)["DEEPSEEK_API_KEY"] == "dummy-deepseek-key-from-file"

    def test_no_key_anywhere_is_refused_and_writes_nothing(self, env_file, config_dir):
        """Dead-end rule over the wire: a cloud config without its key is
        known-broken, exactly like a base_url nothing answers on."""
        client, state = make_client()
        headers = pair(client)
        resp = client.post(
            "/api/setup/configure", json={"provider": "openai"}, headers=headers,
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"] == "cloud_key_missing"
        assert "OPENAI_API_KEY" in body["detail"]
        assert not (config_dir / "prometheus.yaml").exists()
        assert state.configured is False

    def test_base_url_with_a_cloud_provider_is_a_400_not_silently_dropped(self, env_file, config_dir):
        client, _ = make_client()
        headers = pair(client)
        resp = client.post(
            "/api/setup/configure",
            json={"provider": "anthropic", "api_key": KEY, "base_url": "http://127.0.0.1:8080"},
            headers=headers,
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "unexpected_base_url"
        assert not (config_dir / "prometheus.yaml").exists()
        # And nothing was persisted on the way to the refusal.
        assert not env_file.exists() or "ANTHROPIC_API_KEY" not in parse_env_file(env_file)

    def test_local_provider_path_is_unchanged(self, env_file, config_dir):
        client, _ = make_client()
        headers = pair(client)
        with _serve(_FakeLlamaCppHandler) as url:
            resp = client.post(
                "/api/setup/configure",
                json={"provider": "llama_cpp", "base_url": url, "model": "gemma4-26b"},
                headers=headers,
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["base_url"] == url
        assert body["api_key_env"] is None
        assert body["api_key_saved"] is False


class TestDetectAdvertisesCloud:
    def test_cloud_presets_ride_along_with_key_presence_not_values(self, env_file, config_dir, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key-in-env")
        env_file.write_text("MOONSHOT_API_KEY=dummy-kimi-key-in-file\n")
        client, _ = make_client()
        headers = pair(client)
        resp = client.get(
            "/api/setup/detect", params={"base_url": "http://127.0.0.1:9"}, headers=headers,
        )
        assert resp.status_code == 200
        cloud = {c["provider"]: c for c in resp.json()["cloud_providers"]}
        assert set(cloud) == set(_CLOUD_FAST_PROVIDERS)
        assert cloud["openai"]["key_present"] is True
        assert cloud["kimi"]["key_present"] is True
        assert cloud["anthropic"]["key_present"] is False
        assert cloud["anthropic"]["api_key_env"] == "ANTHROPIC_API_KEY"
        assert cloud["anthropic"]["default_model"] == _CLOUD_FAST_PROVIDERS["anthropic"][1]
        assert "dummy-" not in resp.text
