"""The CLI builds its model provider through the same factory the daemon uses.

Found by the 2026-09-01 installability survey: ``__main__.create_provider``
knew llama_cpp / ollama / anthropic and treated every other name as
llama.cpp, so a stranger whose setup chose ``provider: openai`` got a
daemon that worked and a ``prometheus`` CLI that silently talked to
``localhost:8080``. These tests pin the CLI to ``ProviderRegistry`` and
make the old fallback a loud exit.
"""

from __future__ import annotations

import pytest

from prometheus import __main__ as cli
from prometheus.providers.llama_cpp import LlamaCppProvider
from prometheus.providers.ollama import OllamaProvider
from prometheus.providers.openai_compat import OpenAICompatProvider
from prometheus.providers.registry import CLOUD_DEFAULTS, ProviderRegistry


@pytest.fixture()
def cloud_keys(monkeypatch):
    """Every cloud preset's default env var, set to a dummy value."""
    for name, preset in CLOUD_DEFAULTS.items():
        env = preset.get("default_env")
        if env:
            monkeypatch.setenv(env, f"dummy-{name}")
    # xai resolves through a credential helper that also accepts the plain key.
    monkeypatch.setenv("XAI_API_KEY", "dummy-xai")


class TestCloudProvidersReachTheCli:
    def test_openai_is_not_llama_cpp(self, cloud_keys) -> None:
        provider, model = cli.create_provider(
            {"provider": "openai", "model": "gpt-4o", "api_key_env": "OPENAI_API_KEY"}
        )
        assert isinstance(provider, OpenAICompatProvider)
        assert not isinstance(provider, LlamaCppProvider)
        assert model == "gpt-4o"

    @pytest.mark.parametrize("name", sorted(CLOUD_DEFAULTS))
    def test_every_cloud_preset_matches_the_daemon_factory(self, cloud_keys, name) -> None:
        """Parity, not a class list: whatever the registry builds for the
        daemon, the CLI builds the same type — a new provider added to the
        registry cannot regress to the old llama.cpp fallback here."""
        cfg = {"provider": name}
        expected = type(ProviderRegistry.create(cfg))
        provider, model = cli.create_provider(cfg)
        assert type(provider) is expected
        assert not isinstance(provider, LlamaCppProvider)
        # The default model comes from the preset, not the local default.
        assert model == CLOUD_DEFAULTS[name]["model"]


class TestLocalProvidersUnchanged:
    def test_llama_cpp_keeps_its_base_url(self) -> None:
        provider, model = cli.create_provider(
            {"provider": "llama_cpp", "base_url": "http://gpu-box:8080", "model": "q"}
        )
        assert isinstance(provider, LlamaCppProvider)
        assert provider._base_url == "http://gpu-box:8080"
        assert model == "q"

    def test_ollama(self) -> None:
        provider, _ = cli.create_provider({"provider": "ollama"})
        assert isinstance(provider, OllamaProvider)

    def test_absent_provider_is_llama_cpp(self) -> None:
        provider, model = cli.create_provider({})
        assert isinstance(provider, LlamaCppProvider)
        assert model == "qwen3.5-32b"


class TestNoSilentFallback:
    def test_unknown_provider_exits_2_and_names_the_valid_ones(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            cli.create_provider({"provider": "does_not_exist"})
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "does_not_exist" in err
        assert "Valid providers" in err

    def test_cloud_provider_without_a_key_exits_2_not_localhost(self, monkeypatch, capsys) -> None:
        """The stranger's case: cloud chosen, key missing. The old code
        answered with a llama.cpp client aimed at localhost:8080."""
        for preset in CLOUD_DEFAULTS.values():
            if preset.get("default_env"):
                monkeypatch.delenv(preset["default_env"], raising=False)
        with pytest.raises(SystemExit) as exc:
            cli.create_provider({"provider": "openai", "model": "gpt-4o"})
        assert exc.value.code == 2
        assert "API key" in capsys.readouterr().err
