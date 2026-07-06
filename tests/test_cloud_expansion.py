"""CLOUD EXPANSION (2026-07) — four new LLM providers, fakes only.

Covers the provider layer for DeepSeek / Kimi (Moonshot) / GLM (Z.ai) /
MiMo (Xiaomi):

* ProviderRegistry.create() + default-model pins (incl. the pin that the
  DeepSeek default is NOT a deprecated pre-V4 alias),
* the GLM base-with-path URL join (openai_compat._chat_completions_url),
* OVERRIDE_PRESETS / SLASH_COMMAND_NAMES / display names,
* cmd_provider_override key-missing messages,
* config-surface coverage (wizard tables, ENV_OVERRIDES, yaml defaults,
  PRICING),
* gateway handler spot-checks per the G1 pattern (Telegram behavioral;
  Slack/Discord preset-name source pins on top of test_gateway_parity's
  registration assertions).

No network, no keys — everything here runs against fakes and static tables.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from prometheus.providers.openai_compat import (
    OpenAICompatProvider,
    _chat_completions_url,
)
from prometheus.providers.registry import CLOUD_DEFAULTS, ProviderRegistry
from prometheus.router.model_router import (
    OVERRIDE_PRESETS,
    SLASH_COMMAND_NAMES,
    resolve_slash_command_target,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

NEW_PROVIDERS = ("deepseek", "kimi", "glm", "mimo")

# provider -> (env var, default base_url, default model)
EXPECTED: dict[str, tuple[str, str, str]] = {
    "deepseek": (
        "DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-v4-flash",
    ),
    "kimi": (
        "MOONSHOT_API_KEY", "https://api.moonshot.ai/v1", "kimi-k2.6",
    ),
    "glm": (
        "ZAI_API_KEY", "https://api.z.ai/api/paas/v4", "glm-5.2",
    ),
    "mimo": (
        "MIMO_API_KEY", "https://api.xiaomimimo.com/v1", "mimo-v2.5-pro",
    ),
}


# -----------------------------------------------------------------------
# ProviderRegistry
# -----------------------------------------------------------------------


class TestRegistryCreate:
    @pytest.mark.parametrize("provider", NEW_PROVIDERS)
    def test_create_returns_openai_compat(self, provider: str) -> None:
        env_var, base_url, model = EXPECTED[provider]
        with patch.dict(os.environ, {env_var: "test-key"}):
            p = ProviderRegistry.create({"provider": provider})
        assert isinstance(p, OpenAICompatProvider)
        assert p._base_url == base_url.rstrip("/")
        assert p._model == model
        assert p._api_key == "test-key"

    @pytest.mark.parametrize("provider", NEW_PROVIDERS)
    def test_missing_key_raises_with_env_name(self, provider: str) -> None:
        env_var, _, _ = EXPECTED[provider]
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match=env_var):
                ProviderRegistry.create({"provider": provider})

    @pytest.mark.parametrize("provider", NEW_PROVIDERS)
    def test_is_cloud(self, provider: str) -> None:
        assert ProviderRegistry.is_cloud(provider) is True

    def test_list_providers_includes_new_four(self) -> None:
        providers = ProviderRegistry.list_providers()
        for provider in NEW_PROVIDERS:
            assert provider in providers

    def test_cloud_defaults_cover_new_four(self) -> None:
        for provider in NEW_PROVIDERS:
            env_var, base_url, model = EXPECTED[provider]
            defaults = CLOUD_DEFAULTS[provider]
            assert defaults["default_env"] == env_var
            assert defaults["base_url"] == base_url
            assert defaults["model"] == model


class TestDeepSeekDefaultModelPin:
    """The single most load-bearing pin of the sprint: DeepSeek's legacy
    ``deepseek-chat`` / ``deepseek-reasoner`` aliases are deprecated
    2026-07-24. Every default in this repo must ship a V4 name."""

    DEPRECATED = ("deepseek-chat", "deepseek-reasoner")

    def test_registry_default_is_v4_not_deprecated_alias(self) -> None:
        model = CLOUD_DEFAULTS["deepseek"]["model"]
        assert model == "deepseek-v4-flash"
        assert model not in self.DEPRECATED

    def test_override_preset_is_v4_not_deprecated_alias(self) -> None:
        model = OVERRIDE_PRESETS["deepseek"]["model"]
        assert model == "deepseek-v4-flash"
        assert model not in self.DEPRECATED

    def test_yaml_default_slash_command_is_v4(self) -> None:
        cfg = yaml.safe_load(
            (_REPO_ROOT / "config" / "prometheus.yaml.default").read_text()
        )
        model = cfg["slash_commands"]["deepseek"]["model"]
        assert model == "deepseek-v4-flash"
        assert model not in self.DEPRECATED


# -----------------------------------------------------------------------
# GLM base-with-path URL join
# -----------------------------------------------------------------------


class TestChatCompletionsUrlJoin:
    def test_glm_versioned_path_base_gets_no_extra_v1(self) -> None:
        """GLM's base already contains a path AND a version segment — the
        join must NOT insert another /v1."""
        assert _chat_completions_url("https://api.z.ai/api/paas/v4") == (
            "https://api.z.ai/api/paas/v4/chat/completions"
        )

    def test_glm_trailing_slash_tolerated(self) -> None:
        assert _chat_completions_url("https://api.z.ai/api/paas/v4/") == (
            "https://api.z.ai/api/paas/v4/chat/completions"
        )

    def test_v1_suffix_bases_unchanged(self) -> None:
        # openai / xai / moonshot / mimo style
        assert _chat_completions_url("https://api.openai.com/v1") == (
            "https://api.openai.com/v1/chat/completions"
        )
        assert _chat_completions_url("https://api.moonshot.ai/v1") == (
            "https://api.moonshot.ai/v1/chat/completions"
        )
        assert _chat_completions_url("https://api.xiaomimimo.com/v1") == (
            "https://api.xiaomimimo.com/v1/chat/completions"
        )

    def test_bare_host_gets_v1(self) -> None:
        # deepseek serves both forms; the bare host gets the /v1 prefix
        assert _chat_completions_url("https://api.deepseek.com") == (
            "https://api.deepseek.com/v1/chat/completions"
        )

    def test_gemini_openai_suffix_keeps_historical_behavior(self) -> None:
        """Gemini's base ends in /openai (not a version segment) — the
        historical join is preserved byte-for-byte."""
        assert _chat_completions_url(
            "https://generativelanguage.googleapis.com/v1beta/openai"
        ) == (
            "https://generativelanguage.googleapis.com/v1beta/openai"
            "/v1/chat/completions"
        )

    def test_glm_provider_instance_joins_correctly(self) -> None:
        p = OpenAICompatProvider(
            base_url="https://api.z.ai/api/paas/v4",
            api_key="test-key",
            model="glm-5.2",
        )
        assert _chat_completions_url(p._base_url) == (
            "https://api.z.ai/api/paas/v4/chat/completions"
        )


# -----------------------------------------------------------------------
# Router presets + slash-command resolution
# -----------------------------------------------------------------------


class TestOverridePresets:
    @pytest.mark.parametrize("name", NEW_PROVIDERS)
    def test_preset_shape(self, name: str) -> None:
        env_var, _, model = EXPECTED[name]
        preset = OVERRIDE_PRESETS[name]
        assert preset["provider"] == name
        assert preset["api_key_env"] == env_var
        assert preset["model"] == model

    def test_slash_command_names_include_new_four(self) -> None:
        for name in NEW_PROVIDERS:
            assert name in SLASH_COMMAND_NAMES

    @pytest.mark.parametrize("name", NEW_PROVIDERS)
    def test_resolve_falls_back_to_preset(self, name: str) -> None:
        assert resolve_slash_command_target(name, {}) == dict(
            OVERRIDE_PRESETS[name]
        )

    @pytest.mark.parametrize("name", NEW_PROVIDERS)
    def test_resolve_honors_user_config(self, name: str) -> None:
        cfg = {"slash_commands": {name: {"model": "user-pinned-model"}}}
        target = resolve_slash_command_target(name, cfg)
        assert target["model"] == "user-pinned-model"
        assert target["provider"] == name


# -----------------------------------------------------------------------
# cmd_provider_override — key-missing message + display names
# -----------------------------------------------------------------------


class TestProviderOverrideCommand:
    def _agent_loop_with_router(self) -> MagicMock:
        agent_loop = MagicMock()
        router = MagicMock()
        router.config.overrides_enabled = True
        agent_loop._model_router = router
        return agent_loop

    @pytest.mark.parametrize("name", NEW_PROVIDERS)
    def test_missing_key_message_is_actionable(self, name: str) -> None:
        from prometheus.gateway.commands import (
            PROVIDER_PRESET_DISPLAY_NAMES,
            cmd_provider_override,
        )

        env_var, _, _ = EXPECTED[name]
        agent_loop = self._agent_loop_with_router()
        with patch.dict(os.environ, {}, clear=True):
            text, applied = cmd_provider_override(
                agent_loop, {}, "telegram:1", name,
            )
        assert applied is False
        assert env_var in text
        assert PROVIDER_PRESET_DISPLAY_NAMES[name] in text
        assert "~/.config/prometheus/env" in text
        agent_loop._model_router.set_override.assert_not_called()

    @pytest.mark.parametrize("name", NEW_PROVIDERS)
    def test_key_present_applies_override(self, name: str) -> None:
        from prometheus.gateway.commands import cmd_provider_override

        env_var, _, model = EXPECTED[name]
        agent_loop = self._agent_loop_with_router()
        with patch.dict(os.environ, {env_var: "test-key"}):
            text, applied = cmd_provider_override(
                agent_loop, {}, "telegram:1", name,
            )
        assert applied is True
        assert model in text
        agent_loop._model_router.set_override.assert_called_once_with(
            "telegram:1", dict(OVERRIDE_PRESETS[name]),
        )

    def test_display_names_cover_new_four(self) -> None:
        from prometheus.gateway.commands import PROVIDER_PRESET_DISPLAY_NAMES

        for name in NEW_PROVIDERS:
            assert name in PROVIDER_PRESET_DISPLAY_NAMES

    def test_route_output_lists_new_commands(self) -> None:
        from prometheus.gateway.commands import cmd_route

        agent_loop = MagicMock()
        agent_loop._model_router = None
        text = cmd_route(agent_loop, "telegram:1", "m", "llama_cpp")
        for name in NEW_PROVIDERS:
            assert f"/{name}" in text


# -----------------------------------------------------------------------
# Gateway handler spot-checks (G1 pattern)
# -----------------------------------------------------------------------


class TestTelegramHandlers:
    """Behavioral spot-check on the reference gateway: each new command
    handler routes to _apply_override with its own preset name."""

    def _make_adapter(self):
        from prometheus.gateway.config import Platform, PlatformConfig
        from prometheus.gateway.telegram import TelegramAdapter
        from prometheus.tools.base import ToolRegistry

        agent_loop = AsyncMock()
        agent_loop._model_router = None
        return TelegramAdapter(
            config=PlatformConfig(platform=Platform.TELEGRAM, token="test"),
            agent_loop=agent_loop,
            tool_registry=ToolRegistry(),
            model_name="test-model",
            model_provider="llama_cpp",
        )

    @pytest.mark.parametrize("name", NEW_PROVIDERS)
    async def test_command_maps_to_its_preset(self, name: str) -> None:
        adapter = self._make_adapter()
        seen: list[str] = []

        async def _spy(update, context, preset_name):
            seen.append(preset_name)

        adapter._apply_override = _spy
        handler = getattr(adapter, f"_cmd_{name}")
        await handler(MagicMock(), MagicMock())
        assert seen == [name]


class TestSlackDiscordPresetSourcePins:
    """test_gateway_parity asserts registration + handler existence on all
    three platforms; these pins add the preset-name mapping for the two
    gateways whose adapters need live platform objects to construct."""

    @pytest.mark.parametrize("name", NEW_PROVIDERS)
    def test_slack_handler_passes_own_preset(self, name: str) -> None:
        src = (
            _REPO_ROOT / "src" / "prometheus" / "gateway" / "slack.py"
        ).read_text()
        pattern = (
            rf"async def _slash_{name}\(.*?"
            rf'preset_name="{name}"'
        )
        assert re.search(pattern, src, re.DOTALL), (
            f"slack _slash_{name} must call _apply_override with "
            f'preset_name="{name}"'
        )

    @pytest.mark.parametrize("name", NEW_PROVIDERS)
    def test_discord_handler_passes_own_preset(self, name: str) -> None:
        src = (
            _REPO_ROOT / "src" / "prometheus" / "gateway" / "discord.py"
        ).read_text()
        pattern = (
            rf"async def _app_{name}\(.*?"
            rf'preset_name="{name}"'
        )
        assert re.search(pattern, src, re.DOTALL), (
            f"discord _app_{name} must call _apply_override with "
            f'preset_name="{name}"'
        )


# -----------------------------------------------------------------------
# Config-surface coverage: wizard, env overrides, yaml defaults, pricing
# -----------------------------------------------------------------------


class TestConfigSurfaces:
    def test_wizard_tables_cover_new_four(self) -> None:
        from prometheus.setup_wizard import (
            CLOUD_DEFAULT_ENV_VARS,
            CLOUD_PROVIDER_MODELS,
            PROVIDER_EFFECTIVE_LIMITS,
        )

        for name in NEW_PROVIDERS:
            env_var, _, model = EXPECTED[name]
            assert CLOUD_DEFAULT_ENV_VARS[name] == env_var
            assert model in [m for m, _, _ in CLOUD_PROVIDER_MODELS[name]]
            assert PROVIDER_EFFECTIVE_LIMITS[name] > 0

    def test_fast_path_covers_new_four(self) -> None:
        from prometheus.cli.init import _CLOUD_FAST_PROVIDERS

        for name in NEW_PROVIDERS:
            env_var, _, model = EXPECTED[name]
            key_env, default_model, limit = _CLOUD_FAST_PROVIDERS[name]
            assert key_env == env_var
            assert default_model == model
            assert limit > 0

    def test_env_template_mentions_all_six_service_vars(self) -> None:
        from prometheus.cli.init import _ENV_TEMPLATE

        for var in (
            "DEEPSEEK_API_KEY", "MOONSHOT_API_KEY", "ZAI_API_KEY",
            "MIMO_API_KEY", "DASHSCOPE_API_KEY", "KLING_ACCESS_KEY",
            "KLING_SECRET_KEY",
        ):
            assert var in _ENV_TEMPLATE

    def test_env_overrides_cover_new_vars(self) -> None:
        from prometheus.config.env_override import ENV_OVERRIDES

        assert ENV_OVERRIDES["DEEPSEEK_API_KEY"] == ("providers", "deepseek", "api_key")
        assert ENV_OVERRIDES["MOONSHOT_API_KEY"] == ("providers", "kimi", "api_key")
        assert ENV_OVERRIDES["ZAI_API_KEY"] == ("providers", "glm", "api_key")
        assert ENV_OVERRIDES["MIMO_API_KEY"] == ("providers", "mimo", "api_key")
        assert ENV_OVERRIDES["DASHSCOPE_API_KEY"] == (
            "image_generation", "dashscope", "api_key",
        )
        assert ENV_OVERRIDES["KLING_ACCESS_KEY"] == (
            "video_generation", "kling", "access_key",
        )
        assert ENV_OVERRIDES["KLING_SECRET_KEY"] == (
            "video_generation", "kling", "secret_key",
        )

    def test_yaml_default_slash_commands_blocks(self) -> None:
        cfg = yaml.safe_load(
            (_REPO_ROOT / "config" / "prometheus.yaml.default").read_text()
        )
        for name in NEW_PROVIDERS:
            env_var, _, model = EXPECTED[name]
            block = cfg["slash_commands"][name]
            assert block["provider"] == name
            assert block["api_key_env"] == env_var
            assert block["model"] == model

    def test_yaml_default_media_blocks(self) -> None:
        cfg = yaml.safe_load(
            (_REPO_ROOT / "config" / "prometheus.yaml.default").read_text()
        )
        ds = cfg["image_generation"]["dashscope"]
        assert ds["api_key_env"] == "DASHSCOPE_API_KEY"
        assert ds["model"] == "wan2.5-t2i-preview"
        assert ds["base_url"].startswith("https://dashscope")
        # auto stays the default — dashscope must be a deliberate flip
        assert cfg["image_generation"]["default_backend"] == "auto"
        kling = cfg["video_generation"]["kling"]
        assert kling["access_key_env"] == "KLING_ACCESS_KEY"
        assert kling["secret_key_env"] == "KLING_SECRET_KEY"
        assert kling["model_name"] == "kling-v3"
        assert kling["poll_budget_seconds"] >= 300

    def test_pricing_covers_new_models(self) -> None:
        from prometheus.telemetry.cost import PRICING

        for model in (
            "deepseek-v4-flash", "deepseek-v4-pro", "kimi-k2.6",
            "glm-5.2", "mimo-v2.5-pro",
        ):
            inp, out = PRICING[model]
            assert inp > 0 and out > 0

    def test_model_registry_covers_cloud_models(self) -> None:
        from prometheus.infra.doctor import match_model

        registry = yaml.safe_load(
            (_REPO_ROOT / "config" / "model_registry.yaml").read_text()
        )
        expectations = {
            "deepseek-v4-flash": ("DeepSeek V4 (cloud API)", 1000000),
            "kimi-k2.6": ("Kimi K2 (Moonshot cloud API)", 262144),
            "glm-5.2": ("GLM-5 (Z.ai cloud API)", 1000000),
            "mimo-v2.5-pro": ("MiMo (Xiaomi cloud API)", 1000000),
        }
        for model_name, (display, context) in expectations.items():
            family = match_model(model_name, registry)
            assert family is not None, f"{model_name} not matched in registry"
            assert family["display_name"] == display
            caps = family["capabilities"]
            assert caps["function_calling"]["supported"] is True
            assert caps["streaming"]["supported"] is True
            assert context in family["context_sizes"]


# -----------------------------------------------------------------------
# Adapter tier + secondary cloud-provider surfaces
# -----------------------------------------------------------------------


class TestAdapterTierAndSecondarySurfaces:
    @pytest.mark.parametrize("name", NEW_PROVIDERS)
    def test_build_adapter_for_is_tier_off(self, name: str) -> None:
        """The override path builds adapters per provider name. The new
        providers are API-enforced-structure clouds — they must get
        tier=off (validator NONE, no retries), NOT the local full
        pipeline. This was a real drive-by catch: without the fix they
        fell through to the QwenFormatter local branch."""
        from prometheus.adapter import ModelAdapter
        from prometheus.adapter.formatter import PassthroughFormatter
        from prometheus.router.model_router import _build_adapter_for

        adapter = _build_adapter_for(name)
        assert adapter.tier == ModelAdapter.TIER_OFF
        assert isinstance(adapter.formatter, PassthroughFormatter)

    @pytest.mark.parametrize("name", NEW_PROVIDERS)
    def test_telemetry_cloud_set_includes_new_providers(self, name: str) -> None:
        """Golden-trace capture keys off tracker._CLOUD_PROVIDERS (a
        deliberate duplicate of ProviderRegistry.is_cloud)."""
        from prometheus.telemetry.tracker import _CLOUD_PROVIDERS

        assert name in _CLOUD_PROVIDERS

    @pytest.mark.parametrize("name", NEW_PROVIDERS)
    def test_web_slash_router_declares_boundary(self, name: str) -> None:
        """The web chat surface mirrors Telegram's command registrations in
        WEB_NATIVE_ONLY so /deepseek etc. get an explicit boundary reply
        instead of silently running the agent."""
        from prometheus.web.slash_router import WEB_NATIVE_ONLY

        assert name in WEB_NATIVE_ONLY

    def test_model_rest_catalog_labels(self) -> None:
        """GET /api/models labels (web/server.py _PRESET_LABELS) cover the
        new presets — pinned at the source level (the labels dict is
        function-local to create_app)."""
        src = (
            _REPO_ROOT / "src" / "prometheus" / "web" / "server.py"
        ).read_text()
        for key, label in (
            ("deepseek", "DeepSeek"), ("kimi", "Kimi"),
            ("glm", "GLM"), ("mimo", "MiMo"),
        ):
            assert f'"{key}": "{label}"' in src


# -----------------------------------------------------------------------
# Doctor — cloud-keys info line
# -----------------------------------------------------------------------


class TestDoctorCloudKeys:
    def test_lists_all_six_services_without_values(self) -> None:
        from prometheus.infra.doctor import Doctor

        secret = "sk-super-secret-value-000"
        env = {
            "DEEPSEEK_API_KEY": secret,
            "KLING_ACCESS_KEY": secret,
            # KLING_SECRET_KEY deliberately absent → pair reads not set
        }
        with patch.dict(os.environ, env, clear=True):
            check = Doctor({})._check_cloud_keys()
        assert check.status == "info"
        assert check.category == "connectivity"
        assert "DeepSeek set" in check.message
        assert "Kimi not set" in check.message
        assert "GLM not set" in check.message
        assert "MiMo not set" in check.message
        assert "DashScope/WAN not set" in check.message
        # Kling requires BOTH halves of the pair
        assert "Kling AK+SK not set" in check.message
        # Never the value — not even masked
        assert secret not in check.message
        assert secret[:6] not in check.message

    def test_kling_pair_set_when_both_present(self) -> None:
        from prometheus.infra.doctor import Doctor

        env = {"KLING_ACCESS_KEY": "a", "KLING_SECRET_KEY": "b"}
        with patch.dict(os.environ, env, clear=True):
            check = Doctor({})._check_cloud_keys()
        assert "Kling AK+SK set" in check.message
