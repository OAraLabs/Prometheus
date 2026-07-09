"""ProviderRegistry — factory that creates the right provider from config.

Maps provider name strings to classes. Reads API keys from environment
variables (api_key_env config field), never from the config file itself.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from prometheus.providers.base import ModelProvider

log = logging.getLogger(__name__)

# Default base URLs and models per cloud provider
CLOUD_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "default_env": "OPENAI_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.5-flash",
        "default_env": "GEMINI_API_KEY",
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "model": "grok-3",
        "default_env": "XAI_API_KEY",
    },
    "anthropic": {
        "model": "claude-haiku-4-5-20251001",
        "default_env": "ANTHROPIC_API_KEY",
    },
    # -- CLOUD EXPANSION (2026-07) — endpoints/models verified 2026-07-05 --
    "deepseek": {
        # DeepSeek serves /v1/chat/completions with or without the /v1
        # prefix; the bare host is their documented base.
        "base_url": "https://api.deepseek.com",
        # V4 names ship here deliberately: the legacy `deepseek-chat` /
        # `deepseek-reasoner` aliases are deprecated 2026-07-24. The
        # reasoning flagship is `deepseek-v4-pro` (pin it via
        # slash_commands.deepseek.model or model.model in prometheus.yaml).
        "model": "deepseek-v4-flash",
        "default_env": "DEEPSEEK_API_KEY",
    },
    "kimi": {
        # Moonshot AI international endpoint. A separate CN endpoint
        # (https://api.moonshot.cn/v1) exists with SEPARATE keys — point
        # base_url there in config if your key is CN-issued.
        "base_url": "https://api.moonshot.ai/v1",
        "model": "kimi-k2.6",
        "default_env": "MOONSHOT_API_KEY",
    },
    "glm": {
        # Z.ai (Zhipu) — note the nonstandard /api/paas/v4 path prefix; the
        # OpenAI-compat provider appends /chat/completions to version-suffixed
        # bases (see openai_compat._chat_completions_url). CN mainland
        # endpoint (https://open.bigmodel.cn/api/paas/v4) has the same shape.
        "base_url": "https://api.z.ai/api/paas/v4",
        "model": "glm-5.2",
        "default_env": "ZAI_API_KEY",
    },
    "mimo": {
        # Xiaomi MiMo first-party hosted platform.
        "base_url": "https://api.xiaomimimo.com/v1",
        "model": "mimo-v2.5-pro",
        "default_env": "MIMO_API_KEY",
    },
}

# Providers that use the OpenAI-compatible wire format
_OPENAI_COMPAT_PROVIDERS = {"openai", "gemini", "xai", "deepseek", "kimi", "glm", "mimo"}


def _resolve_api_key(config: dict[str, Any], provider_name: str) -> str:
    """Resolve the API key from config or environment.

    Checks (in order):
      1. config["api_key"] — direct key (not recommended)
      2. config["api_key_env"] — name of env var to read
      3. CLOUD_DEFAULTS[provider_name]["default_env"] — fallback env var
    """
    # Direct key (e.g. from test configs)
    direct = config.get("api_key", "")
    if direct:
        return direct

    # Explicit env var name
    env_name = config.get("api_key_env", "")
    if env_name:
        key = os.environ.get(env_name, "")
        if key:
            return key
        raise ValueError(
            f"Environment variable {env_name} is not set. "
            f"Set it with: export {env_name}=your-key"
        )

    # Default env var for this provider
    defaults = CLOUD_DEFAULTS.get(provider_name, {})
    default_env = defaults.get("default_env", "")
    if default_env:
        key = os.environ.get(default_env, "")
        if key:
            return key
        raise ValueError(
            f"No API key configured for {provider_name}. "
            f"Set {default_env} or add api_key_env to your config."
        )

    raise ValueError(f"No API key source found for provider {provider_name}")


def _resolve_xai_credential(config: dict[str, Any]) -> "str | object":
    """Resolve the xAI bearer, preferring a SuperGrok OAuth token over a key.

    Precedence:
      1. A direct ``config["api_key"]`` (explicit/test override) — static str.
      2. SuperGrok OAuth, if the user has logged in — a *callable* resolved per
         request (the token refreshes mid-session, so it must not be frozen at
         construction). The callable falls back to ``XAI_API_KEY`` if an OAuth
         refresh ever fails, so an outage degrades to the key instead of a hard
         failure.
      3. Otherwise the normal env-var key path (raises if unset).
    """
    from prometheus.providers import xai_oauth

    direct = config.get("api_key", "")
    if direct:
        return direct

    if xai_oauth.is_logged_in():
        env_name = config.get("api_key_env", "") or CLOUD_DEFAULTS["xai"]["default_env"]

        def _bearer() -> str | None:
            token = xai_oauth.get_access_token()
            if token:
                return token
            log.warning("xAI OAuth token unavailable — falling back to %s", env_name)
            return os.environ.get(env_name, "") or None

        return _bearer

    return _resolve_api_key(config, "xai")


class ProviderRegistry:
    """Create providers from prometheus.yaml config."""

    @staticmethod
    def create(config: dict[str, Any]) -> ModelProvider:
        """Create a ModelProvider from the model config section.

        Example config::

            model:
              provider: "openai"
              api_key_env: "OPENAI_API_KEY"
              model: "gpt-4o"
        """
        provider_name = config.get("provider", "llama_cpp")
        defaults = CLOUD_DEFAULTS.get(provider_name, {})

        if provider_name in _OPENAI_COMPAT_PROVIDERS:
            from prometheus.providers.openai_compat import OpenAICompatProvider

            api_key = (
                _resolve_xai_credential(config)
                if provider_name == "xai"
                else _resolve_api_key(config, provider_name)
            )
            return OpenAICompatProvider(
                base_url=config.get("base_url", defaults.get("base_url", "")),
                api_key=api_key,
                model=config.get("model", defaults.get("model", "")),
                default_max_tokens=config.get("max_tokens", 4096),
                timeout=config.get("timeout", 120.0),
            )

        if provider_name == "anthropic":
            from prometheus.providers.anthropic import AnthropicProvider

            api_key = _resolve_api_key(config, provider_name)
            return AnthropicProvider(
                api_key=api_key,
                model=config.get("model", defaults.get("model", "claude-haiku-4-5-20251001")),
                timeout=config.get("timeout", 120.0),
                prompt_caching=config.get("prompt_caching", True),
            )

        if provider_name == "llama_cpp":
            from prometheus.providers.llama_cpp import LlamaCppProvider

            return LlamaCppProvider(
                base_url=config.get("base_url", "http://localhost:8080"),
                timeout=config.get("timeout", 120.0),
                # Defaults to True. The model.suppress_thinking key gates the
                # chat_template_kwargs injection that prevents gemma/qwen from
                # burning their token budgets on a thought channel and emitting
                # empty content (briefing, memory extractor, tool calls).
                suppress_thinking=config.get("suppress_thinking", True),
            )

        if provider_name == "ollama":
            from prometheus.providers.ollama import OllamaProvider

            return OllamaProvider(
                base_url=config.get("base_url", "http://localhost:11434"),
                timeout=config.get("timeout", 120.0),
            )

        if provider_name == "stub":
            from prometheus.providers.stub import StubProvider

            return StubProvider(
                base_url=config.get("base_url", "http://localhost:8080"),
                timeout=config.get("timeout", 120.0),
            )

        raise ValueError(
            f"Unknown provider: {provider_name!r}. "
            f"Valid providers: llama_cpp, ollama, stub, openai, anthropic, "
            f"gemini, xai, deepseek, kimi, glm, mimo"
        )

    @staticmethod
    def is_cloud(provider_name: str) -> bool:
        """Return True if the provider is a cloud API (costs money)."""
        return provider_name in _OPENAI_COMPAT_PROVIDERS | {"anthropic"}

    @staticmethod
    def list_providers() -> list[str]:
        """Return all supported provider names."""
        return [
            "llama_cpp", "ollama", "stub", "openai", "anthropic", "gemini",
            "xai", "deepseek", "kimi", "glm", "mimo",
        ]
