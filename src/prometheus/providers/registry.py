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
        "model": "claude-sonnet-4-6",
        "default_env": "ANTHROPIC_API_KEY",
    },
}

# Providers that use the OpenAI-compatible wire format
_OPENAI_COMPAT_PROVIDERS = {"openai", "gemini", "xai"}


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

            api_key = _resolve_api_key(config, provider_name)
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
                model=config.get("model", defaults.get("model", "claude-sonnet-4-6")),
                timeout=config.get("timeout", 120.0),
                prompt_caching=config.get("prompt_caching", True),
            )

        if provider_name == "llama_cpp":
            from prometheus.providers.llama_cpp import LlamaCppProvider

            return LlamaCppProvider(
                base_url=config.get("base_url", "http://localhost:8080"),
                timeout=config.get("timeout", 120.0),
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
            f"Valid providers: llama_cpp, ollama, stub, openai, anthropic, gemini, xai"
        )

    @staticmethod
    def is_cloud(provider_name: str) -> bool:
        """Return True if the provider is a cloud API (costs money)."""
        return provider_name in _OPENAI_COMPAT_PROVIDERS | {"anthropic"}

    @staticmethod
    def list_providers() -> list[str]:
        """Return all supported provider names."""
        return ["llama_cpp", "ollama", "stub", "openai", "anthropic", "gemini", "xai"]
