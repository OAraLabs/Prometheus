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
        "default_base_url_env": "OPENAI_BASE_URL",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.5-flash",
        "default_env": "GEMINI_API_KEY",
        "default_base_url_env": "GEMINI_BASE_URL",
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        # Explicit grok-4.5 — aliases (grok-3/grok-4/-latest) serve grok-4.3
        # on the OAuth surface (probed 2026-07-10); see OVERRIDE_PRESETS.
        "model": "grok-4.5",
        "default_env": "XAI_API_KEY",
        "default_base_url_env": "XAI_BASE_URL",
    },
    "anthropic": {
        "model": "claude-haiku-4-5-20251001",
        "default_env": "ANTHROPIC_API_KEY",
        "default_base_url_env": "ANTHROPIC_BASE_URL",
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
        "default_base_url_env": "DEEPSEEK_BASE_URL",
    },
    "kimi": {
        # Moonshot AI international endpoint. A separate CN endpoint
        # (https://api.moonshot.cn/v1) exists with SEPARATE keys — point
        # base_url there in config if your key is CN-issued.
        "base_url": "https://api.moonshot.ai/v1",
        "model": "kimi-k2.6",
        "default_env": "MOONSHOT_API_KEY",
        "default_base_url_env": "MOONSHOT_BASE_URL",
    },
    "glm": {
        # Z.ai (Zhipu) — note the nonstandard /api/paas/v4 path prefix; the
        # OpenAI-compat provider appends /chat/completions to version-suffixed
        # bases (see openai_compat._chat_completions_url). CN mainland
        # endpoint (https://open.bigmodel.cn/api/paas/v4) has the same shape.
        "base_url": "https://api.z.ai/api/paas/v4",
        "model": "glm-5.2",
        "default_env": "ZAI_API_KEY",
        "default_base_url_env": "ZAI_BASE_URL",
    },
    "mimo": {
        # Xiaomi MiMo first-party hosted platform.
        "base_url": "https://api.xiaomimimo.com/v1",
        "model": "mimo-v2.5-pro",
        "default_env": "MIMO_API_KEY",
        "default_base_url_env": "MIMO_BASE_URL",
    },
    "qwen": {
        # Alibaba Cloud Model Studio, OpenAI-compatible surface.
        #
        # This default is the INTERNATIONAL PAY-AS-YOU-GO endpoint: it takes an
        # ordinary `sk-` Model Studio key and carries no usage-scope
        # restriction. It is the only Alibaba surface a general-purpose daemon
        # can point at without reading their acceptable-use terms first.
        #
        # The subscription plans are SEPARATE hosts with SEPARATE keys, and the
        # pairs are not interchangeable (Alibaba: the keys and base URLs "are
        # completely isolated and must be used in matching pairs"). Both also
        # restrict usage to interactive programming tools / agents — NOT
        # automated scripts, application backends, or non-interactive calls.
        # This default is also the WRONG endpoint for a WORKSPACE-scoped key: a
        # key issued against a workspace 403s here and must use that workspace's
        # own host (https://ws-<id>.<region>.maas.aliyuncs.com/compatible-mode/v1).
        # That hostname identifies the account, so set QWEN_BASE_URL in the env
        # file rather than base_url in yaml — see _resolve_base_url below.
        #
        # Other hosts, all reachable the same way:
        #   Token Plan (Team):
        #     https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
        #   Coding Plan:
        #     https://coding-intl.dashscope.aliyuncs.com/v1
        #   US region: https://dashscope-us.aliyuncs.com/compatible-mode/v1
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        # Kept in step with OVERRIDE_PRESETS["qwen"]["model"] — this one is the
        # fallback for `model.provider: qwen` as the PRIMARY, where no preset
        # supplies a model. Cheaper tiers pin via slash_commands.qwen.model.
        "model": "qwen3.8-max",
        # Deliberately NOT DASHSCOPE_API_KEY: that var is the WAN 2.5 image
        # backend's, and it addresses a different host (…/api/v1). Sharing one
        # var would mean a text key that breaks image generation, or vice versa.
        "default_env": "QWEN_API_KEY",
        "default_base_url_env": "QWEN_BASE_URL",
    },
}

# Providers that use the OpenAI-compatible wire format
_OPENAI_COMPAT_PROVIDERS = {
    "openai", "gemini", "xai", "deepseek", "kimi", "glm", "mimo", "qwen",
}


def _resolve_base_url(config: dict[str, Any], provider_name: str) -> str:
    """Resolve the API base URL from config, environment, or the built-in default.

    Mirrors :func:`_resolve_api_key`'s shape on purpose, because a base URL can
    be as account-identifying as a key. Alibaba Model Studio is the motivating
    case: a workspace-scoped endpoint embeds the workspace id in its hostname
    (``https://ws-<id>.<region>.maas.aliyuncs.com/compatible-mode/v1``), and a
    key issued for that workspace 403s against the shared regional host. That
    URL belongs in the 0600 env file next to the key, not in a yaml file that
    gets copied between checkouts and pasted into issues.

    Checks, in order:
      1. ``config["base_url"]`` — explicit, wins outright.
      2. ``config["base_url_env"]``, else the provider's
         ``default_base_url_env`` — read from the environment if set.
      3. ``CLOUD_DEFAULTS[provider]["base_url"]`` — the shared public endpoint.

    Unlike the key resolver this never raises: an unset variable simply falls
    through to the default, so adding a ``*_BASE_URL`` name costs nothing until
    someone actually exports it.
    """
    direct = config.get("base_url", "")
    if direct:
        return str(direct)

    defaults = CLOUD_DEFAULTS.get(provider_name, {})
    env_name = config.get("base_url_env", "") or defaults.get("default_base_url_env", "")
    if env_name:
        from_env = os.environ.get(env_name, "").strip()
        if from_env:
            log.info(
                "Provider %s: base URL from %s (overriding the default endpoint)",
                provider_name, env_name,
            )
            return from_env

    return str(defaults.get("base_url", ""))


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
                base_url=_resolve_base_url(config, provider_name),
                api_key=api_key,
                model=config.get("model", defaults.get("model", "")),
                default_max_tokens=config.get("max_tokens", 4096),
                timeout=config.get("timeout", 120.0),
            )

        if provider_name == "anthropic":
            from prometheus.providers.anthropic import AnthropicProvider

            api_key = _resolve_api_key(config, provider_name)
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "model": config.get(
                    "model", defaults.get("model", "claude-haiku-4-5-20251001")
                ),
                "timeout": config.get("timeout", 120.0),
                "prompt_caching": config.get("prompt_caching", True),
            }
            # base_url was previously dropped here entirely — AnthropicProvider
            # has always accepted it, but the factory never passed it, so an
            # Anthropic-compatible third-party endpoint was unreachable by
            # config. Only forwarded when actually set, so the default path
            # keeps AnthropicProvider's own constant.
            #
            # Caveat for third-party shims: the provider appends "/messages" to
            # whatever it is given unless the value already ends in it, so a
            # base of ".../apps/anthropic" becomes ".../apps/anthropic/messages"
            # — pass ".../apps/anthropic/v1" if the shim expects /v1/messages.
            # Headers stay Anthropic's (x-api-key, anthropic-version, and the
            # prompt-caching beta when enabled); a shim that rejects those needs
            # provider-side work, not just this passthrough.
            anthropic_base = _resolve_base_url(config, provider_name)
            if anthropic_base:
                kwargs["base_url"] = anthropic_base
            return AnthropicProvider(**kwargs)

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
            f"gemini, xai, deepseek, kimi, glm, mimo, qwen"
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
            "xai", "deepseek", "kimi", "glm", "mimo", "qwen",
        ]
