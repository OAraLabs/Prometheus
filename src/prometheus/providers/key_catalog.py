# Provenance: Novel code for Prometheus
# License: MIT
# MODELS KEYS UI sprint (2026-07): the ONE catalog behind /api/providers/keys.

"""Provider API-key catalog — which services take keys, and where.

This module DERIVES its entries; it never re-declares what the rest of the
codebase already knows:

- **LLM services** come from :data:`router.model_router.OVERRIDE_PRESETS`,
  resolved through :func:`resolve_slash_command_target` so a user's
  ``slash_commands.<key>`` config merges over the preset exactly like the
  ``/claude`` gateway command and ``GET /api/models`` do. Adding a preset to
  OVERRIDE_PRESETS makes it show up here automatically.
- **Media services** come from the image/video tool constants
  (``image_generate._DASHSCOPE_*`` / ``video_generate._KLING_*``) — the same
  env-var names those tools read at execute time.

The only thing declared HERE is presentation metadata: human labels and the
console URL where a user gets a key (``docs_url``). Unknown/new presets fall
back to a title-cased label and an empty docs_url rather than being dropped.

Secrecy contract: this catalog carries env-var NAMES only. Whether a var is
set (a boolean) is the caller's business (the web layer checks os.environ at
request time); values never enter this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prometheus.router.model_router import (
    OVERRIDE_PRESETS,
    resolve_slash_command_target,
)

# Media tool constants — imported, not re-typed, so a tool-side rename breaks
# loudly here instead of drifting silently.
from prometheus.tools.builtin.image_generate import (
    _DASHSCOPE_DEFAULT_KEY_ENV,
    _DASHSCOPE_DEFAULT_MODEL,
)
from prometheus.tools.builtin.video_generate import (
    _KLING_ACCESS_KEY_ENV,
    _KLING_DEFAULT_MODEL,
    _KLING_SECRET_KEY_ENV,
)


@dataclass(frozen=True)
class ProviderKeyService:
    """One keyable service as the /api/providers/keys surface sees it."""

    id: str                     # stable service id ("claude", "kling-video", …)
    label: str                  # human name for the UI card
    kind: str                   # "llm" | "image" | "video"
    env_vars: tuple[str, ...]   # env-var NAMES this service reads (1..n)
    docs_url: str               # console URL where a user creates a key
    default_model: str          # what the service runs by default


# Presentation metadata for the LLM presets. Labels match the /api/models
# catalog labels (single source — server.py imports PRESET_LABELS from here).
PRESET_LABELS: dict[str, str] = {
    "claude": "Claude",
    "gpt": "GPT",
    "gemini": "Gemini",
    "xai": "xAI",
    # CLOUD EXPANSION (2026-07)
    "deepseek": "DeepSeek",
    "kimi": "Kimi",
    "glm": "GLM",
    "mimo": "MiMo",
}

# Where a human gets a key for each preset (console/API-keys pages).
_PRESET_DOCS_URLS: dict[str, str] = {
    "claude": "https://console.anthropic.com/settings/keys",
    "gpt": "https://platform.openai.com/api-keys",
    "gemini": "https://aistudio.google.com/apikey",
    "xai": "https://console.x.ai/",
    "deepseek": "https://platform.deepseek.com/api_keys",
    "kimi": "https://platform.kimi.ai/",
    "glm": "https://z.ai/",
    "mimo": "https://platform.xiaomimimo.com/",
}


def provider_key_services(
    config: dict[str, Any] | None = None,
) -> list[ProviderKeyService]:
    """Build the full catalog: every OVERRIDE_PRESETS entry + the media tools.

    Order is UI order: chat models first (preset declaration order), then
    image, then video.
    """
    services: list[ProviderKeyService] = []

    for key in OVERRIDE_PRESETS:
        # Same resolution path as /claude and GET /api/models: user
        # slash_commands.<key> config merged over the built-in preset.
        preset = resolve_slash_command_target(key, config) or OVERRIDE_PRESETS[key]
        env_var = preset.get("api_key_env", "")
        if not env_var:
            continue  # a keyless preset has nothing to configure here
        services.append(
            ProviderKeyService(
                id=key,
                label=PRESET_LABELS.get(key, key.title()),
                kind="llm",
                env_vars=(env_var,),
                docs_url=_PRESET_DOCS_URLS.get(key, ""),
                default_model=preset.get("model", "unknown"),
            )
        )

    services.append(
        ProviderKeyService(
            id="wan-image",
            label="WAN Image",
            kind="image",
            env_vars=(_DASHSCOPE_DEFAULT_KEY_ENV,),
            docs_url="https://modelstudio.console.alibabacloud.com/",
            default_model=_DASHSCOPE_DEFAULT_MODEL,
        )
    )
    services.append(
        ProviderKeyService(
            id="kling-video",
            label="Kling Video",
            kind="video",
            # The one two-var service: AccessKey + SecretKey pair.
            env_vars=(_KLING_ACCESS_KEY_ENV, _KLING_SECRET_KEY_ENV),
            docs_url="https://app.klingai.com/global/dev",
            default_model=_KLING_DEFAULT_MODEL,
        )
    )
    return services
