"""Configuration loading with environment variable overrides and secret file support.

Precedence (highest to lowest):
1. Environment variables (PROMETHEUS_*)
2. Secret files (*_FILE env vars)
3. Config file (prometheus.yaml)
4. Defaults

Donor patterns:
- OpenClaw src/infra/secret-file.ts: safe secret file loading with symlink rejection

Source: Prometheus (OAra AI Lab)
License: MIT
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Env var → config path mapping
ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    # Gateway tokens (most common pain point)
    "PROMETHEUS_TELEGRAM_TOKEN": ("gateway", "telegram_token"),
    "PROMETHEUS_SLACK_BOT_TOKEN": ("gateway", "slack_bot_token"),
    "PROMETHEUS_SLACK_APP_TOKEN": ("gateway", "slack_app_token"),

    # Provider API keys
    "ANTHROPIC_API_KEY": ("providers", "anthropic", "api_key"),
    "OPENAI_API_KEY": ("providers", "openai", "api_key"),

    # Provider URLs (for switching machines)
    "PROMETHEUS_LLAMA_CPP_URL": ("providers", "llama_cpp", "base_url"),
    "PROMETHEUS_OLLAMA_URL": ("providers", "ollama", "base_url"),

    # Model/provider override
    "PROMETHEUS_MODEL": ("model", "model"),
    "PROMETHEUS_PROVIDER": ("model", "provider"),

    # Trust level
    "PROMETHEUS_TRUST_LEVEL": ("security", "trust_level"),

    # Permission mode
    "PROMETHEUS_PERMISSION_MODE": ("security", "permission_mode"),
}

# Secret file env vars (alternative to inline secrets)
SECRET_FILE_VARS: dict[str, tuple[str, ...]] = {
    "PROMETHEUS_TELEGRAM_TOKEN_FILE": ("gateway", "telegram_token"),
    "PROMETHEUS_ANTHROPIC_KEY_FILE": ("providers", "anthropic", "api_key"),
    "PROMETHEUS_OPENAI_KEY_FILE": ("providers", "openai", "api_key"),
}

# Keys whose values should be coerced to int
_INT_KEYS = frozenset({"trust_level"})


def _set_nested(config: dict, path: tuple[str, ...], value: Any) -> None:
    """Set a value at a nested config path, creating intermediate dicts."""
    current = config
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    final_key = path[-1]
    if final_key in _INT_KEYS and isinstance(value, str) and value.isdigit():
        value = int(value)

    current[final_key] = value


def read_secret_file(
    file_path: str,
    label: str,
    max_bytes: int = 16384,
) -> str | None:
    """Safely read a secret from a file.

    Pattern from OpenClaw src/infra/secret-file.ts:
    - Reject symlinks (prevent path traversal)
    - Enforce size limit (prevent DoS)
    - Strip whitespace
    """
    raw_path = Path(file_path).expanduser()

    # Security: reject symlinks (check BEFORE resolving)
    if raw_path.is_symlink():
        logger.warning("%s file must not be a symlink: %s", label, file_path)
        return None

    path = raw_path.resolve()

    if not path.is_file():
        logger.warning("%s file not found: %s", label, file_path)
        return None

    # Security: size limit
    if path.stat().st_size > max_bytes:
        logger.warning("%s file exceeds %d bytes: %s", label, max_bytes, file_path)
        return None

    try:
        secret = path.read_text().strip()
        if not secret:
            logger.warning("%s file is empty: %s", label, file_path)
            return None
        return secret
    except Exception as e:
        logger.warning("Failed to read %s file: %s", label, e)
        return None


def apply_env_overrides(config: dict) -> dict:
    """Apply environment variable overrides and secret files to a config dict.

    Mutates *config* in place and returns it for convenience.

    Usage::

        # Option 1: env var directly
        PROMETHEUS_TELEGRAM_TOKEN=123:ABC python -m prometheus

        # Option 2: secret file
        PROMETHEUS_TELEGRAM_TOKEN_FILE=~/.secrets/telegram.txt python -m prometheus
    """
    # Apply secret file overrides first (lower priority than direct env vars)
    for env_var, path in SECRET_FILE_VARS.items():
        file_path = os.environ.get(env_var)
        if file_path:
            label = env_var.replace("_FILE", "").replace("PROMETHEUS_", "")
            secret = read_secret_file(file_path, label)
            if secret:
                _set_nested(config, path, secret)
                logger.info("Loaded %s from secret file", label)

    # Apply direct env var overrides (higher priority)
    applied: list[str] = []
    for env_var, path in ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value:
            _set_nested(config, path, value)
            # Mask secrets in log output
            if "token" in env_var.lower() or "key" in env_var.lower():
                masked = value[:4] + "..." + value[-4:] if len(value) > 12 else "***"
            else:
                masked = value
            applied.append(f"{env_var}={masked}")

    if applied:
        logger.info("Applied env overrides: %s", ", ".join(applied))

    return config
