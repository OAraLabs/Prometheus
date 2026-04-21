"""Sticker cache — JSON file cache mapping Telegram sticker IDs to text descriptions.

Donor pattern: NousResearch/hermes-agent gateway/sticker_cache.py.
Adapted for Prometheus: cache at ~/.prometheus/cache/stickers/cache.json.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from prometheus.config.paths import get_config_dir

logger = logging.getLogger(__name__)

# Prompt used when asking the vision model to describe a sticker
STICKER_VISION_PROMPT = (
    "Describe this sticker in 1-2 sentences. Focus on what it depicts "
    "-- character, action, emotion. Be concise and objective."
)


def _cache_path() -> Path:
    d = get_config_dir() / "cache" / "stickers"
    d.mkdir(parents=True, exist_ok=True)
    return d / "cache.json"


def _load_cache() -> dict:
    p = _cache_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(data: dict) -> None:
    _cache_path().write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_cached_description(file_unique_id: str) -> dict | None:
    """Return cached sticker description dict, or None on cache miss.

    Returns: {"description": str, "emoji": str, "set_name": str, "cached_at": float}
    """
    cache = _load_cache()
    return cache.get(file_unique_id)


def cache_sticker_description(
    file_unique_id: str,
    description: str,
    emoji: str = "",
    set_name: str = "",
) -> None:
    """Store a sticker description in the cache."""
    cache = _load_cache()
    cache[file_unique_id] = {
        "description": description,
        "emoji": emoji,
        "set_name": set_name,
        "cached_at": time.time(),
    }
    _save_cache(cache)
    logger.debug("Cached sticker %s: %s", file_unique_id, description[:60])


def build_sticker_injection(description: str, emoji: str = "", set_name: str = "") -> str:
    """Build the text injection for a described sticker."""
    parts = []
    if emoji:
        parts.append(emoji)
    if set_name:
        parts.append(f'from "{set_name}"')
    prefix = " ".join(parts)
    if prefix:
        return f'[The user sent a sticker {prefix}. It shows: "{description}"]'
    return f'[The user sent a sticker. It shows: "{description}"]'


def build_animated_sticker_injection(emoji: str = "") -> str:
    """Fallback injection for animated/video stickers that can't be analyzed."""
    if emoji:
        return f"[The user sent an animated sticker: {emoji}]"
    return "[The user sent an animated sticker]"
