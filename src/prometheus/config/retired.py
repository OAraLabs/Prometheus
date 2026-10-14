"""Config keys that were retired: present in old configs, read by nothing.

A key with no reader is a config that documents fiction — an operator tunes it
and nothing changes. When one is retired its replacement is named here, and
the daemon says so once at startup instead of ignoring it in silence.
"""

from __future__ import annotations

from typing import Any

#: dotted key -> what to use instead.
RETIRED_KEYS: dict[str, str] = {
    # #773 §5: only ContextCompressor ever read it, and that class is not wired
    # into the loop (context/compression.py, audit H1). Compaction fires on
    # compaction.threshold_pct (or a per-model compaction_threshold_*).
    "context.compression_trigger": "compaction.threshold_pct",
}


def _present(config: dict[str, Any], dotted: str) -> bool:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def retired_key_warnings(config: dict[str, Any] | None) -> list[str]:
    """One sentence per retired key found in *config*."""
    cfg = config or {}
    return [
        f"config key {key} is retired and does nothing; use {instead} instead"
        for key, instead in RETIRED_KEYS.items()
        if _present(cfg, key)
    ]
