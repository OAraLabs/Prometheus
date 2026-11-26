"""The off-by-default gate for the ACP core (experimental).

One reader for ``acp.enabled``, so every future wiring point asks the same
question and the default is false in exactly one place. Nothing imports the
ACP core's heavy modules (types, sessions) unless this returns True.
"""

from __future__ import annotations

from typing import Any, Mapping


def acp_enabled(config: Mapping[str, Any] | None) -> bool:
    """Whether ACP is on. Absent config, absent section, or a non-true value
    all mean OFF — the experimental surface never enables itself."""
    section = (config or {}).get("acp")
    if not isinstance(section, Mapping):
        return False
    return section.get("enabled", False) is True