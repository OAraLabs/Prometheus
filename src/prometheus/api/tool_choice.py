"""Per-call tool_choice — one lever, four states, mirroring the cloud API vocabulary.

A per-message `tool_choice` carried on `ApiMessageRequest` drives BOTH tiers from a single
field:

    "auto"          -> boot grammar as-is / native auto   (default agent path)
    "none"          -> grammar dropped / no tools         (chat / suppress_tools)
    "required"      -> must call SOME tool                 (force-search, any tool)
    {"tool": "X"}   -> must call tool X                    (force a specific tool)

`mode` (agent|chat) is sugar that resolves into this (agent->auto, chat->none). The
provider translates tool_choice per tier: LlamaCppProvider selects a GBNF grammar; cloud
providers set the native tool_choice param (with a synthetic tool_use prefill fallback only
where native `required` is unsupported).
"""

from __future__ import annotations

from typing import Any

AUTO = "auto"
NONE = "none"
REQUIRED = "required"

#: The scalar (non-tool-specific) directives.
_SCALARS = frozenset({AUTO, NONE, REQUIRED})

#: A normalized tool_choice is one of the scalars or ``{"tool": "<name>"}``.
ToolChoice = Any  # str | dict[str, str]


def resolve_mode_to_tool_choice(mode: str | None) -> str:
    """`mode` is sugar: agent->auto, chat->none. ONLY "chat" disables tools — anything
    else (incl. None / an unrecognized value) resolves to "auto", so an unknown mode can
    never silently drop tools (matches the Piece-2 always-agentic default)."""
    return NONE if mode == "chat" else AUTO


def forced_tool_name(tool_choice: ToolChoice) -> str | None:
    """The tool name a ``{"tool": X}`` directive forces, or None for scalar directives."""
    if isinstance(tool_choice, dict):
        return tool_choice.get("tool")
    return None


def normalize_tool_choice(raw: Any, valid_tool_names: "frozenset[str] | set[str] | list[str] | None" = None) -> ToolChoice:
    """Validate + normalize an inbound tool_choice from a WS/REST payload.

    Returns the normalized value (a scalar string or ``{"tool": name}``). ``None`` ->
    "auto" (absent is NEVER an error — same contract as absent `mode`). Raises ValueError
    for a malformed value or a forced tool that isn't registered (the caller maps that to
    a 400 / WS error frame).
    """
    if raw is None:
        return AUTO
    if isinstance(raw, str):
        if raw in _SCALARS:
            return raw
        raise ValueError(
            f"invalid tool_choice {raw!r} — expected 'auto' | 'none' | 'required' | {{'tool': <name>}}"
        )
    if isinstance(raw, dict) and set(raw.keys()) == {"tool"} and isinstance(raw["tool"], str):
        name = raw["tool"]
        if valid_tool_names is not None and name not in valid_tool_names:
            raise ValueError(f"unknown tool {name!r} — not a registered tool")
        return {"tool": name}
    raise ValueError(
        f"invalid tool_choice {raw!r} — expected 'auto' | 'none' | 'required' | {{'tool': <name>}}"
    )
