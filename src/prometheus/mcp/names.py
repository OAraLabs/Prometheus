"""MCP name sanitization — safe tool names without collisions.

Donor: OpenClaw src/agents/pi-bundle-mcp-names.ts
License: MIT (Anthropic)

Source: Prometheus (OAra AI Lab)
License: MIT
"""

from __future__ import annotations

import re

TOOL_NAME_SEPARATOR = "__"


def sanitize_server_name(name: str, used_names: set[str]) -> str:
    """Sanitize server name for use in tool names.

    From OpenClaw pi-bundle-mcp-names.ts: sanitizeServerName
    """
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
    safe = re.sub(r"_+", "_", safe).strip("_")

    if not safe:
        safe = "mcp"

    candidate = safe
    suffix = 0
    while candidate in used_names:
        suffix += 1
        candidate = f"{safe}_{suffix}"

    used_names.add(candidate)
    return candidate


def sanitize_tool_name(name: str) -> str:
    """Sanitize a single tool name component."""
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "tool"


def build_safe_tool_name(
    server_name: str,
    tool_name: str,
    reserved_names: set[str],
) -> str:
    """Build collision-free tool name: mcp__{server}__{tool}.

    From OpenClaw pi-bundle-mcp-names.ts: buildSafeToolName
    """
    safe_tool = sanitize_tool_name(tool_name)
    base = f"mcp{TOOL_NAME_SEPARATOR}{server_name}{TOOL_NAME_SEPARATOR}{safe_tool}"

    candidate = base
    suffix = 0
    while candidate.lower() in reserved_names:
        suffix += 1
        candidate = f"{base}_{suffix}"

    reserved_names.add(candidate.lower())
    return candidate
