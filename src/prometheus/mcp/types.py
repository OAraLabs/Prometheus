"""MCP configuration and catalog types.

Donor: OpenClaw src/agents/pi-bundle-mcp-types.ts + mcp-transport-config.ts
License: MIT (OpenClaw Foundation)
Ported to Python with Pydantic models.

Source: Prometheus (OAra Labs)
License: MIT
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from prometheus.mcp.names import TOOL_NAME_SEPARATOR, sanitize_tool_name


# ---------------------------------------------------------------------------
# Catalog types (from pi-bundle-mcp-types.ts)
# ---------------------------------------------------------------------------


@dataclass
class McpServerCatalog:
    """Catalog entry for one MCP server."""

    server_name: str
    launch_summary: str
    tool_count: int
    # FOUNDATION 2.3a: how many of the offered tools survived the server's
    # allowed_tools allowlist. Equal to tool_count when no allowlist is
    # configured. tool_count stays the OFFERED count — the catalog is the
    # honest record of what the server advertises, not of what we accepted.
    allowed_count: int | None = None


@dataclass
class McpCatalogTool:
    """Tool discovered from an MCP server."""

    server_name: str
    safe_server_name: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]
    # The server's ToolAnnotations.readOnlyHint, when it declares one.
    # None = undeclared. Consumed by McpToolAdapter.is_read_only(), where
    # the trust decision (a hint from a third-party server, honoured for
    # confirmation-skipping inside an operator-allowlisted server) is
    # documented.
    read_only_hint: bool | None = None
    # The name the tool is registered under in the ToolRegistry — set by
    # adapter._register, None until then. build_safe_tool_name sanitises
    # (`resolve-library-id` → `resolve_library_id`) and suffixes on
    # collision, so this is not derivable from tool_name by concatenation;
    # the REST card once did exactly that and reported a name nothing
    # accepted (FOUNDATION §4 acceptance run, 2026-09-01).
    registered_as: str | None = None

    @property
    def registry_name(self) -> str:
        """The name to CALL this tool by — the one surface that answers it.

        ``registered_as`` once registration has happened; before that (a
        disabled server's card, a status line rendered mid-registration)
        the sanitised base name registration WOULD produce absent a
        collision suffix. Never ``mcp__{server}__{tool_name}`` by
        concatenation: the REST card reported that until #372, and the
        ``mcp_status`` tool — the agent's own view — until 2026-09-18, so
        for the reference deployment's one server the agent was told to
        call ``mcp__context7__resolve-library-id``, which nothing accepts.
        """
        if self.registered_as:
            return self.registered_as
        return (
            f"mcp{TOOL_NAME_SEPARATOR}{self.safe_server_name}"
            f"{TOOL_NAME_SEPARATOR}{sanitize_tool_name(self.tool_name)}"
        )


@dataclass
class McpToolCatalog:
    """Full catalog of all MCP servers and tools."""

    version: int = 1
    generated_at: float = 0.0
    servers: dict[str, McpServerCatalog] = field(default_factory=dict)
    tools: list[McpCatalogTool] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Connection status
# ---------------------------------------------------------------------------


@dataclass
class McpConnectionStatus:
    """Runtime status for one MCP server."""

    name: str
    state: Literal["connected", "failed", "pending", "disabled"]
    transport: str = "unknown"
    detail: str = ""
    tool_count: int = 0


# ---------------------------------------------------------------------------
# Helpers (from pi-bundle-mcp-runtime.ts)
# ---------------------------------------------------------------------------


def create_config_fingerprint(servers: dict) -> str:
    """Create hash of server config for change detection."""
    return hashlib.sha1(
        json.dumps(servers, sort_keys=True).encode()
    ).hexdigest()
