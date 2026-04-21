"""MCP configuration and catalog types.

Donor: OpenClaw src/agents/pi-bundle-mcp-types.ts + mcp-transport-config.ts
License: MIT (Anthropic)
Ported to Python with Pydantic models.

Source: Prometheus (OAra AI Lab)
License: MIT
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Catalog types (from pi-bundle-mcp-types.ts)
# ---------------------------------------------------------------------------


@dataclass
class McpServerCatalog:
    """Catalog entry for one MCP server."""

    server_name: str
    launch_summary: str
    tool_count: int


@dataclass
class McpCatalogTool:
    """Tool discovered from an MCP server."""

    server_name: str
    safe_server_name: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]


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
