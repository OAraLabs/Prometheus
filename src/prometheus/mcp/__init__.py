"""mcp — MCP (Model Context Protocol) integration for Prometheus.

Sprint 12: connects to MCP tool servers (e.g. Context7) and exposes
their tools as native Prometheus BaseTool instances in the ToolRegistry.

The re-exports below are LAZY (PEP 562), and that is load-bearing. The
``mcp`` SDK is an optional extra, and only ``runtime`` (and ``adapter``,
through it) imports it. But ``import prometheus.mcp.store`` runs THIS file
first, and until 0.9.2 this file imported ``runtime`` eagerly — so every
SDK-free caller got the SDK anyway. The daemon's boot reads the server store
before deciding whether MCP is configured at all (the "lazy + guarded"
import in daemon.py), so an install without the ``mcp`` extra could not
start ``oara daemon``, configured or not, and the REST routes behind
Beacon's Connectors tab failed the same way. tests/test_web_is_base.py
imports the store with the SDK made unimportable.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# public name -> the submodule that defines it
_EXPORTS = {
    "McpToolAdapter": "adapter",
    "register_mcp_tools": "adapter",
    "build_safe_tool_name": "names",
    "sanitize_server_name": "names",
    "McpConnectionError": "runtime",
    "McpRuntime": "runtime",
    "ResolvedHttpTransport": "transport",
    "ResolvedStdioTransport": "transport",
    "resolve_transport": "transport",
    "McpCatalogTool": "types",
    "McpConnectionStatus": "types",
    "McpServerCatalog": "types",
    "McpToolCatalog": "types",
    "create_config_fingerprint": "types",
}

__all__ = [
    "McpCatalogTool",
    "McpConnectionError",
    "McpConnectionStatus",
    "McpRuntime",
    "McpServerCatalog",
    "McpToolAdapter",
    "McpToolCatalog",
    "ResolvedHttpTransport",
    "ResolvedStdioTransport",
    "build_safe_tool_name",
    "create_config_fingerprint",
    "register_mcp_tools",
    "resolve_transport",
    "sanitize_server_name",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)


if TYPE_CHECKING:
    from prometheus.mcp.adapter import McpToolAdapter, register_mcp_tools
    from prometheus.mcp.names import build_safe_tool_name, sanitize_server_name
    from prometheus.mcp.runtime import McpConnectionError, McpRuntime
    from prometheus.mcp.transport import (
        ResolvedHttpTransport,
        ResolvedStdioTransport,
        resolve_transport,
    )
    from prometheus.mcp.types import (
        McpCatalogTool,
        McpConnectionStatus,
        McpServerCatalog,
        McpToolCatalog,
        create_config_fingerprint,
    )
