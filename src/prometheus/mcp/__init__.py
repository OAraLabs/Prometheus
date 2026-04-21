"""mcp — MCP (Model Context Protocol) integration for Prometheus.

Sprint 12: connects to MCP tool servers (e.g. Context7) and exposes
their tools as native Prometheus BaseTool instances in the ToolRegistry.
"""

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
