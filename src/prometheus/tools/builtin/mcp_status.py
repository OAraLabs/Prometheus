"""MCP Status Tool — check MCP server connections from within Prometheus.

Source: Prometheus (OAra AI Lab)
License: MIT
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from prometheus.mcp.runtime import McpRuntime
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class McpStatusInput(BaseModel):
    """Arguments for the mcp_status tool."""

    server: str | None = Field(
        default=None,
        description="Filter by server name (optional)",
    )


class McpStatusTool(BaseTool):
    """Show MCP server connection status and available tools."""

    name = "mcp_status"
    description = "Show MCP server connection status and available tools."
    input_model = McpStatusInput

    def __init__(self, runtime: McpRuntime) -> None:
        self._runtime = runtime

    def is_read_only(self, arguments: McpStatusInput) -> bool:
        return True

    async def execute(
        self,
        arguments: McpStatusInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        statuses = self._runtime.list_statuses()

        if arguments.server:
            statuses = [s for s in statuses if s.name == arguments.server]

        if not statuses:
            return ToolResult(output="No MCP servers configured.")

        lines = ["MCP Server Status:", ""]

        for s in statuses:
            icon = "[ok]" if s.state == "connected" else "[!!]"
            lines.append(f"{icon} {s.name} ({s.transport}): {s.state}")

            if s.state == "connected":
                lines.append(f"    Tools: {s.tool_count}")
            elif s.detail:
                lines.append(f"    Error: {s.detail[:100]}")

        # List discovered tools
        tools = self._runtime.list_tools()
        if arguments.server:
            tools = [t for t in tools if t.server_name == arguments.server]

        if tools:
            lines.append("")
            lines.append(f"Available MCP Tools ({len(tools)}):")
            for t in tools:
                lines.append(f"  mcp__{t.safe_server_name}__{t.tool_name}")
                if t.description:
                    lines.append(f"    {t.description[:80]}")

        return ToolResult(output="\n".join(lines))
