"""MCP Tool Adapter — wrap MCP tools as Prometheus BaseTool instances.

Donor: OpenClaw src/agents/pi-bundle-mcp-materialize.ts
       + OpenHarness src/openharness/tools/mcp_tool.py
License: MIT (OpenClaw Foundation; OpenHarness Contributors)

Source: Prometheus (OAra Labs)
License: MIT
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict

from prometheus.mcp.names import build_safe_tool_name
from prometheus.mcp.runtime import McpRuntime
from prometheus.mcp.types import McpCatalogTool
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)


class _McpDynamicInput(BaseModel):
    """Flexible input model that accepts any MCP tool arguments."""

    model_config = ConfigDict(extra="allow")


class McpToolAdapter(BaseTool):
    """Wrap an MCP tool as a native Prometheus BaseTool.

    Tool names follow the pattern ``mcp__{server}__{tool}``.
    The MCP-provided JSON schema is used for API/OpenAI schema output
    instead of pydantic introspection, so models see the real parameter
    definitions from the MCP server.
    """

    input_model = _McpDynamicInput

    def __init__(
        self,
        runtime: McpRuntime,
        tool_info: McpCatalogTool,
        safe_name: str,
    ) -> None:
        self._runtime = runtime
        self._tool_info = tool_info

        self.name = safe_name
        self.description = tool_info.description

    def is_read_only(self, arguments: BaseModel) -> bool:
        # FOUNDATION 2.3a: honest, not hopeful. The old hardcoded True made
        # every MCP call fall through the SecurityGate as auto-allow — a
        # third-party tool that writes files or sends money was "read-only"
        # by assertion. Now: only an explicit readOnlyHint=True from the
        # server counts; undeclared means NOT read-only, and the gate
        # requires confirmation for it (checker.evaluate's mcp__ rule).
        #
        # Trust decision, stated: the hint is the server's own claim. It is
        # honoured for confirmation-skipping because the operator already
        # opted into the server (and can scope it with allowed_tools) —
        # a server that lies about read-onlyness has the same power as any
        # tool the operator allowlisted. The floor (denied paths, blocked
        # commands) never keyed off this flag.
        return self._tool_info.read_only_hint is True

    def to_api_schema(self) -> dict[str, Any]:
        """Return the MCP-provided schema (not pydantic introspection)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._tool_info.input_schema,
        }

    def to_openai_schema(self) -> dict[str, Any]:
        """Return the MCP-provided schema in OpenAI format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._tool_info.input_schema,
            },
        }

    async def execute(
        self,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Execute the MCP tool via the runtime."""
        # Extract all fields (including extra) as a plain dict
        kwargs = arguments.model_dump()

        try:
            result = await self._runtime.call_tool(
                self._tool_info.server_name,
                self._tool_info.tool_name,
                kwargs,
            )
            return ToolResult(output=result)
        except Exception as e:
            logger.error("MCP tool error: %s - %s", self.name, e)
            return ToolResult(output=str(e), is_error=True)


def register_mcp_tools(registry: Any, runtime: McpRuntime) -> int:
    """Register all discovered MCP tools with a ToolRegistry.

    Returns the number of tools registered.
    """
    return len(_register(registry, runtime, runtime.list_tools()))


def register_server_tools(
    registry: Any, runtime: McpRuntime, server_name: str
) -> list[str]:
    """Register ONE server's discovered tools; returns the registered names
    (#332 — the REST lifecycle registers per server and must know the exact
    names to unregister later, including any collision suffixes)."""
    return _register(
        registry, runtime,
        [t for t in runtime.list_tools() if t.server_name == server_name],
    )


def unregister_tools(registry: Any, names: list[str]) -> int:
    """Remove previously registered MCP tool names. Returns how many were
    actually present — a caller logging this catches double-removal bugs."""
    return sum(1 for name in names if registry.unregister(name))


def _register(registry: Any, runtime: McpRuntime, tool_infos: list) -> list[str]:
    reserved_names: set[str] = {t.name.lower() for t in registry.list_tools()}
    names: list[str] = []
    for tool_info in tool_infos:
        safe_name = build_safe_tool_name(
            tool_info.safe_server_name,
            tool_info.tool_name,
            reserved_names,
        )
        adapter = McpToolAdapter(runtime, tool_info, safe_name)
        registry.register(adapter)
        tool_info.registered_as = safe_name
        logger.info("Registered MCP tool: %s", adapter.name)
        names.append(safe_name)
        # Bookkeeping for the REST lifecycle: which registry names belong
        # to which server. Lives on the runtime because the runtime is the
        # long-lived object the routes hold.
        registered = getattr(runtime, "registered_tool_names", None)
        if registered is None:
            registered = {}
            try:
                runtime.registered_tool_names = registered  # type: ignore[attr-defined]
            except Exception:
                registered = None
        if registered is not None:
            registered.setdefault(tool_info.server_name, []).append(safe_name)
    return names
