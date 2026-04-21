"""MCP Runtime — connect to servers, discover tools, call them.

Donor: OpenClaw src/agents/pi-bundle-mcp-runtime.ts
License: MIT (Anthropic)

Source: Prometheus (OAra AI Lab)
License: MIT
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from prometheus.mcp.names import sanitize_server_name
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

logger = logging.getLogger(__name__)


class McpConnectionError(Exception):
    """MCP connection failed."""


@dataclass
class _McpSession:
    """Active connection to one MCP server."""

    server_name: str
    session: ClientSession
    transport_type: str
    _exit_stack: AsyncExitStack


# ---------------------------------------------------------------------------
# Helpers ported from OpenClaw pi-bundle-mcp-runtime.ts
# ---------------------------------------------------------------------------


async def _connect_with_timeout(session: ClientSession, timeout_ms: int) -> None:
    """Connect with timeout (from OpenClaw connectWithTimeout)."""
    try:
        await asyncio.wait_for(session.initialize(), timeout=timeout_ms / 1000)
    except asyncio.TimeoutError:
        raise McpConnectionError(
            f"Connection timed out after {timeout_ms}ms"
        ) from None


async def _list_all_tools(session: ClientSession) -> list:
    """List all tools, handling pagination (from OpenClaw listAllTools)."""
    tools: list = []
    cursor = None
    while True:
        result = await session.list_tools(cursor=cursor if cursor else None)
        tools.extend(result.tools)
        cursor = getattr(result, "nextCursor", None)
        if not cursor:
            break
    return tools


# ---------------------------------------------------------------------------
# McpRuntime
# ---------------------------------------------------------------------------


class McpRuntime:
    """MCP runtime — manages server connections and provides tool access.

    Simplified from OpenClaw SessionMcpRuntime: single runtime instance
    (not per-session) since Prometheus typically runs one agent loop.
    """

    def __init__(self, server_configs: dict[str, dict]) -> None:
        self._server_configs = server_configs
        self._config_fingerprint = create_config_fingerprint(server_configs)
        self._sessions: dict[str, _McpSession] = {}
        self._catalog: McpToolCatalog | None = None
        self._statuses: dict[str, McpConnectionStatus] = {
            name: McpConnectionStatus(name=name, state="pending")
            for name in server_configs
        }

    @property
    def config_fingerprint(self) -> str:
        return self._config_fingerprint

    async def connect_all(self) -> None:
        """Connect all configured MCP servers and discover tools."""
        if not self._server_configs:
            logger.info("MCP: no servers configured")
            return

        used_names: set[str] = set()
        servers: dict[str, McpServerCatalog] = {}
        tools: list[McpCatalogTool] = []

        for server_name, raw_config in self._server_configs.items():
            resolved = resolve_transport(server_name, raw_config)
            if not resolved:
                self._statuses[server_name] = McpConnectionStatus(
                    name=server_name,
                    state="failed",
                    detail="Could not resolve transport config",
                )
                continue

            safe_name = sanitize_server_name(server_name, used_names)

            try:
                if isinstance(resolved, ResolvedStdioTransport):
                    mcp_session = await self._connect_stdio(server_name, resolved)
                elif isinstance(resolved, ResolvedHttpTransport):
                    self._statuses[server_name] = McpConnectionStatus(
                        name=server_name,
                        state="failed",
                        transport=resolved.transport_type,
                        detail="HTTP/SSE transport not yet implemented",
                    )
                    continue
                else:
                    continue

                # Discover tools
                listed_tools = await _list_all_tools(mcp_session.session)

                for tool in listed_tools:
                    tool_name = tool.name.strip()
                    if not tool_name:
                        continue
                    tools.append(McpCatalogTool(
                        server_name=server_name,
                        safe_server_name=safe_name,
                        tool_name=tool_name,
                        description=tool.description or f"MCP tool from {server_name}",
                        input_schema=dict(tool.inputSchema) if tool.inputSchema else {
                            "type": "object", "properties": {},
                        },
                    ))

                servers[server_name] = McpServerCatalog(
                    server_name=server_name,
                    launch_summary=resolved.description,
                    tool_count=len(listed_tools),
                )

                self._sessions[server_name] = mcp_session
                self._statuses[server_name] = McpConnectionStatus(
                    name=server_name,
                    state="connected",
                    transport=resolved.kind,
                    tool_count=len(listed_tools),
                )
                logger.info(
                    "MCP connected: %s (%d tools)", server_name, len(listed_tools)
                )

            except Exception as e:
                logger.warning("MCP connection failed: %s - %s", server_name, e)
                self._statuses[server_name] = McpConnectionStatus(
                    name=server_name,
                    state="failed",
                    transport=resolved.kind,
                    detail=str(e)[:200],
                )

        self._catalog = McpToolCatalog(
            version=1,
            generated_at=time.time(),
            servers=servers,
            tools=tools,
        )

        connected = sum(1 for s in self._statuses.values() if s.state == "connected")
        logger.info("MCP: %d/%d servers connected", connected, len(self._statuses))

    async def _connect_stdio(
        self,
        server_name: str,
        config: ResolvedStdioTransport,
    ) -> _McpSession:
        """Connect to a stdio MCP server."""
        logger.info("MCP connecting (stdio): %s -> %s", server_name, config.description)

        stack = AsyncExitStack()

        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env,
                cwd=config.cwd,
            ))
        )

        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        await _connect_with_timeout(session, config.timeout_ms)

        return _McpSession(
            server_name=server_name,
            session=session,
            transport_type="stdio",
            _exit_stack=stack,
        )

    # ------------------------------------------------------------------
    # Public query interface
    # ------------------------------------------------------------------

    def get_catalog(self) -> McpToolCatalog:
        """Get the tool catalog (call connect_all first)."""
        return self._catalog or McpToolCatalog()

    def list_statuses(self) -> list[McpConnectionStatus]:
        """Get connection status for all servers."""
        return sorted(self._statuses.values(), key=lambda s: s.name)

    def list_tools(self) -> list[McpCatalogTool]:
        """Get all discovered tools across connected servers."""
        return self._catalog.tools if self._catalog else []

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Call an MCP tool and return stringified result."""
        if server_name not in self._sessions:
            raise ValueError(f"MCP server not connected: {server_name}")

        session = self._sessions[server_name]
        logger.debug("MCP call: %s/%s", server_name, tool_name)

        result = await session.session.call_tool(tool_name, arguments)

        # Stringify result (from OpenClaw pi-bundle-mcp-materialize.ts)
        parts: list[str] = []
        for item in result.content:
            if getattr(item, "type", None) == "text":
                parts.append(getattr(item, "text", ""))
            else:
                parts.append(str(item))

        if not parts:
            parts.append("(no output)")

        return "\n".join(parts).strip()

    async def close(self) -> None:
        """Close all MCP connections."""
        for session in self._sessions.values():
            try:
                await session._exit_stack.aclose()
            except Exception as e:
                logger.warning("Error closing MCP session %s: %s", session.server_name, e)

        self._sessions.clear()
        self._catalog = None
