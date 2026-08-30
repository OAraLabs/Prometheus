"""MCP Runtime — connect to servers, discover tools, call them.

Donor: OpenClaw src/agents/pi-bundle-mcp-runtime.ts
License: MIT (OpenClaw Foundation)

Source: Prometheus (OAra Labs)
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
        # FOUNDATION 2.3a: per-server allowlist. None = no allowlist (every
        # offered tool loads). Parsed once here so call_tool() can enforce
        # the same set the discovery filter applied — a catalog-only filter
        # is bypassable by the registry's "lucky guess" execution path.
        self._allowed_tools: dict[str, set[str] | None] = {
            name: self._parse_allowed(raw)
            for name, raw in server_configs.items()
        }
        # #332 (REST lifecycle): safe-name dedup must survive individual
        # connects/disconnects, so the set lives on the runtime instead of
        # being minted per connect_all pass.
        self._used_safe_names: set[str] = set()
        self._safe_names: dict[str, str] = {}

    @staticmethod
    def _parse_allowed(raw: Any) -> set[str] | None:
        if isinstance(raw, dict) and isinstance(
            raw.get("allowed_tools"), (list, tuple, set)
        ):
            return {str(t) for t in raw["allowed_tools"]}
        return None

    @property
    def config_fingerprint(self) -> str:
        return self._config_fingerprint

    async def connect_all(self) -> None:
        """Connect all configured MCP servers and discover tools."""
        if not self._server_configs:
            logger.info("MCP: no servers configured")
            return

        self._catalog = self._catalog or McpToolCatalog(
            version=1, generated_at=time.time(),
        )
        for server_name, raw_config in list(self._server_configs.items()):
            await self.connect_server(server_name, raw_config)

        connected = sum(1 for s in self._statuses.values() if s.state == "connected")
        logger.info("MCP: %d/%d servers connected", connected, len(self._statuses))

    async def connect_server(
        self, server_name: str, raw_config: dict
    ) -> McpServerCatalog | None:
        """Connect ONE server and add its tools to the catalog (#332).

        Extracted from connect_all so the REST lifecycle can add/reconnect a
        server on a running daemon. Idempotent-ish: an already-connected
        server is disconnected first, so a reconnect re-discovers with the
        current allowed_tools. Returns the server's catalog entry, or None
        on failure (the failure is recorded in the status map — same
        fail-soft contract connect_all always had).
        """
        if server_name in self._sessions:
            await self.disconnect_server(server_name)
        self._server_configs[server_name] = raw_config
        self._allowed_tools[server_name] = self._parse_allowed(raw_config)
        if self._catalog is None:
            self._catalog = McpToolCatalog(version=1, generated_at=time.time())

        resolved = resolve_transport(server_name, raw_config)
        if not resolved:
            self._statuses[server_name] = McpConnectionStatus(
                name=server_name,
                state="failed",
                detail="Could not resolve transport config",
            )
            return None

        if server_name in self._safe_names:
            safe_name = self._safe_names[server_name]
        else:
            safe_name = sanitize_server_name(server_name, self._used_safe_names)
            self._safe_names[server_name] = safe_name

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
                return None
            else:
                return None

            # Discover tools
            listed_tools = await _list_all_tools(mcp_session.session)

            allowed = self._allowed_tools.get(server_name)
            excluded = 0
            for tool in listed_tools:
                tool_name = tool.name.strip()
                if not tool_name:
                    continue
                if allowed is not None and tool_name not in allowed:
                    excluded += 1
                    continue
                annotations = getattr(tool, "annotations", None)
                self._catalog.tools.append(McpCatalogTool(
                    server_name=server_name,
                    safe_server_name=safe_name,
                    tool_name=tool_name,
                    description=tool.description or f"MCP tool from {server_name}",
                    input_schema=dict(tool.inputSchema) if tool.inputSchema else {
                        "type": "object", "properties": {},
                    },
                    read_only_hint=getattr(
                        annotations, "readOnlyHint", None
                    ),
                ))
            if excluded:
                # No silent caps: a filtered offering must be visible in
                # the boot log, or "covered everything" is inferred.
                logger.info(
                    "MCP %s: allowed_tools excluded %d of %d offered "
                    "tool(s)", server_name, excluded, len(listed_tools),
                )

            entry = McpServerCatalog(
                server_name=server_name,
                launch_summary=resolved.description,
                tool_count=len(listed_tools),
                allowed_count=len(listed_tools) - excluded,
            )
            self._catalog.servers[server_name] = entry

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
            return entry

        except Exception as e:
            logger.warning("MCP connection failed: %s - %s", server_name, e)
            self._statuses[server_name] = McpConnectionStatus(
                name=server_name,
                state="failed",
                transport=resolved.kind,
                detail=str(e)[:200],
            )
            return None

    async def disconnect_server(
        self, server_name: str, *, forget: bool = False
    ) -> None:
        """Close one server's session and drop its tools from the catalog.

        ``forget=True`` (DELETE) also drops the config/status/allowlist
        entries; the default (disable/reconnect) keeps them, with the
        status marked ``disabled`` so a card can say so honestly.
        """
        session = self._sessions.pop(server_name, None)
        if session is not None:
            try:
                await session._exit_stack.aclose()
            except Exception as e:  # noqa: BLE001 — teardown is best-effort
                logger.warning(
                    "Error closing MCP session %s: %s", server_name, e
                )
        if self._catalog is not None:
            self._catalog.tools = [
                t for t in self._catalog.tools
                if t.server_name != server_name
            ]
            self._catalog.servers.pop(server_name, None)
        if forget:
            self._server_configs.pop(server_name, None)
            self._allowed_tools.pop(server_name, None)
            self._statuses.pop(server_name, None)
        else:
            self._statuses[server_name] = McpConnectionStatus(
                name=server_name, state="disabled",
            )

    async def probe(self, server_name: str, timeout_s: float = 3.0) -> bool:
        """A PROBED health answer, not a config echo (#332's amber-lock
        discipline): ping the live session; a dead subprocess reads as
        unhealthy on the card, never as an empty success. Downgrades the
        status on failure so the next GET agrees with reality."""
        session = self._sessions.get(server_name)
        if session is None:
            return False
        try:
            await asyncio.wait_for(
                session.session.send_ping(), timeout=timeout_s
            )
            return True
        except Exception as e:  # noqa: BLE001 — any failure = unhealthy
            self._statuses[server_name] = McpConnectionStatus(
                name=server_name,
                state="failed",
                transport=session.transport_type,
                detail=f"probe failed: {str(e)[:150]}",
            )
            return False

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

        # FOUNDATION 2.3a: the allowlist is enforced at the CALL, not only
        # at discovery. Discovery filtering keeps excluded tools out of the
        # catalog and registry, but the registry executes lucky-guess names
        # and adapters outlive config edits — the call seam is the one
        # every path funnels through.
        allowed = self._allowed_tools.get(server_name)
        if allowed is not None and tool_name not in allowed:
            raise PermissionError(
                f"MCP tool {tool_name!r} is not in allowed_tools for "
                f"server {server_name!r}"
            )

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
