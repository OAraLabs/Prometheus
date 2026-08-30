"""MCP bootstrap — one construction path for CLI and daemon.

FOUNDATION 2.3a makes MCP the sanctioned way third parties add tools, and
the survey behind spec v3 found the runtime had only ever been constructed
on the CLI path — the daemon (Telegram, Beacon, cron) had never registered
an MCP tool. This module is the single factory both entry points call, so
"MCP is wired" can never again be true for one surface and false for the
other.

Also owned here: the advertise-or-defer decision for MCP tools. Deferred
loading filters by literal name membership in a static config list, and
MCP names exist only at runtime — so without an explicit decision every
MCP tool lands silently unadvertised to local models (the ``vault_search``
failure class). ``tools.deferred_loading.mcp_always_deferred`` is that
decision, and this module is its reader; either branch logs what happened.

Source: Prometheus (OAra Labs)
License: MIT
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def create_mcp_runtime(
    config: dict[str, Any],
    registry: Any,
    tool_loader: Any | None = None,
) -> Any:
    """Create the MCP runtime, connect servers, register tools.

    Returns the runtime (caller owns ``await runtime.close()`` at
    shutdown), or None when no servers are configured or the ``mcp``
    package is unavailable. Fail-soft by design: a broken MCP config must
    not take the daemon down — the failure is logged and the boot
    continues without third-party tools, which is the pre-MCP baseline.
    """
    # #332: servers come from TWO sources — the operator's yaml map and the
    # REST-managed store (yaml wins on collision; disabled store entries are
    # structurally absent). See mcp/store.py for why the yaml is never
    # written.
    from prometheus.mcp.store import McpServerStore, merged_server_configs
    mcp_servers = merged_server_configs(config, McpServerStore())
    if not mcp_servers:
        logger.debug("MCP: no servers configured")
        return None

    try:
        from prometheus.mcp.adapter import register_mcp_tools
        from prometheus.mcp.runtime import McpRuntime
        from prometheus.tools.builtin.mcp_status import McpStatusTool

        runtime = McpRuntime(mcp_servers)
        await runtime.connect_all()

        before = {t.name for t in registry.list_tools()}
        count = register_mcp_tools(registry, runtime)
        mcp_names = sorted(
            {t.name for t in registry.list_tools()} - before
        )
        registry.register(McpStatusTool(runtime))
        logger.info("MCP: registered %d tools + mcp_status", count)

        dl_cfg = (config.get("tools") or {}).get("deferred_loading") or {}
        if tool_loader is not None and mcp_names:
            if dl_cfg.get("mcp_always_deferred", True):
                logger.info(
                    "MCP: %d tool(s) DEFERRED "
                    "(tools.deferred_loading.mcp_always_deferred: true) — "
                    "reachable via tool_search and direct call, not in the "
                    "advertised catalog",
                    len(mcp_names),
                )
            else:
                tool_loader.add_always_loaded(mcp_names)
                logger.info(
                    "MCP: %d tool(s) ADVERTISED "
                    "(tools.deferred_loading.mcp_always_deferred: false)",
                    len(mcp_names),
                )

        # tools.deferred_loading.search_mcp: whether tool_search offers MCP
        # tools in its results. Same post-hoc injection pattern the daemon
        # uses for the SkillRegistry — create_tool_registry never sees the
        # tools config, so the reader lives here.
        search_tool = registry.get("tool_search")
        if search_tool is not None and hasattr(search_tool, "include_mcp"):
            search_tool.include_mcp = bool(dl_cfg.get("search_mcp", True))

        return runtime
    except Exception as exc:
        logger.warning("MCP runtime not available: %s", exc)
        return None
