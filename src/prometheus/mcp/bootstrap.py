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


# The shipped advertise-or-defer default for MCP tools (#369). The v3 spec
# and #315 demanded that the decision be EXPLICIT; neither recorded a reason
# for deferring, and the config comment for deferral in general is about
# prompt size on small windows. Measured on the live box 2026-09-01: a
# deferred MCP tool was loaded by tool_search and the local model still
# declined to call it — the web_search precedent (zero calls from the day
# deferral activated) in a new form. A configured server IS the operator's
# advertise decision, so its tools are advertised unless the operator says
# otherwise with `tools.deferred_loading.mcp_always_deferred: true`.
MCP_ALWAYS_DEFERRED_DEFAULT = False


def sync_mcp_advertisement(
    config: dict[str, Any],
    tool_loader: Any,
    runtime: Any,
    *,
    when: str = "registry change",
) -> tuple[bool, list[str]]:
    """Make the loader's advertised set follow the runtime's registered
    MCP tools, per ``tools.deferred_loading.mcp_always_deferred``.

    The one place the decision is applied: at boot, and again whenever
    the REST surface registers or unregisters a server's tools between
    runs (the daemon's on_tools_changed hook, #370). Returns
    ``(advertised, names)`` and logs which branch ran — either way, so
    "are my MCP tools in the prompt?" is answerable from the log.
    """
    dl_cfg = (config.get("tools") or {}).get("deferred_loading") or {}
    registered = getattr(runtime, "registered_tool_names", {}) or {}
    names = sorted({n for ns in registered.values() for n in ns})
    deferred = bool(dl_cfg.get("mcp_always_deferred", MCP_ALWAYS_DEFERRED_DEFAULT))
    if tool_loader is None:
        return False, names
    if deferred:
        tool_loader.sync_dynamic_always_loaded(set())
        logger.info(
            "MCP: %d tool(s) DEFERRED at %s "
            "(tools.deferred_loading.mcp_always_deferred: true) — reachable "
            "via tool_search and exact name, not in the advertised catalog",
            len(names), when,
        )
        return False, names
    tool_loader.sync_dynamic_always_loaded(set(names))
    logger.info(
        "MCP: %d tool(s) ADVERTISED at %s "
        "(tools.deferred_loading.mcp_always_deferred: false)",
        len(names), when,
    )
    return True, names


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

        if tool_loader is not None and mcp_names:
            sync_mcp_advertisement(config, tool_loader, runtime, when="boot")

        # tools.deferred_loading.search_mcp: whether tool_search offers MCP
        # tools in its results. Same post-hoc injection pattern the daemon
        # uses for the SkillRegistry — create_tool_registry never sees the
        # tools config, so the reader lives here.
        dl_cfg = (config.get("tools") or {}).get("deferred_loading") or {}
        search_tool = registry.get("tool_search")
        if search_tool is not None and hasattr(search_tool, "include_mcp"):
            search_tool.include_mcp = bool(dl_cfg.get("search_mcp", True))

        return runtime
    except Exception as exc:
        logger.warning("MCP runtime not available: %s", exc)
        return None
