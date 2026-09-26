"""Dump the builtin tool registry's (name, description) pairs as JSON — public code only.

Runs on a DEVELOPMENT machine against a checkout, never against live state:

    PROMETHEUS_CONFIG_DIR=$(mktemp -d) PROMETHEUS_DATA_DIR=$(mktemp -d) \\
    PROMETHEUS_LOGS_DIR=$(mktemp -d) PYTHONPATH=src \\
        .venv/bin/python docs/audits/skill-usage/dump_tool_catalog.py > tools.json

``tool_search_replay.py`` ranks recorded tool_search queries against this list
plus the mini's skills, because a result's ranking depends on every tool's
name and description (tools and skills share one top-5).
"""

from __future__ import annotations

import json
import logging

logging.disable(logging.CRITICAL)


def main() -> None:
    from prometheus.__main__ import create_tool_registry
    from prometheus.tools.tool_search import ToolSearchTool

    registry = create_tool_registry({})
    names = {t.name for t in registry.list_tools()}
    if "tool_search" not in names:  # registered inside create_tool_registry today; keep robust
        registry.register(ToolSearchTool())
    out = [{"name": t.name, "description": t.description} for t in registry.list_tools()]
    # The daemon registers these three outside create_tool_registry (audit_query
    # needs a security gate with an audit log; lsp and mcp_status are wired in
    # daemon.py). Class attributes only — nothing is constructed.
    from prometheus.tools.builtin.audit_query import AuditQueryTool
    from prometheus.tools.builtin.lsp import LSPTool
    from prometheus.tools.builtin.mcp_status import McpStatusTool

    for cls in (AuditQueryTool, LSPTool, McpStatusTool):
        if cls.name not in {t["name"] for t in out}:
            out.append({"name": cls.name, "description": cls.description})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
