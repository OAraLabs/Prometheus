"""``mcp_status`` names tools the way the registry holds them.

The tool printed ``mcp__{server}__{tool_name}`` by concatenation.
Registration sanitises (``resolve-library-id`` → ``resolve_library_id``)
and suffixes on collision, so for the one server on the reference
deployment the agent's own status tool told it to call
``mcp__context7__resolve-library-id`` — a name nothing accepts. The REST
card had the same bug and #372 fixed it there (``registered_as``); this
file was missed. Both now read one property, ``McpCatalogTool.registry_name``.

The assertion is that every name the tool prints resolves in the registry
it describes — not a string pin on today's wording.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mcp")

from prometheus.mcp.adapter import register_mcp_tools  # noqa: E402
from prometheus.mcp.runtime import McpRuntime  # noqa: E402
from prometheus.mcp.types import McpCatalogTool, McpConnectionStatus  # noqa: E402
from prometheus.tools.base import ToolExecutionContext, ToolRegistry  # noqa: E402
from prometheus.tools.builtin.mcp_status import McpStatusTool  # noqa: E402


def _tool(server: str, name: str) -> McpCatalogTool:
    return McpCatalogTool(
        server_name=server, safe_server_name=server, tool_name=name,
        description=f"{name} desc",
        input_schema={"type": "object", "properties": {}},
    )


def _runtime(tools: list[McpCatalogTool]) -> MagicMock:
    runtime = MagicMock(spec=McpRuntime)
    runtime.list_tools.return_value = tools
    runtime.list_statuses.return_value = [
        McpConnectionStatus(name=server, state="connected", transport="stdio",
                            tool_count=sum(t.server_name == server for t in tools))
        for server in sorted({t.server_name for t in tools})
    ]
    return runtime


def _status_output(runtime: MagicMock) -> str:
    tool = McpStatusTool(runtime)
    result = asyncio.run(tool.execute(
        tool.input_model(), ToolExecutionContext(cwd=Path(".")),
    ))
    return result.output


def _printed_names(out: str) -> list[str]:
    return re.findall(r"^\s+(mcp__\S+)$", out, re.M)


def test_every_name_the_status_tool_prints_resolves_in_the_registry():
    runtime = _runtime([
        _tool("context7", "resolve-library-id"), _tool("context7", "query-docs"),
    ])
    registry = ToolRegistry()
    register_mcp_tools(registry, runtime)                 # sets registered_as

    out = _status_output(runtime)
    printed = _printed_names(out)
    assert len(printed) == 2, out
    unresolved = [n for n in printed if registry.get(n) is None]
    assert not unresolved, (
        f"the status tool names tools nothing accepts: {unresolved}\n{out}"
    )
    assert "mcp__context7__resolve-library-id" not in out   # the old line


def test_a_collision_suffix_is_reported_as_registered():
    runtime = _runtime([_tool("srv", "bash")])
    registry = ToolRegistry()
    taken = MagicMock()
    taken.name = "mcp__srv__bash"
    registry.register(taken)
    register_mcp_tools(registry, runtime)                 # → mcp__srv__bash_1

    out = _status_output(runtime)
    assert _printed_names(out) == ["mcp__srv__bash_1"], out
    assert registry.get("mcp__srv__bash_1") is not None


def test_an_unregistered_tool_reports_the_sanitised_base_name():
    # A disabled server's tools are discovered but not registered:
    # registered_as is None, and the sanitised base is the honest answer —
    # what registration WOULD produce absent a collision. Never the raw
    # concatenation.
    t = _tool("srv", "look-up")
    assert t.registered_as is None
    assert t.registry_name == "mcp__srv__look_up"
    assert _printed_names(_status_output(_runtime([t]))) == ["mcp__srv__look_up"]
