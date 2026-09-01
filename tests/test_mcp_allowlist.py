"""Per-server allowed_tools + readOnlyHint capture — FOUNDATION 2.3a.

The allowlist is enforced at BOTH seams: discovery (an excluded tool never
enters the catalog or registry) and call_tool (the seam every path funnels
through — a catalog-only filter is bypassable by the registry's lucky-guess
execution path). tool_count stays the OFFERED count; allowed_count says
what survived.

Follows tests/test_mcp_adapter.py's convention: importorskip("mcp") — the
SDK is an optional extra CI does not install; these run wherever the mcp
extra is present (the operator's boxes, and any `--extra mcp` env).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("mcp")

from prometheus.mcp.adapter import McpToolAdapter  # noqa: E402
from prometheus.mcp.runtime import McpRuntime, _McpSession  # noqa: E402
from prometheus.mcp.types import McpCatalogTool  # noqa: E402


def _offered(name: str, read_only: bool | None = None) -> SimpleNamespace:
    annotations = (
        SimpleNamespace(readOnlyHint=read_only) if read_only is not None else None
    )
    return SimpleNamespace(
        name=name,
        description=f"{name} desc",
        inputSchema={"type": "object", "properties": {}},
        annotations=annotations,
    )


def _connect(runtime: McpRuntime, offered: list, monkeypatch) -> None:
    """Drive the REAL connect_all discovery loop with a stubbed transport."""
    fake_session = _McpSession(
        server_name="srv",
        session=MagicMock(),
        transport_type="stdio",
        _exit_stack=MagicMock(),
    )

    async def _fake_connect_stdio(self, server_name, config):  # noqa: ANN001
        return fake_session

    async def _fake_list(session):  # noqa: ANN001
        return offered

    monkeypatch.setattr(McpRuntime, "_connect_stdio", _fake_connect_stdio)
    monkeypatch.setattr(
        "prometheus.mcp.runtime._list_all_tools", _fake_list
    )
    asyncio.run(runtime.connect_all())


class TestDiscoveryFilter:
    def test_allowlist_excludes_at_discovery(self, monkeypatch) -> None:
        runtime = McpRuntime(
            {"srv": {"command": "x", "allowed_tools": ["keep"]}}
        )
        _connect(runtime, [_offered("keep"), _offered("drop")], monkeypatch)
        names = [t.tool_name for t in runtime.list_tools()]
        assert names == ["keep"]

    def test_counts_stay_honest(self, monkeypatch) -> None:
        runtime = McpRuntime(
            {"srv": {"command": "x", "allowed_tools": ["keep"]}}
        )
        _connect(runtime, [_offered("keep"), _offered("drop")], monkeypatch)
        server = runtime.get_catalog().servers["srv"]
        assert server.tool_count == 2      # what the server OFFERS
        assert server.allowed_count == 1   # what survived the allowlist

    def test_no_allowlist_loads_everything(self, monkeypatch) -> None:
        runtime = McpRuntime({"srv": {"command": "x"}})
        _connect(runtime, [_offered("a"), _offered("b")], monkeypatch)
        assert len(runtime.list_tools()) == 2

    def test_read_only_hint_is_captured(self, monkeypatch) -> None:
        runtime = McpRuntime({"srv": {"command": "x"}})
        _connect(
            runtime,
            [_offered("ro", read_only=True), _offered("rw", read_only=False),
             _offered("undeclared")],
            monkeypatch,
        )
        hints = {t.tool_name: t.read_only_hint for t in runtime.list_tools()}
        assert hints == {"ro": True, "rw": False, "undeclared": None}


class TestCallSeam:
    def _runtime_with_session(self, config: dict) -> McpRuntime:
        runtime = McpRuntime(config)
        session = MagicMock()
        session.session.call_tool = AsyncMock(
            return_value=SimpleNamespace(
                content=[SimpleNamespace(type="text", text="ok")]
            )
        )
        runtime._sessions["srv"] = session
        return runtime

    def test_disallowed_tool_is_refused_at_call(self) -> None:
        runtime = self._runtime_with_session(
            {"srv": {"command": "x", "allowed_tools": ["keep"]}}
        )
        with pytest.raises(PermissionError, match="allowed_tools"):
            asyncio.run(runtime.call_tool("srv", "drop", {}))

    def test_allowed_tool_passes(self) -> None:
        runtime = self._runtime_with_session(
            {"srv": {"command": "x", "allowed_tools": ["keep"]}}
        )
        assert asyncio.run(runtime.call_tool("srv", "keep", {})) == "ok"

    def test_no_allowlist_passes(self) -> None:
        runtime = self._runtime_with_session({"srv": {"command": "x"}})
        assert asyncio.run(runtime.call_tool("srv", "anything", {})) == "ok"


class TestAdapterHonesty:
    def _adapter(self, hint: bool | None) -> McpToolAdapter:
        info = McpCatalogTool(
            server_name="srv",
            safe_server_name="srv",
            tool_name="t",
            description="d",
            input_schema={"type": "object", "properties": {}},
            read_only_hint=hint,
        )
        return McpToolAdapter(MagicMock(spec=McpRuntime), info, "mcp__srv__t")

    def test_declared_read_only_is_read_only(self) -> None:
        adapter = self._adapter(True)
        assert adapter.is_read_only(adapter.input_model()) is True

    @pytest.mark.parametrize("hint", [False, None])
    def test_everything_else_is_not(self, hint: bool | None) -> None:
        # The old hardcoded True is the whole gate bypass; undeclared must
        # mean NOT read-only, because "the server didn't say" is not a
        # safety property.
        adapter = self._adapter(hint)
        assert adapter.is_read_only(adapter.input_model()) is False


class TestBootstrap:
    def test_bootstrap_registers_and_advertises_by_default(self, monkeypatch) -> None:
        from prometheus.context.dynamic_tools import DynamicToolLoader
        from prometheus.mcp.bootstrap import create_mcp_runtime
        from prometheus.tools.base import ToolRegistry

        async def _fake_connect_all(self):  # noqa: ANN001
            pass

        monkeypatch.setattr(McpRuntime, "connect_all", _fake_connect_all)
        monkeypatch.setattr(
            McpRuntime, "list_tools",
            lambda self: [McpCatalogTool(
                server_name="srv", safe_server_name="srv", tool_name="echo",
                description="d", input_schema={"type": "object"},
                read_only_hint=True,
            )],
        )
        from prometheus.tools.tool_search import ToolSearchTool

        registry = ToolRegistry()
        # A real tool_search, so the search_mcp reader at the end of the
        # bootstrap RUNS — the daemon always has one, and a NameError
        # there is swallowed into "MCP runtime not available" (caught
        # live 2026-09-01 while the loop-level assertions stayed green).
        search = ToolSearchTool()
        search.set_registry(registry)
        registry.register(search)
        loader = DynamicToolLoader(
            registry, {"enabled": True, "always_loaded": []}
        )
        config = {"mcp_servers": {"srv": {"command": "x"}}}
        runtime = asyncio.run(create_mcp_runtime(config, registry, tool_loader=loader))
        assert runtime is not None, "bootstrap returned None — see the WARNING it logged"
        assert registry.get("mcp__srv__echo") is not None
        assert registry.get("mcp_status") is not None
        assert search.include_mcp is True                    # the reader ran
        # Default (#369): a configured server IS the advertise decision —
        # its tools are in the advertised schemas without any config.
        names = {s["name"] for s in loader.schemas_for_run(True)}
        assert "mcp__srv__echo" in names

    def test_bootstrap_defers_when_the_operator_says_so(self, monkeypatch) -> None:
        from prometheus.context.dynamic_tools import DynamicToolLoader
        from prometheus.mcp.bootstrap import create_mcp_runtime
        from prometheus.tools.base import ToolRegistry

        async def _fake_connect_all(self):  # noqa: ANN001
            pass

        monkeypatch.setattr(McpRuntime, "connect_all", _fake_connect_all)
        monkeypatch.setattr(
            McpRuntime, "list_tools",
            lambda self: [McpCatalogTool(
                server_name="srv", safe_server_name="srv", tool_name="echo",
                description="d", input_schema={"type": "object"},
                read_only_hint=True,
            )],
        )
        registry = ToolRegistry()
        loader = DynamicToolLoader(
            registry, {"enabled": True, "always_loaded": []}
        )
        config = {
            "mcp_servers": {"srv": {"command": "x"}},
            "tools": {"deferred_loading": {"mcp_always_deferred": True}},
        }
        runtime = asyncio.run(create_mcp_runtime(config, registry, tool_loader=loader))
        assert runtime is not None
        assert registry.get("mcp__srv__echo") is not None      # registered …
        names = {s["name"] for s in loader.schemas_for_run(True)}
        assert "mcp__srv__echo" not in names                    # … not advertised

    def test_bootstrap_advertises_when_configured(self, monkeypatch) -> None:
        from prometheus.context.dynamic_tools import DynamicToolLoader
        from prometheus.mcp.bootstrap import create_mcp_runtime
        from prometheus.tools.base import ToolRegistry

        async def _fake_connect_all(self):  # noqa: ANN001
            pass

        monkeypatch.setattr(McpRuntime, "connect_all", _fake_connect_all)
        monkeypatch.setattr(
            McpRuntime, "list_tools",
            lambda self: [McpCatalogTool(
                server_name="srv", safe_server_name="srv", tool_name="echo",
                description="d", input_schema={"type": "object"},
            )],
        )
        registry = ToolRegistry()
        loader = DynamicToolLoader(
            registry, {"enabled": True, "always_loaded": []}
        )
        config = {
            "mcp_servers": {"srv": {"command": "x"}},
            "tools": {"deferred_loading": {"mcp_always_deferred": False}},
        }
        asyncio.run(create_mcp_runtime(config, registry, tool_loader=loader))
        names = {s["name"] for s in loader.schemas_for_run(True)}
        assert "mcp__srv__echo" in names
