"""#332 — the MCP management REST surface (Beacon Connectors, B1).

Drives the real routes against a REAL McpRuntime whose transport is stubbed
at the same two seams tests/test_mcp_allowlist.py established
(_connect_stdio + _list_all_tools), a REAL McpServerStore on the isolated
data dir, and a REAL ToolRegistry reached the way the routes reach it (via
app.state.ws_bridge.loop_context). The acceptance items come from the
issue: a server added over REST is *called in a live loop*; disabling it or
shrinking allowed_tools demonstrably removes the tool; GET never returns a
credential value.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("mcp")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from prometheus.mcp.runtime import McpRuntime, _McpSession  # noqa: E402
from prometheus.mcp.store import (  # noqa: E402
    McpServerStore,
    McpStoreError,
    merged_server_configs,
)
from prometheus.tools.base import ToolRegistry  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

SECRET = "sk-mcp-super-secret-value-1234"


def _offered(name: str, read_only: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=f"{name} desc",
        inputSchema={"type": "object", "properties": {}},
        annotations=SimpleNamespace(readOnlyHint=read_only),
    )


@pytest.fixture()
def rig(monkeypatch):
    """App + live runtime + registry, transport stubbed to offer two tools."""
    offered = [_offered("lookup"), _offered("search")]

    def _fake_session() -> _McpSession:
        inner = MagicMock()
        inner.send_ping = AsyncMock()
        inner.call_tool = AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="mcp says ok")]
        ))
        return _McpSession(
            server_name="?", session=inner,
            transport_type="stdio", _exit_stack=AsyncMock(),
        )

    async def _fake_connect_stdio(self, server_name, config):  # noqa: ANN001
        return _fake_session()

    async def _fake_list(session):  # noqa: ANN001
        return offered

    monkeypatch.setattr(McpRuntime, "_connect_stdio", _fake_connect_stdio)
    monkeypatch.setattr("prometheus.mcp.runtime._list_all_tools", _fake_list)

    runtime = McpRuntime({})
    registry = ToolRegistry()
    app = create_app({"mcp_servers": {"yaml-srv": {"command": "cfg"}}})
    app.state.mcp_runtime = runtime
    app.state.ws_bridge = SimpleNamespace(
        loop_context=SimpleNamespace(tool_registry=registry)
    )
    return TestClient(app), runtime, registry


class _PlainTool:
    """Minimal registry citizen for the grammar test's boot set."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"{name} desc"

    def to_api_schema(self) -> dict:
        return {"name": self.name, "description": self.description,
                "input_schema": {"type": "object", "properties": {
                    "path": {"type": "string"}}, "required": ["path"]}}

    def is_read_only(self, arguments) -> bool:  # noqa: ANN001
        return True


def _post_server(client, name: str = "docs", **extra):
    body = {"name": name, "command": "npx", "args": ["-y", "docs-mcp"],
            "env": {"DOCS_TOKEN": SECRET}, **extra}
    return client.post("/api/mcp/servers", json=body)


# --------------------------------------------------------------------------- #
# Store unit behaviour
# --------------------------------------------------------------------------- #


class TestStore:
    def test_validate_refuses_the_right_things(self) -> None:
        v = McpServerStore.validate
        with pytest.raises(McpStoreError, match="name"):
            v("bad name!", {"command": "x"})
        with pytest.raises(McpStoreError, match="unknown key"):
            v("ok", {"command": "x", "mystery": 1})
        with pytest.raises(McpStoreError, match="stdio"):
            v("ok", {"args": ["only"]})
        with pytest.raises(McpStoreError, match="control"):
            v("ok", {"command": "x", "env": {"A": "b\nc"}})
        with pytest.raises(McpStoreError, match="allowed_tools"):
            v("ok", {"command": "x", "allowed_tools": "not-a-list"})

    def test_crud_roundtrip_and_secret_projection(self) -> None:
        store = McpServerStore()
        store.upsert("s1", {"command": "x", "env": {"KEY": SECRET}})
        loaded = store.load()["s1"]
        assert loaded["env"]["KEY"] == SECRET  # stored for the subprocess
        view = McpServerStore.public_view(loaded)
        assert "env" not in view
        assert view["env_names"] == ["KEY"]
        store.patch("s1", {"allowed_tools": ["a"]})
        assert store.load()["s1"]["allowed_tools"] == ["a"]
        assert store.delete("s1") is True
        assert store.delete("s1") is False

    def test_merge_yaml_wins_and_disabled_is_absent(self) -> None:
        store = McpServerStore()
        store.upsert("dup", {"command": "store-side"})
        store.upsert("off", {"command": "x", "enabled": False})
        store.upsert("on", {"command": "y"})
        merged = merged_server_configs(
            {"mcp_servers": {"dup": {"command": "yaml-side"}}}, store,
        )
        assert merged["dup"]["command"] == "yaml-side"
        assert "off" not in merged           # structurally absent, not hidden
        assert merged["on"] == {"command": "y"}


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


class TestRoutes:
    def test_add_goes_live_and_get_never_leaks_the_secret(self, rig) -> None:
        client, runtime, registry = rig
        resp = _post_server(client)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["applies"] == "live"
        assert SECRET not in resp.text                      # write-only
        assert body["server"]["env_names"] == ["DOCS_TOKEN"]

        # Live: the tools exist in the registry NOW, no restart.
        assert registry.get("mcp__docs__lookup") is not None
        assert registry.get("mcp__docs__search") is not None

        listing = client.get("/api/mcp/servers")
        assert SECRET not in listing.text
        cards = {s["name"]: s for s in listing.json()["servers"]}
        assert cards["docs"]["health"]["state"] == "connected"
        assert {t["tool_name"] for t in cards["docs"]["tools"]} == {
            "lookup", "search",
        }
        # The yaml server rides along, marked config-managed.
        assert cards["yaml-srv"]["source"] == "config"

    def test_card_reports_the_name_the_registry_actually_holds(self, rig, monkeypatch) -> None:
        """FOUNDATION §4 live run, 2026-09-01: the card said
        ``mcp__context7__resolve-library-id`` while the registry held
        ``mcp__context7__resolve_library_id`` — build_safe_tool_name sanitises
        and the card rebuilt the name by concatenation. The surface that tells
        an operator what to call must report what registration produced."""
        client, runtime, registry = rig

        async def _offered_with_hyphen(session):  # noqa: ANN001
            return [_offered("resolve-library-id"), _offered("query-docs")]

        monkeypatch.setattr("prometheus.mcp.runtime._list_all_tools", _offered_with_hyphen)
        resp = _post_server(client, name="c7")
        assert resp.status_code == 200, resp.text

        reported = {t["tool_name"]: t["registered_as"] for t in resp.json()["server"]["tools"]}
        assert reported == {
            "resolve-library-id": "mcp__c7__resolve_library_id",
            "query-docs": "mcp__c7__query_docs",
        }
        # Every reported name resolves; the naive concatenation does not.
        for name in reported.values():
            assert registry.get(name) is not None, name
        assert registry.get("mcp__c7__resolve-library-id") is None

        # Same truth from the listing route, not only the POST echo.
        cards = {s["name"]: s for s in client.get("/api/mcp/servers").json()["servers"]}
        assert {t["registered_as"] for t in cards["c7"]["tools"]} == set(reported.values())

    def test_added_tool_is_called_in_a_live_loop(self, rig) -> None:
        # THE acceptance line from the issue — driven through run_loop, not
        # asserted off a flag.
        from prometheus.engine.agent_loop import LoopContext, run_loop
        from prometheus.engine.messages import (
            ConversationMessage, TextBlock, ToolUseBlock,
        )
        from prometheus.engine.usage import UsageSnapshot
        from prometheus.providers.base import (
            ApiMessageCompleteEvent, ModelProvider,
        )

        client, runtime, registry = rig
        assert _post_server(client).status_code == 200

        class _P(ModelProvider):
            def __init__(self) -> None:
                self.requests: list = []

            async def stream_message(self, request):  # noqa: ANN001
                self.requests.append(request)
                content = (
                    [ToolUseBlock(id="t1", name="mcp__docs__lookup", input={})]
                    if len(self.requests) == 1 else [TextBlock(text="done")]
                )
                yield ApiMessageCompleteEvent(
                    message=ConversationMessage(role="assistant", content=content),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                    stop_reason="stop",
                )

        provider = _P()
        ctx = LoopContext(
            provider=provider, model="stub", system_prompt="",
            max_tokens=64, tool_registry=registry,
        )

        async def _drain() -> None:
            async for _ in run_loop(
                ctx, [ConversationMessage.from_user_text("go")]
            ):
                pass

        asyncio.run(_drain())
        assert len(provider.requests) >= 2
        # The tool result the model got back came from the MCP session.
        assert "mcp says ok" in str(provider.requests[1].messages)

    def test_patch_allowed_tools_removes_the_tool_everywhere(self, rig) -> None:
        client, runtime, registry = rig
        _post_server(client)
        resp = client.patch(
            "/api/mcp/servers/docs", json={"allowed_tools": ["lookup"]}
        )
        assert resp.status_code == 200
        assert registry.get("mcp__docs__lookup") is not None
        assert registry.get("mcp__docs__search") is None    # gone, not hidden
        # And the call seam refuses a forced call for the excluded name.
        with pytest.raises(PermissionError):
            asyncio.run(runtime.call_tool("docs", "search", {}))

    def test_disable_removes_tools_and_reads_disabled(self, rig) -> None:
        client, runtime, registry = rig
        _post_server(client)
        resp = client.patch("/api/mcp/servers/docs", json={"enabled": False})
        assert resp.status_code == 200
        assert registry.get("mcp__docs__lookup") is None
        cards = {s["name"]: s
                 for s in client.get("/api/mcp/servers").json()["servers"]}
        assert cards["docs"]["enabled"] is False

    def test_delete_removes_everything(self, rig) -> None:
        client, runtime, registry = rig
        _post_server(client)
        assert client.delete("/api/mcp/servers/docs").status_code == 200
        assert registry.get("mcp__docs__lookup") is None
        names = [s["name"]
                 for s in client.get("/api/mcp/servers").json()["servers"]]
        assert "docs" not in names
        assert client.delete("/api/mcp/servers/docs").status_code == 404

    # ------------------------------------------------------------------ #
    # #370 — a tool registered after boot must reach the model's grammar
    # ------------------------------------------------------------------ #

    def test_every_live_change_calls_the_grammar_refresh(self, rig) -> None:
        """The daemon hands the routes its grammar refresh; add, allowlist
        change, disable and delete each fire it. Before #370 nothing did."""
        client, runtime, registry = rig
        calls: list[str] = []
        client.app.state.on_tools_changed = lambda: calls.append("refresh")

        assert _post_server(client).status_code == 200
        assert calls == ["refresh"]                       # register
        client.patch("/api/mcp/servers/docs", json={"allowed_tools": ["lookup"]})
        assert calls == ["refresh"] * 3                   # unregister + register
        client.patch("/api/mcp/servers/docs", json={"enabled": False})
        assert calls == ["refresh"] * 4                   # unregister
        client.patch("/api/mcp/servers/docs", json={"enabled": True})
        assert calls == ["refresh"] * 5                   # register (nothing to unregister)
        assert client.delete("/api/mcp/servers/docs").status_code == 200
        assert calls == ["refresh"] * 6                   # unregister

    def test_rest_added_tool_is_producible_by_a_real_llama_grammar(self, rig) -> None:
        """End to end through the real enforcer and the real provider,
        wired exactly as daemon._update_grammar wires them: before the
        server is added a forced turn on its tool cannot even derive a
        grammar; after, the boot grammar admits the tool and the derived
        one is llama-valid; after delete, the boot grammar drops it."""
        from prometheus.adapter.enforcer import StructuredOutputEnforcer
        from prometheus.providers.llama_cpp import LlamaCppProvider

        client, runtime, registry = rig
        registry.register(_PlainTool("read_file"))        # something in the boot set
        enforcer = StructuredOutputEnforcer()
        provider = LlamaCppProvider()

        def _update_grammar() -> None:                    # mirror of daemon.py:949-966
            schemas = registry.to_api_schema()
            provider.set_grammar(enforcer.generate_grammar(schemas))
            provider.set_grammar_source(enforcer, schemas)

        _update_grammar()                                  # boot
        client.app.state.on_tools_changed = _update_grammar
        tool = "mcp__docs__lookup"

        assert not StructuredOutputEnforcer.grammar_admits_tool(provider._grammar, tool)
        with pytest.raises(Exception, match="not among the provided tool schemas"):
            provider._derived_grammar(tool, only_tool=tool)   # the #370 failure, verbatim

        assert _post_server(client).status_code == 200
        assert registry.get(tool) is not None
        assert StructuredOutputEnforcer.grammar_admits_tool(provider._grammar, tool)
        derived = provider._derived_grammar(tool, only_tool=tool)
        assert StructuredOutputEnforcer.grammar_admits_tool(derived, tool)
        # #77: llama.cpp silently rejects continuation-line alternates and
        # then decodes unconstrained. Whatever a refresh emits must pass the
        # same lint the boot grammar does.
        for g in (provider._grammar, derived):
            bad = [ln for ln in g.splitlines() if ln.lstrip().startswith("|")]
            assert not bad, f"llama-invalid continuation lines: {bad[:2]}"

        assert client.delete("/api/mcp/servers/docs").status_code == 200
        assert registry.get(tool) is None
        assert not StructuredOutputEnforcer.grammar_admits_tool(provider._grammar, tool)

    def test_a_failed_refresh_is_a_500_and_rolls_the_registration_back(self, rig) -> None:
        """Loud, not stale: the response says it, the log says it, and the
        registry does not hold a tool the grammar cannot produce. The stored
        definition survives so the operator can see and retry it."""
        client, runtime, registry = rig

        def _broken_refresh() -> None:
            raise RuntimeError("enforcer exploded")

        client.app.state.on_tools_changed = _broken_refresh
        resp = _post_server(client)
        assert resp.status_code == 500, resp.text
        body = resp.json()
        assert body["error"] == "grammar_refresh_failed"
        assert "enforcer exploded" in body["detail"]
        assert "rolled back" in body["applies"]
        assert registry.get("mcp__docs__lookup") is None
        assert registry.get("mcp__docs__search") is None
        # Stored, honestly: the card is there, source=store, and a later
        # PATCH can retry the live half.
        cards = {s["name"]: s for s in client.get("/api/mcp/servers").json()["servers"]}
        assert cards["docs"]["source"] == "store"

        client.app.state.on_tools_changed = lambda: None   # refresh works again
        assert client.patch("/api/mcp/servers/docs", json={"enabled": True}).status_code == 200
        assert registry.get("mcp__docs__lookup") is not None

    def test_without_a_hook_the_routes_still_apply(self, rig) -> None:
        """No daemon grammar (cloud provider, grammar off, this rig): the
        absence of a hook is not an error."""
        client, runtime, registry = rig
        assert not hasattr(client.app.state, "on_tools_changed")
        assert _post_server(client).status_code == 200
        assert registry.get("mcp__docs__lookup") is not None

    # ------------------------------------------------------------------ #
    # #369 — a server added over REST is advertised on the next run
    # ------------------------------------------------------------------ #

    def _loader_hook(self, registry, config):
        from prometheus.context.dynamic_tools import DynamicToolLoader
        from prometheus.mcp.bootstrap import sync_mcp_advertisement

        loader = DynamicToolLoader(registry, {"enabled": True, "always_loaded": ["read_file"]})
        return loader, lambda runtime: sync_mcp_advertisement(config, loader, runtime)

    def test_rest_added_tools_are_advertised_by_default_and_leave_on_delete(self, rig) -> None:
        client, runtime, registry = rig
        registry.register(_PlainTool("read_file"))
        loader, sync = self._loader_hook(registry, {})
        client.app.state.on_tools_changed = lambda: sync(runtime)   # the daemon's composition

        advertised = lambda: {s["name"] for s in loader.schemas_for_run(True)}  # noqa: E731
        assert advertised() == {"read_file"}
        assert _post_server(client).status_code == 200
        assert advertised() == {"read_file", "mcp__docs__lookup", "mcp__docs__search"}
        client.patch("/api/mcp/servers/docs", json={"allowed_tools": ["lookup"]})
        assert advertised() == {"read_file", "mcp__docs__lookup"}        # the excluded one left
        assert client.delete("/api/mcp/servers/docs").status_code == 200
        assert advertised() == {"read_file"}                             # static list untouched

    def test_rest_added_tools_stay_deferred_when_the_operator_says_so(self, rig) -> None:
        client, runtime, registry = rig
        registry.register(_PlainTool("read_file"))
        loader, sync = self._loader_hook(
            registry, {"tools": {"deferred_loading": {"mcp_always_deferred": True}}},
        )
        client.app.state.on_tools_changed = lambda: sync(runtime)
        assert _post_server(client).status_code == 200
        assert registry.get("mcp__docs__lookup") is not None            # reachable by name …
        assert {s["name"] for s in loader.schemas_for_run(True)} == {"read_file"}   # … not advertised

    def test_yaml_managed_servers_are_read_only(self, rig) -> None:
        client, _runtime, _registry = rig
        assert _post_server(client, name="yaml-srv").status_code == 409
        assert client.patch("/api/mcp/servers/yaml-srv",
                            json={"enabled": False}).status_code == 409
        assert client.delete("/api/mcp/servers/yaml-srv").status_code == 409

    def test_bad_definition_is_a_400_not_a_partial_persist(self, rig) -> None:
        client, _runtime, _registry = rig
        resp = client.post("/api/mcp/servers",
                           json={"name": "half", "mystery": True})
        assert resp.status_code == 400
        assert "half" not in McpServerStore().load()

    def test_without_a_runtime_the_answer_is_honest(self) -> None:
        # A boot with MCP dark: definitions persist, and the response SAYS
        # a restart is needed instead of claiming live effect.
        app = create_app({})
        app.state.mcp_runtime = None
        client = TestClient(app)
        resp = _post_server(client, name="later")
        assert resp.status_code == 200
        assert "restart" in resp.json()["applies"]
        card = client.get("/api/mcp/servers").json()
        assert card["wired"] is False
        assert card["servers"][0]["health"]["state"] == "not_running"
