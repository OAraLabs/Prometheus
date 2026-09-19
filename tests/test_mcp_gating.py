"""MCP calls reach the SecurityGate — FOUNDATION 2.3a, prerequisite 2.

Before #315, every MCP call bypassed the gate in full: the adapter
hardcoded is_read_only=True, its input model declared no fields so path
extraction saw nothing, and there was no command — three misses composing
to auto-allow for arbitrary third-party code. These tests pin the posture
since 2026-09-18: EVERY ``mcp__*`` call requires confirmation. A server's
``readOnlyHint`` is recorded and named in the reason but never trusted to
skip the prompt — the gate still sees no MCP argument (the adapter's input
model declares no fields), so nothing beneath the hint could catch a
"read-only" tool that reads ``~/.ssh``. An operator grant of kind ``tool``
silences the prompt; AUTONOMOUS mode waives it like every other APPROVE
tier (never the floor).

Deliberately NO ``mcp`` SDK import anywhere here: the gate rule keys on
the tool NAME. (CI has installed the ``mcp`` extra since 2026-08-28 —
``.github/workflows/ci.yml`` — so the SDK-dependent files run there too;
this file simply does not need it.)
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, ConfigDict

from prometheus.context.dynamic_tools import DynamicToolLoader
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.permissions.checker import Grant, PermissionMode, SecurityGate
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult


class _AnyInput(BaseModel):
    model_config = ConfigDict(extra="allow")


class _McpShapedTool(BaseTool):
    """The shape register_mcp_tools produces, without the SDK: an
    ``mcp__{server}__{tool}`` name, a permissive input model, and an
    is_read_only that reflects the server's declaration."""

    input_model = _AnyInput

    def __init__(self, name: str, *, read_only: bool) -> None:
        self.name = name
        self.description = f"{name} (mcp-shaped test tool)"
        self._read_only = read_only
        self.calls: list[dict] = []

    def is_read_only(self, arguments: BaseModel) -> bool:
        return self._read_only

    async def execute(self, arguments, context):  # noqa: ANN001
        self.calls.append(arguments.model_dump())
        return ToolResult(output="mcp says ok")


class TestGateRule:
    def test_non_read_only_mcp_requires_approval(self) -> None:
        gate = SecurityGate()
        decision = gate.evaluate("mcp__srv__write_thing", is_read_only=False)
        assert decision.action == "APPROVE"
        assert decision.requires_confirmation
        assert "mcp__srv__write_thing" in decision.reason

    def test_read_only_hint_does_not_skip_the_prompt(self) -> None:
        # Ruled 2026-09-18: the hint is a third party's self-declaration
        # with no check beneath it — the gate never sees an MCP argument.
        # It is recorded and named in the reason, not trusted.
        gate = SecurityGate()
        decision = gate.evaluate("mcp__srv__lookup", is_read_only=True)
        assert decision.action == "APPROVE"
        assert decision.requires_confirmation
        assert "read-only" in decision.reason
        assert "not trusted" in decision.reason

    def test_an_operator_tool_grant_silences_the_prompt(self) -> None:
        # The one thing that does, today: a `security.grants` entry of kind
        # "tool" — hand-written, because /approve always cannot mint one for
        # an MCP request (tests/test_mcp_doc_claims.py pins that). Grants
        # are checked before this tier, so it is honoured for exactly the
        # named tool and nothing else.
        gate = SecurityGate(
            grants=[Grant(kind="tool", value="", tool_name="mcp__srv__lookup")]
        )
        assert gate.evaluate("mcp__srv__lookup", is_read_only=True).action == "ALLOW"
        assert gate.evaluate("mcp__srv__other", is_read_only=True).action == "APPROVE"

    @pytest.mark.parametrize("origin", ["user", "system"])
    def test_both_origins_prompt(self, origin: str) -> None:
        # A present human asked for the TASK, not for whatever a
        # third-party server does with it — user origin does not relax
        # this tier the way it relaxes the bash network patterns.
        gate = SecurityGate()
        decision = gate.evaluate(
            "mcp__srv__write_thing", is_read_only=False, origin=origin
        )
        assert decision.action == "APPROVE"

    def test_autonomous_mode_waives_the_prompt(self) -> None:
        gate = SecurityGate(mode=PermissionMode.AUTONOMOUS)
        decision = gate.evaluate("mcp__srv__write_thing", is_read_only=False)
        assert decision.action == "ALLOW"

    def test_non_mcp_tools_unaffected(self) -> None:
        # The rule is scoped to the mcp__ namespace; a builtin that is not
        # read-only must not start prompting because of it.
        gate = SecurityGate()
        decision = gate.evaluate("bash", is_read_only=False)
        assert decision.action == "ALLOW"


class _ScriptedProvider(ModelProvider):
    """Round 1: calls the given tool. Round 2: ends the turn. Records the
    tool RESULTS it is sent back, which is the proof the call executed."""

    def __init__(self, tool_name: str) -> None:
        self._tool_name = tool_name
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        if len(self.requests) == 1:
            content = [ToolUseBlock(id="t1", name=self._tool_name, input={})]
        else:
            content = [TextBlock(text="done")]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


def _run(ctx) -> None:
    async def _drain() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")]):
            pass
    asyncio.run(_drain())


class TestMcpToolInALiveLoop:
    """Acceptance (FOUNDATION Part 4): an MCP tool is *called* in a live
    loop — run_loop, real registry, real gate — not merely registered."""

    def test_mcp_tool_executes_through_the_loop_under_an_operator_grant(self) -> None:
        tool = _McpShapedTool("mcp__srv__lookup", read_only=True)
        registry = ToolRegistry()
        registry.register(tool)
        provider = _ScriptedProvider(tool.name)
        ctx = LoopContext(
            provider=provider,
            model="stub",
            system_prompt="",
            max_tokens=128,
            tool_registry=registry,
            permission_checker=SecurityGate(
                grants=[Grant(kind="tool", value="", tool_name=tool.name)]
            ),
        )
        _run(ctx)
        assert tool.calls, "the MCP tool never executed"
        assert len(provider.requests) >= 2, "no round followed the tool call"

    def test_read_only_hinted_mcp_tool_is_stopped_by_the_gate(self) -> None:
        # The case that used to execute unprompted. Same rig as the grant
        # test above minus the grant: the hint alone must not get it run.
        tool = _McpShapedTool("mcp__srv__lookup", read_only=True)
        registry = ToolRegistry()
        registry.register(tool)
        provider = _ScriptedProvider(tool.name)
        ctx = LoopContext(
            provider=provider,
            model="stub",
            system_prompt="",
            max_tokens=128,
            tool_registry=registry,
            permission_checker=SecurityGate(),
        )
        _run(ctx)
        assert tool.calls == [], (
            "a read-only-hinted MCP tool executed without confirmation"
        )

    def test_non_read_only_mcp_tool_is_stopped_by_the_gate(self) -> None:
        tool = _McpShapedTool("mcp__srv__write_thing", read_only=False)
        registry = ToolRegistry()
        registry.register(tool)
        provider = _ScriptedProvider(tool.name)
        ctx = LoopContext(
            provider=provider,
            model="stub",
            system_prompt="",
            max_tokens=128,
            tool_registry=registry,
            permission_checker=SecurityGate(),
        )
        _run(ctx)
        # The gate returned APPROVE (requires_confirmation): the tool body
        # must NOT have run — whatever the loop does with the pending
        # approval, executing anyway would be the old bypass back again.
        assert tool.calls == [], (
            "a non-read-only MCP tool executed without confirmation"
        )


class TestAdvertisementDecision:
    """FOUNDATION 2.3a prerequisite 4: dynamically-registered tools get an
    explicit advertise-or-defer decision instead of silently landing on
    the invisible side (the vault_search failure class)."""

    def _registry(self) -> tuple[ToolRegistry, _McpShapedTool]:
        registry = ToolRegistry()
        for name in ("bash",):
            registry.register(_McpShapedTool(name, read_only=True))
        mcp_tool = _McpShapedTool("mcp__srv__echo", read_only=True)
        registry.register(mcp_tool)
        return registry, mcp_tool

    def test_loader_alone_advertises_only_the_static_list(self) -> None:
        # The LOADER has no MCP knowledge: a registered mcp__ tool that no
        # one added to the baseline is not advertised. The MCP default
        # (#369: advertise) is applied by mcp.bootstrap on top of this.
        registry, _ = self._registry()
        loader = DynamicToolLoader(
            registry, {"enabled": True, "always_loaded": ["bash"]}
        )
        names = {s["name"] for s in loader.schemas_for_run(True)}
        assert "mcp__srv__echo" not in names

    def test_add_always_loaded_advertises_them(self) -> None:
        # The advertise branch: bootstrap calls this when
        # tools.deferred_loading.mcp_always_deferred is false.
        registry, _ = self._registry()
        loader = DynamicToolLoader(
            registry, {"enabled": True, "always_loaded": ["bash"]}
        )
        loader.add_always_loaded(["mcp__srv__echo"])
        names = {s["name"] for s in loader.schemas_for_run(True)}
        assert "mcp__srv__echo" in names

    def test_deferred_mcp_tool_is_discoverable_via_tool_search(self) -> None:
        # Deferral is only honest if discovery works: the deferred tool
        # must surface for a query naming it.
        from prometheus.tools.tool_search import ToolSearchTool

        registry, _ = self._registry()
        ts = ToolSearchTool()
        ts.set_registry(registry)
        result = asyncio.run(ts.execute(
            ts.input_model(action="search", query="echo"), None  # type: ignore[arg-type]
        ))
        assert "mcp__srv__echo" in result.output

    def test_search_mcp_false_hides_them_from_search_not_select(self) -> None:
        from prometheus.tools.tool_search import ToolSearchTool

        registry, _ = self._registry()
        ts = ToolSearchTool()
        ts.set_registry(registry)
        ts.include_mcp = False
        searched = asyncio.run(ts.execute(
            ts.input_model(action="search", query="echo"), None  # type: ignore[arg-type]
        ))
        assert "mcp__srv__echo" not in searched.output
        # Naming a tool exactly is explicit in a way fuzzy search is not.
        selected = asyncio.run(ts.execute(
            ts.input_model(action="select", query="mcp__srv__echo"), None  # type: ignore[arg-type]
        ))
        assert "mcp__srv__echo" in selected.output


class TestDaemonWiring:
    """Source-order assertions, same style as the vault-marker wiring
    tests: the daemon constructs MCP (for the first time ever) and closes
    it at shutdown."""

    @pytest.fixture()
    def daemon_source(self) -> str:
        from pathlib import Path

        import prometheus.daemon as daemon_mod
        return Path(daemon_mod.__file__).read_text(encoding="utf-8")

    def test_daemon_constructs_mcp_after_loader_before_baseline_log(
        self, daemon_source: str
    ) -> None:
        loader = daemon_source.index("tool_loader = DynamicToolLoader(")
        mcp = daemon_source.index("create_mcp_runtime(")
        baseline = daemon_source.index("Tool advertisement baseline")
        assert loader < mcp < baseline, (
            "MCP must register after the loader exists (so the advertise "
            "decision can extend it) and before the baseline log (so the "
            "logged counts include it)"
        )

    def test_daemon_closes_mcp_at_shutdown(self, daemon_source: str) -> None:
        shutdown = daemon_source.index("await shutdown_event.wait()")
        close = daemon_source.index("await mcp_runtime.close()")
        assert shutdown < close
