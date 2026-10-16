"""The ``agent`` tool gets its spawner from the daemon's loop (#761), and the subagent is gated.

``AgentTool`` reads ``context.metadata["subagent_spawner"]`` and answers "SubagentSpawner not
configured" when it is missing. Nothing in ``src/`` ever set that key, so on the production daemon the
tool was offered to the model and could never succeed. The tests that exercised the tool set the key by
hand, which is why nobody saw it.

These tests do not set the key. They build a loop the way ``run_daemon()`` does, let the production
wiring attach the spawner, and run a turn whose model calls ``agent``; the tool then reaches the spawner
through the loop's own context path (``AgentLoop`` -> ``LoopContext`` -> ``ToolExecutionContext``).

The second half is a safety property. A subagent's loop is built by ``SubagentSpawner.spawn()``, and a
loop with no ``permission_checker`` skips the whole permission block, so a spawner wired without the
gate would run every tool it hands out (before #790 an unknown ``subagent_type`` handed out the parent's
whole registry) with no gate, no approvals, no denied paths and no audit rows. The spawner is therefore
given the parent loop's gate, and a test pins that a subagent's tool call reaches it and that a denial
stops the call.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest
from pydantic import BaseModel

import prometheus.coordinator.agent_definitions as agent_definitions
from prometheus.coordinator.agent_definitions import AgentDefinition
from prometheus.engine.agent_loop import AgentLoop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolResultBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.permissions.checker import PermissionDecision
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult
from prometheus.tools.builtin.agent import AgentTool
from tests.support.doubles import register_double

PARENT_MARK = "PARENT-TURN"
SUB_REPLY = "sub-says-hi"
NOT_WIRED = "SubagentSpawner not configured"
PROBING = "probing-agent"  # a registered definition that names exactly the two test tools it may use


@pytest.fixture(autouse=True)
def _probing_agent_type(monkeypatch):
    monkeypatch.setattr(agent_definitions, "_BUILTIN_AGENTS", dict(agent_definitions._BUILTIN_AGENTS))
    agent_definitions.register_agent_definition(
        AgentDefinition(name=PROBING, description="test agent", tools=["probe", "agent"]))


class _NoInput(BaseModel):
    pass


class _Probe(BaseTool):
    """A tool the subagent can call; counts how often it really ran."""

    name = "probe"
    description = "test tool"
    input_model = _NoInput

    def __init__(self) -> None:
        self.ran = 0

    async def execute(self, arguments, context):  # noqa: ANN001
        self.ran += 1
        return ToolResult(output="probed", is_error=False)


@register_double("agent_tool_wiring.TwoLoopProvider", replaces="prometheus.providers.base.ModelProvider")
class _TwoLoopProvider(ModelProvider):
    """One scripted model for both loops, told apart by the system prompt.

    The parent turn calls ``agent`` and then answers; the subagent turn answers ``SUB_REPLY``, optionally
    after calling ``sub_calls`` first. ``parent_saw`` / ``sub_saw`` hold the tool results each loop got.

    ``subagent_type`` picks the subagent's tool set. The gate and no-recursion tests use ``PROBING``, a definition
    registered by the fixture below that names exactly ``probe`` and ``agent``; an agent type with no definition is
    refused outright now (#790), so it can no longer be used to hand a subagent the parent's whole registry.
    """

    def __init__(self, sub_calls: str | None = None, subagent_type: str = "general-purpose") -> None:
        self.sub_calls = sub_calls
        self.subagent_type = subagent_type
        self.parent_saw: list[str] = []
        self.sub_saw: list[str] = []
        self.sub_requests = 0

    async def stream_message(self, request):  # noqa: ANN001
        results = [b for b in request.messages[-1].content if isinstance(b, ToolResultBlock)]
        if PARENT_MARK in (request.system_prompt or ""):
            if results:
                self.parent_saw.extend(b.content for b in results)
                blocks = [TextBlock(text="parent done")]
            else:
                blocks = [ToolUseBlock(id="p1", name="agent", input={
                    "description": "say hi", "prompt": "say hi", "subagent_type": self.subagent_type})]
        else:
            self.sub_requests += 1
            self.sub_saw.extend(b.content for b in results)
            if self.sub_calls and not results:
                call_input = ({"description": "again", "prompt": "again", "subagent_type": self.subagent_type}
                              if self.sub_calls == "agent" else {})
                blocks = [ToolUseBlock(id="s1", name=self.sub_calls, input=call_input)]
            else:
                blocks = [TextBlock(text=SUB_REPLY)]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=blocks),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


class _Gate:
    """Stands in for the SecurityGate: records every tool it is asked about, allows or denies ``probe``."""

    def __init__(self, deny_probe: bool = False) -> None:
        self.deny_probe = deny_probe
        self.asked: list[str] = []

    def evaluate(self, tool_name, **kw):  # noqa: ANN001
        self.asked.append(tool_name)
        if tool_name == "probe" and self.deny_probe:
            return PermissionDecision.deny("probe is not allowed here")
        return PermissionDecision.allow("test")


def _parent_loop(provider: _TwoLoopProvider, probe: _Probe, gate: _Gate | None = None) -> AgentLoop:
    """The parent loop, built with the kwargs the daemon's loop carries for this purpose."""
    registry = ToolRegistry()
    registry.register(AgentTool())
    registry.register(probe)
    return AgentLoop(provider=provider, model="m", tool_registry=registry, permission_checker=gate)


def _run(loop: AgentLoop):
    return asyncio.run(loop.run_async(system_prompt=PARENT_MARK, user_message="please delegate"))


def test_without_the_wiring_the_agent_tool_is_not_configured():
    """The control: the loop's own context path reproduces #761 when nothing attaches a spawner."""
    provider = _TwoLoopProvider()
    _run(_parent_loop(provider, _Probe()))
    assert provider.parent_saw and NOT_WIRED in provider.parent_saw[0]
    assert provider.sub_requests == 0


def test_the_wired_loop_runs_a_subagent_and_returns_its_answer():
    from prometheus.coordinator.subagent import attach_subagent_spawner

    provider = _TwoLoopProvider()
    loop = _parent_loop(provider, _Probe())
    attach_subagent_spawner(loop)
    _run(loop)
    assert provider.parent_saw == [SUB_REPLY]
    assert provider.sub_requests >= 1


def test_a_subagents_tool_call_goes_through_the_parents_gate():
    from prometheus.coordinator.subagent import attach_subagent_spawner

    gate, probe = _Gate(), _Probe()
    provider = _TwoLoopProvider(sub_calls="probe", subagent_type=PROBING)
    loop = _parent_loop(provider, probe, gate)
    attach_subagent_spawner(loop)
    _run(loop)
    assert "agent" in gate.asked, "the parent's own call is gated, as before"
    assert "probe" in gate.asked, "the subagent's call must reach the same gate"
    assert probe.ran == 1
    assert provider.parent_saw == [SUB_REPLY]


def test_a_subagent_cannot_run_what_the_gate_denies():
    from prometheus.coordinator.subagent import attach_subagent_spawner

    gate, probe = _Gate(deny_probe=True), _Probe()
    provider = _TwoLoopProvider(sub_calls="probe", subagent_type=PROBING)
    loop = _parent_loop(provider, probe, gate)
    attach_subagent_spawner(loop)
    _run(loop)
    assert probe.ran == 0, "a denied tool call must not execute inside a subagent"
    assert any("probe is not allowed here" in s for s in provider.sub_saw), "the gate's reason comes back to the subagent"


def test_a_subagent_cannot_spawn_another():
    """By construction the subagent's loop carries no spawner, so ``agent`` inside it is not configured."""
    from prometheus.coordinator.subagent import attach_subagent_spawner

    provider = _TwoLoopProvider(sub_calls="agent", subagent_type=PROBING)
    loop = _parent_loop(provider, _Probe())
    attach_subagent_spawner(loop)
    _run(loop)
    assert any(NOT_WIRED in s for s in provider.sub_saw), provider.sub_saw
    assert provider.parent_saw == [SUB_REPLY]


def test_run_daemon_attaches_the_spawner_right_after_it_builds_the_loop():
    """The production wiring is in ``run_daemon()``: the loop is built, then the spawner is attached."""
    import prometheus.daemon as daemon

    src = inspect.getsource(daemon.run_daemon)
    built = src.index("agent_loop = AgentLoop(")
    assert "attach_subagent_spawner(agent_loop)" in src[built:], "run_daemon() never attaches a spawner"
    assert src.index("attach_subagent_spawner(agent_loop)", built) > built


# ── The Beacon loop (the daemon's second construction) ──────────────────────────────────────────────────
#
# run_daemon() builds the loop twice: the AgentLoop above (telegram, CLI) and a LoopContext that every
# Beacon, REST and WebSocket turn runs on through ``run_loop``. #761 attached the spawner to the AgentLoop
# only, so a Beacon turn's ``agent`` call still answered "not configured" on 0.9.10.


def _beacon_context(loop: AgentLoop, *, share_metadata: bool):
    """The web/Beacon LoopContext, built from the same pieces the daemon hands it, and sealed as it is."""
    from prometheus.engine.agent_loop import LoopContext

    return LoopContext(
        provider=loop._provider,
        model=loop._model,
        system_prompt=PARENT_MARK,
        max_tokens=4096,
        tool_registry=loop._tool_registry,
        tool_metadata=loop._tool_metadata if share_metadata else None,
        permission_checker=loop._permission_checker,
    ).seal()


def _run_beacon_turn(ctx) -> None:  # noqa: ANN001
    from prometheus.engine.agent_loop import run_loop

    async def go() -> None:
        msgs = [ConversationMessage.from_user_text("please delegate")]
        async for _event, _usage in run_loop(ctx, msgs, session_id="beacon:test", surface="beacon"):
            pass

    asyncio.run(go())


def test_a_beacon_context_without_the_loops_metadata_is_not_configured():
    """The control: 0.9.10's web LoopContext had no tool metadata, so ``agent`` failed from Beacon."""
    from prometheus.coordinator.subagent import attach_subagent_spawner

    provider = _TwoLoopProvider()
    loop = _parent_loop(provider, _Probe())
    attach_subagent_spawner(loop)
    _run_beacon_turn(_beacon_context(loop, share_metadata=False))
    assert provider.parent_saw and NOT_WIRED in provider.parent_saw[0]
    assert provider.sub_requests == 0


def test_a_beacon_turn_runs_a_subagent_through_the_shared_metadata():
    from prometheus.coordinator.subagent import attach_subagent_spawner

    gate, probe = _Gate(), _Probe()
    provider = _TwoLoopProvider(sub_calls="probe", subagent_type=PROBING)
    loop = _parent_loop(provider, probe, gate)
    attach_subagent_spawner(loop)
    _run_beacon_turn(_beacon_context(loop, share_metadata=True))
    assert provider.parent_saw == [SUB_REPLY]
    assert probe.ran == 1
    assert "probe" in gate.asked, "a subagent started from Beacon is gated like one started from telegram"


def test_run_daemon_hands_the_beacon_context_the_agent_loops_own_metadata():
    """The production wiring: the web LoopContext gets the AgentLoop's dict itself, after the attach."""
    import prometheus.daemon as daemon

    src = inspect.getsource(daemon.run_daemon)
    attached = src.index("attach_subagent_spawner(agent_loop)")
    ctx_at = src.index("loop_context = LoopContext(")
    assert ctx_at > attached, "the web LoopContext must be built after the spawner is attached"
    ctx_src = src[ctx_at:src.index("loop_context.seal()", ctx_at)]
    assert "tool_metadata=agent_loop._tool_metadata," in ctx_src


def test_loop_tool_metadata_divergence_names_the_missing_keys():
    from types import SimpleNamespace

    from prometheus.daemon import loop_tool_metadata_divergence

    loop = SimpleNamespace(_tool_metadata={"subagent_spawner": object(), "printing_press": object()})
    assert loop_tool_metadata_divergence(loop, SimpleNamespace(tool_metadata=None)) == [
        "printing_press", "subagent_spawner"]
    assert loop_tool_metadata_divergence(loop, SimpleNamespace(tool_metadata=loop._tool_metadata)) == []
    assert loop_tool_metadata_divergence(SimpleNamespace(_tool_metadata=None),
                                         SimpleNamespace(tool_metadata=None)) == []
