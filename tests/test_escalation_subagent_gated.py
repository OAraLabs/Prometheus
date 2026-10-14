"""#789 — the router-escalation subagent runs its tool under the gate.

When a tool call fails validation more times than the adapter allows and ``router.escalation.enabled`` is on,
``_try_escalate_tool_call`` spawns a subagent on the escalation provider to attempt the call. That subagent's loop
was built with no ``permission_checker``, and a loop with none skips the whole permission block, so an escalated
``bash`` call ran with no gate: no approval for the commands the gate holds, no denied paths, no audit row.

These tests drive the real path: a real ``ModelRouter`` with escalation enabled, a real ``ModelAdapter`` whose retry
engine returns ESCALATE after repeated invalid calls, and the real ``SecurityGate`` as the gate. The tool
itself is a recording double, so a test that fails on the unfixed code cannot run anything dangerous.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from prometheus.adapter import ModelAdapter
from prometheus.adapter.validator import Strictness
from prometheus.engine.agent_loop import AgentLoop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolResultBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.permissions.checker import PermissionDecision, SecurityGate
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.providers.registry import ProviderRegistry
from prometheus.router.model_router import ModelRouter, RouterConfig
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult
from tests.support.doubles import register_double

PRIMARY_MARK = "PRIMARY-TURN"
ESCALATED_OK = "escalated-ok"
#: Matches the gate's ``git\s+push`` approve pattern but only echoes, so it is safe even if it runs.
HELD_COMMAND = 'echo "escalation test: git push"'
PLAIN_COMMAND = "echo escalation-test"


class _BashInput(BaseModel):
    command: str


class _Bash(BaseTool):
    """Stands in for the real bash tool: records the commands it is asked to run, runs none."""

    name = "bash"
    description = "test double for bash"
    input_model = _BashInput

    def __init__(self) -> None:
        self.ran: list[str] = []

    async def execute(self, arguments, context):  # noqa: ANN001
        self.ran.append(arguments.command)
        return ToolResult(output="ran", is_error=False)


def _reply(blocks):  # noqa: ANN001
    return ApiMessageCompleteEvent(
        message=ConversationMessage(role="assistant", content=blocks),
        usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        stop_reason="stop",
    )


@register_double("escalation_gated.PrimaryProvider", replaces="prometheus.providers.base.ModelProvider")
class _Primary(ModelProvider):
    """The local model: keeps emitting a bash call with no ``command`` (invalid) until the escalated answer arrives.

    Each bad call carries a different junk field: the loop's repeat guard blocks an identical re-run of a failing
    call, so only a model that varies its mistakes (as a real one does) can run the retry count up to escalation.
    """

    def __init__(self) -> None:
        self.attempts = 0

    async def stream_message(self, request):  # noqa: ANN001
        results = [b for b in request.messages[-1].content if isinstance(b, ToolResultBlock)]
        if any(b.content == ESCALATED_OK for b in results):
            yield _reply([TextBlock(text="primary done")])
        else:
            self.attempts += 1
            yield _reply([ToolUseBlock(id=f"bad{self.attempts}", name="bash", input={f"oops{self.attempts}": 1})])


@register_double("escalation_gated.EscalatedProvider", replaces="prometheus.providers.base.ModelProvider")
class _Escalated(ModelProvider):
    """The stronger model the router escalates to: makes the corrected bash call, then reports."""

    def __init__(self, command: str) -> None:
        self.command = command
        self.requests = 0

    async def stream_message(self, request):  # noqa: ANN001
        self.requests += 1
        if any(isinstance(b, ToolResultBlock) for b in request.messages[-1].content):
            yield _reply([TextBlock(text=ESCALATED_OK)])
        else:
            yield _reply([ToolUseBlock(id="good", name="bash", input={"command": self.command})])


class _Gate:
    """Records every tool it is asked about and allows or denies ``bash``."""

    def __init__(self, deny: bool = False) -> None:
        self.deny = deny
        self.asked: list[str] = []

    def evaluate(self, tool_name, **kw):  # noqa: ANN001
        self.asked.append(tool_name)
        if tool_name == "bash" and self.deny:
            return PermissionDecision.deny("bash is not allowed here")
        return PermissionDecision.allow("test")


def _run(gate, command: str, monkeypatch) -> tuple[_Bash, _Escalated]:  # noqa: ANN001
    """A loop whose primary model keeps failing validation, with escalation switched on in the real router."""
    escalated = _Escalated(command)
    monkeypatch.setattr(ProviderRegistry, "create", staticmethod(lambda cfg: escalated))
    router = ModelRouter(
        RouterConfig(escalation_enabled=True,
                     escalation_provider={"provider": "anthropic", "model": "big", "api_key": "k"}),
        primary_provider=_Primary(), primary_adapter=None, primary_model="m",
    )
    # Two retries, so the third invalid call escalates. The loop's circuit breaker ends a turn after three
    # consecutive failures; with the default of three retries escalation would only come on a later turn.
    adapter = ModelAdapter(strictness=Strictness.MEDIUM, max_retries=2, router=router)
    bash = _Bash()
    registry = ToolRegistry()
    registry.register(bash)
    loop = AgentLoop(provider=_Primary(), model="m", tool_registry=registry, adapter=adapter,
                     model_router=router, permission_checker=gate)
    asyncio.run(loop.run_async(system_prompt=PRIMARY_MARK, user_message="go"))
    return bash, escalated


def test_the_escalation_path_is_really_taken(monkeypatch):
    """The control: the repeated invalid calls end in escalation, and the escalated model's call is what runs."""
    bash, escalated = _run(_Gate(), PLAIN_COMMAND, monkeypatch)
    assert escalated.requests >= 1, "the router never escalated, so nothing below proves anything"
    assert bash.ran == [PLAIN_COMMAND], "only the escalated call is valid, so only it can have run"


def test_the_escalated_bash_call_reaches_the_gate(monkeypatch):
    gate = _Gate()
    _run(gate, PLAIN_COMMAND, monkeypatch)
    assert "bash" in gate.asked, "the escalated subagent's call must be evaluated by the gate"


def test_a_gate_denial_stops_the_escalated_call(monkeypatch):
    bash, _ = _run(_Gate(deny=True), PLAIN_COMMAND, monkeypatch)
    assert bash.ran == [], "a denied call must not run inside the escalation subagent"


def test_under_the_real_gate_a_command_that_needs_approval_is_refused(monkeypatch):
    """The normal rules: ``git push`` is held for approval, and with nobody to ask the real gate fails closed."""
    bash, _ = _run(SecurityGate(), HELD_COMMAND, monkeypatch)
    assert bash.ran == [], "an escalated command the gate holds for approval must not run unasked"


def test_under_the_real_gate_an_ordinary_command_still_runs(monkeypatch):
    """The gate is not a blanket refusal: the same escalation with an ordinary command goes through."""
    bash, _ = _run(SecurityGate(), PLAIN_COMMAND, monkeypatch)
    assert bash.ran == [PLAIN_COMMAND]


@pytest.mark.parametrize("deny", [False, True])
def test_the_primary_loops_own_gate_is_unchanged(deny, monkeypatch):
    """The parent's invalid calls fail validation before the gate, so they never reach it, as before."""
    gate = _Gate(deny=deny)
    _run(gate, PLAIN_COMMAND, monkeypatch)
    assert gate.asked.count("bash") <= 1, "only the escalated call is evaluated"
