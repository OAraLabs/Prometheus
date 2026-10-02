"""A cloud run that trips the circuit breaker stays at adapter tier ``off`` (WP-X.41).

The loop reads tier ``off`` as "this is a cloud model": microcompaction, the
``<tool_call>`` markup filter and text extraction of tool calls all skip on
``off`` and run on every other tier. The breaker's one-shot recovery used to
bump the run's adapter one rung up, ``off`` -> ``light``, and the run kept
that tier to its end. On 2026-09-23 that bumped a ``qwen3.8-max`` run, which
then microcompacted 65 times; each rewrite threw away the provider's cached
prompt prefix. The bump could not have helped: the error (``bash`` called with
``{}``) came from the tool, and a cloud API already enforces call structure,
so the adapter had nothing to repair.

Pinned here:

* a cloud run that trips the breaker keeps tier ``off`` and never
  microcompacts, through a real ``run_loop`` turn and the rows it writes;
* it still gets the one more chance the bump used to give (counters reset,
  the run continues), recorded as ``retry:cloud_tier_off``;
* a LOCAL model whose tier is ``off`` (an ``adapter.model_tiers`` override)
  still gets the bump, as does a provider we cannot name.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import LoopContext, _CircuitBreaker, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult


class _CommandInput(BaseModel):
    command: str


class _NeedsCommand(BaseTool):
    """Fails on ``{}`` the way bash did on 09-23: a required field is missing."""

    name = "needs_command"
    description = "runs a command"
    input_model = _CommandInput

    def is_read_only(self, arguments) -> bool:  # noqa: ANN001
        return True

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output=f"ran {arguments.command}", is_error=False)


class _NInput(BaseModel):
    n: int


class _BigRead(BaseTool):
    """A large result that differs per call, so microcompaction has work to do
    and the repeat detector never sees the same bytes twice."""

    name = "big_read"
    description = "returns a large result"
    input_model = _NInput

    def is_read_only(self, arguments) -> bool:  # noqa: ANN001
        return True

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output=f"result {arguments.n}\n" + "x" * 2000, is_error=False)


# Two identical failures, then the repeat guard blocks the call three times;
# the third identical BLOCKED result (and the fifth error) trips the breaker.
_FAILS = 5
_READS = 5   # rounds after the trip, enough for microcompaction to fire repeatedly


class _CloudProvider(ModelProvider):
    """A cloud provider (``qwen``) that trips the breaker, then works."""

    provider_name = "qwen"

    def __init__(self) -> None:
        self.calls = 0

    async def stream_message(self, request):  # noqa: ANN001
        n = self.calls
        self.calls += 1
        if n < _FAILS:
            content = [ToolUseBlock(id=f"f{n}", name="needs_command", input={})]
        elif n < _FAILS + _READS:
            content = [ToolUseBlock(id=f"r{n}", name="big_read", input={"n": n})]
        else:
            content = [TextBlock(text="done")]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


def test_a_cloud_run_that_trips_the_breaker_never_microcompacts(tmp_path):
    from prometheus.adapter import ModelAdapter

    registry = ToolRegistry()
    registry.register(_NeedsCommand())
    registry.register(_BigRead())
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    provider = _CloudProvider()
    ctx = LoopContext(
        provider=provider,
        model="qwen3.8-max",
        system_prompt="",
        max_tokens=128,
        tool_registry=registry,
        adapter=ModelAdapter(tier="off"),
        telemetry=tel,
        session_id="web",
        microcompact_after_turns=2,
    )
    messages = [ConversationMessage.from_user_text("go")]

    async def drain() -> None:
        async for _ in run_loop(ctx, messages, session_id="web:conv-1"):
            pass

    asyncio.run(drain())
    tel.close()
    con = sqlite3.connect(db)
    trips = con.execute(
        "SELECT adapter_tier, recovered, recovery_method FROM circuit_breaker_diagnostics"
    ).fetchall()
    compactions = con.execute(
        "SELECT round_index FROM subsystem_runs"
        " WHERE subsystem = 'agent_loop' AND operation = 'microcompact'"
    ).fetchall()
    con.close()

    # The harness must actually trip the breaker, or this measures nothing.
    assert [(tier, recovered) for tier, recovered, _ in trips] == [("off", 1)]
    assert compactions == [], (
        f"a cloud run microcompacted {len(compactions)} times after the breaker "
        f"tripped ({trips[0][2]}); every rewrite throws away the provider's "
        f"cached prompt prefix"
    )
    assert trips[0][2] == "retry:cloud_tier_off"
    # The trip is not the end of the run: it gets its one more chance.
    assert provider.calls == _FAILS + _READS + 1, "the run stopped at the trip"
    assert not any(
        "[microcompacted]" in getattr(block, "content", "")
        for msg in messages
        if isinstance(msg.content, list)
        for block in msg.content
    )


def _tripped(provider_name: str | None) -> tuple[LoopContext, object, _CircuitBreaker]:
    from prometheus.adapter import ModelAdapter

    provider = _CloudProvider()
    if provider_name is None:
        provider.provider_name = ""   # unknown: no name, no class to map
    else:
        provider.provider_name = provider_name
    adapter = ModelAdapter(tier="off")
    ctx = LoopContext(
        provider=provider, model="m", system_prompt="", max_tokens=256, adapter=adapter,
    )
    breaker = _CircuitBreaker(max_identical=3)
    for _ in range(3):
        breaker.record_error("bash", "Invalid input for bash: command Field required")
    return ctx, adapter, breaker


def test_a_cloud_trip_keeps_tier_off_and_still_gets_its_retry():
    ctx, adapter, breaker = _tripped("qwen")

    result = breaker.diagnose_and_recover(context=ctx, tool_name="bash", intended_action="{}")

    assert result.recovered is True
    assert result.recovery_method == "retry:cloud_tier_off"
    assert result.new_tier is None
    assert ctx.adapter is adapter, "the run's adapter was replaced"
    assert ctx.adapter.tier == "off"
    assert breaker.recovery_attempted is True
    # Counters cleared, so the loop can continue.
    assert breaker._identical_count == 0 and breaker._any_error_count == 0
    assert "tier_bump" not in result.diagnostic_message


@pytest.mark.parametrize("provider_name", ["llama_cpp", None], ids=["local", "unknown"])
def test_a_local_or_unknown_provider_at_tier_off_still_bumps(provider_name):
    ctx, adapter, breaker = _tripped(provider_name)

    result = breaker.diagnose_and_recover(context=ctx, tool_name="bash", intended_action="{}")

    assert result.recovered is True
    assert result.recovery_method == "tier_bump:off->light"
    assert ctx.adapter.tier == "light"
    assert adapter.tier == "off", "the shared adapter was mutated"


@pytest.fixture(autouse=True)
def _no_global_handle(monkeypatch):
    import prometheus.telemetry.tracker as tracker

    monkeypatch.setattr(tracker, "_telemetry_singleton", None)
