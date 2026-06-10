"""Tier-1 loop-hardening batch (audit M5, M2, M1, M4).

- M5: a hung tool is bounded by a per-tool timeout, not allowed to freeze the
  turn/session.
- M2: adapter repairs (incl. silent fuzzy tool-name renames) reach telemetry
  and the logs instead of vanishing.
- M1: synthetic ``_loop_transition`` rows no longer inflate /health and report
  tool-call stats.
- M4: the router evaluates the MOST RECENT user message, not the first.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import LoopContext, _execute_tool_call, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiTextDeltaEvent,
    ModelProvider,
)
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult


class _EmptyInput(BaseModel):
    pass


class _RecordingTelemetry:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record(self, **kw):  # noqa: ANN003
        self.calls.append(kw)


def _ctx(reg: ToolRegistry, tel=None, **kw) -> LoopContext:
    return LoopContext(
        provider=None,
        model="stub",
        system_prompt="",
        max_tokens=1024,
        tool_registry=reg,
        telemetry=tel,
        **kw,
    )


# ---------------------------------------------------------------------------
# M5 — per-tool execution timeout
# ---------------------------------------------------------------------------


class _SlowTool(BaseTool):
    name = "slow"
    description = "sleeps a long time"
    input_model = _EmptyInput

    async def execute(self, arguments, context):  # noqa: ANN001
        await asyncio.sleep(5.0)
        return ToolResult(output="done")


class _PatientTool(BaseTool):
    name = "patient"
    description = "sleeps a little, but is allowed to"
    input_model = _EmptyInput
    execution_timeout_seconds = 5.0  # per-tool override

    async def execute(self, arguments, context):  # noqa: ANN001
        await asyncio.sleep(0.05)
        return ToolResult(output="finished")


class TestToolTimeoutM5:
    def test_hung_tool_times_out_with_error_block_and_telemetry(self):
        reg = ToolRegistry()
        reg.register(_SlowTool())
        tel = _RecordingTelemetry()
        ctx = _ctx(reg, tel, tool_timeout_seconds=0.05)

        block = asyncio.run(_execute_tool_call(ctx, "slow", "tu_slow", {}))
        assert block.is_error
        assert block.tool_use_id == "tu_slow"  # correct correlation, not propagated
        assert "timed out" in block.content
        rec = next(c for c in tel.calls if c.get("tool_name") == "slow")
        assert rec["error_type"] == "tool_timeout"

    def test_per_tool_override_beats_the_context_default(self):
        # Context default would kill a 0.05s tool, but the tool overrides to 5s.
        reg = ToolRegistry()
        reg.register(_PatientTool())
        ctx = _ctx(reg, tool_timeout_seconds=0.01)

        block = asyncio.run(_execute_tool_call(ctx, "patient", "tu_p", {}))
        assert not block.is_error
        assert "finished" in block.content


# ---------------------------------------------------------------------------
# M2 — adapter repairs reach telemetry + logs
# ---------------------------------------------------------------------------


class _RunnerInput(BaseModel):
    command: str


class _Runner(BaseTool):
    name = "runner"
    description = "runs a command"
    input_model = _RunnerInput

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output="ran")


def _adapter_ctx(reg, tel):
    from prometheus.adapter import ModelAdapter

    return _ctx(reg, tel, adapter=ModelAdapter(strictness="MEDIUM"))  # tier full


class TestAdapterRepairsM2:
    def test_repairs_column_persists_and_is_queryable(self, tmp_path):
        tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
        tel.record(model="m", tool_name="bash", success=True, repairs=2)
        row = tel._conn.execute(
            "SELECT repairs FROM tool_calls WHERE tool_name = 'bash'"
        ).fetchone()
        assert row[0] == 2

    def test_fuzzy_rename_records_repair_count_and_warns(self, caplog):
        reg = ToolRegistry()
        reg.register(_Runner())
        tel = _RecordingTelemetry()
        ctx = _adapter_ctx(reg, tel)

        with caplog.at_level(logging.WARNING, logger="prometheus.engine.agent_loop"):
            # Model misnames the tool ("runer"); the adapter fuzzy-repairs it to
            # "runner" and executes it — previously logged/recorded nowhere.
            block = asyncio.run(
                _execute_tool_call(ctx, "runer", "tu_r", {"command": "ls"})
            )

        assert not block.is_error
        rec = next(c for c in tel.calls if c.get("tool_name") == "runner")
        assert rec.get("repairs", 0) >= 1
        assert any("repaired tool name" in r.message for r in caplog.records)

    def test_clean_call_records_zero_repairs(self):
        reg = ToolRegistry()
        reg.register(_Runner())
        tel = _RecordingTelemetry()
        ctx = _adapter_ctx(reg, tel)
        asyncio.run(_execute_tool_call(ctx, "runner", "tu_ok", {"command": "ls"}))
        rec = next(c for c in tel.calls if c.get("tool_name") == "runner")
        assert rec.get("repairs", 0) == 0


# ---------------------------------------------------------------------------
# M1 — synthetic _loop_transition rows excluded from stats
# ---------------------------------------------------------------------------


class TestLoopTransitionExclusionM1:
    def test_health_summary_does_not_double_count_loop_transitions(self, tmp_path):
        tel = ToolCallTelemetry(db_path=tmp_path / "h.db")
        # A real failure, its synthetic loop-echo, and a real success.
        tel.record(model="m", tool_name="bash", success=False, error_type="tool_error")
        tel.record(
            model="m", tool_name="_loop_transition", success=False,
            error_type="tool_error_retry",
        )
        tel.record(model="m", tool_name="bash", success=True)

        tc = tel.health_summary(since=0.0)["tool_calls"]
        assert tc["total"] == 2      # not 3 — the loop-echo is excluded
        assert tc["failures"] == 1   # the single real failure, not double-counted

    def test_report_excludes_loop_transition_tool(self, tmp_path):
        tel = ToolCallTelemetry(db_path=tmp_path / "r.db")
        tel.record(model="m", tool_name="bash", success=True)
        tel.record(model="m", tool_name="_loop_transition", success=True)
        rep = tel.report()
        assert "_loop_transition" not in rep["tools"]
        assert "bash" in rep["tools"]


# ---------------------------------------------------------------------------
# M4 — route on the most recent user message
# ---------------------------------------------------------------------------


class _OneShotProvider(ModelProvider):
    """Ends the turn immediately with a tool-call-free assistant message."""

    async def stream_message(self, request):  # noqa: ANN001
        msg = ConversationMessage(role="assistant", content=[TextBlock(text="done")])
        yield ApiTextDeltaEvent(text="done")
        yield ApiMessageCompleteEvent(
            message=msg, usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


class _RecordingRouter:
    def __init__(self) -> None:
        self.routed: list[str] = []

    def route(self, text, context=None):  # noqa: ANN001
        self.routed.append(text)
        # No swap (all None) so the loop runs on the supplied provider.
        return SimpleNamespace(
            reason="primary", provider=None, adapter=None,
            model_name=None, provider_name=None,
        )


class TestRouteOnLatestMessageM4:
    def test_router_sees_the_latest_user_message_not_the_first(self):
        provider = _OneShotProvider()
        router = _RecordingRouter()
        ctx = LoopContext(
            provider=provider,
            model="test",
            system_prompt="- Model: test (provider: test)",
            max_tokens=256,
            model_router=router,
        )
        messages = [
            ConversationMessage.from_user_text("OLD stale first request"),
            ConversationMessage(role="assistant", content=[TextBlock(text="ok")]),
            ConversationMessage.from_user_text("NEW current request"),
        ]

        async def _run():
            async for _ in run_loop(ctx, messages):
                pass

        asyncio.run(_run())
        assert router.routed == ["NEW current request"]
