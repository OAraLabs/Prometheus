"""Regression tests for the 2026-06-21 "frozen agent" incident.

Root cause chain (a "check GPU models" request that ran an unbounded
``find /home/will``):

1. The bash tool killed only ``/bin/bash`` on timeout/cancel, orphaning the
   ``find … | grep …`` pipeline — which kept thrashing the disk for minutes
   after the turn was abandoned.
2. The agent loop re-issued the byte-identical doomed command every iteration,
   paying the full tool timeout on each one, so the turn looked frozen.

These tests pin both fixes:
  - BashTool kills the whole process GROUP on its own timeout AND on outer
    cancellation (the agent-loop wait_for), leaving no orphans.
  - run_loop refuses to re-execute a tool call whose exact (name, input)
    already failed this turn, instead of running it max_tool_iterations times.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import (
    ConversationMessage,
    ToolUseBlock,
)
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult
from prometheus.tools.builtin.bash import BashTool, BashToolInput


# ---------------------------------------------------------------------------
# Bug 1 — bash tool kills the whole process group (no orphans)
# ---------------------------------------------------------------------------

# A backgrounded child that creates ``marker`` 3s from now. It only survives to
# write the file if the parent shell's death failed to take the group with it.
def _orphan_probe_cmd(marker) -> str:
    return f"( sleep 3 && touch {marker} ) & echo bg=$!; sleep 3"


def test_bash_timeout_kills_child_process_group(tmp_path):
    marker = tmp_path / "marker_timeout"
    tool = BashTool()
    ctx = ToolExecutionContext(cwd=tmp_path)

    res = asyncio.run(
        tool.execute(BashToolInput(command=_orphan_probe_cmd(marker), timeout_seconds=1), ctx)
    )

    assert res.is_error
    assert "timed out" in res.output
    # If the backgrounded child were orphaned it would touch the marker at ~3s.
    time.sleep(4)
    assert not marker.exists(), "background child survived bash timeout — process group not killed"


def test_bash_outer_cancellation_kills_child_process_group(tmp_path):
    """The exact incident shape: the tool's own timeout is generous (30s) but
    the agent loop cancels execute() after 1s. The subprocess must still die."""
    marker = tmp_path / "marker_cancel"
    tool = BashTool()
    ctx = ToolExecutionContext(cwd=tmp_path)

    async def _run():
        await asyncio.wait_for(
            tool.execute(BashToolInput(command=_orphan_probe_cmd(marker), timeout_seconds=30), ctx),
            timeout=1.0,  # outer cancel, like LoopContext.tool_timeout_seconds
        )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(_run())

    time.sleep(4)
    assert not marker.exists(), "background child survived outer cancellation — group not killed"


def test_bash_normal_command_still_works(tmp_path):
    tool = BashTool()
    ctx = ToolExecutionContext(cwd=tmp_path)
    res = asyncio.run(tool.execute(BashToolInput(command="echo hello-prometheus"), ctx))
    assert not res.is_error
    assert "hello-prometheus" in res.output


# ---------------------------------------------------------------------------
# Bug 2 — run_loop refuses to re-run an identical failing call all turn
# ---------------------------------------------------------------------------


class _EmptyInput(BaseModel):
    pass


class _AlwaysFailTool(BaseTool):
    name = "failer"
    description = "always errors"
    input_model = _EmptyInput

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, arguments, context):  # noqa: ANN001
        self.calls += 1
        return ToolResult(output="boom", is_error=True)


class _RepeatProvider(ModelProvider):
    """Always emits the SAME failing tool call (byte-identical input)."""

    async def stream_message(self, request):  # noqa: ANN001
        msg = ConversationMessage(
            role="assistant",
            content=[ToolUseBlock(name="failer", input={})],
        )
        yield ApiMessageCompleteEvent(
            message=msg,
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="tool_calls",
        )


def test_repeat_guard_stops_reexecuting_identical_failing_call():
    reg = ToolRegistry()
    tool = _AlwaysFailTool()
    reg.register(tool)
    ctx = LoopContext(
        provider=_RepeatProvider(),
        model="stub",
        system_prompt="",
        max_tokens=256,
        tool_registry=reg,
    )
    messages = [ConversationMessage.from_user_text("which qwen models are on the 4090?")]

    async def _run():
        async for _ in run_loop(ctx, messages):
            pass

    asyncio.run(_run())

    # Without the guard the tool would execute up to max_tool_iterations (25)
    # times. With it, real execution stops at _REPEAT_FAIL_LIMIT (2) and every
    # later identical call is blocked without running.
    assert tool.calls <= 2, f"repeat guard didn't fire — tool executed {tool.calls}×"
