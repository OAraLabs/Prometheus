"""RetryEngine budget must reset on a successful call (audit H5).

Before the fix, ``RetryEngine._state`` counted validation failures per tool
name for the entire daemon lifetime and ``reset()`` was never called from
anywhere. On tier light (``max_retries=1``) the *second-ever* failure for a
tool — days apart, hundreds of successes between — permanently exhausted that
tool's budget, after which every failure returned ABORT (or, with escalation
on, spawned a paid cloud subagent per failure).

The fix resets a tool's failure streak whenever a call for it validates
successfully, so the budget is per-streak, not per-lifetime. These tests pin
both halves: the streak resets across a success, and the budget still aborts
*within* a single streak.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from prometheus.adapter import ModelAdapter, RetryAction
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult, ToolRegistry


class _RunnerInput(BaseModel):
    command: str  # required — an empty input fails validation


class _Runner(BaseTool):
    name = "runner"
    description = "needs a command"
    input_model = _RunnerInput

    def is_read_only(self, arguments: _RunnerInput) -> bool:
        return False

    async def execute(self, arguments: _RunnerInput, context: ToolExecutionContext) -> ToolResult:
        return ToolResult(output="ran")


def _adapter(max_retries: int = 1):
    reg = ToolRegistry()
    reg.register(_Runner())
    # strictness MEDIUM + tier full so _do_validate_and_repair actually runs
    # (tier "off" short-circuits validation entirely).
    adapter = ModelAdapter(strictness="MEDIUM", max_retries=max_retries)
    return reg, adapter


def test_successful_call_resets_retry_streak():
    reg, adapter = _adapter(max_retries=1)

    # Streak 1: a bad call fails validation; the first retry is within budget.
    with pytest.raises(ValueError):
        adapter.validate_and_repair("runner", {}, reg)
    action1, _ = adapter.handle_retry("runner", "missing command", reg)
    assert action1 == RetryAction.RETRY

    # A successful call clears this tool's streak.
    name, _inp, _repairs = adapter.validate_and_repair("runner", {"command": "ls"}, reg)
    assert name == "runner"

    # Streak 2 must start fresh — RETRY again, NOT ABORT. Pre-fix this was
    # ABORT because the lifetime counter was already at the budget head.
    with pytest.raises(ValueError):
        adapter.validate_and_repair("runner", {}, reg)
    action2, _ = adapter.handle_retry("runner", "missing command", reg)
    assert action2 == RetryAction.RETRY


def test_budget_still_aborts_within_a_single_streak():
    # Regression guard: the reset must NOT defang the budget within one streak.
    reg, adapter = _adapter(max_retries=1)

    with pytest.raises(ValueError):
        adapter.validate_and_repair("runner", {}, reg)
    a1, _ = adapter.handle_retry("runner", "missing command", reg)
    assert a1 == RetryAction.RETRY

    # No successful call between failures → the budget is exhausted.
    with pytest.raises(ValueError):
        adapter.validate_and_repair("runner", {}, reg)
    a2, _ = adapter.handle_retry("runner", "missing command", reg)
    assert a2 == RetryAction.ABORT


def test_reset_is_per_tool():
    # A success for one tool must not reset another tool's streak.
    reg = ToolRegistry()
    reg.register(_Runner())

    class _Other(BaseTool):
        name = "other"
        description = "needs a command too"
        input_model = _RunnerInput

        def is_read_only(self, arguments):
            return False

        async def execute(self, arguments, context):
            return ToolResult(output="ok")

    reg.register(_Other())
    adapter = ModelAdapter(strictness="MEDIUM", max_retries=1)

    # Build up "other"'s streak to the budget head.
    with pytest.raises(ValueError):
        adapter.validate_and_repair("other", {}, reg)
    assert adapter.handle_retry("other", "e", reg)[0] == RetryAction.RETRY

    # A success for "runner" must NOT reset "other".
    adapter.validate_and_repair("runner", {"command": "ls"}, reg)

    with pytest.raises(ValueError):
        adapter.validate_and_repair("other", {}, reg)
    assert adapter.handle_retry("other", "e", reg)[0] == RetryAction.ABORT
