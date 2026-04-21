"""Tests for parallel tool dispatch in the agent loop.

Covers: read-only parallelism, mutating sequential order, mixed dispatch,
error isolation, and hook integration.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic import BaseModel

from prometheus.engine.agent_loop import (
    LoopContext,
    _dispatch_tool_calls,
    _is_tool_read_only,
)
from prometheus.engine.messages import ToolResultBlock, ToolUseBlock
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult, ToolRegistry


# ======================================================================
# Test helpers
# ======================================================================


class DummyInput(BaseModel):
    """Minimal input model for test tools."""
    pass


class ReadOnlyTool(BaseTool):
    """A tool that declares itself read-only."""

    name = "read_tool"
    description = "A read-only tool"
    input_model = DummyInput

    def __init__(self, name: str = "read_tool", *, delay: float = 0):
        self.name = name
        self._delay = delay

    def is_read_only(self, arguments: DummyInput) -> bool:
        return True

    async def execute(self, arguments: DummyInput, context: ToolExecutionContext) -> ToolResult:
        if self._delay:
            await asyncio.sleep(self._delay)
        return ToolResult(output=f"{self.name} result")


class MutatingTool(BaseTool):
    """A tool that mutates state (not read-only)."""

    name = "mutate_tool"
    description = "A mutating tool"
    input_model = DummyInput

    def __init__(self, name: str = "mutate_tool", *, execution_log: list | None = None):
        self.name = name
        self._log = execution_log

    def is_read_only(self, arguments: DummyInput) -> bool:
        return False

    async def execute(self, arguments: DummyInput, context: ToolExecutionContext) -> ToolResult:
        if self._log is not None:
            self._log.append(self.name)
        return ToolResult(output=f"{self.name} result")


class FailingTool(BaseTool):
    """A read-only tool that raises during execution."""

    name = "failing_tool"
    description = "A tool that fails"
    input_model = DummyInput

    def is_read_only(self, arguments: DummyInput) -> bool:
        return True

    async def execute(self, arguments: DummyInput, context: ToolExecutionContext) -> ToolResult:
        raise RuntimeError("tool exploded")


def _make_context(tools: list[BaseTool], **kwargs) -> LoopContext:
    """Build a LoopContext with a tool registry from the given tools."""
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    return LoopContext(
        provider=MagicMock(),
        model="test",
        system_prompt="",
        max_tokens=100,
        tool_registry=registry,
        **kwargs,
    )


def _tc(name: str, tool_id: str = "") -> ToolUseBlock:
    """Shorthand for creating a ToolUseBlock."""
    return ToolUseBlock(id=tool_id or f"t_{name}", name=name, input={})


# ======================================================================
# _is_tool_read_only
# ======================================================================


class TestIsToolReadOnly:

    def test_read_only_method_returns_true(self):
        tool = ReadOnlyTool()
        assert _is_tool_read_only(tool, {}) is True

    def test_mutating_method_returns_false(self):
        tool = MutatingTool()
        assert _is_tool_read_only(tool, {}) is False

    def test_missing_method_returns_false(self):
        tool = MagicMock(spec=[])  # No is_read_only attribute
        assert _is_tool_read_only(tool, {}) is False


# ======================================================================
# Single tool call
# ======================================================================


class TestSingleToolCall:

    @pytest.mark.asyncio
    async def test_single_call_executes_normally(self):
        tool = ReadOnlyTool()
        ctx = _make_context([tool])
        results = await _dispatch_tool_calls(ctx, [_tc("read_tool")])
        assert len(results) == 1
        assert results[0].content == "read_tool result"
        assert not results[0].is_error


# ======================================================================
# Multiple read-only tools
# ======================================================================


class TestParallelReadOnly:

    @pytest.mark.asyncio
    async def test_multiple_read_only_run_in_parallel(self):
        """Three read-only tools with delays should finish faster than sequential."""
        t1 = ReadOnlyTool("read_a", delay=0.05)
        t2 = ReadOnlyTool("read_b", delay=0.05)
        t3 = ReadOnlyTool("read_c", delay=0.05)
        ctx = _make_context([t1, t2, t3])

        start = time.monotonic()
        results = await _dispatch_tool_calls(
            ctx, [_tc("read_a"), _tc("read_b"), _tc("read_c")]
        )
        elapsed = time.monotonic() - start

        assert len(results) == 3
        # If sequential, would take ~0.15s; parallel should be ~0.05s
        assert elapsed < 0.12, f"Expected parallel execution, took {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_results_preserve_original_order(self):
        """Even though parallel, results match input order."""
        t1 = ReadOnlyTool("alpha", delay=0.03)
        t2 = ReadOnlyTool("beta", delay=0.01)  # Finishes first
        ctx = _make_context([t1, t2])

        results = await _dispatch_tool_calls(ctx, [_tc("alpha"), _tc("beta")])
        assert results[0].content == "alpha result"
        assert results[1].content == "beta result"


# ======================================================================
# Mixed read-only + mutating
# ======================================================================


class TestMixedDispatch:

    @pytest.mark.asyncio
    async def test_read_only_before_mutating(self):
        """Read-only tools run first (parallel), then mutating (sequential)."""
        execution_order = []

        class TrackedReadTool(BaseTool):
            name = "tracked_read"
            description = "tracked"
            input_model = DummyInput

            def is_read_only(self, arguments):
                return True

            async def execute(self, arguments, context):
                execution_order.append("read")
                return ToolResult(output="read")

        class TrackedMutateTool(BaseTool):
            name = "tracked_mutate"
            description = "tracked"
            input_model = DummyInput

            def is_read_only(self, arguments):
                return False

            async def execute(self, arguments, context):
                execution_order.append("mutate")
                return ToolResult(output="mutate")

        ctx = _make_context([TrackedReadTool(), TrackedMutateTool()])
        results = await _dispatch_tool_calls(
            ctx, [_tc("tracked_mutate"), _tc("tracked_read")]
        )
        assert len(results) == 2
        # Read should execute before mutate (parallel phase runs first)
        assert execution_order[0] == "read"
        assert execution_order[1] == "mutate"

    @pytest.mark.asyncio
    async def test_results_match_original_call_order(self):
        """Even with mixed dispatch, results align to original tool_calls order."""
        rt = ReadOnlyTool("reader")
        mt = MutatingTool("writer")
        ctx = _make_context([rt, mt])

        # writer first, reader second in the call list
        results = await _dispatch_tool_calls(
            ctx, [_tc("writer"), _tc("reader")]
        )
        assert results[0].content == "writer result"  # index 0 = writer
        assert results[1].content == "reader result"  # index 1 = reader

    @pytest.mark.asyncio
    async def test_multiple_mutating_run_sequentially(self):
        """Multiple mutating tools preserve execution order."""
        log: list[str] = []
        m1 = MutatingTool("first", execution_log=log)
        m2 = MutatingTool("second", execution_log=log)
        m3 = MutatingTool("third", execution_log=log)
        ctx = _make_context([m1, m2, m3])

        await _dispatch_tool_calls(
            ctx, [_tc("first"), _tc("second"), _tc("third")]
        )
        assert log == ["first", "second", "third"]


# ======================================================================
# Error isolation
# ======================================================================


class TestErrorIsolation:

    @pytest.mark.asyncio
    async def test_failed_parallel_tool_does_not_block_others(self):
        """One failing read-only tool shouldn't prevent others from completing."""
        good = ReadOnlyTool("good_tool")
        bad = FailingTool()
        bad.name = "failing_tool"
        ctx = _make_context([good, bad])

        results = await _dispatch_tool_calls(
            ctx, [_tc("good_tool"), _tc("failing_tool")]
        )
        assert len(results) == 2
        # good_tool should succeed
        good_result = [r for r in results if "good_tool" in r.content or not r.is_error]
        assert len(good_result) >= 1

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        """A tool not in the registry returns an error result."""
        ctx = _make_context([])
        results = await _dispatch_tool_calls(ctx, [_tc("nonexistent")])
        assert len(results) == 1
        assert results[0].is_error
        assert "Unknown tool" in results[0].content


# ======================================================================
# Hook integration
# ======================================================================


class TestHookIntegration:

    @pytest.mark.asyncio
    async def test_pre_hook_deny_blocks_individual_tool(self):
        """A denied pre-hook blocks that specific tool, not others."""
        from prometheus.hooks import HookEvent

        tool_a = ReadOnlyTool("allowed_tool")
        tool_b = ReadOnlyTool("denied_tool")

        # Mock hook executor that denies "denied_tool"
        mock_hook = AsyncMock()
        mock_result = MagicMock()
        mock_result.blocked = False
        mock_result.reason = None
        mock_denied = MagicMock()
        mock_denied.blocked = True
        mock_denied.reason = "Blocked by security gate"

        async def hook_side_effect(event, payload):
            if payload.get("tool_name") == "denied_tool":
                return mock_denied
            return mock_result

        mock_hook.execute = AsyncMock(side_effect=hook_side_effect)

        ctx = _make_context([tool_a, tool_b], hook_executor=mock_hook)
        results = await _dispatch_tool_calls(
            ctx, [_tc("allowed_tool"), _tc("denied_tool")]
        )
        assert len(results) == 2

        allowed = results[0]  # allowed_tool is index 0
        denied = results[1]   # denied_tool is index 1
        assert not allowed.is_error
        assert denied.is_error
        assert "Blocked" in denied.content
