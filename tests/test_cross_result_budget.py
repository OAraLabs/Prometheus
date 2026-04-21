"""Tests for Feature 2: Cross-Result Token Budget."""

import pytest
from unittest.mock import MagicMock

from prometheus.engine.agent_loop import _apply_cross_result_budget, LoopContext
from prometheus.engine.messages import ToolResultBlock, ToolUseBlock


def _make_context(budget: int = 8000) -> LoopContext:
    ctx = MagicMock(spec=LoopContext)
    ctx.tool_results_turn_budget = budget
    ctx.tool_registry = None
    return ctx


def _make_tc(name: str, input_: dict | None = None) -> ToolUseBlock:
    return ToolUseBlock(name=name, id=f"id_{name}", input=input_ or {})


def _make_result(tool_use_id: str, content: str, is_error: bool = False) -> ToolResultBlock:
    return ToolResultBlock(tool_use_id=tool_use_id, content=content, is_error=is_error)


class TestCrossResultBudget:
    def test_single_result_under_budget(self):
        ctx = _make_context(budget=8000)
        tcs = [_make_tc("bash")]
        results = [_make_result("id_bash", "short output")]
        out = _apply_cross_result_budget(ctx, tcs, results)
        assert out[0].content == "short output"

    def test_multiple_under_budget(self):
        ctx = _make_context(budget=8000)
        tcs = [_make_tc("bash"), _make_tc("grep")]
        results = [
            _make_result("id_bash", "a" * 100),
            _make_result("id_grep", "b" * 100),
        ]
        out = _apply_cross_result_budget(ctx, tcs, results)
        assert out[0].content == "a" * 100
        assert out[1].content == "b" * 100

    def test_multiple_over_budget_truncates(self):
        ctx = _make_context(budget=500)  # ~500 tokens = ~2000 chars
        tcs = [_make_tc("bash"), _make_tc("grep")]
        results = [
            _make_result("id_bash", "x" * 5000),
            _make_result("id_grep", "y" * 5000),
        ]
        out = _apply_cross_result_budget(ctx, tcs, results)
        # At least one should be truncated
        assert any("[truncated" in r.content for r in out)

    def test_error_results_not_truncated(self):
        ctx = _make_context(budget=100)
        tcs = [_make_tc("bash")]
        results = [_make_result("id_bash", "error: something", is_error=True)]
        out = _apply_cross_result_budget(ctx, tcs, results)
        assert out[0].content == "error: something"

    def test_zero_budget_passthrough(self):
        ctx = _make_context(budget=0)
        tcs = [_make_tc("bash")]
        results = [_make_result("id_bash", "x" * 50000)]
        out = _apply_cross_result_budget(ctx, tcs, results)
        assert out[0].content == "x" * 50000

    def test_read_only_truncated_first(self):
        """Read-only tools should be truncated before mutating tools."""
        ctx = _make_context(budget=200)
        # Set up registry with read-only tool
        registry = MagicMock()
        ro_tool = MagicMock()
        ro_tool.is_read_only.return_value = True
        mut_tool = MagicMock()
        mut_tool.is_read_only.return_value = False
        registry.get.side_effect = lambda n: ro_tool if n == "grep" else mut_tool
        ctx.tool_registry = registry

        tcs = [_make_tc("grep"), _make_tc("bash")]
        results = [
            _make_result("id_grep", "g" * 5000),
            _make_result("id_bash", "b" * 5000),
        ]
        out = _apply_cross_result_budget(ctx, tcs, results)
        # grep (read-only) should be truncated more aggressively
        assert len(out[0].content) <= len(out[1].content)
