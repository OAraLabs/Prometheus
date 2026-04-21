"""Tests for Feature 5: Adaptive Per-Tool Strictness."""

import pytest

from prometheus.adapter import ModelAdapter
from prometheus.adapter.validator import Strictness
from prometheus.adapter.formatter import QwenFormatter
from prometheus.tools.base import ToolRegistry
from prometheus.tools.builtin.bash import BashTool
from prometheus.tools.builtin.file_read import FileReadTool


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(BashTool())
    reg.register(FileReadTool())
    return reg


class TestAdaptiveStrictness:
    def test_disabled_by_default(self):
        adapter = ModelAdapter(strictness="MEDIUM")
        assert adapter._adaptive_strictness is False

    def test_enabled_records_calls(self):
        adapter = ModelAdapter(
            strictness="MEDIUM",
            adaptive_strictness=True,
        )
        adapter.record_tool_call("bash", True)
        adapter.record_tool_call("bash", True)
        assert len(adapter._tool_call_history["bash"]) == 2

    def test_no_bump_above_threshold(self):
        adapter = ModelAdapter(
            strictness="MEDIUM",
            adaptive_strictness=True,
            strictness_threshold=0.5,
        )
        # 9 successes, 1 failure = 90% — above 50% threshold
        for _ in range(9):
            adapter.record_tool_call("bash", True)
        adapter.record_tool_call("bash", False)
        assert "bash" not in adapter._tool_strictness

    def test_bump_below_threshold(self):
        adapter = ModelAdapter(
            strictness="MEDIUM",
            adaptive_strictness=True,
            strictness_threshold=0.8,
        )
        # 6 successes, 4 failures = 60% — below 80% threshold
        for _ in range(6):
            adapter.record_tool_call("bash", True)
        for _ in range(4):
            adapter.record_tool_call("bash", False)
        assert adapter._tool_strictness.get("bash") == Strictness.STRICT

    def test_window_limits_history(self):
        adapter = ModelAdapter(
            strictness="MEDIUM",
            adaptive_strictness=True,
            strictness_window=20,
        )
        for _ in range(30):
            adapter.record_tool_call("bash", True)
        assert len(adapter._tool_call_history["bash"]) == 20

    def test_manual_override_takes_precedence(self):
        adapter = ModelAdapter(
            strictness="NONE",
            adaptive_strictness=True,
        )
        adapter.set_tool_strictness("bash", Strictness.STRICT)
        assert adapter.get_effective_strictness("bash") == Strictness.STRICT

    def test_get_effective_falls_back_to_base(self):
        adapter = ModelAdapter(strictness="MEDIUM", adaptive_strictness=True)
        assert adapter.get_effective_strictness("bash") == Strictness.MEDIUM

    def test_validate_and_repair_uses_per_tool_strictness(self, registry):
        adapter = ModelAdapter(
            formatter=QwenFormatter(),
            strictness="NONE",
            adaptive_strictness=True,
        )
        # With NONE strictness, validation passes everything
        name, inp, repairs = adapter.validate_and_repair("bash", {"command": "ls"}, registry)
        assert name == "bash"

        # Now override bash to MEDIUM — should still pass valid calls
        adapter.set_tool_strictness("bash", Strictness.MEDIUM)
        name, inp, repairs = adapter.validate_and_repair("bash", {"command": "ls"}, registry)
        assert name == "bash"

    def test_validate_records_success_on_valid(self, registry):
        adapter = ModelAdapter(
            formatter=QwenFormatter(),
            strictness="MEDIUM",
            adaptive_strictness=True,
        )
        adapter.validate_and_repair("bash", {"command": "ls"}, registry)
        assert adapter._tool_call_history["bash"] == [True]

    def test_validate_records_failure_on_invalid(self, registry):
        adapter = ModelAdapter(
            formatter=QwenFormatter(),
            strictness="MEDIUM",
            adaptive_strictness=True,
        )
        with pytest.raises(ValueError):
            adapter.validate_and_repair("nonexistent_tool", {}, registry)
        assert adapter._tool_call_history["nonexistent_tool"] == [False]

    def test_none_to_medium_bump(self):
        adapter = ModelAdapter(
            strictness="NONE",
            adaptive_strictness=True,
            strictness_threshold=0.8,
        )
        for _ in range(7):
            adapter.record_tool_call("bash", True)
        for _ in range(3):
            adapter.record_tool_call("bash", False)
        assert adapter._tool_strictness.get("bash") == Strictness.MEDIUM
