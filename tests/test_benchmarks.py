"""Tests for the benchmarks module (Sprint 8)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prometheus.benchmarks.suite import (
    BenchmarkSuite,
    TestCase,
    TestTier,
    load_suite,
)
from prometheus.benchmarks.runner import (
    BenchmarkRunner,
    Score,
    ScoreResult,
    _score_result,
    print_report,
)
from prometheus.coordinator.subagent import SubagentResult
from prometheus.engine.agent_loop import RunResult
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock


# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------


class TestTestCase:
    def test_defaults(self):
        tc = TestCase(id="t1", name="test", tier=1, prompt="do something")
        assert tc.max_turns == 10
        assert tc.expected_tools == []
        assert tc.tags == []
        assert tc.setup_commands == []
        assert tc.teardown_commands == []

    def test_full(self):
        tc = TestCase(
            id="t1_full",
            name="Full test",
            tier=1,
            prompt="Do the thing",
            expected_tools=["bash"],
            expected_output_contains=["hello"],
            expected_file_exists=["/tmp/test.txt"],
            max_turns=5,
            tags=["bash"],
        )
        assert tc.expected_tools == ["bash"]
        assert tc.tier == 1


class TestTestTier:
    def test_tier_values(self):
        assert TestTier.TIER_1 == 1
        assert TestTier.TIER_2 == 2


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------


class TestBenchmarkSuite:
    def test_load_builtin_all(self):
        suite = load_suite()
        assert len(suite) >= 25  # 20+ tier1 + 5 tier2

    def test_load_tier1(self):
        suite = load_suite(tier=1)
        for c in suite.cases:
            assert c.tier == 1
        assert len(suite) >= 20

    def test_load_tier2(self):
        suite = load_suite(tier=2)
        for c in suite.cases:
            assert c.tier == 2
        assert len(suite) == 5

    def test_filter_tags(self):
        suite = load_suite()
        bash_tests = suite.filter_tags(["bash"])
        assert len(bash_tests) > 0
        for tc in bash_tests:
            assert "bash" in tc.tags

    def test_get_by_id(self):
        suite = load_suite()
        tc = suite.get("t1_bash_echo")
        assert tc is not None
        assert tc.name == "Bash echo"

    def test_get_missing(self):
        suite = load_suite()
        assert suite.get("nonexistent") is None

    def test_add(self):
        suite = BenchmarkSuite()
        tc = TestCase(id="custom", name="Custom", tier=1, prompt="test")
        suite.add(tc)
        assert len(suite) == 1
        assert suite.get("custom") is tc

    def test_yaml_roundtrip(self):
        suite = BenchmarkSuite([
            TestCase(id="a", name="A", tier=1, prompt="do A", expected_tools=["bash"]),
            TestCase(id="b", name="B", tier=2, prompt="do B", tags=["multi_step"]),
        ])
        yaml_str = suite.to_yaml()
        assert "id: a" in yaml_str
        assert "id: b" in yaml_str

        loaded = BenchmarkSuite.from_yaml(yaml_str)
        assert len(loaded) == 2
        assert loaded.get("a").expected_tools == ["bash"]
        assert loaded.get("b").tags == ["multi_step"]

    def test_from_yaml_empty(self):
        suite = BenchmarkSuite.from_yaml("")
        assert len(suite) == 0

    def test_unique_ids(self):
        """All built-in test case IDs should be unique."""
        suite = load_suite()
        ids = [c.id for c in suite.cases]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def _make_result(self, text: str, tool_names: list[str] | None = None) -> RunResult:
        """Build a RunResult with optional tool use blocks."""
        content = []
        if tool_names:
            for name in tool_names:
                content.append(ToolUseBlock(name=name, input={}))
        content.append(TextBlock(text=text))
        msg = ConversationMessage(role="assistant", content=content)
        return RunResult(text=text, messages=[msg], turns=1)

    def test_score_success_tools_and_output(self):
        case = TestCase(
            id="test",
            name="Test",
            tier=1,
            prompt="test",
            expected_tools=["bash"],
            expected_output_contains=["hello"],
        )
        result = self._make_result("hello world", ["bash"])
        sr = _score_result(case, result, Path("/tmp"))
        assert sr.score == Score.SUCCESS
        assert "2/2" in sr.details

    def test_score_partial_missing_output(self):
        case = TestCase(
            id="test",
            name="Test",
            tier=1,
            prompt="test",
            expected_tools=["bash"],
            expected_output_contains=["missing_string"],
        )
        result = self._make_result("hello world", ["bash"])
        sr = _score_result(case, result, Path("/tmp"))
        assert sr.score == Score.PARTIAL
        assert "1/2" in sr.details

    def test_score_fail_nothing_matches(self):
        case = TestCase(
            id="test",
            name="Test",
            tier=1,
            prompt="test",
            expected_tools=["grep"],
            expected_output_contains=["xyz_not_present"],
        )
        result = self._make_result("hello world", ["bash"])
        sr = _score_result(case, result, Path("/tmp"))
        assert sr.score == Score.FAIL

    def test_score_no_expectations(self):
        """No expectations + tool called = SUCCESS."""
        case = TestCase(id="test", name="Test", tier=1, prompt="test")
        result = self._make_result("done", ["bash"])
        sr = _score_result(case, result, Path("/tmp"))
        assert sr.score == Score.SUCCESS

    def test_score_no_expectations_no_tools(self):
        """No expectations + no tools = PARTIAL."""
        case = TestCase(id="test", name="Test", tier=1, prompt="test")
        result = self._make_result("done")
        sr = _score_result(case, result, Path("/tmp"))
        assert sr.score == Score.PARTIAL

    def test_score_file_exists(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content")
        case = TestCase(
            id="test",
            name="Test",
            tier=1,
            prompt="test",
            expected_file_exists=[str(f)],
        )
        result = self._make_result("done", ["write_file"])
        sr = _score_result(case, result, Path("/tmp"))
        assert sr.score == Score.SUCCESS

    def test_score_file_contains(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        case = TestCase(
            id="test",
            name="Test",
            tier=1,
            prompt="test",
            expected_file_contains={str(f): "hello"},
        )
        result = self._make_result("done", ["write_file"])
        sr = _score_result(case, result, Path("/tmp"))
        assert sr.score == Score.SUCCESS

    def test_score_output_not_contains(self):
        case = TestCase(
            id="test",
            name="Test",
            tier=1,
            prompt="test",
            expected_output_not_contains=["error"],
        )
        result = self._make_result("all good", ["bash"])
        sr = _score_result(case, result, Path("/tmp"))
        assert sr.score == Score.SUCCESS

    def test_score_output_not_contains_fail(self):
        case = TestCase(
            id="test",
            name="Test",
            tier=1,
            prompt="test",
            expected_output_not_contains=["error"],
        )
        result = self._make_result("error occurred", ["bash"])
        sr = _score_result(case, result, Path("/tmp"))
        assert sr.score == Score.FAIL


# ---------------------------------------------------------------------------
# ScoreResult
# ---------------------------------------------------------------------------


class TestScoreResult:
    def test_defaults(self):
        sr = ScoreResult(case_id="t1", case_name="test", score=Score.SUCCESS)
        assert sr.turns == 0
        assert sr.latency_ms == 0.0
        assert sr.tool_calls == []

    def test_score_enum(self):
        assert Score.SUCCESS.value == "SUCCESS"
        assert Score.PARTIAL.value == "PARTIAL"
        assert Score.RETRY_SUCCESS.value == "RETRY_SUCCESS"
        assert Score.FAIL.value == "FAIL"
        assert Score.CRASH.value == "CRASH"


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


class TestBenchmarkRunner:
    @pytest.mark.asyncio
    async def test_run_case_crash(self):
        """Runner should catch exceptions and return CRASH."""
        mock_provider = MagicMock()

        from prometheus.tools.base import ToolRegistry
        registry = ToolRegistry()

        runner = BenchmarkRunner(
            provider=mock_provider,
            tool_registry=registry,
            model="test-model",
        )

        case = TestCase(
            id="crash_test",
            name="Crash test",
            tier=1,
            prompt="Do something",
            max_turns=2,
        )

        with patch("prometheus.engine.agent_loop.AgentLoop.run_async", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = RuntimeError("Provider exploded")
            result = await runner.run_case(case)
            assert result.score == Score.CRASH
            assert "Provider exploded" in result.details

    @pytest.mark.asyncio
    async def test_run_case_success(self):
        """Runner should score SUCCESS when expectations are met."""
        mock_provider = MagicMock()

        from prometheus.tools.base import ToolRegistry
        registry = ToolRegistry()

        runner = BenchmarkRunner(
            provider=mock_provider,
            tool_registry=registry,
            model="test-model",
        )

        case = TestCase(
            id="success_test",
            name="Success test",
            tier=1,
            prompt="Do something",
            expected_output_contains=["hello"],
        )

        mock_result = RunResult(
            text="hello world",
            messages=[
                ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text="hello world")],
                )
            ],
            turns=1,
        )

        with patch("prometheus.engine.agent_loop.AgentLoop.run_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            result = await runner.run_case(case)
            assert result.score == Score.SUCCESS

    def test_print_report(self, capsys):
        """print_report should output formatted text."""
        results = [
            ScoreResult(
                case_id="t1_test",
                case_name="Test",
                score=Score.SUCCESS,
                turns=1,
                latency_ms=100.0,
                details="1/1 checks passed",
            ),
            ScoreResult(
                case_id="t1_fail",
                case_name="Fail",
                score=Score.FAIL,
                turns=2,
                latency_ms=200.0,
                details="0/1 checks passed",
            ),
        ]
        print_report(results)
        captured = capsys.readouterr()
        assert "BENCHMARK RESULTS" in captured.out
        assert "PASS" in captured.out
        assert "FAIL" in captured.out
        assert "Total: 2" in captured.out


# ---------------------------------------------------------------------------
# AgentTool
# ---------------------------------------------------------------------------


class TestAgentTool:
    def test_schema(self):
        from prometheus.tools.builtin.agent import AgentTool

        tool = AgentTool()
        assert tool.name == "Agent"
        schema = tool.to_api_schema()
        assert schema["name"] == "Agent"
        assert "input_schema" in schema

    @pytest.mark.asyncio
    async def test_execute_no_spawner(self):
        from prometheus.tools.builtin.agent import AgentTool, AgentToolInput
        from prometheus.tools.base import ToolExecutionContext

        tool = AgentTool()
        args = AgentToolInput(
            description="Test",
            prompt="Do something",
        )
        ctx = ToolExecutionContext(cwd=Path("/tmp"), metadata={})
        result = await tool.execute(args, ctx)
        assert result.is_error is True
        assert "SubagentSpawner not configured" in result.output

    @pytest.mark.asyncio
    async def test_execute_with_spawner(self):
        from prometheus.tools.builtin.agent import AgentTool, AgentToolInput
        from prometheus.tools.base import ToolExecutionContext

        tool = AgentTool()
        mock_spawner = AsyncMock()
        mock_spawner.spawn.return_value = SubagentResult(
            agent_id="sub_test",
            agent_type="general-purpose",
            text="Result from subagent",
            turns=1,
        )

        args = AgentToolInput(
            description="Test task",
            prompt="Do something useful",
        )
        ctx = ToolExecutionContext(
            cwd=Path("/tmp"),
            metadata={"subagent_spawner": mock_spawner},
        )
        result = await tool.execute(args, ctx)
        assert result.is_error is False
        assert "Result from subagent" in result.output
        mock_spawner.spawn.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_spawner_failure(self):
        from prometheus.tools.builtin.agent import AgentTool, AgentToolInput
        from prometheus.tools.base import ToolExecutionContext

        tool = AgentTool()
        mock_spawner = AsyncMock()
        mock_spawner.spawn.return_value = SubagentResult(
            agent_id="sub_fail",
            agent_type="general-purpose",
            text="",
            success=False,
            error="Timeout",
        )

        args = AgentToolInput(
            description="Test fail",
            prompt="This will fail",
        )
        ctx = ToolExecutionContext(
            cwd=Path("/tmp"),
            metadata={"subagent_spawner": mock_spawner},
        )
        result = await tool.execute(args, ctx)
        assert result.is_error is True
        assert "Timeout" in result.output
