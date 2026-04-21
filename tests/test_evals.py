"""Tests for Sprint 13: DeepEval + Phoenix evaluation suite.

All external dependencies (deepeval, phoenix, httpx) are mocked —
these tests run without any optional packages installed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prometheus.evals.golden_dataset import (
    GoldenTask,
    TaskTier,
    load_golden_dataset,
)
from prometheus.evals.judge import JUDGE_SCORE_SCHEMA, JudgeVerdict, PrometheusJudge
from prometheus.evals.metrics import (
    NoHallucinationMetric,
    TaskCompletionMetric,
    ToolUsageMetric,
)
from prometheus.evals.classifier import (
    FailureCategory,
    FailureSource,
    classify_failure,
)
from prometheus.evals.runner import EvalResult, EvalRunner, MetricScore, _SimpleTestCase
from prometheus.evals.trends import TrendTracker, TrendRow


# ---------------------------------------------------------------------------
# Golden Dataset
# ---------------------------------------------------------------------------


class TestGoldenDataset:
    def test_load_all(self):
        """Should return 26 tasks when skip_network=False."""
        tasks = load_golden_dataset(skip_network=False)
        assert len(tasks) == 26

    def test_load_skip_network(self):
        """Default should skip network-dependent tasks."""
        tasks = load_golden_dataset()
        network_tasks = [t for t in tasks if t.requires_network]
        assert len(network_tasks) == 0
        assert len(tasks) == 24  # 26 - 2 network tasks

    def test_load_tier1(self):
        """Should return 21 Tier 1 tasks."""
        tasks = load_golden_dataset(tier=1, skip_network=False)
        assert len(tasks) == 21
        assert all(t.tier == TaskTier.TIER_1 for t in tasks)

    def test_load_tier2(self):
        """Should return 5 Tier 2 tasks."""
        tasks = load_golden_dataset(tier=2, skip_network=False)
        assert len(tasks) == 5
        assert all(t.tier == TaskTier.TIER_2 for t in tasks)

    def test_unique_ids(self):
        """All task IDs should be unique."""
        tasks = load_golden_dataset(skip_network=False)
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_required_fields(self):
        """Every task should have id, name, input, expected_behavior."""
        for task in load_golden_dataset(skip_network=False):
            assert task.id, f"Task missing id"
            assert task.name, f"Task {task.id} missing name"
            assert task.input, f"Task {task.id} missing input"
            assert task.expected_behavior, f"Task {task.id} missing expected_behavior"
            assert task.tier in (TaskTier.TIER_1, TaskTier.TIER_2)

    def test_network_tasks_flagged(self):
        """Tasks with web tools should have requires_network=True."""
        all_tasks = load_golden_dataset(skip_network=False)
        web_tasks = [t for t in all_tasks if t.requires_network]
        assert len(web_tasks) >= 2
        for t in web_tasks:
            assert any(
                tool in t.expected_tools for tool in ("web_search", "web_fetch")
            ) or "web" in t.tags


# ---------------------------------------------------------------------------
# PrometheusJudge
# ---------------------------------------------------------------------------


class TestPrometheusJudge:
    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        """Judge should return a JudgeVerdict with score and reasoning."""
        judge = PrometheusJudge(base_url="http://test:8080")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"score": 0.85, "reasoning": "Task completed well"}'
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.get = AsyncMock(
                return_value=MagicMock(
                    json=lambda: {"data": [{"id": "test-model"}]},
                    raise_for_status=MagicMock(),
                )
            )
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            verdict = await judge.evaluate(
                task_input="test task",
                agent_output="test output",
                expected_behavior="should work",
            )

        assert isinstance(verdict, JudgeVerdict)
        assert verdict.score == 0.85
        assert verdict.reasoning == "Task completed well"

    def test_parse_verdict_valid_json(self):
        """Should parse valid JSON verdict."""
        judge = PrometheusJudge()
        verdict = judge._parse_verdict('{"score": 0.9, "reasoning": "great"}')
        assert verdict.score == 0.9
        assert verdict.reasoning == "great"

    def test_parse_verdict_json_with_surrounding_text(self):
        """Should extract JSON from surrounding text."""
        judge = PrometheusJudge()
        verdict = judge._parse_verdict(
            'Here is my evaluation: {"score": 0.7, "reasoning": "mostly done"} end'
        )
        assert verdict.score == 0.7

    def test_parse_verdict_invalid_json(self):
        """Should fallback to regex score extraction."""
        judge = PrometheusJudge()
        verdict = judge._parse_verdict("I rate this 0.6 out of 1.0")
        assert verdict.score == 0.6

    def test_parse_verdict_clamp(self):
        """Score should be clamped to [0, 1]."""
        judge = PrometheusJudge()
        verdict = judge._parse_verdict('{"score": 1.5, "reasoning": "over"}')
        assert verdict.score == 1.0
        verdict = judge._parse_verdict('{"score": -0.5, "reasoning": "under"}')
        assert verdict.score == 0.0

    def test_parse_geval_verdict(self):
        """Should extract score from SCORE: line."""
        judge = PrometheusJudge()
        raw = (
            "Step 1: The agent used bash correctly.\n"
            "Step 2: Output looks good.\n"
            "SCORE: 0.85"
        )
        verdict = judge._parse_geval_verdict(raw)
        assert verdict.score == 0.85
        assert "bash correctly" in verdict.reasoning

    def test_parse_geval_verdict_case_insensitive(self):
        """Should handle Score: and score: variations."""
        judge = PrometheusJudge()
        verdict = judge._parse_geval_verdict("Good work.\nscore: 0.7")
        assert verdict.score == 0.7

    def test_parse_geval_verdict_clamp(self):
        """G-Eval score should be clamped to [0, 1]."""
        judge = PrometheusJudge()
        verdict = judge._parse_geval_verdict("SCORE: 1.5")
        assert verdict.score == 1.0

    def test_parse_geval_verdict_no_score(self):
        """Should fall back when no SCORE: line found."""
        judge = PrometheusJudge()
        verdict = judge._parse_geval_verdict("I think this is 0.6 quality")
        assert verdict.score == 0.6  # fallback regex

    def test_parse_geval_verdict_alt_patterns(self):
        """Should find score from alternative patterns like 'final score'."""
        judge = PrometheusJudge()
        # "final score: X"
        verdict = judge._parse_geval_verdict(
            "1. Good work.\n2. Mostly done.\nFinal score: 0.8"
        )
        assert verdict.score == 0.8
        # "score is X"
        verdict = judge._parse_geval_verdict("Overall the score is 0.75")
        assert verdict.score == 0.75

    def test_parse_geval_verdict_last_line_decimal(self):
        """Should find standalone 0.X on last line as last resort."""
        judge = PrometheusJudge()
        verdict = judge._parse_geval_verdict(
            "1. Agent did the task well.\n2. Output is correct.\n0.9"
        )
        assert verdict.score == 0.9

    def test_parse_geval_verdict_empty(self):
        """Should handle empty response."""
        judge = PrometheusJudge()
        verdict = judge._parse_geval_verdict("")
        assert verdict.score == 0.0
        verdict = judge._parse_geval_verdict("   ")
        assert verdict.score == 0.0

    @pytest.mark.asyncio
    async def test_call_llm_retries_on_empty(self):
        """Should retry when model returns empty response."""
        judge = PrometheusJudge(base_url="http://test:8080")

        call_count = 0

        def make_response(content):
            mock = MagicMock()
            mock.json.return_value = {
                "choices": [{"message": {"content": content}}]
            }
            mock.raise_for_status = MagicMock()
            return mock

        async def fake_post(url, json=None):
            nonlocal call_count
            call_count += 1
            # First two calls return empty, third returns real content
            if call_count <= 2:
                return make_response("")
            return make_response("SCORE: 0.9")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = fake_post
            mock_instance.get = AsyncMock(
                return_value=MagicMock(
                    json=lambda: {"data": [{"id": "test-model"}]},
                    raise_for_status=MagicMock(),
                )
            )
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await judge._call_llm("system", "user", retries=2)

        assert result == "SCORE: 0.9"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_call_llm_returns_empty_after_exhausted_retries(self):
        """Should return empty string after all retries exhausted."""
        judge = PrometheusJudge(base_url="http://test:8080")

        def make_empty():
            mock = MagicMock()
            mock.json.return_value = {
                "choices": [{"message": {"content": ""}}]
            }
            mock.raise_for_status = MagicMock()
            return mock

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=make_empty())
            mock_instance.get = AsyncMock(
                return_value=MagicMock(
                    json=lambda: {"data": [{"id": "test-model"}]},
                    raise_for_status=MagicMock(),
                )
            )
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await judge._call_llm("system", "user", retries=1)

        assert result == ""


# ---------------------------------------------------------------------------
# Constrained Decoding (Sprint 14)
# ---------------------------------------------------------------------------


class TestConstrainedDecoding:
    def test_schema_has_required_fields(self):
        """JUDGE_SCORE_SCHEMA should require score and reasoning."""
        assert "score" in JUDGE_SCORE_SCHEMA["properties"]
        assert "reasoning" in JUDGE_SCORE_SCHEMA["properties"]
        assert "score" in JUDGE_SCORE_SCHEMA["required"]
        assert "reasoning" in JUDGE_SCORE_SCHEMA["required"]
        assert JUDGE_SCORE_SCHEMA["additionalProperties"] is False

    @pytest.mark.asyncio
    async def test_call_llm_includes_response_format(self):
        """_call_llm should include response_format in payload when provided."""
        judge = PrometheusJudge(base_url="http://test:8080", model="test")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"score": 0.8, "reasoning": "ok"}'}}]
        }
        mock_resp.raise_for_status = MagicMock()

        captured_payload = {}

        async def capture_post(url, json=None):
            captured_payload.update(json)
            return mock_resp

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = capture_post
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            schema_fmt = {
                "type": "json_schema",
                "json_schema": {"name": "test", "schema": JUDGE_SCORE_SCHEMA},
            }
            await judge._call_llm(
                "system", "user",
                response_format=schema_fmt,
                chat_template_kwargs={"enable_thinking": False},
            )

        assert "response_format" in captured_payload
        assert captured_payload["response_format"]["type"] == "json_schema"
        assert "chat_template_kwargs" in captured_payload
        assert captured_payload["chat_template_kwargs"]["enable_thinking"] is False

    @pytest.mark.asyncio
    async def test_call_llm_no_extras_by_default(self):
        """_call_llm should NOT include response_format when not provided."""
        judge = PrometheusJudge(base_url="http://test:8080", model="test")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "hello"}}]
        }
        mock_resp.raise_for_status = MagicMock()

        captured_payload = {}

        async def capture_post(url, json=None):
            captured_payload.update(json)
            return mock_resp

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = capture_post
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            await judge._call_llm("system", "user")

        assert "response_format" not in captured_payload
        assert "chat_template_kwargs" not in captured_payload

    @pytest.mark.asyncio
    async def test_reasoning_content_fallback(self):
        """Should extract JSON from reasoning_content when content is empty."""
        judge = PrometheusJudge(base_url="http://test:8080", model="test")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {
                "content": "",
                "reasoning_content": 'The agent did well. {"score": 0.9, "reasoning": "completed"} done.',
            }}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_resp)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await judge._call_llm("system", "user")

        parsed = json.loads(result)
        assert parsed["score"] == 0.9
        assert parsed["reasoning"] == "completed"

    def test_extract_json_from_reasoning(self):
        """Should find JSON with score field in reasoning text."""
        judge = PrometheusJudge()
        result = judge._extract_json_from_reasoning(
            'Thinking... {"score": 0.7, "reasoning": "mostly done"} end'
        )
        assert json.loads(result)["score"] == 0.7

    def test_extract_json_no_match(self):
        """Should return empty string when no JSON found."""
        judge = PrometheusJudge()
        result = judge._extract_json_from_reasoning("just plain text no json here")
        assert result == ""

    def test_parse_verdict_direct_json(self):
        """Should parse clean JSON directly (constrained output)."""
        judge = PrometheusJudge()
        verdict = judge._parse_verdict('{"score": 0.85, "reasoning": "well done"}')
        assert verdict.score == 0.85
        assert verdict.reasoning == "well done"

    def test_parse_verdict_empty(self):
        """Should handle empty response."""
        judge = PrometheusJudge()
        verdict = judge._parse_verdict("")
        assert verdict.score == 0.0

    def test_parse_verdict_markdown_fences(self):
        """Should strip markdown fences wrapping JSON (Ollama pattern)."""
        judge = PrometheusJudge()
        verdict = judge._parse_verdict(
            '```json\n{"score": 0.75, "reasoning": "mostly done"}\n```'
        )
        assert verdict.score == 0.75
        assert verdict.reasoning == "mostly done"

    def test_parse_verdict_markdown_fences_no_lang(self):
        """Should strip bare ``` fences too."""
        judge = PrometheusJudge()
        verdict = judge._parse_verdict(
            '```\n{"score": 0.6, "reasoning": "partial"}\n```'
        )
        assert verdict.score == 0.6

    def test_parse_verdict_preamble_text(self):
        """Should extract JSON from text with preamble."""
        judge = PrometheusJudge()
        verdict = judge._parse_verdict(
            'Here is my evaluation:\n{"score": 0.9, "reasoning": "great work"}'
        )
        assert verdict.score == 0.9


# ---------------------------------------------------------------------------
# ToolUsageMetric
# ---------------------------------------------------------------------------


class TestToolUsageMetric:
    def test_all_tools_present(self):
        """Score should be 1.0 when all expected tools were called."""
        metric = ToolUsageMetric(threshold=0.5)
        tc = _SimpleTestCase(
            input="List files",
            actual_output="Here are the files...",
            expected_output="Uses bash",
            context=["Tools expected: bash"],
            additional_metadata={
                "tool_trace": [{"tool_name": "bash", "result": "ok"}]
            },
        )
        score = metric.measure(tc)
        assert score == 1.0
        assert metric.is_successful()

    def test_missing_tools(self):
        """Score should be partial when some tools were not called."""
        metric = ToolUsageMetric(threshold=0.5)
        tc = _SimpleTestCase(
            input="Write and read",
            actual_output="Done",
            expected_output="Uses write_file and read_file",
            context=["Tools expected: write_file, read_file"],
            additional_metadata={
                "tool_trace": [{"tool_name": "write_file", "result": "ok"}]
            },
        )
        score = metric.measure(tc)
        assert score == 0.5  # 1 of 2 expected tools

    def test_no_expected_tools(self):
        """Score should be 1.0 when no tools are expected."""
        metric = ToolUsageMetric()
        tc = _SimpleTestCase(
            input="What is 2+2?",
            actual_output="4",
            expected_output="Returns 4",
            context=[],
        )
        score = metric.measure(tc)
        assert score == 1.0

    def test_no_tools_used(self):
        """Score should be 0.0 when expected tools were not called."""
        metric = ToolUsageMetric(threshold=0.5)
        tc = _SimpleTestCase(
            input="Read file",
            actual_output="...",
            expected_output="Uses read_file",
            context=["Tools expected: read_file"],
            additional_metadata={"tool_trace": []},
        )
        score = metric.measure(tc)
        assert score == 0.0
        assert not metric.is_successful()

    @pytest.mark.asyncio
    async def test_async_measure(self):
        """a_measure should delegate to sync measure."""
        metric = ToolUsageMetric()
        tc = _SimpleTestCase(
            input="test", actual_output="out", expected_output="exp"
        )
        score = await metric.a_measure(tc)
        assert score == 1.0  # no expected tools


# ---------------------------------------------------------------------------
# TaskCompletionMetric
# ---------------------------------------------------------------------------


class TestTaskCompletionMetric:
    @pytest.mark.asyncio
    async def test_high_score(self):
        """Should pass through judge's constrained score."""
        mock_judge = AsyncMock(spec=PrometheusJudge)
        mock_judge.evaluate = AsyncMock(
            return_value=JudgeVerdict(score=0.9, reasoning="well done", raw_response="")
        )

        metric = TaskCompletionMetric(judge=mock_judge, threshold=0.7)
        tc = _SimpleTestCase(
            input="test task",
            actual_output="completed",
            expected_output="should complete",
        )
        score = await metric.a_measure(tc)
        assert score == 0.9
        assert metric.is_successful()
        mock_judge.evaluate.assert_called_once()

    @pytest.mark.asyncio
    async def test_low_score(self):
        """Should fail when below threshold."""
        mock_judge = AsyncMock(spec=PrometheusJudge)
        mock_judge.evaluate = AsyncMock(
            return_value=JudgeVerdict(score=0.3, reasoning="poor", raw_response="")
        )

        metric = TaskCompletionMetric(judge=mock_judge, threshold=0.7)
        tc = _SimpleTestCase(
            input="test", actual_output="failed", expected_output="should pass"
        )
        score = await metric.a_measure(tc)
        assert score == 0.3
        assert not metric.is_successful()


# ---------------------------------------------------------------------------
# NoHallucinationMetric
# ---------------------------------------------------------------------------


class TestNoHallucinationMetric:
    @pytest.mark.asyncio
    async def test_grounded_output(self):
        """Should score high when output is grounded."""
        mock_judge = AsyncMock(spec=PrometheusJudge)
        mock_judge.evaluate = AsyncMock(
            return_value=JudgeVerdict(
                score=1.0, reasoning="fully grounded", raw_response=""
            )
        )

        metric = NoHallucinationMetric(judge=mock_judge, threshold=0.8)
        tc = _SimpleTestCase(
            input="list files",
            actual_output="file1.txt, file2.txt",
            expected_output="lists files",
            additional_metadata={
                "tool_trace": [
                    {"tool_name": "bash", "result": "file1.txt\nfile2.txt"}
                ]
            },
        )
        score = await metric.a_measure(tc)
        assert score == 1.0
        assert metric.is_successful()
        mock_judge.evaluate.assert_called_once()

    @pytest.mark.asyncio
    async def test_hallucinated_output(self):
        """Should score low when output contains hallucinations."""
        mock_judge = AsyncMock(spec=PrometheusJudge)
        mock_judge.evaluate = AsyncMock(
            return_value=JudgeVerdict(
                score=0.2, reasoning="fabricated results", raw_response=""
            )
        )

        metric = NoHallucinationMetric(judge=mock_judge, threshold=0.8)
        tc = _SimpleTestCase(
            input="test", actual_output="made up stuff", expected_output="real data"
        )
        score = await metric.a_measure(tc)
        assert score == 0.2
        assert not metric.is_successful()


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------


class TestEvalRunner:
    @pytest.mark.asyncio
    async def test_run_task(self):
        """Runner should execute a task and return EvalResult."""
        mock_result = MagicMock()
        mock_result.text = "Task completed successfully"
        mock_result.turns = 3

        mock_loop = MagicMock()
        mock_loop.run_async = AsyncMock(return_value=mock_result)
        mock_loop._tool_trace = [
            {"tool_name": "bash", "result": "ok", "is_error": False}
        ]

        mock_judge = AsyncMock(spec=PrometheusJudge)
        mock_judge.evaluate = AsyncMock(
            return_value=JudgeVerdict(score=0.9, reasoning="good", raw_response="")
        )

        runner = EvalRunner(
            agent_loop=mock_loop,
            judge=mock_judge,
            system_prompt="You are helpful.",
        )

        task = GoldenTask(
            id="test-1",
            name="Test task",
            tier=1,
            input="Do something",
            expected_behavior="Should do the thing",
            expected_tools=["bash"],
            tags=["test"],
        )

        result = await runner.run_task(task)

        assert isinstance(result, EvalResult)
        assert result.task_id == "test-1"
        assert result.agent_output == "Task completed successfully"
        assert result.turns == 3
        assert result.latency_ms >= 0
        assert result.error is None
        assert len(result.tool_trace) == 1
        assert result.failure_source == "pass"
        mock_loop.run_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_task_crash(self):
        """Runner should handle agent crashes gracefully."""
        mock_loop = MagicMock()
        mock_loop.run_async = AsyncMock(side_effect=RuntimeError("boom"))
        mock_loop._tool_trace = []

        mock_judge = AsyncMock(spec=PrometheusJudge)

        runner = EvalRunner(
            agent_loop=mock_loop,
            judge=mock_judge,
            system_prompt="test",
        )

        task = GoldenTask(
            id="crash-1",
            name="Crash test",
            tier=1,
            input="Crash me",
            expected_behavior="Should not crash",
            expected_tools=[],
            tags=[],
        )

        result = await runner.run_task(task)

        assert result.error == "boom"
        assert result.agent_output == ""
        assert result.latency_ms >= 0
        assert result.failure_source in ("harness", "unclear")

    @pytest.mark.asyncio
    async def test_run_all(self):
        """Should run all tasks and return results list."""
        mock_result = MagicMock()
        mock_result.text = "done"
        mock_result.turns = 1

        mock_loop = MagicMock()
        mock_loop.run_async = AsyncMock(return_value=mock_result)
        mock_loop._tool_trace = []

        mock_judge = AsyncMock(spec=PrometheusJudge)
        mock_judge.evaluate = AsyncMock(
            return_value=JudgeVerdict(score=0.8, reasoning="ok", raw_response="")
        )

        runner = EvalRunner(
            agent_loop=mock_loop,
            judge=mock_judge,
            system_prompt="test",
        )

        tasks = [
            GoldenTask(
                id=f"t-{i}",
                name=f"Task {i}",
                tier=1,
                input=f"Do thing {i}",
                expected_behavior="Do it",
                expected_tools=[],
                tags=[],
            )
            for i in range(3)
        ]

        results = await runner.run_all(tasks=tasks)
        assert len(results) == 3
        assert mock_loop.run_async.call_count == 3

    def test_save_results(self, tmp_path):
        """Results should be saved as timestamped JSON."""
        mock_judge = MagicMock()
        mock_loop = MagicMock()

        runner = EvalRunner(
            agent_loop=mock_loop,
            judge=mock_judge,
            system_prompt="test",
        )

        results = [
            EvalResult(
                task_id="t-1",
                task_name="Test",
                tier=1,
                agent_output="done",
                turns=2,
                latency_ms=150.0,
                metrics=[
                    MetricScore(
                        metric_name="Task Completion",
                        score=0.9,
                        threshold=0.7,
                        passed=True,
                    )
                ],
            )
        ]

        path = runner.save_results(results, output_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

        data = json.loads(path.read_text())
        assert data["task_count"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["task_id"] == "t-1"
        assert "summary" in data

    def test_print_summary(self, capsys):
        """Should print a readable summary."""
        mock_judge = MagicMock()
        mock_loop = MagicMock()

        runner = EvalRunner(
            agent_loop=mock_loop,
            judge=mock_judge,
            system_prompt="test",
        )

        results = [
            EvalResult(
                task_id="t-1",
                task_name="Test",
                tier=1,
                agent_output="done",
                turns=2,
                latency_ms=100.0,
                metrics=[
                    MetricScore(
                        metric_name="Task Completion",
                        score=0.9,
                        threshold=0.7,
                        passed=True,
                    )
                ],
            )
        ]

        runner.print_summary(results)
        captured = capsys.readouterr()
        assert "EVALUATION SUMMARY" in captured.out
        assert "t-1" in captured.out


# ---------------------------------------------------------------------------
# Failure Classification
# ---------------------------------------------------------------------------


class TestFailureClassifier:
    def test_all_pass(self):
        """Should return PASS when all metrics are good."""
        result = classify_failure(
            task_id="t1",
            expected_tools=["bash"],
            tool_trace=[{"tool_name": "bash", "result": "ok", "is_error": False}],
            agent_output="done",
            error=None,
            metric_scores={"Task Completion": 0.9, "Tool Usage": 1.0, "No Hallucination": 0.95},
        )
        assert result.source == FailureSource.PASS

    def test_crash_harness_keyword(self):
        """Should classify crash with harness keywords as HARNESS."""
        result = classify_failure(
            task_id="t1",
            expected_tools=["glob"],
            tool_trace=[],
            agent_output="",
            error="Non-relative patterns are unsupported",
            metric_scores={},
        )
        assert result.source == FailureSource.HARNESS
        assert result.category == FailureCategory.TOOL_CRASH

    def test_crash_unknown(self):
        """Should classify ambiguous crash as UNCLEAR."""
        result = classify_failure(
            task_id="t1",
            expected_tools=[],
            tool_trace=[],
            agent_output="",
            error="RuntimeError: something weird",
            metric_scores={},
        )
        assert result.source == FailureSource.UNCLEAR

    def test_no_tool_call(self):
        """Should classify as MODEL when expected tools not called."""
        result = classify_failure(
            task_id="t1",
            expected_tools=["bash"],
            tool_trace=[],
            agent_output="The directory is /home/will",
            error=None,
            metric_scores={"Task Completion": 0.9, "Tool Usage": 0.0, "No Hallucination": 0.8},
        )
        assert result.source == FailureSource.MODEL
        assert result.category == FailureCategory.NO_TOOL_CALL

    def test_wrong_tool(self):
        """Should classify as MODEL when different tools used."""
        result = classify_failure(
            task_id="t1",
            expected_tools=["read_file"],
            tool_trace=[{"tool_name": "bash", "result": "ok", "is_error": False}],
            agent_output="done",
            error=None,
            metric_scores={"Task Completion": 0.8, "Tool Usage": 0.0, "No Hallucination": 0.9},
        )
        assert result.source == FailureSource.MODEL
        assert result.category == FailureCategory.WRONG_TOOL

    def test_tool_error_harness(self):
        """Should classify tool execution errors as HARNESS."""
        result = classify_failure(
            task_id="t1",
            expected_tools=["glob"],
            tool_trace=[{"tool_name": "glob", "result": "Error: path not found", "is_error": True}],
            agent_output="",
            error=None,
            metric_scores={"Task Completion": 0.3},
        )
        assert result.source == FailureSource.HARNESS
        assert result.category == FailureCategory.TOOL_ERROR

    def test_permission_denied(self):
        """Should classify permission blocks as HARNESS."""
        result = classify_failure(
            task_id="t1",
            expected_tools=["bash"],
            tool_trace=[{"tool_name": "bash", "result": "Permission denied for bash", "is_error": True}],
            agent_output="",
            error=None,
            metric_scores={"Task Completion": 0.0},
        )
        assert result.source == FailureSource.HARNESS
        assert result.category == FailureCategory.PERMISSION_DENIED

    def test_bad_args_model(self):
        """Should classify validation failures as MODEL (bad args)."""
        result = classify_failure(
            task_id="t1",
            expected_tools=["write_file"],
            tool_trace=[{"tool_name": "write_file", "result": "Invalid input for write_file: path required", "is_error": True}],
            agent_output="",
            error=None,
            metric_scores={"Task Completion": 0.0},
        )
        assert result.source == FailureSource.MODEL
        assert result.category == FailureCategory.BAD_ARGS

    def test_hallucination(self):
        """Should classify low hallucination score as MODEL."""
        result = classify_failure(
            task_id="t1",
            expected_tools=["bash"],
            tool_trace=[{"tool_name": "bash", "result": "ok", "is_error": False}],
            agent_output="made up stuff",
            error=None,
            metric_scores={"Task Completion": 0.7, "Tool Usage": 1.0, "No Hallucination": 0.2},
        )
        assert result.source == FailureSource.MODEL
        assert result.category == FailureCategory.HALLUCINATED_OUTPUT

    def test_incomplete_model(self):
        """Should classify tools OK but completion failed as MODEL incomplete."""
        result = classify_failure(
            task_id="t1",
            expected_tools=["write_file", "bash"],
            tool_trace=[
                {"tool_name": "write_file", "result": "ok", "is_error": False},
            ],
            agent_output="wrote the file",
            error=None,
            metric_scores={"Task Completion": 0.3, "Tool Usage": 0.5, "No Hallucination": 0.9},
        )
        assert result.source == FailureSource.MODEL
        assert result.category == FailureCategory.INCOMPLETE


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


class TestTracing:
    def test_tracing_disabled_by_default(self):
        """Tracing should be off unless env var set."""
        os.environ.pop("PROMETHEUS_TRACING", None)
        from prometheus.tracing.phoenix import is_tracing_enabled

        assert not is_tracing_enabled()

    def test_span_context_noop(self):
        """span_context should be a passthrough when disabled."""
        os.environ.pop("PROMETHEUS_TRACING", None)
        from prometheus.tracing.spans import span_context

        with span_context("test_span", {"key": "value"}) as s:
            pass  # Should not raise

    def test_traced_decorator_sync(self):
        """@traced should not modify sync function behavior when disabled."""
        os.environ.pop("PROMETHEUS_TRACING", None)
        from prometheus.tracing.spans import traced

        @traced("test_func")
        def my_func(x):
            return x * 2

        assert my_func(5) == 10

    @pytest.mark.asyncio
    async def test_traced_decorator_async(self):
        """@traced should not modify async function behavior when disabled."""
        os.environ.pop("PROMETHEUS_TRACING", None)
        from prometheus.tracing.spans import traced

        @traced("test_async")
        async def my_async(x):
            return x + 1

        result = await my_async(10)
        assert result == 11

    def test_init_tracing_disabled(self):
        """init_tracing should return None when disabled."""
        os.environ.pop("PROMETHEUS_TRACING", None)
        from prometheus.tracing.phoenix import init_tracing, shutdown_tracing

        # Reset state
        shutdown_tracing()

        result = init_tracing({})
        assert result is None


# ---------------------------------------------------------------------------
# Trend Tracking
# ---------------------------------------------------------------------------


class TestTrendTracker:
    def test_record_and_get_latest(self, tmp_path):
        """Should store and retrieve run summaries."""
        tracker = TrendTracker(tmp_path / "trends.db")

        tracker.record({
            "total_tasks": 10,
            "completed": 9,
            "errored": 1,
            "avg_latency_ms": 250.0,
            "metric_averages": {"Task Completion": 0.82, "Tool Usage": 0.95},
        })
        tracker.record({
            "total_tasks": 10,
            "completed": 10,
            "errored": 0,
            "avg_latency_ms": 200.0,
            "metric_averages": {"Task Completion": 0.88, "Tool Usage": 0.97},
        })

        rows = tracker.get_latest(5)
        assert len(rows) == 2
        # Latest first
        assert rows[0].completed == 10
        assert rows[0].metric_averages["Task Completion"] == 0.88
        assert rows[1].completed == 9

        tracker.close()

    def test_get_previous(self, tmp_path):
        """Should return the most recent stored run."""
        tracker = TrendTracker(tmp_path / "trends.db")

        assert tracker.get_previous() is None  # empty

        tracker.record({
            "total_tasks": 5,
            "completed": 5,
            "errored": 0,
            "avg_latency_ms": 100.0,
            "metric_averages": {"Task Completion": 0.75},
        })

        prev = tracker.get_previous()
        assert prev is not None
        assert prev.task_count == 5
        assert prev.metric_averages["Task Completion"] == 0.75

        tracker.close()

    def test_format_trend_comparison_with_previous(self, tmp_path):
        """Should show delta vs previous run."""
        tracker = TrendTracker(tmp_path / "trends.db")

        previous = TrendRow(
            timestamp="2026-04-05",
            task_count=10,
            completed=10,
            errored=0,
            avg_latency_ms=200.0,
            metric_averages={"Task Completion": 0.72, "Tool Usage": 0.90},
        )

        current = {"Task Completion": 0.85, "Tool Usage": 0.92}

        output = tracker.format_trend_comparison(current, previous)
        assert "+0.130" in output  # 0.85 - 0.72
        assert "+0.020" in output  # 0.92 - 0.90
        assert "Task Completion" in output

        tracker.close()

    def test_format_trend_comparison_no_previous(self, tmp_path):
        """Should show (new) when no previous run exists."""
        tracker = TrendTracker(tmp_path / "trends.db")

        current = {"Task Completion": 0.85}
        output = tracker.format_trend_comparison(current, None)
        assert "(new)" in output

        tracker.close()

    def test_format_trend_empty_metrics(self, tmp_path):
        """Should handle empty metrics gracefully."""
        tracker = TrendTracker(tmp_path / "trends.db")
        output = tracker.format_trend_comparison({}, None)
        assert "(no metrics)" in output
        tracker.close()
