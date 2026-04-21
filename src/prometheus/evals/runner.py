"""Eval runner — orchestrates golden dataset evaluation with Phoenix tracing.

Runs golden tasks through AgentLoop, scores with custom metrics,
and saves timestamped JSON results.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from prometheus.evals.classifier import (
    FailureClassification,
    FailureSource,
    classify_failure,
)
from prometheus.evals.golden_dataset import GoldenTask, load_golden_dataset
from prometheus.evals.judge import PrometheusJudge
from prometheus.evals.trends import TrendTracker
from prometheus.engine.agent_loop import AgentLoop
from prometheus.tracing.spans import span_context

log = logging.getLogger(__name__)


@dataclass
class MetricScore:
    """Score from a single metric evaluation."""

    metric_name: str
    score: float
    threshold: float
    passed: bool
    reasoning: str = ""


@dataclass
class EvalResult:
    """Result of evaluating a single golden task."""

    task_id: str
    task_name: str
    tier: int
    agent_output: str
    turns: int
    latency_ms: float
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[MetricScore] = field(default_factory=list)
    error: str | None = None
    failure_source: str = "pass"       # "pass", "model", "harness", "unclear"
    failure_category: str = "none"     # e.g. "model:wrong_tool", "harness:tool_crash"
    failure_detail: str = ""


class EvalRunner:
    """Run golden tasks through AgentLoop and evaluate with metrics."""

    def __init__(
        self,
        agent_loop: AgentLoop,
        judge: PrometheusJudge,
        system_prompt: str,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._loop = agent_loop
        self._judge = judge
        self._system_prompt = system_prompt
        self._config = config or {}

    async def run_task(self, task: GoldenTask) -> EvalResult:
        """Run a single golden task and evaluate with all metrics."""
        with span_context("eval_task", {"task_id": task.id, "tier": task.tier}):
            t0 = time.monotonic()
            try:
                result = await self._loop.run_async(
                    system_prompt=self._system_prompt,
                    user_message=task.input,
                )
                # Copy tool trace immediately — it gets cleared by post-task hook
                tool_trace = list(self._loop._tool_trace)
                latency_ms = (time.monotonic() - t0) * 1000

                log.info(
                    "Task %s completed in %.0fms (%d turns)",
                    task.id,
                    latency_ms,
                    result.turns,
                )

                # Run metrics
                metric_scores = await self._evaluate_metrics(
                    task, result.text, tool_trace
                )

                # Classify failure source
                score_map = {m.metric_name: m.score for m in metric_scores}
                classification = classify_failure(
                    task_id=task.id,
                    expected_tools=task.expected_tools,
                    tool_trace=tool_trace,
                    agent_output=result.text,
                    error=None,
                    metric_scores=score_map,
                )

                return EvalResult(
                    task_id=task.id,
                    task_name=task.name,
                    tier=task.tier,
                    agent_output=result.text[:2000],
                    turns=result.turns,
                    latency_ms=round(latency_ms, 1),
                    tool_trace=tool_trace,
                    metrics=metric_scores,
                    failure_source=classification.source.value,
                    failure_category=classification.category.value,
                    failure_detail=classification.detail,
                )

            except Exception as exc:
                latency_ms = (time.monotonic() - t0) * 1000
                log.error("Task %s crashed: %s", task.id, exc)

                classification = classify_failure(
                    task_id=task.id,
                    expected_tools=task.expected_tools,
                    tool_trace=[],
                    agent_output="",
                    error=str(exc),
                    metric_scores={},
                )

                return EvalResult(
                    task_id=task.id,
                    task_name=task.name,
                    tier=task.tier,
                    agent_output="",
                    turns=0,
                    latency_ms=round(latency_ms, 1),
                    error=str(exc),
                    failure_source=classification.source.value,
                    failure_category=classification.category.value,
                    failure_detail=classification.detail,
                )

    async def _evaluate_metrics(
        self,
        task: GoldenTask,
        agent_output: str,
        tool_trace: list[dict[str, Any]],
    ) -> list[MetricScore]:
        """Run all three metrics against a completed task."""
        from prometheus.evals.metrics import (
            TaskCompletionMetric,
            ToolUsageMetric,
            NoHallucinationMetric,
        )

        # Build a test-case-like object for metrics
        test_case = _SimpleTestCase(
            input=task.input,
            actual_output=agent_output,
            expected_output=task.expected_behavior,
            context=[f"Tools expected: {', '.join(task.expected_tools)}"]
            if task.expected_tools
            else [],
            additional_metadata={"tool_trace": tool_trace},
        )

        scores: list[MetricScore] = []

        # ToolUsageMetric — deterministic, always runs
        try:
            tool_metric = ToolUsageMetric(threshold=0.5)
            tool_score = await tool_metric.a_measure(test_case)
            scores.append(
                MetricScore(
                    metric_name="Tool Usage",
                    score=tool_score,
                    threshold=0.5,
                    passed=tool_metric.is_successful(),
                    reasoning=tool_metric.reason or "",
                )
            )
        except Exception as exc:
            log.warning("ToolUsageMetric failed: %s", exc)

        # TaskCompletionMetric — LLM judge
        try:
            completion_metric = TaskCompletionMetric(
                judge=self._judge, threshold=0.7
            )
            completion_score = await completion_metric.a_measure(test_case)
            scores.append(
                MetricScore(
                    metric_name="Task Completion",
                    score=completion_score,
                    threshold=0.7,
                    passed=completion_metric.is_successful(),
                    reasoning=completion_metric.reason or "",
                )
            )
        except Exception as exc:
            log.warning("TaskCompletionMetric failed: %s", exc)

        # NoHallucinationMetric — LLM judge
        try:
            hallucination_metric = NoHallucinationMetric(
                judge=self._judge, threshold=0.8
            )
            hallucination_score = await hallucination_metric.a_measure(test_case)
            scores.append(
                MetricScore(
                    metric_name="No Hallucination",
                    score=hallucination_score,
                    threshold=0.8,
                    passed=hallucination_metric.is_successful(),
                    reasoning=hallucination_metric.reason or "",
                )
            )
        except Exception as exc:
            log.warning("NoHallucinationMetric failed: %s", exc)

        return scores

    async def run_all(
        self,
        tasks: list[GoldenTask] | None = None,
        *,
        tier: int | None = None,
        skip_network: bool = True,
    ) -> list[EvalResult]:
        """Run all golden tasks sequentially."""
        if tasks is None:
            tasks = load_golden_dataset(tier=tier, skip_network=skip_network)

        log.info("Running %d golden tasks...", len(tasks))
        results: list[EvalResult] = []

        for i, task in enumerate(tasks, 1):
            log.info("[%d/%d] %s: %s", i, len(tasks), task.id, task.name)
            result = await self.run_task(task)
            results.append(result)

        return results

    def save_results(
        self,
        results: list[EvalResult],
        output_dir: Path | None = None,
    ) -> Path:
        """Save results to a timestamped JSON file and record trend."""
        out_dir = output_dir or Path(
            self._config.get("results_dir", "~/.prometheus/eval_results")
        ).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"results_{timestamp}.json"
        path = out_dir / filename

        summary = self._compute_summary(results)

        # Serialize JSON
        data = {
            "timestamp": datetime.now().isoformat(),
            "task_count": len(results),
            "results": [asdict(r) for r in results],
            "summary": summary,
        }
        path.write_text(json.dumps(data, indent=2, default=str))
        log.info("Results saved to %s", path)

        # Record trend
        try:
            tracker = TrendTracker(out_dir / "trends.db")
            tracker.record(summary)
            tracker.close()
        except Exception as exc:
            log.warning("Failed to record trend: %s", exc)

        return path

    def _compute_summary(self, results: list[EvalResult]) -> dict[str, Any]:
        """Compute aggregate summary statistics."""
        if not results:
            return {}

        total = len(results)
        errored = sum(1 for r in results if r.error)
        latencies = [r.latency_ms for r in results if not r.error]

        # Aggregate metric scores
        metric_aggs: dict[str, list[float]] = {}
        for r in results:
            for m in r.metrics:
                metric_aggs.setdefault(m.metric_name, []).append(m.score)

        # Failure source breakdown
        source_counts: dict[str, int] = {}
        category_counts: dict[str, int] = {}
        for r in results:
            src = r.failure_source
            source_counts[src] = source_counts.get(src, 0) + 1
            if src != "pass":
                cat = r.failure_category
                category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "total_tasks": total,
            "errored": errored,
            "completed": total - errored,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1)
            if latencies
            else 0,
            "total_latency_ms": round(sum(latencies), 1),
            "metric_averages": {
                name: round(sum(scores) / len(scores), 3)
                for name, scores in metric_aggs.items()
            },
            "failure_sources": source_counts,
            "failure_categories": category_counts,
        }

    def print_summary(
        self, results: list[EvalResult], output_dir: Path | None = None
    ) -> None:
        """Print a summary report to stdout, with trend comparison."""
        summary = self._compute_summary(results)

        print("\n" + "=" * 72)
        print("EVALUATION SUMMARY")
        print("=" * 72)

        for r in results:
            if r.error:
                status = "ERR "
            elif r.failure_source == "pass":
                status = "PASS"
            elif r.failure_source == "model":
                status = "MDL "
            elif r.failure_source == "harness":
                status = "HRN "
            else:
                status = "??? "

            metric_str = "  ".join(
                f"{m.metric_name}={m.score:.2f}" for m in r.metrics
            )
            line = f"  [{status}] {r.task_id:25s} {r.latency_ms:8.0f}ms  {metric_str}"
            if r.failure_source not in ("pass",):
                line += f"  <- {r.failure_category}"
            print(line)

        print("-" * 72)
        print(
            f"  Tasks: {summary.get('total_tasks', 0)}  |  "
            f"OK: {summary.get('completed', 0)}  |  "
            f"Errors: {summary.get('errored', 0)}"
        )
        print(
            f"  Avg latency: {summary.get('avg_latency_ms', 0):.0f}ms  |  "
            f"Total: {summary.get('total_latency_ms', 0):.0f}ms"
        )

        # Failure source breakdown
        sources = summary.get("failure_sources", {})
        pass_count = sources.get("pass", 0)
        model_count = sources.get("model", 0)
        harness_count = sources.get("harness", 0)
        unclear_count = sources.get("unclear", 0)
        print(
            f"  Classification:  PASS={pass_count}  "
            f"MODEL={model_count}  HARNESS={harness_count}  UNCLEAR={unclear_count}"
        )

        categories = summary.get("failure_categories", {})
        if categories:
            print("  Failure breakdown:")
            for cat, count in sorted(categories.items()):
                print(f"    {cat}: {count}")

        # Trend comparison
        avg = summary.get("metric_averages", {})
        if avg:
            try:
                out_dir = output_dir or Path(
                    self._config.get("results_dir", "~/.prometheus/eval_results")
                ).expanduser()
                tracker = TrendTracker(out_dir / "trends.db")
                previous = tracker.get_previous()
                print("  Metric averages:")
                print(tracker.format_trend_comparison(avg, previous))
                tracker.close()
            except Exception:
                print("  Metric averages:")
                for name, score in avg.items():
                    print(f"    {name}: {score:.3f}")

        print("=" * 72 + "\n")


class _SimpleTestCase:
    """Lightweight test case for metrics — avoids deepeval import."""

    def __init__(
        self,
        input: str,
        actual_output: str,
        expected_output: str,
        context: list[str] | None = None,
        additional_metadata: dict[str, Any] | None = None,
    ):
        self.input = input
        self.actual_output = actual_output
        self.expected_output = expected_output
        self.context = context or []
        self.additional_metadata = additional_metadata or {}
