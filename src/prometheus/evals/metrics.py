"""Custom DeepEval metrics for Prometheus evaluation.

Three metrics tailored for tool-heavy agent evaluation:
- TaskCompletionMetric: LLM-judged task completion (0-1)
- ToolUsageMetric: Deterministic tool usage check (no LLM)
- NoHallucinationMetric: LLM-judged groundedness check
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING

log = logging.getLogger(__name__)

# Graceful import — deepeval is optional
try:
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase

    _HAS_DEEPEVAL = True
except ImportError:
    _HAS_DEEPEVAL = False

    class BaseMetric:  # type: ignore[no-redef]
        """Stub when deepeval is not installed."""

        threshold: float = 0.0
        score: float | None = None
        reason: str | None = None
        success: bool | None = None

        def measure(self, test_case: Any) -> float:
            raise NotImplementedError("deepeval not installed")

        async def a_measure(self, test_case: Any) -> float:
            raise NotImplementedError("deepeval not installed")

        def is_successful(self) -> bool:
            return (self.score or 0) >= self.threshold

    class LLMTestCase:  # type: ignore[no-redef]
        """Stub when deepeval is not installed."""

        pass


if TYPE_CHECKING:
    from prometheus.evals.judge import PrometheusJudge


class TaskCompletionMetric(BaseMetric):
    """Measures whether the agent completed the requested task.

    Uses constrained decoding (JSON schema mode) for reliable scoring.
    The grammar constraint guarantees valid JSON output from the judge —
    no parsing heuristics needed.
    """

    def __init__(self, judge: PrometheusJudge, threshold: float = 0.7) -> None:
        self.threshold = threshold
        self.judge = judge
        self.score: float | None = None
        self.reason: str | None = None
        self.success: bool | None = None

    @property
    def __name__(self) -> str:
        return "Task Completion"

    def measure(self, test_case: LLMTestCase) -> float:
        """Sync measurement — wraps a_measure."""
        return asyncio.run(self.a_measure(test_case))

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async measurement via constrained JSON judge."""
        meta = getattr(test_case, "additional_metadata", None) or {}
        tool_trace = meta.get("tool_trace")
        if tool_trace and not isinstance(tool_trace, list):
            tool_trace = None

        verdict = await self.judge.evaluate(
            task_input=test_case.input,
            agent_output=test_case.actual_output or "",
            expected_behavior=test_case.expected_output or "",
            tool_trace=tool_trace,
        )
        self.score = verdict.score
        self.reason = verdict.reasoning
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return (self.score or 0) >= self.threshold


class ToolUsageMetric(BaseMetric):
    """Measures whether the agent used appropriate tools.

    Deterministic metric — no LLM required. Checks that expected tools
    were called by reading the tool trace from additional_metadata.
    Score = (expected tools found) / (total expected tools).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.score: float | None = None
        self.reason: str | None = None
        self.success: bool | None = None

    @property
    def __name__(self) -> str:
        return "Tool Usage"

    def measure(self, test_case: LLMTestCase) -> float:
        """Sync measurement — deterministic, no LLM needed."""
        meta = getattr(test_case, "additional_metadata", None) or {}
        tool_trace = meta.get("tool_trace", [])
        tools_used = {t.get("tool_name", "") for t in tool_trace if isinstance(t, dict)}

        # Extract expected tools from context
        expected = set()
        context = getattr(test_case, "context", None) or []
        for ctx in context:
            if isinstance(ctx, str) and ctx.startswith("Tools expected:"):
                expected = {
                    t.strip()
                    for t in ctx.replace("Tools expected:", "").split(",")
                    if t.strip()
                }

        if not expected:
            self.score = 1.0
            self.reason = "No specific tools expected"
            self.success = True
            return self.score

        # Score based on overlap (any expected tool found counts)
        overlap = len(tools_used & expected)
        self.score = overlap / len(expected)
        self.reason = f"Used {tools_used or 'none'}, expected {expected}"
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async wrapper — delegates to sync (no I/O needed)."""
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return (self.score or 0) >= self.threshold


class NoHallucinationMetric(BaseMetric):
    """Measures whether the agent's output is grounded in tool results.

    Uses constrained decoding (JSON schema mode) for reliable scoring.
    The judge evaluates whether the agent fabricated information not
    present in actual tool outputs.
    """

    def __init__(self, judge: PrometheusJudge, threshold: float = 0.8) -> None:
        self.threshold = threshold
        self.judge = judge
        self.score: float | None = None
        self.reason: str | None = None
        self.success: bool | None = None

    @property
    def __name__(self) -> str:
        return "No Hallucination"

    def measure(self, test_case: LLMTestCase) -> float:
        """Sync measurement — wraps a_measure."""
        return asyncio.run(self.a_measure(test_case))

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async measurement via constrained JSON judge."""
        meta = getattr(test_case, "additional_metadata", None) or {}
        tool_trace = meta.get("tool_trace", [])

        tool_results = "\n".join(
            f"- {t.get('tool_name', '?')}: {t.get('result', '')[:200]}"
            for t in tool_trace
            if isinstance(t, dict)
        ) or "(no tool calls recorded)"

        expected = (
            f"Agent output should be grounded in tool results. "
            f"Check: Did the agent fabricate data not in tool outputs? "
            f"Did it claim actions it didn't take? "
            f"Tool results:\n{tool_results}"
        )

        verdict = await self.judge.evaluate(
            task_input=test_case.input,
            agent_output=test_case.actual_output or "",
            expected_behavior=expected,
            tool_trace=[t for t in tool_trace if isinstance(t, dict)],
        )
        self.score = verdict.score
        self.reason = verdict.reasoning
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return (self.score or 0) >= self.threshold
