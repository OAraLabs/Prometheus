"""Benchmark runner — execute test cases through AgentLoop and score results.

Source: Novel code for Prometheus Sprint 8.

CLI: python -m prometheus.benchmarks.runner --model qwen3.5-32b --tier 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from prometheus.benchmarks.suite import BenchmarkSuite, TestCase, load_suite
from prometheus.engine.agent_loop import AgentLoop, RunResult
from prometheus.engine.messages import ConversationMessage
from prometheus.providers.base import ModelProvider
from prometheus.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


class Score(str, Enum):
    """Benchmark scoring outcomes."""

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    RETRY_SUCCESS = "RETRY_SUCCESS"
    FAIL = "FAIL"
    CRASH = "CRASH"


@dataclass
class ScoreResult:
    """Result of scoring a single test case."""

    case_id: str
    case_name: str
    score: Score
    turns: int = 0
    latency_ms: float = 0.0
    details: str = ""
    tool_calls: list[str] = field(default_factory=list)


def _run_shell(commands: list[str]) -> None:
    """Run shell commands for setup/teardown."""
    for cmd in commands:
        subprocess.run(cmd, shell=True, capture_output=True, timeout=30)


def _score_result(case: TestCase, result: RunResult, cwd: Path) -> ScoreResult:
    """Score a RunResult against a TestCase's expectations."""
    tool_calls = []
    for msg in result.messages:
        for tu in msg.tool_uses:
            tool_calls.append(tu.name)

    checks_passed = 0
    checks_total = 0

    # Check expected tools were called
    if case.expected_tools:
        checks_total += 1
        if all(t in tool_calls for t in case.expected_tools):
            checks_passed += 1

    # Check output contains expected strings
    output_text = result.text.lower()
    for expected in case.expected_output_contains:
        checks_total += 1
        if expected.lower() in output_text:
            checks_passed += 1

    # Check output does NOT contain unwanted strings
    for unwanted in case.expected_output_not_contains:
        checks_total += 1
        if unwanted.lower() not in output_text:
            checks_passed += 1

    # Check expected files exist
    for fpath in case.expected_file_exists:
        checks_total += 1
        if Path(fpath).exists():
            checks_passed += 1

    # Check file content
    for fpath, expected_content in case.expected_file_contains.items():
        checks_total += 1
        try:
            content = Path(fpath).read_text()
            if expected_content in content:
                checks_passed += 1
        except OSError:
            pass

    if checks_total == 0:
        # No expectations — just check it didn't crash and used a tool
        score = Score.SUCCESS if tool_calls else Score.PARTIAL
    elif checks_passed == checks_total:
        score = Score.SUCCESS
    elif checks_passed > 0:
        score = Score.PARTIAL
    else:
        score = Score.FAIL

    return ScoreResult(
        case_id=case.id,
        case_name=case.name,
        score=score,
        turns=result.turns,
        details=f"{checks_passed}/{checks_total} checks passed",
        tool_calls=tool_calls,
    )


class BenchmarkRunner:
    """Run benchmark test cases through an AgentLoop."""

    def __init__(
        self,
        provider: ModelProvider,
        tool_registry: ToolRegistry,
        *,
        model: str = "qwen3.5-32b",
        max_tokens: int = 4096,
        cwd: Path | None = None,
        adapter: Any | None = None,
    ) -> None:
        self._provider = provider
        self._registry = tool_registry
        self._model = model
        self._max_tokens = max_tokens
        self._cwd = cwd or Path.cwd()
        self._adapter = adapter

    async def run_case(self, case: TestCase) -> ScoreResult:
        """Run a single test case and return its score."""
        # Setup
        if case.setup_commands:
            _run_shell(case.setup_commands)

        loop = AgentLoop(
            provider=self._provider,
            model=self._model,
            max_tokens=self._max_tokens,
            max_turns=case.max_turns,
            tool_registry=self._registry,
            adapter=self._adapter,
            cwd=self._cwd,
        )

        t0 = time.monotonic()
        try:
            result = await loop.run_async(
                system_prompt="You are a helpful assistant with access to tools. Complete the task.",
                user_message=case.prompt,
            )
            latency = (time.monotonic() - t0) * 1000
            score_result = _score_result(case, result, self._cwd)
            score_result.latency_ms = latency
        except Exception as exc:
            latency = (time.monotonic() - t0) * 1000
            score_result = ScoreResult(
                case_id=case.id,
                case_name=case.name,
                score=Score.CRASH,
                latency_ms=latency,
                details=str(exc),
            )

        # Teardown
        if case.teardown_commands:
            _run_shell(case.teardown_commands)

        return score_result

    async def run_suite(
        self,
        suite: BenchmarkSuite,
        *,
        concurrency: int = 1,
    ) -> list[ScoreResult]:
        """Run all cases in a suite. Returns list of ScoreResults."""
        results: list[ScoreResult] = []
        cases = suite.cases

        if concurrency <= 1:
            for case in cases:
                logger.info("Running %s: %s", case.id, case.name)
                sr = await self.run_case(case)
                logger.info("  -> %s (%s)", sr.score.value, sr.details)
                results.append(sr)
        else:
            sem = asyncio.Semaphore(concurrency)

            async def _run(c: TestCase) -> ScoreResult:
                async with sem:
                    logger.info("Running %s: %s", c.id, c.name)
                    sr = await self.run_case(c)
                    logger.info("  -> %s (%s)", sr.score.value, sr.details)
                    return sr

            results = list(await asyncio.gather(*[_run(c) for c in cases]))

        return results


def print_report(results: list[ScoreResult]) -> None:
    """Print a summary report of benchmark results."""
    counts = {s: 0 for s in Score}
    total_latency = 0.0

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    for r in results:
        status_icon = {
            Score.SUCCESS: "PASS",
            Score.PARTIAL: "PART",
            Score.RETRY_SUCCESS: "RTRY",
            Score.FAIL: "FAIL",
            Score.CRASH: "CRSH",
        }[r.score]
        print(f"  [{status_icon}] {r.case_id:30s} {r.details:20s} {r.latency_ms:8.0f}ms")
        counts[r.score] += 1
        total_latency += r.latency_ms

    print("-" * 70)
    total = len(results)
    success = counts[Score.SUCCESS] + counts[Score.RETRY_SUCCESS]
    print(f"  Total: {total}  |  Pass: {success}  |  Partial: {counts[Score.PARTIAL]}  "
          f"|  Fail: {counts[Score.FAIL]}  |  Crash: {counts[Score.CRASH]}")
    print(f"  Total latency: {total_latency:.0f}ms  |  Avg: {total_latency / max(total, 1):.0f}ms")
    print("=" * 70 + "\n")


def main() -> None:
    """CLI entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        prog="prometheus.benchmarks.runner",
        description="Run Prometheus benchmark suite against a model.",
    )
    parser.add_argument("--model", default="qwen3.5-32b", help="Model to benchmark")
    parser.add_argument("--tier", type=int, choices=[1, 2], default=None, help="Filter by tier")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="LLM server URL")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel test cases")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--case", type=str, default=None, help="Run a single test case by ID")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Import provider lazily — StubProvider uses httpx
    from prometheus.providers.stub import StubProvider
    from prometheus.tools.builtin import (
        AgentTool,
        BashTool,
        CronCreateTool,
        CronDeleteTool,
        CronListTool,
        FileEditTool,
        FileReadTool,
        FileWriteTool,
        GlobTool,
        GrepTool,
    )
    from prometheus.tools.builtin.skill import SkillTool
    from prometheus.tools.builtin.todo_write import TodoWriteTool

    provider = StubProvider(base_url=args.base_url)

    registry = ToolRegistry()
    registry.register(AgentTool())
    registry.register(BashTool(workspace="/tmp"))
    registry.register(CronCreateTool())
    registry.register(CronDeleteTool())
    registry.register(CronListTool())
    registry.register(FileEditTool())
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    registry.register(SkillTool())
    registry.register(TodoWriteTool())

    suite = load_suite(tier=args.tier)

    if args.case:
        case = suite.get(args.case)
        if case is None:
            print(f"Test case '{args.case}' not found.", file=sys.stderr)
            sys.exit(1)
        suite = BenchmarkSuite([case])

    runner = BenchmarkRunner(
        provider=provider,
        tool_registry=registry,
        model=args.model,
    )

    results = asyncio.run(runner.run_suite(suite, concurrency=args.concurrency))

    if args.json:
        out = []
        for r in results:
            out.append({
                "case_id": r.case_id,
                "case_name": r.case_name,
                "score": r.score.value,
                "turns": r.turns,
                "latency_ms": round(r.latency_ms, 1),
                "details": r.details,
                "tool_calls": r.tool_calls,
            })
        print(json.dumps(out, indent=2))
    else:
        print_report(results)

    # Exit code: 0 if all pass/partial, 1 if any fail/crash
    if any(r.score in (Score.FAIL, Score.CRASH) for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
