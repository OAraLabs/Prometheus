#!/usr/bin/env python3
"""
Vibe Check — Prometheus Web Agent Task Battery
===============================================
20 real-world web tasks. Tests model judgment, tool selection, multi-step
planning, and output quality — not just harness mechanics.

Two grading modes per task:
  Auto:  Did it produce the expected artifact? (pass/fail, no human needed)
  Human: Is the output actually good? (shown in the report for review)

Usage:
    python3 scripts/vibe_check.py                     # all 20 tasks
    python3 scripts/vibe_check.py --category lookup   # one category
    python3 scripts/vibe_check.py --task 09           # single task
    python3 scripts/vibe_check.py --verbose

Output:
    Console: live results with auto pass/fail
    ~/.prometheus/vibe_check/vibe_<ts>.md   — human review report
    ~/.prometheus/vibe_check/vibe_<ts>.json — machine-readable

Notes dir: ~/prometheus-notes/ (created if absent)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Reuse smoke-test wiring (config, provider, adapter, registry build, AgentLoop)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prometheus.__main__ import (  # noqa: E402
    create_adapter,
    create_security_gate,
    create_tool_registry,
    load_config,
)
from prometheus.engine import AgentLoop  # noqa: E402
from prometheus.providers.registry import ProviderRegistry  # noqa: E402
from prometheus.telemetry.tracker import ToolCallTelemetry  # noqa: E402


VIBE_NOTES_DIR = Path.home() / "prometheus-notes"
VIBE_OUTPUT_DIR = Path.home() / ".prometheus" / "vibe_check"

SYSTEM_PROMPT = (
    "You are a capable personal AI assistant with access to web tools.\n\n"
    "When a user pastes a URL, use web_fetch to retrieve it.\n"
    "When asked to search for something, use web_search.\n"
    "When asked to save something, use file_write.\n"
    "When asked about a YouTube video, use youtube_transcript if available.\n"
    "When asked to download a file, use download_file if available.\n\n"
    "If a URL returns a 404 or private error, say so clearly — don't keep retrying.\n"
    "If you can't complete a task, explain specifically why.\n"
    "Always cite where you got information from."
)

CIRCUIT_BREAKER_INDICATORS = (
    "circuit breaker tripped",
    "consecutive identical errors",
    "cannot produce valid tool calls",
    "unknown tool",
    "max_tool_iterations",
)

SKIPPED_MARKER = "skipped"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class VibeResult:
    """Result of one vibe check task."""

    task_id: str
    name: str
    category: str
    message: str

    auto_passed: bool
    auto_checks: list[str]
    auto_failures: list[str]

    human_review_prompt: str
    agent_output: str
    artifact_path: Optional[str]
    artifact_preview: Optional[str]

    duration_ms: float
    tools_used: list[str]
    circuit_breaker_tripped: bool
    timed_out: bool
    error: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expand(path: str) -> Path:
    return Path(os.path.expanduser(path))


def _detect_circuit_breaker(text: str, error: str) -> bool:
    haystack = (text + " " + error).lower()
    return any(ind in haystack for ind in CIRCUIT_BREAKER_INDICATORS)


def _tool_call_counts(report: dict[str, Any] | None) -> dict[str, int]:
    """Flatten telemetry.report() into {tool_name: total_calls}."""
    if not isinstance(report, dict):
        return {}
    tools = report.get("tools") or {}
    if not isinstance(tools, dict):
        return {}
    return {
        name: int(stats.get("calls", 0) if isinstance(stats, dict) else 0)
        for name, stats in tools.items()
    }


def _diff_tool_calls(
    before: dict[str, int], after: dict[str, int]
) -> list[str]:
    """Return tool names whose call count rose between snapshots."""
    return sorted(
        name for name, count in after.items()
        if count > before.get(name, 0)
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class VibeCheckRunner:
    """Runs vibe check tasks and generates a human-reviewable report."""

    def __init__(
        self,
        agent_loop: AgentLoop,
        telemetry: ToolCallTelemetry,
        config: dict[str, Any],
        verbose: bool = False,
    ) -> None:
        self.loop = agent_loop
        self.telemetry = telemetry
        self.config = config
        self.verbose = verbose
        self.results: list[VibeResult] = []
        self.registered_tools: set[str] = set()

    def _skip_result(
        self,
        task_id: str,
        name: str,
        category: str,
        message: str,
        human_review_prompt: str,
        missing_tool: str,
    ) -> VibeResult:
        result = VibeResult(
            task_id=task_id,
            name=name,
            category=category,
            message=message,
            auto_passed=False,
            auto_checks=[],
            auto_failures=[f"Skipped — required tool '{missing_tool}' not registered"],
            human_review_prompt=human_review_prompt,
            agent_output="",
            artifact_path=None,
            artifact_preview=None,
            duration_ms=0.0,
            tools_used=[],
            circuit_breaker_tripped=False,
            timed_out=False,
            error=SKIPPED_MARKER,
        )
        self.results.append(result)
        return result

    async def run_task(
        self,
        task_id: str,
        name: str,
        category: str,
        message: str,
        auto_checks: list[dict[str, Any]],
        human_review_prompt: str,
        time_budget_seconds: int = 60,
        expect_file: str | None = None,
        skip_if_tool_missing: str | None = None,
    ) -> VibeResult:
        if (
            skip_if_tool_missing
            and skip_if_tool_missing not in self.registered_tools
        ):
            print(
                f"  ⏭  TASK {task_id} {name} — skipped "
                f"(tool '{skip_if_tool_missing}' not registered)"
            )
            return self._skip_result(
                task_id, name, category, message,
                human_review_prompt, skip_if_tool_missing,
            )

        print(f"\n  🔍 TASK {task_id}: {name}")
        if self.verbose:
            print(f"     \"{message}\"")

        before_counts = _tool_call_counts(self.telemetry.report())

        start = time.monotonic()
        timed_out = False
        error = ""
        output = ""

        try:
            result = await asyncio.wait_for(
                self.loop.run_async(
                    system_prompt=SYSTEM_PROMPT,
                    user_message=message,
                    session_id="system",
                ),
                timeout=float(time_budget_seconds),
            )
            output = getattr(result, "text", str(result)) or ""
        except asyncio.TimeoutError:
            timed_out = True
            error = f"Timed out after {time_budget_seconds}s"
            output = ""
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            output = ""
            if self.verbose:
                traceback.print_exc()

        duration_ms = (time.monotonic() - start) * 1000.0
        circuit_breaker_tripped = _detect_circuit_breaker(output, error)

        after_counts = _tool_call_counts(self.telemetry.report())
        tools_used = _diff_tool_calls(before_counts, after_counts)

        auto_failures: list[str] = []
        if timed_out:
            auto_failures.append(f"Timed out after {time_budget_seconds}s")
        elif circuit_breaker_tripped:
            auto_failures.append("Circuit breaker tripped")
        elif error:
            auto_failures.append(f"Error: {error}")
        else:
            for check in auto_checks:
                ctype = check.get("type")
                value = check.get("value")
                if ctype == "contains":
                    if str(value).lower() not in output.lower():
                        auto_failures.append(f"Output missing '{value}'")
                elif ctype == "min_length":
                    if len(output) < int(value):
                        auto_failures.append(
                            f"Output too short ({len(output)} < {value} chars)"
                        )
                elif ctype == "file_exists":
                    p = _expand(str(value))
                    if not p.exists():
                        auto_failures.append(
                            f"Expected file not created: {value}"
                        )
                elif ctype == "not_contains":
                    if str(value).lower() in output.lower():
                        auto_failures.append(
                            f"Output should not contain '{value}'"
                        )
                else:
                    auto_failures.append(f"Unknown check type: {ctype}")

        auto_passed = not auto_failures

        artifact_path: Optional[str] = None
        artifact_preview: Optional[str] = None
        if expect_file:
            p = _expand(expect_file)
            if p.exists():
                artifact_path = str(p)
                try:
                    artifact_preview = p.read_text(encoding="utf-8",
                                                    errors="replace")[:500]
                except Exception:
                    artifact_preview = "(could not read)"

        auto_icon = "✅" if auto_passed else "❌"
        timer = f"({duration_ms:.0f}ms)"
        print(f"  {auto_icon} AUTO  {timer}")
        for f in auto_failures:
            print(f"     → {f}")
        print(f"  👁  HUMAN REVIEW: {human_review_prompt}")
        if self.verbose and output:
            preview = output[:300].replace("\n", " ")
            print(f"     Output: {preview}...")

        result_obj = VibeResult(
            task_id=task_id,
            name=name,
            category=category,
            message=message,
            auto_passed=auto_passed,
            auto_checks=[
                f"{c.get('type')}:{c.get('value')}" for c in auto_checks
            ],
            auto_failures=auto_failures,
            human_review_prompt=human_review_prompt,
            agent_output=output,
            artifact_path=artifact_path,
            artifact_preview=artifact_preview,
            duration_ms=duration_ms,
            tools_used=tools_used,
            circuit_breaker_tripped=circuit_breaker_tripped,
            timed_out=timed_out,
            error=error,
        )
        self.results.append(result_obj)
        return result_obj

    def generate_report(self, model_name: str) -> tuple[str, dict[str, Any]]:
        """Generate markdown report and JSON data."""
        skipped_count = sum(1 for r in self.results if r.error == SKIPPED_MARKER)
        runnable = [r for r in self.results if r.error != SKIPPED_MARKER]
        total = len(runnable)
        auto_passed = sum(1 for r in runnable if r.auto_passed)
        circuit_breaker_trips = sum(
            1 for r in self.results if r.circuit_breaker_tripped
        )
        timeouts = sum(1 for r in self.results if r.timed_out)

        categories: dict[str, dict[str, int]] = {}
        for r in self.results:
            cat = categories.setdefault(
                r.category, {"passed": 0, "failed": 0, "skipped": 0}
            )
            if r.error == SKIPPED_MARKER:
                cat["skipped"] += 1
            elif r.auto_passed:
                cat["passed"] += 1
            else:
                cat["failed"] += 1

        md = self._render_markdown(
            model_name=model_name,
            total=total,
            auto_passed=auto_passed,
            circuit_breaker_trips=circuit_breaker_trips,
            timeouts=timeouts,
            skipped=skipped_count,
            categories=categories,
        )

        data = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "summary": {
                "total": total,
                "auto_passed": auto_passed,
                "auto_failed": total - auto_passed,
                "circuit_breaker_trips": circuit_breaker_trips,
                "timeouts": timeouts,
                "skipped": skipped_count,
            },
            "categories": categories,
            "results": [
                {
                    "task_id": r.task_id,
                    "name": r.name,
                    "category": r.category,
                    "auto_passed": r.auto_passed,
                    "auto_failures": r.auto_failures,
                    "circuit_breaker_tripped": r.circuit_breaker_tripped,
                    "timed_out": r.timed_out,
                    "duration_ms": round(r.duration_ms, 0),
                    "tools_used": r.tools_used,
                    "agent_output_length": len(r.agent_output),
                    "has_artifact": r.artifact_path is not None,
                    "skipped": r.error == SKIPPED_MARKER,
                }
                for r in self.results
            ],
        }
        return md, data

    def _render_markdown(
        self,
        *,
        model_name: str,
        total: int,
        auto_passed: int,
        circuit_breaker_trips: int,
        timeouts: int,
        skipped: int,
        categories: dict[str, dict[str, int]],
    ) -> str:
        pct = (100 * auto_passed // total) if total else 0
        lines: list[str] = [
            "# Prometheus Vibe Check",
            f"**Model:** {model_name}  ",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
            f"**Auto Pass Rate:** {auto_passed}/{total} ({pct}%)  ",
            f"**Circuit Breaker Trips:** {circuit_breaker_trips}  ",
            f"**Timeouts:** {timeouts}  ",
            f"**Skipped:** {skipped}",
            "",
            "---",
            "",
            "## Category Summary",
            "",
        ]
        for cat, stats in sorted(categories.items()):
            run = stats["passed"] + stats["failed"]
            icon = "✅" if stats["failed"] == 0 and run > 0 else (
                "⏭" if run == 0 else "❌"
            )
            extra = (
                f" (+{stats['skipped']} skipped)" if stats["skipped"] else ""
            )
            lines.append(
                f"- {icon} **{cat}**: {stats['passed']}/{run} auto passed{extra}"
            )

        lines += ["", "---", "", "## Task Results", ""]

        for r in self.results:
            if r.error == SKIPPED_MARKER:
                icon = "⏭"
            elif r.auto_passed:
                icon = "✅"
            else:
                icon = "❌"
            lines.append(f"### TASK {r.task_id}: {r.name} {icon}")
            lines.append(f"**Category:** {r.category}  ")
            lines.append(f"**Duration:** {r.duration_ms:.0f}ms  ")
            if r.tools_used:
                lines.append(f"**Tools used:** {', '.join(r.tools_used)}  ")
            lines.append("")
            lines.append("**Prompt sent:**")
            lines.append(f"> {r.message}")
            lines.append("")

            if r.circuit_breaker_tripped:
                lines.append(
                    "⚠️ **CIRCUIT BREAKER TRIPPED** — model could not form valid tool calls"
                )
                lines.append("")
            elif r.timed_out:
                lines.append("⏰ **TIMED OUT**")
                lines.append("")
            elif r.error and r.error != SKIPPED_MARKER:
                lines.append(f"💥 **ERROR:** {r.error}")
                lines.append("")
            elif r.error == SKIPPED_MARKER:
                lines.append("⏭ **SKIPPED** — required tool not registered")
                lines.append("")

            if r.auto_failures:
                lines.append("**Auto check failures:**")
                for f in r.auto_failures:
                    lines.append(f"- {f}")
                lines.append("")

            lines.append(f"**👁 Human review:** {r.human_review_prompt}")
            lines.append("")

            if r.agent_output:
                lines.append("**Agent output:**")
                lines.append("```")
                lines.append(r.agent_output[:800])
                if len(r.agent_output) > 800:
                    lines.append(f"... ({len(r.agent_output)} chars total)")
                lines.append("```")
                lines.append("")

            if r.artifact_preview:
                lines.append(f"**Artifact** (`{r.artifact_path}`):")
                lines.append("```")
                lines.append(r.artifact_preview)
                lines.append("```")
                lines.append("")

            lines.append("---")
            lines.append("")

        trips = [r for r in self.results if r.circuit_breaker_tripped]
        if trips:
            lines += [
                "## 🚨 Circuit Breaker Trips — Fix These First",
                "",
                "These tasks caused the model to spin on malformed tool calls.",
                "Each one is a system prompt or tool description problem, not "
                "a model capability problem.",
                "",
            ]
            for r in trips:
                lines.append(f"- **{r.name}**: `{r.message[:80]}`")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The 20 tasks
# ---------------------------------------------------------------------------


VIBE_TASKS: list[dict[str, Any]] = [
    # ── Category 1: lookup ────────────────────────────────────────────
    {
        "task_id": "01",
        "name": "current_python_version",
        "category": "lookup",
        "message": "What's the latest stable version of Python?",
        "auto_checks": [
            {"type": "contains", "value": "3."},
            {"type": "min_length", "value": 20},
        ],
        "human_review_prompt": (
            "Is the version number actually current? Did it search or guess?"
        ),
        "time_budget_seconds": 30,
    },
    {
        "task_id": "02",
        "name": "library_license",
        "category": "lookup",
        "message": "What license does the FastAPI framework use?",
        "auto_checks": [{"type": "contains", "value": "MIT"}],
        "human_review_prompt": (
            "Did it cite a source or just guess from training data?"
        ),
        "time_budget_seconds": 30,
    },
    {
        "task_id": "03",
        "name": "repo_stars",
        "category": "lookup",
        "message": (
            "How many GitHub stars does the 'rich' Python library have?"
        ),
        "auto_checks": [{"type": "min_length", "value": 20}],
        "human_review_prompt": (
            "Is the number plausible (>40k)? Did it actually check GitHub?"
        ),
        "time_budget_seconds": 45,
    },
    {
        "task_id": "04",
        "name": "is_library_maintained",
        "category": "lookup",
        "message": "Is the Python library 'requests' still actively maintained?",
        "auto_checks": [{"type": "min_length", "value": 50}],
        "human_review_prompt": (
            "Did it check recent commits/releases or just say yes from memory?"
        ),
        "time_budget_seconds": 45,
    },
    {
        "task_id": "05",
        "name": "current_news",
        "category": "lookup",
        "message": "What's the most significant AI news from this week?",
        "auto_checks": [{"type": "min_length", "value": 100}],
        "human_review_prompt": (
            "Is it actually current news or hallucinated old events?"
        ),
        "time_budget_seconds": 60,
    },

    # ── Category 2: fetch ─────────────────────────────────────────────
    {
        "task_id": "06",
        "name": "fetch_and_summarize",
        "category": "fetch",
        "message": (
            "Summarize what's on this page: https://httpx.readthedocs.io/en/latest/"
        ),
        "auto_checks": [
            {"type": "contains", "value": "HTTP"},
            {"type": "min_length", "value": 100},
        ],
        "human_review_prompt": (
            "Is the summary accurate? Does it describe httpx correctly?"
        ),
        "time_budget_seconds": 45,
    },
    {
        "task_id": "07",
        "name": "fetch_and_answer",
        "category": "fetch",
        "message": (
            "According to https://docs.python.org/3/library/asyncio-task.html — "
            "what does asyncio.gather() do?"
        ),
        "auto_checks": [
            {"type": "contains", "value": "gather"},
            {"type": "min_length", "value": 50},
        ],
        "human_review_prompt": (
            "Is the answer correct? Does it seem like it actually read the page?"
        ),
        "time_budget_seconds": 45,
    },
    {
        "task_id": "08",
        "name": "fetch_json_api",
        "category": "fetch",
        "message": (
            "Fetch https://api.github.com/repos/tiangolo/fastapi and tell me "
            "the description and star count"
        ),
        "auto_checks": [{"type": "min_length", "value": 30}],
        "human_review_prompt": (
            "Does the star count match the actual FastAPI repo?"
        ),
        "time_budget_seconds": 30,
    },
    {
        "task_id": "09",
        "name": "private_url_graceful",
        "category": "fetch",
        "message": "What's in this repo? https://github.com/whieber1/leaky",
        "auto_checks": [
            {"type": "not_contains", "value": "circuit breaker"},
            {"type": "min_length", "value": 20},
        ],
        "human_review_prompt": (
            "Did it handle the private repo gracefully with a clear explanation?"
        ),
        "time_budget_seconds": 30,
    },

    # ── Category 3: save ──────────────────────────────────────────────
    {
        "task_id": "10",
        "name": "research_and_save",
        "category": "save",
        "message": (
            "Search for what's new in Python 3.13, summarize the top 3 features, "
            "and save it to ~/prometheus-notes/python313.md"
        ),
        "auto_checks": [
            {"type": "file_exists", "value": "~/prometheus-notes/python313.md"},
        ],
        "human_review_prompt": (
            "Is the content accurate? Are the features correct for Python 3.13?"
        ),
        "time_budget_seconds": 90,
        "expect_file": "~/prometheus-notes/python313.md",
    },
    {
        "task_id": "11",
        "name": "fetch_and_save_cheatsheet",
        "category": "save",
        "message": (
            "Fetch https://www.python-httpx.org/quickstart/ and save a cheat "
            "sheet of the most useful code examples to "
            "~/prometheus-notes/httpx_cheatsheet.md"
        ),
        "auto_checks": [
            {"type": "file_exists",
             "value": "~/prometheus-notes/httpx_cheatsheet.md"},
        ],
        "human_review_prompt": (
            "Would you actually use this cheat sheet? Does it have real code "
            "examples?"
        ),
        "time_budget_seconds": 90,
        "expect_file": "~/prometheus-notes/httpx_cheatsheet.md",
    },
    {
        "task_id": "12",
        "name": "download_and_confirm",
        "category": "save",
        "message": (
            "Download https://raw.githubusercontent.com/tiangolo/fastapi/master/README.md "
            "and save it to ~/prometheus-notes/fastapi_readme.md — then tell "
            "me how many lines it is"
        ),
        "auto_checks": [
            {"type": "file_exists",
             "value": "~/prometheus-notes/fastapi_readme.md"},
            {"type": "min_length", "value": 20},
        ],
        "human_review_prompt": (
            "Is the line count accurate? "
            "(check with `wc -l ~/prometheus-notes/fastapi_readme.md`)"
        ),
        "time_budget_seconds": 45,
        "expect_file": "~/prometheus-notes/fastapi_readme.md",
    },
    {
        "task_id": "13",
        "name": "research_comparison",
        "category": "save",
        "message": (
            "Compare Pydantic v1 vs v2 — what are the main breaking changes? "
            "Save a brief summary to ~/prometheus-notes/pydantic_migration.md"
        ),
        "auto_checks": [
            {"type": "file_exists",
             "value": "~/prometheus-notes/pydantic_migration.md"},
        ],
        "human_review_prompt": (
            "Is the migration info accurate and useful for a developer?"
        ),
        "time_budget_seconds": 120,
        "expect_file": "~/prometheus-notes/pydantic_migration.md",
    },

    # ── Category 4: recommend ─────────────────────────────────────────
    {
        "task_id": "14",
        "name": "find_library",
        "category": "recommend",
        "message": (
            "I need a Python library for fuzzy string matching. Find me the "
            "best option, check its GitHub, and tell me if it's actively "
            "maintained."
        ),
        "auto_checks": [{"type": "min_length", "value": 100}],
        "human_review_prompt": (
            "Is the recommendation actually good? (rapidfuzz is the right "
            "answer). Did it check the repo?"
        ),
        "time_budget_seconds": 90,
    },
    {
        "task_id": "15",
        "name": "evaluate_repo",
        "category": "recommend",
        "message": (
            "Look at https://github.com/encode/httpx — is this library "
            "production-ready? Give me a yes/no with 3 reasons."
        ),
        "auto_checks": [{"type": "min_length", "value": 100}],
        "human_review_prompt": (
            "Are the 3 reasons accurate and based on actual repo data vs "
            "generic claims?"
        ),
        "time_budget_seconds": 60,
    },
    {
        "task_id": "16",
        "name": "find_tutorial",
        "category": "recommend",
        "message": (
            "Find me a good beginner tutorial for learning asyncio in Python. "
            "Give me the URL and a one-sentence description of why it's good."
        ),
        "auto_checks": [{"type": "contains", "value": "http"}],
        "human_review_prompt": (
            "Is the linked resource actually good and beginner-friendly?"
        ),
        "time_budget_seconds": 60,
    },
    {
        "task_id": "17",
        "name": "find_recent_release",
        "category": "recommend",
        "message": (
            "What's the latest version of FastAPI and when was it released? "
            "Any notable changes?"
        ),
        "auto_checks": [{"type": "min_length", "value": 50}],
        "human_review_prompt": (
            "Is the version current? Did it find actual release notes or guess?"
        ),
        "time_budget_seconds": 45,
    },

    # ── Category 5: multistep ─────────────────────────────────────────
    {
        "task_id": "18",
        "name": "research_save_and_report",
        "category": "multistep",
        "message": (
            "Research the top 3 Python async HTTP libraries (httpx, aiohttp, "
            "requests). For each find: GitHub stars, last release date, "
            "license. Save a comparison table to "
            "~/prometheus-notes/async_http_comparison.md and tell me which "
            "one you'd recommend."
        ),
        "auto_checks": [
            {"type": "file_exists",
             "value": "~/prometheus-notes/async_http_comparison.md"},
        ],
        "human_review_prompt": (
            "Is the table accurate? Is the recommendation reasonable with "
            "justification?"
        ),
        "time_budget_seconds": 180,
        "expect_file": "~/prometheus-notes/async_http_comparison.md",
    },
    {
        "task_id": "19",
        "name": "youtube_summary",
        "category": "multistep",
        "message": (
            "Summarize this talk: "
            "https://www.youtube.com/watch?v=zjkBMFhNj_g — "
            "what are the 3 main things Karpathy explains?"
        ),
        "auto_checks": [{"type": "min_length", "value": 100}],
        "human_review_prompt": (
            "Are the 3 points accurate to the actual Karpathy video?"
        ),
        "time_budget_seconds": 120,
        "skip_if_tool_missing": "youtube_transcript",
    },
    {
        "task_id": "20",
        "name": "end_to_end_research",
        "category": "multistep",
        "message": (
            "I'm thinking of using Pydantic for data validation in a new "
            "project. Search for recent issues people have had with it, read "
            "at least one relevant page, and give me an honest assessment of "
            "whether I should use it — with specific concerns if any."
        ),
        "auto_checks": [
            {"type": "contains", "value": "Pydantic"},
            {"type": "min_length", "value": 300},
        ],
        "human_review_prompt": (
            "Does it feel like a genuinely informed opinion? Did it actually "
            "search for issues or just give generic advice?"
        ),
        "time_budget_seconds": 180,
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_runtime(verbose: bool) -> tuple[VibeCheckRunner, str]:
    config = load_config()
    print(
        f"Config loaded: provider="
        f"{config.get('model', {}).get('provider', 'unknown')}"
    )

    try:
        provider = ProviderRegistry.create(config["model"])
    except Exception as exc:
        print(f"❌ Cannot create provider: {exc}")
        sys.exit(2)

    security_cfg = config.get("security", {})
    security_gate = create_security_gate(security_cfg)
    registry = create_tool_registry(security_cfg, security_gate=security_gate)

    schemas = registry.list_schemas()
    registered_names = {s["name"] for s in schemas if isinstance(s, dict)}
    web_relevant = sorted(
        n for n in registered_names
        if any(kw in n for kw in
               ("web", "browser", "youtube", "download", "search", "fetch"))
    )
    print(f"Tools registered: {len(registered_names)}")
    print(f"Web-relevant tools: {', '.join(web_relevant) if web_relevant else '(none)'}")

    model_cfg = config.get("model", {})
    adapter = create_adapter(model_cfg, config.get("adapter"))
    print(f"Adapter tier: {adapter.tier}")

    if (
        model_cfg.get("grammar_enforcement", True)
        and hasattr(provider, "set_grammar")
        and adapter is not None
    ):
        grammar = adapter.generate_grammar(registry)
        if grammar:
            provider.set_grammar(grammar)

    telemetry = ToolCallTelemetry()
    model_name = model_cfg.get("model", "unknown")
    loop = AgentLoop(
        provider=provider,
        model=model_name,
        tool_registry=registry,
        adapter=adapter,
        permission_checker=security_gate,
        telemetry=telemetry,
    )
    runner = VibeCheckRunner(
        agent_loop=loop,
        telemetry=telemetry,
        config=config,
        verbose=verbose,
    )
    runner.registered_tools = registered_names
    return runner, model_name


async def main(args: argparse.Namespace) -> int:
    print("🎯 Prometheus Vibe Check — Web Agent Task Battery")
    print("=" * 55)

    VIBE_NOTES_DIR.mkdir(parents=True, exist_ok=True)
    VIBE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runner, model_name = build_runtime(verbose=args.verbose)
    print("Agent loop ready")
    print("=" * 55)

    tasks = list(VIBE_TASKS)
    if args.category:
        tasks = [t for t in tasks if t["category"] == args.category]
    if args.task:
        normalized = args.task.zfill(2)
        tasks = [t for t in tasks if t["task_id"] == normalized]
    if not tasks:
        print("No tasks match the given filters.")
        return 2

    print(f"Running {len(tasks)} task(s)…")

    for task in tasks:
        await runner.run_task(**task)
        await asyncio.sleep(2)

    md, data = runner.generate_report(model_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    md_path = VIBE_OUTPUT_DIR / f"vibe_{timestamp}.md"
    json_path = VIBE_OUTPUT_DIR / f"vibe_{timestamp}.json"
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print("\n" + "=" * 55)
    print("📊 VIBE CHECK RESULTS")
    print("=" * 55)
    summary = data["summary"]
    total = summary["total"]
    passed = summary["auto_passed"]
    pct = (100 * passed // total) if total else 0
    print(f"  Auto pass: {passed}/{total} ({pct}%)")
    if summary["circuit_breaker_trips"]:
        print(
            f"  ⚠️  Circuit breaker trips: "
            f"{summary['circuit_breaker_trips']} — fix these first"
        )
    if summary["timeouts"]:
        print(f"  ⏰ Timeouts: {summary['timeouts']}")
    if summary["skipped"]:
        print(f"  ⏭  Skipped: {summary['skipped']}")
    print(f"\n  Full report: {md_path}")
    print(f"  Machine-readable: {json_path}")
    print("  👁  Open the report to complete human review\n")

    needs_review = [
        r for r in runner.results
        if not r.timed_out and r.error != SKIPPED_MARKER
    ]
    if needs_review:
        print("👁  HUMAN REVIEW QUEUE")
        print("-" * 40)
        for r in needs_review:
            icon = "✅" if r.auto_passed else "❌"
            print(f"  {icon} TASK {r.task_id} {r.name}")
            print(f"     {r.human_review_prompt}")
            if r.artifact_path:
                print(f"     Artifact: {r.artifact_path}")
        print()

    return 0 if summary["auto_failed"] == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prometheus vibe check")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--category", type=str, default=None,
                        help="Run one category: lookup|fetch|save|recommend|multistep")
    parser.add_argument("--task", type=str, default=None,
                        help="Run a single task by ID: 01-20")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)))
