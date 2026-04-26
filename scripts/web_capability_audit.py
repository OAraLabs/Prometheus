#!/usr/bin/env python3
"""
Web Capability Audit (WEAVE Session B)
=======================================
Runs ~40 real-world web research tasks through Prometheus against the
configured local model. Surfaces what the agent can and cannot do as
a web research assistant.

Usage:
    python3 scripts/web_capability_audit.py
    python3 scripts/web_capability_audit.py --verbose
    python3 scripts/web_capability_audit.py --category search
    python3 scripts/web_capability_audit.py --time-limit 3600

Categories:
    search      web_search tool tests
    fetch       web_fetch tool tests
    youtube     youtube_transcript tests
    download    download_file tests
    research    multi-step research workflows
    graceful    error handling and edge cases
    railway     Railway CLI operations

Output:
    Console: live pass/fail with timing
    JSON: ~/.prometheus/audits/web_audit_<ts>.json
    MD:   ~/.prometheus/audits/web_audit_<ts>.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Reuse smoke test infrastructure
sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_test_tool_calling import SmokeTestRunner, TestResult  # type: ignore  # noqa: E402

from prometheus.__main__ import (  # noqa: E402
    create_adapter,
    create_security_gate,
    create_tool_registry,
    load_config,
)
from prometheus.engine import AgentLoop  # noqa: E402
from prometheus.providers.registry import ProviderRegistry  # noqa: E402
from prometheus.telemetry.tracker import ToolCallTelemetry  # noqa: E402


# ── Failure categorization ──────────────────────────────────────────

FAILURE_CIRCUIT_BREAKER = "circuit_breaker"
FAILURE_WRONG_TOOL = "wrong_tool"
FAILURE_EXECUTION = "execution_failure"
FAILURE_WRONG_ANSWER = "wrong_answer"
FAILURE_TIMEOUT = "timeout"

CIRCUIT_BREAKER_INDICATORS = (
    "circuit breaker tripped",
    "consecutive identical errors",
    "unknown tool",
    "cannot produce valid tool calls",
    "tool call limit",
    "max iterations exceeded",
    "max_tool_iterations",
)


AUDIT_WORKSPACE = Path("/tmp/prometheus-web-audit")
AUDIT_OUTPUT_DIR = Path.home() / ".prometheus" / "audits"

SYSTEM_PROMPT = (
    "You are Prometheus, a web research assistant. You have access to tools "
    "for web search, URL fetching, YouTube transcripts, file downloads, and "
    "shell commands. Use the right tool for each task and be concise. When "
    "a URL is provided, prefer fetching it over searching. If something "
    "fails, explain what went wrong rather than retrying blindly."
)


@dataclass
class AuditTestResult(TestResult):
    """Extends TestResult with audit-specific fields."""

    failure_category: str = ""
    tools_attempted: list[str] = field(default_factory=list)
    circuit_breaker_tripped: bool = False
    time_budget_seconds: float = 0
    over_budget: bool = False


@dataclass
class WebAuditRunner(SmokeTestRunner):
    """SmokeTestRunner + audit-specific tracking."""

    time_limit_seconds: float = 3600
    started_at: float = 0
    expect_no_circuit_breaker_default: bool = False

    def has_time_remaining(self) -> bool:
        if self.started_at == 0:
            return True
        return (time.monotonic() - self.started_at) < self.time_limit_seconds

    @staticmethod
    def detect_circuit_breaker(text: str, error: str) -> bool:
        haystack = (text + " " + error).lower()
        return any(ind in haystack for ind in CIRCUIT_BREAKER_INDICATORS)

    async def run_audit_test(
        self,
        *,
        name: str,
        category: str,
        message: str,
        expect_tools: Optional[list[str]] = None,
        expect_in_output: Optional[str] = None,
        expect_in_output_any: Optional[list[str]] = None,
        expect_file_exists: Optional[str] = None,
        expect_no_circuit_breaker: bool = False,
        time_budget: float = 60,
        max_iterations: int = 10,
    ) -> AuditTestResult:
        """Audit-aware version of run_test.

        ``expect_in_output_any`` passes if ANY of the listed substrings appears.
        ``expect_no_circuit_breaker=True`` forces a fail if a CB trip is detected
        regardless of other expectations.
        """
        if not self.has_time_remaining():
            print(f"  ⏰ {name} skipped — time budget exhausted")
            return AuditTestResult(
                name=name, category=category, passed=False, duration_ms=0,
                error="skipped: time budget exhausted",
                failure_category=FAILURE_TIMEOUT,
                time_budget_seconds=time_budget,
            )

        if self.verbose:
            print(f"\n  ▶ {name}")
            print(f"    Message: {message[:80]}{'...' if len(message) > 80 else ''}")

        start = time.monotonic()
        try:
            out = await self.run_agent(message, max_iterations=max_iterations)
            text = out["text"] or ""
            elapsed_ms = out["elapsed_ms"]
            elapsed_s = elapsed_ms / 1000.0
            err_msg = ""
        except Exception as exc:
            text = ""
            elapsed_ms = (time.monotonic() - start) * 1000
            elapsed_s = elapsed_ms / 1000.0
            err_msg = f"{type(exc).__name__}: {exc}"
            if self.verbose:
                traceback.print_exc()

        cb_tripped = self.detect_circuit_breaker(text, err_msg)
        over_budget = elapsed_s > time_budget

        errors: list[str] = []

        if expect_no_circuit_breaker and cb_tripped:
            errors.append("Circuit breaker tripped (graceful handling expected)")

        if expect_in_output and expect_in_output.lower() not in text.lower():
            errors.append(
                f"Expected '{expect_in_output}' in output, got: {text[:160]}"
            )

        if expect_in_output_any:
            lowered = text.lower()
            if not any(s.lower() in lowered for s in expect_in_output_any):
                errors.append(
                    f"Expected any of {expect_in_output_any} in output, "
                    f"got: {text[:160]}"
                )

        if expect_file_exists:
            p = Path(expect_file_exists)
            if not p.exists():
                errors.append(f"Expected file {expect_file_exists} to exist")

        tools_called: list[str] = []
        try:
            report = self.telemetry.report() if hasattr(self.telemetry, "report") else {}
            by_tool = report.get("by_tool", {}) if isinstance(report, dict) else {}
            tools_called = list(by_tool.keys()) if isinstance(by_tool, dict) else []
        except Exception:
            tools_called = []

        if expect_tools:
            if not any(t in tools_called for t in expect_tools):
                errors.append(
                    f"Expected one of tools {expect_tools} to be called, "
                    f"saw: {tools_called or '(none)'}"
                )

        if err_msg and not errors:
            errors.append(err_msg)

        passed = not errors
        failure_category = ""
        if not passed:
            if cb_tripped:
                failure_category = FAILURE_CIRCUIT_BREAKER
            elif expect_tools and not any(t in tools_called for t in expect_tools):
                failure_category = FAILURE_WRONG_TOOL
            elif over_budget:
                failure_category = FAILURE_TIMEOUT
            elif err_msg:
                failure_category = FAILURE_EXECUTION
            else:
                failure_category = FAILURE_WRONG_ANSWER

        result = AuditTestResult(
            name=name,
            category=category,
            passed=passed,
            duration_ms=elapsed_ms,
            details=text[:300] if self.verbose else "",
            error="; ".join(errors) if errors else "",
            tools_called=tools_called,
            failure_category=failure_category,
            tools_attempted=tools_called,
            circuit_breaker_tripped=cb_tripped,
            time_budget_seconds=time_budget,
            over_budget=over_budget,
        )
        self.results.append(result)

        if result.passed:
            status = "✅"
        else:
            status = "❌"
        timing = f"({elapsed_s:.1f}s)"
        print(f"  {status} {name} {timing}")
        if not result.passed:
            print(f"     → {result.error}")
        return result


# ===========================================================================
# Audit task definitions (~40 tasks)
# ===========================================================================


async def category_search(runner: WebAuditRunner) -> None:
    print("\n━━━ Web Search ━━━")
    cat = "search"
    await runner.run_audit_test(
        name="search_current_python_version",
        category=cat,
        message="What is the current stable version of Python? Use web_search.",
        expect_in_output="3.",
        expect_tools=["web_search"],
        time_budget=45,
    )
    await runner.run_audit_test(
        name="search_recent_news",
        category=cat,
        message="Search for recent news about open source AI models.",
        expect_tools=["web_search"],
        time_budget=60,
    )
    await runner.run_audit_test(
        name="search_library_docs",
        category=cat,
        message="Find the documentation URL for the Python httpx library.",
        expect_in_output="httpx",
        expect_tools=["web_search"],
        time_budget=45,
    )
    await runner.run_audit_test(
        name="search_and_summarize",
        category=cat,
        message=(
            "Search for 'llama.cpp performance benchmarks' and give me "
            "a 3-sentence summary of what you find."
        ),
        expect_tools=["web_search"],
        time_budget=90,
    )
    await runner.run_audit_test(
        name="search_github_repo",
        category=cat,
        message="Find the GitHub repository for the 'httpx' Python library.",
        expect_in_output_any=["github.com", "encode/httpx"],
        time_budget=60,
    )
    await runner.run_audit_test(
        name="search_compare_options",
        category=cat,
        message=(
            "Search for Python libraries for parsing HTML. List the top 3 "
            "options with one-line descriptions."
        ),
        time_budget=90,
    )
    await runner.run_audit_test(
        name="search_specific_fact",
        category=cat,
        message="What is the license of the FastAPI Python framework?",
        expect_in_output="MIT",
        time_budget=60,
    )
    await runner.run_audit_test(
        name="search_no_results_handled",
        category=cat,
        message="Search for 'xyzzy_frobnosticator_quux_2026_internal_unicorn'.",
        expect_in_output_any=["no results", "nothing found", "couldn't find",
                              "no matches", "not found"],
        time_budget=45,
    )


async def category_fetch(runner: WebAuditRunner) -> None:
    print("\n━━━ Web Fetch ━━━")
    cat = "fetch"
    await runner.run_audit_test(
        name="fetch_public_page",
        category=cat,
        message=(
            "Fetch https://www.python.org/about/ and tell me in one sentence "
            "what Python is."
        ),
        expect_tools=["web_fetch"],
        time_budget=45,
    )
    await runner.run_audit_test(
        name="fetch_github_readme",
        category=cat,
        message=(
            "Fetch https://raw.githubusercontent.com/encode/httpx/master/README.md "
            "and tell me what HTTP versions it supports."
        ),
        expect_tools=["web_fetch"],
        time_budget=45,
    )
    await runner.run_audit_test(
        name="fetch_json_api",
        category=cat,
        message=(
            "Fetch https://api.github.com/repos/encode/httpx and tell me "
            "the star count."
        ),
        expect_tools=["web_fetch"],
        expect_in_output_any=["star", "stargazer"],
        time_budget=45,
    )
    await runner.run_audit_test(
        name="fetch_404_graceful",
        category=cat,
        message=(
            "Fetch https://www.python.org/this-page-does-not-exist-xyz "
            "and tell me what you find."
        ),
        expect_in_output_any=["404", "not found", "error", "unavailable"],
        time_budget=45,
    )
    await runner.run_audit_test(
        name="fetch_nonexistent_domain",
        category=cat,
        message=(
            "Fetch https://this-domain-truly-does-not-exist-prometheus-test.invalid"
        ),
        expect_in_output_any=["error", "not found", "unavailable", "cannot",
                              "failed"],
        expect_no_circuit_breaker=True,
        time_budget=45,
    )
    await runner.run_audit_test(
        name="fetch_pypi_metadata",
        category=cat,
        message=(
            "Fetch https://pypi.org/pypi/httpx/json and tell me the latest "
            "version of httpx."
        ),
        expect_tools=["web_fetch"],
        time_budget=45,
    )
    await runner.run_audit_test(
        name="fetch_and_save_summary",
        category=cat,
        message=(
            "Fetch https://docs.python.org/3/library/asyncio.html, summarize "
            f"in 3 bullets, and save the summary to {AUDIT_WORKSPACE}/asyncio_summary.txt"
        ),
        expect_tools=["web_fetch"],
        expect_file_exists=f"{AUDIT_WORKSPACE}/asyncio_summary.txt",
        time_budget=120,
    )
    await runner.run_audit_test(
        name="fetch_hn_front_page",
        category=cat,
        message="Fetch https://news.ycombinator.com and list the top 3 story titles.",
        expect_tools=["web_fetch"],
        time_budget=60,
    )


async def category_youtube(runner: WebAuditRunner) -> None:
    print("\n━━━ YouTube Transcript ━━━")
    cat = "youtube"
    short_video = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    await runner.run_audit_test(
        name="youtube_transcript_basic",
        category=cat,
        message=f"Get the transcript of this YouTube video: {short_video}",
        expect_tools=["youtube_transcript"],
        time_budget=90,
    )
    await runner.run_audit_test(
        name="youtube_short_url",
        category=cat,
        message="Get the transcript from https://youtu.be/dQw4w9WgXcQ",
        expect_tools=["youtube_transcript"],
        time_budget=90,
    )
    await runner.run_audit_test(
        name="youtube_save_transcript",
        category=cat,
        message=(
            f"Get the transcript of {short_video} and save it to "
            f"{AUDIT_WORKSPACE}/transcript.txt"
        ),
        expect_tools=["youtube_transcript"],
        expect_file_exists=f"{AUDIT_WORKSPACE}/transcript.txt",
        time_budget=120,
    )
    await runner.run_audit_test(
        name="youtube_invalid_id_graceful",
        category=cat,
        message="Get the transcript from https://www.youtube.com/watch?v=ZZZZZZZZZZZ",
        expect_in_output_any=["unavailable", "error", "not found", "no transcript",
                              "private", "invalid"],
        expect_no_circuit_breaker=True,
        time_budget=60,
    )
    await runner.run_audit_test(
        name="youtube_summarize_after",
        category=cat,
        message=(
            f"Get the transcript of {short_video} and tell me the song's "
            "main theme in one sentence."
        ),
        expect_tools=["youtube_transcript"],
        time_budget=120,
    )


async def category_download(runner: WebAuditRunner) -> None:
    print("\n━━━ File Download ━━━")
    cat = "download"
    AUDIT_WORKSPACE.mkdir(parents=True, exist_ok=True)
    await runner.run_audit_test(
        name="download_text_file",
        category=cat,
        message=(
            "Download https://raw.githubusercontent.com/encode/httpx/master/README.md "
            f"and save it to {AUDIT_WORKSPACE}/httpx_readme.md"
        ),
        expect_tools=["download_file"],
        expect_file_exists=f"{AUDIT_WORKSPACE}/httpx_readme.md",
        time_budget=60,
    )
    await runner.run_audit_test(
        name="download_json",
        category=cat,
        message=(
            "Download https://api.github.com/repos/encode/httpx and save it to "
            f"{AUDIT_WORKSPACE}/httpx_meta.json"
        ),
        expect_tools=["download_file"],
        expect_file_exists=f"{AUDIT_WORKSPACE}/httpx_meta.json",
        time_budget=45,
    )
    await runner.run_audit_test(
        name="download_default_location",
        category=cat,
        message="Download https://www.python.org/robots.txt — save it wherever makes sense.",
        expect_tools=["download_file"],
        time_budget=45,
    )
    await runner.run_audit_test(
        name="download_overwrite_blocked",
        category=cat,
        message=(
            f"Download https://raw.githubusercontent.com/encode/httpx/master/README.md "
            f"and save it to {AUDIT_WORKSPACE}/httpx_readme.md (this file already exists)."
        ),
        expect_in_output_any=["exists", "overwrite", "already"],
        time_budget=45,
    )


async def category_research(runner: WebAuditRunner) -> None:
    print("\n━━━ Multi-Step Research ━━━")
    cat = "research"
    await runner.run_audit_test(
        name="research_compare_libraries",
        category=cat,
        message=(
            "Compare httpx and requests for Python HTTP. Search for one, "
            "give me a one-paragraph recommendation with reasoning."
        ),
        expect_tools=["web_search"],
        time_budget=180,
    )
    await runner.run_audit_test(
        name="research_github_repo_analysis",
        category=cat,
        message=(
            "Look up the GitHub repo for 'fastapi'. Tell me: star count, "
            "license, and a one-sentence description."
        ),
        time_budget=120,
    )
    await runner.run_audit_test(
        name="research_technical_question",
        category=cat,
        message=(
            "Search for and explain in simple terms how Python's asyncio "
            "event loop differs from threading. 3 sentences max."
        ),
        time_budget=180,
    )
    await runner.run_audit_test(
        name="research_save_structured_report",
        category=cat,
        message=(
            "Look up the latest version of 'httpx' and 'requests' on PyPI. "
            f"Save a small comparison to {AUDIT_WORKSPACE}/http_libs.md "
            "(library, latest version, license)."
        ),
        expect_file_exists=f"{AUDIT_WORKSPACE}/http_libs.md",
        time_budget=240,
    )


async def category_graceful(runner: WebAuditRunner) -> None:
    print("\n━━━ Graceful Error Handling ━━━")
    cat = "graceful"
    await runner.run_audit_test(
        name="bare_url_paste",
        category=cat,
        message="https://www.python.org/about/",
        expect_no_circuit_breaker=True,
        time_budget=60,
    )
    await runner.run_audit_test(
        name="github_org_url",
        category=cat,
        message="https://github.com/OAraLabs?tab=repositories",
        expect_no_circuit_breaker=True,
        time_budget=60,
    )
    await runner.run_audit_test(
        name="ambiguous_request",
        category=cat,
        message="leaky",
        expect_no_circuit_breaker=True,
        time_budget=45,
    )
    await runner.run_audit_test(
        name="search_no_results",
        category=cat,
        message="Search for 'xyzzy_frobnosticator_quux_2026_prometheus_internal'.",
        expect_in_output_any=["no results", "nothing found", "couldn't find",
                              "not found", "no matches"],
        expect_no_circuit_breaker=True,
        time_budget=45,
    )
    await runner.run_audit_test(
        name="youtube_invalid_url",
        category=cat,
        message="Get the transcript from https://youtube.com/watch?v=NOT_A_VIDEO_AT_ALL",
        expect_no_circuit_breaker=True,
        expect_in_output_any=["error", "unavailable", "invalid", "not found"],
        time_budget=60,
    )


async def category_railway(runner: WebAuditRunner) -> None:
    print("\n━━━ Railway CLI ━━━")
    cat = "railway"
    if shutil.which("railway") is None:
        print("  ⏭  Railway CLI not installed — skipping category")
        return
    await runner.run_audit_test(
        name="railway_version",
        category=cat,
        message="What version of the Railway CLI is installed? Use bash to check.",
        expect_tools=["bash"],
        expect_in_output_any=["railway", "version", "."],
        time_budget=30,
    )
    await runner.run_audit_test(
        name="railway_help",
        category=cat,
        message="Show me the Railway CLI help output (run: railway --help).",
        expect_tools=["bash"],
        time_budget=30,
    )
    await runner.run_audit_test(
        name="railway_status",
        category=cat,
        message="Check Railway login status — am I logged in? Run: railway whoami",
        expect_tools=["bash"],
        time_budget=30,
    )


CATEGORY_FUNCS = {
    "search": category_search,
    "fetch": category_fetch,
    "youtube": category_youtube,
    "download": category_download,
    "research": category_research,
    "graceful": category_graceful,
    "railway": category_railway,
}


# ===========================================================================
# Reports
# ===========================================================================


def _category_summary(results: list[AuditTestResult]) -> dict[str, dict[str, Any]]:
    by_cat: dict[str, list[AuditTestResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
    out: dict[str, dict[str, Any]] = {}
    for cat, items in by_cat.items():
        passed = sum(1 for x in items if x.passed)
        out[cat] = {
            "passed": passed,
            "failed": len(items) - passed,
            "pass_rate": (passed / len(items)) if items else 0.0,
        }
    return out


def _failure_breakdown(results: list[AuditTestResult]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results:
        if not r.passed and r.failure_category:
            out[r.failure_category] = out.get(r.failure_category, 0) + 1
    return out


def write_reports(
    results: list[AuditTestResult],
    *,
    model: str,
    duration_seconds: float,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"web_audit_{ts}.json"
    md_path = output_dir / f"web_audit_{ts}.md"

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    pass_rate = (passed / total) if total else 0.0
    cats = _category_summary(results)
    fail_breakdown = _failure_breakdown(results)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "duration_seconds": round(duration_seconds, 1),
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(pass_rate, 3),
        "categories": cats,
        "failure_breakdown": fail_breakdown,
        "results": [
            {
                "name": r.name,
                "category": r.category,
                "passed": r.passed,
                "duration_ms": round(r.duration_ms, 0),
                "tools_called": r.tools_called,
                "circuit_breaker_tripped": r.circuit_breaker_tripped,
                "failure_category": r.failure_category,
                "over_budget": r.over_budget,
                "time_budget_seconds": r.time_budget_seconds,
                "error": r.error,
            }
            for r in results
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2))

    md = _render_markdown(model, duration_seconds, total, passed,
                          failed, pass_rate, cats, fail_breakdown, results)
    md_path.write_text(md)
    return json_path, md_path


def _render_markdown(
    model: str,
    duration_seconds: float,
    total: int,
    passed: int,
    failed: int,
    pass_rate: float,
    cats: dict[str, dict[str, Any]],
    fail_breakdown: dict[str, int],
    results: list[AuditTestResult],
) -> str:
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    lines: list[str] = []
    lines.append("# Prometheus Web Capability Audit")
    lines.append("")
    lines.append(
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  "
        f"**Model:** {model}  "
        f"**Duration:** {minutes}m {seconds}s"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"{passed}/{total} tests passed ({pass_rate * 100:.0f}%)")
    lines.append("")

    lines.append("## By Category")
    lines.append("")
    for cat, stats in sorted(cats.items()):
        rate = stats["pass_rate"] * 100
        lines.append(
            f"- **{cat}**: {stats['passed']}/{stats['passed'] + stats['failed']} "
            f"({rate:.0f}%)"
        )
    lines.append("")

    if fail_breakdown:
        lines.append("## Failure Breakdown")
        lines.append("")
        for fc, n in sorted(fail_breakdown.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {fc}: {n}")
        lines.append("")

    cb_failures = [r for r in results if r.circuit_breaker_tripped]
    if cb_failures:
        lines.append("## Circuit Breaker Trips (Fix These First)")
        lines.append("")
        for r in cb_failures:
            lines.append(f"- **{r.name}** ({r.category}): {r.error[:200]}")
        lines.append("")

    lines.append("## Failures")
    lines.append("")
    for r in results:
        if r.passed:
            continue
        lines.append(f"- **{r.name}** [{r.category}]: {r.error[:240]}")
    lines.append("")

    lines.append("## All Results")
    lines.append("")
    lines.append("| Test | Category | Result | Time | Tools |")
    lines.append("|------|----------|--------|------|-------|")
    for r in results:
        check = "✅" if r.passed else "❌"
        secs = r.duration_ms / 1000.0
        tools = ", ".join(r.tools_called) if r.tools_called else "—"
        lines.append(
            f"| {r.name} | {r.category} | {check} | {secs:.1f}s | {tools} |"
        )
    lines.append("")
    return "\n".join(lines)


# ===========================================================================
# Setup + main
# ===========================================================================


def build_runtime(verbose: bool, time_limit: float) -> WebAuditRunner:
    config = load_config()
    print(f"Config loaded: provider={config.get('model', {}).get('provider', 'unknown')}")

    try:
        provider = ProviderRegistry.create(config["model"])
    except Exception as exc:
        print(f"❌ Cannot create provider: {exc}")
        sys.exit(2)

    security_cfg = config.get("security", {})
    security_gate = create_security_gate(security_cfg)
    registry = create_tool_registry(security_cfg, security_gate=security_gate)
    print(f"Tools registered: {len(registry.list_schemas())}")

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
    model_name = model_cfg.get("model", "gemma4-26b")
    loop = AgentLoop(
        provider=provider,
        model=model_name,
        tool_registry=registry,
        adapter=adapter,
        permission_checker=security_gate,
        telemetry=telemetry,
    )

    runner = WebAuditRunner(
        config=config,
        provider=provider,
        adapter=adapter,
        loop=loop,
        telemetry=telemetry,
        verbose=verbose,
        time_limit_seconds=time_limit,
        started_at=time.monotonic(),
    )
    return runner


async def main(args: argparse.Namespace) -> int:
    print("🔬 Prometheus — Web Capability Audit")
    print("=" * 60)

    if AUDIT_WORKSPACE.exists():
        shutil.rmtree(AUDIT_WORKSPACE)
    AUDIT_WORKSPACE.mkdir(parents=True, exist_ok=True)

    runner = build_runtime(verbose=args.verbose, time_limit=args.time_limit)
    print("Agent loop ready")
    print("=" * 60)

    selected: list[str]
    if args.category:
        if args.category not in CATEGORY_FUNCS:
            print(f"Unknown category: {args.category}")
            print(f"Available: {', '.join(CATEGORY_FUNCS)}")
            return 2
        selected = [args.category]
    else:
        selected = list(CATEGORY_FUNCS.keys())

    started = time.monotonic()
    for cat in selected:
        if not runner.has_time_remaining():
            print(f"\n⏰ Time budget exhausted — stopping before {cat}")
            break
        await CATEGORY_FUNCS[cat](runner)

    duration = time.monotonic() - started

    print("\n" + "=" * 60)
    print("📊 AUDIT REPORT")
    print("=" * 60)
    total = len(runner.results)
    passed = sum(1 for r in runner.results if r.passed)
    failed = total - passed
    print(f"Total: {passed}/{total} passed  ({duration:.0f}s)")
    cats = _category_summary(runner.results)
    for cat, stats in sorted(cats.items()):
        rate = stats["pass_rate"] * 100
        icon = "✅" if stats["failed"] == 0 else "❌"
        n = stats["passed"] + stats["failed"]
        print(f"  {icon} {cat}: {stats['passed']}/{n} ({rate:.0f}%)")

    fb = _failure_breakdown(runner.results)
    if fb:
        print("\nFailure breakdown:")
        for fc, n in sorted(fb.items(), key=lambda kv: -kv[1]):
            print(f"  {fc}: {n}")

    model_name = runner.config.get("model", {}).get("model", "unknown")
    json_path, md_path = write_reports(
        runner.results,
        model=model_name,
        duration_seconds=duration,
        output_dir=AUDIT_OUTPUT_DIR,
    )
    print(f"\nReports written:\n  {json_path}\n  {md_path}")

    if AUDIT_WORKSPACE.exists():
        shutil.rmtree(AUDIT_WORKSPACE, ignore_errors=True)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prometheus web capability audit")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full agent output for each test")
    parser.add_argument("--category", "-c", default=None,
                        help=f"Run a single category: {', '.join(CATEGORY_FUNCS)}")
    parser.add_argument("--time-limit", "-t", type=float, default=3600.0,
                        help="Hard time limit in seconds (default 3600)")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)))
