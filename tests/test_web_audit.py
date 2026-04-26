"""Tests for the WEAVE web capability audit infrastructure.

These tests exercise pure-function logic of the audit harness without
spawning the full agent loop or touching the network.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

# Allow direct import from scripts/
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from web_capability_audit import (  # type: ignore  # noqa: E402
    AuditTestResult,
    CATEGORY_FUNCS,
    FAILURE_CIRCUIT_BREAKER,
    FAILURE_EXECUTION,
    FAILURE_TIMEOUT,
    FAILURE_WRONG_TOOL,
    WebAuditRunner,
    _category_summary,
    _failure_breakdown,
    _render_markdown,
    write_reports,
)


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


def _make_result(
    name: str = "t",
    category: str = "search",
    passed: bool = True,
    duration_ms: float = 100.0,
    failure_category: str = "",
    circuit_breaker_tripped: bool = False,
    tools_called: list[str] | None = None,
) -> AuditTestResult:
    return AuditTestResult(
        name=name,
        category=category,
        passed=passed,
        duration_ms=duration_ms,
        tools_called=tools_called or [],
        failure_category=failure_category,
        circuit_breaker_tripped=circuit_breaker_tripped,
    )


class TestCategorySummary:
    def test_empty_results(self):
        assert _category_summary([]) == {}

    def test_groups_by_category(self):
        results = [
            _make_result(category="search", passed=True),
            _make_result(category="search", passed=False),
            _make_result(category="fetch", passed=True),
        ]
        summary = _category_summary(results)
        assert summary["search"] == {
            "passed": 1, "failed": 1, "pass_rate": 0.5
        }
        assert summary["fetch"] == {
            "passed": 1, "failed": 0, "pass_rate": 1.0
        }


class TestFailureBreakdown:
    def test_only_counts_failures(self):
        results = [
            _make_result(passed=True, failure_category=""),
            _make_result(passed=False, failure_category=FAILURE_TIMEOUT),
            _make_result(passed=False, failure_category=FAILURE_TIMEOUT),
            _make_result(passed=False, failure_category=FAILURE_WRONG_TOOL),
        ]
        fb = _failure_breakdown(results)
        assert fb == {FAILURE_TIMEOUT: 2, FAILURE_WRONG_TOOL: 1}

    def test_skips_failures_with_no_category(self):
        results = [_make_result(passed=False, failure_category="")]
        assert _failure_breakdown(results) == {}


# ---------------------------------------------------------------------------
# Circuit-breaker detection
# ---------------------------------------------------------------------------


class TestCircuitBreakerDetection:
    def test_detects_explicit_message(self):
        assert WebAuditRunner.detect_circuit_breaker(
            "circuit breaker tripped after 5 errors", ""
        )

    def test_detects_unknown_tool(self):
        assert WebAuditRunner.detect_circuit_breaker("", "unknown tool: foo")

    def test_detects_max_iterations(self):
        assert WebAuditRunner.detect_circuit_breaker(
            "max_tool_iterations exceeded", ""
        )

    def test_clean_text_returns_false(self):
        assert not WebAuditRunner.detect_circuit_breaker("Here are 3 results", "")


# ---------------------------------------------------------------------------
# Time-limit
# ---------------------------------------------------------------------------


class TestTimeBudget:
    def test_unstarted_runner_has_time(self):
        runner = WebAuditRunner.__new__(WebAuditRunner)
        runner.started_at = 0
        runner.time_limit_seconds = 100
        assert runner.has_time_remaining()

    def test_within_budget_returns_true(self):
        runner = WebAuditRunner.__new__(WebAuditRunner)
        runner.started_at = time.monotonic()
        runner.time_limit_seconds = 3600
        assert runner.has_time_remaining()

    def test_exceeded_budget_returns_false(self):
        runner = WebAuditRunner.__new__(WebAuditRunner)
        runner.started_at = time.monotonic() - 10000
        runner.time_limit_seconds = 100
        assert not runner.has_time_remaining()


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


class TestWriteReports:
    def test_json_report_is_valid_json(self, tmp_path):
        results = [
            _make_result(name="t1", passed=True, category="search"),
            _make_result(
                name="t2",
                passed=False,
                category="fetch",
                failure_category=FAILURE_EXECUTION,
            ),
        ]
        json_path, md_path = write_reports(
            results,
            model="test-model",
            duration_seconds=42.0,
            output_dir=tmp_path,
        )
        assert json_path.exists()
        payload = json.loads(json_path.read_text())
        assert payload["model"] == "test-model"
        assert payload["total_tests"] == 2
        assert payload["passed"] == 1
        assert payload["failed"] == 1
        assert "categories" in payload
        assert "results" in payload

    def test_markdown_report_generated(self, tmp_path):
        results = [_make_result(passed=True)]
        json_path, md_path = write_reports(
            results,
            model="m",
            duration_seconds=10,
            output_dir=tmp_path,
        )
        assert md_path.exists()
        body = md_path.read_text()
        assert "# Prometheus Web Capability Audit" in body
        assert "1/1" in body  # 1 passed of 1 total

    def test_markdown_includes_circuit_breaker_section_when_present(self):
        results = [
            _make_result(
                name="cb_t",
                passed=False,
                circuit_breaker_tripped=True,
                failure_category=FAILURE_CIRCUIT_BREAKER,
            ),
        ]
        cats = _category_summary(results)
        fb = _failure_breakdown(results)
        md = _render_markdown(
            "m", 5, len(results), 0, 1, 0.0, cats, fb, results
        )
        assert "Circuit Breaker Trips" in md

    def test_markdown_no_circuit_breaker_section_when_clean(self):
        results = [_make_result(passed=True)]
        cats = _category_summary(results)
        fb = _failure_breakdown(results)
        md = _render_markdown(
            "m", 5, 1, 1, 0, 1.0, cats, fb, results
        )
        assert "Circuit Breaker Trips" not in md


# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------


class TestCategoryFuncs:
    def test_all_categories_registered(self):
        assert set(CATEGORY_FUNCS.keys()) == {
            "search", "fetch", "youtube", "download",
            "research", "graceful", "railway",
        }

    def test_each_is_async(self):
        import inspect
        for name, fn in CATEGORY_FUNCS.items():
            assert inspect.iscoroutinefunction(fn), \
                f"{name} should be async"


# ---------------------------------------------------------------------------
# Telegram /audit handler — wiring smoke check
# ---------------------------------------------------------------------------


class TestTelegramAuditWired:
    def test_telegram_has_cmd_audit(self):
        """TelegramAdapter._cmd_audit exists (wired in start())."""
        from prometheus.gateway.telegram import TelegramAdapter
        assert hasattr(TelegramAdapter, "_cmd_audit")
        assert hasattr(TelegramAdapter, "_audit_show_last")
        assert hasattr(TelegramAdapter, "_audit_kick_off")
