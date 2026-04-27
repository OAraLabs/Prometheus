"""Tests for the VIBE web agent task battery.

Pure-function coverage of the runner — no agent loop, no network.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Allow direct import from scripts/
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from vibe_check import (  # type: ignore  # noqa: E402
    SKIPPED_MARKER,
    VIBE_NOTES_DIR,
    VIBE_TASKS,
    VibeCheckRunner,
    VibeResult,
    _detect_circuit_breaker,
    _diff_tool_calls,
    _expand,
    _tool_call_counts,
)


# ---------------------------------------------------------------------------
# Task structure
# ---------------------------------------------------------------------------


class TestVibeTaskStructure:
    REQUIRED_KEYS = {
        "task_id", "name", "category", "message",
        "auto_checks", "human_review_prompt", "time_budget_seconds",
    }
    VALID_CATEGORIES = {"lookup", "fetch", "save", "recommend", "multistep"}
    VALID_CHECK_TYPES = {
        "contains", "min_length", "file_exists", "not_contains",
    }

    def test_all_tasks_have_required_fields(self):
        for task in VIBE_TASKS:
            missing = self.REQUIRED_KEYS - set(task.keys())
            assert not missing, f"Task {task.get('task_id')} missing: {missing}"

    def test_task_ids_unique(self):
        ids = [t["task_id"] for t in VIBE_TASKS]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"

    def test_task_count_matches_spec(self):
        assert len(VIBE_TASKS) == 20

    def test_categories_are_valid(self):
        for task in VIBE_TASKS:
            assert task["category"] in self.VALID_CATEGORIES, (
                f"Task {task['task_id']} has invalid category {task['category']}"
            )

    def test_check_types_are_valid(self):
        for task in VIBE_TASKS:
            for check in task["auto_checks"]:
                assert check["type"] in self.VALID_CHECK_TYPES, (
                    f"Task {task['task_id']} unknown check type {check['type']}"
                )
                assert "value" in check

    def test_each_category_has_expected_count(self):
        counts: dict[str, int] = {}
        for task in VIBE_TASKS:
            counts[task["category"]] = counts.get(task["category"], 0) + 1
        assert counts.get("lookup") == 5
        assert counts.get("fetch") == 4
        assert counts.get("save") == 4
        assert counts.get("recommend") == 4
        assert counts.get("multistep") == 3

    def test_time_budgets_are_positive(self):
        for task in VIBE_TASKS:
            assert task["time_budget_seconds"] > 0

    def test_youtube_task_has_skip_guard(self):
        yt = next(t for t in VIBE_TASKS if t["task_id"] == "19")
        assert yt.get("skip_if_tool_missing") == "youtube_transcript"

    def test_notes_paths_are_absolute_in_messages(self):
        """Tasks with notes-dir paths must hand the model an absolute path.
        Gemma 26B can't reliably expand ``~`` and sometimes guesses
        ``/home/user/`` — handing it the absolute path avoids that failure.
        """
        for task in VIBE_TASKS:
            assert "~/prometheus-notes/" not in task["message"], (
                f"Task {task['task_id']} still has ~/prometheus-notes/ in "
                f"message — model would have to guess HOME"
            )
            for check in task["auto_checks"]:
                value = check.get("value")
                if isinstance(value, str):
                    assert "~/prometheus-notes/" not in value, (
                        f"Task {task['task_id']} auto-check has unexpanded path"
                    )
            ef = task.get("expect_file")
            if isinstance(ef, str):
                assert "~/prometheus-notes/" not in ef, (
                    f"Task {task['task_id']} expect_file unexpanded"
                )

    def test_notes_paths_point_to_real_home(self):
        """The substitution should produce a path under the actual user home."""
        from pathlib import Path
        home_prefix = str(Path.home() / "prometheus-notes") + "/"
        # At least one save task should now have an absolutized path.
        save_tasks = [t for t in VIBE_TASKS if t["category"] == "save"]
        assert any(home_prefix in t["message"] for t in save_tasks), (
            f"No save task contains the absolutized notes path {home_prefix}"
        )


# ---------------------------------------------------------------------------
# Circuit-breaker detection
# ---------------------------------------------------------------------------


class TestCircuitBreakerDetection:
    def test_detects_explicit_message(self):
        assert _detect_circuit_breaker(
            "circuit breaker tripped after 5 errors", ""
        )

    def test_detects_unknown_tool(self):
        assert _detect_circuit_breaker("", "unknown tool: foo")

    def test_detects_max_iterations(self):
        assert _detect_circuit_breaker(
            "max_tool_iterations exceeded", ""
        )

    def test_detects_via_error_field(self):
        assert _detect_circuit_breaker("", "consecutive identical errors")

    def test_clean_text_returns_false(self):
        assert not _detect_circuit_breaker("Here are 3 results", "")


# ---------------------------------------------------------------------------
# Telemetry helpers
# ---------------------------------------------------------------------------


class TestTelemetryHelpers:
    def test_tool_call_counts_extracts_calls(self):
        report = {
            "tools": {
                "web_fetch": {"calls": 3, "successes": 3},
                "bash": {"calls": 1, "successes": 1},
            },
        }
        assert _tool_call_counts(report) == {"web_fetch": 3, "bash": 1}

    def test_tool_call_counts_handles_empty(self):
        assert _tool_call_counts({}) == {}
        assert _tool_call_counts(None) == {}

    def test_tool_call_counts_handles_missing_calls_key(self):
        report = {"tools": {"web_fetch": {}}}
        assert _tool_call_counts(report) == {"web_fetch": 0}

    def test_diff_tool_calls_finds_new_uses(self):
        before = {"web_fetch": 1, "bash": 1}
        after = {"web_fetch": 2, "bash": 1, "web_search": 1}
        assert _diff_tool_calls(before, after) == ["web_fetch", "web_search"]

    def test_diff_tool_calls_empty_when_unchanged(self):
        before = {"web_fetch": 3}
        assert _diff_tool_calls(before, before) == []


# ---------------------------------------------------------------------------
# Path expansion
# ---------------------------------------------------------------------------


class TestExpand:
    def test_expands_tilde(self):
        assert str(_expand("~/x.md")) == os.path.expanduser("~/x.md")

    def test_passes_absolute_through(self):
        assert _expand("/tmp/x") == Path("/tmp/x")


# ---------------------------------------------------------------------------
# Runner skip behavior
# ---------------------------------------------------------------------------


def _make_runner(registered: set[str]) -> VibeCheckRunner:
    fake_loop = MagicMock()
    fake_loop.run_async = AsyncMock(return_value=MagicMock(text="ok"))
    fake_telemetry = MagicMock()
    fake_telemetry.report = MagicMock(return_value={"tools": {}})
    runner = VibeCheckRunner(
        agent_loop=fake_loop,
        telemetry=fake_telemetry,
        config={},
        verbose=False,
    )
    runner.registered_tools = registered
    return runner


class TestSkipMissingTool:
    def test_skipped_when_required_tool_missing(self):
        runner = _make_runner(registered={"web_fetch"})
        result = asyncio.run(runner.run_task(
            task_id="99",
            name="t",
            category="multistep",
            message="x",
            auto_checks=[{"type": "min_length", "value": 1}],
            human_review_prompt="?",
            time_budget_seconds=10,
            skip_if_tool_missing="youtube_transcript",
        ))
        assert result.error == SKIPPED_MARKER
        assert not result.auto_passed
        runner.loop.run_async.assert_not_called()

    def test_not_skipped_when_tool_present(self):
        runner = _make_runner(registered={"youtube_transcript", "web_fetch"})
        result = asyncio.run(runner.run_task(
            task_id="99",
            name="t",
            category="multistep",
            message="x",
            auto_checks=[{"type": "min_length", "value": 1}],
            human_review_prompt="?",
            time_budget_seconds=10,
            skip_if_tool_missing="youtube_transcript",
        ))
        assert result.error != SKIPPED_MARKER
        runner.loop.run_async.assert_called_once()


# ---------------------------------------------------------------------------
# Auto-check evaluation
# ---------------------------------------------------------------------------


class TestAutoChecks:
    def test_contains_pass(self):
        runner = _make_runner(registered=set())
        runner.loop.run_async = AsyncMock(
            return_value=MagicMock(text="Python 3.13.1 is current")
        )
        result = asyncio.run(runner.run_task(
            task_id="t1", name="t", category="lookup", message="x",
            auto_checks=[{"type": "contains", "value": "3."}],
            human_review_prompt="?",
            time_budget_seconds=10,
        ))
        assert result.auto_passed

    def test_contains_fail(self):
        runner = _make_runner(registered=set())
        runner.loop.run_async = AsyncMock(
            return_value=MagicMock(text="No version info")
        )
        result = asyncio.run(runner.run_task(
            task_id="t2", name="t", category="lookup", message="x",
            auto_checks=[{"type": "contains", "value": "3."}],
            human_review_prompt="?",
            time_budget_seconds=10,
        ))
        assert not result.auto_passed
        assert any("missing '3.'" in f for f in result.auto_failures)

    def test_min_length_fail(self):
        runner = _make_runner(registered=set())
        runner.loop.run_async = AsyncMock(return_value=MagicMock(text="hi"))
        result = asyncio.run(runner.run_task(
            task_id="t3", name="t", category="lookup", message="x",
            auto_checks=[{"type": "min_length", "value": 50}],
            human_review_prompt="?",
            time_budget_seconds=10,
        ))
        assert not result.auto_passed

    def test_not_contains_fail(self):
        runner = _make_runner(registered=set())
        runner.loop.run_async = AsyncMock(
            return_value=MagicMock(text="circuit breaker tripped on URL")
        )
        result = asyncio.run(runner.run_task(
            task_id="t4", name="t", category="fetch", message="x",
            auto_checks=[
                {"type": "not_contains", "value": "circuit breaker"},
                {"type": "min_length", "value": 5},
            ],
            human_review_prompt="?",
            time_budget_seconds=10,
        ))
        assert not result.auto_passed
        # circuit-breaker indicator also tripped
        assert result.circuit_breaker_tripped

    def test_file_exists_pass(self, tmp_path, monkeypatch):
        target = tmp_path / "out.md"
        target.write_text("hello")
        monkeypatch.setenv("HOME", str(tmp_path))

        runner = _make_runner(registered=set())
        runner.loop.run_async = AsyncMock(
            return_value=MagicMock(text="Saved")
        )
        result = asyncio.run(runner.run_task(
            task_id="t5", name="t", category="save", message="x",
            auto_checks=[{"type": "file_exists", "value": str(target)}],
            human_review_prompt="?",
            time_budget_seconds=10,
            expect_file=str(target),
        ))
        assert result.auto_passed
        assert result.artifact_path == str(target)
        assert result.artifact_preview == "hello"

    def test_file_exists_fail(self, tmp_path):
        runner = _make_runner(registered=set())
        runner.loop.run_async = AsyncMock(
            return_value=MagicMock(text="Saved")
        )
        result = asyncio.run(runner.run_task(
            task_id="t6", name="t", category="save", message="x",
            auto_checks=[{"type": "file_exists",
                          "value": str(tmp_path / "missing.md")}],
            human_review_prompt="?",
            time_budget_seconds=10,
        ))
        assert not result.auto_passed


# ---------------------------------------------------------------------------
# Tool-usage tracking
# ---------------------------------------------------------------------------


class TestToolUsageTracking:
    def test_tools_used_recorded_via_telemetry_diff(self):
        runner = _make_runner(registered=set())
        runner.loop.run_async = AsyncMock(
            return_value=MagicMock(text="The latest is 3.13")
        )

        # Telemetry returns "no tools called" before, then "web_search called once"
        # after the run. The runner should pick up that diff.
        reports = iter([
            {"tools": {}},
            {"tools": {"web_search": {"calls": 1}}},
        ])
        runner.telemetry.report = MagicMock(side_effect=lambda *a, **kw: next(reports))

        result = asyncio.run(runner.run_task(
            task_id="t7", name="t", category="lookup", message="x",
            auto_checks=[{"type": "min_length", "value": 1}],
            human_review_prompt="?",
            time_budget_seconds=10,
        ))
        assert result.tools_used == ["web_search"]


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _result(
    task_id: str = "t1",
    auto_passed: bool = True,
    category: str = "lookup",
    error: str = "",
    cb: bool = False,
    timed_out: bool = False,
) -> VibeResult:
    return VibeResult(
        task_id=task_id,
        name="t",
        category=category,
        message="msg",
        auto_passed=auto_passed,
        auto_checks=[],
        auto_failures=[] if auto_passed else ["fail reason"],
        human_review_prompt="check it",
        agent_output="some output",
        artifact_path=None,
        artifact_preview=None,
        duration_ms=100.0,
        tools_used=[],
        circuit_breaker_tripped=cb,
        timed_out=timed_out,
        error=error,
    )


class TestGenerateReport:
    def test_markdown_includes_header_and_results(self):
        runner = _make_runner(registered=set())
        runner.results = [
            _result("01", auto_passed=True),
            _result("02", auto_passed=False),
        ]
        md, _ = runner.generate_report("test-model")
        assert "# Prometheus Vibe Check" in md
        assert "test-model" in md
        assert "TASK 01" in md
        assert "TASK 02" in md
        assert "1/2" in md  # auto pass rate

    def test_markdown_circuit_breaker_section_when_present(self):
        runner = _make_runner(registered=set())
        runner.results = [_result("01", auto_passed=False, cb=True)]
        md, _ = runner.generate_report("m")
        # The "Fix These First" section is the diagnostic block; only emitted
        # when at least one task tripped. The summary always shows the count.
        assert "Fix These First" in md

    def test_markdown_no_circuit_breaker_section_when_clean(self):
        runner = _make_runner(registered=set())
        runner.results = [_result("01", auto_passed=True)]
        md, _ = runner.generate_report("m")
        assert "Fix These First" not in md

    def test_skipped_excluded_from_pass_rate_denominator(self):
        runner = _make_runner(registered=set())
        runner.results = [
            _result("01", auto_passed=True),
            _result("02", auto_passed=False, error=SKIPPED_MARKER),
        ]
        _, data = runner.generate_report("m")
        assert data["summary"]["total"] == 1
        assert data["summary"]["auto_passed"] == 1
        assert data["summary"]["skipped"] == 1

    def test_json_payload_is_valid_and_complete(self):
        runner = _make_runner(registered=set())
        runner.results = [_result("01", auto_passed=True)]
        _, data = runner.generate_report("m")
        # round-trip through JSON to confirm serializability
        json.loads(json.dumps(data))
        assert data["model"] == "m"
        assert "summary" in data
        assert "categories" in data
        assert "results" in data
        assert data["results"][0]["task_id"] == "01"


# ---------------------------------------------------------------------------
# Notes-dir contract
# ---------------------------------------------------------------------------


class TestNotesDirectory:
    def test_notes_dir_path_is_tilde_prometheus_notes(self):
        assert VIBE_NOTES_DIR == Path.home() / "prometheus-notes"
