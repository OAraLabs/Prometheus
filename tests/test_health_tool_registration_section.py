"""Tests for the /health Tool registration section (Phase 2 — orphan-tools).

The section is rendered only when at least one ``subsystem_runs`` row
with ``subsystem='tool_registration'`` and ``outcome='failed'`` exists
in the lookback window. Successful-only startups suppress the section
per ``docs/audits/ORPHAN-TOOLS-AUDIT.md`` Phase 2 spec.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prometheus.gateway.commands import cmd_health
from prometheus.telemetry import tracker as _tracker
from prometheus.telemetry.tracker import ToolCallTelemetry


pytestmark = pytest.mark.integration


@pytest.fixture
def telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ToolCallTelemetry:
    """Fresh on-disk telemetry, wired as the cmd_health singleton."""
    tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    monkeypatch.setattr(_tracker, "_telemetry_singleton", tel)
    return tel


class TestHealthShowsToolRegistrationFailures:
    def test_section_appears_when_any_tool_failed(
        self, telemetry: ToolCallTelemetry
    ) -> None:
        # Synthesize: 46 successes + 1 failure
        for i in range(46):
            telemetry.record_run(
                subsystem="tool_registration",
                operation=f"GoodTool{i}",
                outcome="success",
                duration_ms=1.0,
                summary={
                    "module_path": f"prometheus.tools.builtin.good_{i}",
                    "class_name": f"GoodTool{i}",
                },
            )
        telemetry.record_run(
            subsystem="tool_registration",
            operation="SymbioteCoordinator",
            outcome="failed",
            duration_ms=2.5,
            summary={
                "module_path": "prometheus.symbiote.coordinator",
                "class_name": "SymbioteCoordinator",
                "exception_type": "ImportError",
                "exception_msg": "libgit2 not found",
            },
        )

        out = cmd_health(verbose=False, since_hours=24.0)
        assert "Tool registration" in out, (
            f"Expected 'Tool registration' section in /health output:\n{out}"
        )
        assert "SymbioteCoordinator" in out
        assert "ImportError" in out
        assert "libgit2 not found" in out
        # The successful-count appears too.
        assert "46" in out or "46 of" in out, (
            "Expected success count in section.\nGot:\n" + out
        )

    def test_section_groups_repeated_failures_with_count(
        self, telemetry: ToolCallTelemetry
    ) -> None:
        """Two daemon restarts with the same failure should collapse to one row."""
        for _ in range(3):
            telemetry.record_run(
                subsystem="tool_registration",
                operation="BrowserTool",
                outcome="failed",
                duration_ms=1.0,
                summary={
                    "module_path": "prometheus.tools.builtin.browser",
                    "class_name": "BrowserTool",
                    "exception_type": "ModuleNotFoundError",
                    "exception_msg": "No module named 'playwright'",
                },
            )

        out = cmd_health(verbose=False, since_hours=24.0)
        assert "BrowserTool" in out
        assert "3 occurrences" in out, (
            "Expected '[3 occurrences]' grouping suffix in section.\n" + out
        )

    def test_section_suppressed_when_all_registrations_succeed(
        self, telemetry: ToolCallTelemetry
    ) -> None:
        for i in range(10):
            telemetry.record_run(
                subsystem="tool_registration",
                operation=f"Tool{i}",
                outcome="success",
                duration_ms=1.0,
                summary={"module_path": f"x.{i}", "class_name": f"Tool{i}"},
            )

        out = cmd_health(verbose=False, since_hours=24.0)
        # No detailed section header when clean — and the generic
        # subsystems-loop row should also be suppressed (popped).
        assert "Tool registration" not in out, (
            f"Expected /health to suppress Tool registration section "
            f"when all 10 registrations succeeded. Got:\n{out}"
        )

    def test_section_includes_failure_in_subsystems_count_path(
        self, telemetry: ToolCallTelemetry
    ) -> None:
        """When only failures exist (no successes), the section still renders."""
        telemetry.record_run(
            subsystem="tool_registration",
            operation="GhostTool",
            outcome="failed",
            duration_ms=0.5,
            summary={
                "module_path": "x.nonexistent",
                "class_name": "GhostTool",
                "exception_type": "ModuleNotFoundError",
                "exception_msg": "No module named 'x.nonexistent'",
            },
        )
        out = cmd_health(verbose=False, since_hours=24.0)
        assert "Tool registration" in out
        assert "GhostTool" in out
        assert "ModuleNotFoundError" in out
