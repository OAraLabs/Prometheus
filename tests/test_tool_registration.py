"""Tests for prometheus.tools.registration.try_register.

Phase 2 (orphan-tools): the helper replaces 17 silently-failable
``try/except: pass`` blocks in ``__main__.py::create_tool_registry``.
Each registration now logs, telemetry-records, and never throws —
these tests freeze that contract so a future refactor can't regress
back to the silent-failure pattern.

Three failure modes are exercised:
  1. Module path doesn't exist           (ModuleNotFoundError)
  2. Module imports but class is missing (AttributeError)
  3. Class exists but constructor raises (RuntimeError synthetic)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from prometheus.tools.base import (
    BaseTool,
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
)
from prometheus.tools.registration import try_register
from prometheus.telemetry.tracker import ToolCallTelemetry


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def telemetry(tmp_path: Path) -> ToolCallTelemetry:
    """Fresh on-disk telemetry per test — verifies real schema migration."""
    return ToolCallTelemetry(db_path=tmp_path / "telemetry.db")


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestTryRegisterSuccess:
    """The happy-path — a real, importable, no-arg tool round-trips."""

    def test_registers_real_tool_into_registry(
        self,
        registry: ToolRegistry,
        telemetry: ToolCallTelemetry,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO, logger="prometheus.tools.registration")
        ok = try_register(
            registry, "FileReadTool",
            "prometheus.tools.builtin.file_read", "FileReadTool",
            telemetry=telemetry,
        )
        assert ok is True
        names = {t.name for t in registry.list_tools()}
        assert "read_file" in names or "file_read" in names or any(
            "read" in n.lower() for n in names
        ), f"FileReadTool registered but name not in {sorted(names)}"

    def test_logs_info_on_success(
        self,
        registry: ToolRegistry,
        telemetry: ToolCallTelemetry,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO, logger="prometheus.tools.registration")
        try_register(
            registry, "FileReadTool",
            "prometheus.tools.builtin.file_read", "FileReadTool",
            telemetry=telemetry,
        )
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any(
            "FileReadTool" in r.getMessage() and "registered" in r.getMessage()
            for r in info_records
        ), f"Expected INFO log mentioning FileReadTool + registered. Got: {[r.getMessage() for r in info_records]}"

    def test_writes_success_row_to_telemetry(
        self,
        registry: ToolRegistry,
        telemetry: ToolCallTelemetry,
    ) -> None:
        try_register(
            registry, "FileReadTool",
            "prometheus.tools.builtin.file_read", "FileReadTool",
            telemetry=telemetry,
        )
        rows = telemetry.runs_since(0.0, subsystem="tool_registration")
        assert len(rows) == 1
        row = rows[0]
        assert row["subsystem"] == "tool_registration"
        assert row["operation"] == "FileReadTool"
        assert row["outcome"] == "success"
        summary = json.loads(row["summary_json"])
        assert summary["module_path"] == "prometheus.tools.builtin.file_read"
        assert summary["class_name"] == "FileReadTool"

    def test_factory_runs_for_constructor_args(
        self,
        registry: ToolRegistry,
        telemetry: ToolCallTelemetry,
    ) -> None:
        """Tools with non-default constructors register via the factory hook."""
        from prometheus.tools.builtin.bash import BashTool

        ok = try_register(
            registry, "BashTool",
            "prometheus.tools.builtin.bash", "BashTool",
            factory=lambda: BashTool(workspace="/tmp"),
            telemetry=telemetry,
        )
        assert ok is True
        assert any(isinstance(t, BashTool) for t in registry.list_tools())


# ---------------------------------------------------------------------------
# Failure paths — three flavors of break
# ---------------------------------------------------------------------------


class TestTryRegisterImportFailure:
    """Module path doesn't resolve — the load-bearing silent-failure case."""

    def test_returns_false_on_missing_module(
        self,
        registry: ToolRegistry,
        telemetry: ToolCallTelemetry,
    ) -> None:
        ok = try_register(
            registry, "GhostTool",
            "prometheus.nonexistent.module", "GhostTool",
            telemetry=telemetry,
        )
        assert ok is False
        assert len(registry.list_tools()) == 0

    def test_does_not_raise_on_missing_module(
        self,
        registry: ToolRegistry,
        telemetry: ToolCallTelemetry,
    ) -> None:
        """The whole point of try_register: never propagate."""
        try:
            try_register(
                registry, "GhostTool",
                "prometheus.nonexistent.module", "GhostTool",
                telemetry=telemetry,
            )
        except Exception as exc:
            pytest.fail(f"try_register raised: {exc!r}")

    def test_logs_warning_on_import_failure(
        self,
        registry: ToolRegistry,
        telemetry: ToolCallTelemetry,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING, logger="prometheus.tools.registration")
        try_register(
            registry, "GhostTool",
            "prometheus.nonexistent.module", "GhostTool",
            telemetry=telemetry,
        )
        warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "GhostTool" in r.getMessage() and "FAILED" in r.getMessage()
            for r in warns
        ), (
            f"Expected WARN mentioning GhostTool + FAILED. "
            f"Got: {[r.getMessage() for r in warns]}"
        )

    def test_writes_failure_row_with_exception_details(
        self,
        registry: ToolRegistry,
        telemetry: ToolCallTelemetry,
    ) -> None:
        try_register(
            registry, "GhostTool",
            "prometheus.nonexistent.module", "GhostTool",
            telemetry=telemetry,
        )
        rows = telemetry.runs_since(0.0, subsystem="tool_registration")
        assert len(rows) == 1, f"Expected 1 telemetry row, got {len(rows)}"
        row = rows[0]
        assert row["outcome"] == "failed"
        assert row["operation"] == "GhostTool"
        summary = json.loads(row["summary_json"])
        assert summary["exception_type"] in (
            "ModuleNotFoundError",
            "ImportError",
        ), f"Expected import-related exception, got {summary['exception_type']}"
        assert "nonexistent" in summary["exception_msg"].lower() or \
               "no module" in summary["exception_msg"].lower(), \
               f"exception_msg='{summary['exception_msg']}'"


class TestTryRegisterMissingClass:
    """Module imports but the named class isn't there — AttributeError surface."""

    def test_writes_failure_row_for_missing_class(
        self,
        registry: ToolRegistry,
        telemetry: ToolCallTelemetry,
    ) -> None:
        ok = try_register(
            registry, "NotARealClass",
            "prometheus.tools.builtin.file_read", "NotARealClass",
            telemetry=telemetry,
        )
        assert ok is False
        rows = telemetry.runs_since(0.0, subsystem="tool_registration")
        assert len(rows) == 1
        summary = json.loads(rows[0]["summary_json"])
        assert summary["exception_type"] == "AttributeError"


class TestTryRegisterConstructorRaises:
    """Class exists, constructor explodes — RuntimeError surface."""

    def test_writes_failure_row_when_factory_raises(
        self,
        registry: ToolRegistry,
        telemetry: ToolCallTelemetry,
    ) -> None:
        def _exploding_factory():
            raise RuntimeError("dependency 'libgit2' not found")

        ok = try_register(
            registry, "FileReadTool",
            "prometheus.tools.builtin.file_read", "FileReadTool",
            factory=_exploding_factory,
            telemetry=telemetry,
        )
        assert ok is False
        rows = telemetry.runs_since(0.0, subsystem="tool_registration")
        assert len(rows) == 1
        summary = json.loads(rows[0]["summary_json"])
        assert summary["exception_type"] == "RuntimeError"
        assert "libgit2" in summary["exception_msg"]


# ---------------------------------------------------------------------------
# Telemetry-handle fallback
# ---------------------------------------------------------------------------


class TestTryRegisterTelemetryHandle:
    """When telemetry= is omitted, helper falls back to get_telemetry_handle()."""

    def test_falls_back_to_singleton(
        self,
        registry: ToolRegistry,
        telemetry: ToolCallTelemetry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from prometheus.telemetry import tracker

        monkeypatch.setattr(tracker, "_telemetry_singleton", telemetry)
        try_register(
            registry, "FileReadTool",
            "prometheus.tools.builtin.file_read", "FileReadTool",
        )
        rows = telemetry.runs_since(0.0, subsystem="tool_registration")
        assert len(rows) == 1
        assert rows[0]["outcome"] == "success"

    def test_runs_without_telemetry_at_all(
        self,
        registry: ToolRegistry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No singleton, no explicit handle — registration still works."""
        from prometheus.telemetry import tracker

        monkeypatch.setattr(tracker, "_telemetry_singleton", None)
        ok = try_register(
            registry, "FileReadTool",
            "prometheus.tools.builtin.file_read", "FileReadTool",
        )
        assert ok is True
