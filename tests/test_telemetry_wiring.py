"""Tests for telemetry wiring — verifies that tool execution paths record to DB."""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import LoopContext, _execute_tool_call, _dispatch_tool_calls
from prometheus.engine.messages import ToolUseBlock
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _EchoInput(BaseModel):
    text: str = "hello"


class _EchoTool(BaseTool):
    name = "echo"
    description = "Echo text back"
    input_model = _EchoInput

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        return ToolResult(output=arguments.text)

    def is_read_only(self, arguments: BaseModel) -> bool:
        return True


class _FailTool(BaseTool):
    name = "fail_tool"
    description = "Always fails"
    input_model = _EchoInput

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        return ToolResult(output="something went wrong", is_error=True)


class _StrictInput(BaseModel):
    required_field: int


class _StrictTool(BaseTool):
    name = "strict_tool"
    description = "Requires an int"
    input_model = _StrictInput

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        return ToolResult(output=str(arguments.required_field))


def _make_context(tel: ToolCallTelemetry, registry: ToolRegistry | None = None, **overrides) -> LoopContext:
    """Build a minimal LoopContext for testing."""
    stub_provider = AsyncMock()
    return LoopContext(
        provider=stub_provider,
        model="test-model",
        system_prompt="test",
        max_tokens=1024,
        tool_registry=registry,
        telemetry=tel,
        **overrides,
    )


def _rows(tel: ToolCallTelemetry) -> list[dict]:
    """Return all rows from tool_calls as dicts."""
    cur = tel._conn.execute(
        "SELECT model, tool_name, success, retries, latency_ms, error_type, error_detail FROM tool_calls"
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


@pytest.fixture
def tel(tmp_path: Path) -> ToolCallTelemetry:
    return ToolCallTelemetry(db_path=tmp_path / "telemetry.db")


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_EchoTool())
    reg.register(_FailTool())
    reg.register(_StrictTool())
    return reg


# ---------------------------------------------------------------------------
# Tests: successful tool execution
# ---------------------------------------------------------------------------


class TestSuccessfulToolRecording:

    def test_successful_tool_records_telemetry(self, tel, registry):
        ctx = _make_context(tel, registry)
        result = asyncio.run(
            _execute_tool_call(ctx, "echo", "tu_1", {"text": "hi"})
        )
        assert not result.is_error
        rows = _rows(tel)
        assert len(rows) == 1
        assert rows[0]["model"] == "test-model"
        assert rows[0]["tool_name"] == "echo"
        assert rows[0]["success"] == 1
        assert rows[0]["latency_ms"] > 0
        assert rows[0]["error_type"] is None

    def test_failed_tool_records_telemetry(self, tel, registry):
        ctx = _make_context(tel, registry)
        result = asyncio.run(
            _execute_tool_call(ctx, "fail_tool", "tu_2", {"text": "x"})
        )
        assert result.is_error
        rows = _rows(tel)
        assert len(rows) == 1
        assert rows[0]["success"] == 0
        assert rows[0]["error_type"] == "tool_error"


# ---------------------------------------------------------------------------
# Tests: error paths
# ---------------------------------------------------------------------------


class TestErrorPathRecording:

    def test_unknown_tool_records_telemetry(self, tel, registry):
        ctx = _make_context(tel, registry)
        result = asyncio.run(
            _execute_tool_call(ctx, "nonexistent_tool", "tu_3", {})
        )
        assert result.is_error
        rows = _rows(tel)
        assert len(rows) == 1
        assert rows[0]["tool_name"] == "nonexistent_tool"
        assert rows[0]["error_type"] == "unknown_tool"
        assert rows[0]["success"] == 0

    def test_input_validation_error_records(self, tel, registry):
        ctx = _make_context(tel, registry)
        result = asyncio.run(
            _execute_tool_call(ctx, "strict_tool", "tu_4", {"required_field": "not_an_int"})
        )
        assert result.is_error
        rows = _rows(tel)
        assert len(rows) == 1
        assert rows[0]["error_type"] == "input_validation"
        assert rows[0]["success"] == 0

    def test_no_registry_records_telemetry(self, tel):
        ctx = _make_context(tel, registry=None)
        result = asyncio.run(
            _execute_tool_call(ctx, "echo", "tu_5", {})
        )
        assert result.is_error
        rows = _rows(tel)
        assert len(rows) == 1
        assert rows[0]["error_type"] == "no_registry"

    def test_permission_denied_records_telemetry(self, tel, registry):
        @dataclass
        class _Decision:
            allowed: bool = False
            requires_confirmation: bool = False
            reason: str = "blocked by policy"

        mock_checker = type("Checker", (), {
            "evaluate": lambda self, *a, **kw: _Decision(),
        })()

        ctx = _make_context(tel, registry, permission_checker=mock_checker)
        result = asyncio.run(
            _execute_tool_call(ctx, "echo", "tu_6", {"text": "hi"})
        )
        assert result.is_error
        rows = _rows(tel)
        assert len(rows) == 1
        assert rows[0]["error_type"] == "permission_denied"

    def test_hook_blocked_records_telemetry(self, tel, registry):
        @dataclass
        class _HookResult:
            blocked: bool = True
            reason: str = "hook says no"

        mock_hook = AsyncMock()
        mock_hook.execute = AsyncMock(return_value=_HookResult())

        ctx = _make_context(tel, registry, hook_executor=mock_hook)
        result = asyncio.run(
            _execute_tool_call(ctx, "echo", "tu_7", {"text": "hi"})
        )
        assert result.is_error
        rows = _rows(tel)
        assert len(rows) == 1
        assert rows[0]["error_type"] == "hook_blocked"


# ---------------------------------------------------------------------------
# Tests: reset commands
# ---------------------------------------------------------------------------


class TestResetCommands:

    def test_reset_telemetry_deletes_db(self, tmp_path, monkeypatch):
        from prometheus.__main__ import _reset_telemetry
        from prometheus.config import paths

        db = tmp_path / "telemetry.db"
        db.write_text("")
        wal = tmp_path / "telemetry.db-wal"
        wal.write_text("")

        monkeypatch.setattr(paths, "get_config_dir", lambda: tmp_path)
        monkeypatch.setattr("builtins.input", lambda _: "y")

        _reset_telemetry()

        assert not db.exists()
        assert not wal.exists()

    def test_reset_telemetry_cancelled(self, tmp_path, monkeypatch):
        from prometheus.__main__ import _reset_telemetry
        from prometheus.config import paths

        db = tmp_path / "telemetry.db"
        db.write_text("")

        monkeypatch.setattr(paths, "get_config_dir", lambda: tmp_path)
        monkeypatch.setattr("builtins.input", lambda _: "n")

        _reset_telemetry()

        assert db.exists()

    def test_reset_data_deletes_all_targets(self, tmp_path, monkeypatch):
        from prometheus.__main__ import _reset_data
        from prometheus.config import paths

        # Create file targets
        (tmp_path / "telemetry.db").write_text("")
        (tmp_path / "memory.db").write_text("")
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "lcm.db").write_text("")
        (data_dir / "security").mkdir()
        (data_dir / "security" / "audit.db").write_text("")

        # Create dir targets
        (tmp_path / "eval_results").mkdir()
        (tmp_path / "eval_results" / "result.json").write_text("{}")
        (tmp_path / "wiki").mkdir()
        (tmp_path / "sentinel").mkdir()
        (tmp_path / "skills" / "auto").mkdir(parents=True)

        # Create config files that should be preserved
        (tmp_path / "prometheus.yaml").write_text("test: true")

        monkeypatch.setattr(paths, "get_config_dir", lambda: tmp_path)
        monkeypatch.setattr(paths, "get_data_dir", lambda: data_dir)
        monkeypatch.setattr("builtins.input", lambda _: "y")

        _reset_data()

        assert not (tmp_path / "telemetry.db").exists()
        assert not (tmp_path / "memory.db").exists()
        assert not (data_dir / "lcm.db").exists()
        assert not (data_dir / "security" / "audit.db").exists()
        assert not (tmp_path / "eval_results").exists()
        assert not (tmp_path / "wiki").exists()
        assert not (tmp_path / "sentinel").exists()
        assert not (tmp_path / "skills" / "auto").exists()
        # Config preserved
        assert (tmp_path / "prometheus.yaml").exists()

    def test_reset_data_cancelled(self, tmp_path, monkeypatch):
        from prometheus.__main__ import _reset_data
        from prometheus.config import paths

        (tmp_path / "telemetry.db").write_text("")
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        monkeypatch.setattr(paths, "get_config_dir", lambda: tmp_path)
        monkeypatch.setattr(paths, "get_data_dir", lambda: data_dir)
        monkeypatch.setattr("builtins.input", lambda _: "n")

        _reset_data()

        assert (tmp_path / "telemetry.db").exists()
