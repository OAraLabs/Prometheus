"""Tests for LSP diagnostics hook (Sprint 20: auto-check after file mutations)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from prometheus.hooks.lsp_diagnostics import LSPDiagnosticsHook
from prometheus.lsp.client import Diagnostic


# ------------------------------------------------------------------
# Minimal ToolResultBlock stand-in (matches engine.messages)
# ------------------------------------------------------------------

@dataclass
class FakeToolResult:
    tool_use_id: str = "test-123"
    content: str = "Updated /tmp/test.py"
    is_error: bool = False


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def mock_orch():
    orch = MagicMock()
    orch.notify_file_changed = AsyncMock()
    orch.get_diagnostics = AsyncMock(return_value=[])
    return orch


@pytest.fixture
def hook(mock_orch):
    return LSPDiagnosticsHook(orchestrator=mock_orch, delay_ms=0)


# ------------------------------------------------------------------
# Fires for write_file
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fires_after_write_file(hook, mock_orch, tmp_path):
    """Fires after write_file, appends errors to result."""
    f = tmp_path / "test.py"
    f.write_text("x: int = 'hello'\n")

    mock_orch.get_diagnostics = AsyncMock(return_value=[
        Diagnostic(
            path=str(f), line=1, col=10,
            severity=1, message="Type 'str' not assignable to 'int'",
        ),
    ])

    result = FakeToolResult(content=f"Wrote {f}")
    new_result = await hook("write_file", {"path": str(f)}, result)

    assert "\u26a0\ufe0f LSP detected 1 issue(s)" in new_result.content
    assert "Type 'str'" in new_result.content
    mock_orch.notify_file_changed.assert_called_once()


# ------------------------------------------------------------------
# Fires for edit_file
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fires_after_edit_file(hook, mock_orch, tmp_path):
    """Fires after edit_file, appends errors to result."""
    f = tmp_path / "test.py"
    f.write_text("x = 1\n")

    mock_orch.get_diagnostics = AsyncMock(return_value=[
        Diagnostic(path=str(f), line=1, col=1, severity=1, message="Error 1"),
        Diagnostic(path=str(f), line=2, col=1, severity=2, message="Warning 1"),
    ])

    result = FakeToolResult(content=f"Updated {f}")
    new_result = await hook("edit_file", {"path": str(f)}, result)

    assert "\u26a0\ufe0f LSP detected 2 issue(s)" in new_result.content


# ------------------------------------------------------------------
# Does NOT fire for other tools
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_does_not_fire_for_file_read(hook, mock_orch):
    """Does NOT fire after file_read or grep."""
    result = FakeToolResult(content="file contents...")
    new_result = await hook("file_read", {"path": "/tmp/test.py"}, result)

    # Result should be unchanged
    assert new_result is result
    mock_orch.notify_file_changed.assert_not_called()


@pytest.mark.asyncio
async def test_does_not_fire_for_grep(hook, mock_orch):
    """Does NOT fire after grep."""
    result = FakeToolResult(content="matches...")
    new_result = await hook("grep", {"pattern": "foo"}, result)

    assert new_result is result
    mock_orch.notify_file_changed.assert_not_called()


@pytest.mark.asyncio
async def test_does_not_fire_for_bash(hook, mock_orch):
    """Does NOT fire after bash."""
    result = FakeToolResult(content="output")
    new_result = await hook("bash", {"command": "ls"}, result)

    assert new_result is result


# ------------------------------------------------------------------
# Appends nothing when no errors
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_errors_returns_unchanged(hook, mock_orch, tmp_path):
    """Appends nothing when no errors detected."""
    f = tmp_path / "clean.py"
    f.write_text("x: int = 42\n")

    mock_orch.get_diagnostics = AsyncMock(return_value=[])

    result = FakeToolResult(content=f"Wrote {f}")
    new_result = await hook("write_file", {"path": str(f)}, result)

    # Should return original result unchanged
    assert new_result is result


@pytest.mark.asyncio
async def test_info_hints_not_appended(hook, mock_orch, tmp_path):
    """Only ERROR and WARNING are appended, not INFO/HINT."""
    f = tmp_path / "test.py"
    f.write_text("x = 1\n")

    mock_orch.get_diagnostics = AsyncMock(return_value=[
        Diagnostic(path=str(f), line=1, col=1, severity=3, message="Info message"),
        Diagnostic(path=str(f), line=1, col=1, severity=4, message="Hint message"),
    ])

    result = FakeToolResult(content=f"Wrote {f}")
    new_result = await hook("write_file", {"path": str(f)}, result)

    # No errors/warnings — result unchanged
    assert new_result is result


# ------------------------------------------------------------------
# Handles missing LSP server gracefully
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handles_missing_server_gracefully(hook, mock_orch):
    """Handles missing LSP server gracefully (no crash, no output)."""
    mock_orch.notify_file_changed = AsyncMock(side_effect=Exception("No server"))

    result = FakeToolResult(content="Wrote /tmp/test.py")
    new_result = await hook("write_file", {"path": "/tmp/test.py"}, result)

    # Should return original result, not crash
    assert new_result is result


# ------------------------------------------------------------------
# Handles error results (doesn't fire)
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_skips_error_results(hook, mock_orch, tmp_path):
    """Does not fire when the tool result itself is an error."""
    f = tmp_path / "test.py"
    f.write_text("x = 1\n")

    result = FakeToolResult(content="File not found", is_error=True)
    new_result = await hook("write_file", {"path": str(f)}, result)

    assert new_result is result
    mock_orch.notify_file_changed.assert_not_called()


# ------------------------------------------------------------------
# Disabled hook
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_disabled_hook_passes_through(mock_orch, tmp_path):
    """Disabled hook returns result unchanged."""
    hook = LSPDiagnosticsHook(orchestrator=mock_orch, delay_ms=0, enabled=False)
    f = tmp_path / "test.py"
    f.write_text("x = 1\n")

    result = FakeToolResult(content=f"Wrote {f}")
    new_result = await hook("write_file", {"path": str(f)}, result)

    assert new_result is result
    mock_orch.notify_file_changed.assert_not_called()
