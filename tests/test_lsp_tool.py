"""Tests for LSP tool (Sprint 20: model-facing tool interface)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from prometheus.lsp.client import Diagnostic, DocumentSymbol, HoverInfo, Location
from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin.lsp import LSPTool, LSPToolInput, set_lsp_orchestrator


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def mock_orch():
    orch = MagicMock()
    orch.get_definition = AsyncMock(return_value=[])
    orch.get_references = AsyncMock(return_value=[])
    orch.get_hover = AsyncMock(return_value=None)
    orch.get_diagnostics = AsyncMock(return_value=[])
    orch.get_symbols = AsyncMock(return_value=[])
    orch.rename = AsyncMock(return_value={})
    orch.get_symbol_context = AsyncMock(return_value="No info")
    return orch


@pytest.fixture
def ctx(tmp_path):
    return ToolExecutionContext(cwd=tmp_path)


@pytest.fixture
def tool():
    return LSPTool()


@pytest.fixture
def py_file(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("def foo():\n    return 42\n\nclass Bar:\n    pass\n")
    return f


# ------------------------------------------------------------------
# is_read_only
# ------------------------------------------------------------------

def test_is_read_only_for_definition(tool):
    args = LSPToolInput(action="definition", file="test.py", line=1)
    assert tool.is_read_only(args) is True


def test_is_not_read_only_for_rename(tool):
    args = LSPToolInput(action="rename", file="test.py", line=1, new_name="baz")
    assert tool.is_read_only(args) is False


# ------------------------------------------------------------------
# Action: definition
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_definition_returns_formatted_locations(tool, mock_orch, ctx, py_file):
    mock_orch.get_definition = AsyncMock(return_value=[
        Location(path=str(py_file), line=1, col=5),
    ])
    ctx.metadata["lsp_orchestrator"] = mock_orch

    args = LSPToolInput(action="definition", file=str(py_file), line=1, column=5)
    result = await tool.execute(args, ctx)

    assert not result.is_error
    assert "Definition (1):" in result.output
    assert "test.py" in result.output


# ------------------------------------------------------------------
# Action: references
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_references_returns_formatted_locations(tool, mock_orch, ctx, py_file):
    mock_orch.get_references = AsyncMock(return_value=[
        Location(path=str(py_file), line=1, col=5),
        Location(path=str(py_file), line=5, col=7),
    ])
    ctx.metadata["lsp_orchestrator"] = mock_orch

    args = LSPToolInput(action="references", file=str(py_file), line=1, column=5)
    result = await tool.execute(args, ctx)

    assert not result.is_error
    assert "References (2):" in result.output


# ------------------------------------------------------------------
# Action: hover
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hover_returns_type_info(tool, mock_orch, ctx, py_file):
    mock_orch.get_hover = AsyncMock(
        return_value=HoverInfo(contents="(function) def foo() -> int"),
    )
    ctx.metadata["lsp_orchestrator"] = mock_orch

    args = LSPToolInput(action="hover", file=str(py_file), line=1, column=5)
    result = await tool.execute(args, ctx)

    assert not result.is_error
    assert "def foo()" in result.output


@pytest.mark.asyncio
async def test_hover_none_response(tool, mock_orch, ctx, py_file):
    mock_orch.get_hover = AsyncMock(return_value=None)
    ctx.metadata["lsp_orchestrator"] = mock_orch

    args = LSPToolInput(action="hover", file=str(py_file), line=1, column=5)
    result = await tool.execute(args, ctx)

    assert not result.is_error
    assert "No hover" in result.output


# ------------------------------------------------------------------
# Action: diagnostics
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_diagnostics_returns_errors(tool, mock_orch, ctx, py_file):
    mock_orch.get_diagnostics = AsyncMock(return_value=[
        Diagnostic(
            path=str(py_file), line=2, col=5,
            severity=1, message="Type 'str' not assignable to 'int'",
        ),
    ])
    ctx.metadata["lsp_orchestrator"] = mock_orch

    args = LSPToolInput(action="diagnostics", file=str(py_file))
    result = await tool.execute(args, ctx)

    assert not result.is_error
    assert "Diagnostics" in result.output
    assert "Type 'str'" in result.output


@pytest.mark.asyncio
async def test_diagnostics_no_errors(tool, mock_orch, ctx, py_file):
    ctx.metadata["lsp_orchestrator"] = mock_orch

    args = LSPToolInput(action="diagnostics", file=str(py_file))
    result = await tool.execute(args, ctx)

    assert not result.is_error
    assert "No diagnostics" in result.output


# ------------------------------------------------------------------
# Action: symbols
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_symbols_returns_file_structure(tool, mock_orch, ctx, py_file):
    mock_orch.get_symbols = AsyncMock(return_value=[
        DocumentSymbol(name="foo", kind=12, range_start_line=1, range_end_line=2),
        DocumentSymbol(name="Bar", kind=5, range_start_line=4, range_end_line=5),
    ])
    ctx.metadata["lsp_orchestrator"] = mock_orch

    args = LSPToolInput(action="symbols", file=str(py_file))
    result = await tool.execute(args, ctx)

    assert not result.is_error
    assert "foo" in result.output
    assert "Bar" in result.output


# ------------------------------------------------------------------
# Action: context (the money test)
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_context_returns_combined_info(tool, mock_orch, ctx, py_file):
    mock_orch.get_symbol_context = AsyncMock(return_value=(
        "Type: class Bar\n"
        "Defined: test.py:4:7\n"
        "References (3):\n"
        "  - test.py:4:7\n"
        "  - views.py:10:5\n"
        "  - api.py:22:12"
    ))
    ctx.metadata["lsp_orchestrator"] = mock_orch

    args = LSPToolInput(action="context", file=str(py_file), line=4, column=7)
    result = await tool.execute(args, ctx)

    assert not result.is_error
    assert "class Bar" in result.output
    assert "Defined:" in result.output
    assert "References (3):" in result.output


# ------------------------------------------------------------------
# Action: rename
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rename_returns_edit_summary(tool, mock_orch, ctx, py_file):
    mock_orch.rename = AsyncMock(return_value={
        str(py_file): [{"start_line": 1, "end_line": 1, "newText": "bar"}],
        str(py_file.parent / "other.py"): [{"start_line": 5, "end_line": 5, "newText": "bar"}],
    })
    ctx.metadata["lsp_orchestrator"] = mock_orch

    args = LSPToolInput(
        action="rename", file=str(py_file), line=1, column=5, new_name="bar",
    )
    result = await tool.execute(args, ctx)

    assert not result.is_error
    assert "Renamed to 'bar'" in result.output
    assert "2 file(s)" in result.output


# ------------------------------------------------------------------
# Symbol name resolution
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolves_symbol_name_to_position(tool, mock_orch, ctx, py_file):
    """When line/col not given but symbol is, search file for it."""
    mock_orch.get_symbol_context = AsyncMock(return_value="Type: function foo")
    ctx.metadata["lsp_orchestrator"] = mock_orch

    args = LSPToolInput(action="context", file=str(py_file), symbol="foo")
    result = await tool.execute(args, ctx)

    assert not result.is_error
    # Should have resolved 'foo' to line 1
    mock_orch.get_symbol_context.assert_called_once()
    call_args = mock_orch.get_symbol_context.call_args
    assert call_args[0][1] == 1  # line
    assert call_args[0][2] == 5  # col (position of 'foo' in 'def foo():')


@pytest.mark.asyncio
async def test_symbol_not_found_returns_error(tool, mock_orch, ctx, py_file):
    ctx.metadata["lsp_orchestrator"] = mock_orch

    args = LSPToolInput(action="context", file=str(py_file), symbol="nonexistent")
    result = await tool.execute(args, ctx)

    assert result.is_error
    assert "not found" in result.output


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_orchestrator_returns_error(tool, ctx, py_file):
    """Handles missing language server gracefully."""
    args = LSPToolInput(action="definition", file=str(py_file), line=1)
    result = await tool.execute(args, ctx)

    assert result.is_error
    assert "not available" in result.output


@pytest.mark.asyncio
async def test_missing_file_returns_error(tool, mock_orch, ctx):
    ctx.metadata["lsp_orchestrator"] = mock_orch

    args = LSPToolInput(action="definition", file="/nonexistent.py", line=1)
    result = await tool.execute(args, ctx)

    assert result.is_error
    assert "not found" in result.output


@pytest.mark.asyncio
async def test_unknown_action_returns_error(tool, mock_orch, ctx, py_file):
    ctx.metadata["lsp_orchestrator"] = mock_orch

    args = LSPToolInput(action="badaction", file=str(py_file))
    result = await tool.execute(args, ctx)

    assert result.is_error
    assert "Unknown action" in result.output


@pytest.mark.asyncio
async def test_missing_line_for_definition(tool, mock_orch, ctx, py_file):
    ctx.metadata["lsp_orchestrator"] = mock_orch

    args = LSPToolInput(action="definition", file=str(py_file))
    result = await tool.execute(args, ctx)

    assert result.is_error
    assert "Line number" in result.output
