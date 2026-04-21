"""Tests for Sprint 2: tool registry, dispatch, and builtin tool behavior."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult
from prometheus.tools.builtin import (
    BashTool,
    FileEditTool,
    FileReadTool,
    FileWriteTool,
    GlobTool,
    GrepTool,
)
from prometheus.tools.builtin.bash import BashToolInput
from prometheus.tools.builtin.file_edit import FileEditToolInput
from prometheus.tools.builtin.file_read import FileReadToolInput
from prometheus.tools.builtin.file_write import FileWriteToolInput
from prometheus.tools.builtin.glob import GlobToolInput
from prometheus.tools.builtin.grep import GrepToolInput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Return a fresh temp directory for each test."""
    return tmp_path


@pytest.fixture
def ctx(workspace: Path) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=workspace)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

def test_registry_register_and_get():
    registry = ToolRegistry()
    tool = BashTool()
    registry.register(tool)
    assert registry.get("bash") is tool


def test_registry_get_missing_returns_none():
    registry = ToolRegistry()
    assert registry.get("nonexistent") is None


def test_registry_list_tools():
    registry = ToolRegistry()
    registry.register(BashTool())
    registry.register(FileReadTool())
    names = {t.name for t in registry.list_tools()}
    assert names == {"bash", "read_file"}


def test_registry_to_api_schema():
    registry = ToolRegistry()
    registry.register(BashTool())
    schemas = registry.to_api_schema()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "bash"
    assert "input_schema" in schemas[0]


def test_registry_list_schemas_alias():
    registry = ToolRegistry()
    registry.register(BashTool())
    assert registry.list_schemas() == registry.to_api_schema()


def test_registry_to_openai_schemas():
    registry = ToolRegistry()
    registry.register(BashTool())
    schemas = registry.to_openai_schemas()
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "bash"
    assert "parameters" in schemas[0]["function"]


def test_registry_list_schemas_for_task_matches():
    registry = ToolRegistry()
    registry.register(BashTool())
    registry.register(FileReadTool())
    schemas = registry.list_schemas_for_task("run a bash command")
    names = {s["name"] for s in schemas}
    assert "bash" in names


def test_registry_list_schemas_for_task_fallback():
    registry = ToolRegistry()
    registry.register(BashTool())
    schemas = registry.list_schemas_for_task("xyzzy unrecognized words")
    # Falls back to all schemas
    assert len(schemas) == 1


# ---------------------------------------------------------------------------
# BaseTool schema methods
# ---------------------------------------------------------------------------

def test_tool_to_api_schema():
    tool = BashTool()
    schema = tool.to_api_schema()
    assert schema["name"] == "bash"
    assert "input_schema" in schema


def test_tool_to_openai_schema():
    tool = BashTool()
    schema = tool.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "bash"
    assert "parameters" in schema["function"]


# ---------------------------------------------------------------------------
# BashTool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bash_tool_basic(ctx):
    tool = BashTool()
    result = await tool.execute(BashToolInput(command="echo hello"), ctx)
    assert result.output == "hello"
    assert not result.is_error


@pytest.mark.asyncio
async def test_bash_tool_captures_stderr(ctx):
    tool = BashTool()
    result = await tool.execute(BashToolInput(command="echo err >&2"), ctx)
    assert "err" in result.output


@pytest.mark.asyncio
async def test_bash_tool_error_exit_code(ctx):
    tool = BashTool()
    result = await tool.execute(BashToolInput(command="exit 1"), ctx)
    assert result.is_error


@pytest.mark.asyncio
async def test_bash_tool_no_output(ctx):
    tool = BashTool()
    result = await tool.execute(BashToolInput(command="true"), ctx)
    assert result.output == "(no output)"
    assert not result.is_error


@pytest.mark.asyncio
async def test_bash_tool_timeout(ctx):
    tool = BashTool()
    result = await tool.execute(
        BashToolInput(command="sleep 10", timeout_seconds=1), ctx
    )
    assert result.is_error
    assert "timed out" in result.output


@pytest.mark.asyncio
async def test_bash_tool_output_truncation(ctx):
    tool = BashTool(max_output=100)
    result = await tool.execute(
        BashToolInput(command="python3 -c \"print('x' * 500)\""), ctx
    )
    assert "[truncated]" in result.output
    assert len(result.output) < 200


@pytest.mark.asyncio
async def test_bash_tool_workspace_lock_allows_inside(workspace: Path):
    ctx = ToolExecutionContext(cwd=workspace)
    tool = BashTool(workspace=workspace)
    result = await tool.execute(BashToolInput(command="echo ok"), ctx)
    assert result.output == "ok"
    assert not result.is_error


@pytest.mark.asyncio
async def test_bash_tool_workspace_lock_falls_back_when_no_explicit_cwd(workspace: Path):
    """When context cwd is outside workspace and no explicit cwd is given,
    BashTool falls back to the workspace root instead of blocking."""
    outside = Path(tempfile.gettempdir())
    ctx = ToolExecutionContext(cwd=outside)
    tool = BashTool(workspace=workspace)
    result = await tool.execute(BashToolInput(command="pwd"), ctx)
    assert not result.is_error
    assert str(workspace) in result.output


@pytest.mark.asyncio
async def test_bash_tool_workspace_lock_blocks_cwd_override(workspace: Path):
    ctx = ToolExecutionContext(cwd=workspace)
    tool = BashTool(workspace=workspace)
    result = await tool.execute(
        BashToolInput(command="echo pwned", cwd="/tmp"), ctx
    )
    assert result.is_error
    assert "Workspace lock violation" in result.output


# ---------------------------------------------------------------------------
# FileReadTool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_file_read_basic(workspace: Path, ctx):
    p = workspace / "hello.txt"
    p.write_text("line1\nline2\nline3\n")
    tool = FileReadTool()
    result = await tool.execute(FileReadToolInput(path=str(p)), ctx)
    assert "line1" in result.output
    assert "line2" in result.output
    assert not result.is_error


@pytest.mark.asyncio
async def test_file_read_with_offset_and_limit(workspace: Path, ctx):
    p = workspace / "multi.txt"
    p.write_text("\n".join(f"line{i}" for i in range(1, 11)))
    tool = FileReadTool()
    result = await tool.execute(FileReadToolInput(path=str(p), offset=2, limit=3), ctx)
    lines = result.output.splitlines()
    assert len(lines) == 3
    assert "line3" in lines[0]


@pytest.mark.asyncio
async def test_file_read_missing(ctx):
    tool = FileReadTool()
    result = await tool.execute(FileReadToolInput(path="/nonexistent/path.txt"), ctx)
    assert result.is_error
    assert "not found" in result.output


@pytest.mark.asyncio
async def test_file_read_binary(workspace: Path, ctx):
    p = workspace / "bin.dat"
    p.write_bytes(b"\x00\x01\x02")
    tool = FileReadTool()
    result = await tool.execute(FileReadToolInput(path=str(p)), ctx)
    assert result.is_error
    assert "Binary" in result.output


def test_file_read_is_read_only():
    tool = FileReadTool()
    assert tool.is_read_only(FileReadToolInput(path="x"))


# ---------------------------------------------------------------------------
# FileWriteTool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_file_write_creates_file(workspace: Path, ctx):
    p = workspace / "out.txt"
    tool = FileWriteTool()
    result = await tool.execute(FileWriteToolInput(path=str(p), content="hello\n"), ctx)
    assert not result.is_error
    assert p.read_text() == "hello\n"


@pytest.mark.asyncio
async def test_file_write_creates_directories(workspace: Path, ctx):
    p = workspace / "deep" / "nested" / "file.txt"
    tool = FileWriteTool()
    result = await tool.execute(FileWriteToolInput(path=str(p), content="data"), ctx)
    assert not result.is_error
    assert p.exists()


@pytest.mark.asyncio
async def test_file_write_relative_path(workspace: Path, ctx):
    tool = FileWriteTool()
    result = await tool.execute(FileWriteToolInput(path="relative.txt", content="hi"), ctx)
    assert not result.is_error
    assert (workspace / "relative.txt").read_text() == "hi"


# ---------------------------------------------------------------------------
# FileEditTool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_file_edit_replaces_string(workspace: Path, ctx):
    p = workspace / "edit.txt"
    p.write_text("hello world\n")
    tool = FileEditTool()
    result = await tool.execute(
        FileEditToolInput(path=str(p), old_str="world", new_str="prometheus"), ctx
    )
    assert not result.is_error
    assert p.read_text() == "hello prometheus\n"


@pytest.mark.asyncio
async def test_file_edit_replace_all(workspace: Path, ctx):
    p = workspace / "dup.txt"
    p.write_text("a b a b\n")
    tool = FileEditTool()
    result = await tool.execute(
        FileEditToolInput(path=str(p), old_str="a", new_str="x", replace_all=True), ctx
    )
    assert not result.is_error
    assert p.read_text() == "x b x b\n"


@pytest.mark.asyncio
async def test_file_edit_old_str_not_found(workspace: Path, ctx):
    p = workspace / "nope.txt"
    p.write_text("content\n")
    tool = FileEditTool()
    result = await tool.execute(
        FileEditToolInput(path=str(p), old_str="missing", new_str="x"), ctx
    )
    assert result.is_error
    assert "not found" in result.output


@pytest.mark.asyncio
async def test_file_edit_missing_file(ctx):
    tool = FileEditTool()
    result = await tool.execute(
        FileEditToolInput(path="/nonexistent.txt", old_str="x", new_str="y"), ctx
    )
    assert result.is_error


# ---------------------------------------------------------------------------
# GrepTool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_finds_matches(workspace: Path, ctx):
    (workspace / "a.txt").write_text("foo bar\nbaz\n")
    (workspace / "b.txt").write_text("no match here\n")
    tool = GrepTool()
    result = await tool.execute(GrepToolInput(pattern="foo"), ctx)
    assert not result.is_error
    assert "a.txt" in result.output
    assert "foo" in result.output


@pytest.mark.asyncio
async def test_grep_no_matches(workspace: Path, ctx):
    (workspace / "c.txt").write_text("nothing\n")
    tool = GrepTool()
    result = await tool.execute(GrepToolInput(pattern="zzznope"), ctx)
    assert "(no matches)" in result.output


@pytest.mark.asyncio
async def test_grep_case_insensitive(workspace: Path, ctx):
    (workspace / "d.txt").write_text("Hello World\n")
    tool = GrepTool()
    result = await tool.execute(
        GrepToolInput(pattern="hello", case_sensitive=False), ctx
    )
    assert "Hello" in result.output


def test_grep_is_read_only():
    tool = GrepTool()
    assert tool.is_read_only(GrepToolInput(pattern="x"))


# ---------------------------------------------------------------------------
# GlobTool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_glob_finds_files(workspace: Path, ctx):
    (workspace / "foo.py").write_text("")
    (workspace / "bar.py").write_text("")
    (workspace / "readme.md").write_text("")
    tool = GlobTool()
    result = await tool.execute(GlobToolInput(pattern="*.py"), ctx)
    assert "foo.py" in result.output
    assert "bar.py" in result.output
    assert "readme.md" not in result.output


@pytest.mark.asyncio
async def test_glob_no_matches(workspace: Path, ctx):
    tool = GlobTool()
    result = await tool.execute(GlobToolInput(pattern="*.nonexistent"), ctx)
    assert "(no matches)" in result.output


def test_glob_is_read_only():
    tool = GlobTool()
    assert tool.is_read_only(GlobToolInput(pattern="*"))
