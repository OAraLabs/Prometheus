"""Tests for ToolSearchTool: deferred tool loading via search/select."""

from __future__ import annotations

import json

import pytest

from prometheus.tools.base import ToolExecutionContext, ToolRegistry
from prometheus.tools.builtin.bash import BashTool
from prometheus.tools.builtin.file_read import FileReadTool
from prometheus.tools.builtin.grep import GrepTool
from prometheus.tools.tool_search import ToolSearchTool, _levenshtein


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(BashTool())
    reg.register(FileReadTool())
    reg.register(GrepTool())
    return reg


@pytest.fixture
def tool(registry):
    t = ToolSearchTool()
    t.set_registry(registry)
    return t


@pytest.fixture
def ctx(tmp_path):
    return ToolExecutionContext(cwd=tmp_path)


# ---------------------------------------------------------------------------
# Levenshtein helper
# ---------------------------------------------------------------------------

def test_levenshtein_identical():
    assert _levenshtein("abc", "abc") == 0


def test_levenshtein_empty():
    assert _levenshtein("", "abc") == 3
    assert _levenshtein("abc", "") == 3


def test_levenshtein_single_edit():
    assert _levenshtein("cat", "bat") == 1


def test_levenshtein_insertion():
    assert _levenshtein("bsh", "bash") == 1


# ---------------------------------------------------------------------------
# Search by name
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_by_exact_name(tool, ctx):
    from prometheus.tools.tool_search import ToolSearchInput

    result = await tool.execute(ToolSearchInput(query="bash"), ctx)
    assert not result.is_error
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert data[0]["name"] == "bash"


@pytest.mark.asyncio
async def test_search_by_partial_name(tool, ctx):
    from prometheus.tools.tool_search import ToolSearchInput

    result = await tool.execute(ToolSearchInput(query="grep"), ctx)
    assert not result.is_error
    data = json.loads(result.output)
    names = [entry["name"] for entry in data]
    assert "grep" in names


# ---------------------------------------------------------------------------
# Search by description keywords
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_by_description(tool, ctx):
    from prometheus.tools.tool_search import ToolSearchInput

    result = await tool.execute(ToolSearchInput(query="shell"), ctx)
    assert not result.is_error
    data = json.loads(result.output)
    names = [entry["name"] for entry in data]
    # BashTool description contains "shell"
    assert names[0] == "bash"


@pytest.mark.asyncio
async def test_search_by_description_keyword_regex(tool, ctx):
    from prometheus.tools.tool_search import ToolSearchInput

    result = await tool.execute(ToolSearchInput(query="regular expression"), ctx)
    assert not result.is_error
    data = json.loads(result.output)
    names = [entry["name"] for entry in data]
    # GrepTool description contains "regular expression"
    assert names[0] == "grep"


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fuzzy_match_bsh_finds_bash(tool, ctx):
    """Typo 'bsh' should rank 'bash' as the best fuzzy match."""
    from prometheus.tools.tool_search import ToolSearchInput

    result = await tool.execute(ToolSearchInput(query="bsh"), ctx)
    assert not result.is_error
    data = json.loads(result.output)
    names = [entry["name"] for entry in data]
    assert names[0] == "bash"


@pytest.mark.asyncio
async def test_fuzzy_match_grp_finds_grep(tool, ctx):
    from prometheus.tools.tool_search import ToolSearchInput

    result = await tool.execute(ToolSearchInput(query="grp"), ctx)
    assert not result.is_error
    data = json.loads(result.output)
    names = [entry["name"] for entry in data]
    assert names[0] == "grep"


# ---------------------------------------------------------------------------
# Exact select
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_select_existing_tool(tool, ctx):
    from prometheus.tools.tool_search import ToolSearchInput

    result = await tool.execute(
        ToolSearchInput(query="bash", action="select"), ctx
    )
    assert not result.is_error
    data = json.loads(result.output)
    assert data["name"] == "bash"
    assert "input_schema" in data


@pytest.mark.asyncio
async def test_select_returns_full_schema(tool, ctx):
    from prometheus.tools.tool_search import ToolSearchInput

    result = await tool.execute(
        ToolSearchInput(query="grep", action="select"), ctx
    )
    assert not result.is_error
    data = json.loads(result.output)
    assert data["name"] == "grep"
    assert "description" in data
    assert "input_schema" in data
    # Verify the schema has properties from GrepToolInput
    props = data["input_schema"].get("properties", {})
    assert "pattern" in props


# ---------------------------------------------------------------------------
# Empty query returns all tool names
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_query_returns_all_names(tool, ctx):
    from prometheus.tools.tool_search import ToolSearchInput

    result = await tool.execute(ToolSearchInput(query=""), ctx)
    assert not result.is_error
    data = json.loads(result.output)
    names = data["tools"]
    assert sorted(names) == ["bash", "grep", "read_file"]


@pytest.mark.asyncio
async def test_whitespace_query_returns_all_names(tool, ctx):
    from prometheus.tools.tool_search import ToolSearchInput

    result = await tool.execute(ToolSearchInput(query="   "), ctx)
    assert not result.is_error
    data = json.loads(result.output)
    assert "tools" in data


# ---------------------------------------------------------------------------
# Select nonexistent tool returns helpful error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_select_nonexistent_returns_error(tool, ctx):
    from prometheus.tools.tool_search import ToolSearchInput

    result = await tool.execute(
        ToolSearchInput(query="nonexistent_tool", action="select"), ctx
    )
    assert result.is_error
    data = json.loads(result.output)
    assert "error" in data
    assert "nonexistent_tool" in data["error"]
    assert "available" in data
    assert "bash" in data["available"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_registry_returns_error(ctx):
    from prometheus.tools.tool_search import ToolSearchInput

    t = ToolSearchTool()  # no set_registry call
    result = await t.execute(ToolSearchInput(query="bash"), ctx)
    assert result.is_error
    assert "registry" in result.output.lower()


def test_is_read_only_always_true(tool):
    from prometheus.tools.tool_search import ToolSearchInput

    assert tool.is_read_only(ToolSearchInput(query="anything"))
    assert tool.is_read_only(ToolSearchInput(query="x", action="select"))


@pytest.mark.asyncio
async def test_search_results_capped_at_five(ctx):
    """Even with many tools registered, search returns at most 5."""
    from prometheus.tools.tool_search import ToolSearchInput

    reg = ToolRegistry()
    # Register more than 5 tools: reuse same types with different names
    for i in range(10):
        t = BashTool()
        t.name = f"bash_{i}"
        t.description = f"Shell variant {i}"
        reg.register(t)

    ts = ToolSearchTool()
    ts.set_registry(reg)
    result = await ts.execute(ToolSearchInput(query="bash"), ctx)
    data = json.loads(result.output)
    assert len(data) == 5
