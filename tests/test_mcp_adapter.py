"""Tests for Sprint 12: MCP adapter + runtime (unit tests with mocks)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prometheus.mcp.adapter import McpToolAdapter, register_mcp_tools
from prometheus.mcp.runtime import McpRuntime, _list_all_tools
from prometheus.mcp.types import (
    McpCatalogTool,
    McpConnectionStatus,
    McpToolCatalog,
    create_config_fingerprint,
)
from prometheus.tools.base import ToolExecutionContext, ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# McpToolAdapter
# ---------------------------------------------------------------------------


class TestMcpToolAdapter:
    def _make_adapter(self):
        runtime = MagicMock(spec=McpRuntime)
        tool_info = McpCatalogTool(
            server_name="context7",
            safe_server_name="context7",
            tool_name="resolve_library_id",
            description="Resolve a library ID",
            input_schema={
                "type": "object",
                "properties": {
                    "libraryName": {"type": "string", "description": "Library name"},
                },
                "required": ["libraryName"],
            },
        )
        adapter = McpToolAdapter(runtime, tool_info, "mcp__context7__resolve_library_id")
        return adapter, runtime

    def test_name_and_description(self):
        adapter, _ = self._make_adapter()
        assert adapter.name == "mcp__context7__resolve_library_id"
        assert adapter.description == "Resolve a library ID"

    def test_is_read_only(self):
        adapter, _ = self._make_adapter()
        assert adapter.is_read_only(MagicMock()) is True

    def test_api_schema_uses_mcp_schema(self):
        adapter, _ = self._make_adapter()
        schema = adapter.to_api_schema()
        assert schema["name"] == "mcp__context7__resolve_library_id"
        assert "libraryName" in schema["input_schema"]["properties"]

    def test_openai_schema_uses_mcp_schema(self):
        adapter, _ = self._make_adapter()
        schema = adapter.to_openai_schema()
        assert schema["type"] == "function"
        assert "libraryName" in schema["function"]["parameters"]["properties"]

    def test_execute_calls_runtime(self):
        adapter, runtime = self._make_adapter()
        runtime.call_tool = AsyncMock(return_value="result text")

        ctx = ToolExecutionContext(cwd=Path("."))
        input_model = adapter.input_model(libraryName="nextjs")
        result = asyncio.run(adapter.execute(input_model, ctx))

        assert isinstance(result, ToolResult)
        assert result.output == "result text"
        assert result.is_error is False
        runtime.call_tool.assert_called_once_with(
            "context7", "resolve_library_id", {"libraryName": "nextjs"}
        )

    def test_execute_handles_error(self):
        adapter, runtime = self._make_adapter()
        runtime.call_tool = AsyncMock(side_effect=ValueError("boom"))

        ctx = ToolExecutionContext(cwd=Path("."))
        input_model = adapter.input_model(libraryName="nextjs")
        result = asyncio.run(adapter.execute(input_model, ctx))

        assert result.is_error is True
        assert "boom" in result.output


# ---------------------------------------------------------------------------
# register_mcp_tools
# ---------------------------------------------------------------------------


class TestRegisterMcpTools:
    def test_registers_tools(self):
        runtime = MagicMock(spec=McpRuntime)
        runtime.list_tools.return_value = [
            McpCatalogTool(
                server_name="srv",
                safe_server_name="srv",
                tool_name="tool_a",
                description="Tool A",
                input_schema={"type": "object", "properties": {}},
            ),
            McpCatalogTool(
                server_name="srv",
                safe_server_name="srv",
                tool_name="tool_b",
                description="Tool B",
                input_schema={"type": "object", "properties": {}},
            ),
        ]

        registry = ToolRegistry()
        count = register_mcp_tools(registry, runtime)

        assert count == 2
        assert registry.get("mcp__srv__tool_a") is not None
        assert registry.get("mcp__srv__tool_b") is not None

    def test_avoids_name_collisions_with_existing(self):
        runtime = MagicMock(spec=McpRuntime)
        runtime.list_tools.return_value = [
            McpCatalogTool(
                server_name="srv",
                safe_server_name="srv",
                tool_name="bash",  # name collision with builtin
                description="Custom bash",
                input_schema={"type": "object", "properties": {}},
            ),
        ]

        registry = ToolRegistry()
        # Simulate existing tool
        existing = MagicMock()
        existing.name = "mcp__srv__bash"
        registry.register(existing)

        count = register_mcp_tools(registry, runtime)
        assert count == 1
        # Should get a suffixed name
        assert registry.get("mcp__srv__bash_1") is not None


# ---------------------------------------------------------------------------
# McpRuntime (unit tests — no real MCP connections)
# ---------------------------------------------------------------------------


class TestMcpRuntimeUnit:
    def test_empty_config(self):
        runtime = McpRuntime({})
        assert runtime.list_tools() == []
        assert runtime.list_statuses() == []

    def test_initial_statuses_pending(self):
        runtime = McpRuntime({"server_a": {}, "server_b": {}})
        statuses = runtime.list_statuses()
        assert len(statuses) == 2
        assert all(s.state == "pending" for s in statuses)

    def test_config_fingerprint_changes(self):
        r1 = McpRuntime({"a": {"command": "x"}})
        r2 = McpRuntime({"a": {"command": "y"}})
        assert r1.config_fingerprint != r2.config_fingerprint

    def test_config_fingerprint_stable(self):
        cfg = {"a": {"command": "x", "args": ["1"]}}
        r1 = McpRuntime(cfg)
        r2 = McpRuntime(cfg)
        assert r1.config_fingerprint == r2.config_fingerprint

    def test_connect_all_fails_gracefully_on_bad_config(self):
        runtime = McpRuntime({"bad": {"foo": "bar"}})
        asyncio.run(runtime.connect_all())
        statuses = runtime.list_statuses()
        assert statuses[0].state == "failed"
        assert "Could not resolve transport" in statuses[0].detail

    def test_call_tool_raises_for_unconnected_server(self):
        runtime = McpRuntime({})
        with pytest.raises(ValueError, match="not connected"):
            asyncio.run(runtime.call_tool("fake", "tool", {}))


# ---------------------------------------------------------------------------
# Config fingerprint helper
# ---------------------------------------------------------------------------


class TestConfigFingerprint:
    def test_deterministic(self):
        cfg = {"ctx7": {"command": "npx", "args": ["-y", "ctx7"]}}
        assert create_config_fingerprint(cfg) == create_config_fingerprint(cfg)

    def test_changes_on_diff(self):
        a = create_config_fingerprint({"x": {"command": "a"}})
        b = create_config_fingerprint({"x": {"command": "b"}})
        assert a != b

    def test_key_order_irrelevant(self):
        a = create_config_fingerprint({"a": 1, "b": 2})
        b = create_config_fingerprint({"b": 2, "a": 1})
        assert a == b
