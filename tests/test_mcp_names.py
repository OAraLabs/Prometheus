"""Tests for Sprint 12: MCP name sanitization."""

from __future__ import annotations

import pytest

from prometheus.mcp.names import (
    TOOL_NAME_SEPARATOR,
    build_safe_tool_name,
    sanitize_server_name,
    sanitize_tool_name,
)


class TestSanitizeServerName:
    def test_simple_name(self):
        used = set()
        assert sanitize_server_name("context7", used) == "context7"
        assert "context7" in used

    def test_strips_special_chars(self):
        used = set()
        assert sanitize_server_name("my-server.v2", used) == "my_server_v2"

    def test_deduplicates(self):
        used = set()
        assert sanitize_server_name("server", used) == "server"
        assert sanitize_server_name("server", used) == "server_1"
        assert sanitize_server_name("server", used) == "server_2"

    def test_empty_name_fallback(self):
        used = set()
        assert sanitize_server_name("---", used) == "mcp"

    def test_preserves_lowercase(self):
        used = set()
        assert sanitize_server_name("MyServer", used) == "myserver"


class TestSanitizeToolName:
    def test_clean_name(self):
        assert sanitize_tool_name("resolve_library_id") == "resolve_library_id"

    def test_hyphens_become_underscores(self):
        assert sanitize_tool_name("resolve-library-id") == "resolve_library_id"

    def test_empty_fallback(self):
        assert sanitize_tool_name("---") == "tool"


class TestBuildSafeToolName:
    def test_basic_pattern(self):
        reserved = set()
        name = build_safe_tool_name("context7", "resolve_library_id", reserved)
        assert name == "mcp__context7__resolve_library_id"
        assert name.lower() in reserved

    def test_collision_avoidance(self):
        reserved = {"mcp__srv__tool"}
        name = build_safe_tool_name("srv", "tool", reserved)
        assert name == "mcp__srv__tool_1"

    def test_separator(self):
        reserved = set()
        name = build_safe_tool_name("s", "t", reserved)
        assert TOOL_NAME_SEPARATOR in name
        parts = name.split(TOOL_NAME_SEPARATOR)
        assert parts[0] == "mcp"
        assert parts[1] == "s"
        assert parts[2] == "t"

    def test_sanitizes_tool_name(self):
        reserved = set()
        name = build_safe_tool_name("srv", "get-library-docs", reserved)
        assert name == "mcp__srv__get_library_docs"
