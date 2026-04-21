"""Tests for Feature 6: Structured Error Feedback."""

from __future__ import annotations

import pytest

from prometheus.adapter.validator import Strictness, ToolCallValidator
from prometheus.tools.base import BaseTool, ToolRegistry
from prometheus.tools.builtin.bash import BashTool
from prometheus.tools.builtin.file_read import FileReadTool
from prometheus.tools.builtin.grep import GrepTool


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(BashTool())
    reg.register(FileReadTool())
    reg.register(GrepTool())
    return reg


@pytest.fixture
def validator():
    return ToolCallValidator(strictness=Strictness.MEDIUM)


class TestStructuredErrors:
    def test_empty_tool_name_includes_available_tools(self, validator, registry):
        result = validator.validate("", {}, registry)
        assert not result.valid
        assert "Available tools:" in result.error
        assert "bash" in result.error

    def test_unknown_tool_includes_example(self, validator, registry):
        result = validator.validate("nonexistent", {}, registry)
        assert not result.valid
        assert "Expected format:" in result.error
        assert '"name"' in result.error

    def test_repair_failure_includes_structured_error(self, validator, registry):
        # Tool name too far from anything real — repair should fail with rich msg
        result = validator.repair("xyzzy_no_match_at_all", {}, "some error", registry)
        assert not result.repaired
        assert "Available tools:" in result.error
        assert "Expected format:" in result.error

    def test_example_call_field_exists(self):
        """BaseTool subclass with example_call set is accessible."""
        tool = BashTool()
        assert tool.example_call == {"command": "ls -la"}

    def test_example_call_default_none(self):
        """BaseTool.example_call defaults to None when not set."""
        assert BaseTool.example_call is None

    def test_structured_error_contains_example_line(self, validator, registry):
        """When a tool has example_call, the error includes an Example line."""
        result = validator.validate("nonexistent", {}, registry)
        assert "Example:" in result.error
        # The example should be valid JSON containing the tool name and arguments
        assert '"bash"' in result.error
        assert '"ls -la"' in result.error
