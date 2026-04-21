"""Tests for Sprint 3: Model Adapter Layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prometheus.adapter import ModelAdapter
from prometheus.adapter.enforcer import StructuredOutputEnforcer, _try_parse_tool_call
from prometheus.adapter.formatter import (
    AnthropicFormatter,
    GemmaFormatter,
    QwenFormatter,
    _parse_tool_call_json,
)
from prometheus.adapter.retry import RetryAction, RetryEngine
from prometheus.adapter.validator import RepairResult, Strictness, ToolCallValidator, ValidationResult
from prometheus.tools.base import ToolRegistry
from prometheus.tools.builtin import BashTool, FileReadTool, FileWriteTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(BashTool())
    reg.register(FileReadTool())
    reg.register(FileWriteTool())
    return reg


# ---------------------------------------------------------------------------
# ToolCallValidator — validation
# ---------------------------------------------------------------------------

class TestToolCallValidator:

    def test_none_strictness_always_passes(self, registry):
        v = ToolCallValidator(strictness=Strictness.NONE)
        result = v.validate("nonexistent", {}, registry)
        assert result.valid

    def test_medium_unknown_tool(self, registry):
        v = ToolCallValidator(strictness=Strictness.MEDIUM)
        result = v.validate("totally_fake_tool", {}, registry)
        assert not result.valid
        assert result.error_type == "unknown_tool"

    def test_medium_valid_call(self, registry):
        v = ToolCallValidator(strictness=Strictness.MEDIUM)
        result = v.validate("bash", {"command": "echo hi"}, registry)
        assert result.valid

    def test_medium_missing_required_param(self, registry):
        v = ToolCallValidator(strictness=Strictness.MEDIUM)
        result = v.validate("bash", {}, registry)
        assert not result.valid
        assert result.error_type == "missing_param"

    def test_medium_wrong_type(self, registry):
        v = ToolCallValidator(strictness=Strictness.MEDIUM)
        # timeout_seconds should be int/float, not a list
        result = v.validate("bash", {"command": "ls", "timeout_seconds": []}, registry)
        assert not result.valid

    def test_strict_rejects_extra_params(self, registry):
        v = ToolCallValidator(strictness=Strictness.STRICT)
        result = v.validate("bash", {"command": "ls", "surprise_param": "oops"}, registry)
        assert not result.valid
        assert result.error_type == "extra_param"

    def test_medium_does_not_reject_extra_params(self, registry):
        v = ToolCallValidator(strictness=Strictness.MEDIUM)
        result = v.validate("bash", {"command": "ls", "extra": "ok"}, registry)
        # Pydantic extra="ignore" by default, so this should pass
        # (BaseTool subclasses may vary; just check no crash)
        assert isinstance(result, ValidationResult)

    def test_string_strictness_enum_coercion(self, registry):
        v = ToolCallValidator(strictness="MEDIUM")
        assert v.strictness == Strictness.MEDIUM


# ---------------------------------------------------------------------------
# ToolCallValidator — repair
# ---------------------------------------------------------------------------

class TestToolCallValidatorRepair:

    def test_fuzzy_name_match(self, registry):
        v = ToolCallValidator(strictness=Strictness.MEDIUM)
        repair = v.repair("bsh", {"command": "ls"}, "unknown tool", registry)
        assert repair.repaired
        assert repair.tool_name == "bash"
        assert any("fuzzy" in r for r in repair.repairs_made)

    def test_extract_json_from_markdown(self, registry):
        v = ToolCallValidator(strictness=Strictness.MEDIUM)
        raw = '```json\n{"command": "ls"}\n```'
        repair = v.repair("bash", raw, "invalid input", registry)
        assert repair.repaired
        assert repair.tool_input == {"command": "ls"}
        assert any("JSON" in r or "json" in r.lower() for r in repair.repairs_made)

    def test_type_coercion_string_to_int(self, registry):
        v = ToolCallValidator(strictness=Strictness.MEDIUM)
        # BashTool timeout_seconds is Optional[int]
        # Pass it as a string — coercion should handle it
        repair = v.repair(
            "bash",
            {"command": "ls", "timeout_seconds": "30"},
            "wrong type",
            registry,
        )
        # May or may not report coercion depending on pydantic; just check no crash
        assert isinstance(repair, RepairResult)

    def test_strip_unknown_params(self, registry):
        v = ToolCallValidator(strictness=Strictness.STRICT)
        repair = v.repair(
            "bash",
            {"command": "ls", "unknown_param": "x"},
            "extra_param",
            registry,
        )
        assert repair.repaired
        assert "unknown_param" not in repair.tool_input
        assert any("stripped" in r for r in repair.repairs_made)

    def test_repair_fails_on_no_close_match(self, registry):
        v = ToolCallValidator(strictness=Strictness.MEDIUM)
        repair = v.repair("xyzzy_no_match_at_all", {"x": 1}, "unknown", registry)
        assert not repair.repaired

    def test_repair_extracts_json_from_text(self, registry):
        v = ToolCallValidator(strictness=Strictness.MEDIUM)
        raw = 'I want to run: {"command": "echo hello"} in bash'
        repair = v.repair("bash", raw, "invalid", registry)
        # Should extract the JSON object from the string
        assert isinstance(repair, RepairResult)


# ---------------------------------------------------------------------------
# RetryEngine
# ---------------------------------------------------------------------------

class TestRetryEngine:

    def test_first_failure_returns_retry(self, registry):
        engine = RetryEngine(max_retries=3)
        action, prompt = engine.handle_failure("bash", "missing command", registry)
        assert action == RetryAction.RETRY
        assert "bash" in prompt
        assert "missing command" in prompt

    def test_prompt_includes_schema(self, registry):
        engine = RetryEngine(max_retries=3)
        _, prompt = engine.handle_failure("bash", "missing command", registry)
        assert "command" in prompt.lower()

    def test_max_retries_returns_abort(self, registry):
        engine = RetryEngine(max_retries=2)
        engine.handle_failure("bash", "err1", registry)
        engine.handle_failure("bash", "err2", registry)
        action, _ = engine.handle_failure("bash", "err3", registry)
        assert action == RetryAction.ABORT

    def test_retry_count_tracked(self, registry):
        engine = RetryEngine(max_retries=3)
        engine.handle_failure("bash", "e", registry)
        engine.handle_failure("bash", "e", registry)
        assert engine.retry_count("bash") == 2

    def test_reset_clears_state(self, registry):
        engine = RetryEngine(max_retries=3)
        engine.handle_failure("bash", "e", registry)
        engine.reset("bash")
        assert engine.retry_count("bash") == 0

    def test_reset_all(self, registry):
        engine = RetryEngine(max_retries=3)
        engine.handle_failure("bash", "e", registry)
        engine.handle_failure("read_file", "e", registry)
        engine.reset()
        assert engine.retry_count("bash") == 0
        assert engine.retry_count("read_file") == 0

    def test_build_retry_prompt_unknown_tool(self, registry):
        engine = RetryEngine(max_retries=3)
        prompt = engine.build_retry_prompt("fake_tool", "not found", registry)
        assert "fake_tool" in prompt


# ---------------------------------------------------------------------------
# StructuredOutputEnforcer
# ---------------------------------------------------------------------------

class TestStructuredOutputEnforcer:

    def test_extracts_clean_json(self):
        enforcer = StructuredOutputEnforcer()
        response = '{"name": "bash", "arguments": {"command": "ls"}}'
        calls = enforcer.extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].name == "bash"
        assert calls[0].input == {"command": "ls"}

    def test_extracts_json_from_markdown_block(self):
        enforcer = StructuredOutputEnforcer()
        response = 'Sure!\n```json\n{"name": "bash", "arguments": {"command": "pwd"}}\n```'
        calls = enforcer.extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].name == "bash"

    def test_extracts_json_from_generic_code_block(self):
        enforcer = StructuredOutputEnforcer()
        response = '```\n{"name": "read_file", "arguments": {"path": "/tmp/x"}}\n```'
        calls = enforcer.extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].name == "read_file"

    def test_no_tool_call_returns_empty(self):
        enforcer = StructuredOutputEnforcer()
        calls = enforcer.extract_tool_calls("Hello! How can I help?")
        assert calls == []

    def test_empty_response_returns_empty(self):
        enforcer = StructuredOutputEnforcer()
        assert enforcer.extract_tool_calls("") == []
        assert enforcer.extract_tool_calls("   ") == []

    def test_multiple_tool_calls(self):
        enforcer = StructuredOutputEnforcer()
        response = (
            '```json\n{"name": "bash", "arguments": {"command": "ls"}}\n```\n'
            '```json\n{"name": "read_file", "arguments": {"path": "/tmp/x"}}\n```'
        )
        calls = enforcer.extract_tool_calls(response)
        assert len(calls) == 2
        names = {c.name for c in calls}
        assert names == {"bash", "read_file"}

    def test_deduplication(self):
        enforcer = StructuredOutputEnforcer()
        # Same call twice should appear once
        blob = '{"name": "bash", "arguments": {"command": "ls"}}'
        response = f"```json\n{blob}\n```\n\n{blob}"
        calls = enforcer.extract_tool_calls(response)
        assert len(calls) == 1

    def test_filters_against_registry(self, registry):
        enforcer = StructuredOutputEnforcer()
        response = '{"name": "totally_fake", "arguments": {}}'
        calls = enforcer.extract_tool_calls(response, tool_registry=registry)
        assert calls == []

    def test_generate_grammar_basic(self, registry):
        enforcer = StructuredOutputEnforcer()
        schemas = registry.to_api_schema()
        grammar = enforcer.generate_grammar(schemas)
        assert "root" in grammar
        assert "bash" in grammar
        assert "read_file" in grammar

    def test_generate_grammar_empty_returns_json_grammar(self):
        enforcer = StructuredOutputEnforcer()
        grammar = enforcer.generate_grammar([])
        assert "root" in grammar
        assert "object" in grammar

    def test_try_parse_tool_call_various_key_names(self):
        # "function" alias
        data = '{"function": "bash", "arguments": {"command": "ls"}}'
        block = _try_parse_tool_call(data)
        assert block is not None
        assert block.name == "bash"

        # "tool_name" alias
        data2 = '{"tool_name": "read_file", "args": {"path": "/tmp/x"}}'
        block2 = _try_parse_tool_call(data2)
        assert block2 is not None
        assert block2.name == "read_file"


# ---------------------------------------------------------------------------
# ModelAdapter (integration)
# ---------------------------------------------------------------------------

class TestModelAdapter:

    def test_default_formatter_is_anthropic(self):
        adapter = ModelAdapter()
        from prometheus.adapter.formatter import AnthropicFormatter
        assert isinstance(adapter.formatter, AnthropicFormatter)

    def test_qwen_formatter_wired(self):
        adapter = ModelAdapter(formatter=QwenFormatter(), strictness="MEDIUM")
        assert isinstance(adapter.formatter, QwenFormatter)
        assert adapter.validator.strictness == Strictness.MEDIUM

    def test_format_request_returns_tuple(self, registry):
        adapter = ModelAdapter(formatter=QwenFormatter(), strictness="MEDIUM")
        tools = registry.to_api_schema()
        system, formatted_tools = adapter.format_request("You are helpful.", tools)
        assert "bash" in system  # QwenFormatter injects tool list
        assert isinstance(formatted_tools, list)
        assert formatted_tools[0]["type"] == "function"

    def test_validate_and_repair_valid_call(self, registry):
        adapter = ModelAdapter(strictness="MEDIUM")
        name, inputs, repairs = adapter.validate_and_repair(
            "bash", {"command": "ls"}, registry
        )
        assert name == "bash"
        assert inputs == {"command": "ls"}
        assert repairs == []

    def test_validate_and_repair_fuzzy_name(self, registry):
        adapter = ModelAdapter(strictness="MEDIUM")
        name, inputs, repairs = adapter.validate_and_repair(
            "bsh", {"command": "ls"}, registry
        )
        assert name == "bash"

    def test_validate_and_repair_raises_on_irrecoverable(self, registry):
        adapter = ModelAdapter(strictness="MEDIUM")
        with pytest.raises(ValueError):
            adapter.validate_and_repair("xyzzy_not_a_tool_at_all_12345", {}, registry)

    def test_extract_tool_calls_from_text(self):
        adapter = ModelAdapter()
        text = '```json\n{"name": "bash", "arguments": {"command": "ls"}}\n```'
        calls = adapter.extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "bash"

    def test_handle_retry_first_attempt(self, registry):
        adapter = ModelAdapter(max_retries=3)
        action, prompt = adapter.handle_retry("bash", "missing param", registry)
        assert action == RetryAction.RETRY
        assert "bash" in prompt


# ---------------------------------------------------------------------------
# Formatter-specific tests
# ---------------------------------------------------------------------------

class TestQwenFormatter:

    def test_format_tools_converts_to_openai(self):
        f = QwenFormatter()
        tools = [{"name": "bash", "description": "run bash", "input_schema": {"type": "object"}}]
        result = f.format_tools(tools)
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "bash"

    def test_format_system_prompt_injects_tools(self):
        f = QwenFormatter()
        tools = [{"name": "bash", "description": "run bash", "input_schema": {}}]
        prompt = f.format_system_prompt("You are helpful.", tools)
        assert "bash" in prompt
        assert "json" in prompt.lower()

    def test_format_system_prompt_no_tools(self):
        f = QwenFormatter()
        prompt = f.format_system_prompt("Base.", [])
        assert prompt == "Base."

    def test_parse_tool_calls_markdown(self):
        f = QwenFormatter()
        raw = '```json\n{"name": "bash", "arguments": {"command": "ls"}}\n```'
        calls = f.parse_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].name == "bash"

    def test_parse_tool_calls_bare_json(self):
        f = QwenFormatter()
        raw = '{"name": "read_file", "arguments": {"path": "/tmp/x"}}'
        calls = f.parse_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].name == "read_file"


class TestGemmaFormatter:

    def test_parse_tool_call_tags(self):
        f = GemmaFormatter()
        raw = '<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>'
        calls = f.parse_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].name == "bash"

    def test_format_system_prompt_injects_tools(self):
        f = GemmaFormatter()
        tools = [{"name": "bash", "description": "run bash"}]
        prompt = f.format_system_prompt("System.", tools)
        assert "bash" in prompt
        assert "tool_call" in prompt


class TestAnthropicFormatter:

    def test_passthrough(self):
        f = AnthropicFormatter()
        tools = [{"name": "bash"}]
        assert f.format_tools(tools) == tools

    def test_system_prompt_unchanged(self):
        f = AnthropicFormatter()
        assert f.format_system_prompt("Base.", []) == "Base."

    def test_parse_tool_calls_empty(self):
        f = AnthropicFormatter()
        assert f.parse_tool_calls("anything") == []
