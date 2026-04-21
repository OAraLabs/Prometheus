"""Tests for Sprint 4: TokenBudget, ToolResultTruncator, ContextCompressor, DynamicToolLoader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prometheus.context import (
    ContextCompressor,
    DynamicToolLoader,
    TokenBudget,
    ToolResultTruncator,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_four_chars_is_one_token(self):
        assert estimate_tokens("abcd") == 1

    def test_longer_text(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100

    def test_never_negative(self):
        assert estimate_tokens("x") >= 0


# ---------------------------------------------------------------------------
# TokenBudget — sprint acceptance test
# ---------------------------------------------------------------------------


class TestTokenBudgetAcceptance:
    """Mirror the sprint acceptance test exactly."""

    def test_sprint_acceptance(self):
        budget = TokenBudget(effective_limit=24000)
        budget.add("system", "You are helpful." * 100)
        used = budget.used
        headroom = budget.headroom()
        approaching = budget.is_approaching_limit()
        assert used >= 0
        assert headroom >= 0
        assert isinstance(approaching, bool)


class TestTokenBudget:
    def test_starts_at_zero(self):
        b = TokenBudget(effective_limit=1000)
        assert b.used == 0

    def test_add_accumulates(self):
        b = TokenBudget(effective_limit=1000)
        b.add("system", "a" * 40)   # 10 tokens
        b.add("messages", "b" * 80) # 20 tokens
        assert b.used == 30

    def test_headroom_subtracts_reserved_output(self):
        b = TokenBudget(effective_limit=1000, reserved_output=200)
        b.add("x", "a" * 400)  # 100 tokens used
        # available = 1000 - 200 = 800; headroom = 800 - 100 = 700
        assert b.headroom() == 700

    def test_headroom_never_negative(self):
        b = TokenBudget(effective_limit=100, reserved_output=0)
        b.add("x", "a" * 10000)  # way over limit
        assert b.headroom() == 0

    def test_is_approaching_limit_false_below_threshold(self):
        b = TokenBudget(effective_limit=1000, reserved_output=0)
        b.add("x", "a" * 100)  # 25 tokens — 2.5% of 1000
        assert b.is_approaching_limit(threshold=0.75) is False

    def test_is_approaching_limit_true_above_threshold(self):
        b = TokenBudget(effective_limit=1000, reserved_output=0)
        b.add("x", "a" * 3200)  # 800 tokens — 80% of 1000
        assert b.is_approaching_limit(threshold=0.75) is True

    def test_reset_clears_usage(self):
        b = TokenBudget(effective_limit=1000)
        b.add("system", "a" * 400)
        b.reset()
        assert b.used == 0

    def test_usage_by_category(self):
        b = TokenBudget(effective_limit=1000)
        b.add("system", "a" * 40)
        b.add("messages", "b" * 80)
        cats = b.usage_by_category()
        assert cats["system"] == 10
        assert cats["messages"] == 20

    def test_from_config_loads_yaml(self, tmp_path):
        config = tmp_path / "prometheus.yaml"
        config.write_text(
            "context:\n"
            "  effective_limit: 8000\n"
            "  reserved_output: 500\n"
            "  model_overrides:\n"
            "    testmodel:\n"
            "      effective_limit: 16000\n"
        )
        b = TokenBudget.from_config(config_path=str(config))
        assert b.effective_limit == 8000
        assert b.reserved_output == 500

    def test_from_config_model_override(self, tmp_path):
        config = tmp_path / "prometheus.yaml"
        config.write_text(
            "context:\n"
            "  effective_limit: 8000\n"
            "  model_overrides:\n"
            "    bigmodel:\n"
            "      effective_limit: 32000\n"
        )
        b = TokenBudget.from_config(model="bigmodel", config_path=str(config))
        assert b.effective_limit == 32000

    def test_from_config_graceful_on_missing_file(self):
        b = TokenBudget.from_config(config_path="/nonexistent/prometheus.yaml")
        assert b.effective_limit == 24000  # default


# ---------------------------------------------------------------------------
# ToolResultTruncator
# ---------------------------------------------------------------------------


class TestToolResultTruncator:
    def test_short_output_unchanged(self):
        t = ToolResultTruncator(max_tokens=4000)
        text = "hello world"
        assert t.truncate("bash", text) == text

    def test_bash_keeps_last_100_lines(self):
        t = ToolResultTruncator(max_tokens=10)
        lines = [f"line {i}" for i in range(200)]
        output = "\n".join(lines)
        result = t.truncate("bash", output)
        result_lines = result.splitlines()
        # Should end with the last 100 lines (lines 100–199 in a 200-line output)
        assert "line 199" in result
        assert "line 100" in result
        assert "truncated" in result

    def test_file_read_head_and_tail(self):
        t = ToolResultTruncator(max_tokens=10)
        lines = [f"L{i}" for i in range(200)]
        output = "\n".join(lines)
        result = t.truncate("read_file", output)
        assert "L0" in result        # head preserved
        assert "L199" in result      # tail preserved
        assert "truncated" in result

    def test_grep_keeps_top_20(self):
        t = ToolResultTruncator(max_tokens=10)
        lines = [f"match {i}" for i in range(50)]
        output = "\n".join(lines)
        result = t.truncate("grep", output)
        result_lines = [l for l in result.splitlines() if l.strip() and "truncated" not in l]
        assert len(result_lines) <= 20
        assert "truncated" in result

    def test_default_truncation(self):
        t = ToolResultTruncator(max_tokens=10)
        # 10 tokens * 4 chars = 40 char limit
        long_text = "x" * 200
        result = t.truncate("unknown_tool", long_text)
        assert "truncated" in result
        assert len(result) < len(long_text) + 100

    def test_callable_interface(self):
        t = ToolResultTruncator(max_tokens=4000)
        result = t("bash", "hello")
        assert result == "hello"

    def test_from_config(self, tmp_path):
        config = tmp_path / "prometheus.yaml"
        config.write_text("context:\n  tool_result_max: 2000\n")
        t = ToolResultTruncator.from_config(str(config))
        assert t._max_tokens == 2000


# ---------------------------------------------------------------------------
# ContextCompressor
# ---------------------------------------------------------------------------


class TestContextCompressor:
    def _make_messages(self, n_turns: int):
        """Build a simple conversation: alternating user/assistant messages with tool results."""
        from prometheus.engine.messages import (
            ConversationMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
        )

        messages = []
        for i in range(n_turns):
            # User turn with a tool result
            messages.append(
                ConversationMessage(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_use_id=f"id_{i}",
                            content=f"output of turn {i}: " + "x" * 200,
                            is_error=False,
                        )
                    ],
                )
            )
            # Assistant turn
            messages.append(
                ConversationMessage(
                    role="assistant",
                    content=[TextBlock(type="text", text=f"assistant response {i}")],
                )
            )
        return messages

    def test_no_compression_when_under_limit(self):
        budget = TokenBudget(effective_limit=100000, reserved_output=0)
        compressor = ContextCompressor(budget, fresh_tail_count=32)
        messages = self._make_messages(5)
        result = compressor.maybe_compress(messages)
        assert result == messages  # unchanged

    def test_prunes_old_tool_results_when_over_threshold(self):
        from prometheus.engine.messages import ToolResultBlock

        # Small budget so it's immediately "approaching limit"
        budget = TokenBudget(effective_limit=10, reserved_output=0)
        budget.add("x", "a" * 40)  # 10 tokens — 100% → approaching

        compressor = ContextCompressor(budget, fresh_tail_count=1)
        messages = self._make_messages(5)
        result = compressor.maybe_compress(messages)

        # Old user messages should have pruned content
        pruned_any = False
        for msg in result:
            if msg.role == "user":
                for block in msg.content:
                    if isinstance(block, ToolResultBlock) and "pruned" in block.content:
                        pruned_any = True
        assert pruned_any

    def test_fresh_tail_protected(self):
        from prometheus.engine.messages import ToolResultBlock

        budget = TokenBudget(effective_limit=10, reserved_output=0)
        budget.add("x", "a" * 40)  # over threshold

        n_turns = 5
        compressor = ContextCompressor(budget, fresh_tail_count=n_turns)
        messages = self._make_messages(n_turns)
        result = compressor.maybe_compress(messages)

        # With fresh_tail_count = n_turns, ALL user messages are protected → no pruning
        for msg in result:
            if msg.role == "user":
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        assert "pruned" not in block.content

    def test_from_config(self, tmp_path):
        config = tmp_path / "prometheus.yaml"
        config.write_text("context:\n  fresh_tail_count: 16\n")
        budget = TokenBudget(effective_limit=1000)
        compressor = ContextCompressor.from_config(budget, config_path=str(config))
        assert compressor._fresh_tail_count == 16


# ---------------------------------------------------------------------------
# DynamicToolLoader
# ---------------------------------------------------------------------------


class TestDynamicToolLoader:
    def _make_registry(self):
        from prometheus.tools.base import ToolRegistry
        from prometheus.tools.builtin import (
            BashTool,
            FileReadTool,
            FileWriteTool,
            GrepTool,
            GlobTool,
            FileEditTool,
        )

        registry = ToolRegistry()
        for tool in [
            BashTool(),
            FileReadTool(),
            FileWriteTool(),
            GrepTool(),
            GlobTool(),
            FileEditTool(),
        ]:
            registry.register(tool)
        return registry

    def test_core_tools_always_included(self):
        registry = self._make_registry()
        loader = DynamicToolLoader(registry)
        schemas = loader.active_schemas("do something")
        names = {s["name"] for s in schemas}
        assert "bash" in names
        assert "read_file" in names
        assert "write_file" in names

    def test_grep_added_by_keyword(self):
        registry = self._make_registry()
        loader = DynamicToolLoader(registry)
        schemas = loader.active_schemas("grep for errors in the logs")
        names = {s["name"] for s in schemas}
        assert "grep" in names

    def test_edit_added_by_keyword(self):
        registry = self._make_registry()
        loader = DynamicToolLoader(registry)
        schemas = loader.active_schemas("edit the config file")
        names = {s["name"] for s in schemas}
        assert "edit_file" in names

    def test_no_task_returns_all(self):
        registry = self._make_registry()
        loader = DynamicToolLoader(registry)
        all_schemas = loader.active_schemas(None)
        assert len(all_schemas) == len(registry.list_tools())

    def test_on_demand_returns_schema(self):
        registry = self._make_registry()
        loader = DynamicToolLoader(registry)
        schema = loader.on_demand("grep")
        assert schema is not None
        assert schema["name"] == "grep"

    def test_on_demand_returns_none_for_unknown(self):
        registry = self._make_registry()
        loader = DynamicToolLoader(registry)
        assert loader.on_demand("nonexistent_tool") is None

    def test_all_schemas_returns_everything(self):
        registry = self._make_registry()
        loader = DynamicToolLoader(registry)
        all_schemas = loader.all_schemas()
        assert len(all_schemas) == len(registry.list_tools())


# ---------------------------------------------------------------------------
# Integration: TokenBudget + ToolResultTruncator (triggers truncation)
# ---------------------------------------------------------------------------


class TestTruncationIntegration:
    """Integration test: run a task that triggers truncation."""

    def test_truncation_triggered_on_large_output(self):
        budget = TokenBudget(effective_limit=1000, reserved_output=0)
        truncator = ToolResultTruncator(max_tokens=100)

        # Simulate a large bash output
        large_output = "\n".join(f"line {i}: some output content here" for i in range(300))

        # Add to budget as-is first
        budget.add("tool_results", large_output)
        assert budget.is_approaching_limit(threshold=0.01)  # definitely approaching

        # Truncate
        truncated = truncator.truncate("bash", large_output)
        assert len(truncated) < len(large_output)
        assert "truncated" in truncated

        # Budget after truncation would be smaller
        fresh_budget = TokenBudget(effective_limit=1000, reserved_output=0)
        fresh_budget.add("tool_results", truncated)
        assert fresh_budget.used < budget.used
