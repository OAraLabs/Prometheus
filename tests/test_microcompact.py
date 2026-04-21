"""Tests for Feature 3: Tool Result MicroCompaction."""

import pytest
from unittest.mock import MagicMock

from prometheus.engine.agent_loop import _microcompact_old_results, LoopContext
from prometheus.engine.messages import ConversationMessage, ToolResultBlock, TextBlock


def _make_context(**overrides) -> LoopContext:
    ctx = MagicMock(spec=LoopContext)
    ctx.microcompact_after_turns = overrides.get("microcompact_after_turns", 3)
    ctx.microcompact_keep_chars = overrides.get("microcompact_keep_chars", 200)
    ctx.microcompact_keep_chars_no_lcm = overrides.get("microcompact_keep_chars_no_lcm", 500)
    ctx.lcm_engine = overrides.get("lcm_engine", None)
    return ctx


def _user_msg(text: str = "hi") -> ConversationMessage:
    return ConversationMessage(role="user", content=[TextBlock(text=text)])


def _tool_result_msg(content: str, tool_use_id: str = "t1") -> ConversationMessage:
    return ConversationMessage(
        role="user",
        content=[ToolResultBlock(tool_use_id=tool_use_id, content=content)],
    )


def _assistant_msg(text: str = "ok") -> ConversationMessage:
    return ConversationMessage(role="assistant", content=[TextBlock(text=text)])


class TestMicroCompaction:
    def test_no_compaction_before_threshold(self):
        ctx = _make_context(microcompact_after_turns=3)
        long_content = "x" * 1000
        msgs = [_tool_result_msg(long_content), _user_msg(), _assistant_msg()]
        _microcompact_old_results(ctx, msgs, current_turn=2)
        # Should not compact — turn < threshold
        assert msgs[0].content[0].content == long_content

    def test_compacts_old_results_after_threshold(self):
        ctx = _make_context(microcompact_after_turns=2, microcompact_keep_chars_no_lcm=100)
        long_content = "first line\n" + "x" * 2000
        msgs = [
            _tool_result_msg(long_content),
            _user_msg("turn 1"),
            _assistant_msg(),
            _user_msg("turn 2"),
            _assistant_msg(),
            _user_msg("turn 3"),
        ]
        _microcompact_old_results(ctx, msgs, current_turn=3)
        result = msgs[0].content[0].content
        assert "[microcompacted]" in result
        assert len(result) < len(long_content)

    def test_preserves_recent_results(self):
        ctx = _make_context(microcompact_after_turns=2, microcompact_keep_chars_no_lcm=100)
        long_content = "y" * 2000
        msgs = [
            _user_msg("old"),
            _assistant_msg(),
            _user_msg("recent"),
            _tool_result_msg(long_content),  # This is in the fresh window
        ]
        _microcompact_old_results(ctx, msgs, current_turn=3)
        # Recent result should NOT be compacted
        assert msgs[3].content[0].content == long_content

    def test_skips_error_results(self):
        ctx = _make_context(microcompact_after_turns=1, microcompact_keep_chars_no_lcm=50)
        msgs = [
            ConversationMessage(
                role="user",
                content=[ToolResultBlock(
                    tool_use_id="t1",
                    content="x" * 1000,
                    is_error=True,
                )],
            ),
            _user_msg("turn 1"),
            _user_msg("turn 2"),
        ]
        _microcompact_old_results(ctx, msgs, current_turn=2)
        # Error results should be preserved
        assert msgs[0].content[0].content == "x" * 1000

    def test_skips_already_pruned(self):
        ctx = _make_context(microcompact_after_turns=1, microcompact_keep_chars_no_lcm=50)
        msgs = [
            _tool_result_msg("[content pruned — context compression]"),
            _user_msg("turn 1"),
            _user_msg("turn 2"),
        ]
        _microcompact_old_results(ctx, msgs, current_turn=2)
        assert "[content pruned" in msgs[0].content[0].content

    def test_lcm_not_ingested_uses_longer_chars(self):
        lcm = MagicMock()
        lcm.is_ingested = MagicMock(return_value=False)
        ctx = _make_context(
            microcompact_after_turns=1,
            microcompact_keep_chars=50,
            microcompact_keep_chars_no_lcm=300,
            lcm_engine=lcm,
        )
        long_content = "a" * 1000
        msgs = [
            _tool_result_msg(long_content),
            _user_msg("turn 1"),
            _user_msg("turn 2"),
        ]
        _microcompact_old_results(ctx, msgs, current_turn=2)
        result = msgs[0].content[0].content
        assert "[microcompacted]" in result
        # Should keep ~300 chars (no_lcm), not 50
        assert len(result) > 200

    def test_short_results_not_compacted(self):
        ctx = _make_context(microcompact_after_turns=1, microcompact_keep_chars_no_lcm=500)
        msgs = [
            _tool_result_msg("short"),
            _user_msg("turn 1"),
            _user_msg("turn 2"),
        ]
        _microcompact_old_results(ctx, msgs, current_turn=2)
        assert msgs[0].content[0].content == "short"
