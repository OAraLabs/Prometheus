"""Microcompaction never makes a tool result longer.

The pass skipped results of ``microcompact_keep_chars`` (200) or fewer, then
kept ``microcompact_keep_chars_no_lcm`` (500) of the rest, because
``is_ingested`` never matches a tool_use_id. A result of 201-500 characters
was therefore kept WHOLE, plus a ``[microcompacted] <first line>...`` wrapper
of 21 + up to 80 characters. Live, 2026-07-31 to 2026-09-30: 458 of 1,787
microcompaction events made history bigger, every single-result event with
201-500 characters of input grew (388 of 388), and 57 of 76 in 501-600 did.
Each of those rewrites also invalidated the provider's prompt cache from that
point on, for no saving.

Now a replacement that is not shorter than the original is not made: the
original stays verbatim and is not counted.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prometheus.engine.agent_loop import LoopContext, _microcompact_old_results
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolResultBlock

WRAPPER = len("[microcompacted] ") + len("...\n")  # 21


def _ctx(telemetry=None) -> LoopContext:
    ctx = MagicMock(spec=LoopContext)
    ctx.microcompact_after_turns = 1
    ctx.microcompact_keep_chars = 200
    ctx.microcompact_keep_chars_no_lcm = 500
    ctx.lcm_engine = None
    ctx.telemetry = telemetry
    return ctx


def _old(*contents: str) -> list[ConversationMessage]:
    msgs = [
        ConversationMessage(role="user", content=[
            ToolResultBlock(tool_use_id=f"t{i}", content=c)]) for i, c in enumerate(contents)
    ]
    return msgs + [ConversationMessage(role="user", content=[TextBlock(text="turn 1")]),
                   ConversationMessage(role="user", content=[TextBlock(text="turn 2")])]


def _result(msgs, i: int = 0) -> str:
    return msgs[i].content[0].content


def _content(size: int, first_line_len: int = 40) -> str:
    first = "f" * first_line_len
    return (first + "\n" + "x" * size)[:size]


class TestNeverGrows:
    @pytest.mark.parametrize("size", [201, 250, 300, 400, 499, 500])
    def test_a_result_the_wrapper_would_lengthen_is_kept_verbatim(self, size):
        content = _content(size)
        msgs = _old(content)
        _microcompact_old_results(_ctx(), msgs, current_turn=2)
        assert _result(msgs) == content

    def test_the_cut_must_actually_save_characters(self):
        """keep 500 + an 80-char first line: the replacement is 601 characters."""
        at = 500 + WRAPPER + 80
        equal, longer = _content(at, first_line_len=100), _content(at + 1, first_line_len=100)
        msgs = _old(equal, longer)
        _microcompact_old_results(_ctx(), msgs, current_turn=2)
        assert _result(msgs, 0) == equal, "an equal-length rewrite saves nothing"
        assert _result(msgs, 1).startswith("[microcompacted] ")
        assert len(_result(msgs, 1)) == at

    def test_a_long_result_is_still_compacted(self):
        content = _content(2_000)
        msgs = _old(content)
        _microcompact_old_results(_ctx(), msgs, current_turn=2)
        assert _result(msgs).startswith("[microcompacted] ")
        assert len(_result(msgs)) < len(content)

    @pytest.mark.parametrize("first_line_len", [1, 10, 40, 80, 200])
    def test_no_rewrite_is_ever_longer(self, first_line_len):
        sizes = list(range(150, 1_300, 7))
        contents = [_content(s, first_line_len) for s in sizes]
        msgs = _old(*contents)
        _microcompact_old_results(_ctx(), msgs, current_turn=2)
        for i, before in enumerate(contents):
            after = _result(msgs, i)
            assert after == before or len(after) < len(before), (len(before), len(after))


class TestTelemetry:
    def test_a_pass_that_would_only_grow_records_nothing(self):
        tel = MagicMock()
        msgs = _old(_content(300), _content(450))
        _microcompact_old_results(_ctx(tel), msgs, current_turn=2)
        tel.record_run.assert_not_called()

    def test_a_mixed_pass_counts_only_what_it_rewrote(self):
        tel = MagicMock()
        short, long = _content(300), _content(2_000)
        msgs = _old(short, long)
        _microcompact_old_results(_ctx(tel), msgs, current_turn=2)
        tel.record_run.assert_called_once()
        summary = tel.record_run.call_args.kwargs["summary"]
        assert summary["results_compacted"] == 1
        assert summary["chars_before"] == len(long)
        assert summary["chars_after"] == len(_result(msgs, 1))
        assert summary["chars_dropped"] > 0
