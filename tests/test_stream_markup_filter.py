"""Visible-stream hygiene: <tool_call> grammar markup must not reach users.

Local tiers (light/full) make the model emit tool calls as inline
``<tool_call>{…}</tool_call>`` markup. The dispatch path parses it from the
COMPLETE turn text, but the token stream is forwarded to every gateway — so
the tags rendered verbatim in Beacon web/desktop and Telegram chat bubbles
(observed live 2026-06-10, session desktop:smoke). ToolCallMarkupFilter
strips the spans from what users SEE while the parser keeps the raw text.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from prometheus.adapter.formatter import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    ToolCallMarkupFilter,
    _partial_tag_tail,
)
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiTextDeltaEvent,
    ModelProvider,
)

MARKUP = '<tool_call>{"name": "sessions_list", "arguments": {}}</tool_call>'


def _run_filter(chunks: list[str]) -> str:
    f = ToolCallMarkupFilter()
    out = [f.feed(c) for c in chunks]
    out.append(f.flush())
    return "".join(out)


class TestToolCallMarkupFilter:
    def test_passthrough_without_markup(self):
        assert _run_filter(["hello ", "world"]) == "hello world"

    def test_strips_a_whole_tag_in_one_delta(self):
        assert _run_filter([f"before {MARKUP} after"]) == "before  after"

    def test_strips_a_tag_split_across_deltas(self):
        chunks = ["before <to", 'ol_call>{"name": "x"', "}</tool", "_call> after"]
        assert _run_filter(chunks) == "before  after"

    def test_strips_char_by_char_streaming(self):
        text = f"a {MARKUP} b {MARKUP} c"
        assert _run_filter(list(text)) == "a  b  c"

    def test_strips_multiple_tags_in_one_delta(self):
        assert _run_filter([f"{MARKUP}{MARKUP}DONE"]) == "DONE"

    def test_lone_angle_brackets_pass_through(self):
        assert _run_filter(["1 < 2 and 3 > 2"]) == "1 < 2 and 3 > 2"

    def test_lookalike_prefix_is_released_on_flush(self):
        # '<tool_cal' never completes into the tag — it is real text.
        assert _run_filter(["see <tool_cal"]) == "see <tool_cal"

    def test_lookalike_resolves_to_text_mid_stream(self):
        # '<tool_' followed by something that breaks the tag → emitted verbatim.
        assert _run_filter(["x <tool_", "box> y"]) == "x <tool_box> y"

    def test_unterminated_tag_is_dropped_on_flush(self):
        # An opened tag that never closes is grammar markup either way.
        assert _run_filter(['mid <tool_call>{"name": "x"']) == "mid "

    def test_close_tag_split_to_the_last_char(self):
        chunks = [f"a {TOOL_CALL_OPEN}{{}}{TOOL_CALL_CLOSE[:-1]}", TOOL_CALL_CLOSE[-1:], " b"]
        assert _run_filter(chunks) == "a  b"

    def test_partial_tag_tail_helper(self):
        assert _partial_tag_tail("abc <tool", TOOL_CALL_OPEN) == "<tool"
        assert _partial_tag_tail("abc", TOOL_CALL_OPEN) == ""
        # A full tag is not a *partial* tail.
        assert _partial_tag_tail(TOOL_CALL_OPEN, TOOL_CALL_OPEN) == ""


# ---------------------------------------------------------------------------
# run_loop integration: deltas are filtered for local tiers, untouched for cloud
# ---------------------------------------------------------------------------

_FINAL_TEXT = f"Before {MARKUP} after"
_CHUNKS = ["Before <to", 'ol_call>{"name": "sessions_list", "arg', 'uments": {}}</tool', "_call> after"]


class _MarkupStreamProvider(ModelProvider):
    """Streams a gemma-style turn: prose + tool-call markup split across deltas."""

    async def stream_message(self, request):  # noqa: ANN001
        for chunk in _CHUNKS:
            yield ApiTextDeltaEvent(text=chunk)
        msg = ConversationMessage(role="assistant", content=[TextBlock(text=_FINAL_TEXT)])
        yield ApiMessageCompleteEvent(
            message=msg, usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


def _collect_visible_text(ctx: LoopContext) -> str:
    messages = [ConversationMessage.from_user_text("list sessions then say DONE")]

    async def _run() -> str:
        parts: list[str] = []
        async for event, _usage in run_loop(ctx, messages):
            if type(event).__name__ == "AssistantTextDelta":
                parts.append(event.text)
        return "".join(parts)

    return asyncio.run(_run())


class TestRunLoopStreamHygiene:
    def test_local_tier_deltas_carry_no_grammar_markup(self):
        ctx = LoopContext(
            provider=_MarkupStreamProvider(),
            model="test",
            system_prompt="- Model: test (provider: test)",
            max_tokens=256,
            adapter=SimpleNamespace(
                tier="full",
                extract_tool_calls=lambda text, reg=None: [],
            ),
        )
        visible = _collect_visible_text(ctx)
        assert TOOL_CALL_OPEN not in visible
        assert TOOL_CALL_CLOSE not in visible
        assert visible == "Before  after"

    def test_cloud_path_streams_verbatim(self):
        # No adapter (tier off / cloud): quoted markup in prose must NOT be eaten.
        ctx = LoopContext(
            provider=_MarkupStreamProvider(),
            model="test",
            system_prompt="- Model: test (provider: test)",
            max_tokens=256,
        )
        assert _collect_visible_text(ctx) == _FINAL_TEXT
