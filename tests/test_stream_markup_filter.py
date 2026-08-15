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
    strip_tool_call_markup,
)
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
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

    def test_strip_tool_call_markup_oneshot_matches_stream(self):
        text = f"Before {MARKUP} after"
        assert strip_tool_call_markup(text) == "Before  after"
        assert strip_tool_call_markup(MARKUP) == ""
        assert strip_tool_call_markup("no tags here") == "no tags here"
        assert strip_tool_call_markup("") == ""
        # Unterminated open is dropped (same as flush).
        assert strip_tool_call_markup('mid <tool_call>{"name": "x"') == "mid "


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


def _collect_final_text(ctx: LoopContext) -> str:
    """What Telegram/Slack/Discord actually deliver: AssistantTurnComplete.message.text."""
    messages = [ConversationMessage.from_user_text("list sessions then say DONE")]

    async def _run() -> str:
        last = ""
        try:
            async for event, _usage in run_loop(ctx, messages):
                if type(event).__name__ == "AssistantTurnComplete":
                    last = event.message.text
        except RuntimeError as exc:
            # Dual-emit path may hit max_turns after the first complete yield
            # (tools execute, loop wants another round). First complete is enough.
            if "maximum turn limit" not in str(exc) or not last:
                raise
        return last

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


class _DualEmitProvider(ModelProvider):
    """Structured tool_calls AND leftover <tool_call> markup in content.

    Mirrors the dual-emit path where extract_tool_calls is skipped because
    tool_uses is already non-empty, so residual tags used to survive into
    result.text (Telegram bubble).
    """

    async def stream_message(self, request):  # noqa: ANN001
        yield ApiTextDeltaEvent(text=f"Working {MARKUP} done")
        msg = ConversationMessage(
            role="assistant",
            content=[
                TextBlock(text=f"Working {MARKUP} done"),
                ToolUseBlock(
                    id="toolu_test",
                    name="sessions_list",
                    input={},
                ),
            ],
        )
        yield ApiMessageCompleteEvent(
            message=msg,
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="tool_use",
        )


class TestRunLoopFinalTextHygiene:
    def test_local_tier_final_text_strips_markup(self):
        # extract returns [] so text would otherwise keep the tags; final-text
        # hygiene must still scrub them for result.text consumers.
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
        final = _collect_final_text(ctx)
        assert TOOL_CALL_OPEN not in final
        assert TOOL_CALL_CLOSE not in final
        assert final == "Before  after"

    def test_dual_emit_final_text_strips_leftover_markup(self):
        # extract is skipped because tool_uses is already non-empty; residual
        # tags in the TextBlock must still be scrubbed before commit.
        adapter = SimpleNamespace(
            tier="light",
            extract_tool_calls=lambda text, reg=None: [],
            # Tool path may still call these if the loop proceeds past commit;
            # return identity so we don't AttributeError before the yield.
            validate_and_repair=lambda name, inp, reg: (name, inp, []),
        )
        ctx = LoopContext(
            provider=_DualEmitProvider(),
            model="test",
            system_prompt="- Model: test (provider: test)",
            max_tokens=256,
            adapter=adapter,
            tool_registry=_FakeRegistry(),
            max_turns=1,
        )
        final = _collect_final_text(ctx)
        assert TOOL_CALL_OPEN not in final
        assert TOOL_CALL_CLOSE not in final
        assert "Working" in final
        assert "done" in final

    def test_cloud_final_text_keeps_quoted_markup(self):
        ctx = LoopContext(
            provider=_MarkupStreamProvider(),
            model="test",
            system_prompt="- Model: test (provider: test)",
            max_tokens=256,
        )
        assert _collect_final_text(ctx) == _FINAL_TEXT


class _FakeRegistry:
    """Minimal registry so a ToolUseBlock can round-trip without hanging the loop."""

    def get(self, name):  # noqa: ANN001
        return None

    def list_schemas(self):
        return []

    def to_api_schema(self):
        return []
