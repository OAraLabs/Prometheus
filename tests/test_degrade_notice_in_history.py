"""A degrade must survive a re-read of the conversation, not just the live stream.

The fallback wrapper yields the notice as a text delta, which the UI renders as it arrives.
Stored history is rebuilt from `ApiMessageCompleteEvent`, which that delta is not part of — so
before this, re-opening a conversation showed the degraded answer with no trace of why it was
degraded. Loud once, silent forever after, which is the failure the sprint exists to prevent.

The injection point is the whole subtlety. `raw_model_output_this_turn` is captured earlier as
"what we would want to train a local model to emit", and `extract_tool_calls` parses the same
text — so injecting before either would put our outage banner into the golden-trace corpus and
hand the tool parser a prefix the model never wrote.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.fallback import FallbackTarget
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.stream_events import AssistantTurnComplete
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiStreamEvent,
    ModelProvider,
)

MODEL_TEXT = "The answer is 4."


class _ExpiredKey(ModelProvider):
    """401 on every call — the Token-Plan-expiry shape."""

    def __init__(self) -> None:
        self._suppress_thinking = True
        self.calls = 0

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        self.calls += 1

        class R:
            status_code = 401

        class AuthError(Exception):
            response = R()

        raise AuthError("expired credential")
        yield  # pragma: no cover — makes this an async generator


class _WorkingLocal(ModelProvider):
    def __init__(self) -> None:
        self._suppress_thinking = True
        self.requests: list[ApiMessageRequest] = []

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        self.requests.append(request)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text=MODEL_TEXT)]),
            usage=UsageSnapshot(input_tokens=10, output_tokens=4),
            stop_reason="stop",
        )


def _run():
    primary, local = _ExpiredKey(), _WorkingLocal()
    context = LoopContext(
        provider=primary,
        model="qwen3.8-max",
        system_prompt="- Model: qwen3.8-max (provider: qwen)",
        max_tokens=512,
        session_id="s1",
        fallback=FallbackTarget(
            model="Qwen3.8-27B", provider_name="llama_cpp",
            provider=local, is_local_backend=True,
        ),
    )
    turns = []

    async def go():
        async for event, _usage in run_loop(context, [ConversationMessage.from_user_text("2+2?")]):
            if isinstance(event, AssistantTurnComplete):
                turns.append(event)

    asyncio.run(go())
    return primary, local, context, turns


def test_the_degraded_turn_reaches_history_carrying_its_reason():
    primary, local, _ctx, turns = _run()
    assert primary.calls == 1 and len(local.requests) == 1, "the fallback served the turn"
    assert turns, "no assistant turn completed"

    text = turns[0].message.text
    assert MODEL_TEXT in text, "the fallback's answer is present"
    assert "Qwen3.8-27B" in text, "and history says which model produced it"
    assert "qwen3.8-max" in text, "and which one failed"


def test_the_notice_is_a_SEPARATE_block_ahead_of_the_model_s_own_output():
    """Additive, never a rewrite of what the model said. The model's block must survive
    byte-identical — that is what keeps the stored turn an honest record of its output."""
    _p, _l, _ctx, turns = _run()
    blocks = turns[0].message.content

    assert len(blocks) == 2, f"expected notice + model output, got {len(blocks)} blocks"
    assert blocks[0].text.startswith("⚠"), "the notice leads"
    assert blocks[1].text == MODEL_TEXT, "the model's own block is untouched"


def test_the_request_sent_to_the_fallback_names_the_FALLBACK_model():
    """Reusing the failed request would ask the local backend to serve `qwen3.8-max`."""
    _p, local, _ctx, _t = _run()
    assert local.requests[0].model == "Qwen3.8-27B"


def test_the_prompt_identity_line_follows_the_serving_model():
    """Otherwise the degraded model answers "what model is this?" as the primary."""
    _p, _l, ctx, _t = _run()
    assert "Qwen3.8-27B" in ctx.system_prompt
    # is_local_backend=True, so the "separate local backend, not you" clause must be ABSENT —
    # here the serving model IS the local backend and the clause would be false.
    assert "not you" not in ctx.system_prompt


def test_a_healthy_turn_gains_no_notice_and_stays_a_single_block():
    local = _WorkingLocal()
    context = LoopContext(
        provider=local, model="Qwen3.8-27B", system_prompt="s", max_tokens=512, session_id="s2",
    )
    turns = []

    async def go():
        async for event, _u in run_loop(context, [ConversationMessage.from_user_text("hi")]):
            if isinstance(event, AssistantTurnComplete):
                turns.append(event)

    asyncio.run(go())
    assert len(turns[0].message.content) == 1
    assert turns[0].message.text == MODEL_TEXT
    assert "⚠" not in turns[0].message.text


class _RecordingAdapter:
    """Captures the text handed to extract_tool_calls — the golden-trace/parse boundary."""

    tier = "off"  # cloud tier: no markup filter, so the text arrives unfiltered

    def __init__(self) -> None:
        self.extracted_from: list[str] = []

    def format_request(self, system_prompt, tools):
        return system_prompt, tools

    def extract_tool_calls(self, text, registry):
        self.extracted_from.append(text)
        return []


def test_extraction_sees_the_MODEL_s_text_not_our_notice():
    """The ordering guard, asserted rather than assumed.

    `raw_model_output_this_turn` is captured from the same string on the line above
    extract_tool_calls, as "what we would want to train a local model to emit". If the notice
    were injected before that point, our outage banner would enter the golden-trace corpus
    attributed to the model, and the tool parser would receive a prefix the model never wrote.
    """
    adapter = _RecordingAdapter()
    local = _WorkingLocal()
    context = LoopContext(
        provider=_ExpiredKey(),
        model="qwen3.8-max",
        system_prompt="- Model: qwen3.8-max (provider: qwen)",
        max_tokens=512,
        session_id="s3",
        adapter=adapter,
        fallback=FallbackTarget(
            model="Qwen3.8-27B", provider_name="llama_cpp",
            provider=local, is_local_backend=True,
        ),
    )
    turns = []

    async def go():
        async for event, _u in run_loop(context, [ConversationMessage.from_user_text("2+2?")]):
            if isinstance(event, AssistantTurnComplete):
                turns.append(event)

    asyncio.run(go())

    assert adapter.extracted_from, "extract_tool_calls never ran — the guard is untested"
    seen = adapter.extracted_from[0]
    assert seen == MODEL_TEXT, f"extraction saw text the model did not write: {seen!r}"
    assert "⚠" not in seen
    assert "unavailable" not in seen
    # ...and the notice still reached history, so this is the ordering being right, not the
    # injection being absent.
    assert "⚠" in turns[0].message.text
