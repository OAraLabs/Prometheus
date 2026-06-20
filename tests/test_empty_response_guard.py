"""Empty-response guard (agent_loop).

A 0-token / empty assistant reply — no text AND no tool calls — must never enter
the message list. If it does, the conversation history carries it and llama.cpp
400s the NEXT request ("Assistant message must contain either 'content' or
'tool_calls'"), wedging the session. The guard retries the model call once; if
it's still empty, it surfaces a VALID error turn instead of committing the empty
one. (Incident 2026-06-20, session telegram:8139235390.)
"""

from __future__ import annotations

import asyncio

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiTextDeltaEvent,
    ModelProvider,
)


def _empty_msg() -> ConversationMessage:
    # No text, no tool calls — exactly what llama.cpp rejects.
    return ConversationMessage(role="assistant", content=[])


def _text_msg(t: str) -> ConversationMessage:
    return ConversationMessage(role="assistant", content=[TextBlock(text=t)])


def _has_empty_assistant(messages) -> bool:
    """The serialization-invalid condition: an assistant turn with no text and
    no tool calls (what would 400 the next request)."""
    return any(
        m.role == "assistant" and not m.text.strip() and not m.tool_uses
        for m in messages
    )


class _EmptyThenRecover(ModelProvider):
    """First call empty, second call real — exercises retry-once → recovery."""

    def __init__(self) -> None:
        self.calls = 0
        self.system_prompts: list[str] = []

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        self.system_prompts.append(request.system_prompt or "")
        if self.calls == 1:
            yield ApiMessageCompleteEvent(
                message=_empty_msg(),
                usage=UsageSnapshot(input_tokens=1, output_tokens=0),
                stop_reason="stop",
            )
        else:
            yield ApiTextDeltaEvent(text="recovered")
            yield ApiMessageCompleteEvent(
                message=_text_msg("recovered"),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                stop_reason="stop",
            )


class _AlwaysEmpty(ModelProvider):
    """Every call empty — exercises retry-once → surface."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        yield ApiMessageCompleteEvent(
            message=_empty_msg(),
            usage=UsageSnapshot(input_tokens=1, output_tokens=0),
            stop_reason="stop",
        )


def _drive(provider) -> list:
    ctx = LoopContext(provider=provider, model="test", system_prompt="", max_tokens=256)
    messages = [ConversationMessage.from_user_text("hi")]

    async def _run():
        async for _ in run_loop(ctx, messages):
            pass

    asyncio.run(_run())
    return messages


def test_empty_reply_retried_once_then_recovers():
    p = _EmptyThenRecover()
    messages = _drive(p)
    assert p.calls == 2, "should retry the model call exactly once"
    assert not _has_empty_assistant(messages), "no empty assistant turn may enter the history"
    assert any(m.role == "assistant" and m.text.strip() == "recovered" for m in messages)


def test_retry_nudge_is_request_only_never_in_history():
    """The retry nudge rides the per-call request (system prompt) ONLY — it is
    never appended to messages, so it can't leak into session history / LCM."""
    p = _EmptyThenRecover()
    messages = _drive(p)
    assert p.calls == 2

    # The nudge rode the RETRY request's system prompt...
    assert "RETRY" in p.system_prompts[1] and "empty" in p.system_prompts[1].lower()
    # ...and NOT the first call's.
    assert "RETRY" not in p.system_prompts[0]
    # ...and it is absent from the message history (request-only — no leak).
    assert not any("RETRY" in m.text for m in messages)
    assert not any("previous response was empty" in m.text.lower() for m in messages)


def test_empty_reply_twice_surfaces_a_valid_turn():
    p = _AlwaysEmpty()
    messages = _drive(p)
    assert p.calls == 2, "retry once, then surface — no infinite loop"
    assert not _has_empty_assistant(messages), "no empty assistant turn may enter the history"
    # The surfaced turn is a valid assistant message (has content) -> won't 400.
    assistants = [m for m in messages if m.role == "assistant"]
    assert assistants, "a turn must be surfaced"
    assert assistants[-1].text.strip(), "surfaced assistant turn must carry content"


# ---------------------------------------------------------------------------
# #65 — extend the invariant to the MALFORMED-empty path
#
# #64 deliberately excludes the malformed case from the empty-response guard
# (`and not dropped_malformed`), routing it to the structured-feedback path.
# But the empty turn was still committed at the model-reply append point, so a
# malformed-empty reply (no text, no SURVIVING tool calls, dropped_malformed>0)
# entered history as a content-less assistant message and 400'd the next
# request — the same wedge #64 fixed elsewhere. The invariant must be total.
# ---------------------------------------------------------------------------


class _MalformedEmptyThenRecover(ModelProvider):
    """First call: an all-dropped malformed-empty turn (empty message,
    ``dropped_malformed>0``). Second call: a real recovery."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        if self.calls == 1:
            yield ApiMessageCompleteEvent(
                message=_empty_msg(),
                usage=UsageSnapshot(input_tokens=1, output_tokens=0),
                stop_reason="stop",
                dropped_malformed=3,
            )
        else:
            yield ApiTextDeltaEvent(text="recovered")
            yield ApiMessageCompleteEvent(
                message=_text_msg("recovered"),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                stop_reason="stop",
            )


def test_malformed_empty_turn_never_enters_history():
    p = _MalformedEmptyThenRecover()
    messages = _drive(p)
    assert p.calls == 2, "malformed-empty turn retried once, then recovered"
    # The core #65 invariant — the content-less assistant turn never lands.
    assert not _has_empty_assistant(messages), (
        "no empty assistant turn may enter history via the malformed path"
    )
    # The structured feedback IS committed (load-bearing — the model reads it
    # from history to self-correct).
    feedback = [
        m for m in messages
        if m.role == "user" and m.provenance == "orchestrator"
    ]
    assert len(feedback) == 1
    assert "malformed tool call" in feedback[0].text
    # And the model recovered after reading it.
    assert any(
        m.role == "assistant" and m.text.strip() == "recovered" for m in messages
    )


def test_malformed_with_residual_prose_still_commits():
    """The non-empty malformed case must be untouched: a turn that dropped
    garbage calls but produced prose is a valid, non-empty assistant turn —
    it commits and ends the turn (no synthetic feedback, no retry)."""

    class _DroppedButProse(ModelProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def stream_message(self, request):  # noqa: ANN001
            self.calls += 1
            yield ApiMessageCompleteEvent(
                message=_text_msg("here is my answer"),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                stop_reason="stop",
                dropped_malformed=2,
            )

    p = _DroppedButProse()
    messages = _drive(p)
    assert p.calls == 1, "prose turn ends normally — no retry"
    assert messages[-1].role == "assistant"
    assert messages[-1].text == "here is my answer"
    assert not any(m.provenance == "orchestrator" for m in messages), (
        "non-empty malformed turn must not trigger feedback"
    )
