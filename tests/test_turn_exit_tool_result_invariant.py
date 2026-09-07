"""P0.4 — every turn exit answers the tool calls it committed.

THE INVARIANT (session.py: "a hard 400 from every provider"): once an
assistant message carrying tool_use blocks is in history, the NEXT message
must be a user message with a tool_result for every one of those ids.

The iteration-cap exit and the two circuit-breaker exits used to append their
exit text with the committed calls unanswered. The turn ended cleanly
LOCALLY — and poisoned the session: the next user message serialized
assistant(tool_use) -> assistant(text) -> user(text), which Anthropic rejects
("tool_use ids were found without tool_result blocks immediately after") and
OpenAI-compat rejects ("must be followed by tool messages"). The session was
stuck until in-memory state AND the LCM tail were both cleared.

These tests drive the real run_loop through both exit families and assert the
invariant ON THE HISTORY, the same shape every provider builder enforces —
so the property is enforced rather than remembered by whichever exit path
gets edited next.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

import pytest

from prometheus.engine.agent_loop import (
    LoopContext,
    _unanswered_tool_results,
    run_loop,
)
from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ModelProvider,
)


class _ToolCaller(ModelProvider):
    """Emits the same tool call(s) forever — the turn can only end via a guard."""

    def __init__(self, calls: list[tuple[str, dict]], *, distinct_ids: bool = True):
        self._calls = calls
        self._distinct = distinct_ids
        self._n = 0
        self.requests: list[ApiMessageRequest] = []

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator:
        self.requests.append(request)
        self._n += 1
        blocks = [
            ToolUseBlock(
                id=f"t{self._n}-{i}" if self._distinct else f"t{i}",
                name=name,
                input=args,
            )
            for i, (name, args) in enumerate(self._calls)
        ]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=blocks),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="tool_calls",
        )


def _assert_tool_uses_answered(messages: list[ConversationMessage]) -> None:
    """The provider-side invariant, checked on the committed history.

    For EVERY assistant message carrying tool_use blocks, the immediately
    following message must be a user message whose tool_result blocks cover
    every tool_use id. This is exactly what Anthropic and OpenAI-compat
    enforce at serialization; checking it here means a 400 is impossible
    regardless of which exit appended what.
    """
    for i, m in enumerate(messages):
        if m.role != "assistant":
            continue
        uses = [b for b in m.content if getattr(b, "type", None) == "tool_use"]
        if not uses:
            continue
        assert i + 1 < len(messages), (
            f"assistant tool_use at index {i} is the LAST message — the "
            f"calls {[(u.id, u.name) for u in uses]} were never answered"
        )
        nxt = messages[i + 1]
        assert nxt.role == "user", (
            f"the message after a tool_use turn must be the tool_results "
            f"(role=user), got role={nxt.role!r}: "
            f"{[getattr(b, 'type', '?') for b in nxt.content]}"
        )
        answered = {
            b.tool_use_id
            for b in nxt.content
            if getattr(b, "type", None) == "tool_result"
        }
        for u in uses:
            assert u.id in answered, (
                f"tool_use {u.id} ({u.name}) has no tool_result in the next "
                f"message — answered: {sorted(answered)}. This history is a "
                f"hard 400 on every provider."
            )


def _run(provider: ModelProvider, tmp_path: Path, **ctx_kw):
    from prometheus.__main__ import create_tool_registry

    ctx = LoopContext(
        provider=provider,
        model="t",
        system_prompt="",
        max_tokens=256,
        tool_registry=create_tool_registry({}, None),
        cwd=tmp_path,
        **ctx_kw,
    )
    messages = [ConversationMessage.from_user_text("go")]
    finals: list[str] = []

    async def go():
        async for event, _ in run_loop(ctx, messages, session_id="s"):
            msg = getattr(event, "message", None)
            if event.__class__.__name__ == "AssistantTurnComplete" and msg:
                finals.append(msg.text or "")

    asyncio.run(go())
    return messages, finals


# --------------------------------------------------------------------------- #
# Exit 1: the iteration cap fires BEFORE dispatch — nothing ran.
# --------------------------------------------------------------------------- #


def test_iteration_cap_exit_answers_the_calls_it_never_dispatched(tmp_path):
    """max_tool_iterations=1, one message carrying 2 calls: cap trips on
    round 1, before _dispatch_tool_calls ever runs. The committed tool_use
    blocks must still be answered — with NOT-EXECUTED results, never a
    synthesized success."""
    provider = _ToolCaller([("no_such_tool", {}), ("other_missing_tool", {})])
    messages, finals = _run(provider, tmp_path, max_tool_iterations=1)

    _assert_tool_uses_answered(messages)

    # The exit text still reaches the user — the guard stays loud.
    assert any("iteration limit" in t.lower() for t in finals), finals

    # The synthetic answers must be errors that say the call never ran.
    results = [
        b
        for m in messages
        for b in m.content
        if getattr(b, "type", None) == "tool_result"
    ]
    assert results, "no tool_result blocks were appended at all"
    assert all(r.is_error for r in results)
    assert all("NOT EXECUTED" in r.content for r in results), (
        "an unanswered call must never be answered with a plausible result"
    )
    # Only one provider round happened — the cap fired before any dispatch.
    assert len(provider.requests) == 1


# --------------------------------------------------------------------------- #
# Exit 2/3: the circuit breaker trips AFTER dispatch — real results exist.
# --------------------------------------------------------------------------- #


def test_breaker_exit_appends_the_real_results_not_a_synthetic(tmp_path):
    """The same failing call every round trips the breaker (max_identical=3)
    on round 3. Dispatch DID run, so history must carry the REAL error
    results — which are also the evidence the diagnostic refers to."""
    provider = _ToolCaller([("no_such_tool", {"x": 1})])
    messages, finals = _run(provider, tmp_path)

    _assert_tool_uses_answered(messages)

    # The breaker (or its recovery diagnostic) still ends the turn loudly.
    assert finals, "the turn produced no terminal message"
    last = finals[-1].lower()
    assert "circuit breaker" in last or "diagnos" in last or "cannot" in last, (
        f"unexpected terminal message: {finals[-1]!r}"
    )

    # Several rounds ran before the trip; every round's results are in
    # history (the `continue` rounds always appended; the exit round is the
    # fix). Rounds 1-2 dispatch for real ("Unknown tool"); the repeat guard
    # blocks the identical failing call from round 3 ("BLOCKED") until the
    # breaker trips on those.
    result_blocks = [
        b
        for m in messages
        for b in m.content
        if getattr(b, "type", None) == "tool_result"
    ]
    assert len(result_blocks) >= 3, (
        f"expected the real per-round results in history, got "
        f"{len(result_blocks)}"
    )
    # They are the REAL dispatch/guard errors — never NOT-EXECUTED synthetics
    # on a path where dispatch ran.
    assert all(
        ("Unknown tool" in b.content) or ("BLOCKED" in b.content)
        for b in result_blocks
    ), [b.content for b in result_blocks]
    assert all(b.is_error for b in result_blocks)


# --------------------------------------------------------------------------- #
# The helper's own contract.
# --------------------------------------------------------------------------- #


def test_unanswered_tool_results_pairs_every_call():
    calls = [
        ToolUseBlock(id="a", name="t1", input={}),
        ToolUseBlock(id="b", name="t2", input={}),
    ]
    msg = _unanswered_tool_results(calls)
    assert msg.role == "user"
    ids = [b.tool_use_id for b in msg.content]
    assert ids == ["a", "b"]
    assert all(isinstance(b, ToolResultBlock) and b.is_error for b in msg.content)
    assert all("NOT EXECUTED" in b.content for b in msg.content)


def test_unanswered_tool_results_empty_calls_is_an_empty_user_turn():
    msg = _unanswered_tool_results([])
    assert msg.role == "user" and msg.content == []
