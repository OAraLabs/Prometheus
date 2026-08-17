"""A failed turn must not leave its work on the session (fix/discard-poisoned-turn).

``run_loop`` appends assistant + tool-result rows onto ``session.messages`` IN
PLACE (see ChatSession.persist_loop_result). The bridge's broad except used to
return without touching them, so a turn that died left its half-turn behind and
the NEXT message rebuilt a prompt containing whatever killed it.

That is how one session bricked on 2026-08-17: a ``bash`` result carrying
llama.cpp's per-process media marker (a curl of the backend's ``/props``) made
the server reject the prompt with 400 "Failed to tokenize prompt". The rows
stayed, every later message re-sent the marker and took the same 400, and the
web chat surface has no ``/reset`` to clear it. Microcompaction would have
truncated the offending result away, but it cannot fire before round
``microcompact_after_turns`` — and the turn dies on round 0.

Tested by OUTCOME, not by call: a turn that 400s leaves the message list at
``original_len``, and the next message succeeds on a clean history.

The asymmetry with the interrupt path is deliberate and is asserted here too —
a user stop KEEPS what it got (test_ws_interrupt), a failure discards.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import httpx

from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.session import SessionManager
from prometheus.web.ws_server import WebSocketBridge
from tests.support.doubles import register_double

# The literal shape that killed the real session: llama.cpp publishes a random
# per-process media marker in /props, and mtmd rejects any prompt carrying one
# with no accompanying image ("number of media markers in text (1) exceeds
# number of bitmaps (0)").
POISON = "<__media_wfAtGbJouOGjuzjOuqrAwvZjWKBAwQ9z__>"


@dataclass(frozen=True)
class AssistantTextDelta:
    """Name-matched stand-in — _run_agent routes events by type().__name__."""

    text: str


def _tool_result_msg(text: str) -> ConversationMessage:
    return ConversationMessage(role="user", content=[TextBlock(text=text)])


def _http_400() -> httpx.HTTPStatusError:
    """The real failure: httpx raise_for_status on the backend's 400."""
    request = httpx.Request("POST", "http://backend:8080/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        json={"error": {"code": 400, "message": "Failed to tokenize prompt",
                        "type": "invalid_request_error"}},
    )
    return httpx.HTTPStatusError(
        "Client error '400 Bad Request' for url "
        "'http://backend:8080/v1/chat/completions'",
        request=request,
        response=response,
    )


@register_double(
    "turn_discard.poison_then_400",
    replaces="prometheus.engine.agent_loop.run_loop",
)
class _PoisonThen400:
    """Stream a little, append a poisoned tool result in place, then 400.

    Mirrors the real sequence: rounds succeed, a tool result lands on
    ``session.messages``, and the NEXT round's request is the one the backend
    refuses.
    """

    def __call__(self, ctx, messages, *, mode="agent", session_id=None, tool_choice=None):
        async def gen():
            yield AssistantTextDelta(text="checking the server…"), None
            messages.append(_tool_result_msg(f"props output:\n{POISON}\n"))
            raise _http_400()

        return gen()


@register_double(
    "turn_discard.records_prompt",
    replaces="prometheus.engine.agent_loop.run_loop",
)
class _RecordsPrompt:
    """A healthy turn that records the history it was handed, then replies."""

    def __init__(self) -> None:
        self.seen: list[list] = []

    def __call__(self, ctx, messages, *, mode="agent", session_id=None, tool_choice=None):
        self.seen.append(list(messages))

        async def gen():
            yield AssistantTextDelta(text="all good"), None
            messages.append(ConversationMessage(
                role="assistant", content=[TextBlock(text="all good")]))

        return gen()


def _bridge(state: dict) -> WebSocketBridge:
    bridge = WebSocketBridge(
        session_mgr=SessionManager(), loop_context=object(), agent_state_ref=state
    )
    return bridge


def _text_of(msg) -> str:
    parts = []
    for block in getattr(msg, "content", []) or []:
        parts.append(getattr(block, "text", ""))
    return "".join(parts)


def test_failed_turn_leaves_the_message_list_at_original_len(monkeypatch):
    import prometheus.engine.agent_loop as al

    monkeypatch.setattr(al, "run_loop", _PoisonThen400())

    state = {"state": "thinking"}
    bridge = _bridge(state)
    frames: list[dict] = []

    async def cap(ev):
        frames.append(ev)

    bridge.broadcast = cap

    session = bridge.session_mgr.get_or_create("web:s1")
    session.add_user_message("what model is this?")
    original_len = len(session.messages)
    assert original_len == 1

    asyncio.run(bridge._run_agent("web:s1", session))

    # THE OUTCOME: the failed turn's tail is gone; the user's turn survives.
    assert len(session.messages) == original_len
    assert session.messages[0].role == "user"
    assert POISON not in "".join(_text_of(m) for m in session.messages)
    # Still loud: the error frame goes out and state returns to idle.
    assert any(f.get("type") == "error" for f in frames)
    assert state["state"] == "idle"


def test_next_message_succeeds_after_a_failed_turn(monkeypatch):
    """The recovery property: one 400 must not brick the session."""
    import prometheus.engine.agent_loop as al

    state = {"state": "thinking"}
    bridge = _bridge(state)

    async def cap(ev):
        return None

    bridge.broadcast = cap
    session = bridge.session_mgr.get_or_create("web:s2")
    session.add_user_message("what model is this?")

    monkeypatch.setattr(al, "run_loop", _PoisonThen400())
    asyncio.run(bridge._run_agent("web:s2", session))

    # Second message on the SAME session, now with a healthy backend.
    healthy = _RecordsPrompt()
    monkeypatch.setattr(al, "run_loop", healthy)
    session.add_user_message("try again")
    text, _usage = asyncio.run(bridge._run_agent("web:s2", session))

    assert text == "all good"
    # The prompt the second turn was handed carried NO trace of the poison —
    # this is the assertion that would have caught the real defect.
    assert healthy.seen, "the healthy turn never ran"
    assert POISON not in "".join(_text_of(m) for m in healthy.seen[0])
    assert [m.role for m in healthy.seen[0]] == ["user", "user"]
    assert state["state"] == "idle"


def test_rollback_to_retreats_the_lcm_watermark():
    """Freed positions must persist again, not be skipped as already-written."""
    session = SessionManager().get_or_create("web:s3")
    session.add_user_message("one")
    original_len = len(session.messages)

    session.messages.append(_tool_result_msg("poison"))
    session.messages.append(
        ConversationMessage(role="assistant", content=[TextBlock(text="half")]))
    session._lcm_persisted_len = len(session.messages)
    session._lcm_persisted_ahead.add(original_len)

    assert session.rollback_to(original_len) == 2
    assert len(session.messages) == original_len
    assert session._lcm_persisted_len == original_len
    assert original_len not in session._lcm_persisted_ahead
    # Idempotent, and never eats rows below the mark.
    assert session.rollback_to(original_len) == 0
    assert len(session.messages) == original_len


def test_error_frame_still_names_the_session_and_status(monkeypatch):
    """Discarding must not cost the operator the diagnosis (fail-loud intact)."""
    import prometheus.engine.agent_loop as al

    monkeypatch.setattr(al, "run_loop", _PoisonThen400())
    bridge = _bridge({"state": "thinking"})
    frames: list[dict] = []

    async def cap(ev):
        frames.append(ev)

    bridge.broadcast = cap
    session = bridge.session_mgr.get_or_create("web:s4")
    session.add_user_message("go")

    asyncio.run(bridge._run_agent("web:s4", session))

    err = next(f for f in frames if f.get("type") == "error")
    assert err["payload"]["session_id"] == "web:s4"
    assert err["payload"]["status"] == 400
    assert json.dumps(err["payload"])  # frame is serialisable as sent
