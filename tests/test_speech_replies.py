"""``speech/replies.py``: each session's reply, held from the frames a real ``WebSocketBridge`` broadcasts.

The rules (docs/design/speech-route.md, section 2.3, D14): a reply is a turn; it binds by the first ``message_id`` its
text or ``chat_done`` carries (turns in a session run one at a time); it ends on its own ``chat_done``, an ``error``
while it runs, or the session's open-turn count (``agent_state`` thinking/idle) reaching zero. Held by
``(session_id, message_id)``; a finished one is kept 120 s. A desktop task's ``chat_done`` is not a reply.

Every frame here is the bridge's own (``tests/support/bridge_frames.py``); only the model is scripted.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.speech.replies import FINISHED_KEEP, ReplyTap
from tests.support.bridge_frames import HOLD, Bridge, settle, text, tool

S = "desktop:speech-a"
OTHER = "desktop:speech-b"


class _Clock:
    def __init__(self) -> None:
        self.now = 500.0

    def __call__(self) -> float:
        return self.now


def _texts(reply) -> str:
    return "".join(v for k, v in reply.events if k == "text")


def _kinds(reply) -> list[str]:
    return [k for k, _ in reply.events]


@pytest.fixture
def tap():
    return ReplyTap(clock=_Clock())


@pytest.mark.asyncio
async def test_a_finished_turn_is_held_by_its_message_id(monkeypatch, tap):
    b = Bridge(monkeypatch, tap.observe)
    await b.start(S, [text("Hello there. "), text("Bye.")]).task
    mid = next(f["payload"]["message_id"] for f in b.frames if f["type"] == "chat_delta")
    reply = await tap.find(S, message_id=mid)
    assert reply is not None and reply.message_id == mid
    assert _texts(reply) == "Hello there. Bye."
    assert reply.ended == "complete"
    assert _kinds(reply)[0] == "reply" and _kinds(reply)[-1] == "end"


@pytest.mark.asyncio
async def test_another_sessions_id_finds_nothing(monkeypatch, tap):
    b = Bridge(monkeypatch, tap.observe)
    await b.start(S, [text("Secret words.")]).task
    mid = next(f["payload"]["message_id"] for f in b.frames if f["type"] == "chat_delta")
    assert await tap.find(OTHER, message_id=mid, wait=0.01) is None


@pytest.mark.asyncio
async def test_waiting_for_the_next_turn(monkeypatch, tap):
    b = Bridge(monkeypatch, tap.observe)
    finding = asyncio.ensure_future(tap.find(S, wait=2.0))
    await asyncio.sleep(0.01)
    await b.start(S, [text("Fresh reply.")]).task
    reply = await finding
    assert reply is not None and _texts(reply) == "Fresh reply." and reply.ended == "complete"


@pytest.mark.asyncio
async def test_no_turn_within_wait_is_none(tap):
    assert await tap.find(S, wait=0.02) is None


@pytest.mark.asyncio
async def test_a_running_turn_is_found_from_its_first_word(monkeypatch, tap):
    b = Bridge(monkeypatch, tap.observe)
    turn = b.start(S, [text("Early words. "), HOLD, text("Late words.")])
    await settle(lambda: any(f["type"] == "chat_delta" for f in b.frames))
    reply = await tap.find(S, wait=0.01)
    assert reply is not None and _texts(reply) == "Early words. "
    turn.release()
    await turn.task
    assert _texts(reply) == "Early words. Late words." and reply.ended == "complete"


@pytest.mark.asyncio
async def test_a_turn_that_runs_tools_first_is_still_the_reply(monkeypatch, tap):
    b = Bridge(monkeypatch, tap.observe)
    finding = asyncio.ensure_future(tap.find(S, wait=0.05))
    await asyncio.sleep(0.01)
    turn = b.start(S, [HOLD, tool(), text("Found it.")])
    await settle(lambda: any(f["type"] == "agent_state" for f in b.frames))
    await asyncio.sleep(0.1)  # longer than wait: the turn has started, so wait no longer applies
    turn.release()
    await turn.task
    reply = await finding
    assert reply is not None and _texts(reply) == "Found it."
    assert "boundary" in _kinds(reply)


@pytest.mark.asyncio
async def test_a_stopped_turn_ends_interrupted(monkeypatch, tap):
    b = Bridge(monkeypatch, tap.observe)
    turn = b.start(S, [text("Half a "), HOLD])
    await settle(lambda: any(f["type"] == "chat_delta" for f in b.frames))
    reply = await tap.find(S, wait=0.01)
    assert b.interrupt(S)
    await turn.task
    assert reply.ended == "interrupted"


@pytest.mark.asyncio
async def test_a_failed_turn_ends_failed(monkeypatch, tap):
    b = Bridge(monkeypatch, tap.observe)
    finding = asyncio.ensure_future(tap.find(S, wait=2.0))
    await asyncio.sleep(0.01)
    await b.start(S, [text("Start. "), RuntimeError("model went away")]).task
    reply = await finding
    assert reply.ended == "failed" and _texts(reply) == "Start. "


@pytest.mark.asyncio
async def test_a_turn_that_fails_before_a_word_ends_failed(monkeypatch, tap):
    b = Bridge(monkeypatch, tap.observe)
    finding = asyncio.ensure_future(tap.find(S, wait=2.0))
    await asyncio.sleep(0.01)
    await b.start(S, [RuntimeError("no words at all")]).task
    reply = await finding
    assert reply is not None and reply.ended == "failed"


@pytest.mark.asyncio
async def test_a_queued_turn_never_ends_the_running_one(monkeypatch, tap):
    """Review N1: B's thinking arrives while A runs; A's done and idle must not end B, nor B's frames end A."""
    b = Bridge(monkeypatch, tap.observe)
    a = b.start(S, [text("Answer A. "), HOLD, text("More A.")])
    await settle(lambda: any(f["type"] == "chat_delta" for f in b.frames))
    reply_a = await tap.find(S, wait=0.01)
    second = b.start(S, [HOLD, text("Answer B.")])
    await settle(lambda: sum(f["type"] == "agent_state" and f["payload"]["state"] == "thinking"
                             for f in b.frames) == 2)
    assert reply_a.ended is None
    a.release()
    await a.task
    assert reply_a.ended == "complete" and _texts(reply_a) == "Answer A. More A."
    # After A ended and before B speaks, the running turn is B.
    reply_b = await tap.find(S, wait=0.01)
    second.release()
    await second.task
    assert reply_b is not None and reply_b is not reply_a
    assert _texts(reply_b) == "Answer B." and reply_b.ended == "complete"


@pytest.mark.asyncio
async def test_a_discarded_round_is_passed_on(tap):
    tap.observe({"type": "agent_state", "payload": {"state": "thinking", "session_id": S}})
    tap.observe({"type": "chat_delta", "payload": {"session_id": S, "message_id": "asst-1", "content": "Good. "}})
    tap.observe({"type": "chat_discard", "payload": {"session_id": S, "message_id": "asst-1", "keep_chars": 3}})
    reply = await tap.find(S, message_id="asst-1")
    assert ("discard", 3) in reply.events


@pytest.mark.asyncio
async def test_a_desktop_task_chat_done_is_not_a_reply(tap):
    finding = asyncio.ensure_future(tap.find(S, wait=0.05))
    await asyncio.sleep(0)
    tap.observe({"type": "chat_done", "payload": {"session_id": S, "message_id": "computer:t1"}})
    assert await finding is None


@pytest.mark.asyncio
async def test_the_count_never_goes_below_zero(tap):
    tap.observe({"type": "agent_state", "payload": {"state": "idle", "session_id": S}})
    tap.observe({"type": "agent_state", "payload": {"state": "idle", "session_id": S}})
    tap.observe({"type": "agent_state", "payload": {"state": "thinking", "session_id": S}})
    reply = await tap.find(S, wait=0.01)
    assert reply is not None and reply.ended is None  # one open turn, so one running reply


@pytest.mark.asyncio
async def test_a_finished_reply_is_kept_for_its_time_then_gone(monkeypatch):
    clock = _Clock()
    tap = ReplyTap(clock=clock)
    b = Bridge(monkeypatch, tap.observe)
    await b.start(S, [text("Kept a while.")]).task
    mid = next(f["payload"]["message_id"] for f in b.frames if f["type"] == "chat_delta")
    clock.now += FINISHED_KEEP - 1
    assert await tap.find(S, message_id=mid) is not None
    clock.now += 2
    tap.observe({"type": "agent_state", "payload": {"state": "idle", "session_id": OTHER}})  # any frame prunes
    assert await tap.find(S, message_id=mid) is None


def test_observe_never_raises(tap):
    for frame in (None, 3, {}, {"type": "chat_delta"}, {"type": "chat_delta", "payload": "x"},
                  {"type": "chat_delta", "payload": {"session_id": 5, "message_id": [], "content": None}},
                  {"type": "chat_discard", "payload": {"session_id": S, "message_id": "m", "keep_chars": "x"}},
                  {"type": "agent_state", "payload": {"state": "thinking"}}):
        tap.observe(frame)


@pytest.mark.asyncio
async def test_following_a_reply_yields_events_as_they_come(monkeypatch, tap):
    b = Bridge(monkeypatch, tap.observe)
    turn = b.start(S, [text("One. "), HOLD, text("Two.")])
    await settle(lambda: any(f["type"] == "chat_delta" for f in b.frames))
    reply = await tap.find(S, wait=0.01)
    seen: list = []

    async def follow():
        async for event in reply.follow():
            seen.append(event)

    task = asyncio.ensure_future(follow())
    await settle(lambda: ("text", "One. ") in seen)
    turn.release()
    await turn.task
    await asyncio.wait_for(task, 1)
    assert seen[-1] == ("end", "complete")
    assert [v for k, v in seen if k == "text"] == ["One. ", "Two."]
