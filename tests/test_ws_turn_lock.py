"""WS bridge per-session turn serialization (2026-08-11 duplicate-rows fix).

``_handle_send_message`` fires ``_run_agent`` as a background task per send.
Before the lock, rapid-fire sends on one session ran N ``run_loop``s
concurrently, all appending in place to the SAME ``session.messages`` list:
model rounds interleaved across turns, and each finishing turn's
``persist_loop_result(original_len)`` re-persisted every other in-flight
turn's rows to LCM — the 4× row fan the 2026-08-11 survey found in the
``desktop:s4-*`` sessions. The telegram gateway has serialized its turns
since audit M6; these tests hold the WS bridge to the same contract:

  * turns on ONE session run strictly one-after-another (``original_len``
    captured only once the previous turn fully finished, persist included);
  * turns on DIFFERENT sessions never wait on each other;
  * the Stop button targets the turn that is actually RUNNING — a queued
    turn is untouched by the interrupt and runs (serialized) afterwards.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.web.ws_server import WebSocketBridge


class _FakeSession:
    """Just enough ChatSession: an in-place messages list + persist recorder."""

    def __init__(self) -> None:
        self.messages: list = []
        self.persist_calls: list[int] = []

    def get_messages(self) -> list:
        return self.messages  # the SAME list run_loop appends onto in place

    def persist_loop_result(self, original_len: int) -> None:
        self.persist_calls.append(original_len)


async def _wait_until(cond, timeout: float = 2.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not cond():
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("condition not reached before timeout")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_same_session_turns_serialize(monkeypatch):
    """Two turns on one session: the second's run_loop must not start (and
    must not capture its original_len) until the first — persist included —
    is completely done."""
    import prometheus.engine.agent_loop as agent_loop_mod

    entered: list[int] = []          # len(messages) at run_loop entry
    release_first = asyncio.Event()

    async def fake_run_loop(context, messages, mode="agent", session_id=None,
                            tool_choice=None):
        entered.append(len(messages))
        if len(entered) == 1:
            await release_first.wait()
        messages.append(f"turn-{len(entered)}-reply")  # in place, like the real loop
        if False:  # pragma: no cover — makes this an async generator
            yield

    monkeypatch.setattr(agent_loop_mod, "run_loop", fake_run_loop)
    bridge = WebSocketBridge(loop_context=object())
    session = _FakeSession()
    session.messages.append("u0")

    t1 = asyncio.ensure_future(bridge._run_agent("desktop:s", session))
    await _wait_until(lambda: len(entered) == 1)

    session.messages.append("u1")    # second send lands its user message...
    t2 = asyncio.ensure_future(bridge._run_agent("desktop:s", session))
    await asyncio.sleep(0.05)        # ...but its turn must be QUEUED, not running
    assert len(entered) == 1, "second turn entered run_loop while first still held the session"

    release_first.set()
    await asyncio.gather(t1, t2)

    # Turn 1 entered at len 1 ([u0]); turn 2 entered only after turn 1
    # appended its reply and persisted — at len 3 ([u0, u1, turn-1-reply]).
    # Overlapping turns would both have entered at low lens and persisted
    # overlapping spans (entered == [1, 2] and persist_calls == [1, 2]).
    assert entered == [1, 3]
    assert session.persist_calls == [1, 3]


@pytest.mark.asyncio
async def test_different_sessions_do_not_block_each_other(monkeypatch):
    import prometheus.engine.agent_loop as agent_loop_mod

    order: list[str] = []
    release = asyncio.Event()

    async def fake_run_loop(context, messages, mode="agent", session_id=None,
                            tool_choice=None):
        order.append(f"enter:{session_id}")
        if session_id == "desktop:a":
            await release.wait()
        order.append(f"exit:{session_id}")
        if False:  # pragma: no cover
            yield

    monkeypatch.setattr(agent_loop_mod, "run_loop", fake_run_loop)
    bridge = WebSocketBridge(loop_context=object())

    t1 = asyncio.ensure_future(bridge._run_agent("desktop:a", _FakeSession()))
    await _wait_until(lambda: "enter:desktop:a" in order)

    # Session b must run to completion while session a is still parked.
    await bridge._run_agent("desktop:b", _FakeSession())
    assert "exit:desktop:b" in order
    assert "exit:desktop:a" not in order

    release.set()
    await t1
    assert "exit:desktop:a" in order


@pytest.mark.asyncio
async def test_interrupt_targets_running_turn_and_queued_turn_proceeds(monkeypatch):
    """With turn A running and turn B queued on the lock, Stop must cancel A
    (the only registered turn) and B must then run to completion."""
    import prometheus.engine.agent_loop as agent_loop_mod

    entered: list[int] = []

    async def fake_run_loop(context, messages, mode="agent", session_id=None,
                            tool_choice=None):
        entered.append(len(entered) + 1)
        if entered[0] == 1 and len(entered) == 1:
            await asyncio.Event().wait()  # A parks here until cancelled
        messages.append("b-reply")
        if False:  # pragma: no cover
            yield

    monkeypatch.setattr(agent_loop_mod, "run_loop", fake_run_loop)
    bridge = WebSocketBridge(loop_context=object())
    session = _FakeSession()

    ta = asyncio.ensure_future(bridge._run_agent("desktop:s", session))
    await _wait_until(lambda: len(entered) == 1 and "desktop:s" in bridge._turn_tasks)

    tb = asyncio.ensure_future(bridge._run_agent("desktop:s", session))
    await asyncio.sleep(0.05)
    # B is queued: not registered, so the Stop button cannot be aiming at it.
    assert bridge._turn_tasks["desktop:s"] is ta

    assert bridge.interrupt_turn("desktop:s") is True
    await ta                       # a user stop RETURNS (does not raise)

    await tb                       # ...and the queued turn then runs normally
    assert len(entered) == 2
    assert "b-reply" in session.messages
    assert "desktop:s" not in bridge._turn_tasks
