"""WS interrupt frame + /api/chat/interrupt — stop a running turn (feat/ws-interrupt-frame).

The chat Stop button was blocked on the bridge having no interrupt path: a
dispatched turn ran to completion no matter what the user did. These tests
assert the interrupt contract end to end:

  * ``interrupt_turn()`` cancels the RUNNING turn's task and only that —
    a quiet session returns ``stopped: false`` instead of erroring.
  * A user stop is caught cleanly: partial streamed text survives as an
    assistant turn (persisted via ``persist_loop_result``), every client gets
    a terminal ``chat_done{interrupted: true}``, and the agent state returns
    to idle. Old clients that ignore the extra field see a normal done.
  * An UNFLAGGED cancel (daemon shutdown) still propagates — the handler must
    never swallow a system cancel, or teardown would hang.
  * The WS ``interrupt`` frame acks to the requesting socket; the HTTP twin
    ``POST /api/chat/interrupt`` mirrors /api/chat/send's failure modes.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

from prometheus.web.ws_server import WebSocketBridge


@dataclass(frozen=True)
class AssistantTextDelta:
    """Name-matched stand-in — _run_agent routes events by type().__name__."""

    text: str


class _FakeSession:
    """Just enough ChatSession: an in-place messages list + persist recorder."""

    def __init__(self) -> None:
        self.messages: list = []
        self.persist_calls: list[int] = []

    def get_messages(self) -> list:
        return self.messages  # the SAME list run_loop appends onto in place

    def persist_loop_result(self, original_len: int) -> None:
        self.persist_calls.append(original_len)


class _Recorder:
    """Fake WS client capturing every broadcast frame."""

    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    def types(self) -> list[str]:
        return [f["type"] for f in self.frames]


def _bridge_with_recorder() -> tuple[WebSocketBridge, _Recorder]:
    bridge = WebSocketBridge(loop_context=object())
    rec = _Recorder()
    bridge._clients.add(rec)
    return bridge, rec


def _hanging_run_loop(deltas: list[str], append_to: list | None = None):
    """Fake run_loop: stream ``deltas``, optionally append a completed-round
    message in place (as the real loop does), then hang until cancelled."""

    async def run_loop(context, messages, mode="agent", session_id=None, tool_choice=None):
        for text in deltas:
            yield AssistantTextDelta(text=text), None
        if append_to is not None:
            append_to.append("completed-round-msg")
        await asyncio.Event().wait()  # park here until cancel lands

    return run_loop


async def _wait_until(cond, timeout: float = 2.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not cond():
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("condition not reached before timeout")
        await asyncio.sleep(0.01)


async def _start_turn(bridge, session, monkeypatch, run_loop, until):
    """Kick off _run_agent as a task and wait until ``until()`` holds (i.e. the
    fake loop has demonstrably progressed to the state the test targets)."""
    import prometheus.engine.agent_loop as agent_loop_mod

    monkeypatch.setattr(agent_loop_mod, "run_loop", run_loop)
    task = asyncio.ensure_future(bridge._run_agent("desktop:t", session))
    await _wait_until(lambda: until() and "desktop:t" in bridge._turn_tasks)
    return task


# ---------------------------------------------------------------------------
# interrupt_turn() — the bridge-level contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_midstream_keeps_partial_and_emits_interrupted_done(monkeypatch):
    bridge, rec = _bridge_with_recorder()
    session = _FakeSession()
    task = await _start_turn(
        bridge, session, monkeypatch, _hanging_run_loop(["Hello ", "wor"]),
        until=lambda: sum(f["type"] == "chat_delta" for f in rec.frames) >= 2,
    )

    assert bridge.interrupt_turn("desktop:t") is True
    text, usage = await task  # a user stop RETURNS — it must not raise

    assert text == "Hello wor"
    # Partial streamed text became a real assistant turn and was persisted.
    assert len(session.messages) == 1
    assert session.messages[0].role == "assistant"
    assert session.messages[0].text == "Hello wor"
    assert session.persist_calls == [0]
    # Terminal frame: chat_done carrying interrupted=True, then idle state.
    done = next(f for f in rec.frames if f["type"] == "chat_done")
    assert done["payload"]["interrupted"] is True
    assert done["payload"]["session_id"] == "desktop:t"
    assert rec.frames[-1]["type"] == "agent_state"
    assert rec.frames[-1]["payload"]["state"] == "idle"
    # Registry and flag are both cleared — nothing leaks to the next turn.
    assert "desktop:t" not in bridge._turn_tasks
    assert "desktop:t" not in bridge._interrupted


@pytest.mark.asyncio
async def test_interrupt_after_completed_round_does_not_duplicate_text(monkeypatch):
    bridge, rec = _bridge_with_recorder()
    session = _FakeSession()
    # run_loop appends a completed round in place (the real in-place contract),
    # so the handler must NOT also append the accumulated text on top of it.
    task = await _start_turn(
        bridge, session, monkeypatch,
        _hanging_run_loop(["round one text"], append_to=session.messages),
        until=lambda: len(session.messages) == 1,  # the in-place append landed
    )

    assert bridge.interrupt_turn("desktop:t") is True
    await task

    assert session.messages == ["completed-round-msg"]  # no synthetic duplicate
    assert session.persist_calls == [0]
    done = next(f for f in rec.frames if f["type"] == "chat_done")
    assert done["payload"]["interrupted"] is True


@pytest.mark.asyncio
async def test_interrupt_quiet_session_returns_false():
    bridge, _ = _bridge_with_recorder()
    assert bridge.interrupt_turn("desktop:quiet") is False


@pytest.mark.asyncio
async def test_system_cancel_still_propagates(monkeypatch):
    # An UNFLAGGED cancel (daemon shutdown) is not a user interrupt: it must
    # re-raise so shutdown never hangs, and no interrupted chat_done is faked.
    bridge, rec = _bridge_with_recorder()
    session = _FakeSession()
    task = await _start_turn(
        bridge, session, monkeypatch, _hanging_run_loop(["x"]),
        until=lambda: any(f["type"] == "chat_delta" for f in rec.frames),
    )

    task.cancel()  # directly — bypassing interrupt_turn, so no flag is set
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    assert "chat_done" not in [f["type"] for f in rec.frames]
    assert "desktop:t" not in bridge._turn_tasks


# ---------------------------------------------------------------------------
# The WS "interrupt" frame — dispatcher ack to the requesting socket
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ws_interrupt_frame_acks_stopped_false_for_quiet_session():
    websockets = pytest.importorskip("websockets")
    bridge = WebSocketBridge()  # no token → auth disabled, same as REST
    await bridge.start(host="127.0.0.1", port=0)
    port = bridge._server.sockets[0].getsockname()[1]
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=6.0))
            assert first["type"] == "connected"
            await ws.send(json.dumps(
                {"type": "interrupt", "payload": {"session_id": "desktop:none"}}
            ))
            ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=6.0))
            assert ack["type"] == "interrupt_ack"
            assert ack["payload"] == {"session_id": "desktop:none", "stopped": False}
    finally:
        await bridge.stop()


# ---------------------------------------------------------------------------
# POST /api/chat/interrupt — HTTP twin, /api/chat/send failure-mode parity
# ---------------------------------------------------------------------------


def _http_client(bridge=None):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.web.server import create_app

    app = create_app({})
    if bridge is not None:
        app.state.ws_bridge = bridge
    return TestClient(app)


class _StubBridge:
    def __init__(self, result: bool) -> None:
        self.result = result
        self.calls: list[str] = []

    def interrupt_turn(self, session_id: str) -> bool:
        self.calls.append(session_id)
        return self.result


def test_http_interrupt_passes_through_stopped_true():
    bridge = _StubBridge(True)
    resp = _http_client(bridge).post("/api/chat/interrupt", json={"session_id": "desktop:t"})
    assert resp.status_code == 200
    assert resp.json() == {"session_id": "desktop:t", "stopped": True}
    assert bridge.calls == ["desktop:t"]


def test_http_interrupt_quiet_session_is_200_stopped_false():
    resp = _http_client(_StubBridge(False)).post(
        "/api/chat/interrupt", json={"session_id": "desktop:t"}
    )
    assert resp.status_code == 200
    assert resp.json()["stopped"] is False


def test_http_interrupt_missing_session_id_is_400():
    resp = _http_client(_StubBridge(True)).post("/api/chat/interrupt", json={})
    assert resp.status_code == 400


def test_http_interrupt_invalid_json_is_400():
    client = _http_client(_StubBridge(True))
    resp = client.post(
        "/api/chat/interrupt", content=b"not json", headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 400


def test_http_interrupt_no_bridge_is_503():
    resp = _http_client(None).post("/api/chat/interrupt", json={"session_id": "s"})
    assert resp.status_code == 503
    assert "ws_bridge" in resp.json()["error"]


def test_http_interrupt_bridge_without_method_is_503():
    class _Wrong:  # no interrupt_turn
        pass

    resp = _http_client(_Wrong()).post("/api/chat/interrupt", json={"session_id": "s"})
    assert resp.status_code == 503
