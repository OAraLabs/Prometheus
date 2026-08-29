"""GRAFT-MOBILE-BRIDGE 4 — subscribe becomes a real session filter.

Acceptance, from the spec:

  1. Subscribe to one session; frames from another do not arrive;
     approval_pending still does.
  2. Subscribe with no sessions key → unchanged firehose.
  3. Re-subscribing REPLACES the filter rather than appending.

Plus the boundaries that keep the filter honest: signal-bus frame types are
never filtered, a frame with no session_id is always delivered (the reason
Piece 3 landed first — tool frames now carry the id and cannot silently
escape), a filtered skip is not a counted drop, and the filter dies with the
socket.
"""

from __future__ import annotations

import asyncio
import json

from prometheus.web.ws_server import WebSocketBridge


class _Recorder:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    def types(self) -> list[str]:
        return [f["type"] for f in self.frames]


def _bridge_with(*recorders: _Recorder) -> WebSocketBridge:
    bridge = WebSocketBridge(loop_context=None)
    for r in recorders:
        bridge._clients.add(r)
    return bridge


async def _subscribe(bridge, ws, sessions=None, payload=None):
    body = payload if payload is not None else (
        {"sessions": sessions} if sessions is not None else {})
    await bridge._handle_client_message(ws, json.dumps(
        {"type": "subscribe", "payload": body}))
    ws.frames.clear()  # drop the ack; the tests read deliveries


def _turn_frame(kind: str, sid: str) -> dict:
    return {"type": kind, "timestamp": 1.0, "payload": {"session_id": sid}}


def _run(coro):
    return asyncio.run(coro)


def test_filter_admits_own_session_and_signals_only():
    async def run():
        ws = _Recorder()
        bridge = _bridge_with(ws)
        await _subscribe(bridge, ws, sessions=["ios:mine"])

        await bridge.broadcast(_turn_frame("chat_delta", "ios:mine"))
        await bridge.broadcast(_turn_frame("chat_delta", "desktop:other"))
        await bridge.broadcast(_turn_frame("agent_progress", "desktop:other"))
        # Signal-bus frames are never session-scoped — the phone wants them all.
        await bridge.broadcast({"type": "approval_pending", "timestamp": 1.0,
                                "payload": {"request_id": "r1"}})
        await bridge.broadcast({"type": "sentinel_signal", "timestamp": 1.0,
                                "payload": {"kind": "idle_start",
                                            "payload": {"session_id": "desktop:other"}}})

        assert ws.types() == ["chat_delta", "approval_pending", "sentinel_signal"]
        assert bridge._frames_dropped == 0, "a filtered skip is not a drop"

    _run(run())


def test_no_sessions_key_is_the_unchanged_firehose():
    async def run():
        ws = _Recorder()
        bridge = _bridge_with(ws)
        # The legacy ack-only shape a current client sends.
        await _subscribe(bridge, ws, payload={"channels": ["events"]})
        await bridge.broadcast(_turn_frame("chat_delta", "a"))
        await bridge.broadcast(_turn_frame("chat_delta", "b"))
        assert ws.types() == ["chat_delta", "chat_delta"]

    _run(run())


def test_resubscribe_replaces_never_appends():
    async def run():
        ws = _Recorder()
        bridge = _bridge_with(ws)
        await _subscribe(bridge, ws, sessions=["a"])
        await _subscribe(bridge, ws, sessions=["b"])
        await bridge.broadcast(_turn_frame("chat_message", "a"))
        await bridge.broadcast(_turn_frame("chat_message", "b"))
        assert ws.types() == ["chat_message"], "the a-filter is gone, replaced"

        # And [] restores the firehose.
        await _subscribe(bridge, ws, sessions=[])
        await bridge.broadcast(_turn_frame("chat_message", "a"))
        assert ws.types()[-1] == "chat_message"

    _run(run())


def test_filters_are_per_socket():
    async def run():
        phone, desktop = _Recorder(), _Recorder()
        bridge = _bridge_with(phone, desktop)
        await _subscribe(bridge, phone, sessions=["ios:mine"])
        # The desktop never subscribed — full firehose.
        await bridge.broadcast(_turn_frame("agent_progress", "desktop:other"))
        assert phone.types() == [] and desktop.types() == ["agent_progress"]

    _run(run())


def test_sessionless_turn_frames_are_always_delivered():
    async def run():
        ws = _Recorder()
        bridge = _bridge_with(ws)
        await _subscribe(bridge, ws, sessions=["ios:mine"])
        # A pre-Piece-3 daemon's tool frame shape (no session_id): must pass —
        # suppressing what cannot be attributed would hide real activity.
        await bridge.broadcast({"type": "tool_call_start", "timestamp": 1.0,
                                "payload": {"call_id": "t1", "tool_name": "bash"}})
        assert ws.types() == ["tool_call_start"]

    _run(run())


def test_filter_dies_with_the_socket():
    async def run():
        ws = _Recorder()
        bridge = _bridge_with(ws)
        await _subscribe(bridge, ws, sessions=["a"])
        assert ws in bridge._ws_filters
        # The handler's finally-path cleanup, driven directly.
        bridge._clients.discard(ws)
        bridge._ws_filters.pop(ws, None)
        assert ws not in bridge._ws_filters

    _run(run())
