"""A client connecting or leaving mid-fan-out must not break the turn.

THE DEFECT
----------
`WebSocketBridge.broadcast` iterated `self._clients` directly while awaiting
`ws.send()` inside the loop. `_handler` mutates that same set on both sides of
a connection's life — `self._clients.add(websocket)` when one connects, and
`self._clients.discard(websocket)` in its `finally` when one goes away. Either,
landing during a fan-out, raised

    RuntimeError: Set changed size during iteration

out of `broadcast()` and into the turn. The caller catches broadly, so the turn
rolled back and the user got an error frame — because somebody else opened a
browser tab.

A snapshot is also the correct semantics, not just a way to stop the raise: the
recipients of an event are the clients connected when it was emitted.

THE SECOND HALF
---------------
`_handler`'s `except` had a branch for close code 1009 and silence for
everything else — an ordinary 1000/1001 close, a malformed frame, a bug in a
command handler. All of them looked identical from outside: a client that
quietly stopped being handled. Now a clean close logs at INFO and anything else
at WARNING with a traceback.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.web.ws_server import WebSocketBridge  # noqa: E402


class _FakeWS:
    """A client whose `send` mutates the bridge's client set, as a real one does."""

    def __init__(self, bridge, on_send=None):
        self._bridge = bridge
        self._on_send = on_send
        self.sent: list[str] = []

    async def send(self, raw: str) -> None:
        if self._on_send is not None:
            hook, self._on_send = self._on_send, None
            hook()
        self.sent.append(raw)


def _bridge(n_quiet: int = 3) -> tuple[WebSocketBridge, _FakeWS]:
    """A bridge with just enough state for broadcast(), plus the hooked client.

    Returning the hooked client explicitly, rather than hunting for it in a
    set, is deliberate: the first draft searched for `_on_send is not None`,
    found nothing because the hook had not been attached yet, and the mutation
    never ran. The test caught its own vacuity, but only because it asserted
    the hook had fired.
    """
    bridge = WebSocketBridge.__new__(WebSocketBridge)
    bridge._clients = set()
    bridge._frames_dropped = 0
    bridge._clients_discarded = 0
    bridge._wants = lambda ws, event: True
    hooked = _FakeWS(bridge)
    bridge._clients.add(hooked)
    for _ in range(n_quiet):
        bridge._clients.add(_FakeWS(bridge))
    return bridge, hooked


def test_a_client_joining_mid_fan_out_does_not_raise():
    """`_handler` line ~226: `self._clients.add(websocket)`."""
    bridge, hooked = _bridge()
    fired: list[bool] = []

    def joins():
        fired.append(True)
        bridge._clients.add(_FakeWS(bridge))

    hooked._on_send = joins
    asyncio.run(bridge.broadcast({"type": "progress", "payload": {}}))
    assert fired, "the join never happened — this test proves nothing"


def test_a_client_leaving_mid_fan_out_does_not_raise():
    """`_handler` line ~257, the `finally`: `self._clients.discard(websocket)`."""
    bridge, hooked = _bridge()
    victim = next(ws for ws in bridge._clients if ws is not hooked)
    fired: list[bool] = []

    def leaves():
        fired.append(True)
        bridge._clients.discard(victim)

    hooked._on_send = leaves
    asyncio.run(bridge.broadcast({"type": "progress", "payload": {}}))
    assert fired, "the departure never happened — this test proves nothing"


def test_the_snapshot_is_taken_before_the_first_await():
    """A client that joins mid-fan-out is not sent THIS event.

    Pins the semantics, not just the absence of a crash: a late joiner has
    missed nothing it was owed. Without this, "fix" the raise by rebuilding the
    set each iteration and the test would still pass while delivery became
    nondeterministic.
    """
    bridge, hooked = _bridge(n_quiet=1)
    latecomer: list[_FakeWS] = []

    def joins():
        ws = _FakeWS(bridge)
        latecomer.append(ws)
        bridge._clients.add(ws)

    hooked._on_send = joins
    asyncio.run(bridge.broadcast({"type": "progress", "payload": {}}))

    assert latecomer, "the join hook never ran — this test proves nothing"
    assert latecomer[0].sent == [], (
        "a client that connected DURING the fan-out received the event it was "
        "not present for"
    )


def test_every_client_present_at_emit_still_receives_the_event():
    """The snapshot must not become an excuse to drop deliveries."""
    bridge, _hooked = _bridge(n_quiet=4)
    before = tuple(bridge._clients)

    asyncio.run(bridge.broadcast({"type": "progress", "payload": {"n": 1}}))

    for ws in before:
        assert len(ws.sent) == 1, (
            f"a client present when the event was emitted got {len(ws.sent)} "
            f"frames, expected 1"
        )


def test_a_failing_client_is_still_discarded():
    """The existing dead-client sweep must survive the change."""
    bridge, _hooked = _bridge(n_quiet=2)

    class _Broken:
        async def send(self, raw):
            raise ConnectionResetError("gone")

    broken = _Broken()
    bridge._clients.add(broken)

    asyncio.run(bridge.broadcast({"type": "progress", "payload": {}}))

    assert broken not in bridge._clients, "a client whose send failed was kept"
    assert bridge._frames_dropped == 1
    assert bridge._clients_discarded == 1


# ── the handler's silent swallow ────────────────────────────────────────────

def test_a_non_1009_exception_is_logged_not_swallowed(caplog):
    """The handler must say something for every way a connection can end.

    Driven through the real `_handler` so the assertion is about the shipped
    code path, not a re-implementation of it.
    """
    import prometheus.web.ws_server as mod

    bridge = WebSocketBridge.__new__(WebSocketBridge)
    bridge._clients = set()
    bridge._ws_identity = {}
    bridge._ws_filters = {}
    bridge._api_token = ""  # auth_required -> False, so _handler runs straight through

    boom = ValueError("a command handler blew up")

    class _WS:
        remote_address = ("198.51.100.7", 51234)

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise boom

        async def send(self, raw):
            pass

    async def _noop_send(ws, payload):
        return None

    bridge._send_one = _noop_send

    with caplog.at_level(logging.DEBUG, logger=mod.__name__):
        asyncio.run(bridge._handler(_WS()))

    assert "a command handler blew up" in caplog.text, (
        "an unhandled exception ended the client handler and NOTHING was "
        f"logged about it. Captured log:\n{caplog.text}"
    )
    assert "ValueError" in caplog.text
