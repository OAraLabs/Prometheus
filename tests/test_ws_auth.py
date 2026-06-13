"""WebSocket first-frame auth on the :8010 bridge (Phase A — feat/ws-auth).

The bridge streams chat content, tool inputs/outputs, and coding-task lifecycle
signals to every connected client. Before this sprint it was fully open. These
tests drive a REAL ``websockets`` client against a REAL ``WebSocketBridge`` on
an ephemeral port and assert the security property that matters: a client that
does not authenticate receives ZERO frames before being closed 4401 — not just
that ``close()`` was called, but that no event was ever delivered.

Mechanism (see audits/20260613T000000Z-ws-auth-phase0.md): first-frame auth,
chosen because a browser client exists and cannot set an upgrade header.

The token here is a random per-test value — never the real PROMETHEUS_API_TOKEN.
"""

from __future__ import annotations

import asyncio
import json
import secrets

import pytest

websockets = pytest.importorskip("websockets")
from websockets.exceptions import ConnectionClosed  # noqa: E402

from prometheus.web.ws_server import WS_CLOSE_UNAUTHORIZED, WebSocketBridge  # noqa: E402

TEST_TOKEN = "test-" + secrets.token_hex(8)  # NOT the real secret


async def _start_bridge(token: str | None) -> tuple[WebSocketBridge, int]:
    bridge = WebSocketBridge(api_token=token)
    await bridge.start(host="127.0.0.1", port=0)
    port = bridge._server.sockets[0].getsockname()[1]
    return bridge, port


def _uri(port: int) -> str:
    return f"ws://127.0.0.1:{port}"


async def _recv_first(ws, timeout: float = 6.0):
    """Return the first frame, or raise ConnectionClosed if the socket is
    closed before any frame arrives (the zero-frame case)."""
    return await asyncio.wait_for(ws.recv(), timeout=timeout)


def _close_code(exc: ConnectionClosed) -> int:
    """Read the server's close code (``.rcvd.code`` in websockets 13.1+)."""
    return exc.rcvd.code if exc.rcvd is not None else exc.code


# ---------------------------------------------------------------------------
# Rejection paths — closed 4401, ZERO frames delivered
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_auth_frame_is_closed_4401_with_zero_frames():
    bridge, port = await _start_bridge(TEST_TOKEN)
    try:
        async with websockets.connect(_uri(port)) as ws:
            # Send nothing. The server must close us out with zero frames.
            with pytest.raises(ConnectionClosed) as exc:
                await _recv_first(ws)
            assert _close_code(exc.value) == WS_CLOSE_UNAUTHORIZED
    finally:
        await bridge.stop()


@pytest.mark.asyncio
async def test_wrong_token_is_closed_4401_with_zero_frames():
    bridge, port = await _start_bridge(TEST_TOKEN)
    try:
        async with websockets.connect(_uri(port)) as ws:
            await ws.send(json.dumps({"type": "auth", "token": "wrong-token"}))
            with pytest.raises(ConnectionClosed) as exc:
                await _recv_first(ws)
            assert _close_code(exc.value) == WS_CLOSE_UNAUTHORIZED
    finally:
        await bridge.stop()


@pytest.mark.asyncio
async def test_malformed_first_frame_is_closed_4401():
    bridge, port = await _start_bridge(TEST_TOKEN)
    try:
        # Not JSON at all.
        async with websockets.connect(_uri(port)) as ws:
            await ws.send("this is not json")
            with pytest.raises(ConnectionClosed) as exc:
                await _recv_first(ws)
            assert _close_code(exc.value) == WS_CLOSE_UNAUTHORIZED
        # Valid JSON but wrong type (a `subscribe`, like the legacy UI sends).
        async with websockets.connect(_uri(port)) as ws:
            await ws.send(json.dumps({"type": "subscribe", "payload": {}}))
            with pytest.raises(ConnectionClosed) as exc:
                await _recv_first(ws)
            assert _close_code(exc.value) == WS_CLOSE_UNAUTHORIZED
    finally:
        await bridge.stop()


@pytest.mark.asyncio
async def test_late_auth_is_closed_4401_with_zero_frames():
    # Patch the timeout down so the test doesn't wait the full 5s.
    import prometheus.web.ws_server as mod

    bridge, port = await _start_bridge(TEST_TOKEN)
    monkeyed = 0.3
    orig = mod.AUTH_FRAME_TIMEOUT_SECONDS
    mod.AUTH_FRAME_TIMEOUT_SECONDS = monkeyed
    try:
        async with websockets.connect(_uri(port)) as ws:
            # Stay silent past the (shortened) deadline.
            await asyncio.sleep(monkeyed + 0.3)
            with pytest.raises(ConnectionClosed) as exc:
                await _recv_first(ws, timeout=2.0)
            assert _close_code(exc.value) == WS_CLOSE_UNAUTHORIZED
    finally:
        mod.AUTH_FRAME_TIMEOUT_SECONDS = orig
        await bridge.stop()


# ---------------------------------------------------------------------------
# Success path — authenticated client streams normally
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_correct_auth_receives_welcome_then_events():
    bridge, port = await _start_bridge(TEST_TOKEN)
    try:
        async with websockets.connect(_uri(port)) as ws:
            await ws.send(json.dumps({"type": "auth", "token": TEST_TOKEN}))
            # First frame after a good auth is the welcome.
            welcome = json.loads(await _recv_first(ws))
            assert welcome["type"] == "connected"
            # A subsequent broadcast reaches the authenticated client.
            await bridge.broadcast({"type": "agent_state",
                                    "payload": {"state": "thinking"}})
            evt = json.loads(await asyncio.wait_for(ws.recv(), timeout=4.0))
            assert evt["type"] == "agent_state"
    finally:
        await bridge.stop()


# ---------------------------------------------------------------------------
# Auth disabled (empty token) — legacy behavior preserved
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_token_disables_auth_legacy_behavior():
    bridge, port = await _start_bridge("")  # auth OFF (REST parity)
    assert bridge.auth_required is False
    try:
        async with websockets.connect(_uri(port)) as ws:
            # No auth frame needed — welcome arrives immediately, and the
            # legacy `subscribe`-first flow keeps working.
            welcome = json.loads(await _recv_first(ws))
            assert welcome["type"] == "connected"
            await ws.send(json.dumps({"type": "subscribe",
                                      "payload": {"channels": ["*"]}}))
            ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=4.0))
            assert ack["type"] == "subscribed"
    finally:
        await bridge.stop()


# ---------------------------------------------------------------------------
# Token validation unit (no socket) — constant-time, shape-strict
# ---------------------------------------------------------------------------


def test_token_ok_validation():
    bridge = WebSocketBridge(api_token=TEST_TOKEN)
    assert bridge._token_ok(json.dumps({"type": "auth", "token": TEST_TOKEN}))
    assert not bridge._token_ok(json.dumps({"type": "auth", "token": "no"}))
    assert not bridge._token_ok(json.dumps({"type": "auth"}))  # missing token
    assert not bridge._token_ok(json.dumps({"type": "subscribe"}))  # wrong type
    assert not bridge._token_ok("not json")
    assert not bridge._token_ok(json.dumps({"type": "auth", "token": 123}))  # non-str
