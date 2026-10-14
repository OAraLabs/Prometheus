"""#760 — a rejected WebSocket auth says WHO was rejected, and why.

Found in the 0.9.8 deploy: 70 ``WS auth rejected: invalid or missing auth
token`` lines in an afternoon, none naming the client, so a stale Beacon left
open in another macOS account could only be traced through the REST access
log. Every rejection now logs the peer, its own reason, the device a revoked
token belongs to, and the handshake's Origin / User-Agent — and never any token
material (not the token, a prefix of it, or its hash). A client retrying every
minute shows as one line plus a periodic count, not hundreds.

Real ``websockets`` client against a real ``WebSocketBridge`` on an ephemeral
port, with a real ``DeviceStore`` on a scratch DB. Tokens are random per test.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets

import pytest

websockets = pytest.importorskip("websockets")

from websockets.exceptions import ConnectionClosed  # noqa: E402

from prometheus.config.device_store import DeviceStore, token_digest  # noqa: E402
from prometheus.web import ws_server  # noqa: E402
from prometheus.web.ws_server import WS_CLOSE_UNAUTHORIZED, WebSocketBridge  # noqa: E402

GLOBAL = "test" + secrets.token_hex(8)  # NOT the real secret
LOGGER = "prometheus.web.ws_server"


async def _bridge(store=None) -> tuple[WebSocketBridge, int]:
    bridge = WebSocketBridge(api_token=GLOBAL, device_store=store)
    await bridge.start(host="127.0.0.1", port=0)
    return bridge, bridge._server.sockets[0].getsockname()[1]


async def _attempt(port: int, frame: str | None, **connect_kw) -> int:
    """Connect, send *frame* (or nothing), return the close code the server sent."""
    async with websockets.connect(f"ws://127.0.0.1:{port}", **connect_kw) as ws:
        if frame is not None:
            await ws.send(frame)
        with pytest.raises(ConnectionClosed) as exc:
            await asyncio.wait_for(ws.recv(), timeout=6.0)
        rcvd = exc.value.rcvd
        assert (rcvd.reason if rcvd is not None else "") == "unauthorized"
        return rcvd.code if rcvd is not None else exc.value.code


def _rejections(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.name == LOGGER and "WS auth rejected" in r.getMessage()]


@pytest.mark.asyncio
async def test_each_rejection_has_its_own_reason_and_names_the_peer(caplog, monkeypatch):
    monkeypatch.setattr(ws_server, "AUTH_FRAME_TIMEOUT_SECONDS", 0.3)
    bridge, port = await _bridge()
    cases = {
        "malformed auth frame": "this is not json",
        "not an auth frame": json.dumps({"type": "subscribe"}),
        "missing token": json.dumps({"type": "auth"}),
        "unknown token": json.dumps({"type": "auth", "token": "wrong" + secrets.token_hex(6)}),
        "auth frame timeout": None,
    }
    try:
        with caplog.at_level(logging.INFO, logger=LOGGER):
            for frame in cases.values():
                assert await _attempt(port, frame) == WS_CLOSE_UNAUTHORIZED
    finally:
        await bridge.stop()

    lines = _rejections(caplog)
    assert len(lines) == len(cases), lines
    for reason, line in zip(cases, lines):
        assert f"({reason})" in line, line
        assert "from 127.0.0.1" in line, line


@pytest.mark.asyncio
async def test_a_revoked_device_is_named_and_no_token_material_is_logged(tmp_path, caplog):
    store = DeviceStore(tmp_path / "devices.db")
    minted = store.mint("Kitchen Mac", "macos")
    token = minted["token"]
    store.revoke(minted["id"])
    bridge, port = await _bridge(store)
    try:
        with caplog.at_level(logging.DEBUG):
            await _attempt(port, json.dumps({"type": "auth", "token": token}),
                           user_agent_header="Beacon/0.4.3 (macOS)",
                           origin="http://localhost:5173")
    finally:
        await bridge.stop()

    [line] = _rejections(caplog)
    assert "(revoked device)" in line
    assert minted["id"] in line and "Kitchen Mac" in line
    assert "Beacon/0.4.3" in line and "localhost:5173" in line
    # Every record the DAEMON side wrote, at any level — including the
    # websockets library's own DEBUG frame log, which used to carry the auth
    # frame verbatim. (websockets.client is this test's client, not the daemon.)
    everything = "\n".join(r.getMessage() for r in caplog.records
                           if not r.name.startswith("websockets.client"))
    for secret in (token, token[:8], token[-8:], token_digest(token), GLOBAL):
        assert secret not in everything, "token material reached the log"


@pytest.mark.asyncio
async def test_a_client_retrying_every_minute_is_one_line_and_a_count(caplog, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(ws_server, "_reject_clock", lambda: clock[0])
    bridge, port = await _bridge()
    bad = json.dumps({"type": "auth", "token": "stale" + secrets.token_hex(6)})
    try:
        with caplog.at_level(logging.INFO, logger=LOGGER):
            for _ in range(5):                       # five retries, a minute apart
                await _attempt(port, bad)
                clock[0] += 60
            clock[0] += ws_server.REJECT_SUMMARY_SECONDS
            await _attempt(port, bad)                # past the window: a summary
    finally:
        await bridge.stop()

    lines = _rejections(caplog)
    assert len(lines) == 2, lines
    assert "4 more" in lines[1] and "(unknown token)" in lines[1]


@pytest.mark.asyncio
async def test_different_peers_or_reasons_are_never_folded_together(caplog, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(ws_server, "_reject_clock", lambda: clock[0])
    bridge, port = await _bridge()
    try:
        with caplog.at_level(logging.INFO, logger=LOGGER):
            await _attempt(port, json.dumps({"type": "auth", "token": "a" + secrets.token_hex(6)}))
            await _attempt(port, json.dumps({"type": "auth"}))
    finally:
        await bridge.stop()
    assert len(_rejections(caplog)) == 2
