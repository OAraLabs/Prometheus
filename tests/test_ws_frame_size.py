"""Inbound WS frame size on the :8010 bridge (fix/ws-upload-max-size).

The websockets library's default ``max_size`` is 1 MiB. Every chat_upload
bigger than that — i.e. any real screenshot — was closed 1009 BEFORE
``_handle_client_message`` saw a byte, and ``_handler``'s bare ``except``
swallowed the close, so nothing surfaced anywhere (measured live from
beacon-desktop, 2026-08-28). These tests drive a REAL ``websockets`` client
against a REAL ``WebSocketBridge`` on an ephemeral port and pin the two
properties that matter:

  1. a multi-megabyte chat_upload actually REACHES the upload handler
     (this test fails on the pre-fix default max_size), and
  2. a frame over ``WS_MAX_FRAME_BYTES`` is still refused — but loudly:
     closed 1009 and logged, never silently dropped.

Plus the layering invariant: the library limit must sit ABOVE the app-level
20 MB decoded guard as base64, or that guard is dead code again.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging

import pytest

websockets = pytest.importorskip("websockets")
from websockets.exceptions import ConnectionClosed  # noqa: E402

from prometheus.web.ws_server import WS_MAX_FRAME_BYTES, WebSocketBridge  # noqa: E402


async def _start_bridge() -> tuple[WebSocketBridge, int]:
    bridge = WebSocketBridge()  # tokenless => auth disabled, like dev setups
    await bridge.start(host="127.0.0.1", port=0)
    port = bridge._server.sockets[0].getsockname()[1]
    return bridge, port


def _uri(port: int) -> str:
    return f"ws://127.0.0.1:{port}"


@pytest.mark.asyncio
async def test_multi_megabyte_chat_upload_reaches_the_handler():
    """A ~2 MB chat_upload must reach _handle_file_upload — with the library
    default (1 MiB) this exact frame was dropped before dispatch."""
    bridge, port = await _start_bridge()
    received: list[tuple[str, str, int]] = []

    async def _record(session_id, filename, content_b64, mime_type, caption):
        received.append((session_id, filename, len(content_b64)))

    bridge._handle_file_upload = _record  # type: ignore[method-assign]
    payload_b64 = base64.b64encode(b"\x89" * (2 * 1024 * 1024)).decode()
    try:
        async with websockets.connect(_uri(port), max_size=None) as ws:
            await ws.recv()  # the "connected" welcome
            await ws.send(json.dumps({
                "type": "chat_upload",
                "payload": {
                    "session_id": "beacon",
                    "filename": "big.png",
                    "content_base64": payload_b64,
                    "mime_type": "image/png",
                },
            }))
            for _ in range(50):
                if received:
                    break
                await asyncio.sleep(0.1)
    finally:
        await bridge.stop()

    assert received == [("beacon", "big.png", len(payload_b64))]


@pytest.mark.asyncio
async def test_over_limit_frame_is_closed_1009_and_logged(caplog):
    """Past WS_MAX_FRAME_BYTES the library still refuses — the fix is that the
    refusal is VISIBLE: close code 1009 to the client, a warning in the log."""
    bridge, port = await _start_bridge()
    try:
        with caplog.at_level(logging.WARNING, logger="prometheus.web.ws_server"):
            async with websockets.connect(_uri(port), max_size=None) as ws:
                await ws.recv()  # welcome
                await ws.send("x" * (WS_MAX_FRAME_BYTES + 1024))
                with pytest.raises(ConnectionClosed) as exc:
                    await asyncio.wait_for(ws.recv(), timeout=10.0)
            code = exc.value.rcvd.code if exc.value.rcvd is not None else None
            assert code == 1009
            # The handler's finally (and its log line) runs in the server task;
            # give it a beat rather than racing it.
            for _ in range(50):
                if any("1009" in r.message for r in caplog.records):
                    break
                await asyncio.sleep(0.1)
    finally:
        await bridge.stop()

    assert any(
        "1009" in r.getMessage() and "never processed" in r.getMessage()
        for r in caplog.records
    ), f"no over-limit warning logged; records: {[r.getMessage() for r in caplog.records]}"


def test_library_limit_sits_above_the_app_guard_as_base64():
    """_handle_file_upload's 20 MB decoded guard is reachable only if the
    library accepts its base64 form (+ JSON envelope). Shrink the constant
    below that and the guard is dead code again — this pins the layering."""
    app_guard_decoded = 20 * 1024 * 1024
    as_base64 = (app_guard_decoded * 4 + 2) // 3
    envelope_slack = 64 * 1024
    assert WS_MAX_FRAME_BYTES > as_base64 + envelope_slack
