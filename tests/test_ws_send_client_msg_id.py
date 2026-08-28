"""GRAFT-MOBILE-BRIDGE 8 — the WS ``send_message`` path drops ``client_msg_id``.

A bug, not a contract: ``_handle_send_message`` takes ``client_msg_id`` and
echoes it on the user ``chat_message`` frame, and the REST ``/api/chat/send``
path (``dispatch_user_message``) passes it through with a docstring saying so.
The WS command handler read ``session_id``, ``content``, ``mode`` and
``tool_choice`` from the payload — and never ``client_msg_id`` — so a socket
send always echoed ``null`` and a client's optimistic re-key could never match.

The load-bearing assertion here is EQUALITY with the value that was sent, not
presence of the key. A presence-only assertion passes against ``null``, which
is exactly how this survived: both Beacon clients' captures agreed with each
other because both recorded the same broken path.
"""

from __future__ import annotations

import json

import pytest

from prometheus.web.ws_server import WebSocketBridge


class _Recorder:
    """Fake WS client capturing every broadcast frame."""

    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    def user_echo(self) -> dict:
        return next(f["payload"] for f in self.frames if f["type"] == "chat_message")


class _FakeSession:
    """Just enough ChatSession for the echo path."""

    def __init__(self) -> None:
        self.messages: list = []

    def add_user_message(self, content: str) -> int:
        self.messages.append(content)
        return len(self.messages)

    def last_persisted_row_id(self) -> int:
        return 41


class _FakeMgr:
    def __init__(self) -> None:
        self.session = _FakeSession()

    def get(self, session_id: str):
        return self.session

    def get_or_create(self, session_id: str):
        return self.session


def _bridge() -> tuple[WebSocketBridge, _Recorder]:
    # loop_context=None: no agent dispatch — the echo is what is under test.
    bridge = WebSocketBridge(session_mgr=_FakeMgr(), loop_context=None)
    rec = _Recorder()
    bridge._clients.add(rec)
    return bridge, rec


def _send_cmd(payload: dict) -> str:
    return json.dumps({"type": "send_message", "payload": payload})


@pytest.mark.asyncio
async def test_ws_send_echoes_the_client_msg_id_that_was_sent():
    bridge, rec = _bridge()
    await bridge._handle_client_message(rec, _send_cmd({
        "session_id": "ios:t1", "content": "hello",
        "client_msg_id": "beacon-3f2a",
    }))
    echo = rec.user_echo()
    # EQUALS what was sent — not merely "key present". This is the test that
    # would have caught the bug.
    assert echo["client_msg_id"] == "beacon-3f2a"
    assert echo["message_id"] == 41


@pytest.mark.asyncio
async def test_ws_send_without_client_msg_id_still_echoes_null():
    bridge, rec = _bridge()
    await bridge._handle_client_message(rec, _send_cmd({
        "session_id": "ios:t2", "content": "hello",
    }))
    assert rec.user_echo()["client_msg_id"] is None


@pytest.mark.asyncio
async def test_ws_send_coerces_a_non_string_client_msg_id_to_null():
    # An arbitrary object must not reach the broadcast payload.
    bridge, rec = _bridge()
    await bridge._handle_client_message(rec, _send_cmd({
        "session_id": "ios:t3", "content": "hello",
        "client_msg_id": {"nested": "object"},
    }))
    assert rec.user_echo()["client_msg_id"] is None


@pytest.mark.asyncio
async def test_rest_dispatch_path_is_unchanged():
    # The REST path already carried it; pin the parity so the two paths cannot
    # drift apart again.
    bridge, rec = _bridge()
    await bridge.dispatch_user_message("ios:t4", "hello", client_msg_id="rest-77")
    assert rec.user_echo()["client_msg_id"] == "rest-77"
