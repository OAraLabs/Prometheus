"""POST /api/chat/send — REST entry point that Beacon's chat surface needs.

Branch fix/api-chat-send. Mirrors Beacon's tRPC contract
(`src/server/routers/sessions.ts` ~line 145).
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402


class _RecordingBridge:
    """Captures dispatched messages without running an agent."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def dispatch_user_message(self, session_id: str, content: str) -> None:
        self.calls.append((session_id, content))


def _client_with_bridge(bridge=None):
    app = create_app({})
    if bridge is not None:
        app.state.ws_bridge = bridge
    return TestClient(app), app


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_dispatches_and_returns_run_id_and_status_sent():
    br = _RecordingBridge()
    client, _ = _client_with_bridge(br)

    resp = client.post(
        "/api/chat/send",
        json={
            "session_id": "telegram:8139235390",
            "message": "whats my name?",
            "idempotency_key": "test-key-1",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    # Matches Beacon's expected response shape (sessions.ts:158-160).
    assert body == {"run_id": "test-key-1", "status": "sent"}
    # Dispatched exactly once with the right args.
    assert br.calls == [("telegram:8139235390", "whats my name?")]


def test_missing_idempotency_key_returns_generated_run_id():
    br = _RecordingBridge()
    client, _ = _client_with_bridge(br)
    resp = client.post(
        "/api/chat/send",
        json={"session_id": "s1", "message": "hi"},
    )
    assert resp.status_code == 200
    rid = resp.json()["run_id"]
    assert rid and rid != "s1"  # generated, not echoed
    assert len(rid) == 32       # uuid4().hex
    assert br.calls == [("s1", "hi")]


def test_strips_whitespace_around_fields():
    br = _RecordingBridge()
    client, _ = _client_with_bridge(br)
    resp = client.post(
        "/api/chat/send",
        json={"session_id": "  s1  ", "message": "  hello  ", "idempotency_key": "k"},
    )
    assert resp.status_code == 200
    assert br.calls == [("s1", "hello")]


# --------------------------------------------------------------------------- #
# Fail-loud paths
# --------------------------------------------------------------------------- #


def test_no_bridge_wired_returns_503_not_silent_success():
    # Daemon launcher should set app.state.ws_bridge — if it didn't, the route
    # must NOT pretend the message was dispatched.
    client, _ = _client_with_bridge(bridge=None)
    resp = client.post("/api/chat/send", json={"session_id": "s", "message": "m"})
    assert resp.status_code == 503
    assert "ws_bridge" in resp.json()["error"]


def test_bridge_missing_dispatch_method_returns_503():
    class _Wrong:  # no dispatch_user_message
        pass

    client, _ = _client_with_bridge(_Wrong())
    resp = client.post("/api/chat/send", json={"session_id": "s", "message": "m"})
    assert resp.status_code == 503


def test_invalid_json_returns_400():
    client, _ = _client_with_bridge(_RecordingBridge())
    resp = client.post(
        "/api/chat/send",
        content=b"not-json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400


def test_non_object_body_returns_400():
    client, _ = _client_with_bridge(_RecordingBridge())
    resp = client.post("/api/chat/send", json=["not", "an", "object"])
    assert resp.status_code == 400


def test_empty_or_missing_session_id_returns_400():
    client, _ = _client_with_bridge(_RecordingBridge())
    for bad in [{}, {"message": "m"}, {"session_id": "", "message": "m"},
                {"session_id": "   ", "message": "m"}]:
        resp = client.post("/api/chat/send", json=bad)
        assert resp.status_code == 400, f"expected 400 for {bad}, got {resp.status_code}"


def test_empty_or_missing_message_returns_400():
    client, _ = _client_with_bridge(_RecordingBridge())
    for bad in [{"session_id": "s"}, {"session_id": "s", "message": ""},
                {"session_id": "s", "message": "   "}]:
        resp = client.post("/api/chat/send", json=bad)
        assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# Sanity — the WebSocketBridge's public dispatch alias points at the
# same flow the WS command uses, so adding HTTP doesn't fork behaviour.
# --------------------------------------------------------------------------- #


def test_ws_bridge_dispatch_method_calls_internal_handler():
    from prometheus.web.ws_server import WebSocketBridge

    br = WebSocketBridge()
    calls: list[tuple[str, str]] = []

    async def fake(session_id: str, content: str) -> None:
        calls.append((session_id, content))

    br._handle_send_message = fake  # type: ignore[assignment]
    asyncio.run(br.dispatch_user_message("sid", "msg"))
    assert calls == [("sid", "msg")]
