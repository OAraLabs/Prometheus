"""The catch-all static mount must not turn a WS handshake into a 500.

The static files mount lives at ``/`` and is registered last, so it receives
every request no route claimed — including WebSocket handshakes aimed at the
wrong port. Upstream ``StaticFiles`` begins with
``assert scope["type"] == "http"``, so those raised AssertionError, surfaced as
an unhandled ASGI exception plus HTTP 500, and the client reconnected forever:
**11,915 tracebacks in two days** (~9/min) on 2026-08-02, which is enough to
make the log useless for spotting anything else.

These tests drive the ASGI app directly rather than through a client, because
the defect is in scope handling — the thing a client would hide behind a status
code.
"""

from __future__ import annotations

import pytest

from prometheus.web.server import _HttpOnlyStaticFiles


@pytest.fixture
def static_app(tmp_path):
    (tmp_path / "index.html").write_text("<h1>ok</h1>", encoding="utf-8")
    return _HttpOnlyStaticFiles(directory=str(tmp_path), html=True)


async def _drive(app, scope):
    """Run one ASGI cycle, returning the messages the app sent."""
    sent: list[dict] = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    return sent


@pytest.mark.asyncio
async def test_websocket_scope_is_closed_not_asserted(static_app):
    """A WS handshake must close cleanly — never raise, never 500."""
    scope = {"type": "websocket", "path": "/", "headers": []}
    try:
        sent = await _drive(static_app, scope)
    except AssertionError as exc:  # pragma: no cover - the bug being fixed
        pytest.fail(
            "the static mount asserted on a WebSocket scope instead of "
            f"declining it — this is the 500-and-retry loop: {exc!r}"
        )

    assert sent, "the app sent nothing; the client would hang"
    assert sent[0]["type"] == "websocket.close", (
        f"expected a clean websocket.close, got {sent[0]['type']!r}"
    )
    assert sent[0].get("code") == 1002, (
        "expected close code 1002 (protocol error) — the peer spoke WebSocket "
        "to an HTTP-only endpoint"
    )


@pytest.mark.asyncio
async def test_websocket_scope_on_a_deep_path_also_closes(static_app):
    """The catch-all sees every unmatched path, not just '/'."""
    for path in ("/ws", "/socket.io/", "/api/does-not-exist"):
        scope = {"type": "websocket", "path": path, "headers": []}
        sent = await _drive(static_app, scope)
        assert sent and sent[0]["type"] == "websocket.close", (
            f"WS to {path} did not close cleanly"
        )


@pytest.mark.asyncio
async def test_lifespan_scope_is_declined_quietly(static_app):
    """Any other non-HTTP scope must be declined, not asserted."""
    scope = {"type": "lifespan"}
    sent = await _drive(static_app, scope)
    assert sent == [], f"expected no messages for a lifespan scope, got {sent}"


@pytest.mark.asyncio
async def test_http_still_serves_the_file(static_app):
    """The whole point of the mount must keep working."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/index.html",
        "headers": [],
    }
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await static_app(scope, receive, send)

    starts = [m for m in sent if m["type"] == "http.response.start"]
    bodies = [m for m in sent if m["type"] == "http.response.body"]
    assert starts and starts[0]["status"] == 200, (
        f"static file did not serve: {starts}"
    )
    assert any(b"ok" in m.get("body", b"") for m in bodies), (
        "the response body did not contain the file contents"
    )
