"""GRAFT-MOBILE-BRIDGE 1 — per-device tokens: mint, authenticate, revoke.

The acceptance criteria, from the spec:

  1. POST /api/devices with the global token → 201, token returned once.
  2. That token authenticates REST and the WS first-frame; GET /api/devices
     shows is_self on its own row.
  3. POST /api/devices with a DEVICE token → 401 (no privilege escalation —
     a stolen phone cannot enrol a second attacker device).
  4. DELETE the device → its next REST call is 401 and its next WS connect
     closes 4401.
  5. The global token still works everywhere; a daemon with no token set is
     still open.
  6. Revoking one device does not affect any other device or the global token.

Tokens here are random per-test values, never the real PROMETHEUS_API_TOKEN.
The store is always an explicit tmp-path instance — bare create_app must not
touch the real data dir, and these tests would catch it doing so only by
polluting it, so they never rely on the default path.
"""

from __future__ import annotations

import asyncio
import json
import secrets

import pytest

from prometheus.config.api_token import GLOBAL_IDENTITY, verify_token
from prometheus.config.device_store import (
    LAST_SEEN_THROTTLE_SECONDS,
    DeviceStore,
    token_digest,
)

GLOBAL = "glob-" + secrets.token_hex(8)


def _store(tmp_path) -> DeviceStore:
    return DeviceStore(tmp_path / "devices.db")


def _client(tmp_path, store=None, token=GLOBAL):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from prometheus.web.server import create_app

    store = store if store is not None else _store(tmp_path)
    app = create_app({"web": {"api_token": token}} if token else {},
                     device_store=store)
    return TestClient(app), store


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------- #
# verify_token
# --------------------------------------------------------------------------- #


def test_verify_token_global_and_device_and_miss(tmp_path):
    store = _store(tmp_path)
    minted = store.mint("Will's iPhone", "ios")

    assert verify_token(GLOBAL, GLOBAL, store) == GLOBAL_IDENTITY
    ident = verify_token(minted["token"], GLOBAL, store)
    assert ident is not None and ident.id == minted["id"] and not ident.is_global
    assert verify_token("nope", GLOBAL, store) is None
    assert verify_token("", GLOBAL, store) is None
    # No registry wired → device tokens are simply unknown.
    assert verify_token(minted["token"], GLOBAL, None) is None


def test_token_is_stored_only_as_a_digest(tmp_path):
    store = _store(tmp_path)
    minted = store.mint("phone", "ios")
    row = store.lookup(token_digest(minted["token"]))
    assert row is not None
    # The plaintext appears nowhere in the database file.
    blob = (tmp_path / "devices.db").read_bytes()
    assert minted["token"].encode() not in blob


def test_last_seen_stamp_is_throttled(tmp_path):
    store = _store(tmp_path)
    minted = store.mint("phone", "ios")
    store.touch(minted["id"])
    first = next(d for d in store.list_devices() if d.id == minted["id"]).last_seen_at
    store.touch(minted["id"])  # inside the window → no write
    again = next(d for d in store.list_devices() if d.id == minted["id"]).last_seen_at
    assert first == again
    # Age the throttle memory; the next touch writes.
    store._last_touch[minted["id"]] -= LAST_SEEN_THROTTLE_SECONDS + 1
    store.touch(minted["id"])
    later = next(d for d in store.list_devices() if d.id == minted["id"]).last_seen_at
    assert later > first


# --------------------------------------------------------------------------- #
# REST — acceptance 1, 2 (REST half), 3, 4 (REST half), 5, 6
# --------------------------------------------------------------------------- #


def test_mint_with_global_token_returns_token_once(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post("/api/devices", json={"name": "Will's iPhone", "platform": "ios"},
                    headers=_auth(GLOBAL))
    assert r.status_code == 201
    body = r.json()
    assert body["token"] and body["id"] and body["platform"] == "ios"
    # Never again: the listing carries no token field for any row.
    listed = client.get("/api/devices", headers=_auth(GLOBAL)).json()
    assert all("token" not in row for row in listed)


def test_device_token_authenticates_rest_and_is_self(tmp_path):
    client, _ = _client(tmp_path)
    minted = client.post("/api/devices", json={"name": "phone", "platform": "ios"},
                         headers=_auth(GLOBAL)).json()
    listed = client.get("/api/devices", headers=_auth(minted["token"]))
    assert listed.status_code == 200
    rows = {r["id"]: r for r in listed.json()}
    assert rows[minted["id"]]["is_self"] is True
    # The global identity is nobody's row.
    for row in client.get("/api/devices", headers=_auth(GLOBAL)).json():
        assert row["is_self"] is False


def test_device_token_cannot_mint(tmp_path):
    client, _ = _client(tmp_path)
    minted = client.post("/api/devices", json={"name": "phone", "platform": "ios"},
                         headers=_auth(GLOBAL)).json()
    r = client.post("/api/devices", json={"name": "attacker", "platform": "other"},
                    headers=_auth(minted["token"]))
    assert r.status_code == 401


def test_revoked_device_is_401_and_others_unaffected(tmp_path):
    client, _ = _client(tmp_path)
    a = client.post("/api/devices", json={"name": "a", "platform": "ios"},
                    headers=_auth(GLOBAL)).json()
    b = client.post("/api/devices", json={"name": "b", "platform": "macos"},
                    headers=_auth(GLOBAL)).json()

    # A phone revoking itself is the designed path — any valid token may revoke.
    r = client.delete(f"/api/devices/{a['id']}", headers=_auth(a["token"]))
    assert r.status_code == 200 and r.json() == {"ok": True, "id": a["id"]}

    assert client.get("/api/devices", headers=_auth(a["token"])).status_code == 401
    assert client.get("/api/devices", headers=_auth(b["token"])).status_code == 200
    assert client.get("/api/devices", headers=_auth(GLOBAL)).status_code == 200
    revoked_row = {r["id"]: r for r in
                   client.get("/api/devices", headers=_auth(GLOBAL)).json()}[a["id"]]
    assert revoked_row["revoked_at"] is not None


def test_delete_unknown_device_is_404(tmp_path):
    client, _ = _client(tmp_path)
    assert client.delete("/api/devices/no-such-id",
                         headers=_auth(GLOBAL)).status_code == 404


def test_open_daemon_stays_open(tmp_path):
    client, _ = _client(tmp_path, token="")
    # No token configured → auth off; the device surface works unauthenticated.
    r = client.post("/api/devices", json={"name": "open", "platform": "other"})
    assert r.status_code == 201
    assert client.get("/api/devices").status_code == 200


def test_wrong_bearer_is_still_401(tmp_path):
    client, _ = _client(tmp_path)
    assert client.get("/api/devices", headers=_auth("wrong")).status_code == 401
    assert client.get("/api/devices").status_code == 401


# --------------------------------------------------------------------------- #
# WS — acceptance 2 and 4 (socket half), against a real bridge + real client
# --------------------------------------------------------------------------- #

def test_bridge_auth_records_identity_and_rejects_revoked(tmp_path):
    # No websockets package needed: drive _authenticate with a fake socket.
    # The real-socket twin below runs where websockets is installed.
    from prometheus.web.ws_server import WebSocketBridge

    class _FakeWS:
        def __init__(self, frame: str) -> None:
            self._frame = frame
            self.closed_with: int | None = None

        async def recv(self) -> str:
            return self._frame

        async def close(self, code: int, reason: str = "") -> None:
            self.closed_with = code

    async def run():
        store = _store(tmp_path)
        minted = store.mint("phone", "ios")
        bridge = WebSocketBridge(api_token=GLOBAL, device_store=store)

        ws = _FakeWS(json.dumps({"type": "auth", "token": minted["token"]}))
        assert await bridge._authenticate(ws) is True
        assert bridge._ws_identity[ws].id == minted["id"]

        store.revoke(minted["id"])
        ws2 = _FakeWS(json.dumps({"type": "auth", "token": minted["token"]}))
        assert await bridge._authenticate(ws2) is False
        assert ws2.closed_with == 4401
        assert ws2 not in bridge._ws_identity

        ws3 = _FakeWS(json.dumps({"type": "auth", "token": GLOBAL}))
        assert await bridge._authenticate(ws3) is True
        assert bridge._ws_identity[ws3].is_global

    asyncio.run(run())


def test_ws_device_token_authenticates_and_revocation_closes_4401(tmp_path):
    # importorskip stays INSIDE the test: at module level it would skip the
    # REST tests above with it on a websockets-less environment.
    websockets = pytest.importorskip("websockets")
    from websockets.exceptions import ConnectionClosed

    from prometheus.web.ws_server import WS_CLOSE_UNAUTHORIZED, WebSocketBridge

    async def _ws_roundtrip(port: int, token: str):
        """Auth with `token`; return ('connected', ...) or ('closed', code)."""
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps({"type": "auth", "token": token}))
            try:
                frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                return ("connected", frame["type"])
            except ConnectionClosed as exc:
                return ("closed", exc.rcvd.code if exc.rcvd else None)

    async def run():
        store = _store(tmp_path)
        minted = store.mint("phone", "ios")
        bridge = WebSocketBridge(api_token=GLOBAL, device_store=store)
        await bridge.start(host="127.0.0.1", port=0)
        port = bridge._server.sockets[0].getsockname()[1]
        try:
            assert await _ws_roundtrip(port, minted["token"]) == ("connected", "connected")
            assert await _ws_roundtrip(port, GLOBAL) == ("connected", "connected")

            store.revoke(minted["id"])
            status, code = await _ws_roundtrip(port, minted["token"])
            assert (status, code) == ("closed", WS_CLOSE_UNAUTHORIZED)
            # The global token is untouched by a device revocation.
            assert await _ws_roundtrip(port, GLOBAL) == ("connected", "connected")
        finally:
            await bridge.stop()

    asyncio.run(run())
