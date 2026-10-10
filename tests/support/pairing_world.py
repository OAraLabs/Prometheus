"""A daemon with an owner device, a scoped device, a fake clock and a recording notifier, for the pairing tests.

Real objects throughout: the FastAPI app, a real ``DeviceStore`` on a real SQLite file, the real instance key.
The only stand-ins are the clock (``runtime.clock``, so a five-minute TTL takes no time) and the notifier's
first listener, which records what the operator's channels would have been told.
"""

from __future__ import annotations

import secrets
import sqlite3

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from fastapi.testclient import TestClient

from prometheus.config import instance_key, pair_seal
from prometheus.config.device_store import DeviceStore
from prometheus.web.server import create_app

GLOBAL = "pair-test-global-" + secrets.token_hex(8)
T0 = 1_760_000_000.0
LOW_ORDER = pair_seal.b64url_encode(bytes(32))


class Clock:
    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class Requester:
    def __init__(self) -> None:
        self.private = X25519PrivateKey.generate()
        self.public = self.private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.public_b64 = pair_seal.b64url_encode(self.public)

    def unseal(self, request_id: str, sealed: dict) -> dict:
        return pair_seal.unseal_token(private_key=self.private, request_id=request_id, sealed=sealed)


class World:
    """A daemon with an owner device and a scoped device, a fake clock, and a recording notifier."""

    def __init__(self, tmp_path, *, recording: bool = True, bridge: bool = False, config: dict | None = None,
                 **pairing) -> None:
        self.path = tmp_path / "devices.db"
        self.devices = DeviceStore(self.path)
        full = {"web": {"api_token": GLOBAL}, **(config or {})}
        if pairing:
            full["pairing"] = pairing
        self.app = create_app(full, device_store=self.devices)
        self.clock = Clock()
        self.runtime = self.app.state.pairing
        self.runtime.clock = self.clock
        self.events: list[tuple[str, dict]] = []
        if recording:
            self.runtime.notifier.subscribe(lambda kind, payload: self.events.append((kind, payload)) or True)
        self.der = instance_key.ensure_instance_key()      # the daemon makes this at boot
        self.owner = self.devices.mint_owner("Beacon on this Mac", "macos", by="same-mac-pairing")
        self.scoped = self.devices.mint("a scoped phone", "ios")
        self.client = TestClient(self.app)
        self.bridge = None
        if bridge:                      # the real :8010 bridge, attached the way the launcher attaches it
            from prometheus.engine.session import SessionManager
            from prometheus.web.ws_server import WebSocketBridge

            self.bridge = WebSocketBridge(
                session_mgr=SessionManager(), api_token=GLOBAL, device_store=self.devices, loop_context=None)
            self.app.state.ws_bridge = self.bridge
            self.bridge.attach_pairing(self.runtime)

    def hdr(self, who: str) -> dict:
        token = {"global": GLOBAL, "owner": self.owner["token"], "scoped": self.scoped["token"]}.get(who, who)
        return {"Authorization": f"Bearer {token}"}

    def as_(self, who: str, method: str, url: str, **kw):
        headers = {**self.hdr(who), **kw.pop("headers", {})}
        return self.client.request(method, url, headers=headers, **kw)

    def request(self, requester=None, *, source="192.0.2.10", **body):
        requester = requester or Requester()
        payload = {"device_name": "Jennifer's MacBook", "platform": "macos", "public_key": requester.public_b64}
        payload.update(body)
        client = TestClient(self.app, client=(source, 50000))
        return client.post("/api/pair/requests", json=payload), requester, client

    def created(self, **kw):
        response, requester, client = self.request(**kw)
        assert response.status_code == 201, response.text
        return response.json(), requester, client

    def poll(self, created, client=None, secret=None, headers=None):
        merged = {"X-Pairing-Secret": secret if secret is not None else created["poll_secret"], **(headers or {})}
        return (client or self.client).get(f"/api/pair/requests/{created['request_id']}", headers=merged)

    def approve(self, created, who="global", **body):
        return self.as_(who, "POST", f"/api/pair/requests/{created['request_id']}/approve", json=body or None)

    def tables(self) -> set[str]:
        with sqlite3.connect(self.path) as conn:
            return {row[0] for row in conn.execute("select name from sqlite_master where type='table'")}
