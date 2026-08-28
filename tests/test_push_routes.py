"""GRAFT-MOBILE-BRIDGE 2 — push/activity registration routes.

A device may manage only ITS OWN push registration (or the global operator
may): device A re-pointing device B's pushes would be a silent notification
hijack. Bad bodies are 400, unknown/revoked devices 404, and the APNs token
never appears in GET /api/devices.
"""

from __future__ import annotations

import secrets

import pytest

from prometheus.config.device_store import DeviceStore

GLOBAL = "glob-" + secrets.token_hex(8)


def _client(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from prometheus.web.server import create_app

    store = DeviceStore(tmp_path / "devices.db")
    app = create_app({"web": {"api_token": GLOBAL}}, device_store=store)
    return TestClient(app), store


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _reg(session=None):
    return {"apns_token": "a" * 64, "environment": "sandbox",
            "bundle_id": "com.oaralabs.beacon"}


def test_device_registers_its_own_push(tmp_path):
    client, store = _client(tmp_path)
    d = client.post("/api/devices", json={"name": "phone", "platform": "ios"},
                    headers=_auth(GLOBAL)).json()
    r = client.put(f"/api/devices/{d['id']}/push", json=_reg(),
                   headers=_auth(d["token"]))
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert [t.id for t in store.push_targets()] == [d["id"]]
    # And can unregister (notifications disabled).
    assert client.delete(f"/api/devices/{d['id']}/push",
                         headers=_auth(d["token"])).status_code == 200
    assert store.push_targets() == []


def test_a_device_cannot_manage_another_devices_push(tmp_path):
    client, _ = _client(tmp_path)
    a = client.post("/api/devices", json={"name": "a", "platform": "ios"},
                    headers=_auth(GLOBAL)).json()
    b = client.post("/api/devices", json={"name": "b", "platform": "ios"},
                    headers=_auth(GLOBAL)).json()
    assert client.put(f"/api/devices/{b['id']}/push", json=_reg(),
                      headers=_auth(a["token"])).status_code == 401
    assert client.post(f"/api/devices/{b['id']}/activity",
                       json={"session_id": "s", "activity_token": "t"},
                       headers=_auth(a["token"])).status_code == 401
    # The global operator may.
    assert client.put(f"/api/devices/{b['id']}/push", json=_reg(),
                      headers=_auth(GLOBAL)).status_code == 200


def test_bad_bodies_are_400_unknown_devices_404(tmp_path):
    client, _ = _client(tmp_path)
    d = client.post("/api/devices", json={"name": "phone", "platform": "ios"},
                    headers=_auth(GLOBAL)).json()
    bad = dict(_reg()); bad["environment"] = "prod"  # not a valid environment
    assert client.put(f"/api/devices/{d['id']}/push", json=bad,
                      headers=_auth(d["token"])).status_code == 400
    assert client.put("/api/devices/nope/push", json=_reg(),
                      headers=_auth(GLOBAL)).status_code == 404


def test_revoked_device_cannot_register_push(tmp_path):
    client, _ = _client(tmp_path)
    d = client.post("/api/devices", json={"name": "phone", "platform": "ios"},
                    headers=_auth(GLOBAL)).json()
    client.delete(f"/api/devices/{d['id']}", headers=_auth(GLOBAL))
    # Its token is dead (401) and even the global cannot re-arm a tombstone.
    assert client.put(f"/api/devices/{d['id']}/push", json=_reg(),
                      headers=_auth(d["token"])).status_code == 401
    assert client.put(f"/api/devices/{d['id']}/push", json=_reg(),
                      headers=_auth(GLOBAL)).status_code == 404


def test_listing_never_leaks_the_apns_token(tmp_path):
    client, _ = _client(tmp_path)
    d = client.post("/api/devices", json={"name": "phone", "platform": "ios"},
                    headers=_auth(GLOBAL)).json()
    client.put(f"/api/devices/{d['id']}/push", json=_reg(), headers=_auth(d["token"]))
    listing = client.get("/api/devices", headers=_auth(GLOBAL)).json()
    flat = str(listing)
    assert "a" * 64 not in flat and "apns" not in flat


def test_activity_token_round_trip(tmp_path):
    client, store = _client(tmp_path)
    d = client.post("/api/devices", json={"name": "phone", "platform": "ios"},
                    headers=_auth(GLOBAL)).json()
    client.put(f"/api/devices/{d['id']}/push", json=_reg(), headers=_auth(d["token"]))
    r = client.post(f"/api/devices/{d['id']}/activity",
                    json={"session_id": "s1", "activity_token": "act-1"},
                    headers=_auth(d["token"]))
    assert r.status_code == 200
    assert [(t.id, tok) for t, tok in store.activity_targets("s1")] == [(d["id"], "act-1")]
    assert client.request("DELETE", f"/api/devices/{d['id']}/activity",
                          json={"session_id": "s1"},
                          headers=_auth(d["token"])).status_code == 200
    assert store.activity_targets("s1") == []
