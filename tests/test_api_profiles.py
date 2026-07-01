"""GET/PUT /api/profiles — the endpoints exist and the desktop profile-switch UI
is built, but the daemon never passed a profile_store, so /api/profiles returned
[] and the UI was dead (Beacon D1). These pin the wire: a store makes the real
profiles flow, and the no-store path reproduces the old dead behavior.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from prometheus.config.profiles import get_profile_store
from prometheus.web.server import create_app


@pytest.fixture
def wired():
    return TestClient(create_app({}, profile_store=get_profile_store()))


def test_profiles_flow_when_store_is_wired(wired):
    resp = wired.get("/api/profiles")
    assert resp.status_code == 200
    profiles = resp.json()
    names = {p["name"] for p in profiles}
    assert {"full", "coder", "minimal", "research"} <= names
    assert all({"name", "description", "is_active"} <= p.keys() for p in profiles)
    # default active profile is "full" (config default)
    assert [p for p in profiles if p["is_active"]] == [
        p for p in profiles if p["name"] == "full"
    ]


def test_set_active_profile_switches(wired):
    assert wired.put("/api/profiles/active", json={"name": "coder"}).status_code == 200
    profiles = wired.get("/api/profiles").json()
    active = [p["name"] for p in profiles if p["is_active"]]
    assert active == ["coder"]


def test_set_unknown_profile_404s(wired):
    assert wired.put("/api/profiles/active", json={"name": "nope"}).status_code == 404


def test_no_store_returns_empty_the_old_dead_behavior():
    # Reproduces the pre-D1 daemon: no profile_store → [] → dead UI.
    client = TestClient(create_app({}))
    resp = client.get("/api/profiles")
    assert resp.status_code == 200
    assert resp.json() == []
