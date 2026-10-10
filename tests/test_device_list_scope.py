"""``GET /api/devices`` shows a scoped device only itself.

#692 scoped a device to its own conversations and left the device LIST open (its contract, section 4: "still
lists every enrolled device"). That was written when the only holders of a device token were the operator's
own phones. Approval changes who holds one: Jennifer's MacBook, approved from Will's, is an ordinary scoped
device, and it could read the name, platform, last-seen time, push status, owner flag and owner source of
every other device on the daemon, including whether one is the owner's cockpit. The Beacon session found it
testing an approved device against the real daemon.

The rule now: an operator (the global token or an owner device) lists every device; a scoped device lists
ITSELF, so a client can still find its own row (``is_self``, its id, its push state) without learning who
else is enrolled. Returning only the caller's row, not a 403, keeps every client that asks "which one am I"
working. With auth off everyone is the operator, as everywhere else.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402
from tests.support.pairing_world import GLOBAL, World  # noqa: E402


@pytest.fixture
def world(tmp_path) -> World:
    w = World(tmp_path, recording=False)
    w.other = w.devices.mint("another scoped phone", "macos")      # type: ignore[attr-defined]
    return w


def test_a_scoped_device_lists_only_itself(world):
    rows = world.as_("scoped", "GET", "/api/devices").json()
    assert [r["id"] for r in rows] == [world.scoped["id"]]
    assert rows[0]["is_self"] is True and rows[0]["name"] == "a scoped phone"


def test_it_learns_nothing_about_the_others(world):
    response = world.as_("scoped", "GET", "/api/devices")
    for name in ("Beacon on this Mac", "another scoped phone"):
        assert name not in response.text
    assert world.owner["id"] not in response.text and world.other["id"] not in response.text


def test_a_device_the_owner_just_approved_lists_only_itself(world):
    created, requester, _ = world.created()
    world.approve(created)
    polled = world.poll(created).json()
    token = requester.unseal(created["request_id"], polled["sealed"])["token"]
    rows = world.client.get("/api/devices", headers={"Authorization": f"Bearer {token}"}).json()
    assert [r["id"] for r in rows] == [polled["device_id"]] and rows[0]["is_self"] is True


def test_an_owner_device_lists_every_device(world):
    rows = world.as_("owner", "GET", "/api/devices").json()
    assert {r["id"] for r in rows} == {world.owner["id"], world.scoped["id"], world.other["id"]}
    assert [r["id"] for r in rows if r["is_self"]] == [world.owner["id"]]


def test_the_global_token_lists_every_device(world):
    rows = world.as_("global", "GET", "/api/devices").json()
    assert {r["id"] for r in rows} == {world.owner["id"], world.scoped["id"], world.other["id"]}
    assert not any(r["is_self"] for r in rows)


def test_a_revoked_devices_row_is_not_shown_to_a_scoped_device(world):
    world.devices.revoke(world.other["id"])
    rows = world.as_("scoped", "GET", "/api/devices").json()
    assert [r["id"] for r in rows] == [world.scoped["id"]]


def test_with_auth_off_everyone_is_the_operator(tmp_path, monkeypatch):
    monkeypatch.delenv("PROMETHEUS_API_TOKEN", raising=False)
    devices = DeviceStore(tmp_path / "devices.db")
    devices.mint("one", "ios")
    devices.mint("two", "macos")
    rows = TestClient(create_app({"web": {}}, device_store=devices)).get("/api/devices").json()
    assert len(rows) == 2


def test_the_global_token_is_unaffected_by_the_scoped_rule(world):
    assert world.client.get("/api/devices", headers={"Authorization": f"Bearer {GLOBAL}"}).status_code == 200
