"""The person mark: the one thing that lets a device resume a pause (Will, 2026-11-12).

It does nothing but allow resume. It is set and removed only from the owner's Telegram (``/person allow <device
id>``, ``/person disallow <device id>``); no REST route sets it, so neither the API token nor a device can. It is
not the computer-use mark: a device marked for computer use cannot resume, and a person-marked one gets no
computer-use power. A held device cannot be marked, and a revoked one never counts.

Why not "any owner device": the API token sits in a file a model with a shell could read on a host whose floors are
down, and with it a process on the daemon's machine can mint an owner device. The test that does exactly that is
here, and it cannot resume.
"""

from __future__ import annotations

import asyncio
import types

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.engine import pause  # noqa: E402
from prometheus.permissions import approver as _approver  # noqa: E402
from tests.support.device_world import World  # noqa: E402


@pytest.fixture
def state(tmp_path):
    s = pause.PauseState(tmp_path / "pause.json")
    pause.install(s)
    yield s
    pause.install(None)


@pytest.fixture
def w(tmp_path, state):
    world = World(tmp_path)
    world.owner = world.devices.mint_owner("owner phone", "ios", by="test")
    return world


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── the store ────────────────────────────────────────────────────────────────

def test_the_mark_is_set_cleared_and_never_counts_for_a_revoked_device(tmp_path):
    store = DeviceStore(tmp_path / "devices.db")
    d = store.mint("phone", "ios")
    assert store.person_allowed(d["id"]) is False                  # no table yet: no one
    assert store.set_person(d["id"], True, by="telegram:1") is True
    assert store.person_allowed(d["id"]) is True and store.person_device_ids() == {d["id"]}
    assert store.set_person(d["id"], False, by="telegram:1") is True
    assert store.person_allowed(d["id"]) is False
    store.set_person(d["id"], True, by="telegram:1")
    store.revoke(d["id"])
    assert store.person_allowed(d["id"]) is False
    assert store.set_person("nope", True, by="telegram:1") is False


def test_a_held_device_cannot_be_marked(tmp_path):
    store = DeviceStore(tmp_path / "devices.db")
    d = store.mint("server", "linux")
    assert store.set_profile(d["id"], "read-only", by="test")
    with pytest.raises(ValueError):
        store.set_person(d["id"], True, by="telegram:1")


def test_it_is_not_the_computer_use_mark(tmp_path):
    store = DeviceStore(tmp_path / "devices.db")
    d = store.mint("phone", "ios")
    store.set_computer(d["id"], True, by="telegram:1")
    assert store.person_allowed(d["id"]) is False
    e = store.mint("phone 2", "ios")
    store.set_person(e["id"], True, by="telegram:1")
    assert store.computer_allowed(e["id"]) is False


# ── /person, from the owner's Telegram only ─────────────────────────────────

def _tg(user_id: str, owners=("42",)):
    return _approver.telegram(types.SimpleNamespace(id=int(user_id), username="u"), owners=list(owners))


def test_the_owner_marks_and_unmarks_from_telegram(tmp_path):
    from prometheus.gateway.commands import cmd_person

    store = DeviceStore(tmp_path / "devices.db")
    d = store.mint("phone", "ios")
    reply = cmd_person(store, f"allow {d['id']}", by=_tg("42"))
    assert store.person_allowed(d["id"]) and d["id"] in reply
    listing = cmd_person(store, "", by=_tg("42"))
    assert d["id"] in listing
    cmd_person(store, f"disallow {d['id']}", by=_tg("42"))
    assert not store.person_allowed(d["id"])


@pytest.mark.parametrize("who", ["not-the-owner", "api-token", "device", "open"])
def test_nobody_else_can_mark(tmp_path, who):
    from prometheus.gateway.commands import cmd_person

    store = DeviceStore(tmp_path / "devices.db")
    d = store.mint("phone", "ios")
    by = {"not-the-owner": _tg("7"),
          "api-token": _approver.Approver(_approver.GLOBAL_TOKEN, "global", "API token"),
          "device": _approver.Approver(_approver.DEVICE, d["id"], "phone"),
          "open": _approver.Approver(_approver.OPEN, "none", "")}[who]
    reply = cmd_person(store, f"allow {d['id']}", by=by)
    assert not store.person_allowed(d["id"]), reply
    store.set_person(d["id"], True, by="telegram:42")
    cmd_person(store, f"disallow {d['id']}", by=by)
    assert store.person_allowed(d["id"]), who


def test_no_rest_route_sets_the_mark(w):
    paths = {getattr(r, "path", "") for r in w.app.routes}
    assert not [p for p in paths if "person" in p], paths
    w.call("op", "PUT", f"/api/devices/{w.owner['id']}/computer", json={"computer": True})
    assert not w.devices.person_allowed(w.owner["id"])


def test_telegram_registers_the_person_command():
    from pathlib import Path

    import prometheus.gateway.telegram as tg

    assert 'CommandHandler("person", self._cmd_person)' in Path(tg.__file__).read_text()


# ── resume needs the mark ────────────────────────────────────────────────────

def test_a_person_marked_device_resumes(w):
    w.call("op", "PUT", "/api/pause", json={})
    assert w.devices.set_person(w.owner["id"], True, by="telegram:42")
    r = w.client.delete("/api/pause", headers=_bearer(w.owner["token"]))
    assert r.status_code == 200 and r.json()["paused"] is False


def test_a_computer_marked_device_cannot_resume(w):
    w.call("op", "PUT", "/api/pause", json={})
    assert w.devices.set_computer(w.owner["id"], True, by="telegram:42")
    r = w.client.delete("/api/pause", headers=_bearer(w.owner["token"]))
    assert r.status_code == 403 and r.json()["error"] == "person_required"


def test_an_owner_device_without_the_mark_cannot_resume(w):
    w.call("op", "PUT", "/api/pause", json={})
    r = w.client.delete("/api/pause", headers=_bearer(w.owner["token"]))
    assert r.status_code == 403 and r.json()["error"] == "person_required"
    assert "/person allow" in r.json()["detail"]


def test_a_model_that_mints_its_own_owner_device_cannot_resume(w):
    """Read the token, mint an owner device on this machine, resume: three requests, and the third is refused."""
    w.call("op", "PUT", "/api/pause", json={})
    local = TestClient(w.app, client=("127.0.0.1", 50123))
    minted = local.post("/api/devices", headers=w.hdr("op"),
                        json={"name": "model-made", "platform": "ios", "owner": True})
    assert minted.status_code in (200, 201), minted.text
    r = local.delete("/api/pause", headers=_bearer(minted.json()["token"]))
    assert r.status_code == 403 and r.json()["error"] == "person_required"
    assert pause.current() is not None


def test_a_person_marked_device_that_is_later_held_cannot_resume(w):
    w.call("op", "PUT", "/api/pause", json={})
    assert w.devices.set_person(w.a["id"], True, by="telegram:42")
    assert w.devices.set_profile(w.a["id"], "read-only", by="test")
    r = w.call("a", "DELETE", "/api/pause")
    assert r.status_code == 403


def test_the_api_token_cannot_resume(w):
    w.call("op", "PUT", "/api/pause", json={})
    r = w.call("op", "DELETE", "/api/pause")
    assert r.status_code == 403 and r.json()["error"] == "person_required"


def test_cmd_person_is_synchronous_and_needs_no_runner():
    """The person mark works whether or not computer use is on (it has nothing to do with it)."""
    from prometheus.gateway.commands import cmd_person

    assert not asyncio.iscoroutinefunction(cmd_person)
