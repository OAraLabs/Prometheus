"""Device scoping (REST) — a device token sees and manages only its own sessions.

Before this change every enrolled device token could list, read, rename, forget
and purge EVERY conversation on the daemon, and revoke every other device. The
rule now (docs/guide/api.md → "Device tokens and sessions"):

  * the operator's global token keeps full access, exactly as before;
  * a device owns the sessions it brought into existence (POST /api/sessions, or
    the first send to an id that exists nowhere and is not a daemon gateway
    namespace) and sees and manages only those;
  * a session nobody owns (a Telegram chat, anything that predates this change)
    is the operator's;
  * a device can revoke only itself.

Everything here goes through the real FastAPI app, the real WebSocketBridge, the
real SessionManager, the real LCM store and a real DeviceStore. Tokens are random
per-test values, never the real PROMETHEUS_API_TOKEN.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.engine.session import SessionManager  # noqa: E402
from prometheus.memory.lcm_conversation_store import LCMConversationStore  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402
from prometheus.web.ws_server import WebSocketBridge  # noqa: E402
from tests.support.device_world import (  # noqa: E402
    A_SECRET,
    TG_SECRET,
    World,
    engine_over,
)


@pytest.fixture
def world(tmp_path) -> World:
    return World(tmp_path)


def _ids(rows) -> set[str]:
    return {r["session_id"] for r in rows}


# --------------------------------------------------------------------------- #
# The leak, stated plainly: listing and reading
# --------------------------------------------------------------------------- #


def test_a_device_does_not_see_another_devices_session_in_the_list(world):
    sid_a = world.device_session("a")
    sid_b = world.device_session("b", text="b's own")
    tg = world.operator_session()

    assert _ids(world.call("a", "GET", "/api/sessions").json()) == {sid_a}
    assert _ids(world.call("b", "GET", "/api/sessions").json()) == {sid_b}
    # The operator sees all three, as before.
    assert _ids(world.call("op", "GET", "/api/sessions").json()) == {sid_a, sid_b, tg}


def test_a_device_cannot_read_another_devices_history(world):
    sid_a = world.device_session("a")

    mine = world.call("a", "GET", f"/api/sessions/{sid_a}/messages")
    assert mine.status_code == 200 and A_SECRET in mine.text

    theirs = world.call("b", "GET", f"/api/sessions/{sid_a}/messages")
    assert theirs.status_code == 404
    assert A_SECRET not in theirs.text


def test_a_device_cannot_read_an_operator_session(world):
    tg = world.operator_session()
    r = world.call("a", "GET", f"/api/sessions/{tg}/messages")
    assert r.status_code == 404
    assert TG_SECRET not in r.text
    assert TG_SECRET in world.call("op", "GET", f"/api/sessions/{tg}/messages").text


def test_a_foreign_session_looks_exactly_like_a_missing_one(world):
    """No existence oracle: for a device, 'not yours' and 'not there' are the same answer."""
    sid_a = world.device_session("a")
    foreign = world.call("b", "GET", f"/api/sessions/{sid_a}/messages")
    missing = world.call("b", "GET", "/api/sessions/desktop:never-existed/messages")
    assert (foreign.status_code, foreign.json()) == (missing.status_code, missing.json())


# --------------------------------------------------------------------------- #
# Every route that is keyed by a session id — found from the app, not listed by hand
# --------------------------------------------------------------------------- #


def _session_routes(app) -> list[tuple[str, str]]:
    """(METHOD, path template) for every route whose path carries {session_id}."""
    out = []
    for route in app.routes:
        path = getattr(route, "path", "")
        if "{session_id}" in path:
            for method in sorted(getattr(route, "methods", None) or ()):
                if method not in ("HEAD", "OPTIONS"):
                    out.append((method, path))
    return out


def _fill(path: str, sid: str) -> str:
    return re.sub(r"\{[a-z_]+\}", "x", path.replace("{session_id}", sid))


def test_the_session_route_sweep_finds_the_routes_we_think_it_does(world):
    """Guards the sweep below against silently sweeping nothing."""
    found = set(_session_routes(world.app))
    for expect in [
        ("GET", "/api/sessions/{session_id}/messages"),
        ("DELETE", "/api/sessions/{session_id}"),
        ("POST", "/api/sessions/{session_id}/purge"),
        ("PUT", "/api/sessions/{session_id}/title"),
        ("POST", "/api/sessions/{session_id}/fork"),
        ("GET", "/api/sessions/{session_id}/checkpoints"),
        ("PUT", "/api/sessions/{session_id}/workspace"),
        ("GET", "/api/lcm/{session_id}"),
    ]:
        assert expect in found, f"{expect} is not a session-keyed route any more?"
    assert len(found) >= 20


def test_every_session_keyed_route_refuses_a_foreign_device(world, monkeypatch):
    """The session guard, on its own. The route gate (web/route_access.py) now denies a scoped device most of
    these routes before the guard is asked, which would make this sweep pass for the wrong reason; so the gate
    is opened for it. Whatever is ever allowed to a scoped device, the guard still answers 404 to a foreign one."""
    monkeypatch.setattr("prometheus.web.server.is_scoped_allowed", lambda *a, **k: True)
    sid_a = world.device_session("a")
    for method, template in _session_routes(world.app):
        url = _fill(template, sid_a)
        r = world.call("b", method, url, json={})
        assert r.status_code == 404, (
            f"{method} {template}: device B reached device A's session "
            f"→ {r.status_code} {r.text[:120]}"
        )


def test_through_the_real_gate_a_foreign_device_gets_404_on_what_it_may_use_and_403_on_the_rest(world):
    """Either way it learns nothing about whose session it is: the answer does not depend on the session."""
    from prometheus.web.route_access import is_scoped_allowed

    sid_a = world.device_session("a")
    for method, template in _session_routes(world.app):
        url = _fill(template, sid_a)
        r = world.call("b", method, url, json={})
        if is_scoped_allowed(method, template.replace("{session_id}", "x")):
            assert r.status_code == 404, (method, template, r.status_code)
        else:
            assert r.status_code == 403 and r.json()["error"] == "operator_only", (method, template, r.status_code)


def test_every_session_keyed_route_still_admits_the_owner(world):
    """The guard must not be a blanket 404: A reaches the handler on all of them."""
    sid_a = world.device_session("a")
    for method, template in _session_routes(world.app):
        url = _fill(template, sid_a)
        r = world.call("a", method, url, json={})
        assert not (r.status_code == 404 and "unknown session" in r.text), (
            f"{method} {template}: the owner was refused by the session guard"
        )


def test_every_session_keyed_route_still_admits_the_operator(world):
    sid_a = world.device_session("a")
    for method, template in _session_routes(world.app):
        if method == "DELETE" and template == "/api/sessions/{session_id}":
            continue  # destroys the fixture; covered below
        r = world.call("op", method, _fill(template, sid_a), json={})
        assert not (r.status_code == 404 and "unknown session" in r.text), (
            f"{method} {template}: the operator was refused by the session guard"
        )


# --------------------------------------------------------------------------- #
# Mutations: refused AND nothing changed
# --------------------------------------------------------------------------- #


def test_a_foreign_device_cannot_forget_rename_or_pin_a_session(world):
    sid_a = world.device_session("a")
    world.call("a", "PUT", f"/api/sessions/{sid_a}/title", json={"title": "Alpha"})

    assert world.call("b", "DELETE", f"/api/sessions/{sid_a}").status_code == 404
    assert world.call("b", "PUT", f"/api/sessions/{sid_a}/title", json={"title": "pwned"}).status_code == 404
    assert world.call("b", "PUT", f"/api/sessions/{sid_a}/pin", json={"pinned": True}).status_code == 404

    row = next(r for r in world.call("a", "GET", "/api/sessions").json() if r["session_id"] == sid_a)
    assert row["title"] == "Alpha" and row["pinned"] is False, "B changed A's session"
    assert sid_a in _ids(world.call("a", "GET", "/api/sessions").json()), "B forgot A's session"
    assert sid_a in world.mgr._sessions


def test_a_foreign_device_cannot_purge_a_session(world, monkeypatch):
    sid_a = world.device_session("a")
    before = world.lcm.count_all(sid_a)
    assert before > 0

    # A purge is the operator's now (web/route_access.py): stopped at the gate.
    r = world.call("b", "POST", f"/api/sessions/{sid_a}/purge", json={"confirm": sid_a})
    assert r.status_code == 403 and r.json()["error"] == "operator_only"
    assert world.lcm.count_all(sid_a) == before, "B purged A's rows"
    # And behind the gate the session guard still refuses it.
    monkeypatch.setattr("prometheus.web.server.is_scoped_allowed", lambda *a, **k: True)
    r = world.call("b", "POST", f"/api/sessions/{sid_a}/purge", json={"confirm": sid_a})
    assert r.status_code == 404
    assert world.lcm.count_all(sid_a) == before, "B purged A's rows"


def test_the_owner_can_still_manage_its_own_session(world):
    sid_a = world.device_session("a")
    assert world.call("a", "PUT", f"/api/sessions/{sid_a}/title", json={"title": "Mine"}).status_code == 200
    assert world.call("a", "PUT", f"/api/sessions/{sid_a}/pin", json={"pinned": True}).status_code == 200
    assert world.call("a", "DELETE", f"/api/sessions/{sid_a}").status_code == 200


def test_the_operator_can_manage_any_session(world):
    sid_a = world.device_session("a")
    tg = world.operator_session()
    assert world.call("op", "PUT", f"/api/sessions/{sid_a}/title", json={"title": "Op"}).status_code == 200
    assert world.call("op", "DELETE", f"/api/sessions/{tg}").status_code == 200


# --------------------------------------------------------------------------- #
# Body-keyed routes: send / interrupt / fork target
# --------------------------------------------------------------------------- #


def test_a_device_cannot_send_into_another_devices_session(world):
    sid_a = world.device_session("a")
    before = world.lcm.count_all(sid_a)

    r = world.call("b", "POST", "/api/chat/send", json={"session_id": sid_a, "message": "injected"})
    assert r.status_code == 404
    assert world.lcm.count_all(sid_a) == before, "B wrote into A's conversation"


def test_a_device_cannot_send_into_an_operator_session(world):
    tg = world.operator_session()
    before = world.lcm.count_all(tg)
    r = world.call("a", "POST", "/api/chat/send", json={"session_id": tg, "message": "injected"})
    assert r.status_code == 404
    assert world.lcm.count_all(tg) == before


def test_a_device_cannot_interrupt_another_devices_turn(world):
    sid_a = world.device_session("a")
    stopped: list[str] = []
    world.bridge.interrupt_turn = lambda sid: stopped.append(sid) or True  # a turn is running

    mine = world.call("a", "POST", "/api/chat/interrupt", json={"session_id": sid_a})
    assert mine.json() == {"session_id": sid_a, "stopped": True}
    stopped.clear()

    theirs = world.call("b", "POST", "/api/chat/interrupt", json={"session_id": sid_a})
    # Same answer a quiet session gives (interrupt is idempotent) — and nothing was stopped.
    assert theirs.status_code == 200 and theirs.json() == {"session_id": sid_a, "stopped": False}
    assert stopped == []


def test_the_operator_can_interrupt_any_session(world):
    sid_a = world.device_session("a")
    world.bridge.interrupt_turn = lambda sid: True
    r = world.call("op", "POST", "/api/chat/interrupt", json={"session_id": sid_a})
    assert r.json()["stopped"] is True


def test_a_fork_belongs_to_the_device_that_made_it(world):
    sid_a = world.device_session("a")
    at = world.lcm.max_rowid(sid_a)

    # B cannot copy A's history out by forking it.
    stolen = world.call("b", "POST", f"/api/sessions/{sid_a}/fork", json={"at_rowid": at})
    assert stolen.status_code == 404

    mine = world.call("a", "POST", f"/api/sessions/{sid_a}/fork", json={"at_rowid": at})
    assert mine.status_code == 200, mine.text
    fork = mine.json()["session_id"]
    assert A_SECRET in world.call("a", "GET", f"/api/sessions/{fork}/messages").text
    assert world.call("b", "GET", f"/api/sessions/{fork}/messages").status_code == 404


def test_a_fork_cannot_be_aimed_at_a_session_the_device_does_not_own(world):
    """fork_session copies history INTO the target id; the target must be the caller's to take."""
    sid_a = world.device_session("a")
    tg = world.operator_session()
    at = world.lcm.max_rowid(sid_a)
    before = world.lcm.count_all(tg)

    r = world.call("a", "POST", f"/api/sessions/{sid_a}/fork",
                   json={"at_rowid": at, "session_id": tg})
    assert r.status_code == 404
    assert world.lcm.count_all(tg) == before, "A's history was copied into the operator's chat"


# --------------------------------------------------------------------------- #
# How a session becomes a device's
# --------------------------------------------------------------------------- #


def test_a_session_created_by_a_device_is_that_devices(world):
    sid = world.call("a", "POST", "/api/sessions").json()["session_id"]
    # Even before its first message the session is A's alone.
    assert world.call("a", "GET", f"/api/sessions/{sid}/messages").status_code == 200
    assert world.call("b", "GET", f"/api/sessions/{sid}/messages").status_code == 404


def test_a_session_created_by_the_operator_is_nobodys_device(world):
    sid = world.call("op", "POST", "/api/sessions").json()["session_id"]
    assert world.call("a", "GET", f"/api/sessions/{sid}/messages").status_code == 404
    assert world.call("op", "GET", f"/api/sessions/{sid}/messages").status_code == 200


def test_the_first_send_to_a_brand_new_id_claims_it(world):
    """Clients may mint their own ids (docs/guide/api.md uses "my-session")."""
    r = world.call("a", "POST", "/api/chat/send", json={"session_id": "my-session", "message": "hi"})
    assert r.status_code == 200
    assert world.call("a", "GET", "/api/sessions/my-session/messages").status_code == 200
    assert world.call("b", "GET", "/api/sessions/my-session/messages").status_code == 404
    # And B cannot now send into it.
    assert world.call("b", "POST", "/api/chat/send",
                      json={"session_id": "my-session", "message": "x"}).status_code == 404


def test_a_device_cannot_claim_an_existing_operator_session_by_sending_to_it(world):
    tg = world.operator_session()
    world.call("a", "POST", "/api/chat/send", json={"session_id": tg, "message": "mine now?"})
    assert world.call("a", "GET", f"/api/sessions/{tg}/messages").status_code == 404
    assert TG_SECRET in world.call("op", "GET", f"/api/sessions/{tg}/messages").text


@pytest.mark.parametrize("sid", ["telegram:555", "slack:C1", "discord:9", "cli:x", "api:x", "coding:t1"])
def test_a_device_cannot_pre_claim_a_namespace_the_daemon_writes_into(world, sid):
    """Otherwise a device could send to telegram:<operator chat id> before it exists and own it."""
    r = world.call("a", "POST", "/api/chat/send", json={"session_id": sid, "message": "squat"})
    assert r.status_code == 404
    assert world.call("a", "GET", f"/api/sessions/{sid}/messages").status_code == 404


def test_the_operator_can_use_a_daemon_namespace(world):
    r = world.call("op", "POST", "/api/chat/send", json={"session_id": "telegram:555", "message": "x"})
    assert r.status_code == 200


def test_a_write_to_a_brand_new_id_claims_it_but_a_read_never_does(world):
    """A client that mints its own id and configures it before its first message keeps
    working (a write claims); merely asking about an id must not take it (a read does not)."""
    # A read of an id that exists nowhere: 404, and nothing is claimed …
    assert world.call("a", "GET", "/api/sessions/ios:fresh/messages").status_code == 404
    assert world.devices.session_owner("ios:fresh") is None
    # … so B can still take it with a write …
    assert world.call("b", "PUT", "/api/sessions/ios:fresh/title", json={"title": "B's"}).status_code == 200
    assert world.devices.session_owner("ios:fresh") == world.b["id"]
    # … after which it is B's and nobody else's.
    assert world.call("b", "GET", "/api/sessions/ios:fresh/messages").status_code == 200
    assert world.call("a", "GET", "/api/sessions/ios:fresh/messages").status_code == 404
    assert world.call("a", "PUT", "/api/sessions/ios:fresh/title", json={"title": "A's"}).status_code == 404


def test_a_write_cannot_claim_a_daemon_namespace_or_an_existing_session_by_path(world):
    tg = world.operator_session()
    assert world.call("a", "PUT", "/api/sessions/telegram:777/title", json={"title": "x"}).status_code == 404
    assert world.call("a", "PUT", f"/api/sessions/{tg}/title", json={"title": "x"}).status_code == 404
    assert world.devices.session_owner("telegram:777") is None
    assert world.devices.session_owner(tg) is None


def test_a_sessions_owner_does_not_change_when_its_device_is_revoked(world):
    sid_a = world.device_session("a")
    assert world.call("op", "DELETE", f"/api/devices/{world.a['id']}").status_code == 200
    # B still cannot reach it, and the operator still can.
    assert world.call("b", "GET", f"/api/sessions/{sid_a}/messages").status_code == 404
    assert world.call("op", "GET", f"/api/sessions/{sid_a}/messages").status_code == 200


# --------------------------------------------------------------------------- #
# /api/chat (synchronous) lives in the web: namespace
# --------------------------------------------------------------------------- #


def test_post_api_chat_does_not_reach_into_another_devices_web_session(world):
    # A's web conversation is web:<id>; B naming the same <id> would land in it.
    sid = "shared-name"
    world.call("a", "POST", "/api/chat/send", json={"session_id": f"web:{sid}", "message": A_SECRET})
    # No agent loop is wired in this world, so the route 503s AFTER the scope check.
    # What must not happen is B getting past the check into A's session.
    r = world.call("b", "POST", "/api/chat", json={"session_id": sid, "content": "injected"})
    assert r.status_code == 404, r.text


# --------------------------------------------------------------------------- #
# Devices: a device can revoke only itself
# --------------------------------------------------------------------------- #


def test_a_device_cannot_revoke_another_device(world):
    r = world.call("b", "DELETE", f"/api/devices/{world.a['id']}")
    assert r.status_code == 403
    assert "only revoke itself" in r.json()["error"]
    # A is still enrolled and still authenticates.
    assert world.call("a", "GET", "/api/sessions").status_code == 200


def test_a_device_can_revoke_itself(world):
    r = world.call("b", "DELETE", f"/api/devices/{world.b['id']}")
    assert r.status_code == 200 and r.json() == {"ok": True, "id": world.b["id"]}
    # Revocation is a tombstone: B's next request fails auth; A is untouched.
    assert world.call("b", "GET", "/api/sessions").status_code == 401
    assert world.call("a", "GET", "/api/sessions").status_code == 200


def test_the_operator_can_revoke_any_device(world):
    assert world.call("op", "DELETE", f"/api/devices/{world.a['id']}").status_code == 200
    assert world.call("a", "GET", "/api/sessions").status_code == 401
    assert world.call("op", "DELETE", "/api/devices/no-such-device").status_code == 404


def test_a_device_naming_an_unknown_device_gets_the_same_refusal(world):
    """No oracle for device ids either: unknown and somebody-else's are one answer."""
    other = world.call("b", "DELETE", f"/api/devices/{world.a['id']}")
    unknown = world.call("b", "DELETE", "/api/devices/ffffffffffffffffffffffffffffffff")
    assert (other.status_code, other.json()["error"]) == (unknown.status_code, unknown.json()["error"])


# --------------------------------------------------------------------------- #
# Auth off: nothing changes for a deliberately open daemon
# --------------------------------------------------------------------------- #


def test_with_auth_off_every_session_is_open_as_before(tmp_path):
    lcm = LCMConversationStore(tmp_path / "lcm.db")
    engine = engine_over(lcm)
    mgr = SessionManager()
    mgr.lcm_engine = engine
    app = create_app({}, session_mgr=mgr, lcm_engine=engine,
                     device_store=DeviceStore(tmp_path / "devices.db"))
    app.state.ws_bridge = WebSocketBridge(session_mgr=mgr, loop_context=None)
    client = TestClient(app)

    sid = client.post("/api/sessions").json()["session_id"]
    assert client.post("/api/chat/send", json={"session_id": sid, "message": "hello"}).status_code == 200
    assert client.get(f"/api/sessions/{sid}/messages").status_code == 200
    assert sid in _ids(client.get("/api/sessions").json())
    assert client.delete(f"/api/devices/{'x' * 32}").status_code == 404  # unknown id, not 403

