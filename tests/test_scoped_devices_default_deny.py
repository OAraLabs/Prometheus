"""A scoped device is a chat client for its own conversations, and nothing else.

THE HOLE (an independent review of #692-#699, verified)
-------------------------------------------------------
``POST /api/pair/requests`` hands a new device a *scoped* token. #692 scoped what that token can see in the
session routes, and left every other route open to any valid token. So an approved phone could:

* list every session's pending tool calls (``GET /api/approvals``), approve any of them with the scope
  ``always`` (a persistent grant), and delete grants;
* create, change and run cron jobs. A cron job is not tagged origin ``model``, so ``execute_job`` runs it
  through ``/bin/bash`` outside the shell floor, with the daemon's unscrubbed environment, which holds the
  global token;
* overwrite provider API keys.

THE RULE
--------
Default-deny. The bearer middleware answers ``403 operator_only`` for a scoped token on any route that is not
in ``web/route_access.SCOPED_ALLOWED``. That one allowlist is: hello, its own device, chat, its own sessions,
and approving or denying the tool calls of ITS OWN sessions with the scope ``once`` or ``until-restart``.

This file is the behaviour: the exploits, and the allowed approvals, through the real app. The table
(every registered route classified exactly once) is ``tests/test_route_access.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.permissions.approval_queue import (  # noqa: E402
    ApprovalQueue,
    ApprovalResult,
    PendingAction,
)
from prometheus.permissions.checker import PermissionMode, SecurityGate  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

GLOBAL = "scoped-deny-global-" + secrets.token_hex(8)

ALICE_SESSION = "web:alice-1"
BOB_SESSION = "web:bob-1"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Nothing here may touch the real ~/.prometheus (cron, provider keys and the device registry live there)."""
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "config"))


class Daemon:
    """The real app, a real device registry, a real approval queue with a real gate."""

    def __init__(self, tmp_path) -> None:
        self.devices = DeviceStore(tmp_path / "devices.db")
        self.app = create_app({"web": {"api_token": GLOBAL}}, device_store=self.devices)
        self.gate = SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None)
        self.queue = ApprovalQueue(security_gate=self.gate)
        self.app.state.approval_queue = self.queue
        self.owner = self.devices.mint_owner("Beacon on this Mac", "macos", by="same-mac-pairing")
        self.alice = self.devices.mint("alice's phone", "ios")
        self.bob = self.devices.mint("bob's phone", "ios")
        self.devices.claim_session(ALICE_SESSION, self.alice["id"])
        self.devices.claim_session(BOB_SESSION, self.bob["id"])
        self.client = TestClient(self.app, client=("127.0.0.1", 50123))

    def as_(self, who: str, method: str, url: str, **kw):
        token = {"global": GLOBAL, "owner": self.owner["token"], "alice": self.alice["token"],
                 "bob": self.bob["token"]}[who]
        return self.client.request(method, url, headers={"Authorization": f"Bearer {token}"}, **kw)

    def pend(self, request_id: str, *, session: str | None, command: str = "echo hi") -> PendingAction:
        action = PendingAction(request_id=request_id, tool_name="bash", description=f"run {command}",
                               grant_command=command, session_id=session)
        self.queue.pending[request_id] = action
        return action

    def waiting(self) -> set[str]:
        return set(self.queue.pending)


@pytest.fixture
def d(tmp_path) -> Daemon:
    return Daemon(tmp_path)


def operator_only(response) -> bool:
    return response.status_code == 403 and response.json().get("error") == "operator_only"


# ── H2: cron, provider keys, config and the rest of the operator's surface ────

@pytest.mark.parametrize("method, url, body", [
    ("GET", "/api/cron", None),
    ("POST", "/api/cron", {"name": "x", "schedule": "* * * * *", "command": "id"}),
    ("PUT", "/api/cron/x", {"command": "id"}),
    ("POST", "/api/cron/x/run", None),
    ("DELETE", "/api/cron/x", None),
    ("GET", "/api/providers/keys", None),
    ("PUT", "/api/providers/keys/openai", {"key": "sk-not-real"}),
    ("GET", "/api/config", None),
    ("POST", "/api/mcp/servers", {"name": "x", "command": "id"}),
    ("PUT", "/api/memory/current", {"content": "x"}),
    ("GET", "/api/files/read", None),
    ("POST", "/api/code", {"repo": "/tmp"}),
    ("POST", "/api/computer/tasks", {"goal": "x"}),
    ("PUT", "/api/devices/x/computer", {"allowed": True}),
    ("POST", "/api/devices", {"name": "x"}),
    ("GET", "/api/status", None),
])
def test_a_scoped_device_is_refused_the_operators_surface(d, method, url, body):
    response = d.as_("alice", method, url, **({"json": body} if body is not None else {}))
    assert operator_only(response), (method, url, response.status_code, response.text[:120])


def test_a_refused_cron_job_was_not_created(d):
    d.as_("alice", "POST", "/api/cron", json={"name": "evil", "schedule": "* * * * *", "command": "id"})
    listed = d.as_("global", "GET", "/api/cron")
    assert listed.status_code == 200 and "evil" not in listed.text


def test_the_operator_surface_still_answers_the_operator(d):
    for who in ("global", "owner"):
        assert d.as_(who, "GET", "/api/status").status_code == 200, who


def test_with_the_token_off_everyone_is_the_operator(tmp_path):
    app = create_app({"web": {"api_token": ""}}, device_store=DeviceStore(tmp_path / "off.db"))
    assert TestClient(app).get("/api/status").status_code == 200


# ── default-deny is a property of the gate, not of a list of bad routes ──────

def test_a_route_nobody_wrote_is_denied_to_a_scoped_device_not_missing(d):
    assert operator_only(d.as_("alice", "GET", "/api/a-route-added-tomorrow"))
    assert d.as_("global", "GET", "/api/a-route-added-tomorrow").status_code == 404


@pytest.mark.parametrize("method, url", [
    ("HEAD", "/api/cron"),          # Starlette serves HEAD wherever it serves GET
    ("GET", "/api/cron/"),          # the trailing-slash redirect target
    ("GET", "/v1/models"),
    ("POST", "/v1/chat/completions"),
])
def test_the_variants_of_a_denied_route_are_denied_too(d, method, url):
    response = d.as_("alice", method, url)
    assert response.status_code == 403, (method, url, response.status_code)


def test_the_openai_surface_is_not_a_back_door(d):
    assert operator_only(d.as_("alice", "POST", "/v1/chat/completions", json={"model": "x", "messages": []}))


def test_what_a_scoped_device_is_meant_to_use_still_works(d):
    own = d.as_("alice", "GET", "/api/devices")
    assert own.status_code == 200 and [x["id"] for x in own.json()] == [d.alice["id"]]
    assert d.as_("alice", "GET", "/api/sessions").status_code == 200
    assert d.as_("alice", "GET", "/api/approvals").status_code == 200
    assert d.as_("alice", "PUT", f"/api/devices/{d.alice['id']}/push", json={}).status_code == 400, "reached the handler"
    assert d.client.get("/api/hello").status_code == 200


# ── H1: approvals ────────────────────────────────────────────────────────────

def test_a_scoped_device_sees_only_the_tool_calls_of_its_own_sessions(d):
    d.pend("aaaa0001", session=ALICE_SESSION)
    d.pend("bbbb0001", session=BOB_SESSION)
    d.pend("cccc0001", session=None)                       # raised outside any session: nobody's but the operator's
    d.pend("dddd0001", session="telegram:42")              # a daemon surface
    seen = {a["request_id"] for a in d.as_("alice", "GET", "/api/approvals").json()}
    assert seen == {"aaaa0001"}
    for who in ("global", "owner"):
        assert {a["request_id"] for a in d.as_(who, "GET", "/api/approvals").json()} == {
            "aaaa0001", "bbbb0001", "cccc0001", "dddd0001"}, who


def test_a_scoped_device_cannot_approve_another_sessions_tool_call(d):
    d.pend("bbbb0001", session=BOB_SESSION)
    response = d.as_("alice", "POST", "/api/approvals/bbbb0001/approve", json={"scope": "once"})
    assert response.status_code == 404, "not yours looks exactly like not there"
    assert d.waiting() == {"bbbb0001"} and d.queue.pending["bbbb0001"]._result == ApprovalResult.TIMEOUT


@pytest.mark.parametrize("request_id", ["cccc0001", "dddd0001"])
def test_nor_one_that_belongs_to_no_session_it_owns(d, request_id):
    d.pend("cccc0001", session=None)
    d.pend("dddd0001", session="telegram:42")
    assert d.as_("alice", "POST", f"/api/approvals/{request_id}/approve", json={}).status_code == 404
    assert d.as_("alice", "POST", f"/api/approvals/{request_id}/deny").status_code == 404
    assert d.waiting() == {"cccc0001", "dddd0001"}


@pytest.mark.parametrize("scope", ["always", "always here", "until-restart here"])
def test_a_scoped_device_cannot_make_a_persistent_or_widened_grant(d, scope):
    d.pend("aaaa0001", session=ALICE_SESSION)
    response = d.as_("alice", "POST", "/api/approvals/aaaa0001/approve", json={"scope": scope})
    assert operator_only(response), (scope, response.status_code, response.text[:100])
    assert d.waiting() == {"aaaa0001"} and d.gate.list_grants() == []


def test_the_scope_may_not_be_smuggled_in_through_the_request_id(d):
    """The handler builds "<scope> <request_id>" and hands the string to the command parser, so an id with a space
    in it is a second argument. A scoped caller's id must be exactly a pending request's id."""
    d.pend("aaaa0001", session=ALICE_SESSION)
    for smuggled in ("always%20aaaa0001", "always%20all", "all", "aaaa0001%20always"):
        response = d.as_("alice", "POST", f"/api/approvals/{smuggled}/approve", json={"scope": "once"})
        assert response.status_code == 404, (smuggled, response.status_code)
    assert d.waiting() == {"aaaa0001"} and d.gate.list_grants() == []


def test_a_scoped_device_cannot_approve_everything_at_once(d):
    d.pend("aaaa0001", session=ALICE_SESSION)
    d.pend("bbbb0001", session=BOB_SESSION)
    assert d.as_("alice", "POST", "/api/approvals/all/approve", json={}).status_code == 404
    assert d.waiting() == {"aaaa0001", "bbbb0001"}


@pytest.mark.parametrize("scope", ["once", "until-restart"])
def test_a_scoped_device_may_approve_its_own_sessions_tool_call_once_or_until_restart(d, scope):
    action = d.pend("aaaa0001", session=ALICE_SESSION)
    response = d.as_("alice", "POST", "/api/approvals/aaaa0001/approve", json={"scope": scope})
    assert response.status_code == 200 and response.json()["ok"] is True, response.text
    assert d.waiting() == set() and action._result == ApprovalResult.APPROVED


def test_the_default_scope_is_once(d):
    d.pend("aaaa0001", session=ALICE_SESSION)
    assert d.as_("alice", "POST", "/api/approvals/aaaa0001/approve").status_code == 200
    assert d.gate.list_grants() == []


def test_a_scoped_device_may_deny_its_own_sessions_tool_call(d):
    action = d.pend("aaaa0001", session=ALICE_SESSION)
    response = d.as_("alice", "POST", "/api/approvals/aaaa0001/deny")
    assert response.status_code == 200 and response.json() == {"ok": True}
    assert action._result == ApprovalResult.DENIED


def test_nor_deny_anothers(d):
    d.pend("bbbb0001", session=BOB_SESSION)
    assert d.as_("alice", "POST", "/api/approvals/bbbb0001/deny").status_code == 404
    assert d.waiting() == {"bbbb0001"}


def test_grants_belong_to_the_operator(d):
    d.pend("aaaa0001", session=ALICE_SESSION)
    assert d.as_("owner", "POST", "/api/approvals/aaaa0001/approve", json={"scope": "always"}).status_code == 200
    grants = d.as_("global", "GET", "/api/approvals/grants").json()
    assert len(grants) == 1
    assert operator_only(d.as_("alice", "GET", "/api/approvals/grants"))
    assert operator_only(d.as_("alice", "DELETE", f"/api/approvals/grants/{grants[0]['id']}"))
    assert len(d.gate.list_grants()) == 1, "the grant is still there"
    assert d.as_("owner", "DELETE", f"/api/approvals/grants/{grants[0]['id']}").status_code == 200


def test_the_operator_keeps_every_scope_and_every_request(d):
    d.pend("bbbb0001", session=BOB_SESSION)
    d.pend("cccc0001", session=None)
    assert d.as_("global", "POST", "/api/approvals/bbbb0001/approve", json={"scope": "always"}).status_code == 200
    assert d.as_("owner", "POST", "/api/approvals/cccc0001/approve", json={"scope": "once"}).status_code == 200
    assert d.waiting() == set()


# ── the link a request needs to have: which session raised it ────────────────

@pytest.mark.asyncio
async def test_an_approval_requested_inside_a_run_remembers_the_session_that_raised_it(tmp_path):
    from prometheus.engine import tool_context

    gate = SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None)
    queue = ApprovalQueue(security_gate=gate)
    token = tool_context.RUN_SESSION.set("web:alice-1")
    try:
        task = asyncio.ensure_future(queue.request_approval("bash", "run echo hi"))
        await asyncio.sleep(0)
    finally:
        tool_context.RUN_SESSION.reset(token)
    [pending] = queue.list_pending()
    assert pending.session_id == "web:alice-1"
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_outside_a_run_it_remembers_none(tmp_path):
    gate = SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None)
    queue = ApprovalQueue(security_gate=gate)
    task = asyncio.ensure_future(queue.request_approval("bash", "run echo hi"))
    await asyncio.sleep(0)
    [pending] = queue.list_pending()
    assert pending.session_id is None
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def test_the_agent_loop_and_the_queue_read_the_same_variable():
    from prometheus.engine import agent_loop, tool_context

    assert agent_loop._RUN_SESSION is tool_context.RUN_SESSION


def test_the_wire_shape_of_a_pending_request_did_not_change(d):
    """The session is for the server's filtering, not a new field every client now receives."""
    d.pend("aaaa0001", session=ALICE_SESSION)
    [item] = d.as_("global", "GET", "/api/approvals").json()
    assert "session_id" not in item


# ── L2: a device registers live-activity tokens for its OWN sessions ─────────

def test_a_device_cannot_attach_a_live_activity_to_a_session_it_does_not_own(d, monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(d.devices, "set_activity_token", lambda *a: calls.append(a))
    response = d.as_("alice", "POST", f"/api/devices/{d.alice['id']}/activity",
                     json={"session_id": BOB_SESSION, "activity_token": "tok"})
    assert response.status_code == 404 and response.json() == {"error": "unknown session"}
    assert calls == []


def test_nor_to_one_that_does_not_exist(d, monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(d.devices, "set_activity_token", lambda *a: calls.append(a))
    response = d.as_("alice", "POST", f"/api/devices/{d.alice['id']}/activity",
                     json={"session_id": "web:nobody-has-this", "activity_token": "tok"})
    assert response.status_code == 404 and calls == []


def test_its_own_session_is_fine(d, monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(d.devices, "set_activity_token", lambda *a: calls.append(a))
    response = d.as_("alice", "POST", f"/api/devices/{d.alice['id']}/activity",
                     json={"session_id": ALICE_SESSION, "activity_token": "tok"})
    assert response.status_code == 200 and calls == [(d.alice["id"], ALICE_SESSION, "tok")]


def test_an_operator_registering_for_its_own_device_is_not_restricted(d, monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(d.devices, "set_activity_token", lambda *a: calls.append(a))
    response = d.as_("owner", "POST", f"/api/devices/{d.owner['id']}/activity",
                     json={"session_id": BOB_SESSION, "activity_token": "tok"})
    assert response.status_code == 200 and len(calls) == 1


def test_a_device_still_cannot_manage_another_devices_activity(d):
    response = d.as_("alice", "POST", f"/api/devices/{d.bob['id']}/activity",
                     json={"session_id": ALICE_SESSION, "activity_token": "tok"})
    assert response.status_code == 401, "the existing own-device rule is unchanged"
