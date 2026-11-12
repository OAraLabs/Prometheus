"""Piece 18 routes: push-to-start registration, and live-activity tokens keyed by ``{kind, id}``.

``PUT /api/devices/{id}/activity-start`` ``{token}`` (and ``DELETE``) registers a phone's push-to-start token, so
the daemon can start a Live Activity on it for a job the phone did not start. The rule is the existing push
routes': a device manages only its own registration (the global operator, any). The token is never listed.

``POST /api/devices/{id}/activity`` takes ``{kind, id, activity_token}``; ``{session_id, activity_token}`` keeps
meaning ``kind: turn``. A turn is registered for a session the device owns; every other kind is the operator's,
because a scoped device cannot see tasks, coding runs or desktop tasks at all.

Also the bridge half: it lists its running turns for the watcher, and a turn that fails ends its Live Activity.
"""

from __future__ import annotations

import asyncio
import secrets

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

GLOBAL = "job-activity-global-" + secrets.token_hex(8)
ALICE_SESSION = "web:alice-1"
BOB_SESSION = "web:bob-1"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "config"))


class _Daemon:
    def __init__(self, tmp_path) -> None:
        self.devices = DeviceStore(tmp_path / "devices.db")
        self.app = create_app({"web": {"api_token": GLOBAL}}, device_store=self.devices)
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

    def start_tokens(self) -> dict[str, str]:
        return {t.id: tok for t, tok in self.devices.activity_start_targets()}


@pytest.fixture
def d(tmp_path) -> _Daemon:
    return _Daemon(tmp_path)


# ── push-to-start ─────────────────────────────────────────────────────────────

def test_a_scoped_device_registers_its_own_push_to_start_token(d):
    r = d.as_("alice", "PUT", f"/api/devices/{d.alice['id']}/activity-start", json={"token": "start-a"})
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert d.start_tokens() == {d.alice["id"]: "start-a"}
    listed = d.as_("global", "GET", "/api/devices")
    assert listed.status_code == 200 and "start-a" not in listed.text, "never returned by GET /api/devices"
    assert d.as_("alice", "DELETE", f"/api/devices/{d.alice['id']}/activity-start").status_code == 200
    assert d.start_tokens() == {}


def test_a_device_cannot_register_a_push_to_start_token_for_another_device(d):
    r = d.as_("alice", "PUT", f"/api/devices/{d.bob['id']}/activity-start", json={"token": "start-a"})
    assert r.status_code == 401
    assert d.as_("alice", "DELETE", f"/api/devices/{d.bob['id']}/activity-start").status_code == 401
    assert d.start_tokens() == {}


def test_the_operator_may_register_for_any_device(d):
    r = d.as_("global", "PUT", f"/api/devices/{d.bob['id']}/activity-start", json={"token": "start-b"})
    assert r.status_code == 200 and d.start_tokens() == {d.bob["id"]: "start-b"}


def test_push_to_start_needs_a_token_and_a_live_device(d):
    assert d.as_("alice", "PUT", f"/api/devices/{d.alice['id']}/activity-start", json={}).status_code == 400
    assert d.as_("alice", "PUT", f"/api/devices/{d.alice['id']}/activity-start",
                 json={"token": "x" * 5000}).status_code == 400
    assert d.as_("global", "PUT", "/api/devices/no-such-device/activity-start",
                 json={"token": "t"}).status_code == 404


# ── {kind, id} activity tokens ────────────────────────────────────────────────

def test_the_session_id_form_still_means_a_turn(d):
    r = d.as_("alice", "POST", f"/api/devices/{d.alice['id']}/activity",
              json={"session_id": ALICE_SESSION, "activity_token": "tok"})
    assert r.status_code == 200
    assert [tok for _, tok in d.devices.activity_targets(ALICE_SESSION)] == []  # no alert registration yet
    d.devices.set_push(d.alice["id"], "alert", "sandbox", "ai.example.app")
    assert [tok for _, tok in d.devices.activity_targets(ALICE_SESSION)] == ["tok"]


def test_kind_turn_is_the_same_as_the_session_id_form(d):
    d.devices.set_push(d.alice["id"], "alert", "sandbox", "ai.example.app")
    r = d.as_("alice", "POST", f"/api/devices/{d.alice['id']}/activity",
              json={"kind": "turn", "id": ALICE_SESSION, "activity_token": "tok"})
    assert r.status_code == 200
    assert [tok for _, tok in d.devices.activity_targets(ALICE_SESSION)] == ["tok"]
    assert d.as_("alice", "DELETE", f"/api/devices/{d.alice['id']}/activity",
                 json={"kind": "turn", "id": ALICE_SESSION}).status_code == 200
    assert d.devices.activity_targets(ALICE_SESSION) == []


def test_a_turn_in_someone_elses_session_is_the_404_of_none(d):
    r = d.as_("alice", "POST", f"/api/devices/{d.alice['id']}/activity",
              json={"kind": "turn", "id": BOB_SESSION, "activity_token": "tok"})
    assert r.status_code == 404 and r.json() == {"error": "unknown session"}


@pytest.mark.parametrize("kind", ["task", "coding", "desktop"])
def test_a_scoped_device_may_not_follow_a_job(d, kind):
    r = d.as_("alice", "POST", f"/api/devices/{d.alice['id']}/activity",
              json={"kind": kind, "id": "j1", "activity_token": "tok"})
    assert r.status_code == 403 and r.json() == {"error": "operator_only"}
    assert d.devices.job_activity_targets(kind, "j1") == []


@pytest.mark.parametrize("kind", ["task", "coding", "desktop"])
def test_an_owner_device_registers_a_job_token_apart_from_turns(d, kind):
    d.devices.set_push(d.owner["id"], "alert", "sandbox", "ai.example.app")
    r = d.as_("owner", "POST", f"/api/devices/{d.owner['id']}/activity",
              json={"kind": kind, "id": "c1", "activity_token": "job-tok"})
    assert r.status_code == 200
    assert [tok for _, tok in d.devices.job_activity_targets(kind, "c1")] == ["job-tok"]
    assert d.devices.activity_targets("c1") == [] and d.devices.activity_targets(f"coding:c1") == []
    assert d.as_("owner", "DELETE", f"/api/devices/{d.owner['id']}/activity",
                 json={"kind": kind, "id": "c1"}).status_code == 200
    assert d.devices.job_activity_targets(kind, "c1") == []


@pytest.mark.parametrize("body", [
    {"kind": "nope", "id": "x", "activity_token": "t"},
    {"kind": "task", "activity_token": "t"},
    {"kind": "task", "id": "x"},
    {"kind": "task", "id": "x" * 1000, "activity_token": "t"},
])
def test_a_bad_activity_body_is_a_400(d, body):
    r = d.as_("owner", "POST", f"/api/devices/{d.owner['id']}/activity", json=body)
    assert r.status_code == 400


def test_a_device_still_cannot_register_a_job_for_another_device(d):
    r = d.as_("alice", "POST", f"/api/devices/{d.bob['id']}/activity",
              json={"kind": "turn", "id": ALICE_SESSION, "activity_token": "tok"})
    assert r.status_code == 401


# ── the bridge half ───────────────────────────────────────────────────────────

class _FakeSession:
    def __init__(self) -> None:
        self.messages: list = []

    def get_messages(self) -> list:
        return self.messages

    def persist_loop_result(self, original_len: int):
        return None

    def rollback_to(self, n: int) -> int:
        del self.messages[n:]
        return 0


async def _wait_until(cond, timeout: float = 2.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not cond():
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("condition not reached before timeout")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_the_bridge_lists_its_running_turns(monkeypatch):
    import prometheus.engine.agent_loop as agent_loop_mod
    from prometheus.web.ws_server import WebSocketBridge

    async def run_loop(context, messages, mode="agent", session_id=None, tool_choice=None, surface=None):
        await asyncio.Event().wait()
        yield None, None  # pragma: no cover

    monkeypatch.setattr(agent_loop_mod, "run_loop", run_loop)
    bridge = WebSocketBridge(loop_context=object())
    assert bridge.running_turns() == []
    task = asyncio.ensure_future(bridge._run_agent("desktop:t", _FakeSession()))
    await _wait_until(lambda: "desktop:t" in bridge._turn_tasks)
    [(session_id, started_at, progress)] = bridge.running_turns()
    assert session_id == "desktop:t" and started_at > 0 and progress["phase"] == "thinking"
    assert bridge.interrupt_turn("desktop:t")
    await task
    assert bridge.running_turns() == []


@pytest.mark.asyncio
async def test_a_failed_turn_ends_its_live_activity_as_failed(monkeypatch):
    import prometheus.engine.agent_loop as agent_loop_mod
    from prometheus.web.ws_server import WebSocketBridge

    async def run_loop(context, messages, mode="agent", session_id=None, tool_choice=None, surface=None):
        raise RuntimeError("the backend fell over")
        yield None, None  # pragma: no cover

    class _Dispatcher:
        def __init__(self) -> None:
            self.ended: list[tuple[str, dict]] = []

        async def on_agent_progress(self, session_id, payload):
            pass

        async def on_chat_done(self, session_id, payload):
            self.ended.append((session_id, payload))

    monkeypatch.setattr(agent_loop_mod, "run_loop", run_loop)
    bridge = WebSocketBridge(loop_context=object())
    bridge.push_dispatcher = _Dispatcher()
    await bridge._run_agent("desktop:t", _FakeSession())
    assert [(s, p.get("failed")) for s, p in bridge.push_dispatcher.ended] == [("desktop:t", True)]
