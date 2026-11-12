"""Pause everything (Beacon iOS "Piece 17", PR 1): the state, the routes, the gates and the pause itself.

Held here:

  * the state is written to disk before it takes effect; pausing twice is one pause (the second answer's
    ``since`` is the first's); a fresh daemon reading the file comes back paused; an unreadable or corrupt file
    means paused (fail closed); resuming clears it and starts nothing;
  * ``PUT /api/pause`` is the operator's (the API token or an owner device); ``DELETE /api/pause`` is a person's:
    an owner device, NEVER the bare API token; a scoped device gets ``403 operator_only`` for both;
  * pausing withdraws every pending approval (approving one afterwards is a 410), interrupts every bridge turn,
    cancels every other running turn in the process, stops running tasks, pauses (not stops) coding runs through
    their control file, stops desktop tasks, answers the counts and broadcasts ``paused_changed``;
  * while paused, ``POST /api/chat/send``, ``POST /api/chat``, story dispatch, the Paperclip wake and
    ``/v1/chat/completions`` answer ``423 {"error": "paused", since, by}`` and run nothing; the WebSocket send
    is refused with an ``error`` frame of kind ``paused``; ``run_loop`` itself refuses to start;
  * reading still works, ``/api/status`` says ``paused``, and ``server_capabilities`` lists ``pause``.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from prometheus.engine import pause  # noqa: E402
from prometheus.permissions.approval_queue import ApprovalQueue, PendingAction  # noqa: E402
from tests.support.device_world import World  # noqa: E402


@pytest.fixture
def state(tmp_path):
    s = pause.PauseState(tmp_path / "pause.json")
    pause.install(s)
    yield s
    pause.install(None)


# ── the state ────────────────────────────────────────────────────────────────

def test_pausing_writes_the_file_first_and_twice_is_one_pause(state):
    info, new = state.pause(by="api token", reason="test")
    assert new and state.current == info
    on_disk = json.loads(state.path.read_text())
    assert on_disk["paused"] is True and on_disk["since"] == info.since and on_disk["by"] == "api token"
    assert stat.S_IMODE(os.stat(state.path).st_mode) == 0o600
    again, new2 = state.pause(by="someone else")
    assert not new2 and again == info


def test_a_restart_comes_back_paused(state):
    info, _ = state.pause(by="owner device", reason="stop")
    assert pause.PauseState(state.path).current == info


@pytest.mark.parametrize("content", ["{not json", "[]", '{"paused": "yes"}', '{"paused": true}', ""])
def test_an_unreadable_file_means_paused(tmp_path, content):
    path = tmp_path / "pause.json"
    path.write_text(content)
    info = pause.PauseState(path).current
    assert info is not None and info.by == "unreadable"


def test_resuming_clears_it_and_a_restart_stays_resumed(state):
    state.pause(by="owner device")
    cleared = state.resume(by="owner device")
    assert cleared is not None and state.current is None
    assert pause.PauseState(state.path).current is None


def test_a_child_process_reads_the_file_and_sees_a_change(tmp_path, monkeypatch):
    path = tmp_path / "pause.json"
    monkeypatch.setattr(pause, "pause_path", lambda: path)
    pause.install(None)
    assert pause.current() is None
    pause.PauseState(path).pause(by="owner device")
    assert pause.current() is not None
    with pytest.raises(pause.Paused):
        pause.check()


def test_run_loop_refuses_to_start_while_paused(state):
    from prometheus.engine.agent_loop import run_loop

    state.pause(by="owner device")

    async def go():
        async for _ in run_loop(SimpleNamespace(), []):
            pass

    with pytest.raises(pause.Paused):
        asyncio.run(go())


# ── the routes ───────────────────────────────────────────────────────────────

@pytest.fixture
def w(tmp_path, state):
    world = World(tmp_path)
    world.owner = world.devices.mint_owner("owner phone", "ios", by="test")
    return world


def _owner(w):
    return {"Authorization": f"Bearer {w.owner['token']}"}


def test_the_api_token_pauses_and_cannot_resume(w):
    r = w.call("op", "PUT", "/api/pause", json={"reason": "stop"})
    assert r.status_code == 200, r.text
    assert r.json()["paused"] is True and r.json()["by"]
    again = w.call("op", "PUT", "/api/pause", json={})
    assert again.json()["since"] == r.json()["since"]
    refused = w.call("op", "DELETE", "/api/pause")
    assert refused.status_code == 403 and refused.json()["error"] == "person_required"
    assert pause.current() is not None


def test_an_owner_device_pauses_and_resumes(w):
    assert w.client.put("/api/pause", headers=_owner(w), json={}).status_code == 200
    r = w.client.delete("/api/pause", headers=_owner(w))
    assert r.status_code == 200 and r.json()["paused"] is False
    assert pause.current() is None


@pytest.mark.parametrize("method", ["PUT", "DELETE"])
def test_a_scoped_device_is_refused_both(w, method):
    r = w.call("a", method, "/api/pause", json={})
    assert r.status_code == 403 and r.json()["error"] == "operator_only"


def test_status_says_paused(w):
    assert w.call("op", "GET", "/api/status").json()["paused"] is None
    w.call("op", "PUT", "/api/pause", json={"reason": "r"})
    paused = w.call("op", "GET", "/api/status").json()["paused"]
    assert paused["reason"] == "r" and paused["since"]


def test_the_routes_are_classified_operator_only():
    from prometheus.web import route_access

    assert ("PUT", "/api/pause") in route_access.OPERATOR_ONLY
    assert ("DELETE", "/api/pause") in route_access.OPERATOR_ONLY


def test_server_capabilities_lists_pause(w):
    assert "pause" in w.bridge._server_capabilities()


# ── the gates ────────────────────────────────────────────────────────────────

def test_a_send_while_paused_is_423_and_runs_nothing(w):
    sid = w.call("op", "POST", "/api/sessions").json()["session_id"]
    w.call("op", "PUT", "/api/pause", json={})
    calls: list = []

    async def _dispatch(*a, **kw):
        calls.append(a)

    w.bridge.dispatch_user_message = _dispatch
    r = w.call("op", "POST", "/api/chat/send", json={"session_id": sid, "message": "hi"})
    assert r.status_code == 423 and r.json()["error"] == "paused" and r.json()["since"]
    assert calls == []


@pytest.mark.parametrize(("path", "body"), [
    ("/api/chat", {"message": "hi"}),
    ("/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "hi"}]}),
    ("/api/paperclip/wake", {}),
])
def test_other_front_doors_are_423(w, path, body):
    w.call("op", "PUT", "/api/pause", json={})
    r = w.call("op", "POST", path, json=body)
    assert r.status_code == 423, (path, r.status_code, r.text)
    assert r.json().get("error") == "paused" or r.json().get("error", {}).get("code") == "paused"


def test_a_websocket_send_is_refused_with_an_error_frame(w):
    frames: list[dict] = []

    async def _broadcast(frame):
        frames.append(frame)

    w.bridge.broadcast = _broadcast
    w.call("op", "PUT", "/api/pause", json={})
    asyncio.run(w.bridge._handle_send_message("web:s1", "hello"))
    errors = [f for f in frames if f.get("type") == "error"]
    assert errors and errors[0]["payload"]["kind"] == "paused"
    assert w.mgr.get("web:s1") is None or not w.mgr.get("web:s1").messages


# ── the pause itself ─────────────────────────────────────────────────────────

class _Gate:
    pass


class _Tasks:
    def __init__(self, records):
        self.records = records
        self.stopped: list[str] = []

    def list_tasks(self, *, status=None):
        return [r for r in self.records if status is None or r.status == status]

    async def stop_task(self, task_id):
        self.stopped.append(task_id)
        return SimpleNamespace(id=task_id, status="killed")


class _Desktop:
    def __init__(self):
        self.stopped: list[str] = []

    def running(self):
        return [SimpleNamespace(task_id="dt1", status="running")]

    def stop(self, task_id):
        self.stopped.append(task_id)
        return True


def test_pausing_stops_everything_and_keeps_coding_work(w, tmp_path, monkeypatch):
    queue = ApprovalQueue(security_gate=_Gate(), telegram_adapter=None)
    queue._register(PendingAction(request_id="p1", tool_name="bash", description="run", session_id="web:x"))
    w.app.state.approval_queue = queue
    tasks = _Tasks([SimpleNamespace(id="b1", type="local_bash", status="running", session_id=None),
                    SimpleNamespace(id="w1", type="local_agent", status="running", session_id="web:x"),
                    SimpleNamespace(id="c1", type="local_bash", status="running", session_id="coding:c1")])
    w.app.state.task_manager = tasks
    desk = _Desktop()
    w.app.state.computer_runner = desk
    from prometheus.coding import managed

    monkeypatch.setattr(managed, "coding_control_dir", lambda tid: tmp_path / "control" / tid)
    frames: list[dict] = []

    async def _broadcast(frame):
        frames.append(frame)

    w.bridge.broadcast = _broadcast

    r = w.call("op", "PUT", "/api/pause", json={"reason": "now"})
    assert r.status_code == 200, r.text
    stopped = r.json()["stopped"]
    assert stopped["approvals_withdrawn"] == 1
    assert sorted(tasks.stopped) == ["b1", "w1"]                     # coding run c1 is paused, not stopped
    assert stopped["tasks"] == 2 and stopped["coding_runs"] == 1 and stopped["desktop_tasks"] == 1
    control = json.loads((tmp_path / "control" / "c1" / "control.json").read_text())
    assert control["paused"] is True
    assert desk.stopped == ["dt1"]
    assert queue.finished("p1")[0] == "withdrawn"
    late = w.call("op", "POST", "/api/approvals/p1/approve", json={"scope": "once"})
    assert late.status_code == 410
    assert any(f.get("type") == "paused_changed" and f["payload"]["paused"] for f in frames)


def test_pausing_cancels_every_running_turn_in_the_process(state):
    from prometheus.web import pause_routes

    async def go():
        async def turn():
            token = pause.turn_started()
            try:
                await asyncio.sleep(60)
            finally:
                pause.turn_ended(token)

        t = asyncio.create_task(turn())
        await asyncio.sleep(0)
        app = SimpleNamespace(state=SimpleNamespace())
        result = await pause_routes.pause_everything(app, by="test", reason="")
        await asyncio.sleep(0)
        return t, result

    t, result = asyncio.run(go())
    assert t.cancelled() and result["stopped"]["turns"] >= 1
