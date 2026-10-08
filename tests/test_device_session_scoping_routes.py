"""Device scoping — the routes that are not `/api/sessions/{id}/…` but still carry a session.

  * POST /api/search            conversation content across sessions
  * GET  /api/events/recent     persisted signals (a teacher escalation holds a whole exchange)
  * GET  /api/activity/recent   the same table, no session filter at all
  * POST /api/stories/{pk}/dispatch   writes a user turn into the session named in its body
  * POST/GET /api/computer/tasks      a desktop task runs inside a chat session

And the guard that keeps the path rule honest: a route whose session parameter is
not literally called `session_id` would slip past the router-level dependency.

NOT covered, on purpose and listed in docs/contracts/device-scoping.md: approvals, background
and coding tasks, /api/tools/recent, push notifications, and stopping a desktop task.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from prometheus.sentinel.signals import ActivitySignal, SignalBus  # noqa: E402
from prometheus.telemetry import tracker  # noqa: E402
from prometheus.telemetry.tracker import ToolCallTelemetry  # noqa: E402
from tests.support.device_world import A_SECRET, TG_SECRET, World  # noqa: E402
from tests.support.doubles import register_double  # noqa: E402


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path))  # the kanban store lands here
    return World(tmp_path)


def _hit_sessions(body: dict) -> set[str]:
    return {h["session_id"] for h in (*body["messages"], *body["summaries"])}


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


def _seed(world: World) -> tuple[str, str, str]:
    sid_a = world.device_session("a", text="zebra from a")
    sid_b = world.device_session("b", text="zebra from b")
    tg = world.operator_session(text="zebra from telegram")
    for sid in (sid_a, sid_b, tg):
        world.seed_summary(sid, f"zebra summary of {sid}")
    return sid_a, sid_b, tg


def test_a_device_search_returns_only_its_own_sessions(world):
    sid_a, sid_b, tg = _seed(world)

    as_a = world.call("a", "POST", "/api/search", json={"q": "zebra"}).json()
    assert _hit_sessions(as_a) == {sid_a}
    assert as_a["messages"] and as_a["summaries"]

    as_op = world.call("op", "POST", "/api/search", json={"q": "zebra"}).json()
    assert _hit_sessions(as_op) == {sid_a, sid_b, tg}


def test_a_device_cannot_search_inside_a_session_it_does_not_own(world):
    sid_a, _, tg = _seed(world)
    assert world.call("b", "POST", "/api/search",
                      json={"q": "zebra", "session_id": sid_a}).status_code == 404
    assert world.call("b", "POST", "/api/search",
                      json={"q": "zebra", "session_id": tg}).status_code == 404
    mine = world.call("a", "POST", "/api/search", json={"q": "zebra", "session_id": sid_a})
    assert mine.status_code == 200 and _hit_sessions(mine.json()) == {sid_a}


def test_a_device_with_no_sessions_finds_nothing(world):
    _seed(world)
    fresh = world.devices.mint("fresh", "ios")
    r = world.client.post("/api/search", json={"q": "zebra"},
                          headers={"Authorization": f"Bearer {fresh['token']}"})
    assert r.status_code == 200 and r.json()["returned"] == 0


def test_a_devices_search_merges_all_its_sessions_and_honours_the_limit(world):
    one = world.device_session("a", text="zebra one")
    two = world.device_session("a", text="zebra two")
    world.device_session("b", text="zebra of b")
    body = world.call("a", "POST", "/api/search",
                      json={"q": "zebra", "scope": "messages", "limit": 50}).json()
    assert _hit_sessions(body) == {one, two}
    capped = world.call("a", "POST", "/api/search",
                        json={"q": "zebra", "scope": "messages", "limit": 1}).json()
    assert len(capped["messages"]) == 1


# --------------------------------------------------------------------------- #
# Persisted signals
# --------------------------------------------------------------------------- #


@pytest.fixture
def signals(world, tmp_path, monkeypatch):
    """Real telemetry + a real SignalBus: emit() persists to signal_events."""
    tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    monkeypatch.setattr(tracker, "get_telemetry_handle", lambda: tel)
    return SignalBus(telemetry=tel, history_limit=100)


async def _emit(bus: SignalBus, kind: str, **payload) -> None:
    await bus.emit(ActivitySignal(kind=kind, payload=payload, source="test"))


@pytest.mark.asyncio
async def test_events_recent_holds_only_the_callers_sessions_and_daemon_level_events(world, signals):
    sid_a = world.device_session("a")
    sid_b = world.device_session("b", text="b's own")
    await _emit(signals, "teacher_escalation", session_id=sid_a, user_request=A_SECRET)
    await _emit(signals, "teacher_escalation", session_id=sid_b, user_request="b's request")
    await _emit(signals, "teacher_escalation", session_id="telegram:123", user_request=TG_SECRET)
    await _emit(signals, "dream_start", note="daemon level")

    for path in ("/api/events/recent", "/api/activity/recent"):
        as_a = world.call("a", "GET", path)
        assert as_a.status_code == 200
        assert A_SECRET in as_a.text and "daemon level" in as_a.text
        assert "b's request" not in as_a.text and TG_SECRET not in as_a.text, path
        as_op = world.call("op", "GET", path).text
        assert A_SECRET in as_op and "b's request" in as_op and TG_SECRET in as_op, path


@pytest.mark.asyncio
async def test_events_recent_refuses_a_foreign_session_filter(world, signals):
    sid_a = world.device_session("a")
    await _emit(signals, "computer_step", session_id=sid_a, seq=1)
    assert world.call("b", "GET", f"/api/events/recent?session_id={sid_a}").status_code == 404
    assert world.call("a", "GET", f"/api/events/recent?session_id={sid_a}").status_code == 200
    assert world.call("op", "GET", f"/api/events/recent?session_id={sid_a}").status_code == 200


# --------------------------------------------------------------------------- #
# Writing into a session by the id in a body
# --------------------------------------------------------------------------- #


def _story(world: World) -> str:
    return world.call("op", "POST", "/api/stories",
                      json={"story_id": "US-1", "title": "Land the eagle"}).json()["story"]["id"]


def test_a_device_cannot_dispatch_a_story_into_another_devices_session(world):
    sid_a = world.device_session("a")
    pk = _story(world)
    before = world.lcm.count_all(sid_a)

    r = world.call("b", "POST", f"/api/stories/{pk}/dispatch", json={"session_key": sid_a})
    assert r.status_code == 404
    assert world.lcm.count_all(sid_a) == before

    # Its own session (or a brand-new id) is fine; so is the operator, anywhere.
    assert world.call("a", "POST", f"/api/stories/{pk}/dispatch",
                      json={"session_key": sid_a}).status_code == 200
    assert world.call("op", "POST", f"/api/stories/{pk}/dispatch",
                      json={"session_key": sid_a}).status_code == 200


# --------------------------------------------------------------------------- #
# Desktop tasks
# --------------------------------------------------------------------------- #


@register_double("device_scoping._Runner", replaces="prometheus.computer.door.ComputerRunner")
class _Runner:
    """Just enough of the door's runner for the routes: everyone is 'a person'."""

    def __init__(self) -> None:
        self.people = SimpleNamespace(check=lambda by: (True, ""))
        self.tasks: dict[str, SimpleNamespace] = {}
        self.stopped: list[str] = []

    async def start(self, task_input, *, session_id, surface, by):
        task = SimpleNamespace(
            task_id=f"t{len(self.tasks) + 1}", status="running", app="gedit",
            session_id=session_id,
            as_dict=lambda: {"task_id": f"t{len(self.tasks)}", "session_id": session_id,
                             "goal": task_input.goal})
        self.tasks[task.task_id] = task
        return task

    def get(self, task_id):
        return self.tasks.get(task_id)

    def stop(self, task_id):
        self.stopped.append(task_id)
        return True


def test_a_device_cannot_start_or_read_a_desktop_task_in_another_devices_session(world):
    runner = _Runner()
    world.app.state.computer_runner = runner
    sid_a = world.device_session("a")

    started = world.call("b", "POST", "/api/computer/tasks",
                         json={"session_id": sid_a, "goal": "close every window"})
    assert started.status_code == 404 and runner.tasks == {}

    mine = world.call("a", "POST", "/api/computer/tasks",
                      json={"session_id": sid_a, "goal": "open gedit"})
    assert mine.status_code == 202
    tid = mine.json()["task_id"]

    assert world.call("b", "GET", f"/api/computer/tasks/{tid}").status_code == 404
    assert "open gedit" in world.call("a", "GET", f"/api/computer/tasks/{tid}").text
    assert world.call("op", "GET", f"/api/computer/tasks/{tid}").status_code == 200


def test_stopping_a_desktop_task_stays_open_to_every_valid_token(world):
    """Deliberate and pinned: a stop can only end a task (computer-use door design)."""
    runner = _Runner()
    world.app.state.computer_runner = runner
    sid_a = world.device_session("a")
    tid = world.call("a", "POST", "/api/computer/tasks",
                     json={"session_id": sid_a, "goal": "x"}).json()["task_id"]
    r = world.call("b", "POST", f"/api/computer/tasks/{tid}/stop")
    assert r.status_code == 200 and runner.stopped == [tid]


# --------------------------------------------------------------------------- #
# Guards on the rule itself
# --------------------------------------------------------------------------- #


def test_no_route_names_its_session_parameter_anything_but_session_id(world):
    """The router-level guard keys on the path parameter `session_id`. A route that called
    it `sid` or `session_key` would be a hole nobody sees — fail here instead."""
    for route in world.app.routes:
        path = getattr(route, "path", "")
        for name in re.findall(r"\{([a-z_]+)(?::[a-z]+)?\}", path):
            assert name == "session_id" or "session" not in name, (
                f"{path}: path parameter {name!r} looks like a session id but the device guard "
                "only recognises `session_id` — rename it, or guard the route by hand"
            )


def test_every_session_route_carries_the_guard_dependency(world):
    """Dependencies are copied onto a route when it is registered, so the guard must be
    installed before the first route is. Sweep the app, do not trust the ordering."""
    from fastapi.routing import APIRoute

    unguarded = []
    for route in world.app.routes:
        if isinstance(route, APIRoute) and "{session_id}" in route.path:
            names = [d.call.__name__ for d in route.dependant.dependencies]
            if "_guard_session_path" not in names:
                unguarded.append((sorted(route.methods), route.path))
    assert not unguarded, f"session routes without the device guard: {unguarded}"


def test_the_daemon_namespaces_cover_every_gateway_platform():
    """DAEMON_GATEWAYS is a hand-kept mirror of gateway.config.Platform: a new platform that
    the daemon writes sessions for must not be claimable by a device."""
    from prometheus.gateway.config import Platform
    from prometheus.web.session_scope import DAEMON_GATEWAYS

    assert {p.value for p in Platform} <= DAEMON_GATEWAYS
    assert "coding" in DAEMON_GATEWAYS  # coding:<task id>, minted by coding/managed.py
