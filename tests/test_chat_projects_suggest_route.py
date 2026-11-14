"""``GET /api/chat-project-suggestions`` (docs/design/chat-projects.md, 3.6), on the real app.

Owner only, on its own path (under ``/api/chat-projects/`` it would match the chat-only ``{id}`` pattern). For unfiled
conversations only, leaving out machine sessions and sessions a chat-only device owns. ``project`` names the existing
project a suggestion would file into.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from prometheus.memory.lcm_types import MessagePart  # noqa: E402
from prometheus.web import route_access  # noqa: E402
from tests.test_chat_projects_routes import World  # noqa: E402

URL = "/api/chat-project-suggestions"


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World(tmp_path)
    monkeypatch.setenv("HOME", str(w.home))
    return w


def _tool_calls(world: World, sid: str, paths: list[str]) -> None:
    for i, path in enumerate(paths):
        blocks = [{"type": "text", "text": "ok"},
                  {"type": "tool_use", "id": f"t{i}", "name": "read_file", "input": {"path": path}}]
        world.conv.add_message(sid, MessagePart(role="assistant", content="ok", content_json=json.dumps(blocks),
                                                session_id=sid, turn_index=i + 1))


def test_it_is_the_owners_and_classified():
    assert ("GET", URL) in route_access.OPERATOR_ONLY
    assert not route_access.is_scoped_allowed("GET", URL)


def test_a_chat_only_device_is_refused(world):
    assert world.call("a", "GET", URL).status_code == 403


def test_a_suggestion_from_tool_calls(world):
    sid = world.session()
    repo = str(world.home / "projects" / "beacon-desktop" / "src")
    _tool_calls(world, sid, [f"{repo}/a.ts", f"{repo}/b.ts", f"{repo}/c.ts"])
    body = world.call("op", "GET", URL).json()
    s = body["suggestions"][sid]
    assert (s["key"], s["label"], s["calls"], s["share"], s["project"]) == ("beacon-desktop", "beacon-desktop", 3, 1.0,
                                                                             None)
    assert isinstance(body["rev"], int)


def test_a_suggestion_names_the_existing_project_by_name(world):
    p = world.project("Beacon-Desktop")
    sid = world.session()
    repo = str(world.home / "projects" / "beacon-desktop")
    _tool_calls(world, sid, [f"{repo}/a", f"{repo}/b", f"{repo}/c"])
    assert world.call("op", "GET", URL).json()["suggestions"][sid]["project"] == {"id": p["id"], "name": "Beacon-Desktop"}


def test_a_path_in_a_project_folder_suggests_that_project(world):
    p = world.project("Kitchen remodel")
    sid = world.session()
    _tool_calls(world, sid, [f"{p['folder']}/a.md", f"{p['folder']}/b.md", f"{p['folder']}/c.md"])
    s = world.call("op", "GET", URL).json()["suggestions"][sid]
    assert s["key"] == p["id"] and s["label"] == "Kitchen remodel" and s["project"]["id"] == p["id"]


def test_filed_machine_and_chat_only_device_sessions_are_left_out(world):
    p = world.project("Elsewhere")
    repo = str(world.home / "projects" / "beacon-desktop")
    paths = [f"{repo}/a", f"{repo}/b", f"{repo}/c"]
    filed = world.session()
    _tool_calls(world, filed, paths)
    world.call("op", "PUT", f"/api/sessions/{filed}/project", json={"chat_project_id": p["id"], "keep_workspace": True})
    device = world.session("a")
    _tool_calls(world, device, paths)
    machine = "coding:run-7"
    world.conv.add_message(machine, MessagePart(role="user", content="x", session_id=machine, turn_index=0))
    _tool_calls(world, machine, paths)
    owner_device = world.session("owner")
    _tool_calls(world, owner_device, paths)
    got = world.call("op", "GET", URL).json()["suggestions"]
    assert filed not in got and device not in got and machine not in got
    assert owner_device in got


def test_a_mixed_session_gets_no_suggestion(world):
    sid = world.session()
    a, b = world.home / "projects" / "one", world.home / "projects" / "two"
    _tool_calls(world, sid, [f"{a}/x", f"{b}/y", f"{a}/z", f"{b}/w"])
    assert sid not in world.call("op", "GET", URL).json()["suggestions"]


def test_a_new_tool_call_is_seen_on_the_next_read(world):
    sid = world.session()
    repo = str(world.home / "projects" / "pilot")
    _tool_calls(world, sid, [f"{repo}/a", f"{repo}/b"])
    assert sid not in world.call("op", "GET", URL).json()["suggestions"]
    world.conv.add_message(sid, MessagePart(
        role="assistant", content="", session_id=sid, turn_index=9,
        content_json=json.dumps([{"type": "tool_use", "id": "t9", "name": "read_file", "input": {"path": f"{repo}/c"}}])))
    assert world.call("op", "GET", URL).json()["suggestions"][sid]["key"] == "pilot"


def test_garbage_content_json_is_skipped(world):
    sid = world.session()
    for i, raw in enumerate(["not json", "[1, 2]", '{"type": "tool_use"}', "[" * 5000]):
        world.conv.add_message(sid, MessagePart(role="assistant", content="", content_json=raw, session_id=sid,
                                                turn_index=i + 1))
    assert world.call("op", "GET", URL).status_code == 200


def test_filing_does_not_empty_the_cache(world, monkeypatch):
    sids = [world.session() for _ in range(3)]
    repo = str(world.home / "projects" / "pilot")
    for sid in sids:
        _tool_calls(world, sid, [f"{repo}/a", f"{repo}/b", f"{repo}/c"])
    p = world.project("Elsewhere")
    world.call("op", "GET", URL)
    reads = []
    real = world.conv.tool_use_json
    monkeypatch.setattr(world.conv, "tool_use_json", lambda sid: reads.append(sid) or real(sid))
    world.call("op", "PUT", f"/api/sessions/{sids[0]}/project", json={"chat_project_id": p["id"], "keep_workspace": True})
    world.call("op", "GET", URL)
    assert reads == []  # filing changed no folder: nothing was read again


def test_a_checkout_folder_maps_to_its_project(world):
    p = world.project("Beacon", folder=world.folder("projects/beacon-deploy"))
    sid = world.session()
    other = str(world.home / "projects" / "beacon-wt-2")
    _tool_calls(world, sid, [f"{other}/a", f"{other}/b", f"{other}/c"])
    assert world.call("op", "GET", URL).json()["suggestions"][sid]["project"]["id"] == p["id"]
