"""The chat projects routes (docs/design/chat-projects.md, sections 3.1-3.3, 4-6), on the real app.

Real throughout: ``create_app``, the bearer middleware and device tokens (the global token, an owner device, two
chat-only devices), the session path guard, ``route_access``, the conversation store and its projects store, and a
real ``WebSocketBridge`` for the frame. Folders are real directories under a temp root.
"""

from __future__ import annotations

import json
import os
import secrets
import stat
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.engine.session import SessionManager  # noqa: E402
from prometheus.memory.lcm_conversation_store import LCMConversationStore  # noqa: E402
from prometheus.web import route_access  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402
from tests.support.device_world import engine_over  # noqa: E402

GLOBAL = "projects-global-" + secrets.token_hex(8)
SCOPED = {("GET", "/api/chat-projects"), ("GET", "/api/chat-projects/{project_id}")}
OWNER = {("POST", "/api/chat-projects"), ("PUT", "/api/chat-projects/{project_id}"), ("DELETE", "/api/chat-projects/{project_id}"),
         ("PUT", "/api/sessions/{session_id}/project")}


class _Socket:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send(self, raw: str) -> None:
        self.frames.append(json.loads(raw))


class World:
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "projects-root"
        self.home = tmp_path / "home"
        self.home.mkdir()
        self.devices = DeviceStore(tmp_path / "devices.db")
        self.conv = LCMConversationStore(tmp_path / "lcm.db")
        self.mgr = SessionManager()
        self.mgr.lcm_engine = engine_over(self.conv)
        config = {"web": {"api_token": GLOBAL}, "projects": {"root": str(self.root)},
                  "security": {"denied_paths": [str(tmp_path / "secret")]}}
        self.app = create_app(config, session_mgr=self.mgr, lcm_engine=self.mgr.lcm_engine, device_store=self.devices)
        from prometheus.web.ws_server import WebSocketBridge

        self.bridge = WebSocketBridge(session_mgr=self.mgr, api_token=GLOBAL, device_store=self.devices,
                                      loop_context=None)
        self.app.state.ws_bridge = self.bridge
        self.client = TestClient(self.app)
        self.a = self.devices.mint("device-a", "ios")
        self.b = self.devices.mint("device-b", "ios")
        self.owner = self.devices.mint_owner("owner-phone", "ios", by="test")
        (tmp_path / "secret").mkdir()

    def hdr(self, who: str) -> dict:
        token = {"op": GLOBAL, "a": self.a["token"], "b": self.b["token"], "owner": self.owner["token"]}[who]
        return {"Authorization": f"Bearer {token}"}

    def call(self, who: str, method: str, url: str, **kw):
        return self.client.request(method, url, headers=self.hdr(who), **kw)

    def session(self, who: str = "op") -> str:
        sid = self.call(who, "POST", "/api/sessions").json()["session_id"]
        self.conv.add_message(sid, _part(sid))
        return sid

    def project(self, name: str = "Kitchen", **body) -> dict:
        r = self.call("op", "POST", "/api/chat-projects", json={"name": name, **body})
        assert r.status_code in (200, 201), r.text
        return r.json()["project"]

    def folder(self, name: str) -> str:
        p = self.home / name
        p.mkdir(parents=True, exist_ok=True)
        return str(p.resolve())


def _part(sid: str):
    from prometheus.memory.lcm_types import MessagePart

    return MessagePart(role="user", content="hello", session_id=sid, turn_index=0)


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World(tmp_path)
    monkeypatch.setenv("HOME", str(w.home))
    return w


# ------------------------------------------------------------------------- access


def test_the_routes_are_classified():
    assert SCOPED <= route_access.SCOPED_ALLOWED
    assert OWNER <= route_access.OPERATOR_ONLY


def test_no_owner_only_route_matches_a_chat_only_pattern():
    """A path shape that happens to match a chat-only entry must not open an owner route (design 4)."""
    import re

    for method, pattern in route_access.OPERATOR_ONLY:
        concrete = re.sub(r"\{[^}]+\}", "x1", pattern)
        assert not route_access.is_scoped_allowed(method, concrete), (method, pattern)


@pytest.mark.parametrize(("method", "url", "body"), [
    ("POST", "/api/chat-projects", {"name": "X"}),
    ("PUT", "/api/chat-projects/cp_000000000000", {"name": "X"}),
    ("DELETE", "/api/chat-projects/cp_000000000000", None),
])
def test_a_chat_only_device_cannot_change_projects(world, method, url, body):
    r = world.call("a", method, url, **({"json": body} if body else {}))
    assert r.status_code == 403


def test_a_chat_only_device_cannot_file(world):
    sid = world.session("a")
    p = world.project()
    r = world.call("a", "PUT", f"/api/sessions/{sid}/project", json={"chat_project_id": p["id"]})
    assert r.status_code == 403


# ------------------------------------------------------------------------- create


def test_create_makes_a_managed_folder(world):
    r = world.call("op", "POST", "/api/chat-projects", json={"name": "Kitchen remodel"})
    assert r.status_code == 201
    body = r.json()
    assert body["ok"] is True and body["created"] is True
    p = body["project"]
    assert p["folder_managed"] is True and p["name"] == "Kitchen remodel"
    assert Path(p["folder"]) == (world.root / "kitchen-remodel").resolve()
    assert stat.S_IMODE(os.stat(p["folder"]).st_mode) == 0o700
    assert {"id", "folder", "instructions_chars", "instructions_rev", "files", "sessions", "import_key",
            "created_at", "updated_at"} <= set(p)


def test_the_same_name_returns_the_existing_project(world):
    first = world.project("Beacon")
    r = world.call("op", "POST", "/api/chat-projects", json={"name": "  BEACON "})
    assert r.status_code == 200 and r.json()["created"] is False
    assert r.json()["project"]["id"] == first["id"]


def test_the_same_name_with_another_folder_is_taken(world):
    world.project("Beacon")
    r = world.call("op", "POST", "/api/chat-projects", json={"name": "beacon", "folder": world.folder("repo")})
    assert r.status_code == 409 and r.json()["kind"] == "name_taken"


def test_an_import_key_twice_is_one_project(world):
    a = world.project("Old name", import_key="local:p1")
    r = world.call("op", "POST", "/api/chat-projects", json={"name": "Other", "import_key": "local:p1"})
    assert r.status_code == 200 and r.json()["project"]["id"] == a["id"]


def test_an_existing_folder_is_used_as_it_is(world):
    f = world.folder("repo")
    p = world.project("Repo", folder=f)
    assert p["folder"] == f and p["folder_managed"] is False
    r = world.call("op", "POST", "/api/chat-projects", json={"name": "Again", "folder": f})
    assert r.status_code == 409 and r.json()["kind"] == "folder_in_use"


@pytest.mark.parametrize("which", ["missing", "relative", "denied", "home", "root", "agent_home", "inside", "around"])
def test_a_bad_folder_is_refused(world, tmp_path, which):
    outer = world.folder("outer")
    world.project("Outer", folder=outer)
    world.root.mkdir(exist_ok=True)
    (world.home / ".prometheus").mkdir(exist_ok=True)
    folder = {
        "missing": str(world.home / "nope"),
        "relative": "some/folder",
        "denied": str(tmp_path / "secret"),
        "home": str(world.home),
        "root": str(world.root),
        "agent_home": world.folder(".prometheus/workspace"),
        "inside": world.folder("outer/inner"),
        "around": str(world.home),
    }[which]
    r = world.call("op", "POST", "/api/chat-projects", json={"name": f"Bad {which}", "folder": folder})
    assert r.status_code == 400 and r.json()["kind"] == "bad_folder", r.text


def test_a_folder_around_another_project_is_refused(world):
    inner = world.folder("work/inner")
    world.project("Inner", folder=inner)
    r = world.call("op", "POST", "/api/chat-projects", json={"name": "Work", "folder": world.folder("work")})
    assert r.status_code == 400 and r.json()["kind"] == "bad_folder"


def test_an_empty_name_is_refused(world):
    r = world.call("op", "POST", "/api/chat-projects", json={"name": "   "})
    assert r.status_code == 400 and r.json()["kind"] == "name_required"


def test_a_symlinked_root_is_refused(world, tmp_path):
    real = tmp_path / "real-root"
    real.mkdir()
    world.root.symlink_to(real)
    r = world.call("op", "POST", "/api/chat-projects", json={"name": "X"})
    assert r.status_code == 400 and r.json()["kind"] == "bad_folder"
    assert list(real.iterdir()) == []


# ------------------------------------------------------------------------- rename, delete


def test_rename(world):
    a = world.project("Alpha")
    world.project("Beta")
    assert world.call("op", "PUT", f"/api/chat-projects/{a['id']}", json={"name": "beta"}).status_code == 409
    r = world.call("op", "PUT", f"/api/chat-projects/{a['id']}", json={"name": "ALPHA"})
    assert r.status_code == 200 and r.json()["project"]["name"] == "ALPHA"
    assert world.call("op", "PUT", "/api/chat-projects/cp_000000000000", json={"name": "Z"}).status_code == 404


def test_delete_unfiles_and_keeps_every_file(world):
    p = world.project()
    (Path(p["folder"]) / "PROMETHEUS.md").write_text("keep")
    sid = world.session()
    world.call("op", "PUT", f"/api/sessions/{sid}/project", json={"chat_project_id": p["id"]})
    r = world.call("op", "DELETE", f"/api/chat-projects/{p['id']}")
    assert r.status_code == 200 and r.json() == {"ok": True, "id": p["id"], "unfiled": 1}
    assert (Path(p["folder"]) / "PROMETHEUS.md").read_text() == "keep"
    assert world.conv.get_session_workspace(sid) is None


# ------------------------------------------------------------------------- filing


def test_filing_binds_the_workspace(world):
    p = world.project()
    sid = world.session()
    r = world.call("op", "PUT", f"/api/sessions/{sid}/project", json={"chat_project_id": p["id"]})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "session_id": sid, "chat_project_id": p["id"], "workspace": p["folder"]}
    assert world.conv.get_session_workspace(sid) == p["folder"]


def test_a_hand_set_workspace_needs_a_flag(world):
    p = world.project()
    sid = world.session()
    world.conv.set_session_workspace(sid, world.folder("mine"), set_by="rest")
    r = world.call("op", "PUT", f"/api/sessions/{sid}/project", json={"chat_project_id": p["id"]})
    assert r.status_code == 409 and r.json()["kind"] == "workspace_set" and r.json()["workspace"].endswith("mine")
    r = world.call("op", "PUT", f"/api/sessions/{sid}/project", json={"chat_project_id": p["id"], "keep_workspace": True})
    assert r.status_code == 200 and r.json()["workspace"].endswith("mine")


def test_unfiling(world):
    p = world.project()
    sid = world.session()
    world.call("op", "PUT", f"/api/sessions/{sid}/project", json={"chat_project_id": p["id"]})
    r = world.call("op", "PUT", f"/api/sessions/{sid}/project", json={"chat_project_id": None})
    assert r.status_code == 200 and r.json()["chat_project_id"] is None
    assert world.conv.get_session_workspace(sid) is None


@pytest.mark.parametrize(("sid", "status", "kind"), [("coding:run-1", 400, "machine_session"),
                                                     ("desktop:never-seen", 404, None)])
def test_machine_and_unknown_sessions(world, sid, status, kind):
    p = world.project()
    r = world.call("op", "PUT", f"/api/sessions/{sid}/project", json={"chat_project_id": p["id"]})
    assert r.status_code == status
    if kind:
        assert r.json()["kind"] == kind


def test_an_unknown_project(world):
    sid = world.session()
    r = world.call("op", "PUT", f"/api/sessions/{sid}/project", json={"chat_project_id": "cp_000000000000"})
    assert r.status_code == 404 and r.json()["kind"] == "unknown_project"


def test_a_chat_only_devices_session_needs_a_flag_to_bind(world):
    p = world.project()
    sid = world.session("a")
    r = world.call("op", "PUT", f"/api/sessions/{sid}/project", json={"chat_project_id": p["id"]})
    assert r.status_code == 409 and r.json()["kind"] == "device_session"
    assert world.call("op", "PUT", f"/api/sessions/{sid}/project",
                      json={"chat_project_id": p["id"], "keep_workspace": True}).status_code == 200
    assert world.call("op", "PUT", f"/api/sessions/{sid}/project",
                      json={"chat_project_id": p["id"], "device_session": True}).status_code == 200


def test_an_owner_devices_session_needs_no_flag(world):
    p = world.project()
    sid = world.session("owner")
    assert world.call("op", "PUT", f"/api/sessions/{sid}/project",
                      json={"chat_project_id": p["id"]}).status_code == 200


def test_a_folder_swapped_for_a_symlink_is_not_bound(world, tmp_path):
    f = world.folder("swap")
    p = world.project("Swap", folder=f)
    os.rmdir(f)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.symlink(elsewhere, f)
    sid = world.session()
    r = world.call("op", "PUT", f"/api/sessions/{sid}/project", json={"chat_project_id": p["id"]})
    assert r.status_code == 400 and r.json()["kind"] == "bad_folder"


@pytest.mark.parametrize("body", [{}, {"chat_project_id": 5}, {"chat_project_id": "x", "keep_workspace": "yes"}])
def test_a_bad_filing_body(world, body):
    sid = world.session()
    assert world.call("op", "PUT", f"/api/sessions/{sid}/project", json=body).status_code == 400


# ------------------------------------------------------------------------- session rows, new sessions


def test_session_rows_carry_the_project_durable_and_live(world):
    p = world.project()
    sid = world.session()
    world.call("op", "PUT", f"/api/sessions/{sid}/project", json={"chat_project_id": p["id"]})
    rows = {r["session_id"]: r for r in world.call("op", "GET", "/api/sessions").json()}
    assert rows[sid]["chat_project_id"] == p["id"] and rows[sid]["live"] is True
    world.mgr._sessions.pop(sid, None)
    rows = {r["session_id"]: r for r in world.call("op", "GET", "/api/sessions").json()}
    assert rows[sid]["chat_project_id"] == p["id"] and rows[sid]["live"] is False
    other = world.session()
    rows = {r["session_id"]: r for r in world.call("op", "GET", "/api/sessions").json()}
    assert rows[other]["chat_project_id"] is None


def test_a_new_session_can_start_in_a_project(world):
    p = world.project()
    r = world.call("op", "POST", "/api/sessions", json={"chat_project_id": p["id"]})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert r.json()["chat_project_id"] == p["id"]
    assert world.conv.get_session_workspace(sid) == p["folder"]


def test_a_chat_only_device_cannot_start_in_a_project(world):
    p = world.project()
    before = len(world.call("op", "GET", "/api/sessions").json())
    r = world.call("a", "POST", "/api/sessions", json={"chat_project_id": p["id"]})
    assert r.status_code == 403
    assert len(world.call("op", "GET", "/api/sessions").json()) == before


def test_a_new_session_in_an_unknown_project(world):
    r = world.call("op", "POST", "/api/sessions", json={"chat_project_id": "cp_000000000000"})
    assert r.status_code == 404


# ------------------------------------------------------------------------- what each caller sees


def test_the_owner_sees_everything(world):
    p = world.project()
    for who in ("op", "owner"):
        body = world.call(who, "GET", "/api/chat-projects").json()
        assert body["role"] == "owner" and isinstance(body["rev"], int)
        assert [x["id"] for x in body["projects"]] == [p["id"]] and "folder" in body["projects"][0]


def test_a_chat_only_device_sees_only_projects_with_its_sessions(world):
    mine = world.project("Mine")
    theirs = world.project("Theirs")
    own = world.session("a")
    other = world.session("b")
    for sid, pid in ((own, mine["id"]), (other, mine["id"]), (world.session("b"), theirs["id"])):
        world.call("op", "PUT", f"/api/sessions/{sid}/project",
                   json={"chat_project_id": pid, "keep_workspace": True})
    body = world.call("a", "GET", "/api/chat-projects").json()
    assert body == {"role": "chat_only", "projects": [{"id": mine["id"], "name": "Mine", "sessions": 1}]}
    one = world.call("a", "GET", f"/api/chat-projects/{mine['id']}").json()
    assert one == {"id": mine["id"], "name": "Mine", "session_ids": [own]}
    assert world.call("a", "GET", f"/api/chat-projects/{theirs['id']}").status_code == 404
    assert world.call("a", "GET", "/api/chat-projects/cp_000000000000").status_code == 404
    assert world.call("b", "GET", f"/api/chat-projects/{mine['id']}").json()["session_ids"] == [other]


def test_a_device_with_no_filed_sessions_sees_nothing(world):
    world.project()
    assert world.call("a", "GET", "/api/chat-projects").json() == {"role": "chat_only", "projects": []}


def test_the_owner_gets_one_project_with_its_sessions(world):
    p = world.project()
    sid = world.session()
    world.call("op", "PUT", f"/api/sessions/{sid}/project", json={"chat_project_id": p["id"]})
    one = world.call("op", "GET", f"/api/chat-projects/{p['id']}").json()
    assert one["session_ids"] == [sid] and one["folder"] == p["folder"]


# ------------------------------------------------------------------------- capability and frame


def test_the_capability():
    from prometheus.web.ws_server import WebSocketBridge

    assert "projects" in WebSocketBridge(loop_context=object(), config={})._server_capabilities()


def test_the_change_frame_goes_to_owner_sockets_only(world):
    from prometheus.config.api_token import GLOBAL_IDENTITY, DeviceIdentity

    owner_sock, device_sock = _Socket(), _Socket()
    world.bridge._clients.update({owner_sock, device_sock})
    world.bridge._ws_identity[owner_sock] = GLOBAL_IDENTITY
    world.bridge._ws_identity[device_sock] = DeviceIdentity(id=world.a["id"])
    p = world.project("Framed")
    frames = [f for f in owner_sock.frames if f["type"] == "chat_project_changed"]
    assert frames and frames[-1]["payload"]["project_id"] == p["id"] and frames[-1]["payload"]["change"] == "created"
    assert "Framed" not in json.dumps(frames) and p["folder"] not in json.dumps(frames)
    assert not [f for f in device_sock.frames if f["type"] == "chat_project_changed"]


# ------------------------------------------------------------------------- reviews of #933


@pytest.mark.parametrize("spelling", ["home_upper", "agent_upper", "around_home"])
def test_folder_checks_go_by_identity_not_spelling(world, spelling):
    (world.home / ".prometheus" / "state").mkdir(parents=True, exist_ok=True)
    folder = {
        "home_upper": str(world.home).upper() if os.path.exists(str(world.home).upper()) else None,
        "agent_upper": str(world.home / ".PROMETHEUS" / "state")
        if os.path.exists(world.home / ".PROMETHEUS") else None,
        "around_home": str(world.home.parent),
    }[spelling]
    if folder is None:
        pytest.skip("a case-sensitive filesystem")
    r = world.call("op", "POST", "/api/chat-projects", json={"name": "Sneaky", "folder": folder})
    assert r.status_code == 400 and r.json()["kind"] == "bad_folder"


def test_the_same_folder_in_another_case_is_in_use(world):
    f = world.folder("work/a")
    world.project("A", folder=f)
    other = f.replace("work/a", "WORK/A")
    if not os.path.exists(other):
        pytest.skip("a case-sensitive filesystem")
    r = world.call("op", "POST", "/api/chat-projects", json={"name": "B", "folder": other})
    assert r.status_code in (400, 409)
    assert len(world.call("op", "GET", "/api/chat-projects").json()["projects"]) == 1


def test_a_symlinked_agent_home_is_followed_for_the_check(world, tmp_path):
    real = tmp_path / "real-agent-home"
    (real / "skills").mkdir(parents=True)
    os.symlink(real, world.home / ".prometheus")
    r = world.call("op", "POST", "/api/chat-projects", json={"name": "Skills", "folder": str(real / "skills")})
    assert r.status_code == 400 and r.json()["kind"] == "bad_folder"


@pytest.mark.parametrize("root", [None, "", "relative/root"])
def test_a_null_empty_or_relative_root(world, root):
    world.app.state.chat_projects._config["projects"] = {"root": root}
    r = world.call("op", "POST", "/api/chat-projects", json={"name": "Rooted"})
    if root == "relative/root":
        assert r.status_code == 400 and r.json()["kind"] == "bad_folder"
    else:
        assert r.status_code == 201
        assert Path(r.json()["project"]["folder"]).parent == (world.home / "prometheus-projects").resolve()


def test_a_root_inside_a_denied_path_makes_nothing(world, tmp_path):
    world.app.state.chat_projects._config["projects"] = {"root": str(tmp_path / "secret" / "deep" / "root")}
    r = world.call("op", "POST", "/api/chat-projects", json={"name": "Nope"})
    assert r.status_code == 400 and r.json()["kind"] == "bad_folder"
    assert not (tmp_path / "secret" / "deep").exists()


def test_a_forgotten_session_no_longer_counts(world):
    p = world.project()
    a, b = world.session(), world.session()
    for sid in (a, b):
        world.call("op", "PUT", f"/api/sessions/{sid}/project", json={"chat_project_id": p["id"]})
    world.call("op", "DELETE", f"/api/sessions/{b}")
    one = world.call("op", "GET", f"/api/chat-projects/{p['id']}").json()
    assert one["sessions"] == 1 and one["session_ids"] == [a]


def test_a_large_instructions_file_has_the_true_rev(world):
    import hashlib

    p = world.project()
    data = ("x" * 1023 + "\n").encode() * 2048  # 2 MB
    (Path(p["folder"]) / "PROMETHEUS.md").write_bytes(data)
    got = world.call("op", "GET", f"/api/chat-projects/{p['id']}").json()
    assert got["instructions_rev"] == hashlib.sha256(data).hexdigest()[:16]
    assert got["instructions_chars"] == len(data)


def test_a_managed_create_never_collides_with_a_stored_folder(world):
    p = world.project("garden")
    os.rmdir(p["folder"])
    r = world.call("op", "POST", "/api/chat-projects", json={"name": "garden!"})
    assert r.status_code == 409 and r.json()["kind"] == "name_taken" or r.status_code in (200, 201)
    r = world.call("op", "POST", "/api/chat-projects", json={"name": "garden two"})
    assert r.status_code == 201


def test_an_existing_name_with_a_bad_folder_is_name_taken(world):
    world.project("Beacon")
    r = world.call("op", "POST", "/api/chat-projects", json={"name": "beacon", "folder": "/no/such/folder"})
    assert r.status_code == 409 and r.json()["kind"] == "name_taken"


def test_a_machine_gateway_in_a_project_creates_nothing(world):
    p = world.project()
    before = set(world.mgr._sessions)
    r = world.call("op", "POST", "/api/sessions", json={"gateway": "smoke", "chat_project_id": p["id"]})
    assert r.status_code == 400 and r.json()["kind"] == "machine_session"
    assert set(world.mgr._sessions) == before


def test_every_error_has_a_kind(world):
    p = world.project()
    r = world.call("op", "PUT", "/api/sessions/desktop:nobody/project", json={"chat_project_id": p["id"]})
    assert r.status_code == 404 and r.json()["kind"] == "unknown_session"
    r = world.call("op", "POST", "/api/sessions", json={"chat_project_id": 5})
    assert r.status_code == 400 and r.json()["kind"] == "bad_request"


def test_a_session_with_a_project_binding_is_not_claimable(world):
    from prometheus.web.session_scope import session_exists

    p = world.project()
    sid = world.call("op", "POST", "/api/sessions", json={"chat_project_id": p["id"]}).json()["session_id"]
    world.mgr._sessions.pop(sid, None)  # as after a restart, with no message yet
    assert session_exists(world.mgr, world.conv, sid)
    assert world.call("a", "PUT", f"/api/sessions/{sid}/title", json={"title": "mine now"}).status_code == 404
