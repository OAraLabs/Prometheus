"""Kanban board API — /api/projects + /api/stories CRUD, reorder, dispatch
(Beacon Desktop, daemon-backed store).

The store DB lands in a tmp ``PROMETHEUS_DATA_DIR`` so tests never touch the
real ~/.prometheus/data/kanban.db. Dispatch's chat send is exercised with a
stub ws_bridge; with no bridge it 503s WITHOUT stamping (web semantics).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path))
    app = create_app({})
    return TestClient(app), app


def _mk_project(c, name="Apollo"):
    return c.post("/api/projects", json={"name": name}).json()["project"]


def _mk_story(c, project_id=None, story_id="US-001", title="Land the eagle"):
    body = {"story_id": story_id, "title": title}
    if project_id:
        body["project_id"] = project_id
    return c.post("/api/stories", json=body).json()["story"]


def test_project_crud(client):
    c, _ = client
    assert c.get("/api/projects").json() == []
    p = _mk_project(c, "Apollo")
    assert p["name"] == "Apollo" and p["color"] == "#58a6ff" and p["id"]
    assert [x["name"] for x in c.get("/api/projects").json()] == ["Apollo"]
    r = c.put(f"/api/projects/{p['id']}", json={"name": "Apollo 11", "color": "#ff0000"})
    assert r.status_code == 200
    assert r.json()["project"]["name"] == "Apollo 11" and r.json()["project"]["color"] == "#ff0000"
    assert c.delete(f"/api/projects/{p['id']}").status_code == 200
    assert c.delete(f"/api/projects/{p['id']}").status_code == 404
    assert c.put("/api/projects/missing", json={"name": "x"}).status_code == 404


def test_project_create_requires_name(client):
    c, _ = client
    assert c.post("/api/projects", json={"name": "  "}).status_code == 400


def test_delete_project_orphans_its_stories(client):
    c, _ = client
    p = _mk_project(c)
    s = _mk_story(c, project_id=p["id"])
    assert c.delete(f"/api/projects/{p['id']}").status_code == 200
    # the story survives, with its project link cleared
    rows = c.get("/api/stories").json()
    assert len(rows) == 1 and rows[0]["id"] == s["id"] and rows[0]["project_id"] is None


def test_story_crud_and_filter(client):
    c, _ = client
    p = _mk_project(c)
    s = _mk_story(c, project_id=p["id"])
    assert s["story_id"] == "US-001" and s["status"] == "todo" and s["priority"] == "medium"
    assert s["labels"] == [] and s["project_id"] == p["id"]
    assert len(c.get("/api/stories").json()) == 1
    assert len(c.get("/api/stories", params={"project_id": p["id"]}).json()) == 1
    assert c.get("/api/stories", params={"project_id": "other"}).json() == []
    r = c.put(f"/api/stories/{s['id']}", json={"status": "in_progress", "labels": ["api", "p1"]})
    assert r.status_code == 200
    assert r.json()["story"]["status"] == "in_progress" and r.json()["story"]["labels"] == ["api", "p1"]
    assert c.delete(f"/api/stories/{s['id']}").status_code == 200
    assert c.delete(f"/api/stories/{s['id']}").status_code == 404


def test_story_create_validation(client):
    c, _ = client
    assert c.post("/api/stories", json={"title": "no story id"}).status_code == 400
    assert c.post("/api/stories", json={"story_id": "US-1"}).status_code == 400  # no title
    assert c.post("/api/stories", json={"story_id": "US-1", "title": "x", "status": "bogus"}).status_code == 400
    assert c.post("/api/stories", json={"story_id": "US-1", "title": "x", "priority": "urgent"}).status_code == 400


def test_story_update_validates_enums_and_404(client):
    c, _ = client
    s = _mk_story(c)
    assert c.put(f"/api/stories/{s['id']}", json={"status": "bogus"}).status_code == 400
    assert c.put("/api/stories/missing", json={"title": "x"}).status_code == 404


def test_reorder(client):
    c, _ = client
    a = _mk_story(c, story_id="US-1", title="a")
    b = _mk_story(c, story_id="US-2", title="b")
    r = c.post(
        "/api/stories/reorder",
        json={
            "items": [
                {"id": b["id"], "position": 0, "status": "in_progress"},
                {"id": a["id"], "position": 1, "status": "todo"},
            ]
        },
    )
    assert r.status_code == 200 and r.json()["count"] == 2
    rows = c.get("/api/stories").json()
    assert rows[0]["id"] == b["id"] and rows[0]["status"] == "in_progress"
    assert rows[1]["id"] == a["id"]
    assert c.post("/api/stories/reorder", json={"items": [{"id": a["id"], "position": 0, "status": "nope"}]}).status_code == 400
    assert c.post("/api/stories/reorder", json={"items": "notalist"}).status_code == 400


def test_dispatch_requires_bridge_then_stamps(client):
    c, app = client
    s = _mk_story(c)
    # no bridge → 503, story not stamped
    assert c.post(f"/api/stories/{s['id']}/dispatch", json={"session_key": "desktop:1"}).status_code == 503
    assert c.get("/api/stories").json()[0]["session_key"] is None

    sent = []

    class FakeBridge:
        async def dispatch_user_message(self, session_id, message, client_msg_id=None):
            sent.append((session_id, message))

    app.state.ws_bridge = FakeBridge()
    r = c.post(f"/api/stories/{s['id']}/dispatch", json={"session_key": "desktop:1"})
    assert r.status_code == 200
    story = r.json()["story"]
    assert story["session_key"] == "desktop:1" and story["status"] == "in_progress" and story["dispatched_at"]
    assert sent and sent[0][0] == "desktop:1"
    assert "Land the eagle" in sent[0][1] and "US-001" in sent[0][1]
    assert c.post("/api/stories/missing/dispatch", json={"session_key": "x"}).status_code == 404
    assert c.post(f"/api/stories/{s['id']}/dispatch", json={}).status_code == 400


def test_undispatch(client):
    c, app = client
    s = _mk_story(c)

    class FakeBridge:
        async def dispatch_user_message(self, *a, **k):
            return None

    app.state.ws_bridge = FakeBridge()
    c.post(f"/api/stories/{s['id']}/dispatch", json={"session_key": "desktop:1"})
    r = c.post(f"/api/stories/{s['id']}/undispatch")
    assert r.status_code == 200
    st = r.json()["story"]
    assert st["session_key"] is None and st["dispatched_at"] is None and st["status"] == "todo"
