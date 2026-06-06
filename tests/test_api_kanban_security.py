"""Adversarial: /api/projects + /api/stories — bearer enforcement, SQL-injection
resistance (queries are parameterized), enum validation, malformed + oversized
bodies rejected.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path))
    return TestClient(create_app({}))


@pytest.fixture()
def client_tok(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path))
    return TestClient(create_app({"web": {"api_token": "secret"}}))


def test_requires_bearer_when_token_set(client_tok):
    c = client_tok
    assert c.get("/api/projects").status_code == 401
    assert c.get("/api/stories").status_code == 401
    assert c.post("/api/projects", json={"name": "x"}).status_code == 401
    assert c.post("/api/stories", json={"story_id": "US-1", "title": "t"}).status_code == 401
    assert c.get("/api/projects", headers={"Authorization": "Bearer wrong"}).status_code == 401
    h = {"Authorization": "Bearer secret"}
    assert c.get("/api/projects", headers=h).status_code == 200
    assert c.post("/api/projects", json={"name": "x"}, headers=h).status_code == 201


def test_sql_injection_in_story_fields_stored_literally(client):
    c = client
    evil = "Robert'); DROP TABLE stories;--"
    r = c.post("/api/stories", json={"story_id": evil, "title": evil, "description": evil})
    assert r.status_code == 201
    assert r.json()["story"]["title"] == evil  # stored verbatim (parameterized), not executed
    # Table intact: the list still works and a second insert succeeds.
    assert len(c.get("/api/stories").json()) == 1
    assert c.post("/api/stories", json={"story_id": "US-2", "title": "still works"}).status_code == 201
    assert len(c.get("/api/stories").json()) == 2


def test_sql_injection_in_project_filter_is_parameterized(client):
    c = client
    c.post("/api/stories", json={"story_id": "US-1", "title": "t"})
    # Injection in the ?project_id= filter is bound as a literal → matches nothing, table untouched.
    r = c.get("/api/stories", params={"project_id": "' OR '1'='1"})
    assert r.status_code == 200 and r.json() == []
    assert len(c.get("/api/stories").json()) == 1


def test_project_update_injection_literal(client):
    c = client
    pid = c.post("/api/projects", json={"name": "p"}).json()["project"]["id"]
    evil = "x', color='#000000', name='HIJACKED"  # tries to smuggle extra columns
    r = c.put(f"/api/projects/{pid}", json={"description": evil})
    assert r.status_code == 200
    proj = r.json()["project"]
    assert proj["description"] == evil and proj["name"] == "p"  # name NOT hijacked


def test_malformed_json_rejected(client):
    c = client
    r = c.post("/api/stories", content="{not valid json", headers={"content-type": "application/json"})
    assert r.status_code in (400, 422)


def test_oversized_body_rejected(client):
    c = client
    big = "x" * (3 * 1024 * 1024)  # 3 MB > the 2 MB cap
    r = c.post("/api/stories", content=big, headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_enum_validation_on_writes(client):
    c = client
    assert c.post("/api/stories", json={"story_id": "US-1", "title": "t", "status": "PWNED"}).status_code == 400
    assert c.post("/api/stories", json={"story_id": "US-1", "title": "t", "priority": "CRITICAL"}).status_code == 400
    assert c.post("/api/stories", json={"title": "no id"}).status_code == 400  # missing story_id
    assert c.post("/api/projects", json={"name": "   "}).status_code == 400  # blank name
