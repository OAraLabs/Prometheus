"""PUT /api/memory/current — Beacon's Memory editor write path.

Full-content replace of MEMORY.md / USER.md through the SAME FileMemoryStore the
agent's memory tool uses: per-entry sanitize + char budgets enforced, previous
content snapshotted to ~/.prometheus/memory-history/ BEFORE the write (every
edit reversible), over-budget content rejected with 400 without touching disk.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    home = tmp_path / "prometheus-home"
    home.mkdir()
    (home / "MEMORY.md").write_text("Lighthouse is the default room.\n", encoding="utf-8")
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(home))
    return TestClient(create_app({})), home


def test_replace_memory_and_read_back(client):
    c, home = client
    r = c.put("/api/memory/current", json={"memory": "Will runs Capstone Recruiting.\nBeacon is the desktop client."})
    assert r.status_code == 200
    assert r.json()["written"]["memory"] == 2
    body = c.get("/api/memory/current").json()
    assert body["memory"]["entry_count"] == 2
    assert "Lighthouse" not in body["memory"]["content"]


def test_previous_content_snapshotted_before_write(client):
    c, home = client
    c.put("/api/memory/current", json={"memory": "new fact"})
    snaps = list((home / "memory-history").glob("*-MEMORY.md"))
    assert len(snaps) == 1
    assert "Lighthouse" in snaps[0].read_text(encoding="utf-8")


def test_absent_key_leaves_file_untouched(client):
    c, home = client
    c.put("/api/memory/current", json={"user": "Will prefers direct answers."})
    body = c.get("/api/memory/current").json()
    assert "Lighthouse" in body["memory"]["content"]  # memory untouched
    assert body["user"]["entry_count"] == 1


def test_over_budget_rejected_without_write(client):
    c, home = client
    huge = "x" * 20_000
    r = c.put("/api/memory/current", json={"memory": huge})
    assert r.status_code == 400
    assert "budget" in r.json()["error"]
    body = c.get("/api/memory/current").json()
    assert "Lighthouse" in body["memory"]["content"]  # original intact, no snapshot burned
    assert not (home / "memory-history").exists()


def test_clear_with_empty_string(client):
    c, home = client
    r = c.put("/api/memory/current", json={"memory": ""})
    assert r.status_code == 200
    assert c.get("/api/memory/current").json()["memory"]["entry_count"] == 0


def test_non_string_rejected(client):
    c, _ = client
    assert c.put("/api/memory/current", json={"memory": 42}).status_code == 400
    assert c.put("/api/memory/current", json="nope").status_code == 400


def test_stale_base_conflicts_409_without_write(client):
    c, home = client
    r = c.put("/api/memory/current", json={"memory": "my edit", "base_memory": "what I loaded (stale)"})
    assert r.status_code == 409
    assert "changed since" in r.json()["error"]
    assert r.json()["current"] == "Lighthouse is the default room."
    body = c.get("/api/memory/current").json()
    assert "Lighthouse" in body["memory"]["content"]  # untouched
    assert not (home / "memory-history").exists()  # no snapshot burned


def test_matching_base_writes(client):
    c, _ = client
    r = c.put("/api/memory/current", json={"memory": "my edit", "base_memory": "Lighthouse is the default room."})
    assert r.status_code == 200
    assert c.get("/api/memory/current").json()["memory"]["content"] == "my edit"
