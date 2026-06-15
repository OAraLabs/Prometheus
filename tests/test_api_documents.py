"""/api/documents — confined disk-backed editor surface (Documents Editor, Phase A).

HTTP-boundary tests mirroring test_api_files.py: real bytes round-trip, real
diffs, and confinement (traversal / absolute / symlink) rejected at the route.
The suggest route's model call is stubbed — the parser/validation is what we pin
here; the live model is a Phase C concern.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    root = tmp_path / "documents"
    root.mkdir()
    (root / "notes.md").write_text("# Notes\n\nHello, the cat sat on the mat.\n", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "deep.txt").write_text("deep\n", encoding="utf-8")
    monkeypatch.setenv("PROMETHEUS_DOCUMENTS_DIR", str(root))
    return TestClient(create_app({})), root


# ── list / read ────────────────────────────────────────────────────────────


def test_list_root(client):
    c, _ = client
    body = c.get("/api/documents").json()
    assert body["path"] == ""
    names = {e["name"]: e["type"] for e in body["entries"]}
    assert names["notes.md"] == "file" and names["sub"] == "dir"


def test_read_text(client):
    c, _ = client
    body = c.get("/api/documents/content", params={"path": "notes.md"}).json()
    assert "Hello, the cat" in body["content"]
    assert body["binary"] is False


# ── save (PUT) round-trips real bytes ───────────────────────────────────────


def test_save_then_read_roundtrip(client):
    c, root = client
    r = c.put("/api/documents/content", json={"path": "letters/a.md", "content": "Dear friend\n"})
    assert r.status_code == 200 and r.json()["created"] is True
    assert (root / "letters" / "a.md").read_text(encoding="utf-8") == "Dear friend\n"
    got = c.get("/api/documents/content", params={"path": "letters/a.md"}).json()
    assert got["content"] == "Dear friend\n"


# ── edit applies + real diff; loud errors are 422 and leave the file alone ───


def test_edit_applies_and_returns_diff(client):
    c, root = client
    r = c.post("/api/documents/edit", json={"path": "notes.md", "old_str": "the cat", "new_str": "the dog"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "-Hello, the cat sat" in body["diff"] and "+Hello, the dog sat" in body["diff"]
    assert "the dog sat on the mat" in (root / "notes.md").read_text(encoding="utf-8")


def test_edit_no_match_is_422_untouched(client):
    c, root = client
    before = (root / "notes.md").read_text(encoding="utf-8")
    r = c.post("/api/documents/edit", json={"path": "notes.md", "old_str": "ABSENT", "new_str": "x"})
    assert r.status_code == 422 and r.json()["ok"] is False
    assert "NO MATCH" in r.json()["error"]
    assert (root / "notes.md").read_text(encoding="utf-8") == before


def test_edit_multi_match_is_422_untouched(client):
    c, root = client
    before = (root / "notes.md").read_text(encoding="utf-8")
    r = c.post("/api/documents/edit", json={"path": "notes.md", "old_str": "the", "new_str": "THE"})
    assert r.status_code == 422
    assert "MATCHES" in r.json()["error"]
    assert (root / "notes.md").read_text(encoding="utf-8") == before


def test_edit_accepts_find_replace_aliases(client):
    c, root = client
    r = c.post("/api/documents/edit", json={"path": "notes.md", "find": "the cat", "replace": "the dog"})
    assert r.status_code == 200 and r.json()["ok"] is True


# ── confinement at the route ────────────────────────────────────────────────


def test_traversal_blocked(client):
    c, _ = client
    assert c.get("/api/documents", params={"path": "../.."}).status_code == 403
    assert c.get("/api/documents/content", params={"path": "../../../etc/passwd"}).status_code == 403


def test_save_traversal_blocked_and_untouched(client, tmp_path):
    c, _ = client
    outside = tmp_path / "outside.txt"
    outside.write_text("SECRET\n", encoding="utf-8")
    r = c.put("/api/documents/content", json={"path": str(outside), "content": "PWNED"})
    assert r.status_code == 403
    assert outside.read_text(encoding="utf-8") == "SECRET\n"


def test_symlink_escape_blocked(client, tmp_path):
    c, root = client
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret\n", encoding="utf-8")
    link = root / "escape.md"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("symlinks not supported on this platform")
    assert c.get("/api/documents/content", params={"path": "escape.md"}).status_code == 403


def test_missing_is_404(client):
    c, _ = client
    assert c.get("/api/documents", params={"path": "nope"}).status_code == 404
    assert c.get("/api/documents/content", params={"path": "nope.md"}).status_code == 404


# ── suggest (model stubbed) — validates, never writes ───────────────────────


def test_suggest_returns_validated_redlines_without_writing(client, monkeypatch):
    c, root = client
    before = (root / "notes.md").read_text(encoding="utf-8")

    async def stub_gen(provider, model, content, instruction, **kw):
        assert "Hello" in content
        return [
            {"find": "Hello", "replace": "Greetings", "reason": "formal"},
            {"find": "the", "replace": "THE", "reason": "ambiguous"},
            {"find": "NOPE", "replace": "x", "reason": "absent"},
        ]

    monkeypatch.setattr("prometheus.documents.ai.generate_suggestions", stub_gen)
    r = c.post("/api/documents/suggest", json={"path": "notes.md", "instruction": "formalize"})
    assert r.status_code == 200
    edits = {e["find"]: e for e in r.json()["edits"]}
    assert edits["Hello"]["applicable"] is True
    assert edits["the"]["applicable"] is False and "ambiguous" in edits["the"]["note"]
    assert edits["NOPE"]["applicable"] is False
    # redline mode wrote nothing
    assert (root / "notes.md").read_text(encoding="utf-8") == before


def test_suggest_requires_path_and_instruction(client):
    c, _ = client
    assert c.post("/api/documents/suggest", json={"path": "notes.md"}).status_code == 400
    assert c.post("/api/documents/suggest", json={"instruction": "x"}).status_code == 400
