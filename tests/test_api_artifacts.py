"""GET /api/artifacts + /api/artifacts/{id} — the agent's outbox, served by content id.

Beacon (remote) lists the manifest and downloads deliverables the agent saved
into ~/.prometheus/files. The wire contract is CONTENT IDS, never paths: ids are
sha256 prefixes of the bytes, so they survive renames, dedup identical files,
and no client-supplied string ever touches filesystem resolution. Symlinks and
dotfiles are never indexed; unknown/malformed ids 404.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    (outbox / "report.docx").write_bytes(b"PK\x03\x04 docx bytes")
    (outbox / "sub").mkdir()
    (outbox / "sub" / "chart.pdf").write_bytes(b"%PDF-1.7 chart")
    (outbox / ".secret").write_text("hidden", encoding="utf-8")
    secret = tmp_path / "outside.txt"
    secret.write_text("keychain", encoding="utf-8")
    (outbox / "leak.txt").symlink_to(secret)
    monkeypatch.setenv("PROMETHEUS_ARTIFACTS_DIR", str(outbox))
    return TestClient(create_app({})), outbox


def _manifest(c):
    r = c.get("/api/artifacts")
    assert r.status_code == 200
    return r.json()["artifacts"]


def test_manifest_lists_regular_files_only(client):
    c, _ = client
    arts = _manifest(c)
    names = {a["name"] for a in arts}
    assert names == {"report.docx", "chart.pdf"}  # no dotfiles, no symlinks
    by_name = {a["name"]: a for a in arts}
    assert by_name["chart.pdf"]["path"] == "sub/chart.pdf"  # recursive, outbox-relative
    assert by_name["report.docx"]["mime"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert all(len(a["id"]) == 16 for a in arts)


def test_download_by_id_returns_bytes_and_filename(client):
    c, _ = client
    art = next(a for a in _manifest(c) if a["name"] == "report.docx")
    r = c.get(f"/api/artifacts/{art['id']}")
    assert r.status_code == 200
    assert r.content == b"PK\x03\x04 docx bytes"
    assert "report.docx" in r.headers.get("content-disposition", "")


def test_id_survives_rename(client):
    c, outbox = client
    before = next(a for a in _manifest(c) if a["name"] == "report.docx")
    (outbox / "report.docx").rename(outbox / "final-report.docx")
    after = next(a for a in _manifest(c) if a["name"] == "final-report.docx")
    assert after["id"] == before["id"]  # content-addressed: the rename is invisible
    assert c.get(f"/api/artifacts/{after['id']}").status_code == 200


def test_identical_bytes_share_an_id(client):
    c, outbox = client
    (outbox / "copy.docx").write_bytes(b"PK\x03\x04 docx bytes")
    ids = {a["id"] for a in _manifest(c) if a["name"].endswith(".docx")}
    assert len(ids) == 1  # dedup by content


def test_unknown_and_malformed_ids_404(client):
    c, _ = client
    assert c.get("/api/artifacts/0123456789abcdef").status_code == 404  # well-formed, unknown
    assert c.get("/api/artifacts/deadbeef").status_code == 404  # too short
    assert c.get("/api/artifacts/..%2F..%2Fetc%2Fpasswd").status_code == 404  # not an id at all


def test_departed_file_404s_after_delete(client):
    c, outbox = client
    art = next(a for a in _manifest(c) if a["name"] == "report.docx")
    (outbox / "report.docx").unlink()
    assert c.get(f"/api/artifacts/{art['id']}").status_code == 404


def test_bearer_gate_covers_artifacts(client, monkeypatch, tmp_path):
    _, outbox = client
    monkeypatch.setenv("PROMETHEUS_ARTIFACTS_DIR", str(outbox))
    gated = TestClient(create_app({"web": {"api_token": "sekrit"}}))
    assert gated.get("/api/artifacts").status_code == 401
    ok = gated.get("/api/artifacts", headers={"Authorization": "Bearer sekrit"})
    assert ok.status_code == 200
