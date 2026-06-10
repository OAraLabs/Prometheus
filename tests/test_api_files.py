"""GET /api/files + /api/files/read — sandboxed workspace file browser (Beacon Desktop Phase 6).

Beacon Desktop runs REMOTE from this host, so it browses the agent workspace
(get_workspace_dir()) through these endpoints rather than the local filesystem.
Every path must stay confined to the workspace root: .. / absolute / symlink
escapes are rejected (the resolver follows symlinks, so an out-of-root link
resolves outside and is denied).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "hello.txt").write_text("hello world", encoding="utf-8")
    (ws / "sub").mkdir()
    (ws / "sub" / "nested.md").write_text("# nested", encoding="utf-8")
    (ws / "blob.bin").write_bytes(b"\x00\x01\x02BINARY")
    monkeypatch.setenv("PROMETHEUS_WORKSPACE_DIR", str(ws))
    monkeypatch.delenv("PROMETHEUS_FILES_ROOT", raising=False)  # isolate from host config
    return TestClient(create_app({})), ws


def test_list_root(client):
    c, _ = client
    r = c.get("/api/files")
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == ""
    names = {e["name"]: e for e in body["entries"]}
    assert names["sub"]["type"] == "dir"
    assert names["hello.txt"]["type"] == "file" and names["hello.txt"]["size"] == 11
    assert body["entries"][0]["type"] == "dir"  # dirs sorted before files


def test_list_subdir(client):
    c, _ = client
    r = c.get("/api/files", params={"path": "sub"})
    assert r.status_code == 200
    assert [e["name"] for e in r.json()["entries"]] == ["nested.md"]


def test_read_text(client):
    c, _ = client
    body = c.get("/api/files/read", params={"path": "hello.txt"}).json()
    assert body["content"] == "hello world"
    assert body["binary"] is False and body["truncated"] is False and body["size"] == 11


def test_read_binary(client):
    c, _ = client
    body = c.get("/api/files/read", params={"path": "blob.bin"}).json()
    assert body["binary"] is True and body["content"] == ""


def test_traversal_blocked(client):
    c, _ = client
    assert c.get("/api/files", params={"path": "../.."}).status_code == 403
    assert c.get("/api/files/read", params={"path": "../../../etc/passwd"}).status_code == 403


def test_absolute_path_is_contained_not_escaped(client):
    # A leading slash is stripped → treated as workspace-relative, never an absolute escape.
    c, _ = client
    r = c.get("/api/files/read", params={"path": "/etc/passwd"})
    assert r.status_code == 404  # ws/etc/passwd doesn't exist; NOT the real /etc/passwd


def test_symlink_escape_blocked(client, tmp_path):
    c, ws = client
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    link = ws / "escape"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("symlinks not supported on this platform")
    assert c.get("/api/files/read", params={"path": "escape"}).status_code == 403


def test_missing(client):
    c, _ = client
    assert c.get("/api/files", params={"path": "nope"}).status_code == 404
    assert c.get("/api/files/read", params={"path": "nope.txt"}).status_code == 404


def test_truncation(client):
    c, ws = client
    (ws / "big.txt").write_text("x" * (300 * 1024), encoding="utf-8")
    body = c.get("/api/files/read", params={"path": "big.txt"}).json()
    assert body["truncated"] is True
    assert body["size"] == 300 * 1024
    assert len(body["content"]) == 256 * 1024


def test_files_root_env_overrides_workspace(tmp_path, monkeypatch):
    """PROMETHEUS_FILES_ROOT repoints the BROWSE root without touching the
    workspace var (which doubles as an image_generate write root)."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "ws-only.txt").write_text("x", encoding="utf-8")
    browse = tmp_path / "agent-home"
    browse.mkdir()
    (browse / "MEMORY.md").write_text("# memory", encoding="utf-8")
    monkeypatch.setenv("PROMETHEUS_WORKSPACE_DIR", str(ws))
    monkeypatch.setenv("PROMETHEUS_FILES_ROOT", str(browse))
    c = TestClient(create_app({}))
    names = {e["name"] for e in c.get("/api/files").json()["entries"]}
    assert "MEMORY.md" in names and "ws-only.txt" not in names
    # confinement still applies to the new root
    assert c.get("/api/files", params={"path": "../"}).status_code == 403
