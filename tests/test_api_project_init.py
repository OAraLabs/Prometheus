"""POST /api/project-init — Loop Manager from-scratch project bootstrap.

Side-effect tests against real tmp paths. The happy path creates the directory AND git-inits
it; each partial starting state (dir exists but no .git, already a git repo) reports honest
flags; every rejection (blank, relative, path-is-a-file) carries its OWN specific reason.
This endpoint creates directories and ``.git`` only — project FILES stay behind
/api/project-file's narrow scope (proven here by absence: init writes no other entries).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app({}))


def test_from_scratch_creates_dir_and_git(client, tmp_path: Path):
    target = tmp_path / "brand" / "new-project"  # nested → proves mkdir -p
    r = client.post("/api/project-init", json={"path": str(target)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["created_dir"] is True and body["initialized_git"] is True
    assert (target / ".git").is_dir()  # a real git repo landed on disk
    # Init seeds NOTHING else — files are /api/project-file's job (client-side templates).
    assert [p.name for p in target.iterdir()] == [".git"]


def test_existing_plain_dir_gets_git_only(client, tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "notes.txt").write_text("keep me")
    r = client.post("/api/project-init", json={"path": str(plain)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["created_dir"] is False and body["initialized_git"] is True
    assert (plain / ".git").is_dir()
    assert (plain / "notes.txt").read_text() == "keep me"  # non-destructive


def test_existing_git_repo_is_idempotent(client, tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    first = client.post("/api/project-init", json={"path": str(repo)})
    assert first.status_code == 200
    again = client.post("/api/project-init", json={"path": str(repo)})
    assert again.status_code == 200
    body = again.json()
    assert body["created_dir"] is False and body["initialized_git"] is False


def test_unlocks_project_file_flow(client, tmp_path: Path):
    """The whole point: after init, the previously-rejected /api/project-file write succeeds."""
    target = tmp_path / "fresh"
    pre = client.put(
        "/api/project-file", json={"repo": str(target), "name": "TASKS.md", "content": "x"}
    )
    assert pre.status_code == 400 and "does not exist" in pre.json()["error"]
    assert client.post("/api/project-init", json={"path": str(target)}).status_code == 200
    post = client.put(
        "/api/project-file",
        json={"repo": str(target), "name": "TASKS.md", "content": "# Tasks\n"},
    )
    assert post.status_code == 200, post.text
    assert (target / "TASKS.md").read_text() == "# Tasks\n"


def test_blank_path_rejected(client):
    r = client.post("/api/project-init", json={"path": ""})
    assert r.status_code == 400 and "path is required" in r.json()["error"]


def test_relative_path_rejected(client):
    r = client.post("/api/project-init", json={"path": "relative/project"})
    assert r.status_code == 400 and "must be absolute" in r.json()["error"]


def test_path_is_a_file_rejected(client, tmp_path: Path):
    f = tmp_path / "occupied"
    f.write_text("a file, not a dir")
    r = client.post("/api/project-init", json={"path": str(f)})
    assert r.status_code == 400 and "not a directory" in r.json()["error"]
    assert f.read_text() == "a file, not a dir"  # untouched
