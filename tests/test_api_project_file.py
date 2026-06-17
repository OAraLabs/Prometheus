"""GET/PUT /api/project-file — Loop Manager remote project-file read/write (Option B).

Side-effect tests against a REAL git-repo fixture. The three allowed files round-trip
byte-identical; every other case (disallowed basename, subdir path, non-git, nonexistent,
missing file, blank name) is rejected with its OWN specific reason — proving the narrow scope
and that the daemon never returns a generic "could not write" (the swallowed error that made
the original bug hard to diagnose). Nothing here touches /api/files or its sandbox.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app({}))


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    r = tmp_path / "project"
    r.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=r, check=True)
    return r


@pytest.mark.parametrize("name", ["TASKS.md", "LOOP.md", "PROGRESS.md"])
def test_write_then_read_round_trips(client, git_repo, name):
    text = f"# {name}\n\n- [ ] remote write from a non-co-located host\n"
    w = client.put("/api/project-file", json={"repo": str(git_repo), "name": name, "content": text})
    assert w.status_code == 200, w.text
    assert (git_repo / name).read_text() == text  # landed on disk byte-identical
    r = client.get("/api/project-file", params={"repo": str(git_repo), "name": name})
    assert r.status_code == 200 and r.json()["content"] == text  # read back byte-identical


def test_disallowed_basename_rejected(client, git_repo):
    r = client.put("/api/project-file", json={"repo": str(git_repo), "name": "secrets.env", "content": "x"})
    assert r.status_code == 400 and "not permitted" in r.json()["error"]
    assert not (git_repo / "secrets.env").exists()  # nothing written


def test_subdir_path_rejected(client, git_repo):
    r = client.put("/api/project-file", json={"repo": str(git_repo), "name": "sub/TASKS.md", "content": "x"})
    assert r.status_code == 400 and "top-level filename" in r.json()["error"]  # distinct reason
    assert not (git_repo / "sub").exists()


def test_non_git_path_rejected(client, tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    r = client.put("/api/project-file", json={"repo": str(plain), "name": "TASKS.md", "content": "x"})
    assert r.status_code == 400 and "not a git repository" in r.json()["error"]  # distinct reason
    assert not (plain / "TASKS.md").exists()


def test_nonexistent_path_rejected(client, tmp_path):
    r = client.put("/api/project-file", json={"repo": str(tmp_path / "nope"), "name": "TASKS.md", "content": "x"})
    assert r.status_code == 400 and "does not exist" in r.json()["error"]  # distinct from non-git


def test_read_missing_file_is_404(client, git_repo):
    r = client.get("/api/project-file", params={"repo": str(git_repo), "name": "PROGRESS.md"})
    assert r.status_code == 404 and "file not found" in r.json()["error"]


def test_blank_name_rejected(client, git_repo):
    r = client.put("/api/project-file", json={"repo": str(git_repo), "name": "", "content": "x"})
    assert r.status_code == 400 and "name is required" in r.json()["error"]


def test_content_must_be_string(client, git_repo):
    r = client.put("/api/project-file", json={"repo": str(git_repo), "name": "TASKS.md", "content": 42})
    assert r.status_code == 400 and "content must be a string" in r.json()["error"]
