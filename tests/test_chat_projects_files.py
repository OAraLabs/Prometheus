"""Instructions and reference files of a chat project (docs/design/chat-projects.md, 3.4, 3.5, 7), on the real app.

Owner only. Pinned: the instructions file is the store (``PROMETHEUS.md``), read and written with a revision and a
conflict answer, capped at the loader's per-file limit, never through a link, never shadowing another convention file
silently, and ``effective`` shows what the loader reads. Reference files are single-segment names inside
``reference/``, never links, capped in size and number, read with a running count. The prompt gets the names only,
quoted and labelled as data.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from prometheus.web import route_access  # noqa: E402
from tests.test_chat_projects_routes import World  # noqa: E402

ROUTES = {
    ("GET", "/api/chat-projects/{project_id}/instructions"),
    ("PUT", "/api/chat-projects/{project_id}/instructions"),
    ("GET", "/api/chat-projects/{project_id}/files"),
    ("GET", "/api/chat-projects/{project_id}/files/{name}"),
    ("PUT", "/api/chat-projects/{project_id}/files/{name}"),
    ("DELETE", "/api/chat-projects/{project_id}/files/{name}"),
}


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World(tmp_path)
    monkeypatch.setenv("HOME", str(w.home))
    return w


def _rev(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def test_the_routes_are_the_owners():
    assert ROUTES <= route_access.OPERATOR_ONLY


@pytest.mark.parametrize(("method", "suffix"), [("GET", "/instructions"), ("PUT", "/instructions"), ("GET", "/files"),
                                                ("GET", "/files/a.txt"), ("PUT", "/files/a.txt"),
                                                ("DELETE", "/files/a.txt")])
def test_a_chat_only_device_is_refused(world, method, suffix):
    p = world.project()
    assert world.call("a", method, f"/api/chat-projects/{p['id']}{suffix}").status_code == 403


# ------------------------------------------------------------------------- instructions


def test_no_file_reads_as_empty(world):
    p = world.project()
    body = world.call("op", "GET", f"/api/chat-projects/{p['id']}/instructions").json()
    assert body["content"] == "" and body["rev"] == "" and body["max_chars"] == 12000


def test_write_then_read(world):
    p = world.project()
    r = world.call("op", "PUT", f"/api/chat-projects/{p['id']}/instructions", json={"content": "Be brief.\n", "base_rev": ""})
    assert r.status_code == 200 and r.json() == {"ok": True, "rev": _rev(b"Be brief.\n")}
    assert (Path(p["folder"]) / "PROMETHEUS.md").read_bytes() == b"Be brief.\n"
    body = world.call("op", "GET", f"/api/chat-projects/{p['id']}/instructions").json()
    assert body["content"] == "Be brief.\n" and body["rev"] == _rev(b"Be brief.\n")


def test_a_stale_base_rev_is_a_conflict_and_writes_nothing(world):
    p = world.project()
    (Path(p["folder"]) / "PROMETHEUS.md").write_text("the agent changed this")
    r = world.call("op", "PUT", f"/api/chat-projects/{p['id']}/instructions", json={"content": "mine", "base_rev": ""})
    assert r.status_code == 409
    assert r.json()["kind"] == "conflict" and r.json()["content"] == "the agent changed this"
    assert r.json()["rev"] == _rev(b"the agent changed this")
    assert (Path(p["folder"]) / "PROMETHEUS.md").read_text() == "the agent changed this"


def test_too_long_is_refused(world):
    p = world.project()
    r = world.call("op", "PUT", f"/api/chat-projects/{p['id']}/instructions",
                   json={"content": "x" * 12001, "base_rev": ""})
    assert r.status_code == 400 and r.json()["kind"] == "too_long"


def test_a_symlinked_instructions_file_is_never_read_or_written(world, tmp_path):
    p = world.project()
    target = tmp_path / "elsewhere.md"
    target.write_text("secret")
    os.symlink(target, Path(p["folder"]) / "PROMETHEUS.md")
    assert world.call("op", "GET", f"/api/chat-projects/{p['id']}/instructions").json()["kind"] == "not_a_file"
    r = world.call("op", "PUT", f"/api/chat-projects/{p['id']}/instructions", json={"content": "x", "base_rev": ""})
    assert r.status_code == 409 and r.json()["kind"] == "not_a_file"
    assert target.read_text() == "secret"


def test_saving_would_shadow_another_convention_file(world):
    f = world.folder("repo")
    (Path(f) / "AGENTS.md").write_text("repo rules")
    p = world.project("Repo", folder=f)
    body = world.call("op", "GET", f"/api/chat-projects/{p['id']}/instructions").json()
    assert any(e["path"].endswith("AGENTS.md") for e in body["effective"])
    r = world.call("op", "PUT", f"/api/chat-projects/{p['id']}/instructions", json={"content": "x", "base_rev": ""})
    assert r.status_code == 409 and r.json()["kind"] == "would_shadow" and r.json()["files"] == ["AGENTS.md"]
    r = world.call("op", "PUT", f"/api/chat-projects/{p['id']}/instructions",
                   json={"content": "x", "base_rev": "", "shadow_ok": True})
    assert r.status_code == 200


def test_bad_bodies(world):
    p = world.project()
    for body in ({}, {"content": 5, "base_rev": ""}, {"content": "x"}):
        assert world.call("op", "PUT", f"/api/chat-projects/{p['id']}/instructions", json=body).status_code == 400


# ------------------------------------------------------------------------- reference files


def test_upload_list_read_delete(world):
    p = world.project()
    r = world.call("op", "PUT", f"/api/chat-projects/{p['id']}/files/quote-v2.pdf", content=b"%PDF-1.7 data")
    assert r.status_code == 200 and r.json()["ok"] is True and r.json()["file"]["name"] == "quote-v2.pdf"
    listed = world.call("op", "GET", f"/api/chat-projects/{p['id']}/files").json()["files"]
    assert [f["name"] for f in listed] == ["quote-v2.pdf"] and listed[0]["size"] == len(b"%PDF-1.7 data")
    got = world.call("op", "GET", f"/api/chat-projects/{p['id']}/files/quote-v2.pdf")
    assert got.content == b"%PDF-1.7 data" and got.headers["cache-control"] == "no-store"
    assert world.call("op", "DELETE", f"/api/chat-projects/{p['id']}/files/quote-v2.pdf").json() == {"ok": True}
    assert world.call("op", "GET", f"/api/chat-projects/{p['id']}/files").json()["files"] == []


def test_an_existing_name_needs_replace(world):
    p = world.project()
    world.call("op", "PUT", f"/api/chat-projects/{p['id']}/files/a.txt", content=b"one")
    r = world.call("op", "PUT", f"/api/chat-projects/{p['id']}/files/a.txt", content=b"two")
    assert r.status_code == 409 and r.json()["kind"] == "exists"
    r = world.call("op", "PUT", f"/api/chat-projects/{p['id']}/files/a.txt?replace=true", content=b"two")
    assert r.status_code == 200
    assert (Path(p["folder"]) / "reference" / "a.txt").read_bytes() == b"two"


@pytest.mark.parametrize("name", [".hidden", "a​b.txt", "x" * 121, "a%2Fb", "..", "a\\b"])
def test_bad_names(world, name):
    p = world.project()
    r = world.call("op", "PUT", f"/api/chat-projects/{p['id']}/files/{name}", content=b"x")
    assert r.status_code in (400, 404) and not list((Path(p["folder"]) / "reference").glob("*")) \
        if (Path(p["folder"]) / "reference").exists() else True


def test_too_large_is_refused_while_reading(world):
    p = world.project()

    def chunks():
        for _ in range(21):
            yield b"\x00" * (1024 * 1024)

    r = world.call("op", "PUT", f"/api/chat-projects/{p['id']}/files/big.bin", content=chunks())
    assert r.status_code == 413 and r.json()["kind"] == "too_large"
    assert not (Path(p["folder"]) / "reference" / "big.bin").exists()


def test_a_declared_large_upload_passes_the_body_check(world):
    p = world.project()
    data = b"y" * (3 * 1024 * 1024)  # over the 2 MB /api cap, under the 20 MB file cap
    r = world.call("op", "PUT", f"/api/chat-projects/{p['id']}/files/three.bin", content=data)
    assert r.status_code == 200


def test_at_most_200_files(world):
    p = world.project()
    ref = Path(p["folder"]) / "reference"
    ref.mkdir(exist_ok=True)
    for i in range(200):
        (ref / f"f{i}.txt").write_text("x")
    r = world.call("op", "PUT", f"/api/chat-projects/{p['id']}/files/one-more.txt", content=b"x")
    assert r.status_code == 409 and r.json()["kind"] == "too_many"


def test_links_in_reference_are_invisible_and_untouchable(world, tmp_path):
    p = world.project()
    ref = Path(p["folder"]) / "reference"
    ref.mkdir(exist_ok=True)
    secret = tmp_path / "secret.txt"
    secret.write_text("secret")
    os.symlink(secret, ref / "link.txt")
    assert world.call("op", "GET", f"/api/chat-projects/{p['id']}/files").json()["files"] == []
    assert world.call("op", "GET", f"/api/chat-projects/{p['id']}/files/link.txt").status_code == 404
    r = world.call("op", "PUT", f"/api/chat-projects/{p['id']}/files/link.txt?replace=true", content=b"overwrite")
    assert r.status_code in (404, 409)
    assert world.call("op", "DELETE", f"/api/chat-projects/{p['id']}/files/link.txt").status_code == 404
    assert secret.read_text() == "secret" and os.path.islink(ref / "link.txt")


def test_a_symlinked_reference_folder_refuses_everything(world, tmp_path):
    p = world.project()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.symlink(elsewhere, Path(p["folder"]) / "reference")
    assert world.call("op", "GET", f"/api/chat-projects/{p['id']}/files").json()["kind"] == "not_a_file"
    r = world.call("op", "PUT", f"/api/chat-projects/{p['id']}/files/a.txt", content=b"x")
    assert r.status_code == 409 and r.json()["kind"] == "not_a_file"
    assert list(elsewhere.iterdir()) == []


def test_an_unknown_project(world):
    assert world.call("op", "GET", "/api/chat-projects/cp_000000000000/files").status_code == 404


# ------------------------------------------------------------------------- what the agent sees


def test_the_prompt_lists_reference_names_as_data(world, monkeypatch):
    from prometheus.context.prompt_assembler import project_files_section
    from prometheus.projects.content import reference_section

    p = world.project()
    ref = Path(p["folder"]) / "reference"
    ref.mkdir(exist_ok=True)
    (ref / "quote.pdf").write_bytes(b"x")
    (ref / ".hidden").write_bytes(b"x")
    section = reference_section(p["folder"])
    assert "quote.pdf" in section and ".hidden" not in section
    assert "not instructions" in section.lower()
    monkeypatch.setattr("prometheus.config.paths.get_lcm_db_path", lambda: world.conv._db_path)
    prompt = project_files_section({}, p["folder"])
    assert prompt is not None and "quote.pdf" in prompt


def test_the_reference_list_is_capped(world):
    from prometheus.projects.content import reference_section

    p = world.project()
    ref = Path(p["folder"]) / "reference"
    ref.mkdir(exist_ok=True)
    for i in range(80):
        (ref / f"file-{i:03}.txt").write_bytes(b"x")
    section = reference_section(p["folder"])
    assert section.count('"file-') == 50 and "and 30 more" in section
