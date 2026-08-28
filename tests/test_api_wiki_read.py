"""GET /api/wiki/{pages,page,search} — the wiki read surface (branch feat/api-wiki-read).

/api/wiki/stats has counted pages for months without offering a way to read one, so "what does
the agent know about X" could only be answered by asking the agent. These routes answer it
directly — and because one of them takes a client-supplied PATH, the confinement tests are the
point of this file, not a formality.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    wiki = tmp_path / "wiki"
    (wiki / "topics").mkdir(parents=True)
    (wiki / "projects").mkdir()
    (wiki / "topics" / "kling.md").write_text("# Kling API\n\nThe kling key rotation happens monthly.\n")
    (wiki / "topics" / "untitled.md").write_text("no heading here, just prose about beacons\n")
    (wiki / "projects" / "beacon.md").write_text("# Beacon Desktop\n\nAn Electron client.\n")
    (wiki / "notes.txt").write_text("not a wiki page")
    # a secret OUTSIDE the wiki root — the thing confinement exists to protect
    (tmp_path / "secret.md").write_text("# Secret\nnot for the wiki reader\n")
    monkeypatch.setattr("prometheus.web.server.get_wiki_root", lambda: wiki)
    return TestClient(create_app({})), wiki, tmp_path


def test_listing_the_root_shows_the_sections(client):
    c, _, _ = client
    body = c.get("/api/wiki/pages").json()
    names = {e["name"] for e in body["entries"]}
    assert {"topics", "projects"} <= names


def test_listing_a_section_titles_each_page(client):
    c, _, _ = client
    entries = {e["name"]: e for e in c.get("/api/wiki/pages?path=topics").json()["entries"]}
    assert entries["kling.md"]["title"] == "Kling API", "the H1 is the title"
    # A page with no heading must still be nameable, or it renders as a blank row.
    assert entries["untitled.md"]["title"] == "untitled"


def test_reading_a_page_returns_its_markdown(client):
    c, _, _ = client
    body = c.get("/api/wiki/page?path=topics/kling.md").json()
    assert "kling key rotation" in body["content"]


def test_a_non_markdown_path_is_refused(client):
    c, _, _ = client
    r = c.get("/api/wiki/page?path=notes.txt")
    assert r.status_code == 400


# ── confinement: the reason this file exists ────────────────────────────────────────────────

@pytest.mark.parametrize(
    "attempt",
    [
        "../secret.md",
        "../../secret.md",
        "topics/../../secret.md",
        "/etc/passwd",
        "//etc/passwd",
        "topics/../../../../../../etc/passwd",
    ],
)
def test_traversal_cannot_escape_the_wiki_root(client, attempt):
    c, _, _ = client
    r = c.get(f"/api/wiki/page?path={attempt}")
    assert r.status_code >= 400, f"{attempt} was READ — confinement failed"
    # Assert on the CONTENT that would prove a leak, not on a word the refusal message itself
    # uses: the error says "escapes the documents root: …", so a bare "root:" check fails on a
    # correct refusal. `root:x:` only appears in a real /etc/passwd.
    assert "not for the wiki reader" not in r.text, "the out-of-root file was served"
    assert "root:x:" not in r.text, "/etc/passwd was served"


def test_a_symlink_out_of_the_root_is_refused(client):
    """The case a 'starts with' check gets wrong: the path looks inside, the file is outside."""
    c, wiki, tmp_path = client
    link = wiki / "topics" / "escape.md"
    try:
        link.symlink_to(tmp_path / "secret.md")
    except OSError:
        pytest.skip("symlinks unavailable")
    r = c.get("/api/wiki/page?path=topics/escape.md")
    assert "not for the wiki reader" not in r.text, "a symlink walked out of the root"


def test_traversal_in_a_LISTING_path_is_also_refused(client):
    c, _, _ = client
    r = c.get("/api/wiki/pages?path=../")
    assert r.status_code >= 400 or all(e["name"] != "secret.md" for e in r.json().get("entries", []))


# ── search ──────────────────────────────────────────────────────────────────────────────────

def test_search_finds_a_page_by_body_text(client):
    c, _, _ = client
    body = c.get("/api/wiki/search?q=rotation").json()
    assert body["returned"] == 1
    hit = body["results"][0]
    assert hit["path"] == "topics/kling.md"
    assert hit["section"] == "topics"
    assert "rotation" in hit["snippet"]
    # a body match is not a title match, and the client should be able to say which
    assert hit["in_title"] is False


def test_search_marks_a_title_match_differently(client):
    c, _, _ = client
    hit = c.get("/api/wiki/search?q=kling").json()["results"][0]
    assert hit["in_title"] is True, "matching the title/path is a different answer from matching the body"


def test_search_is_case_insensitive_and_capped(client):
    c, _, _ = client
    assert c.get("/api/wiki/search?q=KLING").json()["returned"] == 1
    assert c.get("/api/wiki/search?q=beacon&limit=1").json()["returned"] == 1


def test_a_too_short_query_returns_nothing_rather_than_everything(client):
    c, _, _ = client
    assert c.get("/api/wiki/search?q=a").json()["returned"] == 0


def test_search_never_leaves_the_wiki_root(client):
    """The secret is one directory up and contains a word that IS in the wiki."""
    c, _, _ = client
    body = c.get("/api/wiki/search?q=secret").json()
    assert body["returned"] == 0, "the walk reached outside the wiki root"
