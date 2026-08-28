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
    assert hit["in_title"] is True, "matching the title is a different answer from matching the body"


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


def test_a_path_only_match_is_not_reported_as_a_title_match(client):
    """A page whose PATH carries the term but whose TITLE does not is not a title hit.

    `in_title` is what a client renders a "matched the name" badge from. Folding the path into
    it makes a page called topics/vault-runbook.md titled "Daily Notes" claim the word is in its
    title. The existing title-match test cannot catch this: its fixture matches title AND path
    at once, so it passes under either definition.
    """
    c, wiki, _ = client
    (wiki / "topics" / "vault-runbook.md").write_text("# Daily Notes\n\nnothing relevant here\n")
    hit = c.get("/api/wiki/search?q=runbook").json()["results"][0]
    assert hit["path"] == "topics/vault-runbook.md"
    assert hit["in_title"] is False, "'runbook' is in the filename, not in the title 'Daily Notes'"
    assert hit["in_path"] is True, "the match must still be attributable to the path"


def test_search_does_not_follow_a_symlink_out_of_the_root(client):
    """/api/wiki/page refuses this; the walk must refuse it too.

    rglob('*.md') YIELDS a symlinked file (it declines to recurse into symlinked *directories*,
    which is a different thing), and read_text() then follows it. The snippet is the leak.
    """
    c, wiki, tmp_path = client
    try:
        (wiki / "topics" / "escape.md").symlink_to(tmp_path / "secret.md")
    except OSError:
        pytest.skip("symlinks unavailable")
    r = c.get("/api/wiki/search?q=secret")
    assert "not for the wiki reader" not in r.text, "the walk read through a symlink out of the root"


def test_a_listing_does_not_title_a_page_that_points_outside(client):
    """The listing read every .md directly to find its H1 — including symlinked ones.

    `list_dir` uses iterdir(), which lists a symlink as an ordinary file, and the title scan
    then opened it. The heading of a file anywhere on the box became a wiki page title.
    """
    c, wiki, tmp_path = client
    try:
        (wiki / "topics" / "escape.md").symlink_to(tmp_path / "secret.md")
    except OSError:
        pytest.skip("symlinks unavailable")
    entries = c.get("/api/wiki/pages?path=topics").json()["entries"]
    assert "Secret" not in [e.get("title") for e in entries], "an out-of-root heading became a title"
    # /api/wiki/page refuses this path, so a row for it is only a link that 403s.
    assert all(e["name"] != "escape.md" for e in entries)


def test_the_cap_says_that_it_capped(client):
    """`returned` alone cannot separate "that is all of them" from "that is the first N"."""
    c, wiki, _ = client
    for i in range(4):
        (wiki / "topics" / f"cap{i}.md").write_text("# Cap\n\ncapybara\n")
    body = c.get("/api/wiki/search?q=capybara&limit=2").json()
    assert body["returned"] == 2
    assert body["truncated"] is True, "four pages match and two came back"
    assert c.get("/api/wiki/search?q=capybara&limit=10").json()["truncated"] is False


def test_an_unreadable_page_is_skipped_not_a_500(client):
    """DocumentsService.read() opens directly and does not wrap OSError, so one bad page must
    not take out the whole listing."""
    import os
    c, wiki, _ = client
    bad = wiki / "topics" / "locked.md"
    bad.write_text("# Locked\n")
    os.chmod(bad, 0o000)
    try:
        if os.access(bad, os.R_OK):
            pytest.skip("cannot make a file unreadable (running as root?)")
        r = c.get("/api/wiki/pages?path=topics")
        assert r.status_code == 200, "one unreadable page 500'd the listing"
        assert all(e["name"] != "locked.md" for e in r.json()["entries"])
        assert c.get("/api/wiki/search?q=rotation").status_code == 200
    finally:
        os.chmod(bad, 0o644)
