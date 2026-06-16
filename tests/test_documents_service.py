"""Documents service — side-effect tests (Phase A, Documents Editor).

Side effects, not call-counting: every test asserts real bytes on disk, a real
diff, or a real rejection-that-left-the-filesystem-untouched. The confinement
tests are the load-bearing ones — a documents root that can be escaped is the
whole-sprint failure mode.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.documents import DocumentsError, DocumentsService
from prometheus.permissions.checker import SecurityGate


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "documents"
    r.mkdir()
    (r / "notes.md").write_text("# Notes\n\nHello, the cat sat on the mat.\n")
    (r / "sub").mkdir()
    (r / "sub" / "deep.txt").write_text("deep\n")
    return r


@pytest.fixture
def svc(root: Path) -> DocumentsService:
    return DocumentsService(root)


# --------------------------------------------------------------------------- #
# Round-trip: save then read returns the real bytes
# --------------------------------------------------------------------------- #


def test_save_then_read_roundtrips_real_bytes(svc: DocumentsService):
    body = "line one\nline two — café ☕\n"
    info = svc.save("new/letter.md", body)
    assert info["created"] is True
    assert info["bytes_written"] == len(body.encode("utf-8"))
    got = svc.read("new/letter.md")
    assert got["content"] == body
    assert got["binary"] is False
    # And it is really on disk under the root, nowhere else.
    assert (svc.root / "new" / "letter.md").read_text(encoding="utf-8") == body


def test_save_overwrites_existing_reports_not_created(svc: DocumentsService):
    info = svc.save("notes.md", "# Replaced\n")
    assert info["created"] is False
    assert svc.read("notes.md")["content"] == "# Replaced\n"


def test_list_dir_lists_children(svc: DocumentsService):
    rel, entries = svc.list_dir("")
    assert rel == ""
    names = {(e.name, e.type) for e in entries}
    assert ("notes.md", "file") in names
    assert ("sub", "dir") in names


def test_read_missing_is_404(svc: DocumentsService):
    with pytest.raises(DocumentsError) as exc:
        svc.read("nope.md")
    assert exc.value.status == 404


# --------------------------------------------------------------------------- #
# Edit: applies and returns a real diff; loud errors leave the file untouched
# --------------------------------------------------------------------------- #


async def test_apply_edit_changes_bytes_and_returns_real_diff(svc: DocumentsService):
    before = (svc.root / "notes.md").read_text()
    result = await svc.apply_edit("notes.md", "the cat", "the dog")
    assert result.ok is True
    # A real unified diff over the real change.
    assert "-Hello, the cat sat" in result.diff
    assert "+Hello, the dog sat" in result.diff
    after = (svc.root / "notes.md").read_text()
    assert "the dog sat on the mat" in after
    assert after != before


async def test_edit_no_match_returns_loud_error_untouched(svc: DocumentsService):
    before = (svc.root / "notes.md").read_text()
    result = await svc.apply_edit("notes.md", "ABSENT TEXT", "x")
    assert result.ok is False
    assert "NO MATCH" in result.error  # the primitive's loud error, unchanged
    assert result.diff == ""
    assert (svc.root / "notes.md").read_text() == before  # not touched


async def test_edit_multi_match_returns_loud_error_untouched(svc: DocumentsService):
    # "the" appears twice in notes.md → ambiguous.
    before = (svc.root / "notes.md").read_text()
    result = await svc.apply_edit("notes.md", "the", "THE")
    assert result.ok is False
    assert "MATCHES" in result.error  # "2 MATCHES ... is ambiguous"
    assert (svc.root / "notes.md").read_text() == before


async def test_edit_empty_old_str_is_loud_error(svc: DocumentsService):
    result = await svc.apply_edit("notes.md", "", "x")
    assert result.ok is False
    assert "EMPTY old_str" in result.error


async def test_edit_missing_file_is_404(svc: DocumentsService):
    with pytest.raises(DocumentsError) as exc:
        await svc.apply_edit("nope.md", "a", "b")
    assert exc.value.status == 404


# --------------------------------------------------------------------------- #
# Confinement: traversal / absolute-outside / symlink-escape rejected, untouched
# --------------------------------------------------------------------------- #


def test_relative_traversal_rejected(svc: DocumentsService):
    with pytest.raises(DocumentsError) as exc:
        svc.read("../../etc/passwd")
    assert exc.value.status == 403


def test_absolute_outside_root_rejected(svc: DocumentsService, tmp_path: Path):
    outside = tmp_path / "outside.txt"
    outside.write_text("SECRET\n")
    with pytest.raises(DocumentsError) as exc:
        svc.read(str(outside))
    assert exc.value.status == 403
    assert outside.read_text() == "SECRET\n"  # untouched


def test_save_traversal_rejected_and_target_untouched(svc: DocumentsService, tmp_path: Path):
    outside = tmp_path / "outside.txt"
    outside.write_text("SECRET\n")
    # An absolute path escaping the root must not be written through.
    with pytest.raises(DocumentsError) as exc:
        svc.save(str(outside), "PWNED")
    assert exc.value.status == 403
    assert outside.read_text() == "SECRET\n"  # NOT overwritten


async def test_symlink_escape_rejected_and_target_untouched(svc: DocumentsService, tmp_path: Path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET\n")
    # A symlink INSIDE the root pointing OUT of it.
    link = svc.root / "escape.txt"
    link.symlink_to(secret)

    # Read through the symlink → resolved real path is outside → rejected.
    with pytest.raises(DocumentsError) as exc:
        svc.read("escape.txt")
    assert exc.value.status == 403

    # Edit through the symlink → rejected, and the real target is untouched.
    with pytest.raises(DocumentsError):
        await svc.apply_edit("escape.txt", "TOP SECRET", "leaked")
    assert secret.read_text() == "TOP SECRET\n"


# --------------------------------------------------------------------------- #
# Composition: SecurityGate denied-paths still deny a path that is under-root
# --------------------------------------------------------------------------- #


def test_securitygate_denied_under_root_still_denied(root: Path):
    secret_dir = root / "private"
    secret_dir.mkdir()
    (secret_dir / "keys.md").write_text("api-key\n")
    gate = SecurityGate(denied_paths=[str(secret_dir)])
    svc = DocumentsService(root, gate=gate)

    # Under the documents root, but under a denied prefix → still denied.
    with pytest.raises(DocumentsError) as exc:
        svc.read("private/keys.md")
    assert exc.value.status == 403
    assert "SecurityGate" in exc.value.message

    # A sibling NOT under the denied prefix is allowed (gate composes, not blanket-blocks).
    assert svc.read("notes.md")["content"].startswith("# Notes")


async def test_securitygate_denied_blocks_edit_untouched(root: Path):
    secret_dir = root / "private"
    secret_dir.mkdir()
    target = secret_dir / "keys.md"
    target.write_text("api-key\n")
    gate = SecurityGate(denied_paths=[str(secret_dir)])
    svc = DocumentsService(root, gate=gate)

    with pytest.raises(DocumentsError):
        await svc.apply_edit("private/keys.md", "api-key", "leaked")
    assert target.read_text() == "api-key\n"  # untouched


# --------------------------------------------------------------------------- #
# Suggest: validates find-uniqueness, returns redlines, writes NOTHING
# --------------------------------------------------------------------------- #


async def test_suggest_validates_and_does_not_write(svc: DocumentsService):
    before = (svc.root / "notes.md").read_text()

    async def stub_generator(content: str, instruction: str) -> list[dict]:
        assert "Hello" in content  # the span/document was passed to the model
        return [
            {"find": "Hello", "replace": "Greetings", "reason": "formal tone"},
            {"find": "the", "replace": "THE", "reason": "emphasis"},  # ambiguous
            {"find": "NOTHERE", "replace": "x", "reason": "noop"},   # absent
            {"find": "", "replace": "x", "reason": "empty"},          # dropped
        ]

    edits = await svc.suggest("notes.md", "make it formal", stub_generator)
    # empty-find dropped → 3 returned.
    assert len(edits) == 3
    by_find = {e.find: e for e in edits}
    assert by_find["Hello"].applicable is True and by_find["Hello"].note == ""
    assert by_find["the"].applicable is False and "ambiguous" in by_find["the"].note
    assert by_find["NOTHERE"].applicable is False and "not present" in by_find["NOTHERE"].note

    # Redline mode touches NOTHING on disk.
    assert (svc.root / "notes.md").read_text() == before


async def test_suggest_missing_file_is_404(svc: DocumentsService):
    async def gen(content: str, instruction: str) -> list[dict]:
        return []

    with pytest.raises(DocumentsError) as exc:
        await svc.suggest("nope.md", "x", gen)
    assert exc.value.status == 404
