"""Purging a session removes its content from the FILE, not just from the API.

Prometheus#353. `DELETE /api/sessions/{id}` writes a tombstone: it hides a
session and deliberately leaves the append-only rows intact. That is the right
answer to "take this out of my list" and the wrong answer to "I pasted a
customer's details into that chat".

The assertion that matters here is not "the rows are gone" — a plain DELETE
achieves that while the words remain readable in the fts5 index and in freed
pages. It is that the TEXT IS NOT IN THE DATABASE FILE.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_summary_store import LCMSummaryStore
from prometheus.memory.lcm_types import MessagePart, SummaryNode

SECRET = "ZZQXPURGEMARKER7781"
KEEPER = "YYWVKEEPMARKER4412"


def _build(tmp_path: Path) -> tuple[LCMConversationStore, Path]:
    db = tmp_path / "lcm.db"
    store = LCMConversationStore(db_path=db)
    store.add_message("beacon:doomed", MessagePart(role="user", content=f"my card is {SECRET}"))
    store.add_message("beacon:doomed", MessagePart(role="assistant", content="noted"))
    store.add_message("beacon:keeper", MessagePart(role="user", content=f"unrelated {KEEPER}"))
    store.set_session_title("beacon:doomed", "Doomed")
    store.set_session_pinned("beacon:doomed", True)
    return store, db


def _index_terms(store: LCMConversationStore, prefix: str) -> list[str]:
    """Terms still held by the fts5 index itself.

    A byte search cannot answer this: fts5 stores TOKENS, lowercased, in its own
    encoded blobs. The first version of this test searched the file for an
    UPPERCASE marker and therefore could not see index residue at all — a plain
    DELETE that never told the index passed it. The index has to be asked
    directly.
    """
    store._conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS _vocab_probe USING fts5vocab(lcm_messages_fts, 'row')"
    )
    return [
        r[0]
        for r in store._conn.execute(
            "SELECT term FROM _vocab_probe WHERE term LIKE ?", (prefix.lower() + "%",)
        ).fetchall()
    ]


def _file_contains(db: Path, needle: str) -> bool:
    """Search the database AND its sidecars.

    In WAL mode a freshly written row lives in `<db>-wal` until checkpointed,
    so reading only the .db file would report a secret as absent while it sat
    in a file beside it — which is how this helper first failed its own
    precondition, and exactly the illusion a purge must not create.
    """
    for p in (db, Path(str(db) + "-wal"), Path(str(db) + "-shm")):
        if p.exists() and needle.encode() in p.read_bytes():
            return True
    return False


def test_purge_removes_the_text_from_the_file(tmp_path: Path) -> None:
    store, db = _build(tmp_path)

    # precondition: the marker is findable AND present in the file, or the test
    # below would pass on a database that never held it.
    assert store.search(SECRET), "precondition: the secret is searchable"
    assert _index_terms(store, SECRET) == [SECRET.lower()], "precondition: the index holds the term"
    assert _file_contains(db, SECRET), "precondition: the secret is in the file"

    counts = store.purge_session("beacon:doomed")

    assert counts["messages"] == 2, counts
    assert store.search(SECRET) == [], "search must not find it"
    assert store.get_messages("beacon:doomed") == [], "the rows are gone"
    # THE POINT. A DELETE that leaves the words in the fts5 index or in a freed
    # page passes every check above and fails this one.
    assert not _file_contains(db, SECRET), "the text must not be in the database file"
    assert not _file_contains(db, SECRET.lower()), "nor in the lowercased form the index stores"
    # The index is the residue a byte search is worst at seeing, so ask it.
    assert _index_terms(store, SECRET) == [], "the fts5 index must not still hold the term"


def test_purge_does_not_touch_other_sessions(tmp_path: Path) -> None:
    store, db = _build(tmp_path)
    store.purge_session("beacon:doomed")
    assert len(store.get_messages("beacon:keeper")) == 1
    assert store.search(KEEPER), "an unrelated session stays searchable"
    assert _file_contains(db, KEEPER), "and its text stays in the file"


def test_purge_removes_the_side_rows(tmp_path: Path) -> None:
    store, db = _build(tmp_path)
    counts = store.purge_session("beacon:doomed")
    assert counts["session_titles"] == 1, counts
    assert counts["session_pins"] == 1, counts
    assert not _file_contains(db, "Doomed"), "a title is content too"


def test_purge_removes_derived_summaries(tmp_path: Path) -> None:
    """A purged conversation whose SUMMARY still quotes it is not purged."""
    store, db = _build(tmp_path)
    summaries = LCMSummaryStore(db_path=db)
    summaries.add_summary(
        "beacon:doomed", SummaryNode(summary_text=f"user shared {SECRET}")
    )
    assert _file_contains(db, SECRET)

    counts = store.purge_session("beacon:doomed")
    assert counts["summaries"] == 1, counts
    assert not _file_contains(db, SECRET), "the summary's copy must go too"


def test_branches_are_reported_not_silently_orphaned(tmp_path: Path) -> None:
    """A fork holds its own COPIES, so purging the origin cannot reach them.

    That is defensible and astonishing-if-unsaid, so the store surfaces them.
    """
    store, db = _build(tmp_path)
    # NB: get_messages() returns row_id 0 — the store populates it on INSERT
    # only, though the field's comment says "insert/read". Use the value written
    # back onto the inserted object rather than trusting the read path.
    first = MessagePart(role="user", content="forkable")
    store.add_message("beacon:doomed", first)
    assert first.row_id > 0, "the store stamps the rowid onto the inserted object"
    store.fork_session("beacon:doomed", first.row_id, "beacon:branch")
    assert [f["session_id"] for f in store.list_session_forks("beacon:doomed")] == ["beacon:branch"]

    store.purge_session("beacon:doomed")
    # the branch's own copy survives, because it IS its own copy
    assert len(store.get_messages("beacon:branch")) == 3, "the branch keeps its own copies"


def test_counts_distinguish_a_purge_from_a_no_op(tmp_path: Path) -> None:
    store, _ = _build(tmp_path)
    hit = store.purge_session("beacon:doomed")
    miss = store.purge_session("beacon:never-existed")
    assert hit["messages"] == 2
    assert miss["messages"] == 0, "purging nothing must not look like purging something"
