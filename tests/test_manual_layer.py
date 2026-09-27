"""Phase-4a manual layer: store `manual` flag, migration, and /note (cmd_note)."""

from __future__ import annotations

import itertools
import json
import sqlite3
import sys
import time
import types
from pathlib import Path

# Bypass the prometheus.memory circular-import chain (same shim as test_wiki).
if "prometheus.memory" not in sys.modules:
    _pkg = types.ModuleType("prometheus.memory")
    _pkg.__path__ = ["src/prometheus/memory"]
    _pkg.__package__ = "prometheus.memory"
    sys.modules["prometheus.memory"] = _pkg

import pytest  # noqa: E402

from prometheus.gateway.commands import cmd_note  # noqa: E402
from prometheus.memory.store import MemoryStore  # noqa: E402


def _make_pre_manual_db(path: Path) -> None:
    """Create a memories table WITHOUT the `manual` column (pre-4a schema)."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE memories ("
        " id TEXT PRIMARY KEY, entity_type TEXT NOT NULL, entity_name TEXT NOT NULL,"
        " relationship TEXT NOT NULL, fact TEXT NOT NULL,"
        " confidence REAL NOT NULL DEFAULT 0.5,"
        " source_event_ids TEXT NOT NULL DEFAULT '[]', last_mentioned REAL NOT NULL,"
        " mention_count INTEGER NOT NULL DEFAULT 1, tags TEXT NOT NULL DEFAULT '[]',"
        " timestamp REAL NOT NULL)"
    )
    conn.execute(
        "INSERT INTO memories (id, entity_type, entity_name, relationship, fact,"
        " confidence, source_event_ids, last_mentioned, mention_count, tags, timestamp)"
        " VALUES ('old1','person','Old','fact','an old fact',0.5,'[\"e\"]',0,1,'[]',0)"
    )
    conn.commit()
    conn.close()


def _rows(store: MemoryStore):
    return store._conn.execute(
        "SELECT entity_name, fact, confidence, manual, source_event_ids FROM memories"
    ).fetchall()


# --- store flag ------------------------------------------------------------

def test_persist_memory_manual_flag(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.persist_memory("note", "Foo", "a manual fact", 1.0,
                         source_event_ids=["manual"], manual=True)
    rows = _rows(store)
    assert len(rows) == 1
    assert rows[0]["manual"] == 1
    assert rows[0]["confidence"] == 1.0
    assert json.loads(rows[0]["source_event_ids"]) == ["manual"]
    store.close()


# --- /note (cmd_note) — side-effect tests ----------------------------------

def test_note_writes_manual_fact(tmp_path):
    """/note writes a fact: row has manual=1, source=manual, max trust."""
    store = MemoryStore(db_path=tmp_path / "memory.db")
    msg = cmd_note(store, "@Pham started a new clinic")
    assert "Pham" in msg
    rows = _rows(store)
    assert len(rows) == 1
    r = rows[0]
    assert r["entity_name"] == "Pham"
    assert r["fact"] == "started a new clinic"
    assert r["manual"] == 1
    assert r["confidence"] == 1.0
    assert json.loads(r["source_event_ids"]) == ["manual"]
    store.close()


def test_note_flips_existing_row_no_duplicate(tmp_path):
    """/note matching an ambient fact flips that row to manual — row count flat."""
    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.persist_memory("person", "Pham", "is a nephrologist", 0.6,
                         source_event_ids=["evt1"])
    assert store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1

    cmd_note(store, "@Pham is a nephrologist")

    rows = _rows(store)
    assert len(rows) == 1, "row count must stay flat — no duplicate row"
    assert rows[0]["manual"] == 1, "ambient row flipped to manual"
    assert rows[0]["confidence"] == 1.0, "confidence maxed to manual's 1.0"
    assert set(json.loads(rows[0]["source_event_ids"])) == {"evt1", "manual"}
    store.close()


def test_note_without_entity_uses_default_bucket(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory.db")
    cmd_note(store, "remember to file the Q3 report")
    rows = _rows(store)
    assert len(rows) == 1
    assert rows[0]["entity_name"] == "Notes"
    assert rows[0]["manual"] == 1
    store.close()


# --- migration -------------------------------------------------------------

def test_migration_adds_manual_column_and_snapshots(tmp_path):
    db = tmp_path / "memory.db"
    _make_pre_manual_db(db)
    store = MemoryStore(db_path=db)  # __init__ runs the migration
    cols = {r[1] for r in store._conn.execute("PRAGMA table_info(memories)")}
    assert "manual" in cols
    # existing row backfilled to manual=0
    assert store._conn.execute("SELECT manual FROM memories").fetchone()["manual"] == 0
    # snapshot written out-of-tree (sibling of the DB, timestamped)
    assert list(tmp_path.glob("memory.db.backup-*")), "migration must snapshot first"
    store.close()


def test_migration_is_idempotent_no_snapshot_on_reopen(tmp_path):
    """A reopen migrates nothing, so it snapshots nothing: measured as a delta.

    The first open runs BOTH migrations on a pre-manual DB with rows (manual
    column, then FTS rebuild), and each snapshots first. How many files that
    leaves is pinned by test_snapshots_never_overwrite_each_other below; this
    test only asks that the count does not move on reopen.
    """
    db = tmp_path / "memory.db"
    _make_pre_manual_db(db)
    MemoryStore(db_path=db).close()                  # migrates + snapshots
    first = len(list(tmp_path.glob("memory.db.backup-*")))
    assert first, "the first open migrates, and a migration snapshots first"
    MemoryStore(db_path=db).close()                  # column present, user_version 1 → no-op
    assert len(list(tmp_path.glob("memory.db.backup-*"))) == first, \
        "no new snapshot on reopen: nothing is left to migrate"


@pytest.mark.parametrize(
    ("offsets", "expected"),
    [
        # Both snapshots inside one second: the second name takes a sequence number.
        ((0, 0), ["backup-{0}", "backup-{0}-1"]),
        # Straddling a second boundary (the slow-runner case CI hit): two stamps.
        ((0, 1), ["backup-{0}", "backup-{1}"]),
    ],
    ids=["same-second", "next-second"],
)
def test_snapshots_never_overwrite_each_other(tmp_path, monkeypatch, offsets, expected):
    """One open of a pre-manual DB with rows runs two migrations, and each
    snapshots first — with a name that is one-second granular.

    On a fast machine both copies fell inside the same second, so the second
    copy landed on the first one's name and silently replaced it: the log named
    two snapshots, the disk held one. On a slow runner the two straddled a
    second boundary and the disk held two — which is why a test that assumed
    the collision (``== 1``) passed locally and failed on CI. Pin the clock to
    both cases: the disk must hold one readable copy per migration either way.
    """
    db = tmp_path / "memory.db"
    _make_pre_manual_db(db)
    real_localtime = time.localtime
    stamps = [real_localtime(1_700_000_000 + s) for s in offsets]
    calls = itertools.count()
    # The first call gets the first stamp; every later call gets the last one.
    monkeypatch.setattr(
        time, "localtime", lambda *_: stamps[min(next(calls), len(stamps) - 1)]
    )

    MemoryStore(db_path=db).close()

    text = [time.strftime("%Y%m%dT%H%M%S", st) for st in stamps]
    backups = sorted(tmp_path.glob("memory.db.backup-*"))
    assert [b.name for b in backups] == [f"memory.db.{e.format(*text)}" for e in expected], \
        "one snapshot per migration: the second must not overwrite the first"
    for b in backups:
        conn = sqlite3.connect(str(b))
        assert conn.execute("SELECT fact FROM memories").fetchall() == [("an old fact",)], \
            f"{b.name} must be a readable copy of the data"
        conn.close()


def test_snapshot_holds_rows_still_in_the_wal(tmp_path):
    """A snapshot holds every committed row, including those still in the WAL.

    The store runs in ``journal_mode=WAL``: a committed write lives in
    ``memory.db-wal`` until a checkpoint folds it into ``memory.db``, so a plain
    file copy of ``memory.db`` misses it. Park a committed row in the WAL —
    switch the pre-manual DB to WAL, insert, and keep a second connection open
    so that closing the writer cannot checkpoint — then let the store open,
    migrate and snapshot. Every snapshot must hold both rows.
    """
    db = tmp_path / "memory.db"
    _make_pre_manual_db(db)                       # 'an old fact' sits in memory.db itself
    writer = sqlite3.connect(str(db))
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute(
        "INSERT INTO memories (id, entity_type, entity_name, relationship, fact,"
        " confidence, source_event_ids, last_mentioned, mention_count, tags, timestamp)"
        " VALUES ('wal1','person','Old','fact','a fact still in the wal',0.5,'[]',0,1,'[]',0)"
    )
    writer.commit()
    keeper = sqlite3.connect(str(db))
    keeper.execute("SELECT COUNT(*) FROM memories").fetchone()  # opens the WAL and keeps it
    writer.close()                                # not the last connection: no checkpoint
    wal = tmp_path / "memory.db-wal"
    assert wal.exists() and wal.stat().st_size > 0, \
        "precondition: the new row must still sit in the WAL"
    try:
        MemoryStore(db_path=db).close()           # migrates twice, snapshots twice
    finally:
        keeper.close()

    backups = sorted(tmp_path.glob("memory.db.backup-*"))
    assert len(backups) == 2, "one snapshot per migration"
    for b in backups:
        conn = sqlite3.connect(str(b))
        facts = {r[0] for r in conn.execute("SELECT fact FROM memories")}
        conn.close()
        assert facts == {"an old fact", "a fact still in the wal"}, \
            f"{b.name} misses committed rows"


def test_migration_fails_loud_no_half_write(tmp_path, monkeypatch):
    """A broken ALTER halts (raises) and does NOT half-write the column.

    The ALTER fails through a Connection subclass installed as the factory of
    every connection the store opens — a real connection, so the backup API
    accepts it as a snapshot target — and the snapshot still runs first.
    """
    db = tmp_path / "memory.db"
    _make_pre_manual_db(db)
    real_connect = sqlite3.connect

    class _AlterFailsConn(sqlite3.Connection):
        """A real connection, except the manual ALTER raises."""

        def execute(self, sql, *args, **kwargs):
            if "ADD COLUMN manual" in sql:
                raise sqlite3.OperationalError("simulated ALTER failure")
            return super().execute(sql, *args, **kwargs)

    monkeypatch.setattr(
        sqlite3, "connect",
        lambda *a, **k: real_connect(*a, factory=_AlterFailsConn, **k),
    )

    with pytest.raises(sqlite3.OperationalError):
        MemoryStore(db_path=db)

    monkeypatch.undo()  # restore real connect for the assertions
    cols = {r[1] for r in sqlite3.connect(str(db)).execute("PRAGMA table_info(memories)")}
    assert "manual" not in cols, "failed migration must not half-write the column"
    # ...but the snapshot ran first — self-protection precedes the failing apply.
    assert list(tmp_path.glob("memory.db.backup-*")), "snapshot must precede the ALTER"
