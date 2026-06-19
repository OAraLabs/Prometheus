"""Phase-4a manual layer: store `manual` flag, migration, and /note (cmd_note)."""

from __future__ import annotations

import json
import sqlite3
import sys
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
    db = tmp_path / "memory.db"
    _make_pre_manual_db(db)
    MemoryStore(db_path=db).close()                  # migrates + snapshots once
    first = len(list(tmp_path.glob("memory.db.backup-*")))
    assert first == 1
    MemoryStore(db_path=db).close()                  # column present → no-op
    assert len(list(tmp_path.glob("memory.db.backup-*"))) == first, \
        "no new snapshot when the column already exists"


def test_migration_fails_loud_no_half_write(tmp_path, monkeypatch):
    """A broken ALTER halts (raises) and does NOT half-write the column.

    sqlite3.Connection is an immutable C type, so we make the ALTER fail by
    wrapping the connection the store opens — the snapshot still runs first.
    """
    db = tmp_path / "memory.db"
    _make_pre_manual_db(db)
    real_connect = sqlite3.connect

    class _AlterFailsConn:
        """A real connection, except the manual ALTER raises."""

        def __init__(self, real):
            object.__setattr__(self, "_real", real)

        def execute(self, sql, *args, **kwargs):
            if "ADD COLUMN manual" in sql:
                raise sqlite3.OperationalError("simulated ALTER failure")
            return object.__getattribute__(self, "_real").execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_real"), name)

        def __setattr__(self, name, value):
            setattr(object.__getattribute__(self, "_real"), name, value)

    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **k: _AlterFailsConn(real_connect(*a, **k))
    )

    with pytest.raises(sqlite3.OperationalError):
        MemoryStore(db_path=db)

    monkeypatch.undo()  # restore real connect for the assertions
    cols = {r[1] for r in sqlite3.connect(str(db)).execute("PRAGMA table_info(memories)")}
    assert "manual" not in cols, "failed migration must not half-write the column"
    # ...but the snapshot ran first — self-protection precedes the failing apply.
    assert list(tmp_path.glob("memory.db.backup-*")), "snapshot must precede the ALTER"
