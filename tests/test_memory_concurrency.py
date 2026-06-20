"""Concurrency + fail-loud regression for the MemoryStore.

The bug this guards: the daemon shares ONE sqlite connection across threads
(extractor batch, synthesis, sentinel, /note). Concurrent writers corrupted the
shared transaction state and a write was silently lost — a Telegram /note acked
"Noted" but never persisted. The fix funnels every write through one locked
choke-point and makes a failed write RAISE (never a false ack). These tests are
exactly the concurrency case every isolated test was missing.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
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
from prometheus.memory.store import MemoryStore, MemoryWriteError  # noqa: E402


def test_concurrent_notes_and_batch_no_lost_writes(tmp_path):
    """25 /note writes racing a 300-fact ambient batch: ALL 25 persist, no error.

    Pre-fix this lost every concurrent /note (0/25, InterfaceError on the shared
    connection). Post-fix the single locked choke-point serializes them.
    """
    store = MemoryStore(db_path=tmp_path / "memory.db")
    n_notes = 25
    errors: list[tuple[str, str]] = []

    def batch() -> None:
        try:
            for i in range(300):
                store.persist_memory(
                    "concept", f"Ambient{i}", f"ambient fact {i}", 0.7,
                    source_event_ids=[f"e{i}"],
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(("batch", repr(exc)))

    def notes() -> None:
        try:
            for i in range(n_notes):
                reply = cmd_note(store, f"@RaceNote{i} concurrent manual note {i}")
                assert "Noted" in reply, f"note {i} not acked: {reply!r}"
        except Exception as exc:  # noqa: BLE001
            errors.append(("note", repr(exc)))

    t1 = threading.Thread(target=batch)
    t2 = threading.Thread(target=notes)
    t1.start(); t2.start(); t1.join(); t2.join()

    assert not errors, f"concurrent writes raised: {errors}"
    persisted = store._conn.execute(
        "SELECT COUNT(*) FROM memories WHERE manual = 1 AND entity_name LIKE 'RaceNote%'"
    ).fetchone()[0]
    assert persisted == n_notes, f"LOST WRITES: only {persisted}/{n_notes} /note facts persisted"
    store.close()


def test_many_writers_one_connection_all_land(tmp_path):
    """Eight threads each writing 100 facts to ONE store -> all 800 land."""
    store = MemoryStore(db_path=tmp_path / "memory.db")
    errors: list[str] = []

    def writer(w: int) -> None:
        try:
            for i in range(100):
                store.persist_memory(
                    "concept", f"W{w}E{i}", f"fact {w}-{i}", 0.6,
                    source_event_ids=[f"w{w}e{i}"],
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent writers raised: {errors}"
    total = store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert total == 800, f"lost writes: {total}/800 persisted"
    store.close()


def test_concurrent_reads_during_writes_no_error_no_tear(tmp_path):
    """Reads racing a write batch never error and never see torn state — the
    read-vs-write C-API collision is gone now reads share the write lock."""
    store = MemoryStore(db_path=tmp_path / "memory.db")
    errors: list[tuple[str, str]] = []
    stop = threading.Event()

    def writer() -> None:
        try:
            for i in range(500):
                store.persist_memory(
                    "concept", f"E{i}", f"fact {i}", 0.6, source_event_ids=[f"e{i}"]
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(("write", repr(exc)))
        finally:
            stop.set()

    def reader() -> None:
        try:
            while not stop.is_set():
                rows = store.get_all_memories(limit=10000)
                # Torn-state guard: every row is a fully-formed, JSON-decoded dict.
                for r in rows:
                    assert isinstance(r.get("entity_name"), str)
                    assert isinstance(r.get("source_event_ids"), list)
                store.search_memories(entity="E", limit=50)
        except Exception as exc:  # noqa: BLE001
            errors.append(("read", repr(exc)))

    tw = threading.Thread(target=writer)
    trs = [threading.Thread(target=reader) for _ in range(3)]
    tw.start()
    for t in trs:
        t.start()
    tw.join()
    for t in trs:
        t.join()

    assert not errors, f"concurrent read/write raised: {errors}"
    total = store.get_all_memories(limit=10000)
    assert len([m for m in total if m["entity_name"].startswith("E")]) == 500
    store.close()


def test_cmd_note_fails_loud_not_false_ack():
    """A failing store write surfaces as 'NOT saved' — never a false 'Noted'."""
    class _RaisingStore:
        def persist_memory(self, *a, **k):
            raise RuntimeError("simulated write failure")

    reply = cmd_note(_RaisingStore(), "a note whose write fails")
    assert "NOT saved" in reply
    assert "Noted" not in reply


def test_persist_raises_on_write_failure(tmp_path):
    """A write that can't land RAISES (never swallowed) — here a PK collision."""
    store = MemoryStore(db_path=tmp_path / "memory.db")
    mid = store.persist_memory(
        "note", "X", "fact one", 1.0, source_event_ids=["a"], manual=True
    )
    with pytest.raises(sqlite3.IntegrityError):
        # Re-using the id under a different entity dodges dedup and collides on PK.
        store.persist_memory(
            "note", "Y", "fact two", 1.0, source_event_ids=["b"], memory_id=mid
        )
    store.close()


def test_persist_confirms_rowcount_raises_on_silent_noop(tmp_path):
    """persist_memory raises MemoryWriteError if the INSERT silently writes 0 rows.

    sqlite3.Connection.execute is read-only, so wrap the connection: the
    memories INSERT reports rowcount 0; everything else passes through.
    """
    store = MemoryStore(db_path=tmp_path / "memory.db")

    class _ZeroRows:
        rowcount = 0
        def fetchall(self): return []
        def fetchone(self): return None

    class _Wrap:
        def __init__(self, real):
            object.__setattr__(self, "_real", real)

        def execute(self, sql, *a, **k):
            if sql.lstrip().upper().startswith("INSERT INTO MEMORIES"):
                return _ZeroRows()
            return object.__getattribute__(self, "_real").execute(sql, *a, **k)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_real"), name)

    store._conn = _Wrap(store._conn)  # type: ignore[assignment]
    with pytest.raises(MemoryWriteError):
        store.persist_memory("note", "Z", "zfact", 1.0, source_event_ids=["m"], manual=True)
    store.close()
