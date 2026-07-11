"""SQLite-backed memory store with FTS5 search.

Schema mirrors OpenClaw's proven memories table structure.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sqlite3
import string
import threading
import time
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from prometheus.config.paths import get_config_dir

log = logging.getLogger(__name__)

_DB_NAME = "memory.db"

_DEDUP_TRAILING = string.punctuation + " "


def _normalize_for_dedup(text: str) -> str:
    """Normalize a fact/entity string for duplicate detection.

    Lowercase + collapse internal whitespace + strip trailing
    punctuation/whitespace. Deterministic; no embeddings. Strings that
    differ only by case, spacing, or trailing punctuation collapse to the
    same key; genuinely different wording stays distinct.
    """
    if not text:
        return ""
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return collapsed.rstrip(_DEDUP_TRAILING)


def _get_db_path() -> Path:
    return get_config_dir() / _DB_NAME


class MemoryWriteError(RuntimeError):
    """A memory write failed to commit a row.

    Raised — never swallowed — so a failed write surfaces to the caller (and the
    user) instead of being hidden behind a false success. Closes the false-ack
    bug class for every write path, not just the concurrency one.
    """


class MemoryStore:
    """SQLite memory store with FTS5 full-text search.

    Tables:
      messages  — conversation history (with FTS5 index)
      memories  — extracted entity facts (with FTS5 index)
      summaries — compressed conversation summaries

    Thread-safety: the daemon drives one MemoryStore from several threads — the
    memory extractor's batch, synthesis, sentinel, and the Telegram /note
    handler. A single sqlite connection is NOT safe for concurrent writers: the
    shared transaction state corrupts and a write is silently lost (acked but
    never committed). So every mutation funnels through the single ``_write``
    choke-point under one lock — single-writer discipline at the connection
    level, and a future write path physically cannot forget the lock because
    ``_write`` is the only committer.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else _get_db_path()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Serializes ALL access to the shared connection — writes AND reads.
        # Store reads are infrequent (periodic SENTINEL/compile, on-demand
        # /status·/memory; the per-turn path is file-based wiki_query, never a
        # store read), so locking them adds no hot-path latency and removes the
        # read-vs-write sqlite C-API collision entirely. RLock: reentrant, so a
        # write path that composes another (dedupe_existing -> delete/update)
        # can't deadlock; one lock, no lock-ordering risk.
        self._lock = threading.RLock()
        self._apply_schema()

    # ------------------------------------------------------------------
    # Connection choke-points — the ONLY code that touches self._conn
    # ------------------------------------------------------------------

    def _write(self, fn):
        """Run ``fn(conn)`` under the lock, commit, return its result — the sole
        committer.

        On ANY error: roll back and RE-RAISE — never swallow, so a failed or
        lost write surfaces instead of a false success.
        """
        with self._lock:
            try:
                result = fn(self._conn)
                self._conn.commit()
                return result
            except Exception:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                raise

    def _read(self, fn):
        """Run ``fn(conn)`` under the same lock and return its result (no commit).

        Serializes reads with writes on the shared connection so a read can't hit
        the C-API misuse mid-write or see torn state.
        """
        with self._lock:
            return fn(self._conn)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _apply_schema(self) -> None:
        self._write(lambda conn: conn.executescript("""
            PRAGMA journal_mode=WAL;

            -- DEPRECATED 2026-05-26 — superseded by LCM (LCMConversationStore)
            -- as the conversation store. Kept temporarily because tests still
            -- exercise this path; remove in follow-up PR alongside the test
            -- cleanup. No new code should call this. See:
            -- fix/memory-lcm-full-rewire commit message.
            CREATE TABLE IF NOT EXISTS messages (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                timestamp   REAL NOT NULL,
                compressed  INTEGER NOT NULL DEFAULT 0
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                id UNINDEXED,
                content,
                content='messages',
                content_rowid='rowid'
            );

            CREATE TABLE IF NOT EXISTS memories (
                id                TEXT PRIMARY KEY,
                entity_type       TEXT NOT NULL,
                entity_name       TEXT NOT NULL,
                relationship      TEXT NOT NULL,
                fact              TEXT NOT NULL,
                confidence        REAL NOT NULL DEFAULT 0.5,
                source_event_ids  TEXT NOT NULL DEFAULT '[]',
                last_mentioned    REAL NOT NULL,
                mention_count     INTEGER NOT NULL DEFAULT 1,
                tags              TEXT NOT NULL DEFAULT '[]',
                timestamp         REAL NOT NULL,
                manual            INTEGER NOT NULL DEFAULT 0
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id UNINDEXED,
                entity_name,
                fact,
                content='memories',
                content_rowid='rowid'
            );

            CREATE TABLE IF NOT EXISTS summaries (
                id                  TEXT PRIMARY KEY,
                source_message_ids  TEXT NOT NULL DEFAULT '[]',
                summary_text        TEXT NOT NULL,
                level               INTEGER NOT NULL DEFAULT 1,
                timestamp           REAL NOT NULL
            );

            -- FTS sync triggers (passive-recall sprint). Both FTS tables are
            -- external-content (content=); FTS5 requires index maintenance to
            -- go through the special 'delete' command keyed by the CONTENT
            -- table's rowid. The store used to hand-roll DML instead —
            -- INSERT without rowid (mints an orphan index row per write) and
            -- plain DELETE (never reaches the index) — which corrupted the
            -- index over time ("fts5: missing row N from content table").
            -- Triggers make every write path sync by construction, including
            -- raw SQL like dedupe_existing. _migrate_fts_integrity() rebuilds
            -- once to heal indexes corrupted by the old path.
            CREATE TRIGGER IF NOT EXISTS messages_fts_ai AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, id, content)
                VALUES (new.rowid, new.id, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS messages_fts_ad AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, id, content)
                VALUES ('delete', old.rowid, old.id, old.content);
            END;
            CREATE TRIGGER IF NOT EXISTS messages_fts_au AFTER UPDATE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, id, content)
                VALUES ('delete', old.rowid, old.id, old.content);
                INSERT INTO messages_fts(rowid, id, content)
                VALUES (new.rowid, new.id, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, id, entity_name, fact)
                VALUES (new.rowid, new.id, new.entity_name, new.fact);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, id, entity_name, fact)
                VALUES ('delete', old.rowid, old.id, old.entity_name, old.fact);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, id, entity_name, fact)
                VALUES ('delete', old.rowid, old.id, old.entity_name, old.fact);
                INSERT INTO memories_fts(rowid, id, entity_name, fact)
                VALUES (new.rowid, new.id, new.entity_name, new.fact);
            END;
        """))
        self._migrate_manual_column()
        self._migrate_fts_integrity()

    def _migrate_manual_column(self) -> None:
        """Add the ``manual`` flag column to a pre-manual-layer memories table.

        Idempotent: a no-op when the column already exists (the common path on
        every startup — no snapshot, no write). When it is missing (a DB created
        before the manual layer), snapshot the DB out-of-tree first, then ALTER.
        Fail loud — a failed snapshot or ALTER propagates rather than leaving the
        table half-migrated.
        """
        cols = {
            row[1]
            for row in self._read(
                lambda conn: conn.execute("PRAGMA table_info(memories)").fetchall()
            )
        }
        if "manual" in cols:
            return
        self._snapshot_db()
        self._write(
            lambda conn: conn.execute(
                "ALTER TABLE memories ADD COLUMN manual INTEGER NOT NULL DEFAULT 0"
            )
        )

    def _migrate_fts_integrity(self) -> None:
        """One-time rebuild of both FTS indexes (passive-recall sprint).

        DBs written before the sync triggers existed carry corrupted
        external-content indexes: the old hand-rolled DML inserted without a
        rowid (orphan index rows) and deleted without the FTS 'delete'
        command, so any MATCH could raise "fts5: missing row N from content
        table". ``PRAGMA user_version`` gates the rebuild to once per DB;
        the triggers keep it consistent from then on. Fail loud — a failed
        rebuild propagates rather than leaving search broken but silent.
        """
        version = self._read(
            lambda conn: conn.execute("PRAGMA user_version").fetchone()[0]
        )
        if version >= 1:
            return
        has_rows = self._read(
            lambda conn: conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            or conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        )
        if has_rows:
            # Only snapshot DBs with data — fresh/empty stores (every unit
            # test, first boot) have nothing to protect.
            self._snapshot_db(reason="FTS rebuild")

        def _op(conn):
            conn.execute("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')")
            conn.execute("INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')")
            conn.execute("PRAGMA user_version = 1")

        self._write(_op)
        if has_rows:
            log.info("MemoryStore: FTS indexes rebuilt (one-time integrity migration)")

    def _snapshot_db(self, *, reason: str = "manual migration") -> None:
        """Copy the DB file out-of-tree, timestamped, before a migration."""
        src = self._db_path
        if not src.exists():
            return  # nothing to back up (in-memory / brand-new DB)
        ts = time.strftime("%Y%m%dT%H%M%S", time.localtime())
        dst = src.with_name(f"{src.name}.backup-{ts}")
        shutil.copy2(src, dst)
        log.info("MemoryStore: snapshotted %s -> %s before %s", src, dst, reason)

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    # DEPRECATED 2026-05-26 — superseded by LCM (LCMConversationStore)
    # as the conversation store. Kept temporarily because tests still
    # exercise this path; remove in follow-up PR alongside the test
    # cleanup. No new code should call this. See:
    # fix/memory-lcm-full-rewire commit message.
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        message_id: str | None = None,
        compressed: bool = False,
    ) -> str:
        """Insert a conversation message. Returns the message ID."""
        mid = message_id or uuid4().hex
        now = time.time()

        def _op(conn):
            # FTS stays in sync via the messages_fts_* triggers (OR REPLACE
            # is DELETE+INSERT under the hood, firing both AD and AI).
            conn.execute(
                "INSERT OR REPLACE INTO messages (id, session_id, role, content, timestamp, compressed)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (mid, session_id, role, content, now, int(compressed)),
            )

        self._write(_op)
        return mid

    def get_messages(
        self,
        session_id: str,
        *,
        since: float | None = None,
        compressed: bool | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """Return messages for a session, newest first."""
        query = "SELECT * FROM messages WHERE session_id = ?"
        params: list = [session_id]
        if since is not None:
            query += " AND timestamp >= ?"
            params.append(since)
        if compressed is not None:
            query += " AND compressed = ?"
            params.append(int(compressed))
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self._read(lambda conn: conn.execute(query, params).fetchall())
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------

    def persist_memory(
        self,
        entity_type: str,
        entity_name: str,
        fact: str,
        confidence: float,
        *,
        relationship: str | None = None,
        source_event_ids: list[str] | None = None,
        tags: list[str] | None = None,
        memory_id: str | None = None,
        manual: bool = False,
    ) -> str:
        """Insert or update a memory fact. Returns the memory ID.

        ``manual`` flags a human-asserted fact (e.g. ``/note``). On a dedup
        hit the flag is *unioned* (``MAX``) onto the existing row — once manual,
        stays manual — so a ``/note`` that matches an ambient fact upgrades that
        one row rather than creating a duplicate.

        Provenance is mandatory: ``source_event_ids`` must be a non-empty
        list, else this raises ``ValueError``. Silent source-less writes
        render as ``source: unknown`` in the wiki and are the bug class
        SPRINT MEMORY-1 closes (forward-regression invariant).

        Deduplication is *normalized*: a fact is a duplicate of an existing
        one when, for the same entity (case-insensitive), the two facts
        normalize equal (see :func:`_normalize_for_dedup`). Lexical
        paraphrases — case / whitespace / trailing punctuation — coalesce
        into one row (``mention_count`` incremented, ``source_event_ids``
        unioned, confidence maxed); genuinely different facts stay separate.
        """
        if not source_event_ids:
            raise ValueError(
                "persist_memory requires a non-empty source_event_ids — "
                "provenance is mandatory (no silent 'unknown' writes)"
            )

        now = time.time()
        rel = relationship or "fact"
        norm_fact = _normalize_for_dedup(fact)

        def _op(conn):
            # Normalized duplicate check: scan this entity's rows (few) and
            # match on the normalized fact. Entity match is case-insensitive.
            candidates = conn.execute(
                "SELECT id, fact, source_event_ids FROM memories"
                " WHERE entity_name = ? COLLATE NOCASE",
                (entity_name,),
            ).fetchall()
            existing = next(
                (c for c in candidates if _normalize_for_dedup(c["fact"]) == norm_fact),
                None,
            )

            if existing:
                mid = existing["id"]
                merged_sources = sorted(
                    set(json.loads(existing["source_event_ids"] or "[]"))
                    | set(source_event_ids)
                )
                cur = conn.execute(
                    "UPDATE memories SET confidence = MAX(confidence, ?),"
                    " mention_count = mention_count + 1, last_mentioned = ?,"
                    " source_event_ids = ?, manual = MAX(manual, ?) WHERE id = ?",
                    (confidence, now, json.dumps(merged_sources), int(manual), mid),
                )
                if cur.rowcount != 1:
                    raise MemoryWriteError(
                        f"persist_memory: dedup UPDATE matched {cur.rowcount} rows "
                        f"for id={mid} (expected 1)"
                    )
                return mid

            mid = memory_id or uuid4().hex
            cur = conn.execute(
                "INSERT INTO memories"
                " (id, entity_type, entity_name, relationship, fact, confidence,"
                "  source_event_ids, last_mentioned, mention_count, tags, timestamp,"
                "  manual)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)",
                (
                    mid,
                    entity_type,
                    entity_name,
                    rel,
                    fact,
                    confidence,
                    json.dumps(source_event_ids or []),
                    now,
                    json.dumps(tags or []),
                    now,
                    int(manual),
                ),
            )
            if cur.rowcount != 1:
                raise MemoryWriteError(
                    f"persist_memory: INSERT wrote {cur.rowcount} rows (expected 1)"
                )
            # FTS index entry is created by the memories_fts_ai trigger.
            return mid

        return self._write(_op)

    def dedupe_existing(self) -> int:
        """One-time backfill: collapse normalized-equal rows per entity.

        For each group of rows that are normalized-equal (same normalized
        entity_name + normalized fact), keep the earliest row and merge the
        rest into it: ``source_event_ids`` UNIONed, ``mention_count`` SUMMED,
        confidence MAXed. Dropping either source or count would silently
        shred provenance — so union-and-sum is the invariant, not a count
        target. Returns the number of rows removed. No-op on an empty store.
        """
        rows = self._read(
            lambda conn: conn.execute(
                "SELECT id, entity_name, fact, confidence, source_event_ids,"
                " mention_count, timestamp FROM memories"
            ).fetchall()
        )

        groups: dict[tuple[str, str], list] = defaultdict(list)
        for r in rows:
            key = (
                _normalize_for_dedup(r["entity_name"]),
                _normalize_for_dedup(r["fact"]),
            )
            groups[key].append(r)

        removed = 0
        for group in groups.values():
            if len(group) < 2:
                continue
            # Earliest row (smallest timestamp) is the survivor.
            group = sorted(group, key=lambda r: r["timestamp"])
            survivor = group[0]
            sources = set(json.loads(survivor["source_event_ids"] or "[]"))
            mentions = survivor["mention_count"]
            conf = survivor["confidence"]
            for loser in group[1:]:
                sources |= set(json.loads(loser["source_event_ids"] or "[]"))
                mentions += loser["mention_count"]
                conf = max(conf, loser["confidence"])
                self.delete_memory(loser["id"])
                removed += 1
            self.update_memory(
                survivor["id"],
                source_event_ids=sorted(sources),
                mention_count=mentions,
                confidence=conf,
            )
        return removed

    def search_memories(
        self,
        *,
        query: str | None = None,
        entity: str | None = None,
        entity_type: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 20,
        match_any: bool = False,
    ) -> list[dict]:
        """Search memories by full-text query or by entity name / type.

        ``match_any`` switches the FTS query from implicit AND to OR over
        the tokens. The default AND is right for targeted lookups; recall
        passes whole-message keyword sets where requiring every token to
        appear in one fact would almost never match (bm25 rank still puts
        multi-token hits first).
        """
        if query:
            # FTS5 search. Sanitize like the LCM stores — this site passed
            # the raw query into MATCH, so punctuation ('.', '?') raised
            # fts5 syntax errors (same class as the lcm_expand_query bug).
            from prometheus.memory.lcm_fts5 import sanitize_fts5_query

            safe_query = sanitize_fts5_query(query)
            if not safe_query:
                return []
            if match_any:
                # The sanitizer emits space-joined quoted word tokens (no
                # spaces inside a token), so splitting on " " is exact.
                safe_query = " OR ".join(safe_query.split(" "))
            sql = (
                "SELECT m.* FROM memories m"
                " JOIN memories_fts fts ON m.id = fts.id"
                " WHERE memories_fts MATCH ? AND m.confidence >= ?"
                " ORDER BY rank LIMIT ?"
            )
            params = (safe_query, min_confidence, limit)
        elif entity:
            sql = (
                "SELECT * FROM memories"
                " WHERE entity_name LIKE ? AND confidence >= ?"
                " ORDER BY confidence DESC LIMIT ?"
            )
            params = (f"%{entity}%", min_confidence, limit)
        elif entity_type:
            sql = (
                "SELECT * FROM memories"
                " WHERE entity_type = ? AND confidence >= ?"
                " ORDER BY confidence DESC LIMIT ?"
            )
            params = (entity_type, min_confidence, limit)
        else:
            sql = (
                "SELECT * FROM memories WHERE confidence >= ?"
                " ORDER BY confidence DESC LIMIT ?"
            )
            params = (min_confidence, limit)
        rows = self._read(lambda conn: conn.execute(sql, params).fetchall())

        results = []
        for row in rows:
            d = dict(row)
            d["source_event_ids"] = json.loads(d["source_event_ids"])
            d["tags"] = json.loads(d["tags"])
            results.append(d)
        return results

    def get_all_memories(
        self,
        *,
        min_confidence: float = 0.0,
        limit: int = 1000,
    ) -> list[dict]:
        """Return all memories above *min_confidence*, newest first."""
        rows = self._read(
            lambda conn: conn.execute(
                "SELECT * FROM memories WHERE confidence >= ?"
                " ORDER BY timestamp DESC LIMIT ?",
                (min_confidence, limit),
            ).fetchall()
        )
        results = []
        for row in rows:
            d = dict(row)
            d["source_event_ids"] = json.loads(d["source_event_ids"])
            d["tags"] = json.loads(d["tags"])
            results.append(d)
        return results

    def update_memory(self, memory_id: str, **fields: object) -> None:
        """Update arbitrary columns on a memory by ID."""
        if not fields:
            return
        allowed = {
            "entity_type", "entity_name", "relationship", "fact",
            "confidence", "source_event_ids", "last_mentioned",
            "mention_count", "tags",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return

        # Serialize JSON fields
        for key in ("source_event_ids", "tags"):
            if key in updates and isinstance(updates[key], list):
                updates[key] = json.dumps(updates[key])

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [memory_id]

        def _op(conn):
            # FTS re-index happens via the memories_fts_au trigger.
            conn.execute(
                f"UPDATE memories SET {set_clause} WHERE id = ?", values  # noqa: S608
            )

        self._write(_op)

    def delete_memory(self, memory_id: str) -> None:
        """Delete a memory (its FTS entry follows via the AD trigger)."""
        def _op(conn):
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

        self._write(_op)

    def get_memory(self, memory_id: str) -> dict | None:
        """Return a single memory by ID."""
        row = self._read(
            lambda conn: conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        )
        if row is None:
            return None
        d = dict(row)
        d["source_event_ids"] = json.loads(d["source_event_ids"])
        d["tags"] = json.loads(d["tags"])
        return d

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def add_summary(
        self,
        summary_text: str,
        source_message_ids: list[str],
        *,
        level: int = 1,
        summary_id: str | None = None,
    ) -> str:
        """Store a conversation summary. Returns the summary ID."""
        sid = summary_id or uuid4().hex
        self._write(
            lambda conn: conn.execute(
                "INSERT INTO summaries (id, source_message_ids, summary_text, level, timestamp)"
                " VALUES (?, ?, ?, ?, ?)",
                (sid, json.dumps(source_message_ids), summary_text, level, time.time()),
            )
        )
        return sid

    def get_summaries(self, *, level: int | None = None, limit: int = 10) -> list[dict]:
        """Return summaries, newest first."""
        if level is not None:
            sql = (
                "SELECT * FROM summaries WHERE level = ?"
                " ORDER BY timestamp DESC LIMIT ?"
            )
            params: tuple = (level, limit)
        else:
            sql = "SELECT * FROM summaries ORDER BY timestamp DESC LIMIT ?"
            params = (limit,)
        rows = self._read(lambda conn: conn.execute(sql, params).fetchall())
        results = []
        for row in rows:
            d = dict(row)
            d["source_message_ids"] = json.loads(d["source_message_ids"])
            results.append(d)
        return results

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
