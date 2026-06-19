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


class MemoryStore:
    """SQLite memory store with FTS5 full-text search.

    Tables:
      messages  — conversation history (with FTS5 index)
      memories  — extracted entity facts (with FTS5 index)
      summaries — compressed conversation summaries
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else _get_db_path()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._apply_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _apply_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
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
        """)
        self._conn.commit()
        self._migrate_manual_column()

    def _migrate_manual_column(self) -> None:
        """Add the ``manual`` flag column to a pre-manual-layer memories table.

        Idempotent: a no-op when the column already exists (the common path on
        every startup — no snapshot, no write). When it is missing (a DB created
        before the manual layer), snapshot the DB out-of-tree first, then ALTER.
        Fail loud — a failed snapshot or ALTER propagates rather than leaving the
        table half-migrated.
        """
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(memories)")}
        if "manual" in cols:
            return
        self._snapshot_db()
        self._conn.execute(
            "ALTER TABLE memories ADD COLUMN manual INTEGER NOT NULL DEFAULT 0"
        )
        self._conn.commit()

    def _snapshot_db(self) -> None:
        """Copy the DB file out-of-tree, timestamped, before a migration."""
        src = self._db_path
        if not src.exists():
            return  # nothing to back up (in-memory / brand-new DB)
        ts = time.strftime("%Y%m%dT%H%M%S", time.localtime())
        dst = src.with_name(f"{src.name}.backup-{ts}")
        shutil.copy2(src, dst)
        log.info("MemoryStore: snapshotted %s -> %s before manual migration", src, dst)

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
        self._conn.execute(
            "INSERT OR REPLACE INTO messages (id, session_id, role, content, timestamp, compressed)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (mid, session_id, role, content, now, int(compressed)),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO messages_fts (id, content) VALUES (?, ?)",
            (mid, content),
        )
        self._conn.commit()
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
        rows = self._conn.execute(query, params).fetchall()
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

        # Normalized duplicate check: scan this entity's rows (few) and
        # match on the normalized fact. Entity match is case-insensitive.
        candidates = self._conn.execute(
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
            self._conn.execute(
                "UPDATE memories SET confidence = MAX(confidence, ?),"
                " mention_count = mention_count + 1, last_mentioned = ?,"
                " source_event_ids = ?, manual = MAX(manual, ?) WHERE id = ?",
                (confidence, now, json.dumps(merged_sources), int(manual), mid),
            )
            self._conn.commit()
            return mid

        mid = memory_id or uuid4().hex
        self._conn.execute(
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
        self._conn.execute(
            "INSERT OR REPLACE INTO memories_fts (id, entity_name, fact)"
            " VALUES (?, ?, ?)",
            (mid, entity_name, fact),
        )
        self._conn.commit()
        return mid

    def dedupe_existing(self) -> int:
        """One-time backfill: collapse normalized-equal rows per entity.

        For each group of rows that are normalized-equal (same normalized
        entity_name + normalized fact), keep the earliest row and merge the
        rest into it: ``source_event_ids`` UNIONed, ``mention_count`` SUMMED,
        confidence MAXed. Dropping either source or count would silently
        shred provenance — so union-and-sum is the invariant, not a count
        target. Returns the number of rows removed. No-op on an empty store.
        """
        rows = self._conn.execute(
            "SELECT id, entity_name, fact, confidence, source_event_ids,"
            " mention_count, timestamp FROM memories"
        ).fetchall()

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
    ) -> list[dict]:
        """Search memories by full-text query or by entity name / type."""
        if query:
            # FTS5 search. Sanitize like the LCM stores — this site passed
            # the raw query into MATCH, so punctuation ('.', '?') raised
            # fts5 syntax errors (same class as the lcm_expand_query bug).
            from prometheus.memory.lcm_fts5 import sanitize_fts5_query

            safe_query = sanitize_fts5_query(query)
            if not safe_query:
                return []
            rows = self._conn.execute(
                "SELECT m.* FROM memories m"
                " JOIN memories_fts fts ON m.id = fts.id"
                " WHERE memories_fts MATCH ? AND m.confidence >= ?"
                " ORDER BY rank LIMIT ?",
                (safe_query, min_confidence, limit),
            ).fetchall()
        elif entity:
            rows = self._conn.execute(
                "SELECT * FROM memories"
                " WHERE entity_name LIKE ? AND confidence >= ?"
                " ORDER BY confidence DESC LIMIT ?",
                (f"%{entity}%", min_confidence, limit),
            ).fetchall()
        elif entity_type:
            rows = self._conn.execute(
                "SELECT * FROM memories"
                " WHERE entity_type = ? AND confidence >= ?"
                " ORDER BY confidence DESC LIMIT ?",
                (entity_type, min_confidence, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE confidence >= ?"
                " ORDER BY confidence DESC LIMIT ?",
                (min_confidence, limit),
            ).fetchall()

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
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE confidence >= ?"
            " ORDER BY timestamp DESC LIMIT ?",
            (min_confidence, limit),
        ).fetchall()
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
        self._conn.execute(
            f"UPDATE memories SET {set_clause} WHERE id = ?", values  # noqa: S608
        )

        # Update FTS5 if entity_name or fact changed
        if "entity_name" in updates or "fact" in updates:
            row = self._conn.execute(
                "SELECT entity_name, fact FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            if row:
                self._conn.execute(
                    "INSERT OR REPLACE INTO memories_fts (id, entity_name, fact)"
                    " VALUES (?, ?, ?)",
                    (memory_id, row["entity_name"], row["fact"]),
                )
        self._conn.commit()

    def delete_memory(self, memory_id: str) -> None:
        """Delete a memory and its FTS5 entry."""
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.execute(
            "DELETE FROM memories_fts WHERE id = ?", (memory_id,)
        )
        self._conn.commit()

    def get_memory(self, memory_id: str) -> dict | None:
        """Return a single memory by ID."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
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
        self._conn.execute(
            "INSERT INTO summaries (id, source_message_ids, summary_text, level, timestamp)"
            " VALUES (?, ?, ?, ?, ?)",
            (sid, json.dumps(source_message_ids), summary_text, level, time.time()),
        )
        self._conn.commit()
        return sid

    def get_summaries(self, *, level: int | None = None, limit: int = 10) -> list[dict]:
        """Return summaries, newest first."""
        if level is not None:
            rows = self._conn.execute(
                "SELECT * FROM summaries WHERE level = ?"
                " ORDER BY timestamp DESC LIMIT ?",
                (level, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM summaries ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
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
