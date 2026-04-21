"""SQLite-backed memory store with FTS5 search.

Schema mirrors OpenClaw's proven memories table structure.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

from prometheus.config.paths import get_config_dir

_DB_NAME = "memory.db"


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
                timestamp         REAL NOT NULL
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

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

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
    ) -> str:
        """Insert or update a memory fact. Returns the memory ID.

        If a memory with the same entity_name + fact already exists,
        its confidence and mention_count are updated instead.
        """
        now = time.time()
        rel = relationship or "fact"

        # Check for existing duplicate
        existing = self._conn.execute(
            "SELECT id, mention_count FROM memories"
            " WHERE entity_name = ? AND fact = ?",
            (entity_name, fact),
        ).fetchone()

        if existing:
            mid = existing["id"]
            self._conn.execute(
                "UPDATE memories SET confidence = MAX(confidence, ?),"
                " mention_count = mention_count + 1, last_mentioned = ?"
                " WHERE id = ?",
                (confidence, now, mid),
            )
            self._conn.commit()
            return mid

        mid = memory_id or uuid4().hex
        self._conn.execute(
            "INSERT INTO memories"
            " (id, entity_type, entity_name, relationship, fact, confidence,"
            "  source_event_ids, last_mentioned, mention_count, tags, timestamp)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
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
            ),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO memories_fts (id, entity_name, fact)"
            " VALUES (?, ?, ?)",
            (mid, entity_name, fact),
        )
        self._conn.commit()
        return mid

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
            # FTS5 search
            rows = self._conn.execute(
                "SELECT m.* FROM memories m"
                " JOIN memories_fts fts ON m.id = fts.id"
                " WHERE memories_fts MATCH ? AND m.confidence >= ?"
                " ORDER BY rank LIMIT ?",
                (query, min_confidence, limit),
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
