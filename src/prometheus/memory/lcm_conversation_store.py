"""SQLite conversation storage for Lossless Context Management.

Provides a messages table with FTS5 full-text search, WAL journal mode,
and helpers for the compaction pipeline (fresh-tail retrieval, marking
messages as compacted, uncompacted counts).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

from prometheus.config.paths import get_config_dir
from prometheus.memory.lcm_fts5 import sanitize_fts5_query
from prometheus.memory.lcm_types import MessagePart

_DB_NAME = "lcm.db"


def _default_db_path() -> Path:
    return get_config_dir() / _DB_NAME


class LCMConversationStore:
    """SQLite store for conversation messages with FTS5 search.

    The underlying database file is shared with :class:`LCMSummaryStore`;
    each store owns its own tables within the same ``lcm.db`` file.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path if db_path is not None else _default_db_path()
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

            CREATE TABLE IF NOT EXISTS lcm_messages (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                turn_index  INTEGER NOT NULL DEFAULT 0,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0,
                timestamp   REAL NOT NULL,
                compacted   INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_lcm_messages_session
                ON lcm_messages (session_id, turn_index);

            CREATE INDEX IF NOT EXISTS idx_lcm_messages_compacted
                ON lcm_messages (session_id, compacted);

            CREATE VIRTUAL TABLE IF NOT EXISTS lcm_messages_fts USING fts5(
                content,
                content='lcm_messages',
                content_rowid='rowid'
            );

            -- Checkpoint table for divergence detection (Sprint 10)
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                goal_hash TEXT NOT NULL,
                goal_description TEXT,
                messages_json TEXT NOT NULL,
                tool_calls_json TEXT NOT NULL,
                divergence_score REAL DEFAULT 0.0,
                created_at REAL NOT NULL,
                UNIQUE(task_id, step_number)
            );

            CREATE INDEX IF NOT EXISTS idx_checkpoints_task
                ON checkpoints(task_id, step_number DESC);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Row <-> dataclass helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> MessagePart:
        return MessagePart(
            role=row["role"],
            content=row["content"],
            timestamp=row["timestamp"],
            message_id=row["id"],
            session_id=row["session_id"],
            turn_index=row["turn_index"],
            token_count=row["token_count"],
        )

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert_message(self, msg: MessagePart) -> str:
        """Insert a message and update the FTS5 index. Returns the message id."""
        mid = msg.message_id or uuid4().hex
        ts = msg.timestamp or time.time()

        self._conn.execute(
            "INSERT OR REPLACE INTO lcm_messages"
            " (id, session_id, turn_index, role, content, token_count, timestamp, compacted)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (mid, msg.session_id, msg.turn_index, msg.role, msg.content, msg.token_count, ts),
        )

        # Sync FTS index — use the rowid of the just-inserted row.
        rowid = self._conn.execute(
            "SELECT rowid FROM lcm_messages WHERE id = ?", (mid,)
        ).fetchone()[0]
        self._conn.execute(
            "INSERT OR REPLACE INTO lcm_messages_fts (rowid, content) VALUES (?, ?)",
            (rowid, msg.content),
        )
        self._conn.commit()
        return mid

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_messages(
        self,
        session_id: str,
        *,
        since_turn: int | None = None,
        limit: int = 500,
    ) -> list[MessagePart]:
        """Return messages for a session ordered by turn_index ascending."""
        query = "SELECT * FROM lcm_messages WHERE session_id = ?"
        params: list[object] = [session_id]
        if since_turn is not None:
            query += " AND turn_index >= ?"
            params.append(since_turn)
        query += " ORDER BY turn_index ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_message(r) for r in rows]

    def get_fresh_tail(self, session_id: str, count: int) -> list[MessagePart]:
        """Return the last *count* uncompacted messages for a session.

        Results are ordered oldest-first (ascending turn_index) so they can
        be appended directly to a prompt.
        """
        rows = self._conn.execute(
            "SELECT * FROM lcm_messages"
            " WHERE session_id = ? AND compacted = 0"
            " ORDER BY turn_index DESC LIMIT ?",
            (session_id, count),
        ).fetchall()
        # Reverse so the caller gets chronological order.
        return [self._row_to_message(r) for r in reversed(rows)]

    def mark_compacted(self, message_ids: list[str]) -> int:
        """Mark messages as compacted. Returns the number of rows affected."""
        if not message_ids:
            return 0
        placeholders = ",".join("?" for _ in message_ids)
        cur = self._conn.execute(
            f"UPDATE lcm_messages SET compacted = 1 WHERE id IN ({placeholders})",
            message_ids,
        )
        self._conn.commit()
        return cur.rowcount

    def search(
        self,
        query: str,
        *,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[MessagePart]:
        """Full-text search across message content.

        An empty or all-punctuation query returns an empty list.
        """
        safe_query = sanitize_fts5_query(query)
        if not safe_query:
            return []

        if session_id is not None:
            rows = self._conn.execute(
                "SELECT m.* FROM lcm_messages m"
                " JOIN lcm_messages_fts fts ON m.rowid = fts.rowid"
                " WHERE lcm_messages_fts MATCH ? AND m.session_id = ?"
                " ORDER BY fts.rank LIMIT ?",
                (safe_query, session_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT m.* FROM lcm_messages m"
                " JOIN lcm_messages_fts fts ON m.rowid = fts.rowid"
                " WHERE lcm_messages_fts MATCH ?"
                " ORDER BY fts.rank LIMIT ?",
                (safe_query, limit),
            ).fetchall()

        return [self._row_to_message(r) for r in rows]

    def count_uncompacted(self, session_id: str) -> int:
        """Return the number of uncompacted messages in a session."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM lcm_messages"
            " WHERE session_id = ? AND compacted = 0",
            (session_id,),
        ).fetchone()
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> LCMConversationStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
