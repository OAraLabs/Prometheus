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
                content_json TEXT,
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
        self._migrate_add_content_json()

    def _migrate_add_content_json(self) -> None:
        """Additive, idempotent migration for the structured-content column.

        Older DBs predate ``content_json``. ``ALTER TABLE ADD COLUMN`` is O(1) in SQLite
        (metadata-only — the append-only rows are never rewritten and read back as NULL), and
        the guard makes reopening an already-migrated DB a no-op. Fresh DBs already have the
        column from ``CREATE TABLE`` above, so this is a no-op for them too.
        """
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(lcm_messages)")}
        if "content_json" not in cols:
            self._conn.execute("ALTER TABLE lcm_messages ADD COLUMN content_json TEXT")
            self._conn.commit()

    # ------------------------------------------------------------------
    # Row <-> dataclass helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> MessagePart:
        keys = row.keys()
        return MessagePart(
            role=row["role"],
            content=row["content"],
            timestamp=row["timestamp"],
            message_id=row["id"],
            session_id=row["session_id"],
            turn_index=row["turn_index"],
            token_count=row["token_count"],
            # Present only when the query SELECTs `rowid AS row_id` (messages_after_id);
            # other read paths don't need it.
            row_id=row["row_id"] if "row_id" in keys else 0,
            content_json=row["content_json"] if "content_json" in keys else None,
        )

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def add_message(self, session_id: str, msg: MessagePart) -> str:
        """Insert a message, forcing ``msg.session_id = session_id``.

        Thin adapter that closes the contract gap between ``LCMEngine``
        (which constructs ``MessagePart`` then passes session_id as a
        separate arg) and the internal ``insert_message`` (which derives
        session_id from the MessagePart). Overwrites unconditionally so
        the caller's argument always wins, matching the long-standing
        test-shim behaviour before this method landed in the class.

        Prefer :meth:`insert_message` for internal callers that already
        construct the MessagePart with session_id set.
        """
        msg.session_id = session_id
        return self.insert_message(msg)

    def insert_message(self, msg: MessagePart) -> str:
        """Insert a message and update the FTS5 index. Returns the message id."""
        mid = msg.message_id or uuid4().hex
        ts = msg.timestamp or time.time()

        self._conn.execute(
            "INSERT OR REPLACE INTO lcm_messages"
            " (id, session_id, turn_index, role, content, content_json, token_count, timestamp, compacted)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (mid, msg.session_id, msg.turn_index, msg.role, msg.content, msg.content_json, msg.token_count, ts),
        )

        # Sync FTS index — use the rowid of the just-inserted row.
        rowid = self._conn.execute(
            "SELECT rowid FROM lcm_messages WHERE id = ?", (mid,)
        ).fetchone()[0]
        # Surface the durable rowid back to the caller (canonical wire message id).
        msg.row_id = int(rowid)
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

    def count_all(self, session_id: str) -> int:
        """Return the total number of messages in a session, compacted or not.

        Counterpart to :meth:`count_uncompacted` used by ``LCMAssembler``
        for total-tokens accounting.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM lcm_messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["cnt"]) if row else 0

    def get_all_messages(self, session_id: str) -> list[MessagePart]:
        """All messages for a session ordered by turn_index ASC, no limit.

        Includes compacted messages. Unlike :meth:`get_messages` (which
        caps at ``limit=500``), this returns the full session — used by
        ``LCMAssembler`` to compute total token counts before deciding
        what fits in the assembly budget.
        """
        rows = self._conn.execute(
            "SELECT * FROM lcm_messages WHERE session_id = ? "
            "ORDER BY turn_index ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_message(r) for r in rows]

    def get_uncompacted_messages(self, session_id: str) -> list[MessagePart]:
        """All uncompacted messages for a session, turn_index ASC, no limit.

        Used by :class:`LCMCompactor` (to decide which messages to fold
        into the next summary) and :class:`LCMAssembler` (fresh-tail
        candidates). Differs from :meth:`get_fresh_tail` in that
        ``get_fresh_tail`` caps at ``count`` from the newest end; this
        returns every uncompacted row in chronological order.
        """
        rows = self._conn.execute(
            "SELECT * FROM lcm_messages WHERE session_id = ? AND compacted = 0 "
            "ORDER BY turn_index ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_message(r) for r in rows]

    def messages_since(
        self,
        watermark: float,
        *,
        limit: int = 500,
        session_id: str | None = None,
        include_compacted: bool = False,
    ) -> list[MessagePart]:
        """Strictly-greater watermark read, ordered by timestamp ASC.

        Returns messages with ``timestamp > watermark``. The MemoryExtractor
        read path: matches its existing ``_last_processed_ts`` invariant
        exactly (strictly greater than, not ``>=``).

        By default skips compacted messages so the extractor doesn't
        re-process summaries-of-summaries — pass ``include_compacted=True``
        to override. ``session_id=None`` (the default) reads across
        sessions.
        """
        sql = "SELECT * FROM lcm_messages WHERE timestamp > ?"
        params: list[object] = [watermark]
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        if not include_compacted:
            sql += " AND compacted = 0"
        sql += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_message(r) for r in rows]

    def max_timestamp(self, session_id: str) -> float:
        """Current max ``timestamp`` for a session, or ``0.0`` if it has none.

        This is the "current watermark" the REST history route returns so a
        client can tell it has caught up even when an incremental ``since`` read
        comes back empty.
        """
        row = self._conn.execute(
            "SELECT MAX(timestamp) AS mx FROM lcm_messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return float(row["mx"]) if row and row["mx"] is not None else 0.0

    def messages_after_id(
        self,
        row_id: int,
        *,
        limit: int = 10_000,
        session_id: str | None = None,
        include_compacted: bool = True,
    ) -> list[MessagePart]:
        """Durable, restart-stable read: messages with ``rowid > row_id``, ordered by
        ``rowid`` ASC (insertion order — monotonic and unique, unlike ``turn_index``,
        which is the in-memory list position and repeats across restart/trim).

        This is the canonical history + incremental cursor for the REST surface: the
        rowid is the durable message identity (the store is append-only, so rowids never
        reset or repeat). Distinct from :meth:`messages_since`, which filters by
        timestamp for the MemoryExtractor's ``_last_processed_ts`` contract — that one
        must stay timestamp-based, so this is a separate method.
        """
        sql = "SELECT rowid AS row_id, * FROM lcm_messages WHERE rowid > ?"
        params: list[object] = [row_id]
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        if not include_compacted:
            sql += " AND compacted = 0"
        sql += " ORDER BY rowid ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_message(r) for r in rows]

    def max_rowid(self, session_id: str) -> int:
        """Current max ``rowid`` for a session, or ``0`` if it has none.

        The durable, monotonic, restart-stable watermark the REST history route returns
        and that the WS user-echo reports as the just-persisted message's canonical id.
        """
        row = self._conn.execute(
            "SELECT MAX(rowid) AS mx FROM lcm_messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["mx"]) if row and row["mx"] is not None else 0

    def has_message(self, message_id: str) -> bool:
        """Return ``True`` iff a row with this id is persisted.

        Used by :meth:`LCMEngine.is_ingested` to answer "is this message
        durably stored?" without round-tripping a full row.
        """
        if not message_id:
            return False
        row = self._conn.execute(
            "SELECT 1 FROM lcm_messages WHERE id = ? LIMIT 1",
            (message_id,),
        ).fetchone()
        return row is not None

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
