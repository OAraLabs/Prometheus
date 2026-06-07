"""SQLite DAG summary storage for Lossless Context Management.

Stores :class:`SummaryNode` entries that form a directed acyclic graph of
progressively deeper summaries.  Provides FTS5 search over summary text
and helpers for DAG traversal (leaves, roots, ancestors, children).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

from prometheus.config.paths import get_config_dir
from prometheus.memory.lcm_fts5 import sanitize_fts5_query
from prometheus.memory.lcm_types import SummaryNode

_DB_NAME = "lcm.db"


def _default_db_path() -> Path:
    return get_config_dir() / _DB_NAME


class LCMSummaryStore:
    """SQLite store for the LCM summary DAG with FTS5 search.

    The underlying database file is shared with :class:`LCMConversationStore`;
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

            CREATE TABLE IF NOT EXISTS lcm_summaries (
                id                  TEXT PRIMARY KEY,
                session_id          TEXT,
                parent_ids          TEXT NOT NULL DEFAULT '[]',
                source_message_ids  TEXT NOT NULL DEFAULT '[]',
                summary_text        TEXT NOT NULL,
                depth               INTEGER NOT NULL DEFAULT 0,
                token_count         INTEGER NOT NULL DEFAULT 0,
                created_at          REAL NOT NULL,
                is_leaf             INTEGER NOT NULL DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_lcm_summaries_depth
                ON lcm_summaries (depth);

            CREATE INDEX IF NOT EXISTS idx_lcm_summaries_leaf
                ON lcm_summaries (is_leaf);

            CREATE VIRTUAL TABLE IF NOT EXISTS lcm_summaries_fts USING fts5(
                summary_text,
                content='lcm_summaries',
                content_rowid='rowid'
            );
        """)
        self._conn.commit()
        self._migrate_add_session_id()

    def _migrate_add_session_id(self) -> None:
        """Additive, idempotent migration: session-scope the summary DAG.

        Older DBs predate per-session summaries (the table was global, which left
        the assembler/compactor's ``get_leaf_summaries(session_id)`` etc. with no
        column to filter on). ``ALTER TABLE ADD COLUMN`` is O(1) metadata-only in
        SQLite, and the guard makes reopening a no-op. Fresh DBs already have the
        column from ``CREATE TABLE``. The session index is created here, after the
        column is guaranteed to exist.
        """
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(lcm_summaries)")}
        if "session_id" not in cols:
            self._conn.execute("ALTER TABLE lcm_summaries ADD COLUMN session_id TEXT")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_lcm_summaries_session"
            " ON lcm_summaries (session_id, is_leaf)"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Row <-> dataclass helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> SummaryNode:
        return SummaryNode(
            id=row["id"],
            parent_ids=json.loads(row["parent_ids"]),
            source_message_ids=json.loads(row["source_message_ids"]),
            summary_text=row["summary_text"],
            depth=row["depth"],
            token_count=row["token_count"],
            created_at=row["created_at"],
            is_leaf=bool(row["is_leaf"]),
        )

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert_summary(self, node: SummaryNode, *, session_id: str | None = None) -> str:
        """Insert a summary node into the DAG.

        * Marks all parent nodes as ``is_leaf = 0``.
        * Updates the FTS5 index.
        * Returns the node id.

        ``session_id`` partitions the DAG per conversation (the compactor inserts
        via :meth:`add_summary`); it is optional for back-compat with global
        callers/tests (stored as NULL).
        """
        nid = node.id or uuid4().hex
        created = node.created_at or time.time()

        self._conn.execute(
            "INSERT OR REPLACE INTO lcm_summaries"
            " (id, session_id, parent_ids, source_message_ids, summary_text,"
            "  depth, token_count, created_at, is_leaf)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                nid,
                session_id,
                json.dumps(node.parent_ids),
                json.dumps(node.source_message_ids),
                node.summary_text,
                node.depth,
                node.token_count,
                created,
                int(node.is_leaf),
            ),
        )

        # Mark parents as non-leaf since they now have a child.
        if node.parent_ids:
            placeholders = ",".join("?" for _ in node.parent_ids)
            self._conn.execute(
                f"UPDATE lcm_summaries SET is_leaf = 0 WHERE id IN ({placeholders})",
                node.parent_ids,
            )

        # Sync FTS index.
        rowid = self._conn.execute(
            "SELECT rowid FROM lcm_summaries WHERE id = ?", (nid,)
        ).fetchone()[0]
        self._conn.execute(
            "INSERT OR REPLACE INTO lcm_summaries_fts (rowid, summary_text) VALUES (?, ?)",
            (rowid, node.summary_text),
        )
        self._conn.commit()
        return nid

    # ------------------------------------------------------------------
    # Single-node lookups
    # ------------------------------------------------------------------

    def get_by_id(self, summary_id: str) -> SummaryNode | None:
        """Return a single summary node by id, or ``None``."""
        row = self._conn.execute(
            "SELECT * FROM lcm_summaries WHERE id = ?", (summary_id,)
        ).fetchone()
        return self._row_to_node(row) if row else None

    # ------------------------------------------------------------------
    # DAG queries
    # ------------------------------------------------------------------

    def get_leaves(self, *, max_depth: int | None = None) -> list[SummaryNode]:
        """Return leaf nodes (those not yet summarised further).

        If *max_depth* is given, only leaves at that depth or below are returned.
        """
        if max_depth is not None:
            rows = self._conn.execute(
                "SELECT * FROM lcm_summaries"
                " WHERE is_leaf = 1 AND depth <= ?"
                " ORDER BY created_at ASC",
                (max_depth,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM lcm_summaries WHERE is_leaf = 1"
                " ORDER BY created_at ASC"
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def add_summary(self, session_id: str, node: SummaryNode) -> str:
        """Session-scoped insert — the compactor's accessor (mirrors the
        conversation store's ``add_message(session_id, msg)``)."""
        return self.insert_summary(node, session_id=session_id)

    def get_leaf_summaries(self, session_id: str) -> list[SummaryNode]:
        """Leaf summary nodes for one session (assembler + compactor accessor).

        Restores a method the callers always expected but the store never had:
        its absence made ``LCMAssembler.assemble`` raise ``AttributeError`` on
        every call, which ``GET /api/lcm`` silently turned into all-zeros.
        """
        rows = self._conn.execute(
            "SELECT * FROM lcm_summaries WHERE session_id = ? AND is_leaf = 1"
            " ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_leaf_summaries_at_depth(self, session_id: str, depth: int) -> list[SummaryNode]:
        """Leaf summary nodes at exactly *depth* for one session (cascade input)."""
        rows = self._conn.execute(
            "SELECT * FROM lcm_summaries"
            " WHERE session_id = ? AND is_leaf = 1 AND depth = ?"
            " ORDER BY created_at ASC",
            (session_id, depth),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_max_depth(self, session_id: str) -> int:
        """Highest summary depth present for a session (0 when there are none)."""
        row = self._conn.execute(
            "SELECT MAX(depth) AS d FROM lcm_summaries WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["d"]) if row and row["d"] is not None else 0

    def mark_non_leaf(self, summary_ids: list[str]) -> None:
        """Mark nodes as non-leaf — they've been merged into a higher-depth parent."""
        if not summary_ids:
            return
        placeholders = ",".join("?" for _ in summary_ids)
        self._conn.execute(
            f"UPDATE lcm_summaries SET is_leaf = 0 WHERE id IN ({placeholders})",
            summary_ids,
        )
        self._conn.commit()

    def get_by_depth(self, depth: int, *, limit: int = 100) -> list[SummaryNode]:
        """Return summaries at the given depth, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM lcm_summaries WHERE depth = ?"
            " ORDER BY created_at ASC LIMIT ?",
            (depth, limit),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_roots(self) -> list[SummaryNode]:
        """Return depth-0 summaries (direct summaries of raw messages)."""
        return self.get_by_depth(0, limit=10_000)

    def get_children(self, parent_id: str) -> list[SummaryNode]:
        """Find nodes whose ``parent_ids`` contain *parent_id*.

        Uses a JSON ``LIKE`` check which is sufficient for hex UUIDs that
        cannot collide with JSON syntax characters.
        """
        pattern = f'%"{parent_id}"%'
        rows = self._conn.execute(
            "SELECT * FROM lcm_summaries WHERE parent_ids LIKE ?"
            " ORDER BY created_at ASC",
            (pattern,),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_ancestors(self, node_id: str) -> list[SummaryNode]:
        """Walk up the DAG from *node_id*, collecting all ancestor nodes.

        Returns ancestors in bottom-up order (immediate parents first).
        Performs an iterative BFS to avoid deep recursion.
        """
        ancestors: list[SummaryNode] = []
        visited: set[str] = set()
        queue: list[str] = [node_id]

        while queue:
            current_id = queue.pop(0)
            node = self.get_by_id(current_id)
            if node is None:
                continue
            for pid in node.parent_ids:
                if pid not in visited:
                    visited.add(pid)
                    parent = self.get_by_id(pid)
                    if parent is not None:
                        ancestors.append(parent)
                        queue.append(pid)

        return ancestors

    # ------------------------------------------------------------------
    # FTS5 search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[SummaryNode]:
        """Full-text search across summary text.

        If *session_id* is given, only summaries whose ``source_message_ids``
        include at least one message belonging to that session are returned.
        (The summary table itself has no session column; the filter joins
        against ``lcm_messages`` via the JSON-encoded id list — same pattern
        as :meth:`get_children` uses for parent ids.)

        An empty or all-punctuation query returns an empty list.
        """
        safe_query = sanitize_fts5_query(query)
        if not safe_query:
            return []

        if session_id is None:
            rows = self._conn.execute(
                "SELECT s.* FROM lcm_summaries s"
                " JOIN lcm_summaries_fts fts ON s.rowid = fts.rowid"
                " WHERE lcm_summaries_fts MATCH ?"
                " ORDER BY fts.rank LIMIT ?",
                (safe_query, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT DISTINCT s.*, fts.rank AS _rank FROM lcm_summaries s"
                " JOIN lcm_summaries_fts fts ON s.rowid = fts.rowid"
                " JOIN lcm_messages m"
                "   ON s.source_message_ids LIKE '%\"' || m.id || '\"%'"
                " WHERE lcm_summaries_fts MATCH ? AND m.session_id = ?"
                " ORDER BY _rank LIMIT ?",
                (safe_query, session_id, limit),
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return aggregate statistics about the summary DAG."""
        row = self._conn.execute(
            "SELECT"
            " COUNT(*) AS total,"
            " COALESCE(MAX(depth), 0) AS max_depth,"
            " SUM(CASE WHEN is_leaf = 1 THEN 1 ELSE 0 END) AS leaf_count"
            " FROM lcm_summaries"
        ).fetchone()
        return {
            "count": row["total"],
            "max_depth": row["max_depth"],
            "leaf_count": row["leaf_count"],
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> LCMSummaryStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
