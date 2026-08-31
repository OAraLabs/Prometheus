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

from prometheus.config.paths import get_lcm_db_path
from prometheus.memory.lcm_fts5 import sanitize_fts5_query
from prometheus.memory.lcm_types import MessagePart


class LCMConversationStore:
    """SQLite store for conversation messages with FTS5 search.

    The underlying database file is shared with :class:`LCMSummaryStore` and
    :class:`~prometheus.coordinator.divergence.CheckpointStore`; each store
    owns its own tables within the same ``lcm.db`` file. "Shared" is load
    bearing and was fiction until 2026-08-12 — see
    :func:`~prometheus.config.paths.get_lcm_db_path`.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path if db_path is not None else get_lcm_db_path()
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
                compacted   INTEGER NOT NULL DEFAULT 0,
                provenance  TEXT NOT NULL DEFAULT 'user',
                is_trusted  INTEGER NOT NULL DEFAULT 1
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

            -- Durable "forget session" (feat/durable-session-index): hides a
            -- session from list_sessions() without touching its append-only
            -- rows. deleted_at is a watermark, not a hard delete — activity
            -- NEWER than it revives the session (gateway ids like telegram:123
            -- are stable, so a forgotten chat must resurface when it speaks).
            CREATE TABLE IF NOT EXISTS session_tombstones (
                session_id TEXT PRIMARY KEY,
                deleted_at REAL NOT NULL
            );

            -- Session display titles (GRAFT-MOBILE-BRIDGE 7). A session id is
            -- <gateway>:<uuid> and carries no name; a phone list of those is
            -- unreadable. Generated from the first exchange (or set manually
            -- via PUT), kept OUT of the append-only lcm_messages rows so a
            -- rename never rewrites history. One row per session, last write
            -- wins.
            CREATE TABLE IF NOT EXISTS session_titles (
                session_id TEXT PRIMARY KEY,
                title      TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            -- Pinned sessions. A pin is a property OF the conversation, not of
            -- the client that set it: Beacon Desktop already pins, but stores it
            -- in its own local settings, so a pin never reached the phone and two
            -- surfaces disagreed about the same session. Kept here, beside the
            -- title, for the same reason — outside the append-only rows, so a pin
            -- or unpin never rewrites history. Presence IS the pin; unpinning
            -- deletes the row rather than storing false, so the column stays a
            -- clean present/absent signal (same discipline as a blank title).
            CREATE TABLE IF NOT EXISTS session_pins (
                session_id TEXT PRIMARY KEY,
                pinned_at  REAL NOT NULL
            );

            -- Per-session agent profile. DURABLE on purpose: the model router's
            -- per-session overrides live in a plain dict on the router instance and
            -- vanish on restart, which is the whole class of defect where a setting
            -- silently reverts and nobody can tell it did. A profile chosen for a
            -- conversation is a property OF the conversation, so it outlives the
            -- process. Absence means "use the daemon-wide active profile" — the row
            -- is only ever written when someone chose something.
            CREATE TABLE IF NOT EXISTS session_profiles (
                session_id TEXT PRIMARY KEY,
                profile    TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        self._conn.commit()
        self._migrate_add_content_json()
        self._migrate_add_trust_fields()

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

    def _migrate_add_trust_fields(self) -> None:
        """Additive, idempotent migration for the provenance / is_trusted columns.

        Persists the per-turn trust tag set by ``inject_turn`` so it survives the
        LCM write→read round-trip. ``ALTER TABLE ADD COLUMN`` with a constant
        ``DEFAULT`` is O(1) and **backfills every pre-existing row**: those rows
        predate task injection and are legitimate history, so they resolve to
        ``provenance='user'`` / ``is_trusted=1`` (trusted) — never mis-tagged as
        untrusted. Fresh DBs already have the columns from ``CREATE TABLE`` above,
        so this is a no-op for them. New inserts always write the columns
        explicitly (see :meth:`insert_message`), so they never depend on the
        column default.
        """
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(lcm_messages)")}
        changed = False
        if "provenance" not in cols:
            self._conn.execute(
                "ALTER TABLE lcm_messages ADD COLUMN provenance TEXT NOT NULL DEFAULT 'user'"
            )
            changed = True
        if "is_trusted" not in cols:
            self._conn.execute(
                "ALTER TABLE lcm_messages ADD COLUMN is_trusted INTEGER NOT NULL DEFAULT 1"
            )
            changed = True
        if changed:
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
            # Trust tag. All read paths SELECT *, so these are present; the guards
            # (and the None check on is_trusted) keep the SAFE (user, trusted)
            # fallback for any partial-column read — never a false untrusted tag.
            provenance=row["provenance"] if "provenance" in keys else "user",
            is_trusted=(
                bool(row["is_trusted"])
                if "is_trusted" in keys and row["is_trusted"] is not None
                else True
            ),
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

        # Trust columns are written EXPLICITLY from the MessagePart — never left
        # to the column DEFAULT — so a (task_supervisor, False) turn can never be
        # silently up-tagged to the trusted default on insert.
        self._conn.execute(
            "INSERT OR REPLACE INTO lcm_messages"
            " (id, session_id, turn_index, role, content, content_json, token_count, timestamp, compacted,"
            " provenance, is_trusted)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
            (
                mid, msg.session_id, msg.turn_index, msg.role, msg.content, msg.content_json,
                msg.token_count, ts, msg.provenance, 1 if msg.is_trusted else 0,
            ),
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
        limit: int = 500,
    ) -> list[MessagePart]:
        """Return messages for a session ordered by turn_index ascending.

        ⚠ NO ``since_turn``. It existed here from the initial commit and no
        caller ever passed it, in ``src/`` or in ``tests/`` — so the
        incremental branch it guarded had never executed. The incremental
        read that the wire contract actually uses is
        :meth:`messages_after_id`, keyed on the durable LCM **rowid** rather
        than ``turn_index``, and that is the one the ``?since=`` query
        parameter reaches (``web/server.py``). Two filters for one job, one
        of them wired: the unwired one is gone rather than left as an
        untested second way to do it.
        """
        rows = self._conn.execute(
            "SELECT * FROM lcm_messages WHERE session_id = ?"
            " ORDER BY turn_index ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
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

    def search_snippets(
        self,
        query: str,
        *,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Full-text search returning FTS5 snippet() hits for the wire search API.

        Unlike :meth:`search` (which returns full ``MessagePart`` objects), this
        returns only what a search-results surface needs — the rowid (the durable
        wire message id the client jumps to), the uuid, role, time, BM25 rank, and
        a server-generated snippet with ⟦⟧ match markers. The client never holds
        the full message text, so offsets into it would be unapplyable — markers
        in the snippet are the contract (see the global-search spec).

        An empty or all-punctuation query returns an empty list.
        """
        safe_query = sanitize_fts5_query(query)
        if not safe_query:
            return []

        where = "lcm_messages_fts MATCH ?"
        params: list = [safe_query]
        if session_id is not None:
            where += " AND m.session_id = ?"
            params.append(session_id)
        params.append(limit)

        rows = self._conn.execute(
            "SELECT m.rowid AS row_id, m.id AS uuid, m.session_id, m.role,"
            " m.timestamp, fts.rank AS rank,"
            " snippet(lcm_messages_fts, 0, '⟦', '⟧', '…', 24) AS snippet"
            " FROM lcm_messages m"
            " JOIN lcm_messages_fts fts ON m.rowid = fts.rowid"
            f" WHERE {where}"
            " ORDER BY fts.rank LIMIT ?",
            params,
        ).fetchall()
        return [
            {
                "row_id": int(r["row_id"]),
                "uuid": r["uuid"],
                "session_id": r["session_id"],
                "role": r["role"],
                "timestamp": r["timestamp"],
                "score": float(r["rank"]),
                "snippet": r["snippet"],
            }
            for r in rows
        ]

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

        NOTE: ``row_id`` on the returned parts is 0 — ``SELECT *`` does not
        include SQLite's implicit rowid. If you need durable wire ids, read
        through :meth:`messages_after_id` (which SELECTs ``rowid AS row_id``);
        do not "fix" this by keying anything on the zeros.
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

    def messages_page(
        self,
        *,
        limit: int,
        before: int | None = None,
        session_id: str | None = None,
        include_compacted: bool = True,
    ) -> tuple[list[MessagePart], bool]:
        """One page of history, NEWEST-ANCHORED (GRAFT-MOBILE-BRIDGE 5).

        ``before=None`` starts at the newest row; ``before=<rowid>`` returns
        rows with ``rowid < before``. Selected ``rowid`` DESC (that is what
        makes it a newest-first page), then REVERSED so the returned list is
        ascending like every other read — callers never see the reversal.

        Fetches ``limit + 1`` to answer ``has_more`` honestly instead of
        guessing from a full page. Distinct from :meth:`messages_after_id`
        (the forward ``?since=`` cursor): a forward cursor cannot page
        backwards from the present, which is exactly the mobile cold-open.
        """
        sql = "SELECT rowid AS row_id, * FROM lcm_messages WHERE 1=1"
        params: list[object] = []
        if before is not None:
            sql += " AND rowid < ?"
            params.append(before)
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        if not include_compacted:
            sql += " AND compacted = 0"
        sql += " ORDER BY rowid DESC LIMIT ?"
        params.append(limit + 1)
        rows = self._conn.execute(sql, params).fetchall()
        has_more = len(rows) > limit
        page = rows[:limit]
        page.reverse()
        return [self._row_to_message(r) for r in page], has_more

    def max_rowid_all(self) -> int:
        """Highest rowid in the table, across EVERY session — or 0 if empty.

        Deliberately not derived from :meth:`list_sessions`, which excludes
        tombstoned sessions. A migration floor computed from that aggregate
        sits BELOW a forgotten chat's rows, so those rows read as new and get
        re-mined on the first pass — the exact replay the extractor cursor
        exists to prevent, reintroduced by the thing that prevents it.
        """
        row = self._conn.execute(
            "SELECT MAX(rowid) AS mx FROM lcm_messages"
        ).fetchone()
        return int(row["mx"]) if row and row["mx"] is not None else 0

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

    def rowids_for_message_ids(self, message_ids: list[str]) -> dict[str, int]:
        """Map message UUIDs (``lcm_messages.id``) to their durable rowids.

        The rowid is the wire cursor identity (``?since=``, ``messages_after_id``);
        a UUID is not a scroll position. Ids with no persisted row are simply
        absent from the result — callers decide how to handle the gap.

        Deduplicates and chunks the IN-clause so an unbounded caller-side id
        list (e.g. anchors flattened across many search hits) can never exceed
        SQLITE_MAX_VARIABLE_NUMBER (999 on SQLite < 3.32).
        """
        ids = list(dict.fromkeys(i for i in message_ids if i))
        out: dict[str, int] = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                f"SELECT rowid, id FROM lcm_messages WHERE id IN ({placeholders})",
                chunk,
            ).fetchall()
            out.update({r["id"]: int(r["rowid"]) for r in rows})
        return out

    # ------------------------------------------------------------------
    # Session index (feat/durable-session-index)
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[dict]:
        """Durable session index: one row of aggregates per ``session_id``.

        This is what lets GET /api/sessions survive a daemon restart — the
        in-memory SessionManager starts empty, but every persisted session is
        enumerable here. ``watermark`` is the SAME max-rowid cursor
        :meth:`max_rowid` reports, so a client can hand it straight to the
        incremental messages read.

        Tombstoned sessions are excluded unless they have activity newer than
        ``deleted_at`` (see the schema note — a forgotten stable-id chat must
        resurface when it speaks again). Ordered most-recently-active first.
        """
        rows = self._conn.execute(
            """
            SELECT m.session_id            AS session_id,
                   COUNT(*)                AS message_count,
                   MIN(m.timestamp)        AS first_timestamp,
                   MAX(m.timestamp)        AS last_timestamp,
                   MAX(m.rowid)            AS watermark,
                   ti.title                AS title,
                   pi.pinned_at            AS pinned_at
            FROM lcm_messages m
            LEFT JOIN session_tombstones t ON t.session_id = m.session_id
            LEFT JOIN session_titles ti ON ti.session_id = m.session_id
            LEFT JOIN session_pins pi ON pi.session_id = m.session_id
            GROUP BY m.session_id
            HAVING t.deleted_at IS NULL OR MAX(m.timestamp) > t.deleted_at
            ORDER BY MAX(m.timestamp) DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def tombstone_session(self, session_id: str) -> None:
        """Durably forget a session: hide it from :meth:`list_sessions`.

        The message rows are append-only and stay intact — this only records a
        ``deleted_at`` watermark. Re-forgetting refreshes the watermark, so a
        revived-then-forgotten session hides again.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO session_tombstones (session_id, deleted_at)"
            " VALUES (?, ?)",
            (session_id, time.time()),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Session titles (GRAFT-MOBILE-BRIDGE 7)
    # ------------------------------------------------------------------

    def set_session_title(self, session_id: str, title: str) -> None:
        """Set (or replace) a session's display title. Last write wins.

        A blank title clears the row rather than storing an empty string, so
        ``get_session_title`` stays a clean present/absent signal.
        """
        clean = (title or "").strip()
        if not clean:
            self._conn.execute(
                "DELETE FROM session_titles WHERE session_id = ?", (session_id,)
            )
        else:
            self._conn.execute(
                "INSERT OR REPLACE INTO session_titles (session_id, title, updated_at)"
                " VALUES (?, ?, ?)",
                (session_id, clean, time.time()),
            )
        self._conn.commit()

    def set_session_pinned(self, session_id: str, pinned: bool) -> None:
        """Pin or unpin a session. Unpinning DELETES the row.

        Presence is the pin — storing ``false`` would make "never pinned" and
        "unpinned" indistinguishable in the column, and every reader would have
        to know which falsy value it was looking at.
        """
        if pinned:
            self._conn.execute(
                "INSERT OR REPLACE INTO session_pins (session_id, pinned_at)"
                " VALUES (?, ?)",
                (session_id, time.time()),
            )
        else:
            self._conn.execute(
                "DELETE FROM session_pins WHERE session_id = ?", (session_id,)
            )
        self._conn.commit()

    def is_session_pinned(self, session_id: str) -> bool:
        """True when a pin row exists for this session."""
        row = self._conn.execute(
            "SELECT 1 FROM session_pins WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row is not None

    def set_session_profile(self, session_id: str, profile: str) -> None:
        """Bind a session to an agent profile. Blank CLEARS the binding.

        Mirrors :meth:`set_session_title` deliberately, including the clear-on-blank
        rule: absence has to stay a clean present/absent signal, because absence is
        what means "follow the daemon-wide active profile". Storing an empty string
        would make "no choice" and "chose nothing" indistinguishable.

        The NAME is not validated here — the store does not know what profiles exist.
        Validation belongs at the write surface, where a bad name can be refused with
        a reason instead of silently binding a session to a profile that will never
        resolve.
        """
        clean = (profile or "").strip()
        if not clean:
            self._conn.execute(
                "DELETE FROM session_profiles WHERE session_id = ?", (session_id,)
            )
        else:
            self._conn.execute(
                "INSERT OR REPLACE INTO session_profiles (session_id, profile, updated_at)"
                " VALUES (?, ?, ?)",
                (session_id, clean, time.time()),
            )
        self._conn.commit()

    def get_session_profile(self, session_id: str) -> str | None:
        """The profile bound to this session, or None to follow the global default."""
        row = self._conn.execute(
            "SELECT profile FROM session_profiles WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row[0] if row else None

    def get_session_title(self, session_id: str) -> str | None:
        """The session's title, or None if it has none."""
        row = self._conn.execute(
            "SELECT title FROM session_titles WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row["title"] if row else None

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
