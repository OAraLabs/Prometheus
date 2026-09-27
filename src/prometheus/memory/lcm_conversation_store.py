"""SQLite conversation storage for Lossless Context Management.

Provides a messages table with FTS5 full-text search, WAL journal mode,
and helpers for the compaction pipeline (fresh-tail retrieval, marking
messages as compacted, uncompacted counts).
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

from prometheus.config.paths import get_lcm_db_path
from prometheus.memory.lcm_fts5 import sanitize_fts5_query
from prometheus.memory.lcm_types import MessagePart
from prometheus.security.log_redaction import redact_json_text, redact_secrets

log = logging.getLogger(__name__)

# ``turn_index`` is a message's durable prompt position: unique within its
# session. Two schema objects serve it:
#
# * the plain index on ``(session_id, turn_index)``, which the readers and the
#   guard below use (created with the table);
# * the GUARD, a BEFORE INSERT trigger that refuses a row when another message
#   of the same session already holds its turn_index, with RAISE(ABORT). Only
#   the one-time migration in :mod:`prometheus.memory.lcm_turn_index_migration`
#   creates it; the daemon runs that at start, before any writer.
#
# Deliberately NOT a UNIQUE index. Builds from before the guard insert with
# ``INSERT OR REPLACE``, and REPLACE resolves a UNIQUE conflict by DELETING the
# row that holds the key: a rollback (or a pip downgrade) would then delete
# history on every collision. RAISE(ABORT) fails the statement whatever its
# conflict clause, so an old build's colliding insert fails and the older row
# stays. See docs/audits/LCM-TURN-INDEX-DUPLICATES.md.
TURN_INDEX_INDEX = "idx_lcm_messages_session"
TURN_INDEX_GUARD_TRIGGER = "lcm_messages_turn_index_guard"
TURN_INDEX_GUARD_MESSAGE = (
    "lcm_messages turn_index guard: this session already has a message at this turn_index"
)
# A re-insert of the SAME id is left to the statement's own conflict handling
# (``id <> NEW.id``): an old build's same-id REPLACE still works.
TURN_INDEX_GUARD_SQL = (
    f"CREATE TRIGGER IF NOT EXISTS {TURN_INDEX_GUARD_TRIGGER}"
    " BEFORE INSERT ON lcm_messages"
    " WHEN EXISTS (SELECT 1 FROM lcm_messages"
    "   WHERE session_id = NEW.session_id AND turn_index = NEW.turn_index"
    "   AND id <> NEW.id)"
    f" BEGIN SELECT RAISE(ABORT, '{TURN_INDEX_GUARD_MESSAGE}'); END"
)

_INSERT_COLUMNS = (
    "(id, session_id, turn_index, role, content, content_json, token_count, timestamp,"
    " compacted, provenance, is_trusted)"
)

# ON CONFLICT(id) DO NOTHING: re-inserting an id that is already stored is a
# no-op. It is deliberately NOT ``INSERT OR REPLACE``, which resolves a
# conflict by deleting the row that holds the key. A turn-key conflict (the
# guard trigger) still raises and is resolved in :meth:`insert_message`.
_INSERT_SQL = (
    f"INSERT INTO lcm_messages {_INSERT_COLUMNS}"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)"
    " ON CONFLICT(id) DO NOTHING"
)

# The same row, numbered by the store: the next free index in the session,
# computed and written in ONE statement, so no other writer can take that
# number in between. (The WHERE clause is also what lets SQLite parse the
# upsert after an INSERT ... SELECT.)
_INSERT_APPENDING_SQL = (
    f"INSERT INTO lcm_messages {_INSERT_COLUMNS}"
    " SELECT ?, ?, COALESCE(MAX(turn_index), -1) + 1, ?, ?, ?, ?, ?, 0, ?, ?"
    " FROM lcm_messages WHERE session_id = ?"
    " ON CONFLICT(id) DO NOTHING"
)


def _is_turn_key_conflict(exc: sqlite3.IntegrityError) -> bool:
    """True when *exc* is the turn-index guard refusing a row."""
    return TURN_INDEX_GUARD_MESSAGE in str(exc)


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

            -- Correlation ids for optimistic client rows (audit P9.6 / Beacon#144).
            --
            -- A client writes an optimistic row keyed by its own client_msg_id when it sends,
            -- then re-keys it to the durable rowid when the WS user-echo arrives. MISS that
            -- echo -- a socket drop between the send returning 200 and the echo, or a client
            -- restart in that window -- and nothing ever retires the optimistic row: reconcile
            -- inserts the confirmed row under str(row_id), a DIFFERENT primary key, so both
            -- persist and the duplicate renders forever (row_id IS NULL sorts last). This table
            -- is the mapping that lets reconcile do the same retirement deterministically.
            --
            -- Content-matching is not an alternative and was retired on purpose: two identical
            -- sends are legitimately two messages, so matching on text would merge them.
            --
            -- Kept OUT of lcm_messages deliberately. client_msg_id is a transport correlation
            -- token, not conversation content -- lcm_messages rows are serialized into
            -- content_json, returned by every history read, and rendered into model context.
            -- A wire token does not belong in a prompt. Side table, same reasoning as
            -- session_titles (a rename never rewrites history).
            CREATE TABLE IF NOT EXISTS message_client_ids (
                session_id    TEXT NOT NULL,
                row_id        INTEGER NOT NULL,
                client_msg_id TEXT NOT NULL,
                created_at    REAL NOT NULL,
                PRIMARY KEY (session_id, row_id)
            );

            CREATE INDEX IF NOT EXISTS idx_message_client_ids_lookup
                ON message_client_ids (session_id, client_msg_id);

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
            -- Edit/branch (B4). A fork is a NEW SESSION holding copies of history up to a
            -- point, so every consumer stays linear; this table is the only thing that
            -- remembers the relationship. Keyed by the CHILD: a session has at most one
            -- origin, while an origin may be forked many times.
            CREATE TABLE IF NOT EXISTS session_forks (
                session_id     TEXT PRIMARY KEY,
                origin_session TEXT NOT NULL,
                origin_rowid   INTEGER NOT NULL,
                created_at     REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_profiles (
                session_id TEXT PRIMARY KEY,
                profile    TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_workspaces (
                session_id TEXT PRIMARY KEY,
                path       TEXT NOT NULL,
                set_by     TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_backends (
                session_id TEXT PRIMARY KEY,
                key        TEXT NOT NULL,
                set_by     TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        self._conn.commit()
        self._migrate_add_content_json()
        self._migrate_add_trust_fields()

    @property
    def db_path(self) -> Path:
        """The SQLite file this store reads and writes."""
        return self._db_path

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

    def add_message(self, session_id: str, msg: MessagePart, *, append: bool = False) -> str:
        """Insert a message, forcing ``msg.session_id = session_id``.

        Thin adapter that closes the contract gap between ``LCMEngine``
        (which constructs ``MessagePart`` then passes session_id as a
        separate arg) and the internal ``insert_message`` (which derives
        session_id from the MessagePart). Overwrites unconditionally so
        the caller's argument always wins, matching the long-standing
        test-shim behaviour before this method landed in the class.
        ``append`` is passed through to :meth:`insert_message`.

        Prefer :meth:`insert_message` for internal callers that already
        construct the MessagePart with session_id set.
        """
        msg.session_id = session_id
        return self.insert_message(msg, append=append)

    def insert_message(self, msg: MessagePart, *, append: bool = False) -> str:
        """Insert a message and update the FTS5 index. Returns the message id.

        ``msg.turn_index`` is the row's prompt position, unique within its
        session. With ``append=True`` the store ignores it and numbers the row
        itself: the next free index, taken atomically. That is for callers that
        have no prompt position of their own, such as the CLI REPL. Either way,
        ``msg.turn_index`` and ``msg.row_id`` come back set to what was stored.

        This never deletes or overwrites a stored row:

        * an ``id`` that is already stored is left exactly as it is: the insert
          is a no-op and the FTS index is not touched;
        * a ``turn_index`` another message of the session already holds
          (refused by the guard trigger on a migrated DB) is reassigned to the
          next free index and retried once. The collision is recorded as a
          silent failure (subsystem ``lcm``, operation ``turn_index_collision``),
          so a new producer of duplicate indices shows up loudly instead of
          costing a message.

        Token shapes in ``content`` and ``content_json`` are redacted before the
        row, and its full-text entry, exist (X.37). ``msg`` itself is not
        touched: the live conversation still holds what the user sent, so a
        token pasted for the agent to use keeps working for the rest of the
        session. Only what is kept, and read back later, is redacted.
        """
        mid = msg.message_id or uuid4().hex
        ts = msg.timestamp or time.time()
        requested = msg.turn_index
        content = redact_secrets(msg.content)
        content_json = redact_json_text(msg.content_json)

        # Trust columns are written EXPLICITLY from the MessagePart — never left
        # to the column DEFAULT — so a (task_supervisor, False) turn can never be
        # silently up-tagged to the trusted default on insert.
        values = (
            msg.role, content, content_json, msg.token_count, ts,
            msg.provenance, 1 if msg.is_trusted else 0,
        )
        collision: sqlite3.IntegrityError | None = None
        inserted = False
        if append:
            cur = self._conn.execute(
                _INSERT_APPENDING_SQL, (mid, msg.session_id, *values, msg.session_id)
            )
            inserted = cur.rowcount == 1
        else:
            try:
                cur = self._conn.execute(
                    _INSERT_SQL, (mid, msg.session_id, msg.turn_index, *values)
                )
                inserted = cur.rowcount == 1
            except sqlite3.IntegrityError as exc:
                if not _is_turn_key_conflict(exc):
                    raise
                # The guard fires BEFORE the id conflict is looked at, so a re-insert
                # of a stored id can trip it too. That re-insert is the no-op
                # ON CONFLICT(id) would have made it, not a collision.
                if self._conn.execute(
                    "SELECT 1 FROM lcm_messages WHERE id = ?", (mid,)
                ).fetchone() is None:
                    collision = exc
                    cur = self._conn.execute(
                        _INSERT_APPENDING_SQL,
                        (mid, msg.session_id, *values, msg.session_id),
                    )
                    inserted = cur.rowcount == 1

        stored = self._conn.execute(
            "SELECT rowid, turn_index FROM lcm_messages WHERE id = ?", (mid,)
        ).fetchone()
        # Surface the durable rowid back to the caller (canonical wire message id),
        # and the index the row really holds.
        msg.row_id = int(stored[0])
        msg.turn_index = int(stored[1])
        if inserted:
            # Sync FTS index — use the rowid of the just-inserted row.
            self._conn.execute(
                "INSERT OR REPLACE INTO lcm_messages_fts (rowid, content) VALUES (?, ?)",
                (msg.row_id, content),
            )
        self._conn.commit()
        if collision is not None:
            self._record_turn_index_collision(
                msg.session_id, requested, msg.turn_index, collision
            )
        return mid

    def _record_turn_index_collision(
        self, session_id: str, requested: int, stored: int, exc: sqlite3.IntegrityError
    ) -> None:
        """Make a turn-index collision LOUD. The row is safe (it was stored at
        the next free index), but a collision means some path started numbering
        below the session's durable maximum again, which is exactly what the
        numbering anchor exists to prevent."""
        log.warning(
            "LCM turn_index collision in session %s: index %d is already taken, "
            "stored the row at %d instead",
            session_id, requested, stored,
        )
        try:
            from prometheus.telemetry.tracker import get_telemetry_handle

            tel = get_telemetry_handle()
            if tel is not None:
                tel.record_silent_failure(
                    subsystem="lcm",
                    operation="turn_index_collision",
                    exc=exc,
                    context={
                        "session_id": session_id,
                        "requested_turn_index": requested,
                        "stored_turn_index": stored,
                    },
                )
        except Exception:
            log.warning(
                "telemetry unavailable to record an LCM turn_index collision",
                exc_info=True,
            )

    def next_turn_index(self, session_id: str) -> int:
        """The lowest ``turn_index`` above every row the session holds (0 if none).

        Where a session's numbering must continue after anything that restarts
        it: a new process, ``/reset``, a rollback of a durable row, a rehydrate.
        One indexed seek on ``(session_id, turn_index)``.
        """
        row = self._conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM lcm_messages"
            " WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row[0])

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_messages(
        self,
        session_id: str,
        *,
        limit: int = 500,
    ) -> list[MessagePart]:
        """Return messages for a session in prompt order: ``(turn_index, rowid)`` ascending.

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
            " ORDER BY turn_index ASC, rowid ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [self._row_to_message(r) for r in rows]

    def get_fresh_tail(self, session_id: str, count: int) -> list[MessagePart]:
        """Return the last *count* uncompacted messages for a session.

        Results are in prompt order, oldest first (ascending ``(turn_index,
        rowid)``), so they can be appended directly to a prompt.
        """
        rows = self._conn.execute(
            "SELECT * FROM lcm_messages"
            " WHERE session_id = ? AND compacted = 0"
            " ORDER BY turn_index DESC, rowid DESC LIMIT ?",
            (session_id, count),
        ).fetchall()
        # Reverse so the caller gets chronological order.
        return [self._row_to_message(r) for r in reversed(rows)]

    def messages_before(
        self,
        session_id: str,
        timestamp: float,
        *,
        limit: int,
    ) -> list[MessagePart]:
        """The newest *limit* messages persisted strictly before *timestamp*.

        The golden-trace exporter's read: a tool call's input half is the
        conversation just BEFORE the call. :meth:`get_messages` cannot give
        that — it returns a session's LOWEST ``turn_index`` values, so every
        call after a session's 500th row was paired with the rows around 500.

        Selected newest-first by ``(turn_index, rowid)`` so ``limit`` keeps the
        rows nearest the call, then returned ascending like every other read.
        ``turn_index`` is the prompt position and rowid only breaks ties
        between rows that share one. Not ``ORDER BY rowid``: that is PERSIST
        order, and a message sent mid-turn is persisted before the turn's
        tail although the model saw it after.

        ``timestamp`` is compared with each row's PERSIST time, which for most
        of a turn's rows is when the turn ends, and strictly: a row at the
        call's own time can hold the tool's result. Compacted rows are
        included, as in :meth:`get_messages`.
        """
        rows = self._conn.execute(
            "SELECT * FROM lcm_messages"
            " WHERE session_id = ? AND timestamp < ?"
            " ORDER BY turn_index DESC, rowid DESC LIMIT ?",
            (session_id, timestamp, limit),
        ).fetchall()
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
        """All messages for a session in prompt order (``(turn_index, rowid)`` ASC), no limit.

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
            "ORDER BY turn_index ASC, rowid ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_message(r) for r in rows]

    def get_uncompacted_messages(self, session_id: str) -> list[MessagePart]:
        """All uncompacted messages for a session, ``(turn_index, rowid)`` ASC, no limit.

        Used by :class:`LCMCompactor` (to decide which messages to fold
        into the next summary) and :class:`LCMAssembler` (fresh-tail
        candidates). Differs from :meth:`get_fresh_tail` in that
        ``get_fresh_tail`` caps at ``count`` from the newest end; this
        returns every uncompacted row in chronological order.
        """
        rows = self._conn.execute(
            "SELECT * FROM lcm_messages WHERE session_id = ? AND compacted = 0 "
            "ORDER BY turn_index ASC, rowid ASC",
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
        ``rowid`` ASC (insertion order, monotonic and unique). Insertion order is not
        always prompt order: a message sent in the middle of a turn is persisted at
        once, ahead of the tail its turn writes when it ends, and ``turn_index``
        (unique per session, the prompt position) is what puts it back in place.

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

    def get_by_id(self, message_id: str) -> MessagePart | None:
        """Return one persisted message by its UUID, or ``None`` if absent.

        The id-keyed read the LCM tools need to expand a summary node back to
        its source messages: ``SummaryNode.source_message_ids`` holds these
        UUIDs, and ``lcm_expand`` / ``lcm_expand_query`` resolve each one here.

        Both tools previously called this method when it did not exist, behind
        a ``hasattr`` guard that turned the contract break into a plausible
        "not found" — so every expansion reported the history as gone and the
        model concluded it had no memory. ``has_message`` answers the same
        question as a bool; this returns the row.
        """
        if not message_id:
            return None
        row = self._conn.execute(
            "SELECT * FROM lcm_messages WHERE id = ? LIMIT 1",
            (message_id,),
        ).fetchone()
        return self._row_to_message(row) if row is not None else None

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

    def store_health(self) -> dict[str, int | float]:
        """How much of the durable store is hidden — the figure nobody could see.

        A tombstone hides a session forever without removing it, so the store silently filled with
        conversations no surface will ever show again: it reached 92% before anyone counted, and
        the counting was a person deciding to. That is the definition of a number that belongs in
        a health endpoint rather than in an audit.

        ``hidden`` uses the same predicate ``list_sessions`` does — tombstoned AND no activity
        newer than the tombstone — so this reports exactly what the listing suppresses.

        COST: the hidden count groups over every message row, ~3 ms at 10k messages on the current
        deployment. It scales with MESSAGES, not sessions, so if it ever shows up in a profile the
        fix is a cached watermark, not a cheaper join.
        """
        from prometheus.memory.session_kind import is_machine_session

        sessions = self._conn.execute(
            "SELECT count(DISTINCT session_id) FROM lcm_messages"
        ).fetchone()[0]
        messages = self._conn.execute("SELECT count(*) FROM lcm_messages").fetchone()[0]
        hidden_rows = self._conn.execute(
            """
            SELECT s.session_id AS sid
            FROM (SELECT session_id, MAX(timestamp) AS lt FROM lcm_messages GROUP BY session_id) s
            JOIN session_tombstones t ON t.session_id = s.session_id
            WHERE s.lt <= t.deleted_at
            """
        ).fetchall()
        hidden = len(hidden_rows)
        machine_hidden = sum(1 for r in hidden_rows if is_machine_session(r["sid"]))
        return {
            "sessions": sessions,
            "messages": messages,
            "hidden": hidden,
            # The headline. Rounded to one place because "76.4%" implies a precision that a
            # count of conversations does not have.
            "hidden_pct": round(100.0 * hidden / sessions, 1) if sessions else 0.0,
            # Of the hidden, how many named themselves disposable. A LOW number here with a high
            # hidden_pct is the actionable shape: probes are accumulating under conversation ids,
            # so retention will hold them for the long window (see memory/retention.py).
            "hidden_machine": machine_hidden,
        }

    def purge_session(self, session_id: str) -> dict[str, int]:
        """IRREVERSIBLY remove a session's content. The opposite of a tombstone.

        tombstone_session() hides a session; this deletes it. Both exist because
        they answer different questions — "take this out of my list" and "this
        must not be on the disk any more" — and only one of them can be given as
        an answer to someone who pasted a secret into a chat.

        Three things make this more than a DELETE:

        1. ``lcm_messages_fts`` is an EXTERNAL-CONTENT fts5 table with no
           triggers, so it does not follow the base table. Dropping the row
           alone leaves the words in the index while search stops returning them
           (its JOIN drops the orphan) — invisible AND still there, the worst of
           both. The index must be told, with the original text, BEFORE the row
           goes.
        2. Summaries are derived content: a purged conversation whose summary
           still quotes it has not been purged. They live in the same database
           file, with their own external-content index and the same problem.
        3. SQLite leaves deleted bytes in freed pages until they are reused.
           ``secure_delete`` overwrites them instead, which is the difference
           between "unreachable through the API" and "not in the file".

        Returns per-table row counts, because a purge that reports only "ok" is
        indistinguishable from a purge that matched nothing.

        Forks are NOT followed: a branch holds its own copies (fork_session
        inserts new rows), so purging an origin leaves them standing. That is
        defensible — they are separate conversations — but astonishing if
        unsaid, so the route reports them via the existing
        :meth:`list_session_forks`.
        """
        counts: dict[str, int] = {}
        prior = self._conn.execute("PRAGMA secure_delete").fetchone()[0]
        self._conn.execute("PRAGMA secure_delete = ON")
        try:
            # 1. the messages, and their index entries (index first — the delete
            #    command needs the text that is about to be destroyed).
            rows = self._conn.execute(
                "SELECT rowid AS rid, content FROM lcm_messages WHERE session_id = ?",
                (session_id,),
            ).fetchall()
            for r in rows:
                self._conn.execute(
                    "INSERT INTO lcm_messages_fts (lcm_messages_fts, rowid, content)"
                    " VALUES ('delete', ?, ?)",
                    (r["rid"], r["content"]),
                )
            counts["messages"] = self._conn.execute(
                "DELETE FROM lcm_messages WHERE session_id = ?", (session_id,)
            ).rowcount

            # 2. summaries — same file, same external-content trap.
            if self._table_exists("lcm_summaries"):
                srows = self._conn.execute(
                    "SELECT rowid AS rid, summary_text FROM lcm_summaries WHERE session_id = ?",
                    (session_id,),
                ).fetchall()
                if self._table_exists("lcm_summaries_fts"):
                    for r in srows:
                        self._conn.execute(
                            "INSERT INTO lcm_summaries_fts (lcm_summaries_fts, rowid, summary_text)"
                            " VALUES ('delete', ?, ?)",
                            (r["rid"], r["summary_text"]),
                        )
                counts["summaries"] = self._conn.execute(
                    "DELETE FROM lcm_summaries WHERE session_id = ?", (session_id,)
                ).rowcount
            else:
                counts["summaries"] = 0

            # 3. the per-session metadata. Enumerated rather than looped over a
            #    list so a NEW side table is a compile-time-visible omission
            #    here, not a silent survivor.
            for table in ("session_titles", "session_profiles", "session_pins",
                          "session_tombstones", "session_forks", "session_workspaces",
                          "session_backends", "message_client_ids"):
                counts[table] = self._conn.execute(
                    f"DELETE FROM {table} WHERE session_id = ?", (session_id,)
                ).rowcount if self._table_exists(table) else 0

            # fts5's 'delete' unlinks a document but leaves the TERM in the
            # index b-tree with an empty doclist, and VACUUM preserves that —
            # measured: the token was still readable in the file afterwards.
            # 'optimize' merges the segments and drops it for real. It costs a
            # pass over the index, which is the right trade for an explicit,
            # rare, irreversible operation.
            self._conn.execute(
                "INSERT INTO lcm_messages_fts (lcm_messages_fts) VALUES ('optimize')"
            )
            if self._table_exists("lcm_summaries_fts"):
                self._conn.execute(
                    "INSERT INTO lcm_summaries_fts (lcm_summaries_fts) VALUES ('optimize')"
                )
            self._conn.commit()
        finally:
            self._conn.execute(f"PRAGMA secure_delete = {'ON' if prior else 'OFF'}")
        # Reclaim the freed pages. Without this the overwritten space is still
        # inside the file; with it, the file itself shrinks.
        #
        # The WAL checkpoint is not optional housekeeping: in WAL mode the rows
        # live in the -wal sidecar until it is checkpointed, so a purge that
        # only VACUUMs the main file leaves the purged text sitting in a file
        # next to it. TRUNCATE empties the sidecar rather than merely folding it
        # in. (Found by the test's PRECONDITION, which could not find a freshly
        # written secret in the .db file at all.)
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._conn.execute("VACUUM")
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return counts

    def _table_exists(self, name: str) -> bool:
        return bool(
            self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?", (name,)
            ).fetchone()
        )

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
        ``get_session_title`` stays a clean present/absent signal. A generated
        title comes from the first exchange, so token shapes are redacted.
        """
        clean = redact_secrets((title or "").strip())
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

    def fork_session(self, origin_session: str, at_rowid: int, new_session_id: str) -> dict:
        """Copy `origin_session`'s history up to and including `at_rowid` into a new session.

        The copies get FRESH uuids and therefore fresh rowids, which is what keeps every other
        consumer linear: the original's rows are untouched, its `?since=` cursors do not move,
        and its summaries stay bound to its own message ids (summaries reference uuids, so the
        fork simply starts without any and compacts on its own schedule).

        `turn_index`, `timestamp`, `provenance` and `is_trusted` are copied verbatim — a fork is
        the same conversation up to the branch point, and rewriting its timestamps would make it
        look like it happened now. `compacted` resets to 0 because insert_message hardcodes it;
        the fork therefore carries fuller raw context until it compacts itself.

        Returns the provenance record. Raises ValueError if the origin has nothing at or before
        `at_rowid` — forking from a point that does not exist is a caller error, not an empty fork.
        """
        rows = self._conn.execute(
            "SELECT rowid AS row_id, id, turn_index, role, content, content_json,"
            " token_count, timestamp, provenance, is_trusted"
            " FROM lcm_messages WHERE session_id = ? AND rowid <= ?"
            " ORDER BY rowid",
            (origin_session, int(at_rowid)),
        ).fetchall()
        if not rows:
            raise ValueError(
                f"no messages in {origin_session!r} at or before rowid {at_rowid}"
            )

        from prometheus.memory.lcm_types import MessagePart

        for r in rows:
            self.insert_message(MessagePart(
                session_id=new_session_id,
                turn_index=r["turn_index"],
                role=r["role"],
                content=r["content"],
                content_json=r["content_json"],
                token_count=r["token_count"],
                timestamp=r["timestamp"],
                provenance=r["provenance"],
                is_trusted=bool(r["is_trusted"]),
            ))

        created = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO session_forks"
            " (session_id, origin_session, origin_rowid, created_at) VALUES (?, ?, ?, ?)",
            (new_session_id, origin_session, int(at_rowid), created),
        )
        self._conn.commit()
        return {
            "session_id": new_session_id,
            "origin_session": origin_session,
            "origin_rowid": int(at_rowid),
            "copied": len(rows),
            "created_at": created,
        }

    def get_session_fork(self, session_id: str) -> dict | None:
        """Where this session came from, or None if it was not forked."""
        row = self._conn.execute(
            "SELECT origin_session, origin_rowid, created_at FROM session_forks"
            " WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "origin_session": row["origin_session"],
            "origin_rowid": row["origin_rowid"],
            "created_at": row["created_at"],
        }

    def list_session_forks(self, origin_session: str) -> list[dict]:
        """Every fork taken FROM this session, oldest first — the other direction of the link."""
        rows = self._conn.execute(
            "SELECT session_id, origin_rowid, created_at FROM session_forks"
            " WHERE origin_session = ? ORDER BY created_at", (origin_session,)
        ).fetchall()
        return [{"session_id": r["session_id"], "origin_rowid": r["origin_rowid"],
                 "created_at": r["created_at"]} for r in rows]

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

    def set_session_workspace(self, session_id: str, path: str, *, set_by: str = "rest") -> None:
        """Bind a session to a working directory. Blank CLEARS the binding.

        Item W (2026-09-01): the conversation's cwd — where relative paths
        resolve, where project instruction files are discovered, and (the
        gate follows the session) the write boundary for that conversation.
        Same clear-on-blank rule as :meth:`set_session_profile`: absence means
        "follow the daemon's own cwd and the global workspace roots".

        The PATH is not validated here — the store does not know the
        filesystem or the denied list. Validation belongs at the write
        surfaces (the REST route, the slash command), where a bad path is
        refused with a reason. ``set_by`` records which surface bound it.
        """
        clean = (path or "").strip()
        if not clean:
            self._conn.execute(
                "DELETE FROM session_workspaces WHERE session_id = ?", (session_id,)
            )
        else:
            self._conn.execute(
                "INSERT OR REPLACE INTO session_workspaces (session_id, path, set_by, updated_at)"
                " VALUES (?, ?, ?, ?)",
                (session_id, clean, set_by, time.time()),
            )
        self._conn.commit()

    def set_session_backend(self, session_id: str, key: str | None, *, set_by: str = "router") -> None:
        """Remember which local backend a session is pointed at (`/4090`,
        `mini:qwen2.5:7b-instruct`). None/blank CLEARS. Same shape and rules as
        the workspace binding: absence means "the daemon's primary". Cloud
        overrides are never stored here — they were always RAM-only and stay so.
        """
        clean = (key or "").strip()
        if not clean:
            self._conn.execute("DELETE FROM session_backends WHERE session_id = ?", (session_id,))
        else:
            self._conn.execute(
                "INSERT OR REPLACE INTO session_backends (session_id, key, set_by, updated_at)"
                " VALUES (?, ?, ?, ?)",
                (session_id, clean, set_by, time.time()),
            )
        self._conn.commit()

    def get_session_backend(self, session_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT key FROM session_backends WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row[0] if row else None

    def all_session_backends(self) -> dict[str, str]:
        """Every remembered binding — what boot restore walks."""
        return {
            sid: key for sid, key in self._conn.execute(
                "SELECT session_id, key FROM session_backends"
            ).fetchall()
        }

    def get_session_workspace(self, session_id: str) -> str | None:
        """The path bound to *session_id*, or None (= follow the daemon)."""
        row = self._conn.execute(
            "SELECT path FROM session_workspaces WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row[0] if row else None

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
    # Optimistic-row correlation (audit P9.6 / Beacon#144)
    # ------------------------------------------------------------------

    def set_message_client_id(
        self, session_id: str, row_id: int | None, client_msg_id: str | None
    ) -> None:
        """Record the correlation id a client sent with the message now durable at ``row_id``.

        Called by the WS send path immediately after the user turn is persisted, where both
        halves are already in hand. Best-effort BY CONTRACT: a failure here costs the ability to
        retire a stranded optimistic row after a missed echo, and must never cost the turn
        itself -- so it no-ops on a blank id or a missing rowid rather than raising.

        INSERT OR REPLACE, not INSERT OR IGNORE: a rowid is durable and unique, so a second write
        for the same rowid is a correction of the first, and keeping the stale id would strand
        the row that actually correlates to it.
        """
        cid = (client_msg_id or "").strip()
        if not cid or row_id is None:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO message_client_ids"
            " (session_id, row_id, client_msg_id, created_at) VALUES (?, ?, ?, ?)",
            (session_id, int(row_id), cid, time.time()),
        )
        self._conn.commit()

    def get_message_client_ids(self, session_id: str) -> dict[int, str]:
        """``row_id -> client_msg_id`` for a session, so a history read can return them.

        ONE query for the whole session rather than one per row: history fetches are paginated
        and a per-message lookup would turn a single read into N.
        """
        rows = self._conn.execute(
            "SELECT row_id, client_msg_id FROM message_client_ids WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        return {int(r["row_id"]): r["client_msg_id"] for r in rows}

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
