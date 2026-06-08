"""Durable SQLite store for managed tasks.

The legacy :class:`~prometheus.tasks.manager.BackgroundTaskManager` kept tasks
in an in-memory ``dict`` plus per-task ``.log`` files, so all task state was lost
on a daemon restart. This store persists every :class:`TaskRecord` to
``~/.prometheus/data/tasks.db`` so the supervisor can resume (file_watch/poll) or
reap (process) ``running`` tasks across restarts — no zombie rows.

Single shared connection (``check_same_thread=False``), WAL, ``sqlite3.Row`` —
mirrors the Kanban + LCM stores. Single-operator, low-traffic.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from prometheus.config.paths import get_tasks_db_path
from prometheus.tasks.types import TaskRecord, TaskStatus

# TaskRecord fields persisted as TEXT-JSON (everything else maps to a column).
_JSON_FIELDS = ("metadata", "spec")


class TaskStore:
    """Persist ``TaskRecord`` rows so tasks survive a daemon restart."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path if db_path is not None else get_tasks_db_path()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._apply_schema()

    def _apply_schema(self) -> None:
        self._conn.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS tasks (
                id              TEXT PRIMARY KEY,
                type            TEXT NOT NULL,
                status          TEXT NOT NULL,
                description     TEXT NOT NULL DEFAULT '',
                cwd             TEXT NOT NULL DEFAULT '',
                output_file     TEXT NOT NULL DEFAULT '',
                command         TEXT,
                prompt          TEXT,
                created_at      REAL NOT NULL DEFAULT 0,
                started_at      REAL,
                ended_at        REAL,
                return_code     INTEGER,
                metadata        TEXT NOT NULL DEFAULT '{}',
                session_id      TEXT,
                notify_target   TEXT,
                on_complete     TEXT NOT NULL DEFAULT 'notify',
                reengage_prompt TEXT,
                timeout_seconds INTEGER,
                artifact_path   TEXT,
                error           TEXT,
                spec            TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status);
            """
        )
        self._conn.commit()

    # ── CRUD ───────────────────────────────────────────────────────────

    def upsert(self, record: TaskRecord) -> None:
        """Durably write (insert or replace) a task record. Idempotent on id."""
        row = _record_to_row(record)
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row)
        self._conn.execute(
            f"INSERT OR REPLACE INTO tasks ({cols}) VALUES ({placeholders})",
            row,
        )
        self._conn.commit()

    def get(self, task_id: str) -> TaskRecord | None:
        r = self._conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return _row_to_record(r) if r is not None else None

    def list(self, *, status: TaskStatus | None = None) -> list[TaskRecord]:
        """Return all task records, newest first, optionally filtered by status."""
        if status is not None:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC"
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def delete(self, task_id: str) -> None:
        self._conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ── Row <-> TaskRecord serialization ───────────────────────────────────

# The column order of the schema, used to project a TaskRecord into a row dict.
_COLUMNS = (
    "id", "type", "status", "description", "cwd", "output_file", "command",
    "prompt", "created_at", "started_at", "ended_at", "return_code", "metadata",
    "session_id", "notify_target", "on_complete", "reengage_prompt",
    "timeout_seconds", "artifact_path", "error", "spec",
)


def _record_to_row(record: TaskRecord) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for col in _COLUMNS:
        if col == "output_file":
            row[col] = str(record.output_file)
        elif col in _JSON_FIELDS:
            row[col] = json.dumps(getattr(record, col) or {})
        else:
            row[col] = getattr(record, col)
    return row


def _row_to_record(row: sqlite3.Row) -> TaskRecord:
    data = dict(row)
    return TaskRecord(
        id=data["id"],
        type=data["type"],
        status=data["status"],
        description=data["description"],
        cwd=data["cwd"],
        output_file=Path(data["output_file"]),
        command=data["command"],
        prompt=data["prompt"],
        created_at=data["created_at"] or 0.0,
        started_at=data["started_at"],
        ended_at=data["ended_at"],
        return_code=data["return_code"],
        metadata=_loads(data["metadata"]),
        session_id=data["session_id"],
        notify_target=data["notify_target"],
        on_complete=data["on_complete"] or "notify",
        reengage_prompt=data["reengage_prompt"],
        timeout_seconds=data["timeout_seconds"],
        artifact_path=data["artifact_path"],
        error=data["error"],
        spec=_loads(data["spec"]),
    )


def _loads(blob: str | None) -> dict[str, Any]:
    if not blob:
        return {}
    try:
        val = json.loads(blob)
        return val if isinstance(val, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}
