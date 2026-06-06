"""SQLite store for the Kanban board (projects + stories).

Backs the Beacon Desktop Board view via the daemon's ``/api/projects`` and
``/api/stories`` routes. Single-operator, low-traffic; one shared connection
(``check_same_thread=False``) mirrors the LCM store. Fields are returned
snake_case (as the daemon's other endpoints do); ``labels`` is a parsed list and
timestamps are epoch seconds.

Model mirrors the Beacon web app's Drizzle schema (projects / stories) so a
future migration can point the web client at this daemon for shared state.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from prometheus.config.paths import get_kanban_db_path

STORY_STATUSES = {"todo", "in_progress", "blocked", "done"}
STORY_PRIORITIES = {"low", "medium", "high"}

# Columns a client may set on create/update (everything else is server-managed:
# id, session_key, dispatched_at, created_at, updated_at).
_PROJECT_FIELDS = ("name", "description", "color")
_STORY_FIELDS = (
    "project_id",
    "story_id",
    "title",
    "description",
    "status",
    "priority",
    "assigned_agent",
    "blocked_reason",
    "labels",
    "position",
)


class KanbanStore:
    """Projects + stories persisted in one SQLite file."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path if db_path is not None else get_kanban_db_path()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._apply_schema()

    def _apply_schema(self) -> None:
        self._conn.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS projects (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                description TEXT,
                color       TEXT NOT NULL DEFAULT '#58a6ff',
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS stories (
                id             TEXT PRIMARY KEY,
                project_id     TEXT,
                story_id       TEXT NOT NULL,
                title          TEXT NOT NULL,
                description    TEXT,
                status         TEXT NOT NULL DEFAULT 'todo',
                priority       TEXT NOT NULL DEFAULT 'medium',
                assigned_agent TEXT,
                blocked_reason TEXT,
                labels         TEXT NOT NULL DEFAULT '[]',
                position       REAL NOT NULL DEFAULT 0,
                session_key    TEXT,
                dispatched_at  REAL,
                created_at     REAL NOT NULL,
                updated_at     REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_stories_project
                ON stories (project_id, position);
            """
        )
        self._conn.commit()

    # ── Projects ───────────────────────────────────────────────────

    def list_projects(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM projects ORDER BY created_at ASC"
        ).fetchall()
        return [_project_row(r) for r in rows]

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        r = self._conn.execute(
            "SELECT * FROM projects WHERE id=?", (project_id,)
        ).fetchone()
        return _project_row(r) if r else None

    def create_project(
        self,
        *,
        name: str,
        description: str | None = None,
        color: str = "#58a6ff",
    ) -> dict[str, Any]:
        now = time.time()
        pid = uuid4().hex
        self._conn.execute(
            "INSERT INTO projects (id, name, description, color, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?)",
            (pid, name, description, color, now, now),
        )
        self._conn.commit()
        return self.get_project(pid)  # type: ignore[return-value]

    def update_project(self, project_id: str, **fields: Any) -> dict[str, Any] | None:
        if self.get_project(project_id) is None:
            return None
        sets = {k: fields[k] for k in _PROJECT_FIELDS if k in fields}
        if sets:
            cols = ", ".join(f"{k}=?" for k in sets)
            self._conn.execute(
                f"UPDATE projects SET {cols}, updated_at=? WHERE id=?",
                (*sets.values(), time.time(), project_id),
            )
            self._conn.commit()
        return self.get_project(project_id)

    def delete_project(self, project_id: str) -> bool:
        if self.get_project(project_id) is None:
            return False
        # Orphan its stories (keep them, drop the dangling link) rather than cascade-delete.
        self._conn.execute(
            "UPDATE stories SET project_id=NULL WHERE project_id=?", (project_id,)
        )
        self._conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
        self._conn.commit()
        return True

    # ── Stories ────────────────────────────────────────────────────

    def list_stories(self, *, project_id: str | None = None) -> list[dict[str, Any]]:
        if project_id:
            rows = self._conn.execute(
                "SELECT * FROM stories WHERE project_id=? ORDER BY position ASC",
                (project_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM stories ORDER BY position ASC"
            ).fetchall()
        return [_story_row(r) for r in rows]

    def get_story(self, story_pk: str) -> dict[str, Any] | None:
        r = self._conn.execute(
            "SELECT * FROM stories WHERE id=?", (story_pk,)
        ).fetchone()
        return _story_row(r) if r else None

    def create_story(
        self,
        *,
        story_id: str,
        title: str,
        project_id: str | None = None,
        description: str | None = None,
        status: str = "todo",
        priority: str = "medium",
        assigned_agent: str | None = None,
        labels: list[str] | None = None,
        position: float = 0.0,
    ) -> dict[str, Any]:
        now = time.time()
        sid = uuid4().hex
        self._conn.execute(
            """INSERT INTO stories
               (id, project_id, story_id, title, description, status, priority,
                assigned_agent, blocked_reason, labels, position, session_key,
                dispatched_at, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                sid,
                project_id,
                story_id,
                title,
                description,
                status,
                priority,
                assigned_agent,
                None,
                json.dumps(labels or []),
                position,
                None,
                None,
                now,
                now,
            ),
        )
        self._conn.commit()
        return self.get_story(sid)  # type: ignore[return-value]

    def update_story(self, story_pk: str, **fields: Any) -> dict[str, Any] | None:
        if self.get_story(story_pk) is None:
            return None
        sets: dict[str, Any] = {}
        for k in _STORY_FIELDS:
            if k in fields:
                sets[k] = json.dumps(fields[k] or []) if k == "labels" else fields[k]
        if sets:
            cols = ", ".join(f"{k}=?" for k in sets)
            self._conn.execute(
                f"UPDATE stories SET {cols}, updated_at=? WHERE id=?",
                (*sets.values(), time.time(), story_pk),
            )
            self._conn.commit()
        return self.get_story(story_pk)

    def delete_story(self, story_pk: str) -> bool:
        if self.get_story(story_pk) is None:
            return False
        self._conn.execute("DELETE FROM stories WHERE id=?", (story_pk,))
        self._conn.commit()
        return True

    def reorder_stories(self, items: list[dict[str, Any]]) -> None:
        """Bulk position + status update (the drag-drop persistence)."""
        now = time.time()
        for item in items:
            sid = (item or {}).get("id")
            if not sid:
                continue
            self._conn.execute(
                "UPDATE stories SET position=?, status=?, updated_at=? WHERE id=?",
                (
                    float(item.get("position", 0)),
                    str(item.get("status", "todo")),
                    now,
                    sid,
                ),
            )
        self._conn.commit()

    def mark_dispatched(self, story_pk: str, session_key: str) -> dict[str, Any] | None:
        if self.get_story(story_pk) is None:
            return None
        now = time.time()
        self._conn.execute(
            "UPDATE stories SET session_key=?, status='in_progress', dispatched_at=?,"
            " updated_at=? WHERE id=?",
            (session_key, now, now, story_pk),
        )
        self._conn.commit()
        return self.get_story(story_pk)

    def undispatch(self, story_pk: str) -> dict[str, Any] | None:
        if self.get_story(story_pk) is None:
            return None
        self._conn.execute(
            "UPDATE stories SET session_key=NULL, dispatched_at=NULL, status='todo',"
            " updated_at=? WHERE id=?",
            (time.time(), story_pk),
        )
        self._conn.commit()
        return self.get_story(story_pk)


def _project_row(r: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": r["id"],
        "name": r["name"],
        "description": r["description"],
        "color": r["color"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    }


def _story_row(r: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": r["id"],
        "project_id": r["project_id"],
        "story_id": r["story_id"],
        "title": r["title"],
        "description": r["description"],
        "status": r["status"],
        "priority": r["priority"],
        "assigned_agent": r["assigned_agent"],
        "blocked_reason": r["blocked_reason"],
        "labels": json.loads(r["labels"] or "[]"),
        "position": r["position"],
        "session_key": r["session_key"],
        "dispatched_at": r["dispatched_at"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    }


_STORE: KanbanStore | None = None
_STORE_PATH: Path | None = None


def get_kanban_store() -> KanbanStore:
    """Module-singleton store, rebuilt when the resolved DB path changes.

    The path-change check keeps tests isolated: each monkeypatches
    ``PROMETHEUS_DATA_DIR`` to its own tmp dir, so the singleton is rebuilt
    against that test's database rather than leaking a previous one.
    """
    global _STORE, _STORE_PATH
    path = get_kanban_db_path()
    if _STORE is None or _STORE_PATH != path:
        _STORE = KanbanStore(path)
        _STORE_PATH = path
    return _STORE
