"""The chat projects store (docs/design/chat-projects.md, sections 1 to 3).

Tables in the conversation store, on its connection, beside ``session_workspaces``: filing a session changes its
membership and its workspace binding together, in one transaction, so the two can never disagree.

* ``chat_projects``: one row per project; the cleaned name is unique without regard to case, the folder and an import
  key are unique.
* ``session_chat_projects``: a session is in at most one project; presence is the filing.
* ``chat_projects_rev``: one counter, raised in the same transaction as every change to a project or a membership.

Filing binds the session's workspace to the project's folder with ``set_by = "project:<id>"``. A workspace set any
other way is never replaced unless the caller says so (:class:`WorkspaceSet`). Unfiling, moving and deleting clear
only a binding that same project set, so a session moved to another project as a label keeps (and, on the first
project's delete, loses) the first project's binding correctly. Deleting a project never touches the folder or a file in it.

Folder validation (``validate_workspace_path``) and the managed folder are the caller's: the store does not know the
filesystem or the security config.

Source: novel code for Prometheus, 2026-11-13.
"""

from __future__ import annotations

import re
import secrets
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

NAME_MAX = 60
#: A binding a project set is ``set_by = "project:<id>"``: the id says which, so only that project clears it.
SET_BY_PREFIX = "project:"


def _set_by(project_id: str) -> str:
    return SET_BY_PREFIX + project_id

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_projects (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    name_folded    TEXT NOT NULL UNIQUE,
    folder         TEXT NOT NULL UNIQUE,
    folder_managed INTEGER NOT NULL,
    import_key     TEXT UNIQUE,
    position       REAL NOT NULL,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS session_chat_projects (
    session_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    filed_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_chat_projects_project ON session_chat_projects (project_id);
CREATE TABLE IF NOT EXISTS chat_projects_rev (
    id  INTEGER PRIMARY KEY CHECK (id = 1),
    rev INTEGER NOT NULL
);
"""


def clean_name(raw: str) -> str:
    """Trimmed, inner whitespace collapsed, at most :data:`NAME_MAX` characters. ``""`` means no name."""
    return re.sub(r"\s+", " ", raw or "").strip()[:NAME_MAX].strip()


class ProjectError(Exception):
    kind = "error"


class NameTaken(ProjectError):
    kind = "name_taken"


class FolderInUse(ProjectError):
    kind = "folder_in_use"


class UnknownProject(ProjectError):
    kind = "unknown_project"


class WorkspaceSet(ProjectError):
    """The session's workspace was set another way; filing would replace it."""

    kind = "workspace_set"

    def __init__(self, workspace: str) -> None:
        super().__init__(f"the conversation already works in {workspace}")
        self.workspace = workspace


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    folder: str
    folder_managed: bool
    import_key: str | None
    created_at: float
    updated_at: float
    sessions: int = 0


@dataclass(frozen=True)
class FileResult:
    session_id: str
    project_id: str | None
    workspace: str | None


@dataclass(frozen=True)
class VisibleProject:
    """What a chat-only device may know of a project: its id, its name, and which of the device's sessions are in it."""

    id: str
    name: str
    own_session_ids: list[str] = field(default_factory=list)


class ChatProjectStore:
    """The store, on the conversation store's connection (``LCMConversationStore.chat_projects()``)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.executescript(SCHEMA)
        conn.execute("INSERT OR IGNORE INTO chat_projects_rev (id, rev) VALUES (1, 0)")
        conn.commit()

    # ---------------------------------------------------------------- reads
    def rev(self) -> int:
        row = self._conn.execute("SELECT rev FROM chat_projects_rev WHERE id = 1").fetchone()
        return int(row[0]) if row else 0

    def _rows(self, where: str = "", args: tuple = ()) -> list[Project]:
        sql = (
            "SELECT p.id, p.name, p.folder, p.folder_managed, p.import_key, p.created_at, p.updated_at,"
            " (SELECT COUNT(*) FROM session_chat_projects s WHERE s.project_id = p.id)"
            f" FROM chat_projects p {where} ORDER BY p.position, p.created_at"
        )
        return [Project(r[0], r[1], r[2], bool(r[3]), r[4], r[5], r[6], int(r[7]))
                for r in self._conn.execute(sql, args).fetchall()]

    def all(self) -> list[Project]:
        """Every project, in display order, with its session count."""
        return self._rows()

    def get(self, project_id: str) -> Project | None:
        rows = self._rows("WHERE p.id = ?", (project_id,))
        return rows[0] if rows else None

    def by_name(self, name: str) -> Project | None:
        folded = clean_name(name).lower()
        rows = self._rows("WHERE p.name_folded = ?", (folded,)) if folded else []
        return rows[0] if rows else None

    def by_folder(self, folder: str) -> Project | None:
        rows = self._rows("WHERE p.folder = ?", (folder,))
        return rows[0] if rows else None

    def by_import_key(self, key: str) -> Project | None:
        rows = self._rows("WHERE p.import_key = ?", (key,)) if key else []
        return rows[0] if rows else None

    def project_of(self, session_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT project_id FROM session_chat_projects WHERE session_id = ?", (session_id,)).fetchone()
        return row[0] if row else None

    def projects_of(self, session_ids: Iterable[str]) -> dict[str, str]:
        ids = list(session_ids)
        out: dict[str, str] = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            marks = ",".join("?" * len(chunk))
            out.update(self._conn.execute(
                f"SELECT session_id, project_id FROM session_chat_projects WHERE session_id IN ({marks})",
                chunk).fetchall())
        return out

    def session_ids(self, project_id: str) -> list[str]:
        return [r[0] for r in self._conn.execute(
            "SELECT session_id FROM session_chat_projects WHERE project_id = ? ORDER BY filed_at, session_id",
            (project_id,)).fetchall()]

    def visible_to(self, own_session_ids: set[str]) -> list[VisibleProject]:
        """The projects a chat-only device may see: those with at least one of *own_session_ids*, in display order."""
        if not own_session_ids:
            return []
        filed = self.projects_of(own_session_ids)
        by_project: dict[str, list[str]] = {}
        for sid, pid in filed.items():
            by_project.setdefault(pid, []).append(sid)
        return [VisibleProject(p.id, p.name, sorted(by_project[p.id])) for p in self.all() if p.id in by_project]

    # --------------------------------------------------------------- writes
    def _bump(self) -> None:
        self._conn.execute("UPDATE chat_projects_rev SET rev = rev + 1 WHERE id = 1")

    def insert(self, name: str, folder: str, *, managed: bool, import_key: str | None = None) -> Project:
        """A new project. :class:`NameTaken` / :class:`FolderInUse` when another project has the name or folder."""
        clean = clean_name(name)
        if not clean:
            raise ValueError("a project needs a name")
        now = time.time()
        pid = "cp_" + secrets.token_hex(6)
        try:
            with self._conn:
                position = self._conn.execute(
                    "SELECT COALESCE(MAX(position), 0) + 1 FROM chat_projects").fetchone()[0]
                self._conn.execute(
                    "INSERT INTO chat_projects (id, name, name_folded, folder, folder_managed, import_key, position,"
                    " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (pid, clean, clean.lower(), folder, int(managed), import_key or None, position, now, now))
                self._bump()
        except sqlite3.IntegrityError as exc:
            message = str(exc)
            if "name_folded" in message:
                raise NameTaken(f"a project is already called {clean!r}") from exc
            if "folder" in message:
                raise FolderInUse("another project already uses that folder") from exc
            raise
        project = self.get(pid)
        assert project is not None
        return project

    def rename(self, project_id: str, name: str) -> Project:
        clean = clean_name(name)
        if not clean:
            raise ValueError("a project needs a name")
        try:
            with self._conn:
                cur = self._conn.execute(
                    "UPDATE chat_projects SET name = ?, name_folded = ?, updated_at = ? WHERE id = ?",
                    (clean, clean.lower(), time.time(), project_id))
                if cur.rowcount == 0:
                    raise UnknownProject("no such project")
                self._bump()
        except sqlite3.IntegrityError as exc:
            raise NameTaken(f"a project is already called {clean!r}") from exc
        project = self.get(project_id)
        assert project is not None
        return project

    def delete(self, project_id: str) -> int:
        """Remove the project and its memberships; clear the bindings it set. Returns how many sessions were unfiled.

        Never touches the folder or any file in it.
        """
        with self._conn:
            if self._conn.execute("SELECT 1 FROM chat_projects WHERE id = ?", (project_id,)).fetchone() is None:
                raise UnknownProject("no such project")
            sessions = [r[0] for r in self._conn.execute(
                "SELECT session_id FROM session_chat_projects WHERE project_id = ?", (project_id,)).fetchall()]
            # Every binding this project set, on members and on sessions since moved elsewhere as a label.
            self._conn.execute("DELETE FROM session_workspaces WHERE set_by = ?", (_set_by(project_id),))
            self._conn.execute("DELETE FROM session_chat_projects WHERE project_id = ?", (project_id,))
            self._conn.execute("DELETE FROM chat_projects WHERE id = ?", (project_id,))
            self._bump()
        return len(sessions)

    def file(self, session_id: str, project_id: str | None, *, replace_workspace: bool = False,
             keep_workspace: bool = False) -> FileResult:
        """File *session_id* in *project_id* (``None`` unfiles), binding or clearing its workspace in the same step.

        *keep_workspace* files as a label only: the membership is recorded and the workspace is left as it is (what a
        migration sends, so moving local projects in never changes where an old conversation writes).
        """
        with self._conn:
            if project_id is None:
                old = self.project_of(session_id)
                self._conn.execute("DELETE FROM session_chat_projects WHERE session_id = ?", (session_id,))
                if old is not None:
                    self._clear_project_workspace(session_id, old)
                self._bump()
                left = self._conn.execute(
                    "SELECT path FROM session_workspaces WHERE session_id = ?", (session_id,)).fetchone()
                return FileResult(session_id, None, left[0] if left else None)
            row = self._conn.execute("SELECT folder FROM chat_projects WHERE id = ?", (project_id,)).fetchone()
            if row is None:
                raise UnknownProject("no such project")
            folder = row[0]
            current = self._conn.execute(
                "SELECT path, set_by FROM session_workspaces WHERE session_id = ?", (session_id,)).fetchone()
            bound_by_a_project = current is not None and str(current[1]).startswith(SET_BY_PREFIX)
            if (not keep_workspace and current is not None and not bound_by_a_project and current[0] != folder
                    and not replace_workspace):
                raise WorkspaceSet(current[0])
            now = time.time()
            self._conn.execute(
                "INSERT INTO session_chat_projects (session_id, project_id, filed_at) VALUES (?, ?, ?)"
                " ON CONFLICT(session_id) DO UPDATE SET project_id = excluded.project_id,"
                " filed_at = CASE WHEN session_chat_projects.project_id = excluded.project_id"
                " THEN session_chat_projects.filed_at ELSE excluded.filed_at END",
                (session_id, project_id, now))
            if keep_workspace:
                self._bump()
                return FileResult(session_id, project_id, current[0] if current is not None else None)
            self._conn.execute(
                "INSERT OR REPLACE INTO session_workspaces (session_id, path, set_by, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, folder, _set_by(project_id), now))
            self._bump()
        return FileResult(session_id, project_id, folder)

    def _clear_project_workspace(self, session_id: str, project_id: str) -> None:
        self._conn.execute(
            "DELETE FROM session_workspaces WHERE session_id = ? AND set_by = ?", (session_id, _set_by(project_id)))
