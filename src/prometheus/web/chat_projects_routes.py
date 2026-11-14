"""``/api/chat-projects`` and filing a conversation in one (docs/design/chat-projects.md, sections 3.1-3.3, 4-6).

* ``GET /api/chat-projects`` and ``GET /api/chat-projects/{id}``: the owner sees every project in full; a chat-only
  device sees only the projects one of its own sessions is in, as id and name and its own sessions, and a project
  without one is ``404`` ("not yours" is "not there").
* ``POST /api/chat-projects``, ``PUT|DELETE /api/chat-projects/{id}``, ``PUT /api/sessions/{session_id}/project``:
  the owner's only. Default-deny in ``route_access`` refuses them to a chat-only device, and every handler checks
  again: a path that happens to match a chat-only pattern must not open an owner route.

Filing binds the conversation's workspace to the project's folder unless ``keep_workspace`` (a label only, what a
migration sends). A hand-set workspace is never replaced without ``replace_workspace``; a chat-only device's session is
bound only with ``device_session``; a machine session never; the session must exist; the folder is validated again and
must still resolve to the stored path.

Every change raises the store's ``rev`` and sends ``chat_project_changed`` to owner sockets only (ids and a change
kind, never a name or a path). Never in an answer to a chat-only device: a folder path, instructions, other projects,
other sessions, the ``rev``.

Source: novel code for Prometheus, 2026-11-13.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from prometheus.memory.session_kind import is_machine_session
from prometheus.projects.content import instructions_info, reference_names
from prometheus.projects.folders import make_managed_folder, projects_root
from prometheus.projects.suggest import KnownFolder, suggest, tally
from prometheus.projects.store import (
    FolderInUse,
    ImportKeyTaken,
    NameTaken,
    Project,
    UnknownProject,
    WorkspaceSet,
    clean_name,
)

logger = logging.getLogger(__name__)

_BODY_CAP = 64 * 1024
#: Suggestions are computed for at most this many unfiled conversations, the most recently active.
SUGGEST_SESSIONS = 200
_SUGGEST_CACHE_MAX = 2000
_IMPORT_KEY = re.compile(r"[\x21-\x7e]{1,200}")


def _err(status: int, kind: str, error: str, **extra: Any) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": error, "kind": kind, **extra})


async def _read_json(request: Request) -> Any:
    """The JSON body, read with a cap (a chunked body has no Content-Length for the middleware). None if unusable."""
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > _BODY_CAP:
            return None
    if not body:
        return {}
    try:
        return json.loads(body)
    except (ValueError, RecursionError):
        return None


def _ident(path: Path) -> tuple[int, int] | None:
    """The file's identity (device, inode), or None. Case and spelling do not matter to it (macOS is
    case-insensitive), nor does a symlink in the path."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_dev, st.st_ino


def _chain(path: Path) -> list[tuple[int, int]]:
    """The identities of *path* and every directory above it."""
    out = []
    for p in (path, *path.parents):
        ident = _ident(p)
        if ident is not None:
            out.append(ident)
    return out


def _no_store() -> JSONResponse:
    # 404, not 503: the design tells a client that 404 here means "this daemon has no projects".
    return _err(404, "unavailable", "this daemon has no projects (no durable store)")


class ChatProjectsService:
    """What the routes and ``POST /api/sessions`` share. On ``app.state.chat_projects``."""

    def __init__(self, app: Any, config: dict[str, Any], conversation_store: Callable[[], Any],
                 devices: Callable[[], Any], session_exists: Callable[[str], bool]) -> None:
        self._app = app
        self._config = config
        self._conversation_store = conversation_store
        self._devices = devices
        self._session_exists = session_exists
        # (session_id, its newest rowid, the projects rev) -> the suggestion, or None. An unchanged conversation is
        # not read again; a new message or any project change makes a new key.
        self._suggest_cache: dict[tuple[str, int, int], dict[str, Any] | None] = {}  # (sid, rowid, folders hash)

    def store(self) -> Any:
        conv = self._conversation_store()
        return conv.chat_projects() if conv is not None else None

    def _security(self) -> dict[str, Any]:
        sec = self._config.get("security")
        return sec if isinstance(sec, dict) else {}

    # ------------------------------------------------------------------ views
    @staticmethod
    def full(p: Project) -> dict[str, Any]:
        chars, rev = instructions_info(p.folder)
        return {"id": p.id, "name": p.name, "folder": p.folder, "folder_managed": p.folder_managed,
                "instructions_chars": chars, "instructions_rev": rev, "files": len(reference_names(p.folder)),
                "sessions": p.sessions, "import_key": p.import_key, "created_at": p.created_at,
                "updated_at": p.updated_at}

    # ---------------------------------------------------------------- folders
    @staticmethod
    def _agent_homes() -> list[Path]:
        from prometheus.config.paths import config_dir_path, data_dir_path

        homes = [Path(os.path.expanduser("~/.prometheus"))]
        for get in (config_dir_path, data_dir_path):
            try:
                homes.append(get())
            except Exception:  # noqa: BLE001 - a path that cannot be computed is simply not compared
                continue
        return homes

    def validate(self, raw: str) -> tuple[Path | None, str | None]:
        """A folder a project may use, resolved, or why not (design 3.1 rule 3). Compared by file identity, never by
        spelling: macOS is case-insensitive, and a symlinked agent home must not hide what it points at."""
        from prometheus.context.workspace import validate_workspace_path

        resolved, why = validate_workspace_path(raw, self._security())
        if resolved is None:
            return None, why
        chain = _chain(resolved)
        here = chain[0] if chain else None
        home = _ident(Path(os.path.expanduser("~")))
        if home is not None and (here == home or here in _chain(Path(os.path.expanduser("~")))[1:]):
            return None, "the home directory, or a folder that contains it, cannot be a project folder"
        for agent_home in self._agent_homes():
            ident = _ident(agent_home)
            if ident is None:
                continue
            if ident in chain or (here is not None and here in _chain(agent_home)):
                return None, "a folder inside or around the agent's own home cannot be a project folder"
        root = projects_root(self._config)
        root_ident = _ident(root)
        if root_ident is not None and (here == root_ident or (here is not None and here in _chain(root)[1:])):
            return None, "the projects root, or a folder that contains it, cannot be a project folder"
        store = self.store()
        for other in (store.all() if store is not None else []):
            o = Path(other.folder)
            oid = _ident(o)
            if oid is None:
                continue
            if oid == here:
                return None, f"the project {other.name!r} already uses that folder"
            if oid in chain or (here is not None and here in _chain(o)):
                return None, f"it is inside or around the folder of the project {other.name!r}"
        return resolved, None

    def bindable(self, p: Project) -> str | None:
        """Why *p*'s folder may not be bound now, or None: validated again, and it must still be the stored path."""
        from prometheus.context.workspace import validate_workspace_path

        if os.path.islink(p.folder):
            return "the project's folder is now a link"
        resolved, why = validate_workspace_path(p.folder, self._security())
        if resolved is None:
            return why
        if str(resolved) != p.folder:
            return "the project's folder no longer resolves to where it was"
        return None

    def make_managed(self, name: str) -> tuple[Path | None, str | None]:
        from prometheus.config.shipped_defaults import resolve_denied_paths
        from prometheus.context.workspace import validate_workspace_path
        from prometheus.security.path_guard import denying_entry

        root = projects_root(self._config)
        if not root.is_absolute():
            return None, "projects.root must be an absolute path (or start with ~)"
        # Checked BEFORE anything is made: the nearest folder that exists must be one a project could use, and no
        # denied entry may cover the root's own path, so a refused root leaves nothing behind.
        existing = next((p for p in (root, *root.parents) if p.exists()), None)
        if existing is None:
            return None, "the projects root has no existing parent"
        _ok, why = validate_workspace_path(str(existing), self._security())
        if _ok is None:
            return None, why
        entries = [os.path.expanduser(e) for e in resolve_denied_paths(self._security())]
        if denying_entry(str(existing.resolve() / root.relative_to(existing)), entries) is not None:
            return None, "the projects root is under security.denied_paths"
        store = self.store()
        projects = store.all() if store is not None else []
        root_chain_existing = _chain(existing.resolve())
        for other in projects:
            oid = _ident(Path(other.folder))
            if oid is not None and oid in root_chain_existing:
                return None, f"the projects root is inside the folder of the project {other.name!r}"
        stored = {other.folder for other in projects}
        try:
            made = make_managed_folder(root, name, skip=lambda c: str(c.resolve()) in stored or str(c) in stored)
        except (OSError, ValueError) as exc:
            return None, str(exc)
        resolved, why = self.validate(str(made))
        if resolved is None:
            _remove_empty(made)
            return None, why
        return resolved, None

    # ---------------------------------------------------------------- filing
    def precheck(self, project_id: str) -> JSONResponse | Project:
        store = self.store()
        if store is None:
            return _no_store()
        p = store.get(project_id)
        if p is None:
            return _err(404, "unknown_project", "no such project")
        why = self.bindable(p)
        if why is not None:
            return _err(400, "bad_folder", why)
        return p

    async def file(self, session_id: str, project_id: str | None, *, replace_workspace: bool = False,
                   keep_workspace: bool = False, device_session: bool = False) -> JSONResponse | dict[str, Any]:
        store = self.store()
        if store is None:
            return _no_store()
        if is_machine_session(session_id):
            return _err(400, "machine_session", "a machine session cannot be filed in a project")
        if not self._session_exists(session_id):
            return _err(404, "unknown_session", "unknown session")
        if project_id is not None:
            p = store.get(project_id)
            if p is None:
                return _err(404, "unknown_project", "no such project")
            if not keep_workspace:
                devices = self._devices()
                owner = devices.session_owner(session_id) if devices is not None else None
                if owner and owner not in devices.owner_device_ids() and not device_session:
                    return _err(409, "device_session", "this conversation belongs to a chat-only device; binding it "
                                "to the project's folder needs device_session: true")
                why = self.bindable(p)
                if why is not None:
                    return _err(400, "bad_folder", why)
        try:
            result = store.file(session_id, project_id, replace_workspace=replace_workspace,
                                keep_workspace=keep_workspace)
        except WorkspaceSet as exc:
            return _err(409, "workspace_set", str(exc), workspace=exc.workspace)
        except UnknownProject:
            return _err(404, "unknown_project", "no such project")
        if project_id is not None and not keep_workspace:
            logger.warning("session workspace SET for %s → %s (project %s) — this is now that conversation's write "
                           "boundary", session_id, result.workspace, project_id)
        await self.notify("membership", project_id, session_id)
        return {"ok": True, "session_id": session_id, "chat_project_id": project_id, "workspace": result.workspace}

    # ------------------------------------------------------------ suggestions
    def suggestions(self) -> dict[str, Any]:
        """The suggestion for each recent unfiled conversation (design 3.6). Owner only; the route checks."""
        conv = self._conversation_store()
        store = self.store()
        if conv is None or store is None:
            return {"suggestions": {}, "rev": 0}
        projects = store.all()
        known = [KnownFolder(p.id, p.name, p.folder) for p in projects]
        root = str(projects_root(self._config))
        rev = store.rev()
        # The cache key moves with what changes a suggestion (the projects' ids, names and folders), not with rev:
        # filing a session raises rev and changes no suggestion of another session.
        folders_key = hash(tuple((p.id, p.name, p.folder) for p in projects))
        devices = self._devices()
        owners = devices.owner_device_ids() if devices is not None else set()
        rows = sorted(conv.list_sessions(), key=lambda r: r.get("last_timestamp") or 0.0, reverse=True)
        filed = store.projects_of(r["session_id"] for r in rows)
        out: dict[str, Any] = {}
        considered = 0
        for row in rows:
            sid = row["session_id"]
            if sid in filed or is_machine_session(sid):
                continue
            try:
                owner = devices.session_owner(sid) if devices is not None else None
            except Exception:  # noqa: BLE001 - an owner that cannot be read: fail closed, offer nothing for it
                continue
            if owner and owner not in owners:
                continue  # a chat-only device's conversation: filing it takes the owner's explicit flag
            considered += 1
            if considered > SUGGEST_SESSIONS:
                break
            key = (sid, int(row.get("watermark") or 0), folders_key)
            if key not in self._suggest_cache:
                if len(self._suggest_cache) >= _SUGGEST_CACHE_MAX:
                    self._suggest_cache.clear()
                self._suggest_cache[key] = self._suggest_one(conv, sid, known, root, projects)
            hint = self._suggest_cache[key]
            if hint is not None:
                out[sid] = hint
        return {"suggestions": out, "rev": rev}

    @staticmethod
    def _suggest_one(conv: Any, sid: str, known: list[KnownFolder], root: str,
                     projects: list[Project]) -> dict[str, Any] | None:
        hint = suggest(tally(_tool_inputs(conv, sid), known=known, root=root))
        if hint is None:
            return None
        project = None
        by_id = {p.id: p for p in projects}
        if hint.key in by_id:
            project = by_id[hint.key]
        else:
            from prometheus.projects.store import fold_name

            from prometheus.projects.suggest import project_key_of_folder

            wanted = fold_name(hint.label)
            for p in projects:
                if project_key_of_folder(p.folder) == hint.key or fold_name(p.name) == wanted:
                    project = p
                    break
        return {"key": hint.key, "label": hint.label, "share": hint.share, "calls": hint.calls,
                "project": {"id": project.id, "name": project.name} if project is not None else None}

    async def notify(self, change: str, project_id: str | None, session_id: str | None = None) -> None:
        """``chat_project_changed`` to owner sockets only (ids and a change kind)."""
        bridge = getattr(self._app.state, "ws_bridge", None)
        send = getattr(bridge, "send_to_operators", None)
        store = self.store()
        if send is None or store is None:
            return
        try:
            await send({"type": "chat_project_changed", "payload": {
                "rev": store.rev(), "project_id": project_id, "change": change, "session_id": session_id}})
        except Exception:  # noqa: BLE001 - a frame is a hint; clients refetch on refresh
            logger.debug("chat_project_changed not sent", exc_info=True)


def _tool_inputs(conv: Any, session_id: str) -> list[str]:
    """Each tool call's input in *session_id*, as JSON text (the desktop's evidence: ``tool_use`` blocks on assistant
    rows). A row whose ``content_json`` is not usable is skipped."""
    out: list[str] = []
    for raw in conv.tool_use_json(session_id):
        try:
            blocks = json.loads(raw)
        except (ValueError, RecursionError, TypeError):
            continue
        if isinstance(blocks, dict):
            blocks = blocks.get("content")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use" and "input" in block:
                try:
                    out.append(json.dumps(block["input"], ensure_ascii=False))
                except (TypeError, ValueError, RecursionError):
                    continue
    return out


def _remove_empty(folder: Path) -> None:
    """Remove a folder this request made a moment ago, if it is still empty (never anything else)."""
    try:
        folder.rmdir()
    except OSError:
        pass


def register_chat_projects_routes(
    app: Any, *, config: dict[str, Any], conversation_store: Callable[[], Any], devices: Callable[[], Any],
    is_owner: Callable[[Request], bool], owned_ids: Callable[[Request], set[str] | None],
    session_exists: Callable[[str], bool],
) -> ChatProjectsService:
    service = ChatProjectsService(app, config, conversation_store, devices, session_exists)
    app.state.chat_projects = service

    def _refuse(request: Request) -> JSONResponse | None:
        if not is_owner(request):
            return _err(403, "operator_only", "only the owner may do this")
        if service.store() is None:
            return _no_store()
        return None

    @app.get("/api/chat-projects")
    async def list_chat_projects(request: Request) -> Any:
        store = service.store()
        if store is None:
            return _no_store()
        if is_owner(request):
            projects = store.all()
            # Off the event loop: each project's folder is read (instructions, reference listing).
            full = await asyncio.to_thread(lambda: [service.full(p) for p in projects])
            return {"role": "owner", "rev": store.rev(), "projects": full}
        own = owned_ids(request) or set()
        return {"role": "chat_only", "projects": [
            {"id": v.id, "name": v.name, "sessions": len(v.own_session_ids)} for v in store.visible_to(own)]}

    @app.get("/api/chat-projects/{project_id}")
    async def get_chat_project(project_id: str, request: Request) -> Any:
        store = service.store()
        if store is None:
            return _no_store()
        if is_owner(request):
            p = store.get(project_id)
            if p is None:
                return _err(404, "unknown_project", "no such project")
            full = await asyncio.to_thread(service.full, p)
            return {**full, "session_ids": store.session_ids(project_id)}
        own = owned_ids(request) or set()
        for v in store.visible_to(own):
            if v.id == project_id:
                return {"id": v.id, "name": v.name, "session_ids": v.own_session_ids}
        return _err(404, "unknown_project", "no such project")

    # ── instructions (design 3.4) ────────────────────────────────────────────────────────────────────────────────
    from prometheus.projects import content as _content

    _uploads = asyncio.Semaphore(2)  # at most two 20 MB uploads held in memory at once

    def _project_or_refusal(request: Request, project_id: str) -> JSONResponse | Project:
        refusal = _refuse(request)
        if refusal is not None:
            return refusal
        p = service.store().get(project_id)
        if p is None:
            return _err(404, "unknown_project", "no such project")
        return p

    def _fs_error(exc: BaseException) -> JSONResponse:
        """The answer for a filesystem refusal (never a path in it beyond the file's own name)."""
        if isinstance(exc, _content.FolderMissing):
            return _err(409, "folder_missing", str(exc))
        if isinstance(exc, _content.NotAFile):
            return _err(409, "not_a_file", str(exc))
        if isinstance(exc, _content.TooLarge):
            return _err(409, "too_large", str(exc))
        logger.warning("chat project file operation failed: %s", type(exc).__name__, exc_info=True)
        return _err(500, "io_error", "the file could not be read or written")

    async def _changed(change: str, project_id: str) -> None:
        service.store().touch()  # a real rev move, so a client that follows rev re-reads
        await service.notify(change, project_id)

    @app.get("/api/chat-projects/{project_id}/instructions", response_model=None)
    async def get_instructions(project_id: str, request: Request) -> Any:
        p = _project_or_refusal(request, project_id)
        if isinstance(p, JSONResponse):
            return p
        try:
            text, rev = await asyncio.to_thread(_content.read_instructions, p.folder)
        except (OSError, _content.NotAFile, _content.FolderMissing, _content.TooLarge) as exc:
            return _fs_error(exc)
        effective = await asyncio.to_thread(_content.effective_files, p.folder, config)
        return {"content": text, "rev": rev, "max_chars": _content.loader_limits(config)[0], "effective": effective}

    @app.put("/api/chat-projects/{project_id}/instructions", response_model=None)
    async def put_instructions(project_id: str, request: Request) -> Any:
        p = _project_or_refusal(request, project_id)
        if isinstance(p, JSONResponse):
            return p
        body = await _read_json(request)
        if not isinstance(body, dict) or not isinstance(body.get("content"), str) \
                or not isinstance(body.get("base_rev"), str) or not isinstance(body.get("shadow_ok", False), bool):
            return _err(400, "bad_request", 'send {"content": "...", "base_rev": "..."}')
        text = body["content"]
        max_chars = _content.loader_limits(config)[0]
        if len(text) > max_chars:
            return _err(400, "too_long", f"at most {max_chars} characters (the loader's per-file limit)")
        if not body.get("shadow_ok", False):
            shadowed = _content.shadowed_by_save(p.folder)
            if shadowed:
                return _err(409, "would_shadow", "a PROMETHEUS.md here would stop the loader reading "
                            + ", ".join(shadowed), files=shadowed)
        try:
            rev = await asyncio.to_thread(_content.write_instructions, p.folder, text, body["base_rev"])
        except _content.Conflict as exc:
            return _err(409, "conflict", str(exc), content=exc.content, rev=exc.rev)
        except UnicodeError:
            return _err(400, "bad_request", "the text is not valid Unicode")
        except (OSError, _content.NotAFile, _content.FolderMissing, _content.TooLarge) as exc:
            return _fs_error(exc)
        logger.info("chat project %s: instructions saved by the owner (%d characters)", project_id, len(text))
        await _changed("instructions", project_id)
        return {"ok": True, "rev": rev}

    # ── reference files (design 3.5) ─────────────────────────────────────────────────────────────────────────────
    _INLINE_UNSAFE = ("text/html", "image/svg+xml", "application/xhtml+xml", "text/xml", "application/xml")

    def _bad_name() -> JSONResponse:
        return _err(400, "bad_name", "a file name is one segment: no slash, no leading or trailing dot or space, "
                    "at most 120 characters")

    @app.get("/api/chat-projects/{project_id}/files", response_model=None)
    async def list_files(project_id: str, request: Request) -> Any:
        p = _project_or_refusal(request, project_id)
        if isinstance(p, JSONResponse):
            return p
        try:
            return {"files": await asyncio.to_thread(_content.list_reference, p.folder)}
        except (OSError, _content.NotAFile, _content.FolderMissing) as exc:
            return _fs_error(exc)

    @app.get("/api/chat-projects/{project_id}/files/{name}", response_model=None)
    async def get_file(project_id: str, name: str, request: Request) -> Any:
        import mimetypes

        p = _project_or_refusal(request, project_id)
        if isinstance(p, JSONResponse):
            return p
        if not _content.valid_reference_name(name):
            return _bad_name()
        try:
            data = await asyncio.to_thread(_content.read_reference, p.folder, name)
        except FileNotFoundError:
            return _err(404, "unknown_file", "no such reference file")
        except (OSError, _content.NotAFile, _content.FolderMissing, _content.TooLarge) as exc:
            return _fs_error(exc)
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if mime in _INLINE_UNSAFE:
            mime = "application/octet-stream"  # never markup a browser could run in the daemon's origin
        return Response(content=data, media_type=mime, headers={
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Content-Disposition": "attachment",
            "Content-Security-Policy": "sandbox; default-src 'none'"})

    @app.put("/api/chat-projects/{project_id}/files/{name}", response_model=None)
    async def put_file(project_id: str, name: str, request: Request, replace: str | None = None) -> Any:
        p = _project_or_refusal(request, project_id)
        if isinstance(p, JSONResponse):
            return p
        if not _content.valid_reference_name(name):
            return _bad_name()
        if replace not in (None, "true", "false"):
            return _err(400, "bad_request", "replace is true or false")
        async with _uploads:
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > _content.MAX_FILE_BYTES:
                    return _err(413, "too_large", f"at most {_content.MAX_FILE_BYTES} bytes")
            try:
                info = await asyncio.to_thread(_content.write_reference, p.folder, name, body,
                                               replace=replace == "true")
            except _content.Exists:
                return _err(409, "exists", "a reference file has that name; send ?replace=true to overwrite it")
            except _content.TooMany:
                return _err(409, "too_many", f"at most {_content.MAX_FILES} reference files")
            except (OSError, _content.NotAFile, _content.FolderMissing) as exc:
                return _fs_error(exc)
        await _changed("files", project_id)
        return {"ok": True, "file": info}

    @app.delete("/api/chat-projects/{project_id}/files/{name}", response_model=None)
    async def delete_file(project_id: str, name: str, request: Request) -> Any:
        p = _project_or_refusal(request, project_id)
        if isinstance(p, JSONResponse):
            return p
        if not _content.valid_reference_name(name):
            return _bad_name()
        try:
            await asyncio.to_thread(_content.delete_reference, p.folder, name)
        except FileNotFoundError:
            return _err(404, "unknown_file", "no such reference file")
        except (OSError, _content.NotAFile, _content.FolderMissing) as exc:
            return _fs_error(exc)
        await _changed("files", project_id)
        return {"ok": True}

    @app.get("/api/chat-project-suggestions", response_model=None)
    async def chat_project_suggestions(request: Request) -> Any:
        # Its own path: under /api/chat-projects/ it would match the chat-only {project_id} pattern.
        refusal = _refuse(request)
        if refusal is not None:
            return refusal
        return service.suggestions()

    @app.post("/api/chat-projects", response_model=None)
    async def create_chat_project(request: Request) -> Any:
        refusal = _refuse(request)
        if refusal is not None:
            return refusal
        store = service.store()
        body = await _read_json(request)
        if not isinstance(body, dict):
            return _err(400, "bad_request", "send a JSON object")
        raw_name, folder_raw, import_key = body.get("name"), body.get("folder"), body.get("import_key")
        name = clean_name(raw_name) if isinstance(raw_name, str) else ""
        if not name:
            return _err(400, "name_required", "a project needs a name")
        if folder_raw is not None and not isinstance(folder_raw, str):
            return _err(400, "bad_folder", "folder is an absolute path or null")
        if import_key is not None and not (isinstance(import_key, str) and _IMPORT_KEY.fullmatch(import_key)):
            return _err(400, "bad_request", "import_key is 1 to 200 printable characters")
        if import_key and (found := store.by_import_key(import_key)) is not None:
            return JSONResponse({"ok": True, "project": service.full(found), "created": False})
        existing = store.by_name(name)
        made: Path | None = None
        if folder_raw is None:
            if existing is not None:
                return JSONResponse({"ok": True, "project": service.full(existing), "created": False})
            made, why = service.make_managed(name)
            if made is None:
                return _err(400, "bad_folder", why or "the managed folder could not be made")
            folder, managed = made, True
        else:
            if existing is not None:
                # The name first (design 3.1 rule 2): the same folder by identity is the same project; anything
                # else, even a folder that would not validate, is a name clash.
                same = _ident(Path(os.path.expanduser(folder_raw)))
                if same is not None and same == _ident(Path(existing.folder)):
                    return JSONResponse({"ok": True, "project": service.full(existing), "created": False})
                return _err(409, "name_taken", f"a project is already called {existing.name!r}")
            resolved, why = service.validate(folder_raw)
            if resolved is None:
                if why and "already uses that folder" in why:
                    return _err(409, "folder_in_use", why)
                return _err(400, "bad_folder", why or "that folder cannot be used")
            if store.by_folder(str(resolved)) is not None:
                return _err(409, "folder_in_use", "another project already uses that folder")
            folder, managed = resolved, False
        try:
            project = store.insert(name, str(folder), managed=managed, import_key=import_key)
        except (NameTaken, FolderInUse, ImportKeyTaken) as exc:
            if made is not None:
                _remove_empty(made)
            # Two creates at once: the winner's project when this one would have returned it.
            winner = store.by_import_key(import_key) if isinstance(exc, ImportKeyTaken) and import_key else None
            if winner is None and isinstance(exc, NameTaken):
                same = store.by_name(name)
                if same is not None and (folder_raw is None or same.folder == str(folder)):
                    winner = same
            if winner is not None:
                return JSONResponse({"ok": True, "project": service.full(winner), "created": False})
            return _err(409, exc.kind, str(exc))
        await service.notify("created", project.id)
        return JSONResponse(status_code=201, content={"ok": True, "project": service.full(project), "created": True})

    @app.put("/api/chat-projects/{project_id}", response_model=None)
    async def rename_chat_project(project_id: str, request: Request) -> Any:
        refusal = _refuse(request)
        if refusal is not None:
            return refusal
        body = await _read_json(request)
        raw = body.get("name") if isinstance(body, dict) else None
        if not isinstance(raw, str) or not clean_name(raw):
            return _err(400, "name_required", "a project needs a name")
        try:
            project = service.store().rename(project_id, raw)
        except UnknownProject:
            return _err(404, "unknown_project", "no such project")
        except NameTaken as exc:
            return _err(409, "name_taken", str(exc))
        await service.notify("updated", project_id)
        return {"ok": True, "project": service.full(project)}

    @app.delete("/api/chat-projects/{project_id}", response_model=None)
    async def delete_chat_project(project_id: str, request: Request) -> Any:
        refusal = _refuse(request)
        if refusal is not None:
            return refusal
        try:
            unfiled = service.store().delete(project_id)
        except UnknownProject:
            return _err(404, "unknown_project", "no such project")
        await service.notify("deleted", project_id)
        return {"ok": True, "id": project_id, "unfiled": unfiled}

    @app.put("/api/sessions/{session_id}/project", response_model=None)
    async def file_session(session_id: str, request: Request) -> Any:
        refusal = _refuse(request)
        if refusal is not None:
            return refusal
        body = await _read_json(request)
        if not isinstance(body, dict) or "chat_project_id" not in body:
            return _err(400, "bad_request", 'send {"chat_project_id": "<id>" | null}')
        pid = body["chat_project_id"]
        flags = {k: body.get(k, False) for k in ("replace_workspace", "keep_workspace", "device_session")}
        if (pid is not None and not isinstance(pid, str)) or not all(isinstance(v, bool) for v in flags.values()):
            return _err(400, "bad_request", "chat_project_id is a string or null; the flags are true or false")
        return await service.file(session_id, pid, **flags)

    return service
