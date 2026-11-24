"""``GET /api/sessions/{session_id}/git``: what git says about a conversation's work folder, and the frame that says it
changed (contract: Beacon's ``git-status.md``, capability ``git/1``).

The request names a session, never a path: the folder is the session's bound workspace, resolved by the daemon. The
reading itself is ``security/git_read.py`` (git cannot start a program) and ``security/git_pr.py`` (``gh``, outside
the repository). This module is the route, the cache validator and the watcher.

* **Who may call.** The global token and owner devices. A scoped device gets ``403 operator_only`` from the bearer
  middleware (``web/route_access.py`` lists the route as the operator's); the file names and the state of a work
  folder are the owner's.
* **ETag.** The answer's validator covers everything but ``read_at``. ``If-None-Match`` with it answers ``304``.
* **The frame.** ``workspace_git_changed`` ``{session_id, read_at}`` goes to owner devices watching the session when
  the daemon's reading of the folder differs from the last one a client was given (by a frame or by this route). It
  carries no state; the client fetches. It is re-read after a turn ends, after a tool call in that session, when the
  workspace binding changes, and every 30 seconds for sessions an owner device is watching and that have been active
  lately. At most one frame per session per two seconds. It is not written to the durable event feed (it is sent to
  the sockets directly, not through the signal bus).

Source: novel code for Prometheus, 2026-11.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
from collections.abc import Callable
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from prometheus.security import git_pr, git_read
from prometheus.web.session_scope import event_session_id

logger = logging.getLogger(__name__)

#: At most one frame per session in this long.
COALESCE_S = 2.0
#: A moment for a tool call's writes to land before the folder is read.
SETTLE_S = 0.3
#: How often a watched, recently active session is read.
TIMER_S = 30.0
#: A session counts as recently active for the timer this long after a frame about it or a fetch.
ACTIVE_S = 1800.0
#: The most sessions the timer reads in one tick.
TIMER_MAX = 20
#: How long the route waits for a pull request lookup it had to start (it goes on in the background).
PR_WAIT_S = 3.0
#: Frames that mean "something may have changed in that session's folder".
_TRIGGERS = frozenset({"chat_done", "tool_call_end", "command_done"})


def etag_of(body: dict[str, Any]) -> str:
    """A validator for an answer: everything but the time it was read."""
    stable = {k: v for k, v in body.items() if k != "read_at"}
    digest = hashlib.sha256(json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()[:20]
    return f'W/"{digest}"'


def not_modified(header: str | None, etag: str) -> bool:
    """Whether an ``If-None-Match`` header names this validator (weak comparison; ``*`` names anything)."""
    if not header:
        return False
    if header.strip() == "*":
        return True
    wanted = etag.removeprefix("W/")
    return any(part.strip().removeprefix("W/") == wanted for part in header.split(","))


class GitWatch:
    """Reads a session's work folder for the route, and tells owner devices when it changed."""

    def __init__(self, *, config: dict, conversation_store: Callable[[], Any]) -> None:
        self._config = config
        self._store = conversation_store
        self.bridge: Any = None
        self._sent: dict[str, str] = {}            # session -> validator of the reading its last frame announced
        self._frame_at: dict[str, float] = {}      # session -> when its last frame went out (monotonic)
        self._active: dict[str, float] = {}        # session -> last activity (monotonic)
        self._pending: dict[str, asyncio.Task] = {}
        self._dirty: set[str] = set()              # touched while a re-read was running: read once more
        self._locks: dict[str, asyncio.Lock] = {}
        self._pr_tasks: dict[str, asyncio.Task] = {}
        self._pr_waiters: dict[str, set[str]] = {}  # folder -> sessions waiting on its pull request lookup
        self._timer: asyncio.Task | None = None
        self._rotate = 0
        self._prepared = False

    # ── reading ───────────────────────────────────────────────────────────────────────────────────────────

    def folder_of(self, session_id: str) -> str | None:
        store = self._store()
        if store is None:
            return None
        try:
            bound = store.get_session_workspace(session_id)
        except Exception:
            return None
        return bound if isinstance(bound, str) and bound else None

    def _top_allowed(self) -> Callable[[str], bool]:
        return git_read.acceptable_top(self._config.get("security", {}) or {})

    async def _pr_for(self, session_id: str, folder: str, state: dict[str, Any], *,
                      wait: float) -> dict[str, object] | None:
        """The pull request for the state's branch, if its upstream says there could be one. A lookup that has to be
        started is waited for only ``wait`` seconds; it goes on in the background and every session waiting on it is
        told when it lands (a pull request that appears is a change)."""
        if not state.get("branch") or not state.get("upstream"):
            return None
        self._pr_waiters.setdefault(folder, set()).add(session_id)
        task = self._pr_tasks.get(folder)
        if task is None:
            task = asyncio.ensure_future(git_pr.read_pr(folder, self._top_allowed()))
            self._pr_tasks[folder] = task

            def landed(t: asyncio.Task, f: str = folder) -> None:
                self._pr_tasks.pop(f, None)
                waiting = self._pr_waiters.pop(f, set())
                # A pull request that turns up after the route stopped waiting is a change a client must hear of.
                if not t.cancelled() and t.exception() is None and t.result().kind == "pr":
                    for sid in waiting:
                        self.touch(sid)

            task.add_done_callback(landed)
        try:
            got = await asyncio.wait_for(asyncio.shield(task), wait)
        except (asyncio.TimeoutError, Exception):
            return None
        return got.pr if got.kind == "pr" else None

    async def snapshot(self, session_id: str, *, fresh: bool = False, pr_wait: float = PR_WAIT_S) -> dict[str, Any]:
        """The answer for a session: its folder's state and pull request, as the route sends it."""
        folder = self.folder_of(session_id)
        if folder is None:
            return git_read.payload(git_read.NONE, read_at=time.time())
        read = await git_read.read_git(folder, self._top_allowed(), max_age=0.0 if fresh else git_read.FRESH_S)
        pr = None
        if read.kind == "state":
            pr = await self._pr_for(session_id, folder, read.state, wait=pr_wait)
        return git_read.payload(read, pr=pr)

    # ── telling ───────────────────────────────────────────────────────────────────────────────────────────

    def note_active(self, session_id: str) -> None:
        """A client fetched this session's git state: it counts as active for the timer."""
        self._active[session_id] = time.monotonic()

    def observe(self, event: dict[str, Any]) -> None:
        """A frame the bridge broadcast (called for every one; never raises): a turn ended, or a tool call did."""
        try:
            sid = event_session_id(event)
            if sid is None:
                return
            self._active[sid] = time.monotonic()
            if event.get("type") in _TRIGGERS:
                self.touch(sid)
        except Exception:
            logger.debug("git watch observe failed", exc_info=True)

    def touch(self, session_id: str) -> None:
        """Something may have changed in this session's folder (or its binding did): re-read soon, and say so if it
        differs. Does nothing when no owner device is watching the session. A touch while a re-read is running is
        not lost: the running one reads once more when it finishes."""
        bridge = self.bridge
        if bridge is None or not git_read.capability_ready() or not bridge.operator_sockets_watching(session_id):
            return
        task = self._pending.get(session_id)
        if task is not None and not task.done():
            self._dirty.add(session_id)
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._reread(session_id))
        self._pending[session_id] = task

        def done(t: asyncio.Task, sid: str = session_id) -> None:
            if self._pending.get(sid) is t:
                self._pending.pop(sid, None)

        task.add_done_callback(done)

    async def _reread(self, session_id: str) -> None:
        try:
            while True:
                self._dirty.discard(session_id)
                await asyncio.sleep(SETTLE_S)
                await self.check(session_id)
                if session_id not in self._dirty:
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("git watch re-read failed for a session", exc_info=True)

    async def check(self, session_id: str, *, coalesce: bool = True) -> bool:
        """Read the folder afresh; send the frame if it differs from the reading the last frame announced.

        One at a time per session. At most one frame per session per ``COALESCE_S``: a read that finds a change
        inside the window waits out the rest of it first (``coalesce``), so what is sent is the latest reading.
        """
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            if coalesce:
                wait = self._frame_at.get(session_id, 0.0) + COALESCE_S - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
            body = await self.snapshot(session_id, fresh=True, pr_wait=PR_WAIT_S)
            tag = etag_of(body)
            known = self._sent.get(session_id)
            if known == tag or (known is None and body.get("kind") == "none"):
                return False                       # nothing a client could be missing
            bridge = self.bridge
            if bridge is None:
                return False
            # Recorded before the send and undone if nobody took it, so two checks cannot both announce one change.
            previous = self._sent.get(session_id)
            self._sent.pop(session_id, None)
            self._sent[session_id] = tag               # most recently used last
            try:
                sent = await bridge.send_to_operators_watching(
                    {"type": "workspace_git_changed", "timestamp": time.time(),
                     "payload": {"session_id": session_id, "read_at": body.get("read_at")}},
                    session_id,
                )
            except BaseException:
                self._restore(session_id, previous)
                raise
            if not sent:
                self._restore(session_id, previous)
                return False
            self._frame_at[session_id] = time.monotonic()
            if len(self._sent) > 500:
                for k in list(self._sent)[:250]:
                    self._sent.pop(k, None)
                    self._frame_at.pop(k, None)
                    self._locks.pop(k, None)
            return True

    def _restore(self, session_id: str, previous: str | None) -> None:
        if previous is None:
            self._sent.pop(session_id, None)
        else:
            self._sent[session_id] = previous

    # ── the slow timer ────────────────────────────────────────────────────────────────────────────────────

    def _candidates(self) -> list[str]:
        """Sessions the timer reads: those an owner socket names in its subscribe filter, and those active lately
        (a socket with no filter watches everything, so recent activity is what bounds it). Rotated, so more than
        ``TIMER_MAX`` of them are all read in turn."""
        bridge = self.bridge
        now = time.monotonic()
        for sid in [s for s, at in self._active.items() if now - at > ACTIVE_S]:
            self._active.pop(sid, None)
        named = bridge.operator_watched_sessions() if bridge is not None else set()
        every = sorted(named | set(self._active))
        if len(every) <= TIMER_MAX:
            return every
        start = self._rotate % len(every)
        self._rotate += TIMER_MAX
        return (every + every)[start:start + TIMER_MAX]

    async def _tick(self) -> None:
        bridge = self.bridge
        if bridge is None or not git_read.capability_ready():
            return
        for sid in self._candidates():
            if self.folder_of(sid) is None or not bridge.operator_sockets_watching(sid):
                continue
            if time.monotonic() - self._frame_at.get(sid, 0.0) < COALESCE_S:
                continue                           # said a moment ago: the next tick reads it
            with contextlib.suppress(Exception):
                await self.check(sid, coalesce=False)

    async def _run_timer(self) -> None:
        while True:
            await asyncio.sleep(TIMER_S)
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("git watch timer failed", exc_info=True)

    async def prepare(self) -> bool:
        """The self-test (is git held by the process limit here?), before any client can be told what this daemon can
        do. Whether ``git/1`` is on."""
        self._prepared = True
        try:
            ready = await git_read.prepare()
        except Exception:
            ready = False
            logger.warning("git status self-test could not run", exc_info=True)
        if ready:
            logger.info("git status: read-only reads of work folders are ON (git/1): git cannot start a program here")
        else:
            logger.warning("git status: OFF (git/1 is not advertised): the process limit does not hold for this user "
                           "(root, or a host that does not count processes per user), git was not found or it is "
                           "older than 2.31")
        return ready

    async def start(self) -> None:
        """Startup: the self-test if the launcher did not already run it, then the timer."""
        if not self._prepared:
            await self.prepare()
        if self._timer is None:
            self._timer = asyncio.get_running_loop().create_task(self._run_timer())

    async def stop(self) -> None:
        for task in (self._timer, *self._pending.values(), *self._pr_tasks.values()):
            if task is not None:
                task.cancel()
        self._timer = None


def register_git_routes(app: Any, *, config: dict, conversation_store: Callable[[], Any]) -> GitWatch:
    """The route, and the watcher the bridge feeds (``attach_bridge``)."""
    watch = GitWatch(config=config, conversation_store=conversation_store)
    app.state.git_watch = watch

    @app.get("/api/sessions/{session_id}/git")
    async def get_session_git(session_id: str, request: Request):
        """What git says about this session's work folder (``v: 1``). ``kind``: ``state``, ``none`` (no work folder,
        not in a repository, or no git) or ``unreadable`` with ``why``. Owner devices and the main key only."""
        body = await watch.snapshot(session_id)
        tag = etag_of(body)
        watch.note_active(session_id)
        headers = {"ETag": tag, "Cache-Control": "no-store"}
        if not_modified(request.headers.get("if-none-match"), tag):
            return Response(status_code=304, headers=headers)
        return JSONResponse(body, headers=headers)

    return watch


def attach_bridge(app: Any, bridge: Any) -> None:
    """Let the watcher hear the bridge's frames and send its own to owner sockets; start it with the app."""
    watch: GitWatch = app.state.git_watch
    watch.bridge = bridge
    bridge.frame_observers.append(watch.observe)
    app.router.add_event_handler("startup", watch.start)
    app.router.add_event_handler("shutdown", watch.stop)
