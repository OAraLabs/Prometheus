"""A Live Activity for a job the phone did not start (Beacon iOS "Piece 18").

A Live Activity used to exist only when the app started one, in the foreground, for a turn sent from that phone.
A coding run, a background task, a desktop task, or a turn sent from the desktop never got one, and those are the
long jobs. So the daemon starts one by push (iOS 17.2 push-to-start, ``event: start``) on the owner's devices
once a job has run :data:`START_AFTER_SECONDS` and is still running, updates it at most every
:data:`UPDATE_EVERY_SECONDS`, and ends it with how the job ended.

ONE PERIODIC TICK, NOT HOOKS
----------------------------
:class:`JobActivityWatcher` asks each kind's source what is running every :data:`TICK_SECONDS`. A job that ended
between two ticks is a job that is no longer listed; its source then says how it ended. The sources read state
the daemon already keeps, in memory:

* ``turn``    the bridge's running turns (``WebSocketBridge.running_turns``: every turn the web bridge runs, from
              Beacon, the desktop app, ``POST /api/chat`` or Paperclip). A Telegram turn runs in its gateway and
              is not listed. A turn's updates and its end keep riding the bridge (``agent_progress`` and
              ``chat_done`` in ``PushDispatcher``); the watcher only starts its activity.
* ``task``    the TaskManager's in-memory running records, minus coding runs.
* ``coding``  the same records whose session is ``coding:<id>``; ``detail`` is the last tool and ``steps`` the
              rounds, from the ``coding_tool`` / ``coding_round`` signals the coding live stream emits (a run the
              stream does not tail shows neither), and a run paused through ``/api/code/{id}/pause`` shows
              ``paused``.
* ``desktop`` the computer-use runner's running tasks. No content at all, like a desktop task's approval push:
              not the goal, not the app. ``steps`` is the count.

``title`` is the daemon's own words for the kind ("Coding run", "Background task: shell command"), never a
description or a goal, which are user or model text. ``detail`` is one short line the daemon already shows for
the job (a registered tool's name, a task kind), never a tool's arguments or any message text.

WHO GETS ONE
------------
Every owner device that registered a push-to-start token gets a start for every kind. A scoped device gets one
only for a turn in a session it owns. Nobody gets a start for a job they already show: a phone that started its
own activity for a turn registered that session's token, and is skipped. One start per job per device, ever.

EVERY START AND UPDATE IS STALE IN 45 SECONDS
---------------------------------------------
``stale-date`` is now + :data:`STALE_AFTER_SECONDS` on each. A daemon that dies mid-job stops updating, and the
lock screen then shows the activity as stale instead of a last state that looks alive.

THE PHONE'S CONTRACT
--------------------
:data:`CONTENT_STATE_KEYS` is the one definition of the ``content-state`` keys. ``docs/reference/job-activity.json``
is generated from this module (``scripts/gen_reference.py``) and a test fails when they drift; the phone's decoding
fixture is that file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: The four kinds of job, as ``attributes.kind`` and the ``kind`` of ``POST /api/devices/{id}/activity``.
JOB_KINDS: tuple[str, ...] = ("turn", "task", "coding", "desktop")
#: The ActivityKit attributes type the phone declares for these.
ATTRIBUTES_TYPE = "JobActivityAttributes"
#: ``attributes`` keys; ``session_id`` and ``node_pub`` are left out when there is none.
ATTRIBUTE_KEYS: tuple[str, ...] = ("kind", "id", "title", "session_id", "node_pub")
OPTIONAL_ATTRIBUTE_KEYS: tuple[str, ...] = ("session_id", "node_pub")
#: THE content-state keys. A turn's own updates carry these too, beside the turn's older keys.
CONTENT_STATE_KEYS: tuple[str, ...] = ("phase", "detail", "elapsedS", "steps")
#: ``phase`` while a job runs: a job's ``running`` or ``paused``, or a turn's own phase.
LIVE_PHASES: tuple[str, ...] = ("running", "paused", "thinking", "compacting", "generating", "tool")
#: ``phase`` in the final state: how the job ended. A turn that reached its round cap ends ``paused``.
END_PHASES: tuple[str, ...] = ("done", "failed", "stopped", "paused")

START_AFTER_SECONDS = 60.0
UPDATE_EVERY_SECONDS = 10.0
TICK_SECONDS = 10.0
STALE_AFTER_SECONDS = 45
#: How long the final state stays on the lock screen.
END_LINGER_SECONDS = 600
START_PRIORITY = 10
UPDATE_PRIORITY = 5

TITLE_CAP = 80
DETAIL_CAP = 80
ID_CAP = 200
#: APNs refuses a body over 4096 bytes. Bodies are capped field by field; this is the last guard.
BODY_LIMIT = 4000

#: Titles are the daemon's own words, one per kind (a task adds its kind in words). Never a description: a
#: coding goal and a task description are user or model text.
TURN_TITLE = "Reply in progress"
TASK_TITLE = "Background task"
CODING_TITLE = "Coding run"
DESKTOP_TITLE = "Desktop task"
#: ``detail`` when a tool's name is not a registered tool (or there is no registry to ask).
GENERIC_TOOL_DETAIL = "tool"
#: How long an ended job is remembered, so a token registered after its end is answered with the end (M1).
TOMBSTONE_SECONDS = 600.0
TOMBSTONES_REMEMBERED = 512
#: The most ends the boot sweep sends; rows past it wait for the next boot.
SWEEP_CAP = 200
_START_ALERT = {
    "turn": "A reply is still being written",
    "task": "A background task is still running",
    "coding": "A coding run is still running",
    "desktop": "A desktop task is still running",
}
_TASK_TYPE_WORDS = {
    "local_bash": "shell command",
    "local_agent": "agent",
    "remote_agent": "agent",
    "in_process_teammate": "agent",
    "file_watch": "watching for a file",
    "poll": "polling",
}
_CODING_PREFIX = "coding:"


def one_line(value: Any, cap: int) -> str:
    """The first non-blank line of *value*, stripped and cut to *cap* characters."""
    text = str(value or "").strip()
    return text.splitlines()[0].strip()[:cap] if text else ""


def content_state(*, phase: str, detail: str, elapsed_s: int, steps: int) -> dict[str, Any]:
    """The ``content-state`` of a job activity, in :data:`CONTENT_STATE_KEYS` order. The only place it is built."""
    values = {"phase": phase, "detail": one_line(detail, DETAIL_CAP), "elapsedS": int(elapsed_s),
              "steps": int(steps)}
    return {key: values.get(key) for key in CONTENT_STATE_KEYS}


def tool_detail(name: Any, known_tool: Callable[[str], bool] | None) -> str:
    """A tool's name as ``detail`` only when it is a registered tool; otherwise :data:`GENERIC_TOOL_DETAIL`.

    A name in a progress frame or a telemetry row is whatever the model asked for, and a model can ask for a
    "tool" whose name is a sentence."""
    text = str(name or "").strip()
    if not text:
        return ""
    try:
        known = known_tool is not None and bool(known_tool(text))
    except Exception:
        known = False
    return one_line(text, DETAIL_CAP) if known else GENERIC_TOOL_DETAIL


def task_title(task_type: Any) -> str:
    word = _TASK_TYPE_WORDS.get(str(task_type), "")
    return f"{TASK_TITLE}: {word}" if word else TASK_TITLE


def turn_end_phase(payload: dict) -> str:
    """How a turn ended, from its ``chat_done`` payload (and the bridge's ``failed`` for a turn that raised)."""
    if payload.get("failed"):
        return "failed"
    if payload.get("interrupted"):
        return "stopped"
    if payload.get("stop_reason") == "max_turns_exhausted":
        return "paused"
    return "done"


@dataclass(frozen=True)
class Job:
    """One running job, as a source sees it. Nothing here may be a tool's arguments or message text."""

    kind: str
    id: str
    title: str
    started_at: float
    session_id: str | None = None
    phase: str = "running"
    detail: str = ""
    steps: int = 0


class JobSource(Protocol):
    kind: str

    def running(self) -> Iterable[Job]: ...

    def outcome(self, job_id: str) -> str | None: ...


# ── bodies ────────────────────────────────────────────────────────────────────

def _attributes(job: Job, node_pub: str | None) -> dict[str, Any]:
    attrs: dict[str, Any] = {"kind": job.kind, "id": job.id[:ID_CAP], "title": one_line(job.title, TITLE_CAP)}
    if job.session_id:
        attrs["session_id"] = job.session_id[:ID_CAP]
    if node_pub:
        attrs["node_pub"] = node_pub
    return attrs


def _elapsed(job: Job, now: float) -> int:
    return max(0, int(now - job.started_at))


def start_aps(job: Job, now: float, node_pub: str | None) -> dict[str, Any]:
    return {
        "timestamp": int(now),
        "event": "start",
        "content-state": content_state(phase=job.phase, detail=job.detail, elapsed_s=_elapsed(job, now),
                                       steps=job.steps),
        "stale-date": int(now) + STALE_AFTER_SECONDS,
        "attributes-type": ATTRIBUTES_TYPE,
        "attributes": _attributes(job, node_pub),
        # Daemon-authored words only: the alert names the kind of job, never the job.
        "alert": {"title": "Still running", "body": _START_ALERT.get(job.kind, "A job is still running")},
    }


def update_aps(job: Job, now: float) -> dict[str, Any]:
    return {
        "timestamp": int(now),
        "event": "update",
        "content-state": content_state(phase=job.phase, detail=job.detail, elapsed_s=_elapsed(job, now),
                                       steps=job.steps),
        "stale-date": int(now) + STALE_AFTER_SECONDS,
    }


def end_aps(job: Job, now: float, phase: str) -> dict[str, Any]:
    return {
        "timestamp": int(now),
        "event": "end",
        "dismissal-date": int(now) + END_LINGER_SECONDS,
        "content-state": content_state(phase=phase, detail=job.detail, elapsed_s=_elapsed(job, now),
                                       steps=job.steps),
    }


def turn_end_aps(now: float, outcome: str, *, elapsed_s: int = 0, round_: int = 0, tool_calls: int = 0,
                 chars: int = 0) -> dict[str, Any]:
    """A turn's end, the one shape for the bridge's ``chat_done`` end and a late one: ``phase`` stays exactly
    ``"done"`` (the shipped per-turn decode reads it), how the turn ended is ``outcome``, the numbers are the turn's
    own, and it lingers (``dismissal-date`` :data:`END_LINGER_SECONDS` out) so a locked phone shows how it ended."""
    return {
        "timestamp": int(now),
        "event": "end",
        "dismissal-date": int(now) + END_LINGER_SECONDS,
        "content-state": {**content_state(phase="done", detail="", elapsed_s=elapsed_s, steps=tool_calls),
                          "outcome": outcome, "round": round_, "toolCalls": tool_calls, "chars": chars},
    }


def _fit(body: dict[str, Any]) -> dict[str, Any]:
    """The last guard on size: a body still over :data:`BODY_LIMIT` (titles full of characters JSON escapes six
    bytes wide) has its strings cut short. Never seen with the caps above; kept so APNs never refuses one."""
    if len(json.dumps(body).encode()) <= BODY_LIMIT:
        return body
    aps = dict(body["aps"])
    if "attributes" in aps:
        attrs = dict(aps["attributes"])
        for key, cap in (("title", 24), ("id", 64), ("session_id", 64)):
            if key in attrs:
                attrs[key] = attrs[key][:cap]
        aps["attributes"] = attrs
    state = dict(aps["content-state"])
    state["detail"] = state["detail"][:24]
    aps["content-state"] = state
    return {**body, "aps": aps}


# ── sources ───────────────────────────────────────────────────────────────────

class TurnSource:
    """The bridge's running turns: ``running_turns()`` → ``(session_id, started_at, progress)``."""

    kind = "turn"

    def __init__(self, bridge: Any, *, known_tool: Callable[[str], bool] | None = None) -> None:
        self._bridge = bridge
        self._known_tool = known_tool

    def running(self) -> Iterable[Job]:
        for session_id, started_at, progress in self._bridge.running_turns():
            phase = str(progress.get("phase") or "thinking")
            yield Job(kind="turn", id=session_id, title=TURN_TITLE, started_at=float(started_at),
                      session_id=session_id, phase=phase,
                      detail=tool_detail(progress.get("tool_name"), self._known_tool) if phase == "tool" else "",
                      steps=int(progress.get("tool_calls") or 0))

    def outcome(self, job_id: str) -> str | None:
        return None  # a turn's end rides chat_done (PushDispatcher.on_chat_done)


def _coding_paused(record: Any) -> bool:
    """Whether a coding run is paused: the control file ``/api/code/{id}/pause`` writes. Unreadable is not."""
    from prometheus.coding.control import control_path, read_state_for_write
    from prometheus.coding.managed import coding_control_dir

    coding_id = (record.session_id or "").removeprefix(_CODING_PREFIX) or record.id
    return read_state_for_write(control_path(coding_control_dir(coding_id))).paused


class TaskSource:
    """Background tasks (``kind="task"``) or coding runs (``kind="coding"``), from the TaskManager's memory."""

    #: Coding runs whose progress is remembered at once; far more than ever run together.
    REMEMBERED = 256

    def __init__(self, manager: Any, *, kind: str,
                 is_paused: Callable[[Any], bool] | None = None,
                 known_tool: Callable[[str], bool] | None = None) -> None:
        if kind not in ("task", "coding"):
            raise ValueError(f"TaskSource kind must be task or coding, not {kind!r}")
        self.kind = kind
        self._manager = manager
        self._is_paused = is_paused or _coding_paused
        self._known_tool = known_tool
        # coding session -> rounds seen / the last tool's name (signals from the coding live stream).
        self._rounds: dict[str, int] = {}
        self._last_tool: dict[str, str] = {}

    def _mine(self, record: Any) -> bool:
        coding = str(record.session_id or "").startswith(_CODING_PREFIX)
        return coding if self.kind == "coding" else not coding

    def running(self) -> Iterable[Job]:
        for record in self._manager.running_tasks():
            if not self._mine(record):
                continue
            started = float(record.started_at or record.created_at or 0.0)
            if self.kind == "coding":
                session = str(record.session_id)
                try:
                    paused = bool(self._is_paused(record))
                except Exception:
                    paused = False
                yield Job(kind="coding", id=record.id, title=CODING_TITLE,
                          started_at=started, session_id=session, phase="paused" if paused else "running",
                          detail=self._last_tool.get(session, ""), steps=self._rounds.get(session, 0))
            else:
                yield Job(kind="task", id=record.id, title=task_title(record.type),
                          started_at=started, session_id=record.session_id or None,
                          detail=_TASK_TYPE_WORDS.get(str(record.type), ""))

    def alive(self, job_id: str) -> bool:
        """Whether the task still runs, or will again (a file_watch or poll task resumed at boot keeps its id).
        Reads the durable store too, so it answers the same before and after ``resume_running``."""
        record = self._manager.get_task(job_id)
        return record is not None and str(record.status) in ("running", "pending")

    def outcome(self, job_id: str) -> str | None:
        record = self._manager.get_task(job_id)
        if record is None:
            return None
        if record.session_id:
            self._rounds.pop(record.session_id, None)
            self._last_tool.pop(record.session_id, None)
        return {"completed": "done", "failed": "failed", "blocked": "failed",
                "killed": "stopped"}.get(str(record.status))

    async def on_signal(self, signal: Any) -> None:
        """``coding_round`` / ``coding_tool`` from the coding live stream: count rounds, keep the last tool's name
        when it is a registered tool (its name only: the frame's other fields are not read)."""
        if self.kind != "coding":
            return
        payload = getattr(signal, "payload", None) or {}
        session = str(payload.get("session_id") or "")
        if not session.startswith(_CODING_PREFIX):
            return
        kind = getattr(signal, "kind", "")
        if kind == "coding_round":
            self._rounds[session] = self._rounds.get(session, 0) + 1
        elif kind == "coding_tool" and payload.get("tool_name"):
            self._last_tool[session] = tool_detail(payload["tool_name"], self._known_tool)
        for memory in (self._rounds, self._last_tool):
            while len(memory) > self.REMEMBERED:
                memory.pop(next(iter(memory)))


class DesktopSource:
    """The computer-use runner's running desktop tasks. No content: not the goal, not the app."""

    kind = "desktop"

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def running(self) -> Iterable[Job]:
        for task in self._runner.running():
            yield Job(kind="desktop", id=task.task_id, title=DESKTOP_TITLE, started_at=float(task.started_at),
                      session_id=task.session_id or None, steps=int(task.steps or 0))

    def alive(self, job_id: str) -> bool:
        task = self._runner.get(job_id)
        return task is not None and task.status == "running"

    def outcome(self, job_id: str) -> str | None:
        task = self._runner.get(job_id)
        if task is None:
            return None
        return {"done": "done", "abstained": "done", "stopped": "stopped", "refused": "failed",
                "failed": "failed", "limit": "failed"}.get(str(task.outcome or ""), "failed")


_TITLES = {"turn": TURN_TITLE, "task": TASK_TITLE, "coding": CODING_TITLE, "desktop": DESKTOP_TITLE}


# ── the watcher ───────────────────────────────────────────────────────────────

@dataclass
class _Tracked:
    job: Job
    started_on: set[str] = field(default_factory=set)   # devices sent a start (once each, ever)
    pushed_at: float = float("-inf")                     # the last start or update, for the throttle


@dataclass
class _Tombstone:
    """An ended job, remembered for :data:`TOMBSTONE_SECONDS` (review M1)."""

    job: Job
    phase: str
    ended_at: float
    started_on: set[str]
    ended_for: set[str] = field(default_factory=set)    # devices already sent a late end (once each)


def _key(job: Job) -> tuple[str, str, float]:
    # A turn is keyed by its own start too (review L2): back-to-back turns in one session are separate jobs, each
    # with its own start. Every other kind keeps one id for its whole life.
    return (job.kind, job.id, job.started_at if job.kind == "turn" else 0.0)


class JobActivityWatcher:
    """Starts, updates and ends the Live Activities of jobs the phone did not start. See the module docstring."""

    def __init__(self, dispatcher: Any, store: Any, *, sources: Iterable[JobSource],
                 now: Callable[[], float] = time.time, tick_seconds: float = TICK_SECONDS) -> None:
        self._dispatcher = dispatcher
        self._store = store
        self._sources = {source.kind: source for source in sources}
        self._now = now
        self._tick_seconds = tick_seconds
        self._tracked: dict[tuple[str, str, float], _Tracked] = {}
        # (kind, id) -> how it ended. A turn's id is its session id.
        self._tombstones: OrderedDict[tuple[str, str], _Tombstone] = OrderedDict()
        # Turns that ended (chat_done) -> when, kept like the tombstones. The bridge still lists a turn until its
        # _run_agent finally runs, a moment after chat_done; a tick in that gap must not track it again (it would
        # send a second start and then record the turn as stopped over its real outcome).
        self._ended_turns: OrderedDict[tuple[str, str, float], float] = OrderedDict()
        self._task: asyncio.Task | None = None
        self._sweep_task: asyncio.Task | None = None
        # A turn's end is told at the moment it ends (PushDispatcher.on_chat_done), not found at the next tick:
        # a token registered between the two would otherwise miss its end.
        if hasattr(dispatcher, "turn_end_listener"):
            dispatcher.turn_end_listener = self.turn_ended

    # ── the loop ──

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run(), name="job-activity-watcher")

    def stop(self) -> None:
        for task in (self._task, self._sweep_task):
            if task is not None:
                task.cancel()
        self._task = self._sweep_task = None

    async def run(self) -> None:
        first = True
        while True:
            await asyncio.sleep(self._tick_seconds)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("job activity tick failed", exc_info=True)
            if first:
                # After the first tick and beside the ticks, never in front of them; by then the daemon has
                # resumed its tasks, and sweep() asks each source besides.
                first = False
                self._sweep_task = asyncio.create_task(self._sweep_quietly(), name="job-activity-sweep")

    async def _sweep_quietly(self) -> None:
        try:
            await self.sweep()
        except Exception:
            logger.warning("job activity boot sweep failed", exc_info=True)

    def _alive(self, kind: str, job_id: str) -> bool:
        if any(key[0] == kind and key[1] == job_id for key in self._tracked):
            return True
        source = self._sources.get(kind)
        alive = getattr(source, "alive", None)
        try:
            return bool(alive(job_id)) if callable(alive) else False
        except Exception:
            return True  # unknown: leave the row for a later boot rather than end a live activity

    async def sweep(self) -> None:
        """At boot: a job token left from an earlier run whose job is not running (or coming back, as a resumed
        file_watch or poll task does) gets an ``end`` (phase ``stopped``), and that row, only that row, is dropped
        once the end was sent. At most :data:`SWEEP_CAP` ends. A turn's tokens are not jobs' (review L1)."""
        now = self._now()
        sent = 0
        for kind, job_id, target, token in self._store.all_job_activity_tokens():
            if sent >= SWEEP_CAP:
                break
            try:
                if self._alive(kind, job_id):
                    continue
                job = Job(kind=kind, id=job_id, title=_TITLES.get(kind, ""), started_at=now)
                body = _fit(self._dispatcher._named({"aps": end_aps(job, now, "stopped")}))
                outcome = await self._dispatcher.send_liveactivity(target, token, body, priority=UPDATE_PRIORITY,
                                                                   kind="liveactivity end")
                if outcome is None:
                    continue  # muted: not sent, the row stays
                sent += 1
                self._store.clear_job_activity_token(target.id, kind, job_id)
            except Exception:
                logger.warning("job activity: the boot end for %s %s failed", kind, job_id, exc_info=True)

    async def tick(self) -> None:
        now = self._now()
        self._forget_old_tombstones(now)
        seen: set[tuple[str, str, float]] = set()
        unreadable: set[str] = set()
        owners: set[str] | None = None
        for kind, source in self._sources.items():
            try:
                jobs = list(source.running())
            except Exception:
                # A source that cannot answer is not a source whose jobs all ended.
                logger.warning("job activity: the %s source failed; its jobs carry on", kind, exc_info=True)
                unreadable.add(kind)
                continue
            for job in jobs:
                key = _key(job)
                if key in self._ended_turns:
                    continue  # ended at chat_done; the bridge has not dropped it yet
                seen.add(key)
                tracked = self._tracked.get(key)
                if tracked is None:
                    tracked = self._tracked[key] = _Tracked(job)
                tracked.job = job
                try:  # one bad job must not stop the rest (review L4)
                    if now - job.started_at >= START_AFTER_SECONDS:
                        if owners is None:
                            owners = self._owner_ids()
                        await self._start(tracked, now, owners)
                    if job.kind != "turn":
                        await self._update(tracked, now)
                except Exception:
                    logger.warning("job activity: %s %s failed this tick", job.kind, job.id, exc_info=True)
        for key in [k for k in self._tracked if k not in seen and k[0] not in unreadable]:
            tracked = self._tracked.pop(key)
            try:
                await self._ended(tracked, now)
            except Exception:
                logger.warning("job activity: ending %s %s failed", key[0], key[1], exc_info=True)

    # ── the three pushes ──

    def _owner_ids(self) -> set[str]:
        try:
            return set(self._store.owner_device_ids())
        except Exception:
            logger.warning("job activity: owner devices unreadable", exc_info=True)
            return set()

    def _may_see(self, device_id: str, job: Job, owners: set[str]) -> bool:
        if device_id in owners:
            return True
        # A scoped device: only a turn in a session it owns.
        return job.kind == "turn" and bool(job.session_id) and \
            self._store.session_owner(job.session_id) == device_id

    def _already_shows(self, device_id: str, job: Job) -> bool:
        if job.kind == "turn":
            targets = self._store.activity_targets(job.session_id or job.id)
        else:
            targets = self._store.job_activity_targets(job.kind, job.id)
        return any(target.id == device_id for target, _token in targets)

    async def _start(self, tracked: _Tracked, now: float, owners: set[str]) -> None:
        job = tracked.job
        for target, token in self._store.activity_start_targets():
            if target.id in tracked.started_on or not self._may_see(target.id, job, owners):
                continue
            tracked.started_on.add(target.id)  # one attempt per device, whatever APNs answers: no storm
            if self._already_shows(target.id, job):
                continue
            body = _fit(self._dispatcher._named({"aps": start_aps(job, now, self._dispatcher.node_pub())}))
            outcome = await self._dispatcher.send_liveactivity(target, token, body, priority=START_PRIORITY,
                                                               kind="liveactivity start")
            tracked.pushed_at = now
            if outcome == "unregistered":
                self._store.clear_activity_start_token(target.id, token)

    async def _update(self, tracked: _Tracked, now: float) -> None:
        if now - tracked.pushed_at < UPDATE_EVERY_SECONDS:
            return
        job = tracked.job
        targets = self._store.job_activity_targets(job.kind, job.id)
        if not targets:
            return
        tracked.pushed_at = now
        body = _fit(self._dispatcher._named({"aps": update_aps(job, now)}))
        for target, token in targets:
            outcome = await self._dispatcher.send_liveactivity(target, token, body, priority=UPDATE_PRIORITY,
                                                               kind="liveactivity update")
            if outcome == "unregistered":
                self._store.clear_job_activity_token(target.id, job.kind, job.id)

    async def _ended(self, tracked: _Tracked, now: float) -> None:
        job = tracked.job
        if job.kind == "turn":
            # The turn's own end rode chat_done. Remembered only when a start was sent: a phone registering a
            # token for that session later is then answered with the end (review M1).
            # Normally turn_ended already did this, the moment the turn ended. A turn that vanished without a
            # chat_done (the daemon cancelled it) is remembered here, as stopped.
            if tracked.started_on:
                self._remember(job, "stopped", now, tracked.started_on)
            return
        phase = self._outcome(job) or "stopped"
        self._remember(job, phase, now, tracked.started_on)
        targets = self._store.job_activity_targets(job.kind, job.id)
        if targets:
            body = _fit(self._dispatcher._named({"aps": end_aps(job, now, phase)}))
            for target, token in targets:
                await self._dispatcher.send_liveactivity(target, token, body, priority=UPDATE_PRIORITY,
                                                         kind="liveactivity end")
        self._store.clear_job_activity_tokens(job.kind, job.id)

    def turn_ended(self, session_id: str, started_at: float | None, outcome: str) -> None:
        """A turn ended (``chat_done``): remember it now, with how it ended, if a start was sent for it. Keyed by
        the turn (its session and its start), so a later turn in the session neither races nor overwrites it.
        Without a start time (an older caller), the session's one tracked turn. Never raises."""
        try:
            keys = [k for k in self._tracked if k[0] == "turn" and k[1] == session_id
                    and (started_at is None or k[2] == float(started_at))]
            now = self._now()
            marks = list(keys)
            if started_at is not None:
                marks.append(("turn", session_id, float(started_at)))
            for mark in marks:
                self._ended_turns[mark] = now
                self._ended_turns.move_to_end(mark)
            while len(self._ended_turns) > TOMBSTONES_REMEMBERED:
                self._ended_turns.popitem(last=False)
            for key in keys:
                tracked = self._tracked.pop(key)
                if tracked.started_on:
                    self._remember(tracked.job, outcome, self._now(), tracked.started_on)
        except Exception:
            logger.warning("job activity: remembering the end of a turn in %s failed", session_id, exc_info=True)

    def _outcome(self, job: Job) -> str | None:
        source = self._sources.get(job.kind)
        try:
            return source.outcome(job.id) if source is not None else None
        except Exception:
            logger.warning("job activity: how %s %s ended is unreadable", job.kind, job.id, exc_info=True)
            return None

    # ── tombstones (review M1) ──

    def _remember(self, job: Job, phase: str, now: float, started_on: set[str]) -> None:
        key = (job.kind, job.id)
        self._tombstones[key] = _Tombstone(job=job, phase=phase, ended_at=now, started_on=set(started_on))
        self._tombstones.move_to_end(key)
        while len(self._tombstones) > TOMBSTONES_REMEMBERED:
            self._tombstones.popitem(last=False)

    def _forget_old_tombstones(self, now: float) -> None:
        for key in [k for k, t in self._tombstones.items() if now - t.ended_at > TOMBSTONE_SECONDS]:
            del self._tombstones[key]
        for turn in [k for k, at in self._ended_turns.items() if now - at > TOMBSTONE_SECONDS]:
            del self._ended_turns[turn]

    def _turn_running(self, session_id: str) -> bool:
        source = self._sources.get("turn")
        if source is None:
            return False
        try:
            return any(job.session_id == session_id or job.id == session_id for job in source.running())
        except Exception:
            return True  # unknown: do not end what may be a live activity

    async def end_if_ended(self, device_id: str, kind: str, job_id: str, token: str) -> bool:
        """For a token registered after its job ended: send the ``end`` to *token* now and say so (the caller then
        does not store it). False when the job is running, unknown, or ended over :data:`TOMBSTONE_SECONDS` ago.

        A turn's token counts only from a device that was sent that turn's start, and never while a turn runs in
        that session: a token registered then is for the new turn. Never raises."""
        try:
            now = self._now()
            self._forget_old_tombstones(now)
            stone = self._tombstones.get((kind, job_id))
            if stone is None or device_id in stone.ended_for:
                return False
            if kind == "turn":
                if device_id not in stone.started_on or self._turn_running(job_id):
                    return False
            elif any(key[0] == kind and key[1] == job_id for key in self._tracked):
                return False
            target = self._store.push_target(device_id)
            if target is None:
                return False
            if kind == "turn":
                aps = turn_end_aps(now, stone.phase, elapsed_s=_elapsed(stone.job, stone.ended_at))
            else:
                aps = end_aps(stone.job, now, stone.phase)
            outcome = await self._dispatcher.send_liveactivity(target, token, _fit(self._dispatcher._named(
                {"aps": aps})), priority=UPDATE_PRIORITY, kind="liveactivity end")
            if outcome != "ok":
                return False  # not delivered: the token is kept, and a later registration may try again
            stone.ended_for.add(device_id)  # once per device and job, counted only when it was delivered
            return True
        except Exception:
            logger.warning("job activity: the late end for %s %s failed", kind, job_id, exc_info=True)
            return False


# ── the phone's fixture ───────────────────────────────────────────────────────

_EXAMPLE_PUB = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8="


def fixture() -> dict[str, Any]:
    """``docs/reference/job-activity.json``: the contract the phone decodes, generated from this module."""
    job = Job(kind="coding", id="c1a2b3c4", title=CODING_TITLE, started_at=1_800_000_000.0,
              session_id="coding:c1a2b3c4", detail="edit_file", steps=4)
    now = job.started_at + 75.0
    return {
        "_generated": "DO NOT EDIT BY HAND. Regenerate with: uv run python scripts/gen_reference.py "
                      "(source: src/prometheus/push/job_activity.py)",
        "attributes_type": ATTRIBUTES_TYPE,
        "attributes_keys": list(ATTRIBUTE_KEYS),
        "optional_attributes_keys": list(OPTIONAL_ATTRIBUTE_KEYS),
        "content_state_keys": list(CONTENT_STATE_KEYS),
        "kinds": list(JOB_KINDS),
        "live_phases": list(LIVE_PHASES),
        "end_phases": list(END_PHASES),
        "start_after_seconds": int(START_AFTER_SECONDS),
        "update_every_seconds": int(UPDATE_EVERY_SECONDS),
        "stale_after_seconds": STALE_AFTER_SECONDS,
        # A turn's own end (the bridge path) keeps ``phase: "done"`` for the shipped decode; how it ended is here.
        "turn_end_outcome_key": "outcome",
        "examples": {
            "start": {"aps": start_aps(job, now, _EXAMPLE_PUB), "node_pub": _EXAMPLE_PUB},
            "update": {"aps": update_aps(job, now + 10), "node_pub": _EXAMPLE_PUB},
            "end": {"aps": end_aps(job, now + 20, "done"), "node_pub": _EXAMPLE_PUB},
        },
    }


def render_fixture() -> str:
    return json.dumps(fixture(), indent=2) + "\n"
