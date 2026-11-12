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

``detail`` is one short line the daemon already shows for the job (a tool name, a task kind), never a tool's
arguments or any message text.

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

TURN_TITLE = "Reply in progress"
DESKTOP_TITLE = "Desktop task"
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

    def __init__(self, bridge: Any) -> None:
        self._bridge = bridge

    def running(self) -> Iterable[Job]:
        for session_id, started_at, progress in self._bridge.running_turns():
            phase = str(progress.get("phase") or "thinking")
            yield Job(kind="turn", id=session_id, title=TURN_TITLE, started_at=float(started_at),
                      session_id=session_id, phase=phase,
                      detail=str(progress.get("tool_name") or "") if phase == "tool" else "",
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
                 is_paused: Callable[[Any], bool] | None = None) -> None:
        if kind not in ("task", "coding"):
            raise ValueError(f"TaskSource kind must be task or coding, not {kind!r}")
        self.kind = kind
        self._manager = manager
        self._is_paused = is_paused or _coding_paused
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
                yield Job(kind="coding", id=record.id, title=one_line(record.description, TITLE_CAP),
                          started_at=started, session_id=session, phase="paused" if paused else "running",
                          detail=self._last_tool.get(session, ""), steps=self._rounds.get(session, 0))
            else:
                yield Job(kind="task", id=record.id, title=one_line(record.description, TITLE_CAP),
                          started_at=started, session_id=record.session_id or None,
                          detail=_TASK_TYPE_WORDS.get(str(record.type), ""))

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
        (its name only: the frame's other fields are not read)."""
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
            self._last_tool[session] = one_line(payload["tool_name"], DETAIL_CAP)
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

    def outcome(self, job_id: str) -> str | None:
        task = self._runner.get(job_id)
        if task is None:
            return None
        return {"done": "done", "abstained": "done", "stopped": "stopped", "refused": "failed",
                "failed": "failed", "limit": "failed"}.get(str(task.outcome or ""), "failed")


# ── the watcher ───────────────────────────────────────────────────────────────

@dataclass
class _Tracked:
    job: Job
    started_on: set[str] = field(default_factory=set)   # devices sent a start (once each, ever)
    pushed_at: float = float("-inf")                     # the last start or update, for the throttle


class JobActivityWatcher:
    """Starts, updates and ends the Live Activities of jobs the phone did not start. See the module docstring."""

    def __init__(self, dispatcher: Any, store: Any, *, sources: Iterable[JobSource],
                 now: Callable[[], float] = time.time, tick_seconds: float = TICK_SECONDS) -> None:
        self._dispatcher = dispatcher
        self._store = store
        self._sources = {source.kind: source for source in sources}
        self._now = now
        self._tick_seconds = tick_seconds
        self._tracked: dict[tuple[str, str], _Tracked] = {}
        self._task: asyncio.Task | None = None

    # ── the loop ──

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run(), name="job-activity-watcher")

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self._tick_seconds)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("job activity tick failed", exc_info=True)

    async def tick(self) -> None:
        now = self._now()
        seen: set[tuple[str, str]] = set()
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
                key = (job.kind, job.id)
                seen.add(key)
                tracked = self._tracked.get(key)
                if tracked is None:
                    tracked = self._tracked[key] = _Tracked(job)
                tracked.job = job
                if now - job.started_at >= START_AFTER_SECONDS:
                    if owners is None:
                        owners = self._owner_ids()
                    await self._start(tracked, now, owners)
                if job.kind != "turn":
                    await self._update(tracked, now)
        for key in [k for k in self._tracked if k not in seen and k[0] not in unreadable]:
            tracked = self._tracked.pop(key)
            if key[0] != "turn":
                await self._end(tracked.job, now)

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

    async def _end(self, job: Job, now: float) -> None:
        targets = self._store.job_activity_targets(job.kind, job.id)
        if targets:
            source = self._sources.get(job.kind)
            try:
                phase = source.outcome(job.id) if source is not None else None
            except Exception:
                logger.warning("job activity: how %s %s ended is unreadable", job.kind, job.id, exc_info=True)
                phase = None
            body = _fit(self._dispatcher._named({"aps": end_aps(job, now, phase or "stopped")}))
            for target, token in targets:
                await self._dispatcher.send_liveactivity(target, token, body, priority=UPDATE_PRIORITY,
                                                         kind="liveactivity end")
        self._store.clear_job_activity_tokens(job.kind, job.id)


# ── the phone's fixture ───────────────────────────────────────────────────────

_EXAMPLE_PUB = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8="


def fixture() -> dict[str, Any]:
    """``docs/reference/job-activity.json``: the contract the phone decodes, generated from this module."""
    job = Job(kind="coding", id="c1a2b3c4", title="coding task c1a2b3c4: fix the parser", started_at=1_800_000_000.0,
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
        "examples": {
            "start": {"aps": start_aps(job, now, _EXAMPLE_PUB), "node_pub": _EXAMPLE_PUB},
            "update": {"aps": update_aps(job, now + 10), "node_pub": _EXAMPLE_PUB},
            "end": {"aps": end_aps(job, now + 20, "done"), "node_pub": _EXAMPLE_PUB},
        },
    }


def render_fixture() -> str:
    return json.dumps(fixture(), indent=2) + "\n"
