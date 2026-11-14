"""Each session's reply, held from the frames the bridge broadcasts (docs/design/speech-route.md, section 2.3).

:class:`ReplyTap` is an observer of ``WebSocketBridge.broadcast``: it sees every frame, for every session, the moment
it is broadcast. It holds each session's running reply from its start, so a stream that opens late still speaks the
reply from its first word, and the last finished ones for :data:`FINISHED_KEEP` seconds.

A reply is a turn. Turns in a session run one at a time (the bridge holds a per-session lock), so every ``chat_delta``,
``chat_discard``, ``tool_call_start`` and ``error`` for the session belongs to the turn that holds it:

* ``agent_state`` ``thinking`` counts a turn in, ``idle`` counts one out (never below zero). A queued turn sends
  ``thinking`` too, so the count says how many are open, not which one is running. A count going from zero to one
  makes a running reply, unbound;
* a reply **binds** to the first ``message_id`` its text or ``chat_done`` carries, and is held by
  ``(session_id, message_id)``: ids are ``asst-<milliseconds>`` and can repeat across sessions;
* it **ends** on its own ``chat_done`` (``interrupted`` or ``complete``), an ``error`` for the session (``failed``),
  or the count reaching zero (``ended``, or ``no_reply`` for a reply that never bound);
* a frame while no turn is open is not a reply, and neither is a ``computer:`` id (a desktop task's ``chat_done``) nor
  a desktop task's step (``tool_call_start`` with ``inputs.origin == "user_task"``);
* a turn that ended but has not sent its ``idle`` yet is not open (the bridge still pushes and titles in between);
* at most :data:`TEXT_KEEP` characters of a reply's text are kept.

The tap must be registered when the bridge is made (the launcher does): a turn already running when it starts would
be misread.

A reply's ``events`` are ``("reply", message_id)``, ``("text", str)``, ``("boundary", None)`` (a tool call started),
``("discard", keep_chars)`` and, last, ``("end", reason)``. :meth:`Reply.follow` yields them in order as they come.

:meth:`ReplyTap.observe` is synchronous, never blocks and never raises: it runs inside the bridge's broadcast, where a
raise would roll the turn back.

Source: novel code for Prometheus, 2026-11-13.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: How long a finished reply is kept for a stream that names its ``message_id``.
FINISHED_KEEP = 120.0
#: The most text kept per reply. Only the first 5,000 characters are ever spoken; the rest is slack for a discard.
TEXT_KEEP = 64_000


@dataclass(eq=False)
class Reply:
    """One turn's reply in one session."""

    session_id: str
    message_id: str | None = None
    events: list[tuple[str, Any]] = field(default_factory=list)
    ended: str | None = None
    ended_at: float | None = None
    kept: int = 0
    _changed: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def _add(self, kind: str, value: Any = None) -> None:
        self.events.append((kind, value))
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    def _bind(self, message_id: str) -> None:
        self.message_id = message_id
        self._add("reply", message_id)

    async def follow(self) -> AsyncIterator[tuple[str, Any]]:
        """Every event, from the first, as it comes; the last is ``("end", reason)``."""
        seen = 0
        while True:
            while seen < len(self.events):
                event = self.events[seen]
                seen += 1
                yield event
                if event[0] == "end":
                    return
            await self._changed.wait()


@dataclass
class _Session:
    open_turns: int = 0
    running: Reply | None = None
    finished: dict[str, Reply] = field(default_factory=dict)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    waiters: int = 0
    # Turns that ended (chat_done or error) and have not sent their idle yet. Between the two the bridge still does
    # work (a push, the title), and a turn in that gap is not open: a stream opening then waits for the next one.
    ended_awaiting_idle: int = 0
    # The reply of the turn that last started, for a waiter that wakes after it already ended.
    last_started: Reply | None = None


def _open(session: _Session) -> int:
    """Turns open in the session: counted in by thinking, out by idle, less those that ended and await their idle."""
    return max(0, session.open_turns - session.ended_awaiting_idle)


class ReplyTap:
    """Holds every session's reply from the bridge's frames. Register :meth:`observe` on the bridge."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._sessions: dict[str, _Session] = {}

    # --------------------------------------------------------------- frames
    def observe(self, frame: Any) -> None:
        """One broadcast frame. Never raises."""
        try:
            self._observe(frame)
        except Exception:  # noqa: BLE001 - a raise here would roll the turn back
            logger.warning("speech: a frame could not be read", exc_info=True)

    def _observe(self, frame: Any) -> None:
        if not isinstance(frame, dict):
            return
        payload = frame.get("payload")
        if not isinstance(payload, dict):
            return
        sid = payload.get("session_id")
        if not isinstance(sid, str) or not sid:
            return
        kind = frame.get("type")
        if kind in ("agent_state", "chat_done"):
            self._prune()  # not on every delta: a turn's ends are often enough
        if kind == "agent_state":
            self._agent_state(sid, payload.get("state"))
            return
        if kind not in ("chat_delta", "chat_discard", "chat_done", "tool_call_start", "error"):
            return
        mid = payload.get("message_id")
        if isinstance(mid, str) and mid.startswith("computer:"):
            return
        if kind == "tool_call_start":
            inputs = payload.get("inputs")
            if isinstance(inputs, dict) and inputs.get("origin") == "user_task":
                return  # a desktop task's step, not this reply's tool call
        session = self._sessions.get(sid)
        if session is None or (_open(session) == 0 and session.running is None):
            return  # no turn is open: not a reply
        reply = session.running
        if reply is None:
            reply = session.running = Reply(sid)
        if kind in ("chat_delta", "chat_done") and isinstance(mid, str) and mid:
            if reply.message_id is None:
                reply._bind(mid)
            elif reply.message_id != mid:
                # Another turn's frame while this one never sent chat_done (it was cancelled): its idle has already
                # been counted, so it is not awaiting one.
                self._end(session, "ended", by_idle=True)
                reply = session.running = Reply(sid)
                reply._bind(mid)
        if kind == "chat_delta":
            content = payload.get("content")
            if isinstance(content, str) and content and reply.kept < TEXT_KEEP:
                content = content[: TEXT_KEEP - reply.kept]
                reply.kept += len(content)
                reply._add("text", content)
        elif kind == "tool_call_start":
            reply._add("boundary")
        elif kind == "chat_discard":
            keep = payload.get("keep_chars")
            if isinstance(keep, int) and not isinstance(keep, bool) and mid == reply.message_id:
                reply._add("discard", keep)
        elif kind == "chat_done":
            if isinstance(mid, str) and mid == reply.message_id:
                self._end(session, "interrupted" if payload.get("interrupted") else "complete")
        elif kind == "error":
            self._end(session, "failed")

    def _agent_state(self, sid: str, state: Any) -> None:
        session = self._sessions.setdefault(sid, _Session())
        if state == "thinking":
            session.open_turns += 1
            if session.running is None:
                session.running = Reply(sid)
            session.last_started = session.running
            started, session.started = session.started, asyncio.Event()
            started.set()
        elif state == "idle":
            session.open_turns = max(0, session.open_turns - 1)
            if session.ended_awaiting_idle > 0:
                session.ended_awaiting_idle -= 1
            if _open(session) == 0 and session.running is not None:
                reply = session.running
                self._end(session, "ended" if reply.message_id or reply.events else "no_reply", by_idle=True)

    def _end(self, session: _Session, reason: str, *, by_idle: bool = False) -> None:
        reply = session.running
        if reply is None:
            return
        session.running = None
        if not by_idle:
            session.ended_awaiting_idle = min(session.open_turns, session.ended_awaiting_idle + 1)
        reply.ended, reply.ended_at = reason, self._clock()
        if reply.message_id is not None:
            # Two turns can get the same asst-<ms> id; the first keeps it.
            session.finished.setdefault(reply.message_id, reply)
        reply._add("end", reason)

    def _prune(self) -> None:
        now = self._clock()
        for sid in list(self._sessions):
            session = self._sessions[sid]
            for mid in [m for m, r in session.finished.items() if r.ended_at is not None
                        and now - r.ended_at > FINISHED_KEEP]:
                del session.finished[mid]
            if (not session.finished and session.running is None and session.open_turns == 0
                    and session.waiters == 0):
                del self._sessions[sid]

    # ---------------------------------------------------------------- find
    async def find(self, session_id: str, *, message_id: str | None = None, wait: float = 30.0) -> Reply | None:
        """The reply a stream should speak (design 2.3), or None for ``no_reply``.

        *message_id*: that reply in this session, running or finished within :data:`FINISHED_KEEP`. Else the running
        turn's (made now, unbound, when a turn is open but has sent nothing). Else the next turn to start within
        *wait* seconds.
        """
        self._prune()
        session = self._sessions.get(session_id)
        if message_id is not None:
            if session is None:
                return None
            if session.running is not None and session.running.message_id == message_id:
                return session.running
            return session.finished.get(message_id)
        if session is not None and session.running is not None:
            return session.running
        if session is not None and _open(session) > 0:
            session.running = Reply(session_id)
            return session.running
        session = self._sessions.setdefault(session_id, _Session())
        session.waiters += 1  # a session someone waits on is never pruned
        try:
            await asyncio.wait_for(session.started.wait(), wait)
        except TimeoutError:
            return None
        finally:
            session.waiters -= 1
        return session.last_started
