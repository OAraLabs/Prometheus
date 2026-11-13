"""Pause everything: one daemon-wide state that stops the agent and keeps it stopped (Beacon iOS "Piece 17").

Every other stop is for one thing, by id, and none of them holds: a cron job fires again a minute later, a chat
message starts a new turn, a pending approval answered afterwards runs its tool. A pause is "stop, and stay
stopped until a person says". This module is the state and the two checks every path uses:

  * :class:`PauseState` owns ``<data>/pause.json``. The daemon creates one at boot and :func:`install`\\s it;
    pausing writes the file (atomic, fsync'd, 0600) BEFORE anything is stopped, so nothing new can start while the
    rest is being stopped, and so a restart comes back paused.
  * :func:`current` answers "paused?" everywhere: from the installed state in the daemon, from the file (cached on
    its mtime) in a child process (a worker, a coding run) that has none.
  * A file that cannot be read or parsed means PAUSED (fail closed), with the reason saying so: an emergency stop
    that an unreadable file quietly undoes is not one.
  * :func:`run_as_turn` (and :func:`track_current_task_as_turn` for a per-request handler) list the tasks that
    exist for one turn, so a pause can cancel them; every other turn stops at ``run_loop``'s round check.

Resuming is not decided here: who may resume (a person, never the bare API token) is the route's rule.

Source: novel code for Prometheus, 2026-11-12.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PAUSE_FILENAME = "pause.json"


@dataclass(frozen=True)
class PauseInfo:
    since: float
    by: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"since": self.since, "by": self.by, "reason": self.reason}


#: Who paused and why are shown to the owner on every screen of the phone: one line, no control characters, short.
REASON_MAX_CHARS = 140
BY_MAX_CHARS = 80


#: Letters that render as nothing (Hangul fillers, the braille blank): a reason made of them looks blank.
_INVISIBLE = frozenset("\u115f\u1160\u3164\uffa0\u2800")


def _one_line(text: str, cap: int) -> str:
    """*text* as one printable line: control, format and combining characters dropped (stacked marks spill over the
    lines around it on a phone), invisible letters dropped, whitespace runs collapsed, capped, no trailing space."""
    import unicodedata

    # NFC first, so an accent typed as letter + combining mark becomes one letter and survives.
    kept = "".join(" " if ch.isspace() else ch for ch in unicodedata.normalize("NFC", str(text or ""))
                   if ch.isspace() or (unicodedata.category(ch)[0] not in ("C", "M") and ch not in _INVISIBLE))
    return " ".join(kept.split())[:cap].rstrip()


class Paused(RuntimeError):
    """Raised where a turn would start (or go on) while the agent is paused."""

    def __init__(self, info: PauseInfo) -> None:
        # Not who paused or why: this text can reach any client in a turn's error frame.
        super().__init__(f"Prometheus is paused (since {info.since:.0f})")
        self.info = info


def pause_path() -> Path:
    from prometheus.config.paths import data_dir_path

    return data_dir_path() / PAUSE_FILENAME


def _unreadable(why: str) -> PauseInfo:
    logger.error("pause state unreadable (%s): treating the agent as PAUSED until a person resumes it", why)
    return PauseInfo(since=time.time(), by="unreadable", reason=f"pause state unreadable: {why}")


def read_file(path: Path) -> PauseInfo | None:
    """The state *path* holds: None when not paused (no file, or ``paused: false``); paused when unreadable."""
    # Not re-sanitized here: only ``PauseState.pause`` writes the file, and it stores who and why as one short line
    # already (``_one_line``); the floor keeps the agent's shell from writing it.
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:                  # ValueError: bytes that are not UTF-8
        return _unreadable(type(exc).__name__)
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError):                  # RecursionError: JSON nested deeper than Python goes
        return _unreadable("not JSON")
    if not isinstance(data, dict) or not isinstance(data.get("paused"), bool):
        return _unreadable("no paused flag")
    if not data["paused"]:
        return None
    since, by, reason = data.get("since"), data.get("by"), data.get("reason", "")
    if not isinstance(since, (int, float)) or isinstance(since, bool) or not isinstance(by, str):
        return _unreadable("bad fields")
    return PauseInfo(since=float(since), by=by, reason=reason if isinstance(reason, str) else "")


def _write_file(path: Path, data: dict[str, Any]) -> None:
    """Atomic and durable: a temp file in the same folder, fsync'd, renamed over, the folder fsync'd."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Named with the file's own name as a PREFIX, so the trusted-state floor that covers ``pause.json*`` covers it.
    tmp = path.with_name(f"{path.name}.{os.urandom(4).hex()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, sort_keys=True))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:
        dfd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)


class PauseState:
    """The daemon's pause state, backed by one file. Read once at construction (boot)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or pause_path()
        self._info = read_file(self.path)

    @property
    def current(self) -> PauseInfo | None:
        return self._info

    #: Whether the pause in force is on disk (so it survives a restart). False when the write failed.
    persisted: bool = True

    def pause(self, *, by: str, reason: str = "") -> tuple[PauseInfo, bool]:
        """Pause, or return the pause already in force. ``(info, new)``.

        In force HERE first, then written: an emergency stop must not depend on the disk. A write that fails (a
        full or read-only data folder) is logged and leaves ``persisted`` False; the pause still holds in this
        process, and a later pause call tries the write again.
        """
        if self._info is not None:
            if not self.persisted:
                self._persist(self._info)
            return self._info, False
        info = PauseInfo(since=time.time(), by=_one_line(by, BY_MAX_CHARS) or "unknown",
                         reason=_one_line(reason, REASON_MAX_CHARS))
        self._info = info
        self._persist(info)
        return info, True

    def _persist(self, info: PauseInfo) -> None:
        try:
            _write_file(self.path, {"paused": True, **info.to_dict()})
            self.persisted = True
        except OSError:
            self.persisted = False
            logger.error("pause: could not write %s; the pause holds in this process but will NOT survive a "
                         "restart", self.path, exc_info=True)

    def resume(self, *, by: str) -> PauseInfo | None:
        """Clear the pause; returns the one cleared, or None if there was none. Starts nothing."""
        cleared = self._info
        _write_file(self.path, {"paused": False, "resumed_at": time.time(), "resumed_by": by})
        self._info = None
        self.persisted = True
        return cleared


# ── process-wide accessors ───────────────────────────────────────────────────

_installed: PauseState | None = None
_file_cache: tuple[tuple[int, int], PauseInfo | None] | None = None


def install(state: PauseState | None) -> None:
    """Make *state* this process's answer (the daemon, at boot). ``None`` uninstalls (tests)."""
    global _installed, _file_cache
    _installed = state
    _file_cache = None


def installed() -> PauseState | None:
    return _installed


#: The daemon names its pause file to every child it builds (tasks/child_env.py), so a child never guesses.
CHILD_ENV = "PROMETHEUS_PAUSE_FILE"


def child_env_entry() -> dict[str, str]:
    """``{PROMETHEUS_PAUSE_FILE: <the pause file>}`` for a child's environment: the daemon's own file, or (in a
    child) the file it was told, so a grandchild reads the same file as its parent."""
    state = installed()
    if state is not None:
        return {CHILD_ENV: str(state.path)}
    return {CHILD_ENV: os.environ.get(CHILD_ENV) or str(pause_path())}


def current() -> PauseInfo | None:
    """Is the agent paused? The installed state, or (a child process) the file, re-read when it changes.

    A child the daemon built is told the exact file (:data:`CHILD_ENV`). If it cannot reach that file's FOLDER
    (missing or unreadable), it is paused: fail closed. A readable folder with no file means never paused. A folder
    hidden behind an empty mount looks readable and empty, so it reads as not paused: this check cannot tell, which
    is why the daemon's own state (and its cancelling the turns it runs) is the authority, not the file. A process
    that was not told (a CLI run) reads the default place, where a missing folder is not a pause.
    """
    global _file_cache
    if _installed is not None:
        return _installed.current
    told = os.environ.get(CHILD_ENV)
    path = Path(told) if told else pause_path()
    if told:
        try:
            os.stat(path.parent)
        except OSError as exc:
            return _unreadable(f"pause folder unreachable ({type(exc).__name__})")
        if not os.access(path.parent, os.R_OK | os.X_OK):
            return _unreadable("pause folder unreadable")
    try:
        st = os.stat(path)
    except FileNotFoundError:
        _file_cache = None
        return None
    except OSError as exc:
        return _unreadable(type(exc).__name__)
    key = (st.st_mtime_ns, st.st_size)
    if _file_cache is not None and _file_cache[0] == key:
        return _file_cache[1]
    info = read_file(path)
    _file_cache = (key, info)
    return info


def check() -> None:
    """Raise :class:`Paused` if the agent is paused."""
    info = current()
    if info is not None:
        raise Paused(info)


# ── the turns running in this process ────────────────────────────────────────
#
# Only tasks created FOR a turn are listed, never "whatever task is running a turn": a chat poller can run a
# handler, and so a turn, inline in its own long-lived task, and cancelling that would kill the poller (the one
# a person resumes through). A turn not started through one of these is stopped at run_loop's round check.

_turn_tasks: set[asyncio.Task[Any]] = set()


def _track(task: asyncio.Task[Any]) -> None:
    _turn_tasks.add(task)
    task.add_done_callback(_turn_tasks.discard)


async def run_as_turn(coro: Any) -> Any:
    """Run *coro* (one turn) in a task of its own that a pause may cancel. A pause surfaces to the caller as
    :class:`Paused`; the caller's own cancellation still cancels the turn and propagates as itself."""
    task: asyncio.Task[Any] = asyncio.ensure_future(coro)
    _track(task)
    try:
        return await task
    except asyncio.CancelledError:
        outer = asyncio.current_task()
        if task.cancelled() and (outer is None or outer.cancelling() == 0):
            raise Paused(current() or PauseInfo(since=time.time(), by="unknown")) from None
        raise


def track_current_task_as_turn() -> Any:
    """Mark the CURRENT task as one created for a single turn (a per-request handler, never a long-lived
    poller). Returns a token for :func:`untrack`."""
    task = asyncio.current_task()
    if task is not None:
        _turn_tasks.add(task)
    return task


def untrack(task: Any) -> None:
    if task is not None:
        _turn_tasks.discard(task)


def running_turn_tasks() -> list[asyncio.Task[Any]]:
    return [t for t in _turn_tasks if not t.done()]
