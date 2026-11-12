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
  * :func:`turn_started` / :func:`turn_ended` keep a count of the asyncio tasks running a turn in this process,
    whatever started them (the bridge, ``/v1``, ``POST /api/chat``, a chat gateway, a subagent), so a pause can
    cancel every one.

Resuming is not decided here: who may resume (a person, never the bare API token) is the route's rule.

Source: novel code for Prometheus, 2026-11-12.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import Counter
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


class Paused(RuntimeError):
    """Raised where a turn would start (or go on) while the agent is paused."""

    def __init__(self, info: PauseInfo) -> None:
        super().__init__(f"Prometheus is paused (since {info.since:.0f}, by {info.by})")
        self.info = info


def pause_path() -> Path:
    from prometheus.config.paths import data_dir_path

    return data_dir_path() / PAUSE_FILENAME


def _unreadable(why: str) -> PauseInfo:
    logger.error("pause state unreadable (%s): treating the agent as PAUSED until a person resumes it", why)
    return PauseInfo(since=time.time(), by="unreadable", reason=f"pause state unreadable: {why}")


def read_file(path: Path) -> PauseInfo | None:
    """The state *path* holds: None when not paused (no file, or ``paused: false``); paused when unreadable."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        return _unreadable(type(exc).__name__)
    try:
        data = json.loads(raw)
    except ValueError:
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
    tmp = path.with_name(f".{path.name}.{os.urandom(4).hex()}.tmp")
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

    def pause(self, *, by: str, reason: str = "") -> tuple[PauseInfo, bool]:
        """Pause, or return the pause already in force. ``(info, new)``. Written to disk BEFORE it takes effect
        here, so a crash between the two leaves the daemon paused, never the other way round."""
        if self._info is not None:
            return self._info, False
        info = PauseInfo(since=time.time(), by=by, reason=(reason or "")[:500])
        _write_file(self.path, {"paused": True, **info.to_dict()})
        self._info = info
        return info, True

    def resume(self, *, by: str) -> PauseInfo | None:
        """Clear the pause; returns the one cleared, or None if there was none. Starts nothing."""
        cleared = self._info
        _write_file(self.path, {"paused": False, "resumed_at": time.time(), "resumed_by": by})
        self._info = None
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


def current() -> PauseInfo | None:
    """Is the agent paused? The installed state, or (a child process) the file, re-read when it changes."""
    global _file_cache
    if _installed is not None:
        return _installed.current
    path = pause_path()
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

_running: Counter[asyncio.Task[Any]] = Counter()


def turn_started() -> asyncio.Task[Any] | None:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return None
    if task is not None:
        _running[task] += 1
    return task


def turn_ended(task: asyncio.Task[Any] | None) -> None:
    if task is None:
        return
    _running[task] -= 1
    if _running[task] <= 0:
        del _running[task]


def running_turn_tasks() -> list[asyncio.Task[Any]]:
    return [t for t in _running if not t.done()]
