"""Non-LLM completion detectors for managed tasks.

Pure detection only: each coroutine resolves when its watched condition occurs
or the timeout elapses. They never touch ``TaskRecord`` or the ``SignalBus`` —
the supervisor (:class:`~prometheus.tasks.manager.BackgroundTaskManager`) owns
state transitions and signal emission. Three detection modes:

- process : ``await proc.wait()`` — lives in the manager (already event-driven).
- file    : :func:`watch_for_file` — watchdog.Observer + PatternMatchingEventHandler
            (inotify on Linux, FSEvents/kqueue on macOS). Event-driven.
- poll    : :func:`poll_until` — exponential-backoff fallback ONLY, for
            conditions you can neither ``wait()`` on nor watch.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

log = logging.getLogger(__name__)


def _first_glob_match(directory: str, pattern: str) -> str | None:
    """Return the first existing file in *directory* matching *pattern*, or None."""
    try:
        for p in sorted(Path(directory).glob(pattern)):
            if p.is_file():
                return str(p)
    except OSError:
        return None
    return None


async def watch_for_file(
    directory: str,
    pattern: str,
    *,
    timeout_seconds: float | None,
) -> str | None:
    """Resolve with the path of the first file matching *pattern* in *directory*.

    Non-recursive. Returns the matched path, or ``None`` on timeout. A pre-check
    runs *after* the observer is live, so a file that already exists (or appears
    during observer startup) is never missed. watchdog drives the observer on its
    own thread; the first match is bridged to this event loop via
    ``loop.call_soon_threadsafe``.
    """
    loop = asyncio.get_running_loop()
    found: asyncio.Event = asyncio.Event()
    matched: list[str] = []

    def _record(path: str | None) -> None:
        if matched or not path:
            return
        matched.append(path)
        loop.call_soon_threadsafe(found.set)

    class _Handler(PatternMatchingEventHandler):
        def on_created(self, event):  # type: ignore[override]
            if not event.is_directory:
                _record(event.src_path)

        def on_modified(self, event):  # type: ignore[override]
            if not event.is_directory:
                _record(event.src_path)

        def on_closed(self, event):  # type: ignore[override]
            if not event.is_directory:
                _record(event.src_path)

        def on_moved(self, event):  # type: ignore[override]
            if not event.is_directory:
                _record(getattr(event, "dest_path", None) or event.src_path)

    handler = _Handler(patterns=[pattern], ignore_directories=True)
    observer = Observer()
    try:
        observer.schedule(handler, directory, recursive=False)
        observer.start()
    except (OSError, FileNotFoundError) as exc:
        # Directory missing/unwatchable — fall back to a single glob check.
        log.warning("watch_for_file: cannot watch %s: %s", directory, exc)
        return _first_glob_match(directory, pattern)

    try:
        existing = _first_glob_match(directory, pattern)
        if existing:
            return existing
        if timeout_seconds and timeout_seconds > 0:
            try:
                await asyncio.wait_for(found.wait(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                return None
        else:
            await found.wait()
        return matched[0] if matched else None
    finally:
        observer.stop()
        # Thread.join is blocking — run it off the event loop.
        try:
            await loop.run_in_executor(None, observer.join, 2.0)
        except Exception:
            pass


async def poll_until(
    predicate_cmd: str,
    *,
    cwd: str,
    timeout_seconds: float | None,
    initial_interval: float,
    max_interval: float,
    workspaces: Sequence[str] | None = None,
) -> bool:
    """Run *predicate_cmd* on exponential backoff until it exits 0, or timeout.

    Fallback detector ONLY. Returns True on success, False on timeout. The
    command is assumed already vetted by the supervisor's SecurityGate at
    registration — this function does not gate.

    It DOES floor: the predicate is the model's command, so every run goes
    behind the bash tool's read and write floors with its scrubbed
    environment. A floor that is required and unavailable raises
    ``ShellFloorRefused`` out of here — it is not a predicate that failed.
    """
    loop = asyncio.get_running_loop()
    deadline = (loop.time() + timeout_seconds) if timeout_seconds else None
    interval = max(0.5, float(initial_interval))

    while True:
        if await _run_predicate(predicate_cmd, cwd, workspaces) == 0:
            return True
        now = loop.time()
        if deadline is not None and now >= deadline:
            return False
        sleep_for = interval
        if deadline is not None:
            sleep_for = min(sleep_for, max(0.0, deadline - now))
            if sleep_for <= 0:
                return False
        await asyncio.sleep(sleep_for)
        interval = min(interval * 2, float(max_interval))


async def _run_predicate(
    command: str, cwd: str, workspaces: Sequence[str] | None = None,
) -> int:
    """Run *command* in bash, discard output, return its exit code (1 on error).

    The argv and environment are built OUTSIDE the try below, so a refusal by
    the floor propagates instead of being counted as one more failed poll.
    """
    from prometheus.security.shell_floor import floored_argv, model_shell_env

    argv, _floors = floored_argv(command, cwd=cwd, workspaces=workspaces)
    env = model_shell_env()
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return proc.returncode if proc.returncode is not None else 1
    except Exception as exc:  # noqa: BLE001 — predicate failures are non-fatal
        log.warning("poll predicate failed to run: %s", exc)
        return 1
