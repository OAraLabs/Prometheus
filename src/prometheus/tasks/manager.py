"""Background task manager (managed-tasks sprint).

Owns the full lifecycle of background tasks across three detection modes:

- ``local_bash`` / ``local_agent`` — subprocess; completion via ``proc.wait()``.
- ``file_watch`` — a watchdog observer; completion on first matching file.
- ``poll`` — exponential-backoff predicate; fallback detector only.

On top of the original in-memory manager this adds, all backward-compatibly
(every new dependency is optional and defaults off):

- SecurityGate vetting at launch (origin="system", like cron) — closes the
  unvetted-bash gap and covers restart resume.
- A durable :class:`~prometheus.tasks.store.TaskStore` so ``running`` tasks can
  be resumed (file_watch/poll) or reaped (process) across a daemon restart.
- ``task_completed`` / ``task_failed`` SignalBus emission on resolution, so the
  completion handler (re-engagement) fires event-driven. Notification stays in
  the heartbeat.
- ``timeout_seconds`` enforcement on every mode.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import signal
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from prometheus.config.paths import get_tasks_dir
from prometheus.tasks.types import (
    TERMINAL_STATUSES,
    OnComplete,
    TaskRecord,
    TaskStatus,
    TaskType,
)
from prometheus.tasks.watchers import poll_until, watch_for_file

if TYPE_CHECKING:
    from prometheus.tasks.store import TaskStore

log = logging.getLogger(__name__)


def _signal_process_group(process: asyncio.subprocess.Process, sig: int) -> None:
    """Send ``sig`` to the task shell's whole process group (it + its children).

    Tasks are launched with ``start_new_session=True``, so the shell leads its
    own group; signalling the group kills the actual workload (a ``wget`` etc.),
    not just ``/bin/bash``. Best-effort and idempotent: falls back to signalling
    the process directly, and ignores a process/group that has already exited.
    """
    if process.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except (ProcessLookupError, PermissionError):
        try:
            process.send_signal(sig)
        except (ProcessLookupError, ValueError):
            pass


# Defaults; the daemon overrides from config["tasks"].
DEFAULT_TIMEOUT_SECONDS = 3600
DEFAULT_POLL_INITIAL_INTERVAL = 5.0
DEFAULT_POLL_MAX_INTERVAL = 120.0


class BackgroundTaskManager:
    """Manage shell, agent, file-watch and poll tasks with full lifecycle tracking."""

    def __init__(
        self,
        *,
        signal_bus: Any | None = None,
        security_gate: Any | None = None,
        store: "TaskStore | None" = None,
        poll_initial_interval: float = DEFAULT_POLL_INITIAL_INTERVAL,
        poll_max_interval: float = DEFAULT_POLL_MAX_INTERVAL,
        default_timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._waiters: dict[str, asyncio.Task[None]] = {}
        self._output_locks: dict[str, asyncio.Lock] = {}
        self._input_locks: dict[str, asyncio.Lock] = {}
        self._generations: dict[str, int] = {}
        self._emitted: set[str] = set()

        # Optional collaborators (wired by the daemon; None in lightweight use).
        self.signal_bus = signal_bus
        self.security_gate = security_gate
        self.store = store
        self.poll_initial_interval = poll_initial_interval
        self.poll_max_interval = poll_max_interval
        self.default_timeout_seconds = default_timeout_seconds

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    async def create_shell_task(
        self,
        *,
        command: str,
        description: str,
        cwd: str | Path,
        task_type: TaskType = "local_bash",
        session_id: str | None = None,
        notify_target: str | None = None,
        on_complete: OnComplete = "notify",
        reengage_prompt: str | None = None,
        timeout_seconds: int | None = None,
    ) -> TaskRecord:
        """Start a background shell command and return its TaskRecord.

        Vets *command* through the SecurityGate at system trust before spawning.
        A denied command yields a ``failed`` record (error ``"blocked: ..."``)
        and NO process is launched.
        """
        blocked = self._vet_command(command)
        record = self._new_record(
            task_type=task_type,
            description=description,
            cwd=cwd,
            command=command,
            session_id=session_id,
            notify_target=notify_target,
            on_complete=on_complete,
            reengage_prompt=reengage_prompt,
            timeout_seconds=timeout_seconds,
        )
        if blocked is not None:
            record.status = "failed"
            record.error = blocked
            record.ended_at = time.time()
            self._tasks[record.id] = record
            self._persist(record)
            log.warning("Task %s rejected at register: %s", record.id, blocked)
            return record

        record.output_file.write_text("", encoding="utf-8")
        self._tasks[record.id] = record
        self._output_locks[record.id] = asyncio.Lock()
        self._input_locks[record.id] = asyncio.Lock()
        self._persist(record)
        await self._start_process(record.id)
        return record

    async def create_agent_task(
        self,
        *,
        prompt: str,
        description: str,
        cwd: str | Path,
        task_type: TaskType = "local_agent",
        model: str | None = None,
        api_key: str | None = None,
        command: str | None = None,
        session_id: str | None = None,
        notify_target: str | None = None,
        on_complete: OnComplete = "notify",
        reengage_prompt: str | None = None,
        timeout_seconds: int | None = None,
    ) -> TaskRecord:
        """Start a Prometheus agent as a subprocess task."""
        if command is None:
            effective_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not effective_api_key:
                raise ValueError(
                    "Local agent tasks require ANTHROPIC_API_KEY or an explicit command override"
                )
            cmd = ["python", "-m", "prometheus", "--headless", "--api-key", effective_api_key]
            if model:
                cmd.extend(["--model", model])
            command = " ".join(shlex.quote(part) for part in cmd)

        record = await self.create_shell_task(
            command=command,
            description=description,
            cwd=cwd,
            task_type=task_type,
            session_id=session_id,
            notify_target=notify_target,
            on_complete=on_complete,
            reengage_prompt=reengage_prompt,
            timeout_seconds=timeout_seconds,
        )
        if record.status == "failed":
            return record
        updated = replace(record, prompt=prompt)
        self._tasks[record.id] = updated
        self._persist(updated)
        await self.write_to_task(record.id, prompt)
        return updated

    async def create_file_watch_task(
        self,
        *,
        watch_dir: str,
        watch_pattern: str,
        description: str,
        cwd: str | Path,
        session_id: str | None = None,
        notify_target: str | None = None,
        on_complete: OnComplete = "notify",
        reengage_prompt: str | None = None,
        timeout_seconds: int | None = None,
    ) -> TaskRecord:
        """Watch *watch_dir* for the first file matching *watch_pattern*."""
        record = self._new_record(
            task_type="file_watch",
            description=description,
            cwd=cwd,
            session_id=session_id,
            notify_target=notify_target,
            on_complete=on_complete,
            reengage_prompt=reengage_prompt,
            timeout_seconds=timeout_seconds,
            spec={"dir": str(watch_dir), "pattern": watch_pattern},
        )
        record.output_file.write_text("", encoding="utf-8")
        self._tasks[record.id] = record
        self._persist(record)
        self._waiters[record.id] = asyncio.create_task(self._run_file_watch(record.id))
        return record

    async def create_poll_task(
        self,
        *,
        poll_predicate: str,
        description: str,
        cwd: str | Path,
        session_id: str | None = None,
        notify_target: str | None = None,
        on_complete: OnComplete = "notify",
        reengage_prompt: str | None = None,
        timeout_seconds: int | None = None,
    ) -> TaskRecord:
        """Poll *poll_predicate* (a shell command) until it exits 0, or timeout."""
        blocked = self._vet_command(poll_predicate)
        record = self._new_record(
            task_type="poll",
            description=description,
            cwd=cwd,
            session_id=session_id,
            notify_target=notify_target,
            on_complete=on_complete,
            reengage_prompt=reengage_prompt,
            timeout_seconds=timeout_seconds,
            spec={"predicate_cmd": poll_predicate},
        )
        if blocked is not None:
            record.status = "failed"
            record.error = blocked
            record.ended_at = time.time()
            self._tasks[record.id] = record
            self._persist(record)
            log.warning("Poll task %s rejected at register: %s", record.id, blocked)
            return record
        record.output_file.write_text("", encoding="utf-8")
        self._tasks[record.id] = record
        self._persist(record)
        self._waiters[record.id] = asyncio.create_task(self._run_poll(record.id))
        return record

    # ------------------------------------------------------------------
    # Queries / mutation (unchanged public surface)
    # ------------------------------------------------------------------

    def _load_task(self, task_id: str) -> TaskRecord | None:
        """Resolve a task by id: in-memory first, then the durable TaskStore.

        Restart survival: a task that finished BEFORE this daemon lifetime is
        durably in tasks.db (``_watch_process`` persists the terminal record)
        but absent from ``_tasks`` (``resume_running`` only re-reads *running*
        tasks). Falling back to ``store.get`` rehydrates completed/killed/failed
        tasks ON DEMAND — bounded, unlike eagerly loading every historical row
        at startup — and caches the hit so ``stop_task`` and repeat reads
        resolve without re-querying. Returns None only when the id is unknown
        to BOTH the live map and the durable store (a genuine 404).
        """
        task = self._tasks.get(task_id)
        if task is not None:
            return task
        if self.store is None:
            return None
        try:
            rec = self.store.get(task_id)
        except Exception:
            log.warning("TaskStore.get failed for %s", task_id, exc_info=True)
            return None
        if rec is not None:
            self._tasks[task_id] = rec  # cache the rehydrated record
        return rec

    def get_task(self, task_id: str) -> TaskRecord | None:
        """Return one task record by ID (durable-store fallback; see _load_task)."""
        return self._load_task(task_id)

    def list_tasks(self, *, status: TaskStatus | None = None) -> list[TaskRecord]:
        """Return all tasks, newest first, optionally filtered by status."""
        tasks = list(self._tasks.values())
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def update_task(
        self,
        task_id: str,
        *,
        description: str | None = None,
        progress: int | None = None,
        status_note: str | None = None,
    ) -> TaskRecord:
        """Update mutable task metadata (description, progress, status note)."""
        task = self._require_task(task_id)
        if description is not None and description.strip():
            task.description = description.strip()
        if progress is not None:
            task.metadata["progress"] = str(progress)
        if status_note is not None:
            note = status_note.strip()
            if note:
                task.metadata["status_note"] = note
            else:
                task.metadata.pop("status_note", None)
        self._persist(task)
        return task

    async def stop_task(self, task_id: str) -> TaskRecord:
        """Terminate a running task (SIGTERM then SIGKILL, or cancel a watcher)."""
        task = self._require_task(task_id)
        process = self._processes.get(task_id)

        if process is None:
            if task.status in TERMINAL_STATUSES:
                return task
            # file_watch / poll: cancel the watcher coroutine.
            waiter = self._waiters.get(task_id)
            if waiter is not None and not waiter.done():
                waiter.cancel()
                task.status = "killed"
                task.ended_at = time.time()
                self._waiters.pop(task_id, None)
                self._persist(task)
                await self._emit_completion(task)
                return task
            raise ValueError(f"Task {task_id} is not running")

        _signal_process_group(process, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except asyncio.TimeoutError:
            _signal_process_group(process, signal.SIGKILL)
            await process.wait()

        task.status = "killed"
        task.ended_at = time.time()
        # _watch_process sees the killed status and emits + persists.
        return task

    async def write_to_task(self, task_id: str, data: str) -> None:
        """Write a line to the task's stdin."""
        task = self._require_task(task_id)
        async with self._input_locks[task_id]:
            process = await self._ensure_writable_process(task)
            process.stdin.write((data.rstrip("\n") + "\n").encode("utf-8"))
            try:
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                if task.type not in {"local_agent", "remote_agent", "in_process_teammate"}:
                    raise ValueError(f"Task {task_id} does not accept input") from None
                process = await self._restart_agent_task(task)
                process.stdin.write((data.rstrip("\n") + "\n").encode("utf-8"))
                await process.stdin.drain()

    def read_task_output(self, task_id: str, *, max_bytes: int = 12000) -> str:
        """Return the tail of a task's output log."""
        task = self._require_task(task_id)
        try:
            content = task.output_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        if len(content) > max_bytes:
            return content[-max_bytes:]
        return content

    # ------------------------------------------------------------------
    # Durability across restart
    # ------------------------------------------------------------------

    async def resume_running(self) -> None:
        """Resume or reap tasks left ``running`` by a previous daemon process.

        - file_watch / poll : the detector is stateless, so re-create the watcher.
        - process tasks      : the OS process handle is gone — reap to ``failed``
          (error ``"daemon_restart"``) and emit, so no zombie ``running`` rows
          remain.
        """
        if self.store is None:
            return
        try:
            stale = self.store.list(status="running")
        except Exception:
            log.warning("resume_running: TaskStore.list failed", exc_info=True)
            return

        for rec in stale:
            self._tasks[rec.id] = rec
            if rec.type == "file_watch":
                self._waiters[rec.id] = asyncio.create_task(self._run_file_watch(rec.id))
                log.info("Resumed file_watch task %s", rec.id)
            elif rec.type == "poll":
                self._waiters[rec.id] = asyncio.create_task(self._run_poll(rec.id))
                log.info("Resumed poll task %s", rec.id)
            else:
                rec.status = "failed"
                rec.error = "daemon_restart"
                rec.ended_at = time.time()
                self._persist(rec)
                await self._emit_completion(rec)
                log.info("Reaped orphaned process task %s (daemon_restart)", rec.id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_task(self, task_id: str) -> TaskRecord:
        # Durable-store fallback (see _load_task): a completed/killed task from a
        # prior daemon lifetime resolves here too, so stop_task on it returns the
        # terminal record rather than raising.
        task = self._load_task(task_id)
        if task is None:
            raise ValueError(f"No task found with ID: {task_id}")
        return task

    def _new_record(
        self,
        *,
        task_type: TaskType,
        description: str,
        cwd: str | Path,
        command: str | None = None,
        session_id: str | None = None,
        notify_target: str | None = None,
        on_complete: OnComplete = "notify",
        reengage_prompt: str | None = None,
        timeout_seconds: int | None = None,
        spec: dict[str, Any] | None = None,
    ) -> TaskRecord:
        task_id = _task_id(task_type)
        effective_timeout = (
            timeout_seconds if timeout_seconds is not None else self.default_timeout_seconds
        )
        return TaskRecord(
            id=task_id,
            type=task_type,
            status="running",
            description=description,
            cwd=str(Path(cwd).resolve()),
            output_file=get_tasks_dir() / f"{task_id}.log",
            command=command,
            created_at=time.time(),
            started_at=time.time(),
            session_id=session_id,
            notify_target=notify_target,
            on_complete=on_complete,
            reengage_prompt=reengage_prompt,
            timeout_seconds=effective_timeout,
            spec=spec or {},
        )

    def _vet_command(self, command: str) -> str | None:
        """Return None if the command is allowed, else a ``"blocked: ..."`` reason.

        Vets at system trust (``origin="system"``), exactly like cron: anything
        other than an ALLOW verdict is blocked, since a background task has no
        human approver. Fails closed on any gate error.
        """
        if self.security_gate is None:
            return None
        try:
            decision = self.security_gate.evaluate("bash", command=command, origin="system")
        except Exception as exc:  # noqa: BLE001 — fail closed
            return f"blocked: security gate error ({exc})"
        action = getattr(decision, "action", "DENY")
        if action == "ALLOW":
            return None
        reason = getattr(decision, "reason", "") or f"SecurityGate verdict {action}"
        return f"blocked: {reason}"

    def _persist(self, record: TaskRecord) -> None:
        """Best-effort durable write — never raises into the task path."""
        if self.store is None:
            return
        try:
            self.store.upsert(record)
        except Exception:
            log.warning("TaskStore.upsert failed for %s", record.id, exc_info=True)

    async def _emit_completion(self, task: TaskRecord) -> None:
        """Emit ``task_completed`` / ``task_failed`` once per terminal task."""
        if self.signal_bus is None or task.status not in TERMINAL_STATUSES:
            return
        if task.id in self._emitted:
            return
        self._emitted.add(task.id)
        kind = "task_completed" if task.status == "completed" else "task_failed"
        try:
            from prometheus.sentinel.signals import ActivitySignal

            await self.signal_bus.emit(
                ActivitySignal(
                    kind=kind,
                    payload={
                        "task_id": task.id,
                        "type": task.type,
                        "status": task.status,
                        "description": task.description,
                        "return_code": task.return_code,
                        "exit_code": task.return_code,
                        "error": task.error,
                        "artifact_path": task.artifact_path,
                        "session_id": task.session_id,
                        "notify_target": task.notify_target,
                        "on_complete": task.on_complete,
                        "reengage_prompt": task.reengage_prompt,
                        "output_tail": self.read_task_output(task.id, max_bytes=4000),
                    },
                    source="task_supervisor",
                )
            )
        except Exception:
            log.warning("task signal emit failed for %s", task.id, exc_info=True)

    async def _start_process(self, task_id: str) -> asyncio.subprocess.Process:
        task = self._require_task(task_id)
        if task.command is None:
            raise ValueError(f"Task {task_id} has no command")
        generation = self._generations.get(task_id, 0) + 1
        self._generations[task_id] = generation
        # start_new_session=True: the task shell leads its own process group so
        # stop_task / a timeout can kill the WHOLE job (e.g. a `wget` spawned by
        # the command), not just /bin/bash — otherwise children orphan and run
        # on. Same fix as tools/builtin/bash.py.
        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-lc",
            task.command,
            cwd=task.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        self._processes[task_id] = process
        self._waiters[task_id] = asyncio.create_task(
            self._watch_process(task_id, process, generation)
        )
        return process

    async def _watch_process(
        self,
        task_id: str,
        process: asyncio.subprocess.Process,
        generation: int,
    ) -> None:
        reader = asyncio.create_task(self._copy_output(task_id, process))
        task = self._tasks[task_id]
        timed_out = False
        timeout = task.timeout_seconds
        try:
            if timeout and timeout > 0:
                return_code = await asyncio.wait_for(process.wait(), timeout=timeout)
            else:
                return_code = await process.wait()
        except asyncio.TimeoutError:
            timed_out = True
            _signal_process_group(process, signal.SIGKILL)
            return_code = await process.wait()
        await reader
        if self._generations.get(task_id) != generation:
            return
        task.return_code = return_code
        if timed_out:
            task.status = "failed"
            task.error = "timeout"
        elif task.status != "killed":
            task.status = "completed" if return_code == 0 else "failed"
        task.ended_at = time.time()
        self._processes.pop(task_id, None)
        self._waiters.pop(task_id, None)
        self._persist(task)
        await self._emit_completion(task)

    async def _run_file_watch(self, task_id: str) -> None:
        task = self._tasks[task_id]
        directory = task.spec.get("dir", "")
        pattern = task.spec.get("pattern", "*")
        try:
            matched = await watch_for_file(
                directory, pattern, timeout_seconds=task.timeout_seconds
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            matched = None
            task.error = f"watch error: {exc}"
        if self._tasks.get(task_id) is not task or task.status in TERMINAL_STATUSES:
            return
        if matched is not None:
            task.status = "completed"
            task.artifact_path = matched
        else:
            task.status = "failed"
            if task.error is None:
                task.error = "timeout"
        task.ended_at = time.time()
        self._waiters.pop(task_id, None)
        self._persist(task)
        await self._emit_completion(task)

    async def _run_poll(self, task_id: str) -> None:
        task = self._tasks[task_id]
        predicate = task.spec.get("predicate_cmd", "")
        try:
            ok = await poll_until(
                predicate,
                cwd=task.cwd,
                timeout_seconds=task.timeout_seconds,
                initial_interval=self.poll_initial_interval,
                max_interval=self.poll_max_interval,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            ok = False
            task.error = f"poll error: {exc}"
        if self._tasks.get(task_id) is not task or task.status in TERMINAL_STATUSES:
            return
        if ok:
            task.status = "completed"
        else:
            task.status = "failed"
            if task.error is None:
                task.error = "timeout"
        task.ended_at = time.time()
        self._waiters.pop(task_id, None)
        self._persist(task)
        await self._emit_completion(task)

    async def _copy_output(self, task_id: str, process: asyncio.subprocess.Process) -> None:
        if process.stdout is None:
            return
        while True:
            chunk = await process.stdout.read(4096)
            if not chunk:
                return
            async with self._output_locks[task_id]:
                with self._tasks[task_id].output_file.open("ab") as fh:
                    fh.write(chunk)

    async def _ensure_writable_process(self, task: TaskRecord) -> asyncio.subprocess.Process:
        process = self._processes.get(task.id)
        if process is not None and process.stdin is not None and process.returncode is None:
            return process
        if task.type not in {"local_agent", "remote_agent", "in_process_teammate"}:
            raise ValueError(f"Task {task.id} does not accept input")
        return await self._restart_agent_task(task)

    async def _restart_agent_task(self, task: TaskRecord) -> asyncio.subprocess.Process:
        if task.command is None:
            raise ValueError(f"Task {task.id} has no restart command")
        waiter = self._waiters.get(task.id)
        if waiter is not None and not waiter.done():
            await waiter
        restart_count = int(task.metadata.get("restart_count", "0")) + 1
        task.metadata["restart_count"] = str(restart_count)
        task.status = "running"
        task.started_at = time.time()
        task.ended_at = None
        task.return_code = None
        self._emitted.discard(task.id)
        self._persist(task)
        return await self._start_process(task.id)


# Singleton

_DEFAULT_MANAGER: BackgroundTaskManager | None = None
_DEFAULT_MANAGER_KEY: str | None = None


def get_task_manager() -> BackgroundTaskManager:
    """Return the process-wide singleton BackgroundTaskManager."""
    global _DEFAULT_MANAGER, _DEFAULT_MANAGER_KEY
    current_key = str(get_tasks_dir().resolve())
    if _DEFAULT_MANAGER is None or _DEFAULT_MANAGER_KEY != current_key:
        _DEFAULT_MANAGER = BackgroundTaskManager()
        _DEFAULT_MANAGER_KEY = current_key
    return _DEFAULT_MANAGER


def _task_id(task_type: TaskType) -> str:
    prefixes = {
        "local_bash": "b",
        "local_agent": "a",
        "remote_agent": "r",
        "in_process_teammate": "t",
        "file_watch": "w",
        "poll": "p",
    }
    return f"{prefixes.get(task_type, 'b')}{uuid4().hex[:8]}"
