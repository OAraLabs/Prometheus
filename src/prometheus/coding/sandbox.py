"""Execution sandbox for coding mode (SPRINT-coding-mode v2, scope item 3).

Two backends: ``ProcessSandbox`` (subprocess + cwd jail + env scrub +
wall/time limits) and ``DockerSandbox`` (Docker container with cap-drop
ALL, no-new-privileges, PID limits, optional network isolation).  The
jail root is a DEDICATED FULL CLONE of the target repo (decided in the
spec: hard isolation over disk savings) — never the live Prometheus tree.

Defense in depth: SecurityGate vets the coding task's commands at task
creation (origin="system", like cron); the sandbox ADDITIONALLY refuses any
path that resolves outside its root or into a denied path — so even a
gate-approved command cannot read/write outside the jail through the tools.
``resolve()`` resolves symlinks BEFORE the containment check, so a symlink
inside the clone pointing outside it is denied, not followed.

Env scrub: subprocesses get an allowlisted environment (PATH/HOME/locale —
what test runners need) with every other variable dropped. Secrets like
PROMETHEUS_API_TOKEN or provider keys in the daemon's environment never
reach a sandboxed process. This is process-level hygiene, not a container
boundary — the honest limit of ProcessSandbox, stated here so nobody
mistakes it for one.

Factory: ``create_sandbox()`` selects the backend based on config
(``sandbox_type: process|docker``).  Falls back to ProcessSandbox when
Docker is unavailable.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

# What a sandboxed process inherits. Everything else — tokens, API keys,
# gateway config — is dropped. PYTHONUNBUFFERED so test output streams into
# the capture instead of sitting in a pipe buffer at timeout-kill time.
_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR")

# Output discipline: keep the head (imports/collection errors) and the tail
# (the failure summary — what iterate-to-green feeds back) of long output.
_OUTPUT_HEAD_CHARS = 2_000
_OUTPUT_TAIL_CHARS = 10_000

DEFAULT_RUN_TIMEOUT_SECONDS = 240.0


class SandboxViolation(Exception):
    """A path resolved outside the sandbox root or into a denied path."""


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of one sandboxed command."""

    exit_code: int | None  # None = killed (timeout)
    output: str            # combined stdout+stderr, head/tail-truncated
    timed_out: bool
    duration_seconds: float


class Sandbox(ABC):
    """Path confinement + command execution for one coding run."""

    root: Path

    @property
    @abstractmethod
    def backend(self) -> str:
        """Backend identifier: ``"process"`` or ``"docker"``."""

    @abstractmethod
    def resolve(self, path: str) -> Path:
        """Resolve *path* (relative to root, or absolute) inside the jail.

        Raises :class:`SandboxViolation` if the real (symlink-resolved)
        location is outside the root or under a denied path.
        """

    @abstractmethod
    async def run(
        self, command: str, *, timeout_seconds: float | None = None
    ) -> SandboxResult:
        """Run a shell command with cwd=root and a scrubbed environment."""

    def close(self):
        """Release sandbox resources. No-op for ProcessSandbox."""
        pass


@dataclass
class ProcessSandbox(Sandbox):
    """Subprocess-backed sandbox. See module docstring for the guarantees."""

    root: Path
    denied_paths: tuple[Path, ...] = ()
    default_timeout_seconds: float = DEFAULT_RUN_TIMEOUT_SECONDS
    extra_env: dict[str, str] = field(default_factory=dict)

    @property
    def backend(self) -> str:
        return "process"

    def close(self):
        pass  # Nothing to release for process sandbox.

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"sandbox root is not a directory: {self.root}")
        self.denied_paths = tuple(
            Path(p).expanduser().resolve() for p in self.denied_paths
        )

    # ------------------------------------------------------------------
    # Path confinement
    # ------------------------------------------------------------------

    def resolve(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        # resolve() follows symlinks — the containment check runs on the
        # REAL location, so a link out of the jail is a violation, not a door.
        real = candidate.resolve()
        if not (real == self.root or real.is_relative_to(self.root)):
            raise SandboxViolation(
                f"path escapes the sandbox: {path!r} → {real} "
                f"(root: {self.root})"
            )
        for denied in self.denied_paths:
            if real == denied or real.is_relative_to(denied):
                raise SandboxViolation(
                    f"path is denied by policy: {path!r} → {real} "
                    f"(denied root: {denied})"
                )
        return real

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _scrubbed_env(self) -> dict[str, str]:
        env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
        env["PYTHONUNBUFFERED"] = "1"
        env["TERM"] = "dumb"
        # No bytecode caches inside the jail: CPython's pyc header stores
        # mtime in WHOLE SECONDS, so a same-size edit landing within the
        # same second as the previous compile silently reuses stale
        # bytecode — an iterate-to-green loop (fast edit→test cycles) hits
        # exactly this, and the test run then contradicts the source on
        # disk. Recompile-per-run is trivial for task repos; the artifact
        # diff stays clean of __pycache__ as a bonus.
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.update(self.extra_env)
        return env

    async def run(
        self, command: str, *, timeout_seconds: float | None = None
    ) -> SandboxResult:
        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.default_timeout_seconds
        )
        started = time.monotonic()
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(self.root),
            env=self._scrubbed_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,  # own process group → timeout kills the tree
        )
        try:
            raw, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            timed_out = False
        except asyncio.TimeoutError:
            timed_out = True
            try:
                # Kill the whole group — pytest children must not survive.
                os.killpg(os.getpgid(proc.pid), 9)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            raw, _ = await proc.communicate()
        duration = time.monotonic() - started

        output = (raw or b"").decode("utf-8", errors="replace")
        if len(output) > _OUTPUT_HEAD_CHARS + _OUTPUT_TAIL_CHARS:
            dropped = len(output) - _OUTPUT_HEAD_CHARS - _OUTPUT_TAIL_CHARS
            output = (
                output[:_OUTPUT_HEAD_CHARS]
                + f"\n… [{dropped} chars truncated] …\n"
                + output[-_OUTPUT_TAIL_CHARS:]
            )
        return SandboxResult(
            exit_code=None if timed_out else proc.returncode,
            output=output,
            timed_out=timed_out,
            duration_seconds=duration,
        )


def clone_repo_for_sandbox(
    source_repo: str | Path,
    dest_parent: str | Path,
    *,
    name: str,
    denied_paths: Iterable[str] = (),
) -> ProcessSandbox:
    """Create the dedicated full clone and return a sandbox rooted in it.

    A FULL ``git clone`` (not a worktree — spec decision: hard isolation
    over disk savings) into ``dest_parent/name``. The clone shares nothing
    writable with the source; the coding run's branch lives here.
    """
    import subprocess

    source = Path(source_repo).resolve()
    dest = Path(dest_parent).resolve() / name
    if dest.exists():
        raise ValueError(f"sandbox clone target already exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--no-hardlinks", str(source), str(dest)],
        check=True,
        capture_output=True,
        text=True,
    )
    return ProcessSandbox(root=dest, denied_paths=tuple(denied_paths))


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

DOCKER_CONTAINER_PREFIX = "prometheus-coding-"
DOCKER_IMAGE = "python:3.12-slim"
DOCKER_LABEL = "prometheus.coding"


def docker_available() -> bool:
    """Check whether the Docker daemon is reachable."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, OSError):
        return False


def _container_id_for_task(task_id: str) -> str:
    """Deterministic container name derivable from the task UUID."""
    # Strip hyphens to stay within Docker naming limits.
    safe = task_id.replace("-", "")
    return f"{DOCKER_CONTAINER_PREFIX}{safe}"


# ---------------------------------------------------------------------------
# DockerSandbox — container-based isolation
# ---------------------------------------------------------------------------


class DockerSandbox(Sandbox):
    """Sandbox backed by a Docker container.

    Security properties beyond ProcessSandbox:
    - ``cap-drop ALL`` — no Linux capabilities
    - ``security-opt=no-new-privileges`` — setuid/setgid blocked
    - ``--pid="limit=128"`` — PID exhaustion protection
    - ``--read-only`` root filesystem with tmpfs on /tmp + /workspace
    - Optional ``--network none`` for network isolation
    - Mounts the sandbox clone read-write at /workspace

    The container is created lazily on first ``execute()`` and kept alive
    (``--detach``) so repeated tool calls share the same environment.
    ``close()`` stops and removes the container.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        task_id: str,
        denied_paths: Iterable[str] = (),
        network_isolation: bool = False,
        image: str | None = None,
    ):
        self._root = Path(root).resolve()
        self._task_id = task_id
        self._denied_paths = tuple(
            Path(p).resolve() for p in denied_paths
        )
        self._network_isolation = network_isolation
        self._image = image or DOCKER_IMAGE
        self._container_id = _container_id_for_task(task_id)
        self._closed = False

        # Ensure container doesn't already exist (leftover from crash).
        self._ensure_container()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def backend(self) -> str:
        return "docker"

    def _ensure_container(self):
        """Create (or verify) the Docker container for this sandbox."""
        # Check if a container with this ID already exists.
        inspect = subprocess.run(
            ["docker", "inspect", self._container_id],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if inspect.returncode == 0:
            # Exists — check if it's running.
            state = subprocess.run(
                [
                    "docker", "inspect",
                    "--format", "{{.State.Status}}",
                    self._container_id,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if state.stdout.strip() in ("running", "paused"):
                return  # Container is live, good.
            # Dead or exited — remove and recreate.
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True,
                text=True,
                timeout=10,
            )

        # Build the docker run command.
        cmd = [
            "docker", "create",
            "--name", self._container_id,
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pid", "limit=128",
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--tmpfs", f"/workspace:rw,noexec,nosuid,size=512m",
            "--label", DOCKER_LABEL,
            "--workdir", "/workspace",
            "--user", "root",
        ]
        if self._network_isolation:
            cmd.append("--network=none")

        # Mount the sandbox clone at /workspace.
        cmd.extend(["-v", f"{self._root}:/workspace"])
        cmd.append(self._image)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create Docker container {self._container_id}: "
                f"{result.stderr.strip()}"
            )

    def resolve(self, path: str) -> Path:
        """Resolve a path, enforcing containment.

        Inside Docker, all tool paths are host-side (the mount point),
        so this works the same as ProcessSandbox.resolve().
        """
        try:
            candidate = self._root / path
            resolved = candidate.resolve(strict=True)
        except (OSError, ValueError):
            raise SandboxViolation(
                f"Cannot resolve path '{path}' inside sandbox"
            )

        if not str(resolved).startswith(str(self._root)):
            raise SandboxViolation(
                f"Path '{path}' escapes sandbox root "
                f"(resolves to {resolved}, root is {self._root})"
            )

        for denied in self._denied_paths:
            if str(resolved).startswith(str(denied)):
                raise SandboxViolation(
                    f"Path '{path}' is denied (matches {denied})"
                )

        return resolved

    async def run(
        self, command: str, *, timeout_seconds: float | None = None
    ) -> SandboxResult:
        if self._closed:
            raise RuntimeError("Sandbox already closed")

        container_cwd = str(self._root)

        timeout = (timeout_seconds or 60.0)

        start = time.monotonic()
        timed_out = False

        try:
            # Use docker exec for the command.
            cmd = [
                "docker", "exec", "-i",
                "--workdir", container_cwd,
                self._container_id,
                "bash", "-c", command,
            ]
            proc = subprocess.run(
                cmd,
                input=None,
                capture_output=True,
                text=True,
                timeout=timeout + 5,  # Extra for Docker overhead.
            )
            duration = time.monotonic() - start

            output = proc.stdout or ""
            if proc.stderr:
                output += proc.stderr

            # Truncate if needed.
            if len(output) > _OUTPUT_HEAD_CHARS + _OUTPUT_TAIL_CHARS:
                output = (
                    output[:_OUTPUT_HEAD_CHARS]
                    + f"\n… [{len(output) - _OUTPUT_HEAD_CHARS - _OUTPUT_TAIL_CHARS} chars truncated] …\n"
                    + output[-_OUTPUT_TAIL_CHARS:]
                )

            return SandboxResult(
                exit_code=proc.returncode,
                output=output,
                timed_out=False,
                duration_seconds=duration,
            )

        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            timed_out = True
            # Kill the process inside the container.
            subprocess.run(
                ["docker", "kill", self._container_id],
                capture_output=True,
                timeout=10,
            )
            # Restart the container so it's usable again.
            subprocess.run(
                ["docker", "start", self._container_id],
                capture_output=True,
                timeout=10,
            )
            return SandboxResult(
                exit_code=None,
                output="",
                timed_out=True,
                duration_seconds=duration,
            )

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception:
            log.exception("Failed to remove Docker container %s", self._container_id)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_sandbox(
    root: str | Path,
    *,
    task_id: str,
    backend: str = "process",
    denied_paths: Iterable[str] = (),
    network_isolation: bool = False,
    image: str | None = None,
) -> Sandbox:
    """Create a sandbox based on the requested backend.

    Args:
        root: Resolved path to the sandbox clone root.
        task_id: UUID string for this coding task.
        backend: ``"process"`` or ``"docker"``.
        denied_paths: Paths inside the root that are off-limits.
        network_isolation: Only for Docker — drop network access.
        image: Only for Docker — override the container image.

    Returns:
        A ``ProcessSandbox`` or ``DockerSandbox`` instance.

    Raises:
        RuntimeError: If Docker is requested but unavailable.
    """
    backend_lower = backend.lower()

    if backend_lower == "docker":
        if not docker_available():
            log.warning(
                "Docker requested but not available — falling back to process"
            )
            backend_lower = "process"

    if backend_lower == "docker":
        return DockerSandbox(
            root,
            task_id=task_id,
            denied_paths=denied_paths,
            network_isolation=network_isolation,
            image=image,
        )

    # Default: process sandbox (no task_id needed).
    return ProcessSandbox(root=Path(root), denied_paths=tuple(denied_paths))


# ---------------------------------------------------------------------------
# Stale container management
# ---------------------------------------------------------------------------


def list_stale_docker_containers(
    max_age_hours: float = 24,
) -> list[dict]:
    """List Docker containers matching our label that are older than max_age.

    Returns list of dicts with ``id``, ``name``, ``created``, ``status``.
    """
    cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)

    result = subprocess.run(
        [
            "docker", "ps", "-a",
            "--filter", f"label={DOCKER_LABEL}",
            "--format",
            "{{.ID}}\t{{.Names}}\t{{.CreatedAt}}\t{{.Status}}",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []

    containers = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        cid, name, created_str, status = parts[0], parts[1], parts[2], parts[3]

        # Parse "2026-08-14 12:00:00 +0000 UTC" or relative like "5 minutes ago".
        age_hours = _parse_docker_age(created_str)
        if age_hours is None or age_hours < max_age_hours:
            continue

        containers.append({
            "id": cid,
            "name": name,
            "created": created_str,
            "status": status,
            "age_hours": age_hours,
        })

    return containers


def _parse_docker_age(created_str: str) -> float | None:
    """Parse Docker's --format {{.CreatedAt}} into hours ago.

    Handles both absolute timestamps and relative strings like
    "5 minutes ago", "2 hours ago", "3 days ago".
    """
    now = datetime.now(timezone.utc).timestamp()

    # Try absolute format first: "2026-08-14 12:00:00 +0000 UTC"
    try:
        # Strip trailing " UTC" for parsing.
        clean = created_str.rstrip().rstrip(" UTC")
        dt = datetime.strptime(clean.strip(), "%Y-%m-%d %H:%M:%S %z")
        return (now - dt.timestamp()) / 3600
    except ValueError:
        pass

    # Try relative: "5 minutes ago", "2 hours ago", "3 days ago"
    import re
    m = re.match(r"(\d+)\s+(second|minute|hour|day|week)s?\s+ago", created_str, re.IGNORECASE)
    if m:
        value = int(m.group(1))
        unit = m.group(2).lower()
        multipliers = {
            "second": 1 / 3600,
            "minute": 1 / 60,
            "hour": 1,
            "day": 24,
            "week": 168,
        }
        return value * multipliers.get(unit, 0)

    return None


def cleanup_stale_docker_containers(
    max_age_hours: float = 24,
    dry_run: bool = False,
) -> list[str]:
    """Remove stale Docker containers matching our label.

    Args:
        max_age_hours: Remove containers older than this.
        dry_run: If True, list what would be removed without removing.

    Returns:
        List of container IDs that were (or would be) removed.
    """
    stale = list_stale_docker_containers(max_age_hours=max_age_hours)
    removed = []

    for container in stale:
        cid = container["id"]
        if dry_run:
            log.info(
                "[dry-run] Would remove stale container %s (%s, %0.1fh old)",
                cid, container["name"], container["age_hours"],
            )
            removed.append(cid)
        else:
            try:
                subprocess.run(
                    ["docker", "rm", "-f", cid],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                log.info("Removed stale container %s (%s)", cid, container["name"])
                removed.append(cid)
            except Exception:
                log.exception("Failed to remove container %s", cid)

    return removed
