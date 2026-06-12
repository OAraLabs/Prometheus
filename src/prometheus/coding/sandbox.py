"""Execution sandbox for coding mode (SPRINT-coding-mode v2, scope item 3).

One interface, ``ProcessSandbox`` backend now: subprocess + cwd jail + env
scrub + wall/time limits. The jail root is a DEDICATED FULL CLONE of the
target repo (decided in the spec: hard isolation over disk savings) — never
the live Prometheus tree. ``DockerSandbox`` is interface-shaped future work;
this module is the interface it will implement.

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
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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


@dataclass
class ProcessSandbox(Sandbox):
    """Subprocess-backed sandbox. See module docstring for the guarantees."""

    root: Path
    denied_paths: tuple[Path, ...] = ()
    default_timeout_seconds: float = DEFAULT_RUN_TIMEOUT_SECONDS
    extra_env: dict[str, str] = field(default_factory=dict)

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
