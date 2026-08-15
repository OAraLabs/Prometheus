"""Execution sandbox for coding mode (SPRINT-coding-mode v2, scope item 3).

THREE backends, weakest to strongest, all behind one interface and selected
by ``create_sandbox()`` / ``coding.sandbox_type``:

- ``ProcessSandbox`` — subprocess + cwd jail + env scrub + wall/time limits.
  Confines the FILE TOOLS only; a shell redirect to an absolute path escapes
  it. Honest default, not a container boundary.
- ``BwrapSandbox`` — kernel mount/pid namespace via bubblewrap. Closes the
  shell-redirect hole. Needs unprivileged user namespaces; see its docstring
  for a host finding that blocks it on Ubuntu 24.04 defaults.
- ``DockerSandbox`` — container isolation: cap-drop ALL, no-new-privileges,
  pids-limit, read-only rootfs, optional network isolation. Strongest, and
  the only one that also bounds the process tree's view of the OS itself.

The jail root is a DEDICATED FULL CLONE of the target repo (decided in the
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
import shutil
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


class SandboxConstructionError(Exception):
    """bwrap itself failed to start the sandboxed process.

    Distinct from a command that ran and exited nonzero (that is an ordinary
    ``SandboxResult`` with a real exit code — not an error). This fires only
    when the target command never started at all: bwrap's own namespace/bind
    setup failed first. Conflating the two would misreport "the command
    failed" for "nothing the caller asked for ever ran" — the exact shape
    Standing-Principles calls out: name what you can actually prove.
    """


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


# ---------------------------------------------------------------------------
# BwrapSandbox — kernel namespace containment
# ---------------------------------------------------------------------------

_BWRAP_BIN = "bwrap"

# Read-only binds every invocation needs so a normal toolchain (git, python3,
# a venv's interpreter, uv, pytest) resolves inside the namespace. --ro-bind-
# TRY, not --ro-bind: a directory absent on this distro (e.g. no /lib64) is
# silently skipped rather than failing construction.
_BASE_RO_BIND_DIRS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc")


def _path_dirs() -> tuple[str, ...]:
    """Existing, deduplicated directories from the current $PATH.

    Bound read-only alongside the OS base set so tools installed outside
    /usr — a uv/pip venv, ~/.local/bin — resolve inside the namespace without
    hardcoding a username or an install layout. Computed from the CALLING
    process's environment, not the scrubbed sandbox env: PATH is what the
    daemon actually has right now, which is what needs to be reachable.
    """
    seen: list[str] = []
    for d in os.environ.get("PATH", "").split(":"):
        if d and d not in seen and Path(d).is_dir():
            seen.append(d)
    return tuple(seen)


@dataclass(frozen=True)
class BwrapSelfCheck:
    """Whether bwrap can actually run a namespaced process on this host —
    not merely whether the binary is on PATH. See :meth:`BwrapSandbox.self_check`.
    """

    ok: bool
    detail: str
    isolated_ok: bool   # succeeded with network unshared (the default policy)
    networked_ok: bool  # succeeded with network left attached


@dataclass
class BwrapSandbox(ProcessSandbox):
    """``ProcessSandbox`` + kernel namespace containment on ``run()``.

    Inherits ``resolve()``, env scrub, and the dedicated-clone jail root
    unchanged — those are pure Python path logic and were never the gap (see
    ``audits/20260813T053000Z-kernel-sandbox-scoping.md``). The ONE thing
    this class changes is ``run()``: instead of a bare
    ``create_subprocess_shell(cmd, cwd=root)`` — cwd-only confinement, which a
    shell redirect to an absolute path escapes trivially — the command runs
    inside a ``bwrap`` mount+pid(+net) namespace where the jail root is the
    only writable location and everything outside it either does not exist in
    the mount table or is read-only. The acceptance test:

        sb.run("echo x > <outside the jail>")

    must FAIL to create the file. Under ``ProcessSandbox`` it does not
    (``rc=0``, file lands outside — proved by outcome, see the audit above).
    Under this class the shell's own ``open(2)`` fails before the question of
    "does bwrap have write access" is even reached, because the path is not
    there to open.

    Two things ``ProcessSandbox.run()`` never had, gained as a side effect of
    real namespace isolation rather than as separate features:

    - ``denied_paths`` entries INSIDE ``root`` are additionally bind-mounted
      read-only, shadowing the writable root bind at that path. Previously
      ``denied_paths`` was resolve()-only — a file TOOL honored it; a shell
      command in ``run()`` did not. Now the shell cannot write there either.
    - ``/dev`` is bwrap's minimal synthetic set (null/zero/random/urandom/tty)
      rather than the host's real ``/dev`` — ``ProcessSandbox`` never
      restricted this at all, since a bare subprocess sees the host's whole
      device tree.

    Network policy: ``isolate_network`` defaults to **True** — an untrusted
    coding run has no business reaching the network by default. Set False
    only for a task that genuinely needs it (a `uv sync`/`pip install` step);
    this is a plain constructor flag, never silently toggled by this class.

    ⚠ HOST FINDING, live and unresolved as of 2026-08-13 (deployment host):

    This class is implemented and unit-tested (argv construction, the
    self-check, the sentinel-based construction-failure detector), but its
    CORE containment claim — the acceptance test above actually succeeding —
    is **UNVERIFIED on this host**, and not for a reason any change here can
    fix. `kernel.apparmor_restrict_unprivileged_userns=1` (Ubuntu 24.04's
    default) blocks unprivileged `--unshare-user` UNLESS `--unshare-net` is
    also requested in the same invocation — confirmed by bisecting every
    individual `--unshare-*` flag combination bwrap accepts. Once network IS
    unshared, bwrap's own automatic loopback-interface bring-up then fails
    with `EPERM` on a `RTM_NEWADDR` netlink call — confirmed via `strace` to
    be a kernel-level refusal, NOT a dropped capability (`CAP_NET_ADMIN` is
    present in the calling process's capability BOUNDING set per
    `/proc/self/status`) and NOT a nested-sandbox artifact
    (`systemd-detect-virt` reports `none`; `/proc/1/comm` is `systemd`; this
    is the real host). Net result: as configured today, NO bwrap invocation —
    networked or not — completes for an unprivileged user on this box.

    The minimal, understood fix is a system-level security-setting change
    (loosen or disable `kernel.apparmor_restrict_unprivileged_userns`, or add
    a targeted AppArmor allowance for `/usr/bin/bwrap`), which this agent is
    not permitted to apply unilaterally — surfaced to Will, not worked around.
    `BwrapSandbox.self_check()` reproduces both halves of this finding
    on demand and is what the test suite skips on, loudly, rather than
    reporting a false green.
    """

    isolate_network: bool = True
    _bwrap_path: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        found = shutil.which(_BWRAP_BIN)
        if not found:
            raise RuntimeError(
                "BwrapSandbox requires the 'bwrap' (bubblewrap) binary; "
                "none found on PATH. Install bubblewrap, or use "
                "ProcessSandbox if kernel containment is not required."
            )
        self._bwrap_path = found

    @staticmethod
    def is_available() -> bool:
        """bwrap is on PATH.

        Deliberately NOT the same claim as :meth:`self_check` — a host can
        have a perfectly good bwrap binary and still be unable to use it
        (this host is the reason the two are kept separate).
        """
        return shutil.which(_BWRAP_BIN) is not None

    @staticmethod
    def self_check(timeout_seconds: float = 10.0) -> BwrapSelfCheck:
        """Attempt the smallest possible bwrap invocation, both network
        policies, and report what actually happened — not what should
        happen. Safe to call repeatedly; touches nothing outside a throwaway
        namespace. Used by the test suite to skip (loudly) rather than fail
        for a reason no code change here can fix, and by anyone diagnosing a
        "coding run has no isolation" report to tell "bwrap absent" from
        "bwrap present but blocked" from "works, but only networked" —
        three different findings that all look identical from outside.
        """
        import subprocess

        bwrap_path = shutil.which(_BWRAP_BIN)
        if not bwrap_path:
            return BwrapSelfCheck(False, "bwrap not found on PATH", False, False)

        def _probe(extra: list[str]) -> tuple[bool, str]:
            sentinel = f"__bwrap_selfcheck_{uuid.uuid4().hex}__"
            argv = [
                bwrap_path,
                "--die-with-parent",
                "--unshare-user",
                "--unshare-pid",
                *extra,
                "--proc", "/proc",
                "--dev", "/dev",
                "--ro-bind-try", "/usr", "/usr",
                "--ro-bind-try", "/bin", "/bin",
                "--", "/bin/sh", "-c", f'printf "%s" {sentinel}',
            ]
            try:
                proc = subprocess.run(
                    argv, capture_output=True, text=True, timeout=timeout_seconds
                )
            except subprocess.TimeoutExpired:
                return False, "self-check timed out"
            combined = (proc.stdout or "") + (proc.stderr or "")
            if sentinel in combined:
                return True, "ok"
            return False, combined.strip()[:400] or f"exit {proc.returncode}, no output"

        isolated_ok, isolated_detail = _probe([])
        networked_ok, networked_detail = _probe(["--unshare-net"])

        if isolated_ok:
            return BwrapSelfCheck(
                True, "bwrap functional with network isolated",
                True, networked_ok,
            )
        if networked_ok:
            return BwrapSelfCheck(
                True,
                "bwrap functional ONLY with network retained — the isolated "
                "(no-net) path is blocked on this host, consistent with "
                "kernel.apparmor_restrict_unprivileged_userns=1. See "
                "BwrapSandbox's class docstring.",
                False, True,
            )
        return BwrapSelfCheck(
            False,
            f"bwrap cannot run a namespaced process in EITHER network "
            f"policy on this host. isolated: {isolated_detail!r} | "
            f"networked: {networked_detail!r}",
            False, False,
        )

    # ------------------------------------------------------------------
    # bwrap invocation
    # ------------------------------------------------------------------

    def _bwrap_argv(self, command: str, sentinel: str) -> list[str]:
        root_s = str(self.root)
        argv = [
            self._bwrap_path,
            "--die-with-parent",   # a killed daemon must not orphan a jail
            "--new-session",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup-try",
        ]
        if self.isolate_network:
            argv += ["--unshare-net"]
        argv += [
            "--proc", "/proc",
            "--dev", "/dev",       # minimal synthetic /dev, not the host's
            "--tmpfs", "/tmp",     # own scratch /tmp, isolated from the host's
        ]
        for base in _BASE_RO_BIND_DIRS:
            argv += ["--ro-bind-try", base, base]
        for d in _path_dirs():
            argv += ["--ro-bind-try", d, d]
        # The writable jail...
        argv += ["--bind", root_s, root_s]
        # ...then re-mount any denied_paths INSIDE it read-only. Order
        # matters: bwrap applies binds in argv order, and a later bind at a
        # nested path shadows the earlier one — so this must come AFTER the
        # root bind above, not before.
        for denied in self.denied_paths:
            if denied == self.root or denied.is_relative_to(self.root):
                argv += ["--ro-bind-try", str(denied), str(denied)]
        argv += ["--chdir", root_s]
        argv += ["--clearenv"]
        for k, v in self._scrubbed_env().items():
            argv += ["--setenv", k, v]
        # The sentinel is how run() tells "bwrap's own setup failed" from "the
        # command ran and exited nonzero" — printed BEFORE the real command,
        # so its absence from the output means the inner shell never started.
        inner = f'printf "%s\\n" {sentinel}; {command}'
        argv += ["--", "/bin/sh", "-c", inner]
        return argv

    async def run(
        self, command: str, *, timeout_seconds: float | None = None
    ) -> SandboxResult:
        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.default_timeout_seconds
        )
        sentinel = f"__bwrap_run_{uuid.uuid4().hex}__"
        argv = self._bwrap_argv(command, sentinel)

        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *argv,
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
                # bwrap's own pid namespace means killing bwrap tears down
                # everything inside it too — the process-group kill is a
                # first-line defense on top of that, not the only one.
                os.killpg(os.getpgid(proc.pid), 9)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            raw, _ = await proc.communicate()
        duration = time.monotonic() - started

        output = (raw or b"").decode("utf-8", errors="replace")

        if not timed_out and sentinel not in output:
            # bwrap failed before ever exec'ing the inner shell. Raise rather
            # than return a SandboxResult — proc.returncode here is BWRAP'S
            # exit code, and reporting it as the command's would answer a
            # different question than the one the caller asked.
            raise SandboxConstructionError(
                f"bwrap failed to start the sandboxed process "
                f"(exit {proc.returncode}): {output.strip()[:2000]}"
            )

        if sentinel in output:
            output = output.split(sentinel, 1)[-1].lstrip("\n")

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
