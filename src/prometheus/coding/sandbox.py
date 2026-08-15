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
import json
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


class SandboxBackendUnavailable(RuntimeError):
    """The requested backend cannot run here, and fallback was not permitted.

    Raised rather than silently degrading, at two points:

    * :func:`create_sandbox`, when the backend is not usable on this host. A
      caller who asked for container or namespace isolation and quietly
      received ``ProcessSandbox`` would believe a boundary exists where none
      does — and ``ProcessSandbox``'s boundary is escapable by a one-line
      shell redirect. ``allow_fallback=True`` is available for callers who
      genuinely mean "best effort".
    * :meth:`DockerSandbox._ensure_container`, when a container was built but
      cannot execute anything. Handing that back is worse than degrading:
      every command "fails", which a containment check reads as a PASS.

    Failing loudly is the only honest option in both cases.
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

    @property
    def backend(self) -> str:
        # MUST be overridden: subclassing ProcessSandbox would otherwise make
        # a namespaced run report itself as "process", crediting the weaker
        # backend in every log line and telemetry row it appears in.
        return "bwrap"

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
    backend: str = "process",
    task_id: str | None = None,
    network_isolation: bool = False,
    image: str | None = None,
    allow_fallback: bool = False,
) -> Sandbox:
    """Create the dedicated full clone and return a sandbox rooted in it.

    A FULL ``git clone`` (not a worktree — spec decision: hard isolation
    over disk savings) into ``dest_parent/name``. The clone shares nothing
    writable with the source; the coding run's branch lives here.

    *backend* selects which sandbox wraps the clone — see
    :func:`create_sandbox`. It defaults to ``"process"`` so existing callers
    are unchanged, but it must be threaded through from
    ``coding.sandbox_type``: this is the ONLY place a coding run's sandbox is
    constructed, so a backend that stops here is a backend no user can reach
    no matter what their config says.
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
    return create_sandbox(
        dest,
        task_id=task_id or name,
        backend=backend,
        denied_paths=tuple(denied_paths),
        network_isolation=network_isolation,
        image=image,
        allow_fallback=allow_fallback,
    )


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

DOCKER_CONTAINER_PREFIX = "prometheus-coding-"
DOCKER_IMAGE = "python:3.12-slim"
DOCKER_LABEL = "prometheus.coding"

# Where the jail clone is mounted INSIDE the container. Host paths do not
# exist in the container's mount namespace, so every in-container path —
# --workdir, docker exec --workdir — must use this, never str(self.root).
CONTAINER_WORKDIR = "/workspace"

_DOCKER_PIDS_LIMIT = 128

# Slack added to a run's own timeout before we give up on `docker exec`
# itself, so container round-trip overhead is never mistaken for the
# command overrunning its budget.
_DOCKER_EXEC_GRACE_SECONDS = 5.0


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
        default_timeout_seconds: float = DEFAULT_RUN_TIMEOUT_SECONDS,
    ):
        self._root = Path(root).resolve()
        self._task_id = task_id
        # Same default as ProcessSandbox/BwrapSandbox. Previously this path
        # hardcoded 60s, so an identical run() call silently got a quarter of
        # the time budget depending on which backend the config selected —
        # a per-backend behaviour difference nothing advertised.
        self.default_timeout_seconds = default_timeout_seconds
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

    def _inspect_container(self) -> dict | None:
        """This container's inspect record, or None if absent/unreadable.

        ONE ``docker inspect``, parsed once: status and mounts then come from
        the same snapshot and cannot disagree the way two separate inspect
        calls could. Output that will not parse is reported as "no usable
        record", so the caller rebuilds — the safe direction.
        """
        result = subprocess.run(
            ["docker", "inspect", self._container_id],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        try:
            records = json.loads(result.stdout)
        except (TypeError, ValueError):
            return None
        if not isinstance(records, list) or not records:
            return None
        record = records[0]
        return record if isinstance(record, dict) else None

    def _probe_exec_failure(self) -> str | None:
        """Run a trivial exec at the workdir; return the failure, or None.

        ``--workdir CONTAINER_WORKDIR`` is what makes this representative. A
        dead bind mount surfaces only when the exec has to ENTER the
        workspace — which is exactly what every real :meth:`run` does, and
        what a probe without the flag would miss.
        """
        probe = subprocess.run(
            [
                "docker", "exec",
                "--workdir", CONTAINER_WORKDIR,
                self._container_id,
                "true",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if probe.returncode == 0:
            return None
        detail = (probe.stderr or probe.stdout or "").strip()
        return f"a probe exec failed (exit {probe.returncode}): {detail[:500]}"

    def _adoption_blocker(self, record: dict) -> str | None:
        """Why an existing container must NOT be reused — None means adopt."""
        state = record.get("State")
        status = state.get("Status") if isinstance(state, dict) else None
        if status not in ("running", "paused"):
            return f"its status is {status!r}, not running"

        source = None
        mounts = record.get("Mounts")
        if isinstance(mounts, list):
            for mount in mounts:
                if (
                    isinstance(mount, dict)
                    and mount.get("Destination") == CONTAINER_WORKDIR
                ):
                    source = mount.get("Source")
                    break
        if not source:
            return f"it has no {CONTAINER_WORKDIR} bind mount"

        bound = Path(source)
        if not bound.exists():
            return (
                f"its {CONTAINER_WORKDIR} bind source {bound} no longer "
                f"exists on the host"
            )
        try:
            same_root = bound.resolve() == self._root
        except OSError:
            same_root = False
        if not same_root:
            return (
                f"its {CONTAINER_WORKDIR} bind source {bound} is another "
                f"run's clone, not this run's root {self._root}"
            )

        # Path checks cannot see a RE-CREATED directory. A bind mount holds
        # the original dentry, so deleting the clone and cloning again at the
        # same path leaves the source present and matching while the mount
        # inside the container is a corpse. Only an exec settles it.
        return self._probe_exec_failure()

    def _ensure_container(self):
        """Create — or safely adopt — the Docker container for this sandbox.

        Adoption is the dangerous half. The container name is derived from
        the task id alone, so a run that dies without :meth:`close` leaves a
        RUNNING container that the next run with that task id finds and
        reuses. Its ``/workspace`` bind still points at the dead run's clone,
        which by then is gone (``clone_repo_for_sandbox`` gives each attempt
        a fresh timestamped directory, and refuses to reuse an existing one).
        Every ``docker exec`` into such a container then fails with:

            OCI runtime exec failed: ... current working directory is
            outside of container mount namespace

        That is worse than a crash. NOTHING executes, so a containment check
        watches every escape attempt "fail" and reports a PASS — a false pass
        indistinguishable from a real one, which is how this was found.

        A container is therefore adopted only if it is running, its
        ``/workspace`` bind is still THIS run's root, and it can actually
        exec. Anything else is torn down and rebuilt, and a freshly built
        container that still cannot exec raises rather than being handed
        back.
        """
        record = self._inspect_container()
        if record is not None:
            blocker = self._adoption_blocker(record)
            if blocker is None:
                return  # Container is live and usable, good.
            log.warning(
                "Discarding existing Docker container %s (%s) — rebuilding.",
                self._container_id, blocker,
            )
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True,
                text=True,
                timeout=10,
            )

        # Build the container spec.
        #
        # Flag notes, each verified against the live docker CLI rather than
        # assumed — three of these were wrong in a way that made the backend
        # unusable, and none of them failed at import time:
        #   * `--pids-limit N`, NOT `--pid limit=N` (the latter is the PID
        #     *namespace* flag and docker rejects it outright).
        #   * NO tmpfs on /workspace: the clone is bind-mounted there, and
        #     docker REFUSES a tmpfs and a bind at the same target. /tmp
        #     keeps its tmpfs because nothing is mounted over it.
        #   * The container must be STARTED, not merely created — `docker
        #     exec` against a created-but-never-started container fails with
        #     "container is not running".
        cmd = [
            "docker", "create",
            "--name", self._container_id,
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(_DOCKER_PIDS_LIMIT),
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--label", DOCKER_LABEL,
            "--workdir", CONTAINER_WORKDIR,
            # Run as the HOST user, not root. Two reasons, and the first is
            # not optional:
            #   * --cap-drop ALL removes CAP_DAC_OVERRIDE, which is precisely
            #     the capability that lets root ignore permission bits. A
            #     root container against a user-owned bind mount therefore
            #     cannot even READ the workspace — "Permission denied" on
            #     every file, which reads like a broken mount rather than a
            #     dropped capability.
            #   * Files the run creates stay owned by the invoking user
            #     instead of appearing as root-owned artifacts in the clone.
            "--user", f"{os.getuid()}:{os.getgid()}",
            # The image's default HOME may not be writable for that uid; /tmp
            # is the tmpfs mounted above, so point HOME there rather than
            # letting tools fail on an unwritable home.
            "--env", "HOME=/tmp",
            # Keep the container alive between exec calls; --read-only plus
            # cap-drop ALL is what bounds it, not the entrypoint.
            "--entrypoint", "sh",
        ]
        if self._network_isolation:
            cmd.append("--network=none")

        # Mount the sandbox clone at the container's workdir.
        cmd.extend(["-v", f"{self._root}:{CONTAINER_WORKDIR}"])
        cmd.append(self._image)
        cmd.extend(["-c", "while true; do sleep 3600; done"])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create Docker container {self._container_id}: "
                f"{result.stderr.strip()}"
            )

        start = subprocess.run(
            ["docker", "start", self._container_id],
            capture_output=True, text=True, timeout=30,
        )
        if start.returncode != 0:
            # Don't leave a created-but-dead container behind to confuse the
            # next run's inspect branch.
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True, timeout=15,
            )
            raise RuntimeError(
                f"Failed to start Docker container {self._container_id}: "
                f"{start.stderr.strip()}"
            )

        # A started container is not yet a WORKING one. Prove it can execute
        # at the workdir before handing it back, and refuse loudly if it
        # cannot: a sandbox in which every command fails looks exactly like a
        # sandbox that contains everything.
        problem = self._probe_exec_failure()
        if problem:
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True, timeout=15,
            )
            raise SandboxBackendUnavailable(
                f"Docker container {self._container_id} started but cannot "
                f"execute commands: {problem}. Refusing to return a sandbox "
                f"in which nothing runs — every command would 'fail', which "
                f"a containment check would misread as a PASS."
            )

    def resolve(self, path: str) -> Path:
        """Resolve a path, enforcing containment.

        Tool paths are host-side (the bind-mount source), so containment is
        the same question as ``ProcessSandbox.resolve()`` — and is answered
        with the same logic, deliberately. Two things this must NOT do, both
        of which it did before:

        * **String prefix matching.** ``str(p).startswith(str(root))`` admits
          a sibling whose name merely extends the root's — ``/jail-evil``
          "starts with" ``/jail``. ``is_relative_to`` compares path
          components, so it cannot be fooled that way.
        * **``strict=True``.** That raises for a path that does not exist
          yet, so every write to a NEW file inside the jail was rejected as a
          violation. Non-strict resolution still canonicalises ``..`` and
          symlinks, which is what containment actually needs.
        """
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self._root / candidate
        try:
            real = candidate.resolve()
        except (OSError, ValueError, RuntimeError):
            # RuntimeError covers a symlink loop.
            raise SandboxViolation(f"Cannot resolve path {path!r} inside sandbox")

        if not (real == self._root or real.is_relative_to(self._root)):
            raise SandboxViolation(
                f"path escapes the sandbox: {path!r} → {real} "
                f"(root: {self._root})"
            )

        for denied in self._denied_paths:
            if real == denied or real.is_relative_to(denied):
                raise SandboxViolation(
                    f"path is denied by policy: {path!r} → {real} "
                    f"(denied root: {denied})"
                )

        return real

    async def run(
        self, command: str, *, timeout_seconds: float | None = None
    ) -> SandboxResult:
        if self._closed:
            raise RuntimeError("Sandbox already closed")

        # The container sees the clone at CONTAINER_WORKDIR, never at its
        # host path — passing str(self._root) here made every exec fail with
        # "no such file or directory" for a workdir that cannot exist inside
        # the mount namespace.
        container_cwd = CONTAINER_WORKDIR

        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.default_timeout_seconds
        )

        start = time.monotonic()

        try:
            # Use docker exec for the command. `sh`, not `bash`: the default
            # image is Debian-slim (which has bash), but a user-supplied
            # `docker_image` may be Alpine or distroless-ish, where bash is
            # absent and the exec would fail for a reason that looks like the
            # command's fault.
            cmd = [
                "docker", "exec", "-i",
                "--workdir", container_cwd,
                self._container_id,
                "sh", "-c", command,
            ]
            # asyncio, not subprocess.run: this coroutine runs on the daemon's
            # event loop, and a blocking call here stalls every other session
            # for the duration of a coding command — which can be minutes.
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            raw, _ = await asyncio.wait_for(
                proc.communicate(), timeout=timeout + _DOCKER_EXEC_GRACE_SECONDS
            )
            duration = time.monotonic() - start

            output = (raw or b"").decode("utf-8", errors="replace")

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

        except (asyncio.TimeoutError, subprocess.TimeoutExpired):
            duration = time.monotonic() - start
            # `docker kill` stops the CONTAINER, which is what actually
            # guarantees the runaway process tree inside it is gone — the
            # exec'd child has no life independent of it. Restarting after
            # leaves the sandbox usable for the next iterate-to-green round
            # rather than poisoning every later call in the run.
            await asyncio.get_running_loop().run_in_executor(
                None, self._kill_and_restart_container
            )
            return SandboxResult(
                exit_code=None,
                output="",
                timed_out=True,
                duration_seconds=duration,
            )

    def _kill_and_restart_container(self) -> None:
        """Blocking docker kill+start, for the timeout path's executor."""
        subprocess.run(
            ["docker", "kill", self._container_id],
            capture_output=True, timeout=15,
        )
        subprocess.run(
            ["docker", "start", self._container_id],
            capture_output=True, timeout=15,
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


SANDBOX_BACKENDS: tuple[str, ...] = ("process", "bwrap", "docker")


def create_sandbox(
    root: str | Path,
    *,
    task_id: str,
    backend: str = "process",
    denied_paths: Iterable[str] = (),
    network_isolation: bool = False,
    image: str | None = None,
    allow_fallback: bool = False,
) -> Sandbox:
    """Create a sandbox for the requested backend.

    Args:
        root: Resolved path to the sandbox clone root.
        task_id: UUID string for this coding task (Docker container naming).
        backend: one of :data:`SANDBOX_BACKENDS` — ``"process"``,
            ``"bwrap"``, or ``"docker"``, weakest to strongest.
        denied_paths: Paths inside the root that are off-limits.
        network_isolation: Docker only — drop network access.
        image: Docker only — override the container image.
        allow_fallback: When the requested backend is unavailable, degrade to
            ``ProcessSandbox`` with a WARNING instead of raising. Default
            False: see :class:`SandboxBackendUnavailable` for why silence is
            the wrong default for a security boundary.

    Returns:
        A :class:`ProcessSandbox`, :class:`BwrapSandbox`, or
        :class:`DockerSandbox`.

    Raises:
        ValueError: unknown backend name.
        SandboxBackendUnavailable: backend is known but not usable here, and
            ``allow_fallback`` is False.
    """
    requested = (backend or "process").strip().lower()
    if requested not in SANDBOX_BACKENDS:
        raise ValueError(
            f"unknown sandbox backend {backend!r}; "
            f"expected one of {', '.join(SANDBOX_BACKENDS)}"
        )

    def _unavailable(reason: str) -> Sandbox:
        if not allow_fallback:
            raise SandboxBackendUnavailable(
                f"sandbox backend {requested!r} is unavailable: {reason}. "
                f"Refusing to fall back to 'process', whose confinement a "
                f"shell redirect escapes — pass allow_fallback=True if a "
                f"weaker sandbox is genuinely acceptable for this run."
            )
        log.warning(
            "Sandbox backend %r unavailable (%s) — falling back to 'process', "
            "which does NOT contain shell writes.", requested, reason,
        )
        return ProcessSandbox(root=Path(root), denied_paths=tuple(denied_paths))

    if requested == "docker":
        if not docker_available():
            return _unavailable("the Docker daemon is not reachable")
        return DockerSandbox(
            root,
            task_id=task_id,
            denied_paths=denied_paths,
            network_isolation=network_isolation,
            image=image,
        )

    if requested == "bwrap":
        if not BwrapSandbox.is_available():
            return _unavailable("the 'bwrap' binary is not on PATH")
        # Presence of the binary is not the same claim as "it can actually
        # create a namespace here" — see BwrapSandbox.self_check() and the
        # host finding in its docstring.
        check = BwrapSandbox.self_check()
        if not check.ok:
            return _unavailable(check.detail)
        return BwrapSandbox(
            root=Path(root),
            denied_paths=tuple(denied_paths),
            isolate_network=network_isolation,
        )

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
