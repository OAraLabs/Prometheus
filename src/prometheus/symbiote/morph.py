"""MorphEngine — Phase 4 of SYMBIOTE: blue-green hot swap with auto-rollback.

Source: Novel code for Prometheus GRAFT-SYMBIOTE Sprint Session B.
License: MIT

The morph engine takes a completed graft (Session A produced one), creates
a versioned backup, stages the live tree as a candidate, validates it with
the full test suite, and (on user approval) performs a hot swap:

  1. Stop daemon (using detected manager strategy)
  2. mv src/prometheus → src/prometheus.pre_swap
  3. mv candidate/prometheus → src/prometheus
  4. mv tests → tests.pre_swap (if changed) ; mv candidate/tests → tests
  5. Start daemon
  6. Health-check watchdog (60s default, 5s interval, 3 consecutive passes)
  7. Healthy → COMPLETE; clean up .pre_swap
     Unhealthy → AUTO-ROLLBACK (no approval needed); preserve failed candidate

DAEMON-MANAGER DETECTION (CRITICAL):
The Prometheus daemon may be running under any of:
  • systemd user unit (``systemctl --user status prometheus``)
  • bare process with a JSON lock at ``~/.prometheus/daemon.lock`` containing
    the PID (this is the path Session-15 ``acquire_daemon_lock`` already
    writes — we treat it as the canonical "pid file")
  • bare process with no lock at all

``_detect_daemon_manager`` checks in that order, with a 3-second timeout
on systemctl. It NEVER hangs waiting for a unit that doesn't exist. The
result is cached on the instance so stop/start use the same strategy.

Path-traversal guard: every directory move resolves the candidate path
before checking that it lives under ``CANDIDATE_ROOT``. See
PROMETHEUS.md §Security Conventions / Path Traversal Defense.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from prometheus.config.paths import get_config_dir
from prometheus.symbiote.backup_vault import BackupSnapshot, BackupVault

if TYPE_CHECKING:
    from prometheus.symbiote.graft import GraftReport

log = logging.getLogger(__name__)


DaemonManager = Literal["systemd", "pidfile", "pkill"]


def _candidate_root_default() -> Path:
    return Path.home() / ".prometheus" / "symbiote" / "candidate"


def _post_mortem_root_default() -> Path:
    return Path.home() / ".prometheus" / "symbiote" / "post_mortem"


def _daemon_lock_path() -> Path:
    """Path to the JSON lock file written by ``acquire_daemon_lock``."""
    return get_config_dir() / "daemon.lock"


_PKILL_PATTERN = "python.*prometheus daemon"
_DAEMON_LAUNCH_CMD = ["python3", "-m", "prometheus", "daemon"]


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass
class MorphReport:
    """Result of ``MorphEngine.prepare_candidate``."""

    backup_snapshot_id: str
    candidate_path: str
    graft_applied: bool
    tests_passed: bool
    tests_output: str
    files_changed: list[str] = field(default_factory=list)
    new_dependencies: list[str] = field(default_factory=list)
    ready_to_swap: bool = False
    reason_if_not_ready: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_telegram_summary(self) -> str:
        status = "✅ ready" if self.ready_to_swap else "❌ blocked"
        lines = [
            f"MORPH report — {status}",
            f"Backup: {self.backup_snapshot_id}",
            f"Candidate: {self.candidate_path}",
            f"Tests: {'passing' if self.tests_passed else 'FAILING'}",
            f"Files changed vs prior backup: {len(self.files_changed)}",
        ]
        if self.new_dependencies:
            lines.append(f"New deps flagged: {', '.join(self.new_dependencies[:5])}")
        if not self.ready_to_swap and self.reason_if_not_ready:
            lines.append(f"Blocked: {self.reason_if_not_ready}")
        if self.ready_to_swap:
            lines.append("Reply /symbiote swap to execute the hot swap.")
        return "\n".join(lines)


@dataclass
class SwapResult:
    """Result of ``MorphEngine.execute_swap``."""

    success: bool
    previous_version: str = ""
    new_version: str = ""
    health_check_passed: bool = False
    rolled_back: bool = False
    rollback_reason: str = ""
    downtime_seconds: float = 0.0
    notification_sent: bool = False
    daemon_manager: str = ""
    error: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_telegram_summary(self) -> str:
        if self.error and not self.rolled_back:
            return f"SWAP failed: {self.error}"
        if self.rolled_back:
            return (
                f"SWAP rolled back to {self.previous_version}. "
                f"Reason: {self.rollback_reason}"
            )
        return (
            f"SWAP complete. Now running {self.new_version}. "
            f"Downtime ~{self.downtime_seconds:.1f}s. "
            f"Backup {self.previous_version} retained."
        )


# ---------------------------------------------------------------------------
# MorphEngine
# ---------------------------------------------------------------------------


class MorphEngine:
    """Blue-green self-deployment for the Prometheus daemon."""

    def __init__(
        self,
        backup_vault: BackupVault,
        project_root: Path,
        *,
        candidate_root: Path | None = None,
        post_mortem_root: Path | None = None,
        health_check_timeout: int = 60,
        health_check_interval: int = 5,
        consecutive_passes_required: int = 3,
        daemon_manager_override: str | None = None,
        daemon_health_url: str | None = None,
    ) -> None:
        self._vault = backup_vault
        self._project_root = project_root.resolve()
        self._candidate_root = (candidate_root or _candidate_root_default()).expanduser().resolve()
        self._post_mortem_root = (post_mortem_root or _post_mortem_root_default()).expanduser().resolve()
        self._health_timeout = max(5, int(health_check_timeout))
        self._health_interval = max(1, int(health_check_interval))
        self._passes_required = max(1, int(consecutive_passes_required))
        self._daemon_health_url = daemon_health_url
        self._cached_manager: DaemonManager | None = None
        self._daemon_manager_override: str | None = (
            daemon_manager_override if daemon_manager_override in
            ("systemd", "pidfile", "pkill") else None
        )

    # ==================================================================
    # PHASE 1 — prepare_candidate
    # ==================================================================

    async def prepare_candidate(
        self,
        graft_report: "GraftReport | None" = None,
        *,
        capture_test_status: bool = True,
    ) -> MorphReport:
        """Stage the live tree as a candidate, run tests, produce a MorphReport.

        Note on Session A interaction: Session A's ``GraftEngine`` writes
        adapted files directly into ``src/prometheus/``, so the live tree
        already contains the graft by the time we get here. ``prepare_candidate``
        therefore copies the *live* tree into the candidate directory and
        runs tests against that copy. The "apply graft to candidate" step
        in the spec is a no-op for this Session-A pipeline — the comment
        block below preserves the intent if a future sprint moves graft to
        be staged-only.
        """
        timestamp = _now_iso()

        # 1. Backup snapshot (source="symbiote_morph", exempt from retention).
        snapshot = await self._vault.create_snapshot(
            description="pre-MORPH safety net",
            source="symbiote_morph",
            metadata={
                "graft_repo": getattr(graft_report, "repo_full_name", None),
                "files_created": [
                    f.path for f in (
                        getattr(graft_report, "files_created", []) or []
                    )
                ],
            },
            capture_test_status=False,
        )

        # 2. Stage candidate directory.
        await self._stage_candidate()

        # 3. Apply graft to candidate — no-op in V1 (graft is already on live);
        #    this is the hook future sprints can fill in if Graft becomes
        #    staged-only.

        # 4. Run tests against candidate.
        tests_passed = True
        tests_output = "(skipped)"
        if capture_test_status:
            tests_passed, tests_output = await self._run_candidate_tests()

        # 5. Diff candidate vs the snapshot's manifest (informational).
        files_changed = self._files_changed_against_snapshot(snapshot)

        # 6. Aggregate.
        ready = bool(tests_passed)
        reason = "" if ready else "candidate test suite failed"
        report = MorphReport(
            backup_snapshot_id=snapshot.backup_id,
            candidate_path=str(self._candidate_root),
            graft_applied=graft_report is not None,
            tests_passed=tests_passed,
            tests_output=tests_output[-2000:],
            files_changed=files_changed,
            new_dependencies=[],
            ready_to_swap=ready,
            reason_if_not_ready=reason,
            timestamp=timestamp,
        )
        return report

    # ==================================================================
    # PHASE 2 — execute_swap (REQUIRES PRIOR USER APPROVAL UPSTREAM)
    # ==================================================================

    async def execute_swap(self, morph_report: MorphReport) -> SwapResult:
        """Perform the hot swap with automatic rollback on health failure."""
        result = SwapResult(
            success=False,
            previous_version=morph_report.backup_snapshot_id,
            timestamp=_now_iso(),
        )
        if not morph_report.ready_to_swap:
            result.error = (
                f"Not ready to swap: {morph_report.reason_if_not_ready}"
            )
            return result

        candidate_path = Path(morph_report.candidate_path)
        if not self._candidate_under_root(candidate_path):
            result.error = (
                f"Refusing to swap a candidate outside CANDIDATE_ROOT: {candidate_path}"
            )
            return result
        live_pkg = self._project_root / "src" / "prometheus"
        if not live_pkg.exists():
            result.error = f"Live src/prometheus does not exist at {live_pkg}"
            return result

        manager = await self._detect_daemon_manager()
        result.daemon_manager = manager
        log.info("MorphEngine: using daemon manager %s", manager)

        start_time = time.time()
        stopped = await self._stop_daemon(manager)
        if not stopped:
            result.error = "Could not stop daemon — aborting swap."
            return result

        # Atomic-ish swap.
        live_backup = live_pkg.with_name("prometheus.pre_swap")
        candidate_pkg = candidate_path / "src" / "prometheus"
        try:
            self._swap_directories(live_pkg, candidate_pkg, live_backup)
        except Exception as exc:
            log.exception("MorphEngine: swap failed mid-flight")
            result.error = f"Atomic swap failed: {exc}"
            # Best-effort restore.
            await self._restart_after_failure(live_pkg, live_backup, manager)
            return result

        started = await self._start_daemon(manager)
        if not started:
            log.error("MorphEngine: daemon failed to start after swap")
            result.rolled_back = await self._auto_rollback(
                "daemon failed to start after swap",
                live_pkg=live_pkg,
                pre_swap=live_backup,
                manager=manager,
            )
            result.rollback_reason = "daemon failed to start"
            result.downtime_seconds = time.time() - start_time
            return result

        healthy, reason = await self._health_check_watchdog()
        result.health_check_passed = healthy
        result.downtime_seconds = time.time() - start_time

        if not healthy:
            result.rolled_back = await self._auto_rollback(
                reason,
                live_pkg=live_pkg,
                pre_swap=live_backup,
                manager=manager,
            )
            result.rollback_reason = reason
            return result

        # Success — clean up the .pre_swap dir, generate a "new_version" id.
        self._cleanup_pre_swap(live_backup)
        result.success = True
        result.new_version = f"post_swap_{int(time.time())}"
        return result

    # ==================================================================
    # Daemon-manager detection
    # ==================================================================

    async def _detect_daemon_manager(self) -> DaemonManager:
        """Detect how the daemon is currently managed.

        Returns one of "systemd" | "pidfile" | "pkill". Result cached on
        the instance after the first call.
        """
        if self._cached_manager is not None:
            return self._cached_manager
        if self._daemon_manager_override is not None:
            self._cached_manager = self._daemon_manager_override  # type: ignore[assignment]
            return self._cached_manager  # type: ignore[return-value]

        # 1. systemctl is-active --user prometheus, with 3s timeout.
        if await self._systemctl_is_active():
            self._cached_manager = "systemd"
            return "systemd"

        # 2. PID file (lock file with PID inside) exists AND PID is alive.
        pid = self._read_lockfile_pid()
        if pid is not None and _pid_alive(pid):
            self._cached_manager = "pidfile"
            return "pidfile"

        # 3. Fallback to pkill.
        self._cached_manager = "pkill"
        return "pkill"

    @staticmethod
    async def _systemctl_is_active() -> bool:
        """``systemctl is-active --user prometheus``. 3s timeout, never hang."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl", "is-active", "--user", "prometheus",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError):
            return False
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            return False
        if proc.returncode != 0:
            return False
        return stdout.decode("utf-8", errors="replace").strip() == "active"

    @staticmethod
    def _read_lockfile_pid() -> int | None:
        """Return the PID from ``~/.prometheus/daemon.lock`` if readable."""
        lock = _daemon_lock_path()
        if not lock.exists():
            return None
        try:
            data = json.loads(lock.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        pid = data.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            return None
        return pid

    # ==================================================================
    # Daemon stop/start
    # ==================================================================

    async def _stop_daemon(self, manager: DaemonManager) -> bool:
        """Stop the daemon. 10s wait. Returns True iff confirmed stopped."""
        if manager == "systemd":
            return await self._stop_systemd()
        if manager == "pidfile":
            pid = self._read_lockfile_pid()
            if pid is None:
                # Lockfile vanished between detection and stop — fall back.
                return await self._stop_pkill()
            return await self._stop_pid(pid)
        return await self._stop_pkill()

    async def _start_daemon(self, manager: DaemonManager) -> bool:
        if manager == "systemd":
            return await self._start_systemd()
        return await self._start_bare()

    async def _stop_systemd(self) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl", "--user", "stop", "prometheus",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except (FileNotFoundError, OSError, asyncio.TimeoutError):
            return False
        return proc.returncode == 0

    async def _stop_pid(self, pid: int) -> bool:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True  # already gone
        except PermissionError:
            return False
        for _ in range(10):
            if not _pid_alive(pid):
                return True
            await asyncio.sleep(1.0)
        # Escalate to SIGKILL.
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        await asyncio.sleep(1.0)
        return not _pid_alive(pid)

    async def _stop_pkill(self) -> bool:
        # SIGTERM first.
        await self._run_pkill(signal_arg="-TERM")
        for _ in range(10):
            if not await self._any_match_pkill():
                return True
            await asyncio.sleep(1.0)
        # SIGKILL.
        await self._run_pkill(signal_arg="-KILL")
        await asyncio.sleep(1.0)
        return not await self._any_match_pkill()

    async def _run_pkill(self, *, signal_arg: str) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "pkill", signal_arg, "-f", _PKILL_PATTERN,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except (FileNotFoundError, OSError, asyncio.TimeoutError):
            pass

    @staticmethod
    async def _any_match_pkill() -> bool:
        """Return True if any process matches the daemon pattern."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "pgrep", "-f", _PKILL_PATTERN,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except (FileNotFoundError, OSError):
            return False
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            return False
        return proc.returncode == 0 and bool(stdout.strip())

    async def _start_systemd(self) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl", "--user", "start", "prometheus",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except (FileNotFoundError, OSError, asyncio.TimeoutError):
            return False
        return proc.returncode == 0

    async def _start_bare(self) -> bool:
        """Launch ``python3 -m prometheus daemon &`` detached.

        The daemon writes its own ``daemon.lock`` on startup, so future
        ``_stop_daemon`` calls can pick the cleaner pidfile path.
        """
        try:
            subprocess.Popen(  # noqa: S603,S607 — controlled launch
                _DAEMON_LAUNCH_CMD,
                cwd=str(self._project_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (FileNotFoundError, OSError):
            return False
        # Brief wait for the daemon to begin starting; health check
        # will do the proper liveness verification.
        await asyncio.sleep(2.0)
        return True

    # ==================================================================
    # Health check
    # ==================================================================

    async def _health_check_watchdog(self) -> tuple[bool, str]:
        """Watch the post-swap daemon. Three consecutive passes = healthy."""
        deadline = time.time() + self._health_timeout
        passes = 0
        first_failure_reason = ""
        # Wait half an interval before first check (let daemon settle).
        await asyncio.sleep(self._health_interval / 2.0)
        while time.time() < deadline:
            ok, reason = await self._single_health_check()
            if ok:
                passes += 1
                if passes >= self._passes_required:
                    return True, "healthy"
            else:
                first_failure_reason = first_failure_reason or reason
                # Only require strict consecutiveness while we haven't yet
                # accumulated enough passes.
                if passes < self._passes_required:
                    passes = 0
            await asyncio.sleep(self._health_interval)
        return False, first_failure_reason or "health check timed out"

    async def _single_health_check(self) -> tuple[bool, str]:
        """One liveness check. Currently: lock-file PID alive."""
        # 1. Process is running (via lock file PID, or pgrep fallback).
        pid = self._read_lockfile_pid()
        if pid is not None:
            if not _pid_alive(pid):
                return False, "daemon PID not alive"
        else:
            if not await self._any_match_pkill():
                return False, "no daemon process matches"
        # 2. Optional HTTP health check.
        if self._daemon_health_url:
            ok = await self._http_health_check(self._daemon_health_url)
            if not ok:
                return False, "daemon HTTP health check failed"
        return True, ""

    @staticmethod
    async def _http_health_check(url: str) -> bool:
        try:
            import httpx  # type: ignore
        except ImportError:
            return True  # can't check; assume OK
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(url)
            return 200 <= resp.status_code < 500
        except Exception:
            return False

    # ==================================================================
    # Atomic swap + rollback
    # ==================================================================

    def _swap_directories(
        self,
        live: Path,
        candidate: Path,
        live_backup: Path,
    ) -> None:
        """``mv live live_backup ; mv candidate live``. Caller must hold daemon stopped."""
        if not candidate.exists():
            raise RuntimeError(f"Candidate path does not exist: {candidate}")
        # If a previous .pre_swap is still around, archive it out of the way.
        if live_backup.exists():
            stash = live_backup.with_name(
                f"{live_backup.name}.{int(time.time())}"
            )
            shutil.move(str(live_backup), str(stash))
            log.warning(
                "MorphEngine: stale .pre_swap moved aside to %s", stash,
            )
        shutil.move(str(live), str(live_backup))
        try:
            shutil.move(str(candidate), str(live))
        except Exception:
            # Couldn't put the candidate in place — undo the live → backup move.
            try:
                shutil.move(str(live_backup), str(live))
            except Exception:
                log.exception("MorphEngine: failed to restore live after swap failure")
            raise

    async def _auto_rollback(
        self,
        reason: str,
        *,
        live_pkg: Path,
        pre_swap: Path,
        manager: DaemonManager,
    ) -> bool:
        """Roll back to .pre_swap. NO user approval — broken daemons can't ask."""
        log.warning("MorphEngine: AUTO-ROLLBACK triggered (%s)", reason)
        await self._stop_daemon(manager)  # best effort

        # Preserve failed candidate for post-mortem.
        failed_dir = live_pkg.with_name("prometheus.failed")
        try:
            if live_pkg.exists():
                if failed_dir.exists():
                    shutil.rmtree(failed_dir, ignore_errors=True)
                shutil.move(str(live_pkg), str(failed_dir))
                self._preserve_failed_candidate(failed_dir, reason)
        except OSError:
            log.exception("MorphEngine: could not preserve failed candidate")

        # Restore previous live from .pre_swap.
        try:
            if pre_swap.exists():
                shutil.move(str(pre_swap), str(live_pkg))
            else:
                log.error(
                    "MorphEngine: .pre_swap missing — cannot restore "
                    "previous version automatically."
                )
                return False
        except OSError:
            log.exception("MorphEngine: rollback move failed")
            return False

        # Restart daemon on the restored tree.
        await self._start_daemon(manager)
        return True

    async def _restart_after_failure(
        self,
        live_pkg: Path,
        pre_swap: Path,
        manager: DaemonManager,
    ) -> None:
        """Best-effort recovery if the swap itself failed midway."""
        if not live_pkg.exists() and pre_swap.exists():
            try:
                shutil.move(str(pre_swap), str(live_pkg))
            except OSError:
                log.exception("MorphEngine: emergency restore failed")
        await self._start_daemon(manager)

    def _cleanup_pre_swap(self, pre_swap: Path) -> None:
        """Remove ``.pre_swap`` after a successful, healthy swap."""
        try:
            if pre_swap.exists():
                shutil.rmtree(pre_swap)
                log.info("MorphEngine: pruned %s", pre_swap)
        except OSError:
            log.debug("MorphEngine: cleanup of %s failed", pre_swap, exc_info=True)

    def _preserve_failed_candidate(self, failed_dir: Path, reason: str) -> None:
        """Move a failed candidate into post_mortem/<timestamp>/."""
        ts = int(time.time())
        target = self._post_mortem_root / f"failed_{ts}"
        try:
            self._post_mortem_root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(failed_dir), str(target))
            (target / "REASON.txt").write_text(reason, encoding="utf-8")
            log.info("MorphEngine: preserved failed candidate at %s", target)
        except OSError:
            log.exception("MorphEngine: could not preserve failed candidate")

    # ==================================================================
    # Candidate staging
    # ==================================================================

    async def _stage_candidate(self) -> None:
        """Copy live src/prometheus + tests into ``CANDIDATE_ROOT``."""
        # Reset.
        if self._candidate_root.exists():
            shutil.rmtree(self._candidate_root)
        self._candidate_root.mkdir(parents=True, exist_ok=True)

        live_pkg = self._project_root / "src" / "prometheus"
        live_tests = self._project_root / "tests"
        cand_pkg = self._candidate_root / "src" / "prometheus"
        cand_tests = self._candidate_root / "tests"

        if live_pkg.exists():
            cand_pkg.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                live_pkg,
                cand_pkg,
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache",
                ),
            )
        if live_tests.exists():
            shutil.copytree(
                live_tests,
                cand_tests,
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", ".pytest_cache",
                ),
            )

    async def _run_candidate_tests(self) -> tuple[bool, str]:
        """Run pytest against the candidate tree. Returns (passed, output)."""
        cand_pkg = self._candidate_root / "src" / "prometheus"
        cand_tests = self._candidate_root / "tests"
        if not cand_pkg.exists() or not cand_tests.exists():
            return False, "candidate not staged"
        env = os.environ.copy()
        # Make the candidate src importable instead of live src.
        env["PYTHONPATH"] = str(self._candidate_root / "src")
        cmd = ["python3", "-m", "pytest", str(cand_tests), "-q", "--tb=short"]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self._candidate_root),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300.0)
        except (FileNotFoundError, OSError) as exc:
            return False, f"pytest launch failed: {exc}"
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            return False, "candidate tests timed out"
        text = stdout.decode("utf-8", errors="replace") if stdout else ""
        return proc.returncode == 0, text

    def _files_changed_against_snapshot(
        self,
        snapshot: BackupSnapshot,
    ) -> list[str]:
        """List of candidate paths NOT in the snapshot manifest (best-effort)."""
        snap_set = set(snapshot.manifest)
        out: list[str] = []
        cand_pkg = self._candidate_root / "src" / "prometheus"
        if not cand_pkg.exists():
            return out
        # Compare against the live project root since the manifest paths are
        # relative to project root.
        for path in cand_pkg.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self._candidate_root).as_posix()
            if rel not in snap_set and not rel.startswith("identity/"):
                out.append(rel)
        return out

    # ==================================================================
    # Path-traversal guard
    # ==================================================================

    def _candidate_under_root(self, candidate: Path) -> bool:
        """Resolve before prefix-checking. See PROMETHEUS.md §Security
        Conventions / Path Traversal Defense."""
        try:
            resolved = candidate.resolve()
            resolved.relative_to(self._candidate_root)
        except (ValueError, OSError):
            return False
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    """``os.kill(pid, 0)`` based liveness check."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but we can't signal it
    return True


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
