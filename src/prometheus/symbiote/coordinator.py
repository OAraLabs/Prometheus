"""SymbioteCoordinator — orchestrates Scout → Harvest → Graft.

Source: Novel code for Prometheus GRAFT-SYMBIOTE Sprint Session A.
License: MIT

State machine (Session A scope; MORPH/SWAP phases are Session B):

    IDLE
      └── start_scout(problem) ─► SCOUTING
          └── ScoutReport stored ─► AWAITING_SCOUT_APPROVAL
              └── approve_scout(candidate) ─► HARVESTING
                  └── HarvestReport stored ─► AWAITING_HARVEST_APPROVAL
                      └── approve_harvest() ─► GRAFTING
                          └── GraftReport stored ─► AWAITING_GRAFT_APPROVAL
                              └── approve_graft() ─► COMPLETE

Any phase → FAILED on error.
abort(session_id) → ABORTED.

Mutex: only one non-terminal session at a time.

Persistence: SQLite at ~/.prometheus/symbiote/sessions.db.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from prometheus.config.paths import get_config_dir
from prometheus.symbiote.graft import GraftEngine, GraftReport
from prometheus.symbiote.harvest import HarvestEngine, HarvestReport
from prometheus.symbiote.license_gate import LicenseCheck, LicenseVerdict
from prometheus.symbiote.scout import ScoutEngine, ScoutReport

log = logging.getLogger(__name__)


class SymbiotePhase(str, Enum):
    IDLE = "idle"
    SCOUTING = "scouting"
    AWAITING_SCOUT_APPROVAL = "awaiting_scout_approval"
    HARVESTING = "harvesting"
    AWAITING_HARVEST_APPROVAL = "awaiting_harvest_approval"
    GRAFTING = "grafting"
    AWAITING_GRAFT_APPROVAL = "awaiting_graft_approval"
    # Session B: MORPH phases (added 2026-04-26)
    MORPHING = "morphing"
    AWAITING_SWAP_APPROVAL = "awaiting_swap_approval"
    SWAPPING = "swapping"
    HEALTH_CHECK = "health_check"
    ROLLED_BACK = "rolled_back"
    COMPLETE = "complete"
    FAILED = "failed"
    ABORTED = "aborted"


_TERMINAL = frozenset({
    SymbiotePhase.COMPLETE,
    SymbiotePhase.FAILED,
    SymbiotePhase.ABORTED,
    SymbiotePhase.ROLLED_BACK,
})


@dataclass
class SymbioteSession:
    session_id: str
    problem_statement: str
    phase: SymbiotePhase
    scout_report: dict[str, Any] | None = None
    selected_candidate: str | None = None
    harvest_report: dict[str, Any] | None = None
    graft_report: dict[str, Any] | None = None
    # Session B (MORPH) additions
    morph_report: dict[str, Any] | None = None
    swap_result: dict[str, Any] | None = None
    backup_id: str | None = None
    started_at: str = ""
    updated_at: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["phase"] = self.phase.value
        return d

    def is_terminal(self) -> bool:
        return self.phase in _TERMINAL

    def to_telegram_summary(self) -> str:
        lines = [
            f"SYMBIOTE session {self.session_id[:8]}",
            f"Phase: {self.phase.value}",
            f"Problem: {self.problem_statement}",
        ]
        if self.selected_candidate:
            lines.append(f"Candidate: {self.selected_candidate}")
        if self.scout_report and self.scout_report.get("candidates"):
            cnt = len(self.scout_report["candidates"])
            lines.append(f"Scout: {cnt} candidate(s)")
        if self.harvest_report:
            modules = len(self.harvest_report.get("modules_extracted", []))
            lines.append(f"Harvest: {modules} module(s)")
        if self.graft_report:
            files = len(self.graft_report.get("files_created", []))
            tests = "passed" if self.graft_report.get("tests_passed") else "FAILED"
            lines.append(f"Graft: {files} file(s), tests {tests}")
        if self.morph_report:
            ready = self.morph_report.get("ready_to_swap")
            tests = "passed" if self.morph_report.get("tests_passed") else "FAILED"
            lines.append(
                f"Morph: ready={ready}, tests {tests}, "
                f"backup={self.morph_report.get('backup_snapshot_id', '?')}"
            )
        if self.swap_result:
            if self.swap_result.get("rolled_back"):
                lines.append(
                    f"Swap: ROLLED BACK ({self.swap_result.get('rollback_reason', '')})"
                )
            elif self.swap_result.get("success"):
                lines.append(
                    f"Swap: complete (downtime "
                    f"{self.swap_result.get('downtime_seconds', 0):.1f}s)"
                )
            else:
                lines.append(f"Swap: failed ({self.swap_result.get('error', '')})")
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class SymbioteCoordinator:
    """Top-level orchestrator. State persists across daemon restarts."""

    def __init__(
        self,
        scout_engine: ScoutEngine,
        harvest_engine: HarvestEngine,
        graft_engine: GraftEngine,
        *,
        db_path: Path | None = None,
    ) -> None:
        self._scout = scout_engine
        self._harvest = harvest_engine
        self._graft = graft_engine
        if db_path is None:
            db_path = get_config_dir() / "symbiote" / "sessions.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        # check_same_thread=False because async tasks may run on different threads
        # depending on event-loop policy; SQLite calls are guarded by `self._lock`.
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_scout(self, problem_statement: str) -> SymbioteSession:
        """Run Scout. Returns a session in AWAITING_SCOUT_APPROVAL."""
        if self._get_active_session() is not None:
            raise RuntimeError("Another SYMBIOTE session is already active")
        session = SymbioteSession(
            session_id=uuid4().hex,
            problem_statement=problem_statement,
            phase=SymbiotePhase.SCOUTING,
            started_at=_now_iso(),
            updated_at=_now_iso(),
        )
        self._persist(session)
        try:
            report = await self._scout.scout(problem_statement)
        except Exception as exc:
            session.phase = SymbiotePhase.FAILED
            session.error = f"scout failed: {exc}"
            session.updated_at = _now_iso()
            self._persist(session)
            log.exception("SymbioteCoordinator: scout failed")
            return session
        session.scout_report = report.to_dict()
        session.phase = SymbiotePhase.AWAITING_SCOUT_APPROVAL
        session.updated_at = _now_iso()
        self._persist(session)
        return session

    async def approve_scout(
        self,
        session_id: str,
        candidate_full_name: str,
    ) -> SymbioteSession:
        """User selected a candidate. Run Harvest. Returns session in
        AWAITING_HARVEST_APPROVAL (or FAILED)."""
        session = self._require_session(session_id)
        if session.phase != SymbiotePhase.AWAITING_SCOUT_APPROVAL:
            raise RuntimeError(
                f"Cannot approve scout from phase {session.phase.value}"
            )
        candidate = self._find_candidate(session, candidate_full_name)
        if candidate is None:
            raise RuntimeError(f"Candidate {candidate_full_name!r} not in scout report")
        if candidate.get("recommendation") == "blocked":
            session.phase = SymbiotePhase.FAILED
            session.error = (
                f"Candidate {candidate_full_name} blocked by license gate"
            )
            session.updated_at = _now_iso()
            self._persist(session)
            return session
        license_check = self._rebuild_license_check(candidate.get("license_check") or {})
        session.selected_candidate = candidate_full_name
        session.phase = SymbiotePhase.HARVESTING
        session.updated_at = _now_iso()
        self._persist(session)
        try:
            report = await self._harvest.harvest(
                repo_full_name=candidate_full_name,
                repo_url=candidate.get("url", ""),
                problem_statement=session.problem_statement,
                license_check=license_check,
                repo_size_kb=candidate.get("size_kb"),
            )
        except Exception as exc:
            session.phase = SymbiotePhase.FAILED
            session.error = f"harvest failed: {exc}"
            session.updated_at = _now_iso()
            self._persist(session)
            log.exception("SymbioteCoordinator: harvest failed")
            return session
        session.harvest_report = report.to_dict()
        if report.aborted:
            session.phase = SymbiotePhase.FAILED
            session.error = f"harvest aborted: {report.abort_reason}"
        else:
            session.phase = SymbiotePhase.AWAITING_HARVEST_APPROVAL
        session.updated_at = _now_iso()
        self._persist(session)
        return session

    async def approve_harvest(self, session_id: str) -> SymbioteSession:
        """User approved harvest plan. Run Graft."""
        session = self._require_session(session_id)
        if session.phase != SymbiotePhase.AWAITING_HARVEST_APPROVAL:
            raise RuntimeError(
                f"Cannot approve harvest from phase {session.phase.value}"
            )
        if not session.harvest_report:
            session.phase = SymbiotePhase.FAILED
            session.error = "no harvest report on session"
            session.updated_at = _now_iso()
            self._persist(session)
            return session

        session.phase = SymbiotePhase.GRAFTING
        session.updated_at = _now_iso()
        self._persist(session)

        # Rebuild a HarvestReport-shaped object enough for GraftEngine.
        # GraftEngine reads .repo_full_name / .repo_url / .license / .modules_extracted
        # / .adaptation_plan / .aborted / .abort_reason — we reconstruct from JSON.
        try:
            harvest = self._rebuild_harvest_report(session.harvest_report)
            report = await self._graft.graft(harvest)
        except Exception as exc:
            session.phase = SymbiotePhase.FAILED
            session.error = f"graft failed: {exc}"
            session.updated_at = _now_iso()
            self._persist(session)
            log.exception("SymbioteCoordinator: graft failed")
            return session
        session.graft_report = report.to_dict()
        if report.aborted:
            session.phase = SymbiotePhase.FAILED
            session.error = f"graft aborted: {report.abort_reason}"
        else:
            session.phase = SymbiotePhase.AWAITING_GRAFT_APPROVAL
        session.updated_at = _now_iso()
        self._persist(session)
        return session

    async def approve_graft(self, session_id: str) -> SymbioteSession:
        """User accepted the graft. Mark session COMPLETE (no MORPH).

        Session B's MORPH path is opt-in via ``start_morph()`` from this
        same phase — call this method when the graft is enough and you
        do NOT want to hot-swap the daemon. Calling ``start_morph()``
        instead branches into the blue-green flow.
        """
        session = self._require_session(session_id)
        if session.phase != SymbiotePhase.AWAITING_GRAFT_APPROVAL:
            raise RuntimeError(
                f"Cannot approve graft from phase {session.phase.value}"
            )
        if session.graft_report and not session.graft_report.get("tests_passed"):
            session.phase = SymbiotePhase.FAILED
            session.error = "graft tests did not pass"
        else:
            session.phase = SymbiotePhase.COMPLETE
        session.updated_at = _now_iso()
        self._persist(session)
        return session

    async def start_morph(
        self,
        session_id: str,
        morph_engine: object,
    ) -> SymbioteSession:
        """User approved the graft AND wants to hot-swap.

        Runs ``morph_engine.prepare_candidate(graft_report=...)`` and
        records the resulting MorphReport on the session. Returns the
        session in ``AWAITING_SWAP_APPROVAL`` if ready, or ``FAILED`` if
        the candidate's tests didn't pass.
        """
        session = self._require_session(session_id)
        if session.phase != SymbiotePhase.AWAITING_GRAFT_APPROVAL:
            raise RuntimeError(
                f"Cannot start morph from phase {session.phase.value}"
            )

        session.phase = SymbiotePhase.MORPHING
        session.updated_at = _now_iso()
        self._persist(session)

        try:
            graft = self._rebuild_graft_report(session.graft_report)
            morph_report = await morph_engine.prepare_candidate(graft)  # type: ignore[attr-defined]
        except Exception as exc:
            session.phase = SymbiotePhase.FAILED
            session.error = f"morph prepare failed: {exc}"
            session.updated_at = _now_iso()
            self._persist(session)
            log.exception("SymbioteCoordinator: prepare_candidate failed")
            return session

        session.morph_report = morph_report.to_dict()
        session.backup_id = morph_report.backup_snapshot_id
        if morph_report.ready_to_swap:
            session.phase = SymbiotePhase.AWAITING_SWAP_APPROVAL
        else:
            session.phase = SymbiotePhase.FAILED
            session.error = (
                morph_report.reason_if_not_ready or "candidate not ready to swap"
            )
        session.updated_at = _now_iso()
        self._persist(session)
        return session

    async def approve_swap(
        self,
        session_id: str,
        morph_engine: object,
    ) -> SymbioteSession:
        """User approved the hot swap. Runs ``execute_swap`` and records result."""
        session = self._require_session(session_id)
        if session.phase != SymbiotePhase.AWAITING_SWAP_APPROVAL:
            raise RuntimeError(
                f"Cannot approve swap from phase {session.phase.value}"
            )
        if not session.morph_report:
            session.phase = SymbiotePhase.FAILED
            session.error = "no morph report on session"
            session.updated_at = _now_iso()
            self._persist(session)
            return session

        session.phase = SymbiotePhase.SWAPPING
        session.updated_at = _now_iso()
        self._persist(session)

        try:
            morph_report = self._rebuild_morph_report(session.morph_report)
            swap_result = await morph_engine.execute_swap(morph_report)  # type: ignore[attr-defined]
        except Exception as exc:
            session.phase = SymbiotePhase.FAILED
            session.error = f"swap execution raised: {exc}"
            session.updated_at = _now_iso()
            self._persist(session)
            log.exception("SymbioteCoordinator: execute_swap failed")
            return session

        session.swap_result = swap_result.to_dict()
        if swap_result.rolled_back:
            session.phase = SymbiotePhase.ROLLED_BACK
            session.error = swap_result.rollback_reason or "rolled back"
        elif swap_result.success:
            session.phase = SymbiotePhase.COMPLETE
        else:
            session.phase = SymbiotePhase.FAILED
            session.error = swap_result.error or "swap failed"
        session.updated_at = _now_iso()
        self._persist(session)
        return session

    async def abort(self, session_id: str) -> SymbioteSession:
        """Abort the session at any non-terminal phase."""
        session = self._require_session(session_id)
        if session.is_terminal():
            return session
        session.phase = SymbiotePhase.ABORTED
        session.updated_at = _now_iso()
        self._persist(session)
        return session

    def get_status(self, session_id: str | None = None) -> SymbioteSession | None:
        """Return the active session if no ID given; otherwise the named session."""
        if session_id is None:
            return self._get_active_session()
        return self._load_session(session_id)

    def get_history(self, limit: int = 10) -> list[SymbioteSession]:
        """Return up to `limit` most recent sessions, newest first."""
        with self._lock:
            cur = self._db.execute(
                "SELECT data FROM symbiote_sessions "
                "ORDER BY updated_at DESC LIMIT ?",
                (max(1, int(limit)),),
            )
            rows = cur.fetchall()
        sessions: list[SymbioteSession] = []
        for (data,) in rows:
            try:
                sessions.append(self._deserialize(data))
            except Exception:
                log.debug("get_history: failed to deserialize a row", exc_info=True)
        return sessions

    # ------------------------------------------------------------------
    # SQLite persistence
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS symbiote_sessions (
                    session_id TEXT PRIMARY KEY,
                    phase TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    data TEXT NOT NULL
                )
                """
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbiote_phase "
                "ON symbiote_sessions(phase)"
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbiote_updated "
                "ON symbiote_sessions(updated_at)"
            )
            self._db.commit()

    def _persist(self, session: SymbioteSession) -> None:
        payload = json.dumps(session.to_dict())
        with self._lock:
            self._db.execute(
                """
                INSERT INTO symbiote_sessions
                    (session_id, phase, started_at, updated_at, data)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    phase=excluded.phase,
                    updated_at=excluded.updated_at,
                    data=excluded.data
                """,
                (
                    session.session_id,
                    session.phase.value,
                    session.started_at,
                    session.updated_at,
                    payload,
                ),
            )
            self._db.commit()

    def _load_session(self, session_id: str) -> SymbioteSession | None:
        with self._lock:
            cur = self._db.execute(
                "SELECT data FROM symbiote_sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        try:
            return self._deserialize(row[0])
        except Exception:
            log.debug("Failed to load session %s", session_id, exc_info=True)
            return None

    def _get_active_session(self) -> SymbioteSession | None:
        terminal_values = ",".join(f"'{p.value}'" for p in _TERMINAL)
        with self._lock:
            cur = self._db.execute(
                "SELECT data FROM symbiote_sessions "
                f"WHERE phase NOT IN ({terminal_values}) "
                "ORDER BY updated_at DESC LIMIT 1"
            )
            row = cur.fetchone()
        if not row:
            return None
        try:
            return self._deserialize(row[0])
        except Exception:
            log.debug("Failed to load active session", exc_info=True)
            return None

    def _require_session(self, session_id: str) -> SymbioteSession:
        session = self._load_session(session_id)
        if session is None:
            raise RuntimeError(f"Unknown session_id {session_id!r}")
        return session

    @staticmethod
    def _deserialize(data: str) -> SymbioteSession:
        obj = json.loads(data)
        return SymbioteSession(
            session_id=obj["session_id"],
            problem_statement=obj.get("problem_statement", ""),
            phase=SymbiotePhase(obj.get("phase", SymbiotePhase.IDLE.value)),
            scout_report=obj.get("scout_report"),
            selected_candidate=obj.get("selected_candidate"),
            harvest_report=obj.get("harvest_report"),
            graft_report=obj.get("graft_report"),
            morph_report=obj.get("morph_report"),
            swap_result=obj.get("swap_result"),
            backup_id=obj.get("backup_id"),
            started_at=obj.get("started_at", ""),
            updated_at=obj.get("updated_at", ""),
            error=obj.get("error", ""),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_candidate(
        session: SymbioteSession,
        candidate_full_name: str,
    ) -> dict[str, Any] | None:
        report = session.scout_report or {}
        for c in report.get("candidates", []):
            if c.get("full_name") == candidate_full_name:
                return c
        return None

    @staticmethod
    def _rebuild_license_check(d: dict[str, Any]) -> LicenseCheck:
        verdict_str = d.get("verdict", LicenseVerdict.UNKNOWN.value)
        try:
            verdict = LicenseVerdict(verdict_str)
        except ValueError:
            verdict = LicenseVerdict.UNKNOWN
        return LicenseCheck(
            spdx_id=d.get("spdx_id"),
            verdict=verdict,
            source=d.get("source", "none"),
            obligations=list(d.get("obligations") or []),
            raw_text=d.get("raw_text"),
        )

    @staticmethod
    def _rebuild_graft_report(d: dict[str, Any] | None):
        """Reconstruct a GraftReport-shaped object from the stored dict."""
        from prometheus.symbiote.graft import GraftedFile, GraftReport
        if not d:
            return None
        files = [
            GraftedFile(
                path=f.get("path", ""),
                original_source=f.get("original_source", ""),
                lines_added=int(f.get("lines_added", 0)),
                provenance_header=f.get("provenance_header", ""),
                scan_verdict=f.get("scan_verdict", "clean"),
            )
            for f in d.get("files_created", []) or []
        ]
        return GraftReport(
            repo_full_name=d.get("repo_full_name", ""),
            files_created=files,
            files_modified=list(d.get("files_modified") or []),
            tests_added=list(d.get("tests_added") or []),
            tests_passed=bool(d.get("tests_passed", False)),
            tests_output=d.get("tests_output", ""),
            prometheus_md_updated=bool(d.get("prometheus_md_updated", False)),
            timestamp=d.get("timestamp", ""),
            aborted=bool(d.get("aborted", False)),
            abort_reason=d.get("abort_reason", ""),
        )

    @staticmethod
    def _rebuild_morph_report(d: dict[str, Any]):
        """Reconstruct a MorphReport from a stored dict."""
        from prometheus.symbiote.morph import MorphReport
        return MorphReport(
            backup_snapshot_id=d.get("backup_snapshot_id", ""),
            candidate_path=d.get("candidate_path", ""),
            graft_applied=bool(d.get("graft_applied", False)),
            tests_passed=bool(d.get("tests_passed", False)),
            tests_output=d.get("tests_output", ""),
            files_changed=list(d.get("files_changed") or []),
            new_dependencies=list(d.get("new_dependencies") or []),
            ready_to_swap=bool(d.get("ready_to_swap", False)),
            reason_if_not_ready=d.get("reason_if_not_ready", ""),
            timestamp=d.get("timestamp", ""),
        )

    @staticmethod
    def _rebuild_harvest_report(d: dict[str, Any]) -> HarvestReport:
        from prometheus.symbiote.harvest import (
            AdaptationStep,
            ExtractedModule,
        )

        license_check = SymbioteCoordinator._rebuild_license_check(
            d.get("license") or {}
        )
        modules = [
            ExtractedModule(
                original_path=m.get("original_path", ""),
                content=m.get("content", ""),
                description=m.get("description", ""),
                dependencies=list(m.get("dependencies") or []),
                line_count=int(m.get("line_count", 0)),
                scan_verdict=m.get("scan_verdict", "clean"),
                scan_findings=list(m.get("scan_findings") or []),
            )
            for m in d.get("modules_extracted") or []
        ]
        plan = [
            AdaptationStep(
                action=s.get("action", "create"),
                target_path=s.get("target_path", ""),
                description=s.get("description", ""),
                source_module=s.get("source_module", ""),
            )
            for s in d.get("adaptation_plan") or []
        ]
        return HarvestReport(
            repo_full_name=d.get("repo_full_name", ""),
            repo_url=d.get("repo_url", ""),
            license=license_check,
            problem_statement=d.get("problem_statement", ""),
            modules_extracted=modules,
            total_lines_extracted=int(d.get("total_lines_extracted", 0)),
            external_dependencies=list(d.get("external_dependencies") or []),
            adaptation_plan=plan,
            security_scan_summary=d.get("security_scan_summary", ""),
            sandbox_path=d.get("sandbox_path", ""),
            harvest_dir=d.get("harvest_dir", ""),
            timestamp=d.get("timestamp", ""),
            aborted=bool(d.get("aborted", False)),
            abort_reason=d.get("abort_reason", ""),
        )


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
