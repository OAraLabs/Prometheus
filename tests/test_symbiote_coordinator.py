"""SymbioteCoordinator — state machine, mutex, persistence, abort."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from prometheus.symbiote.coordinator import (
    SymbioteCoordinator,
    SymbiotePhase,
    SymbioteSession,
)
from prometheus.symbiote.graft import GraftReport
from prometheus.symbiote.harvest import HarvestReport
from prometheus.symbiote.license_gate import LicenseCheck, LicenseVerdict
from prometheus.symbiote.scout import (
    ScoutCandidate,
    ScoutEngine,
    ScoutReport,
)


def _scout_report_with_candidate(name: str = "alice/mit-tool") -> ScoutReport:
    return ScoutReport(
        problem_statement="x",
        candidates=[
            ScoutCandidate(
                full_name=name,
                url="https://example.com/" + name,
                description="desc",
                stars=100,
                language="Python",
                license_check=LicenseCheck(
                    spdx_id="MIT",
                    verdict=LicenseVerdict.ALLOW,
                    source="github_api",
                ),
                last_pushed=None,
                topics=[],
                size_kb=50,
                readme_excerpt="",
                relevance_score=0.8,
                relevance_rationale="",
                recommendation="recommended",
            ),
        ],
        search_queries_used=["q"],
        timestamp="2026-04-25T00:00:00Z",
        total_found=1,
    )


def _harvest_report(
    *, repo_full_name: str = "alice/mit-tool", aborted: bool = False
) -> HarvestReport:
    return HarvestReport(
        repo_full_name=repo_full_name,
        repo_url="https://example.com/" + repo_full_name,
        license=LicenseCheck(
            spdx_id="MIT", verdict=LicenseVerdict.ALLOW, source="github_api",
        ),
        problem_statement="x",
        modules_extracted=[],
        total_lines_extracted=0,
        external_dependencies=[],
        adaptation_plan=[],
        security_scan_summary="all clean",
        sandbox_path="",
        harvest_dir="",
        timestamp="2026-04-25T00:00:00Z",
        aborted=aborted,
    )


def _stub_scout(report: ScoutReport):
    scout = AsyncMock()
    scout.scout = AsyncMock(return_value=report)
    return scout


def _stub_harvest(report: HarvestReport):
    harvest = AsyncMock()
    harvest.harvest = AsyncMock(return_value=report)
    return harvest


def _stub_graft(report: GraftReport):
    graft = AsyncMock()
    graft.graft = AsyncMock(return_value=report)
    return graft


def _coordinator(tmp_path, scout=None, harvest=None, graft=None) -> SymbioteCoordinator:
    return SymbioteCoordinator(
        scout_engine=scout or _stub_scout(_scout_report_with_candidate()),
        harvest_engine=harvest or _stub_harvest(_harvest_report()),
        graft_engine=graft or _stub_graft(GraftReport(
            repo_full_name="alice/mit-tool",
            tests_passed=True,
            timestamp="2026-04-25T00:00:00Z",
        )),
        db_path=tmp_path / "sessions.db",
    )


class TestStateMachine:
    def test_start_scout_transitions(self, tmp_path):
        coord = _coordinator(tmp_path)
        session = asyncio.run(coord.start_scout("yaml validation"))
        assert session.phase == SymbiotePhase.AWAITING_SCOUT_APPROVAL
        assert session.scout_report is not None

    def test_approve_scout_runs_harvest(self, tmp_path):
        coord = _coordinator(tmp_path)
        session = asyncio.run(coord.start_scout("x"))
        session2 = asyncio.run(coord.approve_scout(session.session_id, "alice/mit-tool"))
        assert session2.phase == SymbiotePhase.AWAITING_HARVEST_APPROVAL
        assert session2.selected_candidate == "alice/mit-tool"

    def test_approve_unknown_candidate_raises(self, tmp_path):
        coord = _coordinator(tmp_path)
        session = asyncio.run(coord.start_scout("x"))
        with pytest.raises(RuntimeError):
            asyncio.run(coord.approve_scout(session.session_id, "no/such-repo"))

    def test_full_pipeline_to_complete(self, tmp_path):
        # Provide a non-empty adaptation plan so the harvest report is "valid"
        # for the graft engine; the graft step is stubbed anyway.
        coord = _coordinator(tmp_path)
        session = asyncio.run(coord.start_scout("x"))
        session = asyncio.run(coord.approve_scout(session.session_id, "alice/mit-tool"))
        # approve_harvest stubs out graft; it returns AWAITING_GRAFT_APPROVAL.
        # Manually set graft_report to something the rebuilder can use.
        session = asyncio.run(coord.approve_harvest(session.session_id))
        assert session.phase == SymbiotePhase.AWAITING_GRAFT_APPROVAL
        session = asyncio.run(coord.approve_graft(session.session_id))
        assert session.phase == SymbiotePhase.COMPLETE

    def test_aborted_harvest_marks_session_failed(self, tmp_path):
        coord = _coordinator(
            tmp_path,
            harvest=_stub_harvest(_harvest_report(aborted=True)),
        )
        coord._harvest.harvest.return_value = _harvest_report(aborted=True)
        session = asyncio.run(coord.start_scout("x"))
        # Rebuild the harvest report so .aborted=True surfaces.
        from prometheus.symbiote.harvest import HarvestReport
        coord._harvest.harvest = AsyncMock(return_value=HarvestReport(
            repo_full_name="alice/mit-tool",
            repo_url="https://x",
            license=LicenseCheck(spdx_id="MIT", verdict=LicenseVerdict.ALLOW, source="x"),
            problem_statement="x",
            modules_extracted=[],
            total_lines_extracted=0,
            external_dependencies=[],
            adaptation_plan=[],
            security_scan_summary="x",
            sandbox_path="",
            harvest_dir="",
            timestamp="x",
            aborted=True,
            abort_reason="bad scan",
        ))
        session = asyncio.run(coord.approve_scout(session.session_id, "alice/mit-tool"))
        assert session.phase == SymbiotePhase.FAILED
        assert "bad scan" in (session.error or "")


class TestMutex:
    def test_only_one_active_session(self, tmp_path):
        coord = _coordinator(tmp_path)
        asyncio.run(coord.start_scout("first"))
        with pytest.raises(RuntimeError, match="already active"):
            asyncio.run(coord.start_scout("second"))

    def test_can_start_after_abort(self, tmp_path):
        coord = _coordinator(tmp_path)
        s = asyncio.run(coord.start_scout("first"))
        asyncio.run(coord.abort(s.session_id))
        # Now a new session should be allowed.
        s2 = asyncio.run(coord.start_scout("second"))
        assert s2.session_id != s.session_id


class TestPersistence:
    def test_session_persists_across_instances(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        scout_report = _scout_report_with_candidate()
        coord1 = SymbioteCoordinator(
            scout_engine=_stub_scout(scout_report),
            harvest_engine=_stub_harvest(_harvest_report()),
            graft_engine=_stub_graft(GraftReport(
                repo_full_name="x", timestamp="x",
            )),
            db_path=db_path,
        )
        session = asyncio.run(coord1.start_scout("y"))
        del coord1
        # Reopen with a fresh coordinator pointed at the same DB.
        coord2 = SymbioteCoordinator(
            scout_engine=_stub_scout(scout_report),
            harvest_engine=_stub_harvest(_harvest_report()),
            graft_engine=_stub_graft(GraftReport(
                repo_full_name="x", timestamp="x",
            )),
            db_path=db_path,
        )
        loaded = coord2.get_status(session.session_id)
        assert loaded is not None
        assert loaded.problem_statement == "y"
        assert loaded.phase == SymbiotePhase.AWAITING_SCOUT_APPROVAL


class TestAbort:
    def test_abort_terminal_session_no_change(self, tmp_path):
        coord = _coordinator(tmp_path)
        session = asyncio.run(coord.start_scout("x"))
        asyncio.run(coord.abort(session.session_id))
        # abort again is a no-op.
        s2 = asyncio.run(coord.abort(session.session_id))
        assert s2.phase == SymbiotePhase.ABORTED


# ---------------------------------------------------------------------------
# Session B — MORPH / SWAP state transitions
# ---------------------------------------------------------------------------


def _morph_report(*, ready: bool = True, backup_id: str = "v42_test"):
    """Build a MorphReport stand-in compatible with the engine stub."""
    from prometheus.symbiote.morph import MorphReport
    return MorphReport(
        backup_snapshot_id=backup_id,
        candidate_path="/tmp/candidate",
        graft_applied=True,
        tests_passed=ready,
        tests_output="ok" if ready else "FAIL",
        files_changed=[],
        new_dependencies=[],
        ready_to_swap=ready,
        reason_if_not_ready="" if ready else "tests failed",
        timestamp="2026-04-26T00:00:00Z",
    )


def _swap_result(*, success: bool = True, rolled_back: bool = False):
    from prometheus.symbiote.morph import SwapResult
    return SwapResult(
        success=success and not rolled_back,
        previous_version="v42_test",
        new_version="post_swap_1" if success and not rolled_back else "",
        health_check_passed=success and not rolled_back,
        rolled_back=rolled_back,
        rollback_reason="health failed" if rolled_back else "",
        downtime_seconds=2.5,
        timestamp="2026-04-26T00:01:00Z",
    )


class TestMorphTransitions:
    def _drive_to_graft_approval(self, tmp_path):
        coord = _coordinator(tmp_path)
        # Need a non-empty graft_report so _rebuild_graft_report works.
        graft_stub = GraftReport(
            repo_full_name="alice/mit-tool",
            tests_passed=True,
            timestamp="x",
        )
        coord._graft.graft = AsyncMock(return_value=graft_stub)
        session = asyncio.run(coord.start_scout("x"))
        session = asyncio.run(coord.approve_scout(session.session_id, "alice/mit-tool"))
        session = asyncio.run(coord.approve_harvest(session.session_id))
        return coord, session

    def test_start_morph_transitions_to_swap_approval(self, tmp_path):
        coord, session = self._drive_to_graft_approval(tmp_path)
        morph_engine = AsyncMock()
        morph_engine.prepare_candidate = AsyncMock(return_value=_morph_report(ready=True))
        new = asyncio.run(coord.start_morph(session.session_id, morph_engine))
        assert new.phase == SymbiotePhase.AWAITING_SWAP_APPROVAL
        assert new.morph_report is not None
        assert new.backup_id == "v42_test"

    def test_start_morph_fails_when_candidate_not_ready(self, tmp_path):
        coord, session = self._drive_to_graft_approval(tmp_path)
        morph_engine = AsyncMock()
        morph_engine.prepare_candidate = AsyncMock(return_value=_morph_report(ready=False))
        new = asyncio.run(coord.start_morph(session.session_id, morph_engine))
        assert new.phase == SymbiotePhase.FAILED
        assert "tests failed" in (new.error or "")

    def test_start_morph_rejects_wrong_phase(self, tmp_path):
        coord = _coordinator(tmp_path)
        session = asyncio.run(coord.start_scout("x"))
        morph_engine = AsyncMock()
        with pytest.raises(RuntimeError):
            asyncio.run(coord.start_morph(session.session_id, morph_engine))

    def test_approve_swap_success_marks_complete(self, tmp_path):
        coord, session = self._drive_to_graft_approval(tmp_path)
        morph_engine = AsyncMock()
        morph_engine.prepare_candidate = AsyncMock(return_value=_morph_report(ready=True))
        morph_engine.execute_swap = AsyncMock(return_value=_swap_result(success=True))
        session = asyncio.run(coord.start_morph(session.session_id, morph_engine))
        new = asyncio.run(coord.approve_swap(session.session_id, morph_engine))
        assert new.phase == SymbiotePhase.COMPLETE
        assert new.swap_result is not None

    def test_approve_swap_rolled_back(self, tmp_path):
        coord, session = self._drive_to_graft_approval(tmp_path)
        morph_engine = AsyncMock()
        morph_engine.prepare_candidate = AsyncMock(return_value=_morph_report(ready=True))
        morph_engine.execute_swap = AsyncMock(return_value=_swap_result(rolled_back=True))
        session = asyncio.run(coord.start_morph(session.session_id, morph_engine))
        new = asyncio.run(coord.approve_swap(session.session_id, morph_engine))
        assert new.phase == SymbiotePhase.ROLLED_BACK

    def test_approve_swap_rejects_wrong_phase(self, tmp_path):
        coord, session = self._drive_to_graft_approval(tmp_path)
        morph_engine = AsyncMock()
        with pytest.raises(RuntimeError):
            asyncio.run(coord.approve_swap(session.session_id, morph_engine))

    def test_session_to_dict_includes_morph_fields(self, tmp_path):
        coord, session = self._drive_to_graft_approval(tmp_path)
        morph_engine = AsyncMock()
        morph_engine.prepare_candidate = AsyncMock(return_value=_morph_report(ready=True))
        new = asyncio.run(coord.start_morph(session.session_id, morph_engine))
        d = new.to_dict()
        assert "morph_report" in d
        assert d["backup_id"] == "v42_test"
        # Phase serialized as string.
        assert d["phase"] == "awaiting_swap_approval"
