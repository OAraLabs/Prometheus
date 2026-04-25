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
