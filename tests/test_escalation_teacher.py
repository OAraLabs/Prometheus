"""Teacher escalation engine — side-effect tests (files on disk, db rows).

Recorded teacher fixtures only (tests/fixtures/escalation/teacher_reply_*.md);
no live network. The fake provider streams the fixture through the REAL
LLMCallEnvelope so envelope telemetry (subsystem_runs) is exercised too.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from prometheus.escalation.teacher import (
    EscalationOutcome,
    TeacherEscalation,
    parse_teacher_sections,
)
from prometheus.learning.skill_creator import SkillCreator
from prometheus.providers.base import ApiTextDeltaEvent
from prometheus.telemetry.tracker import ToolCallTelemetry

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "escalation"

GOOD_TEACHER = (FIXTURE_DIR / "teacher_reply_good.md").read_text(encoding="utf-8")
STALLED_TEACHER = (FIXTURE_DIR / "teacher_reply_stalled.md").read_text(encoding="utf-8")
MISSING_SKILL_TEACHER = (
    FIXTURE_DIR / "teacher_reply_missing_skill.md").read_text(encoding="utf-8")

# A failed local turn (mirrors pos_unrecovered_error.json).
FAILED_RESULTS = [
    {"tool_name": "bash", "arguments": {"command": "./deploy.sh"},
     "result": "Error: connection refused (deploy target unreachable)",
     "is_error": True},
]
FAILED_REPLY = "The deploy script could not reach the server."
REQUEST = "Deploy the new build to the staging box."


class _FakeTeacherProvider:
    """Streams a recorded teacher response; counts calls (side effect)."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    async def stream_message(self, request):  # noqa: ANN001 — provider duck type
        self.calls += 1
        yield ApiTextDeltaEvent(text=self._text)


def _build(tmp_path: Path, teacher_text: str, *, max_per_session: int = 3):
    telemetry = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    provider = _FakeTeacherProvider(teacher_text)
    auto_dir = tmp_path / "auto"
    auto_dir.mkdir(parents=True, exist_ok=True)
    creator = SkillCreator(
        provider, model="teacher-test", auto_dir=auto_dir, telemetry=telemetry)
    engine = TeacherEscalation(
        teacher_model="teacher-test",
        teacher_provider="anthropic",
        max_per_session=max_per_session,
        telemetry=telemetry,
        provider=provider,
        skill_creator=creator,
    )
    return engine, telemetry, auto_dir, provider


async def _escalate(engine: TeacherEscalation, **overrides):
    kwargs = dict(
        session_id="telegram:1",
        user_request=REQUEST,
        tool_results=FAILED_RESULTS,
        final_reply=FAILED_REPLY,
        agent_mode=True,
        primary_provider="llama_cpp",
    )
    kwargs.update(overrides)
    return await engine.maybe_escalate(**kwargs)


def _signal_rows(tmp_path: Path) -> list[tuple[str, dict, str]]:
    conn = sqlite3.connect(tmp_path / "telemetry.db")
    try:
        rows = conn.execute(
            "SELECT signal_type, payload, source_subsystem FROM signal_events"
            " ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return [(t, json.loads(p), s) for t, p, s in rows]


# ---------------------------------------------------------------------------
# Wiring: skill file EXISTS on disk, telemetry row EXISTS with correct tags
# ---------------------------------------------------------------------------

async def test_wiring_escalation_writes_skill_and_golden_trace(tmp_path):
    engine, _tel, auto_dir, provider = _build(tmp_path, GOOD_TEACHER)
    outcome = await _escalate(engine)

    assert isinstance(outcome, EscalationOutcome)
    assert outcome.status == "escalated"
    assert "connection" in (outcome.corrective_reply or "")
    assert "teacher" in outcome.note  # no silent substitution

    # Side effect 1: the SKILL.md file EXISTS with the fixture's content.
    skill_file = auto_dir / "diagnose-unreachable-deploy-target.md"
    assert skill_file.exists()
    content = skill_file.read_text(encoding="utf-8")
    assert content.startswith("---\nname: diagnose-unreachable-deploy-target")
    assert "## Steps" in content
    assert outcome.skill_path == str(skill_file)

    # Side effect 2: the golden-trace row EXISTS with correct tags.
    rows = _signal_rows(tmp_path)
    assert len(rows) == 1
    signal_type, payload, source = rows[0]
    assert signal_type == "teacher_escalation"
    assert source == "teacher_escalation"
    assert payload["source"] == "teacher_escalation"
    assert payload["status"] == "escalated"
    assert payload["skill_persisted"] is True
    assert payload["skill_path"] == str(skill_file)
    assert payload["detector_reasons"]  # non-empty
    assert payload["matched_patterns"] == ["unrecovered_tool_error"]
    assert payload["teacher_model"] == "teacher-test"

    # Side effect 3: the envelope recorded the teacher call (liveness row).
    conn = sqlite3.connect(tmp_path / "telemetry.db")
    runs = conn.execute(
        "SELECT subsystem, operation, outcome FROM subsystem_runs").fetchall()
    conn.close()
    assert ("teacher_escalation", "teacher_call", "success") in runs
    assert provider.calls == 1


# ---------------------------------------------------------------------------
# Gate: a teacher that fails the detector writes NOTHING
# ---------------------------------------------------------------------------

async def test_gate_teacher_failing_detector_persists_nothing(tmp_path):
    engine, _tel, auto_dir, _provider = _build(tmp_path, STALLED_TEACHER)
    outcome = await _escalate(engine)

    assert outcome.status == "teacher_failed"
    assert outcome.corrective_reply is None  # local reply stands
    assert outcome.skill_rejected_reasons  # detector reasons recorded
    assert "did not succeed" in outcome.note  # loud, visible

    assert list(auto_dir.iterdir()) == []  # NO skill file written

    rows = _signal_rows(tmp_path)
    assert len(rows) == 1
    _t, payload, _s = rows[0]
    assert payload["status"] == "teacher_failed"
    assert payload["skill_persisted"] is False
    assert payload["skill_rejected_reasons"]
    assert payload["failure"] == "teacher reply failed the detector"


async def test_missing_section_is_loud_and_persists_nothing(tmp_path):
    engine, _tel, auto_dir, _provider = _build(tmp_path, MISSING_SKILL_TEACHER)
    outcome = await _escalate(engine)

    assert outcome.status == "teacher_failed"
    assert outcome.corrective_reply is None
    assert list(auto_dir.iterdir()) == []

    # Loud: silent_failures row with the parse operation.
    conn = sqlite3.connect(tmp_path / "telemetry.db")
    fails = conn.execute(
        "SELECT subsystem, operation FROM silent_failures").fetchall()
    conn.close()
    assert ("teacher_escalation", "parse_sections") in fails

    rows = _signal_rows(tmp_path)
    assert rows[0][1]["status"] == "teacher_failed"
    assert "missing sections" in rows[0][1]["failure"]


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

async def test_fourth_escalation_in_session_is_refused_and_logged(tmp_path):
    engine, _tel, _auto, provider = _build(tmp_path, GOOD_TEACHER)

    for _ in range(3):
        outcome = await _escalate(engine)
        assert outcome.status == "escalated"

    fourth = await _escalate(engine)
    assert fourth.status == "refused_budget"
    assert provider.calls == 3  # the 4th attempt made NO teacher call

    rows = _signal_rows(tmp_path)
    assert len(rows) == 4
    assert rows[-1][1]["status"] == "refused_budget"
    assert "budget exhausted" in rows[-1][1]["failure"]
    assert engine.stats()["refused_budget"] == 1
    assert engine.stats()["sessions"] == {"telegram:1": 3}

    # A different session still has budget.
    other = await _escalate(engine, session_id="telegram:2")
    assert other.status == "escalated"


# ---------------------------------------------------------------------------
# Trigger conditions 1–4 block silently (None) with zero side effects
# ---------------------------------------------------------------------------

async def test_trigger_blocks_produce_no_side_effects(tmp_path):
    engine, _tel, auto_dir, provider = _build(tmp_path, GOOD_TEACHER)

    # 1. not agent mode
    assert await _escalate(engine, agent_mode=False) is None
    # 2. cloud primary
    assert await _escalate(engine, primary_provider="anthropic") is None
    # 4. healthy turn
    assert await _escalate(
        engine, tool_results=[], final_reply=(
            "Done — the deploy completed and the health check passed."
        )) is None

    # 3. teacher not configured (inert engine from empty config)
    inert = TeacherEscalation.from_config({}, telemetry=None)
    assert not inert.is_armed
    assert await _escalate(inert) is None

    assert provider.calls == 0
    assert list(auto_dir.iterdir()) == []
    assert _signal_rows(tmp_path) == []


def test_from_config_defaults_and_arming():
    assert TeacherEscalation.from_config(None).is_armed is False
    armed = TeacherEscalation.from_config(
        {"escalation": {"teacher_model": "claude-test", "max_per_session": 5}})
    assert armed.is_armed is True
    stats = armed.stats()
    assert stats["max_per_session"] == 5
    assert stats["teacher"] == "anthropic/claude-test"


# ---------------------------------------------------------------------------
# Section parser + skill-writer confinement
# ---------------------------------------------------------------------------

def test_parse_teacher_sections_good_and_missing():
    c, s, problems = parse_teacher_sections(GOOD_TEACHER)
    assert c and s and problems == []
    assert s.startswith("---\nname: diagnose-unreachable-deploy-target")

    c, s, problems = parse_teacher_sections(MISSING_SKILL_TEACHER)
    assert c is not None and s is None
    assert problems == ["SKILL_DRAFT section missing or empty"]

    c, s, problems = parse_teacher_sections("no sections at all")
    assert c is None and s is None and len(problems) == 2


def test_skill_draft_with_inner_code_fences_survives():
    raw = (
        "```CORRECTIVE_REPLY\nUse the helper script as shown below.\n```\n\n"
        "```SKILL_DRAFT\n---\nname: fenced-skill\ndescription: keeps fences\n"
        "---\n\n# Fenced\n\n```bash\necho hi\n```\n\nDone.\n```\n"
    )
    c, s, problems = parse_teacher_sections(raw)
    assert problems == []
    assert "```bash" in s and s.endswith("Done.")


async def test_hostile_skill_name_cannot_escape_auto_dir(tmp_path):
    """SecurityGate-intent test: the skill writer cannot touch paths outside
    the auto dir — a traversal-shaped name slugifies to a safe filename."""
    auto_dir = tmp_path / "auto"
    auto_dir.mkdir()
    creator = SkillCreator(
        _FakeTeacherProvider(""), model="x", auto_dir=auto_dir)
    hostile = (
        "---\nname: ../../config/prometheus\ndescription: hostile\n---\n# X\n"
    )
    path = await creator.persist_skill_content(hostile, trigger="t")
    assert path is not None
    assert path.parent == auto_dir
    assert path.name == "config-prometheus.md"
    assert not (tmp_path / "config").exists()


def test_detector_is_standalone_loadable():
    """BAKEOFF-harness.md imports detector.py by file path from a checkout —
    it must load without the prometheus package. (sys.modules registration is
    what any correct spec-based loader does; dataclasses requires it.)"""
    import sys

    import prometheus.escalation.detector as det

    spec = importlib.util.spec_from_file_location(
        "standalone_detector_check", det.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules["standalone_detector_check"] = module
    try:
        spec.loader.exec_module(module)
        verdict = module.detect_failure([], "Which file do you mean?")
        assert verdict.failed
        assert "clarification_stall" in verdict.matched_patterns
    finally:
        sys.modules.pop("standalone_detector_check", None)
