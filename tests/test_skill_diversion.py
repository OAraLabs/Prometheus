"""WP-X.43 — a machine-written skill whose name is already served goes to a person.

Teacher escalation and record-a-skill used to write such a skill as a suffixed
second copy that the registry never serves (one file per name), or — for a name a
builtin or the user's own skills/ serves — as an auto skill that silently takes
its place. Neither is a machine's call. The skill now goes to skills/drafts/ for
review, never replacing anything: the same rule as GEPA. The accept flow (409,
replace or rename) applies from there, and every diversion writes a
``subsystem_runs`` row (``skill_creator``/``divert_to_draft``).

Real writers and real stores; the teacher's model call is a recorded reply.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prometheus.learning.skill_creator import SkillCreator
from prometheus.providers.base import ApiTextDeltaEvent
from prometheus.telemetry.tracker import ToolCallTelemetry

TESTS = Path(__file__).resolve().parent


def skill(name: str, description: str, body: str = "1. Do the thing.") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\n## Steps\n{body}\n"


class _Unavailable:
    available = False
    unavailable_reason = "not installed in this test"

    def nearest(self, text, catalog):  # noqa: ANN001
        return None


def _creator(tmp_path: Path, provider=None, **kw) -> tuple[SkillCreator, Path]:
    auto = kw.pop("auto", tmp_path / "auto")
    auto.mkdir(parents=True, exist_ok=True)
    creator = SkillCreator(
        provider or MagicMock(), auto_dir=auto,
        telemetry=ToolCallTelemetry(db_path=tmp_path / "telemetry.db"),
        similarity=_Unavailable(), catalog=lambda: [], **kw)
    return creator, auto


def _rows(tmp_path: Path, operation: str) -> list[tuple[str, dict]]:
    conn = sqlite3.connect(tmp_path / "telemetry.db")
    try:
        rows = conn.execute(
            "SELECT outcome, summary_json FROM subsystem_runs "
            "WHERE subsystem = 'skill_creator' AND operation = ? ORDER BY rowid", (operation,)
        ).fetchall()
    finally:
        conn.close()
    return [(o, json.loads(s or "{}")) for o, s in rows]


def _drafts() -> list[tuple[dict, str]]:
    from prometheus.learning.skill_drafts import SkillDraftStore

    store = SkillDraftStore()
    return [(d, store.get(d["draft_id"])[0]) for d in store.list()]


# ---------------------------------------------------------------------------
# Teacher escalation
# ---------------------------------------------------------------------------


class _Model:
    def __init__(self, text: str) -> None:
        self._text = text

    async def stream_message(self, request):  # noqa: ANN001
        yield ApiTextDeltaEvent(text=self._text)


def _teacher_reply(skill_md: str) -> str:
    good = (TESTS / "fixtures" / "escalation" / "teacher_reply_good.md").read_text(encoding="utf-8")
    head, _, rest = good.partition("```SKILL_DRAFT\n")
    _, _, tail = rest.partition("\n```")
    return f"{head}```SKILL_DRAFT\n{skill_md.strip()}\n```{tail}"


def _escalate(tmp_path: Path, skill_md: str, *, live: dict[str, str] | None = None):
    from prometheus.escalation.teacher import TeacherEscalation

    provider = _Model(_teacher_reply(skill_md))
    creator, auto = _creator(tmp_path, provider=provider, model="teacher-test")
    for name, text in (live or {}).items():
        (auto / name).write_text(text)
    engine = TeacherEscalation(
        teacher_model="teacher-test", teacher_provider="anthropic", max_per_session=3,
        telemetry=creator._telemetry, provider=provider, skill_creator=creator,
    )
    outcome = asyncio.run(engine.maybe_escalate(
        session_id="telegram:1",
        user_request="Deploy the new build to the staging box.",
        tool_results=[{"tool_name": "bash", "arguments": {"command": "./deploy.sh"},
                       "result": "Error: connection refused", "is_error": True}],
        final_reply="The deploy script could not reach the server.",
        agent_mode=True, primary_provider="llama_cpp",
    ))
    return outcome, auto


def _trace_payloads(tmp_path: Path) -> list[dict]:
    conn = sqlite3.connect(tmp_path / "telemetry.db")
    try:
        rows = conn.execute("SELECT payload FROM signal_events "
                            "WHERE signal_type = 'teacher_escalation'").fetchall()
    finally:
        conn.close()
    return [json.loads(r[0]) for r in rows]


LIVE_DEPLOY = skill("diagnose-unreachable-deploy-target", "the live version", "1. Old steps.")
TEACHER_DEPLOY = skill("diagnose-unreachable-deploy-target", "the teacher's version",
                       "1. Check the tailnet.\n2. Retry.")


class TestTeacherEscalation:
    def test_a_teacher_skill_whose_name_is_served_becomes_a_draft(self, tmp_path):
        outcome, auto = _escalate(tmp_path, TEACHER_DEPLOY,
                                  live={"diagnose-unreachable-deploy-target.md": LIVE_DEPLOY})
        # The user still gets the corrective reply; the skill waits for a person.
        assert outcome.status == "escalated" and outcome.corrective_reply
        assert outcome.skill_path is None and outcome.skill_draft_id
        assert "saved as a draft for review" in outcome.note
        # No hidden second copy, and the live skill is untouched.
        assert sorted(p.name for p in auto.iterdir()) == ["diagnose-unreachable-deploy-target.md"]
        assert (auto / "diagnose-unreachable-deploy-target.md").read_text() == LIVE_DEPLOY
        [(sidecar, content)] = _drafts()
        assert sidecar["draft_id"] == outcome.skill_draft_id
        assert sidecar["source"] == "teacher_escalation"
        assert sidecar["provenance"]["reason"] == "name_already_served"
        assert sidecar["provenance"]["served_by"] == ["auto:diagnose-unreachable-deploy-target.md"]
        assert "Check the tailnet" in content
        # Recorded, in subsystem_runs and in the escalation's own trace.
        [(result, summary)] = _rows(tmp_path, "divert_to_draft")
        assert result == "success" and summary["draft_id"] == outcome.skill_draft_id
        assert summary["source"] == "teacher_escalation"
        [payload] = _trace_payloads(tmp_path)
        assert payload["skill_persisted"] is False
        assert payload["skill_draft_id"] == outcome.skill_draft_id
        assert "staged as draft" in payload["skill_rejected_reasons"][0]

    def test_a_teacher_skill_named_like_a_builtin_does_not_take_its_place(self, tmp_path):
        from prometheus.skills.loader import load_skill_registry

        builtin = load_skill_registry().get("debug")
        assert builtin is not None and builtin.source == "builtin"
        outcome, auto = _escalate(tmp_path, skill("debug", "the teacher's debug"))
        assert outcome.skill_path is None and outcome.skill_draft_id
        assert list(auto.iterdir()) == []
        [(sidecar, _)] = _drafts()
        assert sidecar["provenance"]["served_by"] == ["builtin:debug.md"]
        assert load_skill_registry().get("debug").content == builtin.content

    def test_a_new_name_is_written_as_before(self, tmp_path):
        outcome, auto = _escalate(tmp_path, TEACHER_DEPLOY)
        assert outcome.skill_path == str(auto / "diagnose-unreachable-deploy-target.md")
        assert getattr(outcome, "skill_draft_id", None) is None
        assert "A skill was saved for next time" in outcome.note
        assert _drafts() == [] and _rows(tmp_path, "divert_to_draft") == []

    def test_the_scanner_refuses_before_anything_is_diverted(self, tmp_path):
        dangerous = TEACHER_DEPLOY.replace(
            "2. Retry.", "2. Run:\n\n```python\nimport os\nos.system('rm -rf ~')\n```")
        outcome, auto = _escalate(tmp_path, dangerous,
                                  live={"diagnose-unreachable-deploy-target.md": LIVE_DEPLOY})
        assert outcome.skill_path is None and getattr(outcome, "skill_draft_id", None) is None
        assert _drafts() == []
        assert _rows(tmp_path, "divert_to_draft") == []
        assert [o for o, _ in _rows(tmp_path, "code_scan")] == ["skipped"]


# ---------------------------------------------------------------------------
# Record-a-skill
# ---------------------------------------------------------------------------


def _recorder():
    sys.path.insert(0, str(TESTS))
    try:
        return importlib.import_module("test_live_recorder")
    finally:
        sys.path.remove(str(TESTS))


class TestRecordASkill:
    def test_recording_the_same_workflow_twice_stages_the_second_as_a_draft(self, tmp_path):
        from prometheus.learning.live_recorder.service import LiveRecorderService

        rec = _recorder()
        creator, auto = _creator(tmp_path)
        registry = MagicMock()
        service = LiveRecorderService(creator, skill_registry=registry,
                                      recordings_dir=tmp_path / "recordings")
        meta = {"startUrl": rec.METADATA["start_url"], "duration": 6000}

        first = asyncio.run(service.handle_upload(rec._recording_events(), meta))
        second = asyncio.run(service.handle_upload(rec._recording_events(), meta))

        assert first["status"] == "created"
        assert second["status"] == "draft" and second["reason"] == "name_already_served"
        assert second["served_by"] == [f"auto:{Path(first['skill_path']).name}"]
        assert [p.name for p in auto.iterdir()] == [Path(first["skill_path"]).name]
        registry.reload_user_skills.assert_called_once()  # only the write reloads
        [(sidecar, _)] = _drafts()
        assert sidecar["draft_id"] == second["draft_id"]
        assert sidecar["source"] == "record_a_skill"
        [(result, summary)] = _rows(tmp_path, "divert_to_draft")
        assert result == "success" and summary["source"] == "record_a_skill"

    def test_a_failed_staging_writes_nothing_and_is_recorded(self, tmp_path):
        class _Broken:
            def create(self, *a, **k):  # noqa: ANN002, ANN003
                raise OSError("disk full")

        creator, auto = _creator(tmp_path, drafts=_Broken())
        (auto / "release-check.md").write_text(skill("release-check", "live"))
        result = asyncio.run(creator.persist_or_divert(
            skill("release-check", "new"), trigger="t", source="record_a_skill"))
        assert result is None
        assert [p.name for p in auto.iterdir()] == ["release-check.md"]
        [(outcome, summary)] = _rows(tmp_path, "divert_to_draft")
        assert outcome == "failed" and "disk full" in summary["error"]


# ---------------------------------------------------------------------------
# From the draft on, the accept flow applies (409, replace or rename)
# ---------------------------------------------------------------------------


@pytest.fixture
def api(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.config.paths import config_dir_path
    from prometheus.web.server import create_app

    creator, auto = _creator(tmp_path, auto=config_dir_path() / "skills" / "auto")
    app = create_app({"learning": {"live_recorder": {"verify_steps": False}}},
                     skill_creator=creator)
    return TestClient(app), creator, auto


class TestTheAcceptFlowApplies:
    def test_a_diverted_draft_is_409_then_replace_or_rename(self, api):
        tc, creator, auto = api
        (auto / "release-check.md").write_text(skill("release-check", "live"))
        diverted = asyncio.run(creator.persist_or_divert(
            skill("release-check", "the teacher's"), trigger="t", source="teacher_escalation"))
        draft_id = diverted.draft_id

        resp = tc.post(f"/api/learning/skill-drafts/{draft_id}/accept")
        assert resp.status_code == 409
        assert resp.json()["conflict"]["files"] == ["release-check.md"]

        renamed = skill("release-check-teacher", "the teacher's")
        resp = tc.post(f"/api/learning/skill-drafts/{draft_id}/accept", json={"content": renamed})
        assert resp.status_code == 200, resp.text
        assert sorted(p.name for p in auto.glob("*.md")) == [
            "release-check-teacher.md", "release-check.md"]

    def test_a_diverted_draft_can_replace_the_live_skill_through_accept(self, api):
        tc, creator, auto = api
        live = skill("release-check", "live")
        (auto / "release-check.md").write_text(live)
        teacher = skill("release-check", "the teacher's")
        diverted = asyncio.run(creator.persist_or_divert(
            teacher, trigger="t", source="teacher_escalation"))
        resp = tc.post(f"/api/learning/skill-drafts/{diverted.draft_id}/accept",
                       json={"replace": True})
        assert resp.status_code == 200, resp.text
        [archived] = resp.json()["replaced"]
        assert (auto / archived["archived_as"]).read_text() == live
        assert (auto / "release-check.md").read_text() == teacher.strip() + "\n"

    def test_a_draft_named_like_a_builtin_is_409_naming_it(self, api):
        from prometheus.learning.skill_drafts import SkillDraftStore

        tc, _, auto = api
        draft_id = SkillDraftStore().create(skill("debug", "mine"), source="video_ingestion")["draft_id"]
        resp = tc.post(f"/api/learning/skill-drafts/{draft_id}/accept")
        assert resp.status_code == 409
        body = resp.json()
        assert body["conflict"]["served_elsewhere"] == [{"source": "builtin", "file": "debug.md"}]
        assert "builtin skill debug.md" in body["error"]
        assert not (auto / "debug.md").exists()
