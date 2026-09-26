"""The Curator ages a skill on its last LOAD, not on its file mtime.

Before this, ``_discover_skills`` computed ``days_ago`` from mtime and handed
it to the model as ``last_used_days_ago``. No skill had ever been loaded, so
"last used" was really "last edited": 35 active→stale transitions and 22
prunings on the mini were decided on file age (docs/audits/SKILL-USAGE.md §6).

Now "last used" is the last recorded load (``skill_load_stats``), and a skill
with no load data is never stale and never pruned — absence of loads since the
counter began is not evidence that a skill is unused, only that nobody has
measured it yet.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prometheus.learning.curator import Curator
from prometheus.learning.skill_state import (
    SKILL_STATE_ACTIVE,
    SKILL_STATE_ARCHIVED,
    SKILL_STATE_STALE,
    SkillRecord,
    SkillStateStore,
)

DAY = 86400.0


class _Telemetry:
    """Stands in for ToolCallTelemetry: load stats plus the write calls the Curator makes."""

    def __init__(self, stats: dict) -> None:
        self.stats = stats

    def skill_load_stats(self) -> dict:
        return self.stats

    def record_run(self, *args, **kwargs) -> None:
        pass

    def record_silent_failure(self, *args, **kwargs) -> None:
        pass


def _skill(auto: Path, stem: str, *, name: str | None = None, age_days: float = 0.0) -> Path:
    path = auto / f"{stem}.md"
    path.write_text(f"---\nname: {name or stem}\ndescription: does {stem}\n---\n# {stem}\n")
    when = time.time() - age_days * DAY
    os.utime(path, (when, when))
    return path


def _loaded(days_ago: float, *, file: str | None = None) -> dict:
    return {"loads": 1, "last_loaded_at": time.time() - days_ago * DAY, "source": "auto",
            "file": file}


def _curator(tmp_path: Path, stats: dict | None) -> tuple[Curator, Path, SkillStateStore]:
    auto = tmp_path / "auto"
    auto.mkdir(exist_ok=True)
    store = SkillStateStore(tmp_path / "state.json")
    curator = Curator(
        MagicMock(), state_store=store, auto_dir=auto, reports_dir=tmp_path / "reports",
        telemetry=None if stats is None else _Telemetry(stats),
    )
    return curator, auto, store


def _by_name(skills: list[dict]) -> dict[str, dict]:
    return {s["name"]: s for s in skills}


class TestLastUsedIsTheLastLoad:
    def test_an_old_file_loaded_two_days_ago_is_active(self, tmp_path):
        curator, auto, _ = _curator(tmp_path, {"old-file": _loaded(2)})
        _skill(auto, "old-file", age_days=200)
        [s] = curator._discover_skills()
        assert s["days_ago"] == 2
        assert curator._target_state(s["days_ago"]) == SKILL_STATE_ACTIVE

    def test_a_fresh_file_last_loaded_forty_days_ago_goes_stale(self, tmp_path):
        curator, auto, _ = _curator(tmp_path, {"fresh-file": _loaded(40)})
        _skill(auto, "fresh-file", age_days=0)
        [s] = curator._discover_skills()
        assert s["days_ago"] == 40
        assert curator._target_state(s["days_ago"]) == SKILL_STATE_STALE

    def test_a_load_a_hundred_days_ago_is_archive_age(self, tmp_path):
        curator, auto, _ = _curator(tmp_path, {"x": _loaded(100)})
        _skill(auto, "x")
        [s] = curator._discover_skills()
        assert curator._target_state(s["days_ago"]) == SKILL_STATE_ARCHIVED

    def test_loads_are_matched_by_frontmatter_name_or_file(self, tmp_path):
        curator, auto, _ = _curator(tmp_path, {
            "release-check": _loaded(5),
            "other-name": _loaded(7, file="stem-b"),
        })
        _skill(auto, "release-check-1790000000", name="release-check")
        _skill(auto, "stem-b", name="something-else")
        skills = _by_name(curator._discover_skills())
        assert skills["release-check-1790000000"]["days_ago"] == 5
        assert skills["stem-b"]["days_ago"] == 7


class TestNoLoadDataIsNeverStale:
    def test_a_never_loaded_ancient_file_stays_active(self, tmp_path):
        curator, auto, _ = _curator(tmp_path, {})
        _skill(auto, "ancient", age_days=400)
        [s] = curator._discover_skills()
        assert s["days_ago"] is None
        assert curator._target_state(None) == SKILL_STATE_ACTIVE
        assert curator._apply_auto_transitions([s], dry_run=False) == []

    def test_a_legacy_mtime_stale_state_is_restored_to_active(self, tmp_path):
        curator, auto, store = _curator(tmp_path, {})
        _skill(auto, "was-stale", age_days=45)
        store.upsert_skill("was-stale", SkillRecord(state=SKILL_STATE_STALE))
        transitions = curator._apply_auto_transitions(curator._discover_skills(), dry_run=False)
        assert [(t["from_state"], t["to_state"]) for t in transitions] == [
            (SKILL_STATE_STALE, SKILL_STATE_ACTIVE)]
        assert store.get_skill("was-stale").state == SKILL_STATE_ACTIVE

    def test_without_telemetry_there_is_no_load_data(self, tmp_path):
        curator, auto, _ = _curator(tmp_path, None)
        _skill(auto, "x", age_days=400)
        [s] = curator._discover_skills()
        assert s["days_ago"] is None

    def test_the_model_is_told_never_not_a_file_age(self, tmp_path):
        curator, auto, _ = _curator(tmp_path, {"used": _loaded(3)})
        _skill(auto, "never", age_days=90)
        _skill(auto, "used", age_days=90)
        text = curator._format_library_for_prompt(curator._discover_skills())
        assert "last_used_days_ago: never (no load recorded)" in text
        assert "last_used_days_ago: 3" in text
        assert "last_used_days_ago: 90" not in text


class TestPruning:
    def _run(self, curator: Curator, yaml_text: str, monkeypatch) -> object:
        async def fake_call_model(self, prompt):
            return yaml_text

        monkeypatch.setattr(Curator, "_call_model", fake_call_model)
        return asyncio.run(curator.run_once())

    def test_a_never_loaded_skill_is_never_pruned(self, tmp_path, monkeypatch):
        curator, auto, _ = _curator(tmp_path, {})
        _skill(auto, "never-loaded", age_days=400)
        run = self._run(curator, "```yaml\nprunings:\n  - name: never-loaded\n"
                                 "    reason: stale\n```", monkeypatch)
        assert (auto / "never-loaded.md").exists()
        assert run.prunings == []
        assert run.skipped_no_load_data == ["never-loaded"]

    def test_a_loaded_then_stale_skill_can_still_be_pruned(self, tmp_path, monkeypatch):
        curator, auto, _ = _curator(tmp_path, {"went-stale": _loaded(45)})
        _skill(auto, "went-stale")
        run = self._run(curator, "```yaml\nprunings:\n  - name: went-stale\n"
                                 "    reason: stale and superseded\n```", monkeypatch)
        assert not (auto / "went-stale.md").exists()
        assert (auto / ".archive" / "went-stale.md").exists()
        assert [p["name"] for p in run.prunings] == ["went-stale"]

    def test_the_report_says_never_loaded(self, tmp_path, monkeypatch):
        curator, auto, _ = _curator(tmp_path, {})
        _skill(auto, "never-loaded")
        run = self._run(curator, "```yaml\nprunings:\n  - name: never-loaded\n"
                                 "    reason: x\n```", monkeypatch)
        report = Path(run.report_path).read_text()
        assert "never-loaded" in report
        assert "no load recorded" in report
