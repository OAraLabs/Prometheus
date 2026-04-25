"""GEPA — unit tests for the optimizer.

Real instances throughout. Provider and judge calls are stubbed via
``monkeypatch`` to avoid real LLM calls in CI; everything else (file I/O,
trace parsing, archive logic) runs against the real filesystem.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from prometheus.learning.gepa import GEPAOptimizer, GEPAReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_jsonl(path: Path, records: list[dict]) -> None:
    """Write ``records`` as JSONL to ``path``."""
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _skill_invocation_trace(skill_name: str, content: str = "ok") -> dict:
    """Build a JSONL trace record that looks like a Skill tool invocation."""
    parsed = {"name": "Skill", "input": {"skill": skill_name}}
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Call the `Skill` tool appropriately.\n"
                    f"Reference parsed call: {json.dumps(parsed)}"
                ),
            },
            {"role": "assistant", "content": content},
        ],
        "_meta": {
            "model": "test-model",
            "tool_name": "Skill",
            "timestamp": time.time(),
        },
    }


def _stub_optimizer(
    tmp_path: Path,
    *,
    config: dict | None = None,
) -> GEPAOptimizer:
    """Build a GEPAOptimizer rooted at tmp_path with sensible defaults."""
    base_cfg = {
        "gepa_enabled": True,
        "gepa_max_skills_per_cycle": 3,
        "gepa_variants_per_skill": 2,
        "gepa_min_traces_required": 1,
        "gepa_judge_threshold": 0.5,
    }
    if config:
        base_cfg.update(config)
    return GEPAOptimizer(
        provider=None,
        config=base_cfg,
        trajectories_dir=tmp_path / "trajectories",
        skills_auto_dir=tmp_path / "skills" / "auto",
    )


@dataclass
class _StubVerdict:
    score: float
    reasoning: str = ""
    raw_response: str = ""


class _StubJudge:
    """Minimal judge stand-in with a configurable score-by-content map."""

    def __init__(self, scoremap: dict[str, float] | None = None, default: float = 0.5) -> None:
        self.scoremap = scoremap or {}
        self.default = default
        self.calls: list[str] = []

    async def evaluate(
        self,
        task_input: str,
        agent_output: str,
        expected_behavior: str,
        tool_trace=None,
    ) -> _StubVerdict:
        self.calls.append(agent_output)
        for key, score in self.scoremap.items():
            if key in agent_output:
                return _StubVerdict(score=score, reasoning="stub")
        return _StubVerdict(score=self.default, reasoning="stub default")


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------


class TestLoadTraces:
    @pytest.mark.asyncio
    async def test_loads_recent_jsonl(self, tmp_path):
        """_load_recent_traces reads JSONL records from the most recent file."""
        opt = _stub_optimizer(tmp_path)
        opt._trajectories_dir.mkdir(parents=True, exist_ok=True)
        path = opt._trajectories_dir / "golden_traces_1.jsonl"
        _make_jsonl(path, [
            _skill_invocation_trace("alpha"),
            _skill_invocation_trace("beta"),
        ])

        traces = await opt._load_recent_traces()
        assert len(traces) == 2
        assert traces[0]["_meta"]["tool_name"] == "Skill"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_dir(self, tmp_path):
        opt = _stub_optimizer(tmp_path)
        # Don't create the dir — it should not blow up.
        traces = await opt._load_recent_traces()
        assert traces == []

    @pytest.mark.asyncio
    async def test_picks_latest_file(self, tmp_path):
        """When multiple golden_traces_*.jsonl files exist, picks the latest."""
        opt = _stub_optimizer(tmp_path)
        opt._trajectories_dir.mkdir(parents=True, exist_ok=True)
        old = opt._trajectories_dir / "golden_traces_100.jsonl"
        new = opt._trajectories_dir / "golden_traces_200.jsonl"
        _make_jsonl(old, [_skill_invocation_trace("old")])
        _make_jsonl(new, [_skill_invocation_trace("new1"), _skill_invocation_trace("new2")])

        traces = await opt._load_recent_traces()
        assert len(traces) == 2  # picked the new file


# ---------------------------------------------------------------------------
# Candidate skill discovery
# ---------------------------------------------------------------------------


class TestFindCandidateSkills:
    @pytest.mark.asyncio
    async def test_only_auto_dir_skills_are_candidates(self, tmp_path):
        """Skills in the parent dir are NEVER returned."""
        opt = _stub_optimizer(tmp_path)
        # auto/ dir
        opt._skills_auto_dir.mkdir(parents=True, exist_ok=True)
        (opt._skills_auto_dir / "auto-skill.md").write_text("---\nname: auto-skill\n---\n")
        # Sibling root skills/ dir (must NOT be matched)
        root_skills = opt._skills_auto_dir.parent
        (root_skills / "manual-skill.md").write_text("---\nname: manual-skill\n---\n")

        traces = [
            _skill_invocation_trace("auto-skill"),
            _skill_invocation_trace("manual-skill"),
            _skill_invocation_trace("nonexistent"),
        ]
        candidates = await opt._find_candidate_skills(traces)
        assert len(candidates) == 1
        assert candidates[0].name == "auto-skill.md"

    @pytest.mark.asyncio
    async def test_ignores_non_skill_tools(self, tmp_path):
        """Traces where tool_name != 'Skill' are filtered out."""
        opt = _stub_optimizer(tmp_path)
        opt._skills_auto_dir.mkdir(parents=True, exist_ok=True)
        (opt._skills_auto_dir / "x.md").write_text("---\nname: x\n---\n")

        bash_trace = _skill_invocation_trace("x")
        bash_trace["_meta"]["tool_name"] = "bash"

        candidates = await opt._find_candidate_skills([bash_trace])
        assert candidates == []

    @pytest.mark.asyncio
    async def test_handles_malformed_reference(self, tmp_path):
        """Trace user-content without a valid Reference parsed call is skipped."""
        opt = _stub_optimizer(tmp_path)
        opt._skills_auto_dir.mkdir(parents=True, exist_ok=True)
        (opt._skills_auto_dir / "x.md").write_text("---\nname: x\n---\n")

        bad = {
            "messages": [
                {"role": "user", "content": "Call Skill tool. (no parsed reference)"},
                {"role": "assistant", "content": "ok"},
            ],
            "_meta": {"tool_name": "Skill"},
        }
        candidates = await opt._find_candidate_skills([bad])
        assert candidates == []


# ---------------------------------------------------------------------------
# run_optimization_cycle: gating
# ---------------------------------------------------------------------------


class TestCycleGating:
    @pytest.mark.asyncio
    async def test_disabled_returns_immediate_report(self, tmp_path):
        opt = _stub_optimizer(tmp_path, config={"gepa_enabled": False})
        report = await opt.run_optimization_cycle()
        assert isinstance(report, GEPAReport)
        assert report.skills_evaluated == 0
        assert "disabled" in report.notes

    @pytest.mark.asyncio
    async def test_skips_when_below_min_traces(self, tmp_path):
        """Cycle returns empty report if trace count < min."""
        opt = _stub_optimizer(tmp_path, config={"gepa_min_traces_required": 5})
        opt._skills_auto_dir.mkdir(parents=True, exist_ok=True)
        (opt._skills_auto_dir / "x.md").write_text("---\nname: x\n---\n")
        opt._trajectories_dir.mkdir(parents=True, exist_ok=True)
        _make_jsonl(
            opt._trajectories_dir / "golden_traces_1.jsonl",
            [_skill_invocation_trace("x")],  # only 1 trace
        )
        report = await opt.run_optimization_cycle()
        assert report.skills_evaluated == 0
        assert "1 traces" in report.notes

    @pytest.mark.asyncio
    async def test_handles_missing_auto_dir(self, tmp_path):
        """run_optimization_cycle does not crash if auto/ doesn't exist."""
        opt = _stub_optimizer(tmp_path)
        # Trajectories exist but no auto/
        opt._trajectories_dir.mkdir(parents=True, exist_ok=True)
        _make_jsonl(
            opt._trajectories_dir / "golden_traces_1.jsonl",
            [_skill_invocation_trace("x") for _ in range(3)],
        )
        report = await opt.run_optimization_cycle()
        # Auto-dir gets lazy-created but is empty → "auto/ dir empty" note
        assert report.skills_evaluated == 0
        assert "empty" in report.notes
        assert opt._skills_auto_dir.exists()

    @pytest.mark.asyncio
    async def test_no_candidates_returns_note(self, tmp_path):
        """When traces don't match any auto-skill, cycle returns 'no auto-skill matches'."""
        opt = _stub_optimizer(tmp_path)
        opt._skills_auto_dir.mkdir(parents=True, exist_ok=True)
        # Create skill X but traces reference skill Y
        (opt._skills_auto_dir / "x.md").write_text("---\nname: x\n---\nbody")
        opt._trajectories_dir.mkdir(parents=True, exist_ok=True)
        _make_jsonl(
            opt._trajectories_dir / "golden_traces_1.jsonl",
            [_skill_invocation_trace("y")],
        )
        report = await opt.run_optimization_cycle()
        assert report.skills_evaluated == 0
        assert "no auto-skill matches" in report.notes


# ---------------------------------------------------------------------------
# Promotion logic
# ---------------------------------------------------------------------------


class TestPromotion:
    @pytest.mark.asyncio
    async def test_promotes_winner_archives_old(self, tmp_path, monkeypatch):
        """Promotion writes new file with provenance and archives old."""
        opt = _stub_optimizer(tmp_path, config={"gepa_judge_threshold": 0.5})
        opt._skills_auto_dir.mkdir(parents=True, exist_ok=True)
        skill = opt._skills_auto_dir / "x.md"
        original = "---\nname: x\n---\nold body"
        skill.write_text(original)

        opt._trajectories_dir.mkdir(parents=True, exist_ok=True)
        _make_jsonl(
            opt._trajectories_dir / "golden_traces_1.jsonl",
            [_skill_invocation_trace("x") for _ in range(3)],
        )

        async def fake_variants(self, skill_path, traces):
            return [
                "---\nname: x\n---\nVARIANT_A body",
                "---\nname: x\n---\nVARIANT_B body",
            ]

        monkeypatch.setattr(GEPAOptimizer, "_generate_variants", fake_variants)
        # Judge: original=0.4, A=0.6, B=0.9
        opt._judge = _StubJudge(
            scoremap={"old body": 0.4, "VARIANT_A": 0.6, "VARIANT_B": 0.9},
        )

        report = await opt.run_optimization_cycle()
        assert report.skills_evaluated == 1
        assert report.skills_promoted == 1
        # Skill file now contains B
        new_content = skill.read_text()
        assert "VARIANT_B body" in new_content
        # Provenance header
        assert "GEPA-optimized:" in new_content
        # Archive exists with original content
        archives = list((opt._skills_auto_dir / "archive").glob("x_*.md"))
        assert len(archives) == 1
        assert archives[0].read_text() == original

    @pytest.mark.asyncio
    async def test_does_not_promote_below_threshold(self, tmp_path, monkeypatch):
        """A best-variant below the threshold is NOT promoted."""
        opt = _stub_optimizer(tmp_path, config={"gepa_judge_threshold": 0.8})
        opt._skills_auto_dir.mkdir(parents=True, exist_ok=True)
        skill = opt._skills_auto_dir / "x.md"
        original = "---\nname: x\n---\nbody"
        skill.write_text(original)
        opt._trajectories_dir.mkdir(parents=True, exist_ok=True)
        _make_jsonl(
            opt._trajectories_dir / "golden_traces_1.jsonl",
            [_skill_invocation_trace("x") for _ in range(2)],
        )

        async def fake_variants(self, skill_path, traces):
            return ["---\nname: x\n---\nVARIANT body"]

        monkeypatch.setattr(GEPAOptimizer, "_generate_variants", fake_variants)
        # Variant scores higher than current but below 0.8 threshold
        opt._judge = _StubJudge(
            scoremap={"body\n" : 0.4, "VARIANT": 0.7},
        )

        report = await opt.run_optimization_cycle()
        assert report.skills_evaluated == 1
        assert report.skills_promoted == 0
        assert report.skills_unchanged == 1
        # Original unchanged
        assert skill.read_text() == original

    @pytest.mark.asyncio
    async def test_does_not_promote_when_winner_not_better(self, tmp_path, monkeypatch):
        """If best variant scores at or below current, skill is not promoted."""
        opt = _stub_optimizer(tmp_path, config={"gepa_judge_threshold": 0.5})
        opt._skills_auto_dir.mkdir(parents=True, exist_ok=True)
        skill = opt._skills_auto_dir / "x.md"
        skill.write_text("---\nname: x\n---\noriginal high quality")
        opt._trajectories_dir.mkdir(parents=True, exist_ok=True)
        _make_jsonl(
            opt._trajectories_dir / "golden_traces_1.jsonl",
            [_skill_invocation_trace("x") for _ in range(2)],
        )

        async def fake_variants(self, skill_path, traces):
            return ["---\nname: x\n---\nworse"]

        monkeypatch.setattr(GEPAOptimizer, "_generate_variants", fake_variants)
        opt._judge = _StubJudge(
            scoremap={"original high quality": 0.9, "worse": 0.6},
        )

        report = await opt.run_optimization_cycle()
        assert report.skills_promoted == 0
        assert report.skills_unchanged == 1


# ---------------------------------------------------------------------------
# GEPAReport rendering
# ---------------------------------------------------------------------------


class TestReport:
    def test_empty_report_summary(self):
        report = GEPAReport(timestamp=time.time())
        text = report.to_telegram_summary()
        assert "no candidate skills" in text
        assert text  # non-empty

    def test_empty_report_with_notes(self):
        report = GEPAReport(timestamp=time.time(), notes="rate limited")
        text = report.to_telegram_summary()
        assert "rate limited" in text

    def test_promoted_report_summary(self):
        report = GEPAReport(
            timestamp=time.time(),
            skills_evaluated=2,
            skills_promoted=1,
            skills_unchanged=1,
            duration_seconds=12.5,
            promotions=[
                {"skill": "deploy-app", "old_score": 0.5, "new_score": 0.85},
            ],
        )
        text = report.to_telegram_summary()
        assert "12s" in text or "12." in text
        assert "Evaluated: 2" in text
        assert "deploy-app" in text
        assert "0.85" in text


# ---------------------------------------------------------------------------
# from_config gating
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_returns_none_when_disabled(self, tmp_path):
        import yaml
        cfg = tmp_path / "prometheus.yaml"
        cfg.write_text(yaml.safe_dump({"learning": {"gepa_enabled": False}}))
        result = GEPAOptimizer.from_config(provider=None, config_path=str(cfg))
        assert result is None

    def test_builds_when_enabled(self, tmp_path):
        import yaml
        cfg = tmp_path / "prometheus.yaml"
        cfg.write_text(yaml.safe_dump({
            "learning": {
                "gepa_enabled": True,
                "gepa_judge_threshold": 0.6,
                "gepa_max_skills_per_cycle": 5,
            },
            "evals": {"judge_base_url": "http://stub:8080"},
        }))
        result = GEPAOptimizer.from_config(provider=None, config_path=str(cfg))
        assert result is not None
        assert result._enabled is True
        assert result._max_skills == 5
        assert result._threshold == 0.6
        assert result._judge_base_url == "http://stub:8080"
