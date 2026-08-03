"""Every score records who graded it.

Until 2026-08-02 two judges ran concurrently: ``run_nightly_evals.py`` pinned
its model, while the in-daemon GEPA optimizer passed ``model=None`` and graded
with whatever the judge endpoint had loaded. **The records were
indistinguishable** — a GEPA promotion score and a nightly score were both bare
numbers — so cross-comparison was unsound and there was no way to tell after
the fact which judge produced which score.

PR #136 pinned the model. This closes the other half: the score now carries its
judge. Two properties matter and both are asserted below:

* ``pinned`` is NOT inferable from ``model``. An auto-detected judge that
  resolves to ``qwen2.5:7b-instruct`` records the same model name as one pinned
  to it — only the pinned run is reproducible.
* A record with **no** ``judge`` key means UNKNOWN, permanently. The 230
  pre-existing result files are not backfilled: manufacturing provenance is the
  exact failure this change prevents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from prometheus.evals.judge import PrometheusJudge


# ---------------------------------------------------------------------------
# provenance() itself
# ---------------------------------------------------------------------------


def test_pinned_judge_reports_pinned():
    j = PrometheusJudge(base_url="http://judge:11434/", model="qwen2.5:7b-instruct")
    prov = j.provenance()
    assert prov["pinned"] is True
    assert prov["model"] == "qwen2.5:7b-instruct"
    assert prov["base_url"] == "http://judge:11434"  # trailing slash stripped


def test_unpinned_judge_reports_not_pinned():
    j = PrometheusJudge(base_url="http://judge:11434")
    prov = j.provenance()
    assert prov["pinned"] is False, (
        "an auto-detecting judge must not claim to be pinned"
    )
    assert prov["model"] is None, (
        "nothing has been graded yet, so there is no judge identity to record — "
        "None is the honest answer, not a gap"
    )


def test_pinned_is_not_inferable_from_model_alone():
    """The whole point: same model name, different reproducibility."""
    pinned = PrometheusJudge(base_url="http://j", model="qwen2.5:7b-instruct")

    auto = PrometheusJudge(base_url="http://j")
    auto._resolved_model = "qwen2.5:7b-instruct"  # as if detection resolved here

    assert pinned.provenance()["model"] == auto.provenance()["model"], (
        "precondition: the two must be indistinguishable by model name"
    )
    assert pinned.provenance()["pinned"] != auto.provenance()["pinned"], (
        "pinned must distinguish them — without it the records are the same, "
        "which is the defect this change exists to close"
    )


@pytest.mark.asyncio
async def test_detected_model_is_reported_after_detection(monkeypatch):
    """provenance() names the judge that GRADED, not the one requested."""
    j = PrometheusJudge(base_url="http://judge:11434")
    assert j.provenance()["model"] is None

    j._resolved_model = "some-ambient-model"
    prov = j.provenance()
    assert prov["model"] == "some-ambient-model"
    assert prov["pinned"] is False, (
        "resolving a model must not retroactively make the run look pinned"
    )


# ---------------------------------------------------------------------------
# The two writers
# ---------------------------------------------------------------------------


@dataclass
class _FakeResult:
    task_id: str = "t1"
    passed: bool = True
    score: float = 1.0


def test_eval_results_file_carries_judge_provenance(tmp_path):
    """The saved JSON must name its judge."""
    from prometheus.evals.runner import EvalRunner

    runner = object.__new__(EvalRunner)
    runner._judge = PrometheusJudge(base_url="http://j:1", model="pinned-judge")
    runner._config = {"results_dir": str(tmp_path)}
    runner._compute_summary = lambda results: {"n": len(results)}  # type: ignore[method-assign]

    path = EvalRunner.save_results(runner, [_FakeResult()], output_dir=tmp_path)
    data = json.loads(path.read_text())

    assert "judge" in data, (
        "the results file does not record who graded — scores in it cannot be "
        "compared with scores from another run"
    )
    assert data["judge"]["model"] == "pinned-judge"
    assert data["judge"]["pinned"] is True
    assert data["judge"]["base_url"] == "http://j:1"


def test_gepa_provenance_reads_through_the_scoring_accessor():
    """GEPA must report the judge that actually scored, not a lookalike.

    Regression guard: the first version of this referenced a bare ``judge``
    name that is not in scope inside ``_optimize_one`` — it imported fine and
    would have raised NameError only on a real promotion.
    """
    import inspect

    from prometheus.learning.gepa import GEPAOptimizer

    src = inspect.getsource(GEPAOptimizer._optimize_one)
    assert "self._judge_provenance()" in src, (
        "the promotion record does not attach judge provenance"
    )

    opt = object.__new__(GEPAOptimizer)
    opt._judge = PrometheusJudge(base_url="http://j:2", model="gepa-judge")
    prov = GEPAOptimizer._judge_provenance(opt)
    assert prov is not None and prov["model"] == "gepa-judge"
    assert prov["pinned"] is True


def test_gepa_provenance_is_none_when_no_judge_exists():
    """No judge is a legitimate state — it must not raise."""
    from prometheus.learning.gepa import GEPAOptimizer

    opt = object.__new__(GEPAOptimizer)
    opt._judge = None
    opt._judge_base_url = None
    assert GEPAOptimizer._judge_provenance(opt) is None


# ---------------------------------------------------------------------------
# No backfill
# ---------------------------------------------------------------------------


def test_a_record_without_a_judge_key_means_unknown():
    """Old records are not migrated. Absent must read as unknown, never guessed."""
    legacy = {"timestamp": "2026-04-06T03:04:06", "task_count": 3, "results": []}
    assert legacy.get("judge") is None, (
        "a pre-2026-08-02 record has no knowable judge; any code reading these "
        "must treat a missing key as UNKNOWN and must not infer or backfill one"
    )
