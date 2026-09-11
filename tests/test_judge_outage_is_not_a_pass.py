"""A judge that cannot answer must not produce a PASS.

THE DEFECT
----------
The judge is health-checked once at start-up. If it errors afterwards, each of
the three metrics raises, and `_evaluate_metrics` caught the exception, logged
a warning, and simply did not append the score. `metric_scores` therefore came
back EMPTY — and `classify_failure` read:

    all_passed = all(s >= 0.5 for s in metric_scores.values()) if metric_scores else True

An empty map meant "everything passed". Measured before the fix:

    judge DOWN (all metrics raised)
      -> source=pass  category=none  detail='All metrics passed'

Every task in the suite recorded PASS, with only a log warning to show for it,
and `run_nightly_evals.py` exited 0 because it keys the exit code on `r.error`
— which is not set, since the task itself ran fine. It was the SCORING that did
not happen.

THE DISTINCTION THE FIX ADDS
-----------------------------
"Scored zero" and "could not be scored" are different facts. The first says the
model did badly; the second says nothing about the model at all. A metric whose
judge call raises is now NAMED in `unavailable_metrics` rather than silently
dropped, and a named unavailable metric is a HARNESS failure.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.evals.classifier import (  # noqa: E402
    FailureCategory,
    FailureSource,
    classify_failure,
)

TRACE = [{"name": "bash", "result": "ok"}]
ALL_THREE = ["Task Completion", "Tool Usage", "No Hallucination"]


def _classify(**over):
    kwargs = dict(
        task_id="t1",
        expected_tools=["bash"],
        tool_trace=TRACE,
        agent_output="an answer",
        error=None,
        metric_scores={},
    )
    kwargs.update(over)
    return classify_failure(**kwargs)


# ── the classifier ──────────────────────────────────────────────────────────

def test_an_unavailable_judge_is_not_a_pass():
    """THE defect, stated directly."""
    result = _classify(unavailable_metrics=ALL_THREE)

    assert result.source is not FailureSource.PASS, (
        f"a run in which no metric could be evaluated was classified "
        f"{result.source.value!r} with detail {result.detail!r}"
    )
    assert result.source is FailureSource.HARNESS
    assert result.category is FailureCategory.JUDGE_UNAVAILABLE


def test_the_detail_names_which_metrics_could_not_run():
    """'Something failed' is not actionable; say which."""
    result = _classify(unavailable_metrics=["Task Completion"])
    assert "Task Completion" in result.detail
    assert "not a pass" in result.detail


def test_one_unavailable_metric_is_enough():
    """A partial outage is still an unscored task.

    Two metrics passing does not make up for a third that never ran — the
    unscored one might have been the one that failed.
    """
    result = _classify(
        metric_scores={"Tool Usage": 1.0, "No Hallucination": 1.0},
        unavailable_metrics=["Task Completion"],
    )
    assert result.source is FailureSource.HARNESS
    assert result.category is FailureCategory.JUDGE_UNAVAILABLE


def test_an_empty_metric_map_with_no_explanation_is_unclear_not_pass():
    """The vacuous `else True`, removed.

    Nobody said a metric was unavailable and nobody produced a score. That is
    unknown, and unknown is not success.
    """
    result = _classify(metric_scores={})
    assert result.source is not FailureSource.PASS, result.detail
    assert result.source is FailureSource.UNCLEAR


def test_a_genuinely_good_task_still_passes():
    """The fix must not make everything a failure.

    Without this, `return HARNESS` unconditionally would satisfy every test
    above and make the whole eval suite useless in the other direction.
    """
    result = _classify(
        metric_scores={"Task Completion": 0.9, "Tool Usage": 1.0,
                       "No Hallucination": 1.0},
        unavailable_metrics=[],
    )
    assert result.source is FailureSource.PASS
    assert result.category is FailureCategory.NONE


def test_a_genuinely_bad_task_is_still_a_model_failure():
    """An available judge scoring low must stay a MODEL failure.

    "Scored zero" and "could not be scored" must not collapse into each other
    in either direction.
    """
    result = _classify(
        metric_scores={"Task Completion": 0.1, "Tool Usage": 1.0,
                       "No Hallucination": 1.0},
        unavailable_metrics=[],
    )
    assert result.source is FailureSource.MODEL


# ── the runner, through the real _evaluate_metrics ──────────────────────────

class _DeadJudge:
    """A judge that was healthy at start-up and is not any more."""

    def provenance(self):
        return {"model": "dead"}

    async def a_measure(self, *a, **k):
        raise ConnectionError("judge unreachable")


def test_the_runner_names_the_metrics_it_could_not_evaluate(monkeypatch):
    """Driven through the real `_evaluate_metrics`, not a re-implementation."""
    from prometheus.evals.runner import EvalRunner

    runner = EvalRunner.__new__(EvalRunner)
    runner._judge = _DeadJudge()

    class _Boom:
        def __init__(self, *a, **k):
            pass

        async def a_measure(self, *a, **k):
            raise ConnectionError("judge unreachable")

        def is_successful(self):
            return False

        reason = ""

    import prometheus.evals.metrics as metrics_mod

    monkeypatch.setattr(metrics_mod, "ToolUsageMetric", _Boom, raising=False)
    monkeypatch.setattr(metrics_mod, "TaskCompletionMetric", _Boom, raising=False)
    monkeypatch.setattr(metrics_mod, "NoHallucinationMetric", _Boom, raising=False)

    from prometheus.evals.golden_dataset import GoldenTask

    task = GoldenTask(
        id="t1", name="n", tier=1, input="p",
        expected_behavior="does the thing", expected_tools=["bash"],
    )

    scores, unavailable = asyncio.run(
        runner._evaluate_metrics(task, "output", TRACE)
    )

    assert scores == [], "a dead judge produced scores"
    assert sorted(unavailable) == sorted(ALL_THREE), (
        f"the runner did not name the metrics it could not evaluate: "
        f"{unavailable}"
    )
