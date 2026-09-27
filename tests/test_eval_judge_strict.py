"""A judge reply that holds no verdict must not become a score (WP-X.22).

THE DEFECT
----------
``PrometheusJudge`` turned replies that carry no verdict into scores:

* ``_fallback_parse`` took the FIRST number anywhere in the text. A G-Eval
  reply that opens "1. The agent failed the first criterion…" and never
  writes its SCORE line scored 1.0: a reply saying the criteria failed,
  recorded as a pass.
* A JSON reply with no ``"score"`` key scored 0.0, so a judge failure read as
  a model failure. An empty reply did the same.
* An out-of-range number was clamped: a 7 became 1.0.
* The G-Eval fallbacks ("rating", "score is", a bare decimal on the last line)
  picked up numbers the judge was quoting, not grading with.
* The SCORE: regex took the FIRST score line, not the final one.

The ladder refused all of this in ``strict_judge_score``; the nightly evals
did not.

THE RULE
--------
A verdict counts only when the judge gave a finite score in [0, 1] in the
form it was asked for: a JSON ``"score"`` key from ``evaluate()``, the final
explicit ``SCORE:`` line from ``evaluate_geval()``. Anything else is an
UNPARSEABLE verdict with no score, never 0.0 and never a stray number, and
the evals count it as "judge unavailable": excluded from pass rates, never
scored as pass or fail.
"""

from __future__ import annotations

import asyncio
import math
import random
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from prometheus.evals.golden_dataset import GoldenTask
from prometheus.evals.judge import JudgeVerdict, PrometheusJudge
from prometheus.evals.metrics import NoHallucinationMetric, TaskCompletionMetric
from prometheus.evals.runner import EvalRunner, _SimpleTestCase

JUDGE_METRICS = ["No Hallucination", "Task Completion"]

# The reported symptom: the criteria are written out, the SCORE line never is.
CRITERIA_FAILED = (
    "1. The agent failed the first criterion: it never ran the command.\n"
    "2. The output is not grounded in any tool result.\n"
    "3. It claims an action it did not take."
)


def _judge_replying(raw: str) -> PrometheusJudge:
    """A real judge whose endpoint answers ``raw``: only the transport is faked."""
    judge = PrometheusJudge(base_url="http://judge.invalid", model="pinned-judge")

    async def _reply(*_args: object, **_kwargs: object) -> str:
        return raw

    judge._call_llm = _reply  # type: ignore[method-assign]
    return judge


def _evaluate(raw: str) -> JudgeVerdict:
    return asyncio.run(_judge_replying(raw).evaluate(
        task_input="List the files in /tmp.",
        agent_output="The files are a.txt and b.txt.",
        expected_behavior="Runs ls and reports the files.",
    ))


def _evaluate_geval(raw: str) -> JudgeVerdict:
    return asyncio.run(_judge_replying(raw).evaluate_geval(
        criteria=["The agent ran the command.", "The output is grounded."],
        context="Task: list the files in /tmp.",
    ))


def _assert_no_verdict(v: JudgeVerdict, raw: str) -> None:
    assert v.score is None, f"a reply with no verdict scored {v.score!r}: {raw!r}"
    assert v.status == "unparseable", raw
    assert v.raw_response == raw, "the reply the judge gave must be kept"


# ── 1. the first number anywhere ────────────────────────────────────────────

class TestFirstNumberAnywhere:

    def test_a_geval_reply_saying_the_criteria_failed_is_not_a_pass(self):
        """THE reported symptom: its list numbering "1." scored 1.0."""
        _assert_no_verdict(_evaluate_geval(CRITERIA_FAILED), CRITERIA_FAILED)

    def test_a_json_judge_answering_in_prose_is_not_scored_by_its_numbering(self):
        _assert_no_verdict(_evaluate(CRITERIA_FAILED), CRITERIA_FAILED)

    @pytest.mark.parametrize("raw", [
        "I rate this 0.6 out of 1.0",
        "The agent listed 3 of the 4 files.",
        # evaluate() asked for JSON; a SCORE line is the G-Eval form, and its
        # stray "1." is what used to be read
        "1. The agent ran ls.\nSCORE: 0.85",
    ])
    def test_a_number_in_prose_is_not_a_json_verdict(self, raw):
        _assert_no_verdict(_evaluate(raw), raw)


# ── 2. no "score" key, or no reply at all ───────────────────────────────────

class TestNoScoreIsNotZero:

    @pytest.mark.parametrize("raw", [
        '{"reasoning": "the agent never ran the command"}',
        '{"verdict": 0.9, "reasoning": "x"}',
        '{"score": null, "reasoning": "x"}',
        '{"score": "0.9", "reasoning": "x"}',
        '{"score": true, "reasoning": "x"}',
        '{"score": [0.9], "reasoning": "x"}',
    ])
    def test_a_json_reply_without_a_numeric_score_has_no_score(self, raw):
        _assert_no_verdict(_evaluate(raw), raw)

    @pytest.mark.parametrize("raw", ["", "   \n"])
    def test_an_empty_reply_is_not_a_zero(self, raw):
        _assert_no_verdict(_evaluate(raw), raw)
        _assert_no_verdict(_evaluate_geval(raw), raw)


# ── 3. out of range is not clamped ──────────────────────────────────────────

class TestOutOfRangeIsNotClamped:

    @pytest.mark.parametrize("raw", [
        '{"score": 7, "reasoning": "x"}',
        '{"score": 1.5, "reasoning": "x"}',
        '{"score": -0.5, "reasoning": "x"}',
        '{"score": 85, "reasoning": "on a 0-100 scale"}',
        '{"score": NaN, "reasoning": "x"}',
        '{"score": Infinity, "reasoning": "x"}',
        '{"score": -Infinity, "reasoning": "x"}',
    ])
    def test_a_json_score_outside_zero_to_one(self, raw):
        _assert_no_verdict(_evaluate(raw), raw)

    @pytest.mark.parametrize("raw", [
        "1. Done.\nSCORE: 7",
        "1. Done.\nSCORE: 85",
        "1. Done.\nSCORE: 8/10",
    ])
    def test_a_geval_score_outside_zero_to_one(self, raw):
        _assert_no_verdict(_evaluate_geval(raw), raw)


# ── 4. the G-Eval fallbacks ─────────────────────────────────────────────────

class TestGevalFallbacksAreNotVerdicts:

    @pytest.mark.parametrize("raw", [
        # "rating": the judge quoting the task, not grading it
        "1. The agent quoted the product rating: 4.5 stars, as the page shows.\n"
        "2. It never finished the comparison.",
        # "score is"
        "1. The agent says the test score is 0.9, but no tool produced it.\n"
        "2. The task was not completed.",
        # a bare decimal on the last line
        "1. The agent did not complete the task.\n"
        "2. It reported version 0.9 of a package that does not exist.",
        # close to a verdict, but not the SCORE: line the judge was told to write
        "1. Good work.\n2. Mostly done.\nFinal score: 0.8",
    ])
    def test_a_quoted_number_is_not_the_verdict(self, raw):
        _assert_no_verdict(_evaluate_geval(raw), raw)

    def test_evaluate_geval_reads_only_its_score_line(self):
        """G-Eval asks for a SCORE line; a JSON object is not that form."""
        raw = '{"score": 0.9, "reasoning": "fine"}'
        _assert_no_verdict(_evaluate_geval(raw), raw)


# ── 5. the final SCORE line, not the first ──────────────────────────────────

class TestTheFinalScoreLine:

    def test_the_final_score_line_is_the_verdict(self):
        raw = ("SCORE: 0.9\n"
               "1. Checking the tool output, the command never ran.\n"
               "2. The listing was invented.\n"
               "SCORE: 0.2")
        v = _evaluate_geval(raw)
        assert v.score == 0.2, f"read {v.score!r}; the judge's final line says 0.2"
        assert v.status == "parsed"

    def test_a_malformed_final_line_is_not_rescued_by_an_earlier_one(self):
        raw = "SCORE: 0.9\n1. On reflection the task was not done.\nSCORE: N/A"
        _assert_no_verdict(_evaluate_geval(raw), raw)


# ── a real verdict keeps its score ──────────────────────────────────────────

class TestARealVerdictStillCounts:
    """Strictness must not turn real verdicts into "unavailable"."""

    @pytest.mark.parametrize("raw, score", [
        ('{"score": 0.85, "reasoning": "done"}', 0.85),
        ('```json\n{"score": 0.75, "reasoning": "mostly"}\n```', 0.75),
        ('Here is my evaluation:\n{"score": 0.9, "reasoning": "great"}', 0.9),
        ('{"score": 1, "reasoning": "every criterion met"}', 1.0),
        # a real zero is a real FAIL, not a judge failure
        ('{"score": 0, "reasoning": "nothing was done"}', 0.0),
    ])
    def test_a_json_verdict(self, raw, score):
        v = _evaluate(raw)
        assert v.score == score

    @pytest.mark.parametrize("raw, score", [
        ("1. The agent ran ls.\n2. The listing matches.\nSCORE: 0.85", 0.85),
        ("Good work.\nscore: 0.7", 0.7),
        ("1. Ran it.\n**SCORE:** 0.85", 0.85),
        ("1. Nothing was done.\nSCORE: 0", 0.0),
        ("1. Ran it.\nSCORE: 0.6\n(Judged against the reference answer.)", 0.6),
    ])
    def test_a_geval_verdict(self, raw, score):
        v = _evaluate_geval(raw)
        assert v.score == score

    def test_the_geval_reasoning_is_the_text_before_the_score(self):
        v = _evaluate_geval("1. The agent ran ls correctly.\nSCORE: 0.85")
        assert "ran ls correctly" in v.reasoning

    def test_every_verdict_says_whether_it_was_parsed(self):
        assert _evaluate('{"score": 0.85, "reasoning": "done"}').status == "parsed"
        assert _evaluate_geval("1. Ran it.\nSCORE: 0.85").status == "parsed"
        assert _evaluate('{"reasoning": "x"}').status == "unparseable"
        assert _evaluate_geval(CRITERIA_FAILED).status == "unparseable"


# ── the metrics and the runner: judge unavailable, not pass or fail ─────────

def _case() -> _SimpleTestCase:
    return _SimpleTestCase(
        input="List the files in /tmp.", actual_output="a.txt b.txt",
        expected_output="Runs ls and reports the files.",
        additional_metadata={"tool_trace": []},
    )


class TestTheMetricsDoNotScoreANonVerdict:

    @pytest.mark.parametrize("metric_cls", [TaskCompletionMetric, NoHallucinationMetric])
    def test_an_unparseable_verdict_is_neither_pass_nor_fail(self, metric_cls):
        metric = metric_cls(judge=_judge_replying('{"reasoning": "no score"}'))
        got = asyncio.run(metric.a_measure(_case()))
        assert got is None and metric.score is None, f"scored {metric.score!r}"
        assert metric.success is None


TASK = GoldenTask(
    id="ls-tmp", name="List /tmp", tier=1, input="List the files in /tmp.",
    expected_behavior="Runs ls and reports the files.", expected_tools=["bash"],
)


def _runner(raw: str) -> EvalRunner:
    """The real runner, metrics, judge parse and classifier; the agent is faked."""
    loop = SimpleNamespace(
        run_async=AsyncMock(return_value=SimpleNamespace(text="a.txt b.txt", turns=2)),
        _tool_trace=[{"tool_name": "bash", "result": "a.txt b.txt", "is_error": False}],
    )
    return EvalRunner(agent_loop=loop, judge=_judge_replying(raw),  # type: ignore[arg-type]
                      system_prompt="You are helpful.")


def _run(raw: str):
    return asyncio.run(_runner(raw).run_task(TASK))


class TestTheRunner:

    @pytest.mark.parametrize("raw", [
        CRITERIA_FAILED, '{"reasoning": "no score"}', '{"score": 7}', "",
    ])
    def test_an_unparseable_verdict_is_judge_unavailable(self, raw):
        r = _run(raw)
        judged = {m.metric_name: m.score for m in r.metrics}
        assert not set(JUDGE_METRICS) & set(judged), (
            f"a reply with no verdict was scored: {judged}")
        assert sorted(r.unavailable_metrics) == JUDGE_METRICS
        assert sorted(r.unparseable_metrics) == JUDGE_METRICS
        assert (r.failure_source, r.failure_category) == (
            "harness", "harness:judge_unavailable")

    def test_a_good_verdict_still_passes(self):
        r = _run('{"score": 0.9, "reasoning": "done"}')
        assert (r.failure_source, r.unavailable_metrics) == ("pass", [])

    def test_a_low_verdict_is_still_a_model_failure(self):
        r = _run('{"score": 0.1, "reasoning": "not done"}')
        assert r.failure_source == "model"
        assert r.unavailable_metrics == []


class TestTheReport:

    def _results(self):
        return [_run('{"score": 0.9, "reasoning": "done"}'),
                _run('{"reasoning": "no score"}')]

    def test_the_summary_counts_them_and_leaves_them_out_of_pass_rates(self):
        summary = _runner("")._compute_summary(self._results())
        # only the parsed verdict is averaged
        assert summary["metric_averages"]["Task Completion"] == 0.9
        assert summary["judge_unavailable"] == {
            "tasks": 1, "metrics": 2, "unparseable": 2}
        # one task could be scored, and it passed
        assert (summary["scored_tasks"], summary["pass_rate"]) == (1, 1.0)

    def test_the_printed_report_shows_them(self, tmp_path, capsys):
        _runner("").print_summary(self._results(), output_dir=tmp_path)
        out = capsys.readouterr().out
        assert "2 unparseable" in out, out
        assert "1/1 scored" in out, out


# ── the ladder: one parser, identical results ───────────────────────────────

def _ladder_strict_score_at_5fb0000(raw: str) -> float | None:
    """``strict_judge_score`` as it was at 5fb0000, verbatim: the oracle for
    "the ladder reads every reply exactly as it did"."""
    import json as _json
    import math

    candidates = [raw, re.sub(r"```(?:json)?\s*\n?", "", raw).strip()]
    if "{" in raw and "}" in raw:
        candidates.append(raw[raw.index("{"): raw.rindex("}") + 1])
    for text in candidates:
        try:
            obj = _json.loads(text)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        s = obj.get("score")
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            return None
        s = float(s)
        return s if math.isfinite(s) and 0.0 <= s <= 1.0 else None
    return None


LADDER_CORPUS = [
    '{"score": 0.5}', '{"score":0.9,"reasoning":"good"}', '{"score": 1}',
    '{"score": 0}', '{"score": -0.0}', '{"score": 1e-5}', '{"score": 1e999}',
    '```json\n{"score": 0.75}\n```', '```\n{"score": 0.6}\n```',
    'Here: {"score": 0.9} ok', '[{"score": 0.9}]', '{"a": {"score": 0.9}}',
    '{"reasoning": "fine"}', '{"score": 7}', '{"score": true}', '{"score": "0.9"}',
    '{"score": NaN}', '{"score": Infinity}', '{"score": null}', '{"score": [1]}',
    '{"score": 0.2} {"score": 0.9}', '} {', '{', '}', '0.8', '"0.8"', '[]', '',
    '   ', "1 lol", "SCORE: 0.9", "1. Ran it.\nSCORE: 0.85", CRITERIA_FAILED,
    '```json\n{"score": 0.4}', '{"score": 0.3}\n```', '\n{"score": 0.5}\n',
]


def _fuzz_corpus(n: int = 3000) -> list[str]:
    parts = ['{', '}', '[', ']', '"score"', '"reasoning"', ':', ',', ' ', '\n',
             '0.5', '7', '1', '0', '-1', 'NaN', 'Infinity', 'true', 'null', '"x"',
             '```json\n', '```', 'SCORE: 0.9', 'text', '1e3', '.5']
    rng = random.Random(22)
    return ["".join(rng.choice(parts) for _ in range(rng.randint(1, 12)))
            for _ in range(n)]


class TestTheLadder:

    @pytest.mark.parametrize("raw", LADDER_CORPUS)
    def test_the_ladder_reads_every_reply_as_it_did(self, raw):
        from prometheus.gym.ladder.verdict import strict_judge_score

        for text in (raw, raw.strip()):
            assert strict_judge_score(text) == _ladder_strict_score_at_5fb0000(text), text

    def test_the_ladder_reads_generated_replies_as_it_did(self):
        from prometheus.gym.ladder.verdict import strict_judge_score

        diffs = [raw for raw in _fuzz_corpus()
                 if strict_judge_score(raw) != _ladder_strict_score_at_5fb0000(raw)]
        assert diffs == []

    def test_the_ladder_uses_the_shared_parser(self):
        from prometheus.evals.judge import parse_judge_reply
        from prometheus.gym.ladder.verdict import strict_judge_score

        for raw in LADDER_CORPUS + _fuzz_corpus(500):
            assert strict_judge_score(raw) == parse_judge_reply(raw).score, raw


# ── the verdict cannot say two things at once ───────────────────────────────

class TestTheVerdictIsConsistent:

    def test_existing_constructors_still_work(self):
        """Callers that build a verdict with a score keep working unchanged."""
        v = JudgeVerdict(score=0.9, reasoning="good", raw_response="")
        assert (v.score, v.status) == (0.9, "parsed")

    @pytest.mark.parametrize("kwargs", [
        dict(score=None),                                   # parsed, but no score
        dict(score=1.5),                                    # parsed, out of range
        dict(score=math.nan),
        dict(score=0.5, status="unparseable"),              # unparseable, with a score
        dict(score=None, status="maybe"),
    ])
    def test_a_contradictory_verdict_cannot_be_built(self, kwargs):
        with pytest.raises(ValueError):
            JudgeVerdict(reasoning="", raw_response="", **kwargs)
