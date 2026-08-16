"""The divergence aggregation, driven through the REAL detector.

Both live observations of this detector were wrong, in opposite directions,
and both were the same arithmetic: an unweighted mean over a variable-length
list of heterogeneous signals.

  * a compliant deploy-proof turn scored 0.96 and logged "diverged", because
    a checkpoint had just emptied the tool window and the mean collapsed to
    the single lexical term;
  * an agent visibly stuck scored ~0.23 and read "on_track", because a 0.5
    repetition penalty averaged among healthy terms cannot reach a 0.7
    threshold from inside a mean.

So the fix is not a threshold and not a new detector. It is three properties:
repetition FLOORS the score, a lone term may never be the verdict, and the
sub-scores are returned so the next failure is diagnosable rather than
re-derived.

These drive `record_tool_call` + `evaluate` on a real `DivergenceDetector`,
the way the agent loop does — not `_components` directly.
"""

from __future__ import annotations

import pytest

from prometheus.coordinator.divergence import (
    BEHAVIOURAL_TERMS, REPETITION_FLOOR, CheckpointStore, DivergenceDetector)
from tests.fixtures.divergence_traces import (
    FL4_GOAL, FL4_MESSAGES, FL4_RECORDED_SCORE, FL4_TOOL_CALLS,
    SYMBIOTE_GOAL, SYMBIOTE_MESSAGES, SYMBIOTE_TOOL_CALLS)


def _detector(tmp_path, **cfg):
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = CheckpointStore(db_path=tmp_path / "lcm.db")
    return DivergenceDetector(
        {"divergence": {"enabled": True, "threshold": 0.7,
                        "checkpoint_interval": 5, **cfg}},
        checkpoint_store=store,
    )


def _drive(detector, task_id, goal, calls):
    """Feed a trace the way the agent loop does."""
    detector.start_task(task_id, goal)
    for c in calls:
        detector.record_tool_call(
            c["tool"], c["args"], c["result"], c["success"], task_id=task_id)
    return detector


# ---------------------------------------------------------------------------
# FL4 — the recorded false positive. THIS IS THE OUTCOME CHECK.
# ---------------------------------------------------------------------------

def test_fl4_recorded_trace_scores_below_threshold(tmp_path):
    """The recorded 0.96 false positive must no longer read as diverged.

    Provenance: the brain vault, wiki/log.md:217. Six sequential `echo`
    calls on a turn that did exactly what it was asked.
    """
    d = _drive(_detector(tmp_path), "fl4", FL4_GOAL, FL4_TOOL_CALLS)
    r = d.evaluate(FL4_MESSAGES, [], task_id="fl4")

    assert not r.diverged, (
        f"the recorded compliant turn still reads as diverged: "
        f"score={r.score:.2f} components={r.components}"
    )
    assert r.score < 0.7, f"score {r.score:.2f} is still at/above threshold"
    assert r.score < FL4_RECORDED_SCORE, (
        f"score {r.score:.2f} did not improve on the recorded {FL4_RECORDED_SCORE}"
    )
    assert "repetition" not in r.components, (
        "six productive bash calls tripped the repetition floor. Same-tool "
        "repetition alone discriminates NOTHING — a run of bash is the most "
        "ordinary shape in this system, and flooring it made this trace and "
        "the stuck-agent trace score identically (0.750, both diverged)."
    )


def test_fl4_at_a_checkpoint_boundary_still_sees_its_tool_calls(tmp_path):
    """The mechanism, pinned: a checkpoint must not blind the scorer.

    `_create_checkpoint` clears the checkpoint PAYLOAD; the scoring window is
    a separate deque that survives it. Before this, a checkpoint immediately
    before `evaluate` left goal_alignment as the only term — which is exactly
    how a compliant turn reached 0.96.
    """
    d = _drive(_detector(tmp_path), "fl4", FL4_GOAL, FL4_TOOL_CALLS)
    d.maybe_checkpoint(FL4_MESSAGES, task_id="fl4")      # empties the payload
    r = d.evaluate(FL4_MESSAGES, [], task_id="fl4")

    assert len(r.components) > 1, (
        f"only {list(r.components)} survived the checkpoint — the scorer is "
        f"blind on checkpoint-boundary steps again"
    )
    assert "tool_failure_rate" in r.components, (
        "the tool window was emptied by checkpointing; that is the defect"
    )


def test_a_lone_lexical_term_can_never_declare_divergence(tmp_path):
    """Goal alignment may contribute to a verdict; it may not constitute one.

    Driven with NO tool calls at all, which is the state a checkpoint used to
    manufacture. The score may legitimately be high — the text really is
    dissimilar — but `diverged` must be False, and the reason must say why so
    a reader is not left with an unexplained high number.
    """
    d = _detector(tmp_path)
    d.start_task("lonely", FL4_GOAL)
    r = d.evaluate([{"role": "assistant", "content": "24f862d 200 idle"}], [],
                   task_id="lonely")

    assert set(r.components) == {"goal_alignment"}, r.components
    assert not r.diverged, (
        f"a single lexical term declared divergence: {r.score:.2f} {r.components}"
    )
    if r.score >= 0.7:
        assert "no_behavioural_signal" in r.reason, r.reason


# ---------------------------------------------------------------------------
# SYMBIOTE — ⚠ SYNTHETIC trace. See tests/fixtures/divergence_traces.py.
# ---------------------------------------------------------------------------

def test_repetition_floors_the_score_rather_than_being_averaged_away(tmp_path):
    """⚠ Asserted against a SYNTHETIC trace — no recording of this exists.

    Seven greps with different arguments, an agent plainly stuck. As a
    summand, the repetition penalty averaged among healthy terms to ~0.23 and
    read "on_track"; as a floor it survives.

    Note what this also settles: HASH-EXACT repeat detection would see seven
    distinct calls and fire on none of them. It is strictly narrower than the
    same-tool check already present, and would go silent on the one shape that
    motivated this round.
    """
    d = _drive(_detector(tmp_path), "sym", SYMBIOTE_GOAL, SYMBIOTE_TOOL_CALLS)
    r = d.evaluate(SYMBIOTE_MESSAGES, [], task_id="sym")

    assert r.components.get("repetition") == REPETITION_FLOOR, (
        f"repetition did not fire on seven UNPRODUCTIVE same-tool calls: "
        f"{r.components}")
    assert r.score >= REPETITION_FLOOR, (
        f"the repetition signal was averaged away: score={r.score:.2f} "
        f"components={r.components} — a floor is the whole point"
    )
    mean = sum(r.components.values()) / len(r.components)
    assert r.score > mean, (
        f"score {r.score:.2f} equals the plain mean {mean:.2f}; the floor is "
        f"not being applied"
    )


# ---------------------------------------------------------------------------
# Properties of the aggregation itself.
# ---------------------------------------------------------------------------

def test_components_are_returned_so_a_verdict_is_diagnosable(tmp_path):
    """The first false positive took weeks to explain because only the total
    was recorded. Every verdict now carries its inputs."""
    d = _drive(_detector(tmp_path), "t", FL4_GOAL, FL4_TOOL_CALLS)
    r = d.evaluate(FL4_MESSAGES, [], task_id="t")
    assert r.components and all(isinstance(v, float) for v in r.components.values())
    for name in r.components:
        assert f"{name}=" in r.reason, (
            f"{name} is not named in the reason string: {r.reason}")


def test_divergence_requires_a_behavioural_term_to_agree(tmp_path):
    """The rule, stated directly against the classifier."""
    hi = {"goal_alignment": 0.99}
    assert not DivergenceDetector._is_diverged(hi, 0.99, 0.7)
    for term in BEHAVIOURAL_TERMS:
        agreed = {"goal_alignment": 0.99, term: 0.5}
        assert DivergenceDetector._is_diverged(agreed, 0.99, 0.7), term


def test_a_genuinely_diverged_run_still_reports_diverged(tmp_path):
    """ADMISSION HALF (§2c). A control suite that only proves refusals is
    blind in one direction by construction — and this change makes the
    detector strictly harder to trigger, which is exactly when that matters."""
    d = _detector(tmp_path)
    d.start_task("bad", FL4_GOAL)
    for i in range(6):
        d.record_tool_call("grep", {"pattern": f"p{i}"}, "", False, task_id="bad")
    r = d.evaluate([{"role": "assistant", "content": "unrelated prose"}], [],
                   task_id="bad")
    assert r.diverged, f"a failing, repeating, off-goal run was not flagged: {r}"
    assert r.components["tool_failure_rate"] == 1.0


def test_the_two_traces_SEPARATE(tmp_path):
    """THE OUTCOME CHECK. fl4 below threshold, SYMBIOTE above, same aggregation.

    A tie is the failure mode that matters and it is not hypothetical: the
    first attempt at this floor produced 0.750/diverged for BOTH, with
    byte-identical components. A signal that gives the same answer for a
    compliant turn and a stuck one is not a signal, and no threshold fixes it.
    """
    fl4 = _drive(_detector(tmp_path / "a"), "fl4", FL4_GOAL, FL4_TOOL_CALLS)
    fl4_r = fl4.evaluate(FL4_MESSAGES, [], task_id="fl4")
    sym = _drive(_detector(tmp_path / "b"), "sym", SYMBIOTE_GOAL,
                 SYMBIOTE_TOOL_CALLS)
    sym_r = sym.evaluate(SYMBIOTE_MESSAGES, [], task_id="sym")

    assert fl4_r.score != sym_r.score, (
        f"THE TRACES TIE at {fl4_r.score:.3f}. The repetition signal does not "
        f"discriminate; ship rules 2+3 without the floor rather than a floor "
        f"that separates nothing.\n  fl4={fl4_r.components}\n  sym={sym_r.components}"
    )
    assert not fl4_r.diverged and fl4_r.score < 0.7, fl4_r
    assert sym_r.diverged and sym_r.score >= 0.7, sym_r
