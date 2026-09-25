"""The frozen ladder suite (gym/ladder/v1) and the rung manifest.

The load-bearing test is ``test_every_verdict_discriminates``: for EVERY task
with a mechanical verdict it proves the check fails on the untouched setup
and on each listed wrong answer, and passes on the reference solution. A task
that cannot pass that proof is not a measurement.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from prometheus.gym.ladder.selfcheck import selfcheck_task
from prometheus.gym.ladder.suite import CLASS_IDS, load_suite, select_tasks

REPO = Path(__file__).resolve().parents[1]
SUITE_DIR = REPO / "gym" / "ladder" / "v1"
RUNGS = REPO / "gym" / "ladder" / "rungs.yaml"
MIN_TASKS_PER_CLASS = 20

SUITE = load_suite(SUITE_DIR)


# The first cut (WP-2.1 scope, 2026-09-24): four classes populated, four
# defined with their budgets and success criterion but no tasks yet.
ACTIVE = ["qa", "single_tool", "multi_step", "file_edit"]
DEFERRED = ["web_research", "memory_recall", "scheduling", "long_haul"]


def test_all_eight_classes_are_defined_with_a_success_criterion():
    assert list(SUITE.classes) == list(CLASS_IDS)
    for cid, spec in SUITE.classes.items():
        assert len(spec.success_criterion) > 40, cid


def test_first_cut_populates_four_classes_and_defers_four():
    assert SUITE.active_classes == ACTIVE
    assert SUITE.deferred_classes == DEFERRED
    for cid in DEFERRED:
        assert SUITE.by_class()[cid] == [], cid


@pytest.mark.parametrize("cid", ACTIVE)
def test_class_has_enough_default_tasks(cid):
    default = [t for t in select_tasks(SUITE) if t.task_class == cid]
    assert len(default) >= MIN_TASKS_PER_CLASS, (
        f"{cid}: {len(default)} tasks run by default, need {MIN_TASKS_PER_CLASS}")


@pytest.mark.parametrize("cid", ACTIVE)
def test_smoke_set_is_a_few_per_class(cid):
    smoke = [t for t in select_tasks(SUITE, smoke=True) if t.task_class == cid]
    assert 2 <= len(smoke) <= 4, f"{cid}: {len(smoke)} smoke tasks"


@pytest.mark.parametrize("task", SUITE.tasks, ids=lambda t: t.id)
def test_every_verdict_discriminates(task, tmp_path):
    assert selfcheck_task(task, tmp_path / "sandbox") == []


@pytest.mark.parametrize("cid", ACTIVE)
def test_every_task_is_graded_and_the_class_spans_all_three(cid):
    tasks = [t for t in SUITE.tasks if t.task_class == cid]
    missing = [t.id for t in tasks if not t.difficulty]
    assert not missing, f"{cid}: no difficulty on {missing}"
    assert {t.difficulty for t in tasks} == {"easy", "medium", "hard"}, cid


REPLY_KEYS = {"expect_answer", "expect_text_any", "expect_text_all", "expect_text_regex",
              "expect_text_contains"}


def test_every_reply_check_carries_its_proof_both_ways():
    """A check on the final reply must be shown to credit right answers phrased
    differently AND to reject plausible wrong ones — the audit found checks
    that failed '12600.0' and checks that passed 'x = -16'."""
    for t in SUITE.tasks:
        if set(t.score) & REPLY_KEYS:
            ref = t.reference
            assert len(ref.get("right_answers") or []) >= 3, (t.id, "right_answers")
            assert len(ref.get("wrong_answers") or []) >= 2, (t.id, "wrong_answers")


def test_every_answer_line_task_proves_the_format_miss_rule():
    """Format misses stay apart from wrong answers: every answer-line task
    shows a reply without the line that is a FORMAT MISS, and declares the
    answer's shape so a bare wrong value is still scored as wrong."""
    for t in SUITE.tasks:
        if "expect_answer" in t.score:
            assert t.score.get("answer_shape"), (t.id, "answer_shape")
            assert len(t.reference.get("format_misses") or []) >= 1, (t.id, "format_misses")
            assert "ANSWER:" in t.prompt, (t.id, "the prompt must ask for the line")


def test_every_acceptance_goes_through_the_harness_runner():
    # {unittest} <modules>: the isolated runner that requires tests to RUN.
    # A free-form command's exit status proves nothing (see verdict.py).
    from prometheus.gym.ladder.verdict import ACCEPTANCE_RE

    for t in SUITE.tasks:
        if t.acceptance:
            assert ACCEPTANCE_RE.match(t.acceptance), (t.id, t.acceptance)


def test_web_tasks_are_flagged():
    # Vacuous while web_research is deferred; binding the moment it has tasks.
    for t in SUITE.tasks:
        if t.task_class == "web_research" or t.web:
            assert t.web in ("offline", "live"), t.id
            assert isinstance(t.answer_changes_over_time, bool), t.id


def test_no_real_hosts_or_private_addresses_in_the_suite():
    """An allowlist, not a denylist: every URL must sit on a reserved .example
    host (RFC 2606), so no machine name has to be spelled here to be refused
    (the pre-commit hook scans committed files for those)."""
    text = "\n".join(p.read_text() for p in sorted(SUITE_DIR.glob("*.yaml")))
    text += RUNGS.read_text()
    hosts = {h.lower() for h in re.findall(r"https?://([A-Za-z0-9.-]+)", text)}
    assert all(h.endswith(".example") for h in hosts), sorted(hosts)
    assert not re.search(r"\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+\b", text)
    assert not re.search(r"\b(192\.168|10)\.\d+\.\d+\.\d+", text)


def test_rungs_are_well_formed_and_the_judge_is_not_a_rung():
    data = yaml.safe_load(RUNGS.read_text())
    judge = data["judge"]["model"].lower()
    ids = [r["id"] for r in data["rungs"]]
    assert len(ids) == len(set(ids))
    # First run: about 4B, about 14B, and the 27B.
    assert data["first_run"] == ["r04b", "r14b", "r27b"]
    assert set(data["first_run"]) <= set(ids)
    for r in data["rungs"]:
        for key in ("id", "size_class", "model", "match", "quantization", "adapter_tier"):
            assert r.get(key), (r.get("id"), key)
        assert not re.search(r["match"], judge, re.IGNORECASE), r["id"]
        assert judge not in r["model"].lower(), r["id"]
