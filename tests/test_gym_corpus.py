"""Harvest-corpus generator (failure-mode breadth). #40 proved value diversity
but yielded 97.7% one transition; this generator is organized BY transition and
must emit a SPREAD — the tests assert the spread, not just the count, and that
the real loader accepts every generated task.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from prometheus.gym.corpus import generate_harvest_corpus, intended_distribution
from prometheus.gym.tasks import load_taskset
from prometheus.learning.transition_taxonomy import TRANSITION_TYPES


def test_deterministic():
    assert generate_harvest_corpus(15) == generate_harvest_corpus(15)


def test_distinct_ids_and_prompts():
    tasks = generate_harvest_corpus(20)["tasks"]
    ids = [t["id"] for t in tasks]
    prompts = [t["prompt"] for t in tasks]
    assert len(ids) == len(set(ids))
    assert len(prompts) == len(set(prompts))


def test_corpus_spans_at_least_five_transition_types():
    # The whole point: induce a SPREAD, not 97.7% dict-unwrap. Every targeted
    # transition (all but "other") must be represented in the INTENDED mix.
    dist = intended_distribution(generate_harvest_corpus(20))
    targeted = [t for t in TRANSITION_TYPES if t != "other"]
    present = [t for t in targeted if dist.get(t, 0) > 0]
    assert len(present) >= 5, f"only {len(present)} transitions induced: {dist}"


def test_per_transition_cap_respected():
    # No single lesson dominates: each transition capped at per_transition.
    cap = 12
    dist = intended_distribution(generate_harvest_corpus(cap))
    for transition, n in dist.items():
        assert n <= cap, f"{transition} has {n} > cap {cap}"


def test_dict_wrap_does_not_dominate():
    # Balanced: dict-unwrap is at most ~its fair share, not the bulk.
    dist = intended_distribution(generate_harvest_corpus(20))
    total = sum(dist.values())
    assert dist.get("dict_wrap_unwrap", 0) <= total * 0.4


def test_task_create_tasks_stub_the_tool():
    tasks = generate_harvest_corpus(15)["tasks"]
    for t in tasks:
        if t["score"].get("expect_tool") == "task_create":
            assert "task_create" in t.get("stub_tools", [])


def test_real_loader_accepts_every_task(tmp_path: Path):
    corpus = generate_harvest_corpus(20)
    p = tmp_path / "harvest.yaml"
    p.write_text(yaml.safe_dump(corpus, sort_keys=False))
    ts = load_taskset(p)
    assert len(ts.tasks) == len(corpus["tasks"])
    assert all(t.category.startswith("harvest_") for t in ts.tasks)
