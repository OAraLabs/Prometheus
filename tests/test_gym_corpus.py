"""Harvest-corpus generator (flywheel corpus) — deterministic, schema-valid,
diverse. The whole point is DISTINCT values → distinct context_hashes → distinct
training pairs, so the tests assert distinctness and that the real loader accepts
every generated task.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from prometheus.gym.corpus import generate_harvest_corpus
from prometheus.gym.tasks import load_taskset


def test_deterministic():
    assert generate_harvest_corpus(20) == generate_harvest_corpus(20)


def test_distinct_ids_and_prompts():
    corpus = generate_harvest_corpus(25)
    tasks = corpus["tasks"]
    ids = [t["id"] for t in tasks]
    prompts = [t["prompt"] for t in tasks]
    assert len(ids) == len(set(ids)), "task ids must be unique"
    # Distinct prompts are the diversity lever — the harvest dedups on the call,
    # which mirrors the prompt's target value.
    assert len(prompts) == len(set(prompts)), "prompts must be distinct"


def test_high_ceiling_shapes_scale_with_per_shape():
    small = generate_harvest_corpus(5)["tasks"]
    big = generate_harvest_corpus(40)["tasks"]
    # task_create (×2 shapes) and grep are unbounded → more per_shape, more tasks.
    assert len(big) > len(small)
    tc = [t for t in big if t["id"].startswith("hv_tc_cmd_")]
    assert len(tc) == 40  # the command shape yields exactly per_shape distinct tasks


def test_task_list_status_is_capped():
    # Only a handful of valid statuses — must not fabricate duplicates to fill N.
    tasks = generate_harvest_corpus(50)["tasks"]
    statuses = [t for t in tasks if t["id"].startswith("hv_tl_status_")]
    assert 1 <= len(statuses) <= 5


def test_side_effecting_tools_are_stubbed():
    tasks = generate_harvest_corpus(10)["tasks"]
    for t in tasks:
        if t["score"]["expect_tool"] == "task_create":
            assert "task_create" in t.get("stub_tools", []), \
                "task_create tasks must stub the tool (unwrap fires pre-execution)"


def test_real_loader_accepts_the_corpus(tmp_path: Path):
    # The decisive check: the gym's own load_taskset must accept every task
    # (required keys present, score predicates allowed, ids unique).
    corpus = generate_harvest_corpus(30)
    p = tmp_path / "harvest.yaml"
    p.write_text(yaml.safe_dump(corpus, sort_keys=False))
    ts = load_taskset(p)
    assert len(ts.tasks) == len(corpus["tasks"])
    assert all(t.category == "harvest" for t in ts.tasks)
