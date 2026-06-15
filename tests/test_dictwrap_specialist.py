"""Dict-wrap specialist train + held-out corpora. The eval's validity rests on
TRAIN and HELDOUT being disjoint in (tool, param) coverage — the Goodhart guard —
so the tests assert that disjointness, not just that tasks generate.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from prometheus.gym.dictwrap_specialist import (
    HELDOUT_UNWRAP_TOOLS,
    generate_dictwrap_heldout,
    generate_dictwrap_train,
)
from prometheus.gym.tasks import load_taskset


def _tools(corpus):
    return {t["score"].get("expect_tool") for t in corpus["tasks"]}


def test_deterministic():
    assert generate_dictwrap_train(50) == generate_dictwrap_train(50)
    assert generate_dictwrap_heldout(8) == generate_dictwrap_heldout(8)


def test_train_reaches_target_distinct_tasks():
    tasks = generate_dictwrap_train(250)["tasks"]
    ids = [t["id"] for t in tasks]
    prompts = [t["prompt"] for t in tasks]
    assert len(ids) == len(set(ids))
    assert len(prompts) == len(set(prompts))
    assert len(tasks) >= 500  # enough distinct tasks for ~hundreds of pairs


def test_new_heldout_tools_are_disjoint_from_train():
    # The four NEW held-out tools never appear in train (grep is the one shared
    # tool, used with a DIFFERENT param — pattern vs root, asserted below).
    train_tools = _tools(generate_dictwrap_train(50))
    new_ho = {"task_get", "cron_create", "download_file", "task_update"}
    assert new_ho.isdisjoint(train_tools)
    assert train_tools == {"task_create", "grep", "task_list"}


def test_heldout_grep_targets_pattern_not_root():
    # held-out grep prompts must NOT mention a directory (that's the trained
    # grep.root shape); they target grep.pattern (a different param).
    ho = generate_dictwrap_heldout(8)["tasks"]
    grep_prompts = [t["prompt"] for t in ho if t["id"].startswith("ho_greppat_")]
    assert grep_prompts
    assert all("directory" not in p for p in grep_prompts)
    tr = generate_dictwrap_train(50)["tasks"]
    train_grep = [t["prompt"] for t in tr if t["id"].startswith("tr_grep_")]
    assert all("directory" in p for p in train_grep)  # train grep IS root-shaped


def test_train_heldout_prompts_fully_disjoint():
    tr = {t["prompt"] for t in generate_dictwrap_train(250)["tasks"]}
    ho = {t["prompt"] for t in generate_dictwrap_heldout(8)["tasks"]}
    assert tr.isdisjoint(ho)


def test_heldout_unwrap_tools_cover_heldout_tools():
    assert _tools(generate_dictwrap_heldout(8)) <= set(HELDOUT_UNWRAP_TOOLS)


def test_real_loader_accepts_both(tmp_path: Path):
    for name, corpus in [("train", generate_dictwrap_train(60)),
                         ("heldout", generate_dictwrap_heldout(8))]:
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump(corpus, sort_keys=False))
        ts = load_taskset(p)
        assert len(ts.tasks) == len(corpus["tasks"])
