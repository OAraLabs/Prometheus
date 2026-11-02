"""build.py round 2: per-target-area caps (--area-cap-for), eval-set disjointness (--eval-disjoint)
and the round-2 recipe (build.py round2). Synthetic rows only; no model, no network."""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_distill_build import GYM, build, ex, feed, gym, make_builder, word, write_jsonl
from tests.test_distill_round2_build import _dataset, _gym_rows

CAPS = {"injection-resistance": 0.2, "daily-agent": 0.3, "coding": 0.3}
NO_CLASS_CAPS: dict[str, float] = {}


def _area_rows(spec: dict[str, int], *, tasks: int = 5, **kw: Any) -> list[Any]:
    rows, i = [], 0
    for area, n in spec.items():
        for j in range(n):
            rows.append(ex(i, "other", f"{area}-t{j % tasks}", area=area, **kw))
            i += 1
    return rows


def _counts(rows: list[Any]) -> dict[str, int]:
    return dict(collections.Counter(e.area for e in rows))


def test_area_caps_are_shares_of_the_new_rows_and_uncapped_areas_keep_everything() -> None:
    rows = _area_rows({"injection-resistance": 60, "daily-agent": 30, "coding": 10,
                       "long-context": 8, "tool-choice": 2})
    kept, _cats, classes = build.cap_train(rows, 1.0, NO_CLASS_CAPS, "seed", None, CAPS)
    # largest fixed point of T = 10 + min(60, .2T) + min(30, .3T) + min(10, .3T): T = 40
    assert _counts(kept) == {"injection-resistance": 8, "daily-agent": 12, "coding": 10,
                             "long-context": 8, "tool-choice": 2}
    n = len(kept)
    assert all(_counts(kept)[a] / n <= cap + 1e-9 for a, cap in CAPS.items())
    rep = classes["area_caps"]
    assert rep["injection-resistance"]["dropped_by_area_cap"] == 52 and rep["injection-resistance"]["cap"] == 0.2
    assert rep["long-context"]["dropped_by_area_cap"] == 0 and rep["long-context"]["cap"] is None
    assert classes["rows_dropped"]["area_cap"] == 52 + 18
    # round-robin over tasks: the 8 injection rows kept cover all 5 tasks
    assert len({e.task_id for e in kept if e.area == "injection-resistance"}) == 5
    # deterministic for a seed; another seed draws other rows
    again, _, _ = build.cap_train(rows, 1.0, NO_CLASS_CAPS, "seed", None, CAPS)
    assert [e.id for e in again] == [e.id for e in kept]
    other, _, _ = build.cap_train(rows, 1.0, NO_CLASS_CAPS, "other-seed", None, CAPS)
    assert {e.id for e in other} != {e.id for e in kept} and _counts(other) == _counts(kept)


def test_no_area_caps_is_the_old_behaviour() -> None:
    rows = _area_rows({"injection-resistance": 30, "coding": 30}) + [ex(500 + i, "coding", f"c{i}") for i in range(9)]
    a = build.cap_train(rows, 0.35, {"no_tool_answer": 0.2}, "seed")
    b = build.cap_train(rows, 0.35, {"no_tool_answer": 0.2}, "seed", None, None)
    assert [e.id for e in a[0]] == [e.id for e in b[0]] and "area_caps" not in a[2]


def test_class_and_category_caps_never_drop_an_uncapped_areas_row() -> None:
    # every long-context row is a no-tool answer and "other"; daily-agent rows give the class cap its excess
    lc = _area_rows({"long-context": 10}, classes=["no_tool_answer"])
    daily = [ex(100 + i, "other", f"d{i % 4}", area="daily-agent", classes=["no_tool_answer"] if i < 10 else [])
             for i in range(20)]
    coding = [ex(200 + i, "coding", f"c{i % 4}", area="coding") for i in range(20)]
    rows = lc + daily + coding
    kept, cats, classes = build.cap_train(rows, 0.35, {"no_tool_answer": 0.2}, "seed", None,
                                          {"daily-agent": 0.5, "coding": 0.5})
    assert _counts(kept)["long-context"] == 10
    # the class cap took its excess from daily-agent only
    assert classes["area_caps"]["long-context"]["dropped_by_class_cap"] == 0
    assert classes["area_caps"]["daily-agent"]["dropped_by_class_cap"] > 0
    assert classes["area_caps"]["long-context"]["dropped_by_category_cap"] == 0
    # with only protected rows left in the class it stays over its cap: the class_caps gate reports it
    only_lc = _area_rows({"long-context": 10}, classes=["no_tool_answer"]) + _area_rows({"coding": 4})
    kept2, cl2 = build.cap_classes(only_lc, {"no_tool_answer": 0.2}, "seed", build.area_protected({"coding": 0.3}))
    assert _counts(kept2)["long-context"] == 10 and cl2["no_tool_answer"]["dropped"] == 0


def test_protected_category_cap_keeps_protected_rows_and_fills_from_the_rest() -> None:
    rows = ([ex(i, "other", f"o{i % 3}", area="long-context") for i in range(6)]
            + [ex(10 + i, "other", f"x{i % 3}", area="daily-agent") for i in range(20)]
            + [ex(40 + i, "coding", f"c{i}", area="coding") for i in range(10)])
    prot = build.area_protected({"daily-agent": 0.3, "coding": 0.3})
    kept, report = build.cap_categories(rows, 0.35, "seed", protected=prot)
    # two categories cannot both be <= 35%: unprotected, the fixed point is empty; protected,
    # "other" keeps its 6 long-context rows (over its cap: the category_cap gate says so)
    assert sum(1 for e in kept if e.area == "long-context") == 6
    assert report["other"]["kept"] == 6 and report["coding"]["kept"] == 3
    plain, _ = build.cap_categories(rows, 0.35, "seed")
    assert plain == []


def _r2(area: str, task: str, category: str = "other", **meta: Any) -> dict[str, Any]:
    return gym(task, "bash", {"command": f"make {task}"}, category=category, target_area=area, round=2, **meta)


def _r2_rows(spec: dict[str, int], prefix: str = "") -> list[dict[str, Any]]:
    rows, i = [], 0
    for area, n in spec.items():
        for _ in range(n):
            rows.append(_r2(area, prefix + "gym" + word(i)))
            i += 1
    return rows


def test_a_build_caps_round2_areas_reports_them_and_val_mirrors_train(tmp_path: Path) -> None:
    rows = _r2_rows({"injection-resistance": 60, "daily-agent": 30, "coding": 10, "long-context": 8,
                     "tool-choice": 2, "error-recovery": 5})
    b = make_builder(tmp_path, train_sources=(GYM,), eval_task_frac=0.0, val_frac=0.1, category_cap=1.0,
                     holdout_max_group_share=0.5, area_caps=CAPS, area_caps_reason="test")
    feed(b, write_jsonl(tmp_path / "r2.jsonl", rows), "extra")
    info = b.split()
    tr = b.final["train"]
    gates = b.gates()
    g = gates["area_caps"]
    assert g["pass"], g
    assert g["new_train_rows"] == len(tr) and not g["uncapped_dropped"]
    pre = info["train_class_caps"]["area_caps"]
    for a in ("long-context", "tool-choice", "error-recovery"):
        assert pre[a]["kept"] == pre[a]["before"]  # uncapped: everything that reached train stays
    # val keeps the share of each area train kept
    va = collections.Counter(e.area for e in b.final["val"])
    assert info["val_class_caps"]["area_caps"]
    for a, r in pre.items():
        before_val = sum(1 for e in b.examples if e.split == "val" and (e.area or "none") == a)
        if before_val and r["before"]:
            ratio = r["kept"] / r["before"]
            assert abs(va.get(a, 0) - before_val * ratio) <= 1.0 or va.get(a, 0) == 1
    report = b.report(info, gates, catalog="stubs", head_tokens={}, inputs={}, elapsed=0.0)
    assert report["caps"]["area_caps"]["base"].startswith("new round-2 train rows")
    r2 = report["round2"]
    assert r2["injection-resistance"]["input"] == 60 and r2["injection-resistance"]["accepted"] == 60
    assert r2["long-context"]["final"]["train"] + r2["long-context"]["final"]["val"] == 8
    assert report["splits"]["train"]["area"]["injection-resistance"] <= 0.2 * len(tr)
    json.dumps(report)  # aggregates only, serialisable


def test_replay_scales_with_the_capped_new_rows_and_stays_outside_the_area_caps(tmp_path: Path) -> None:
    prev = _dataset(tmp_path, "prev", _gym_rows("old", 80, ["coding", "other"]))
    rows = _r2_rows({"injection-resistance": 40, "daily-agent": 10, "long-context": 10}, prefix="new")
    (tmp_path / "w").mkdir()
    b = make_builder(tmp_path / "w", train_sources=(GYM,), eval_task_frac=0.0, val_frac=0.0, category_cap=1.0,
                     area_caps=CAPS, area_caps_reason="test", replay_from=str(prev), replay_share=0.4)
    feed(b, write_jsonl(tmp_path / "r2.jsonl", rows), "extra")
    b.split()
    new = len(b.final["train"])
    # T = 10 + .2T + .3T: 20 new rows
    assert new == 20 and _counts(b.final["train"])["long-context"] == 10
    info = b.add_replay()
    assert info["added"] == round(0.4 * new / 0.6)
    g = b.gates()["area_caps"]
    assert g["pass"] and g["new_train_rows"] == new
    assert {e.area for e in b.final["train"] if e.kind == "replay"} == {""}
    # the shares of the final train set are the new-row shares x (1 - replay share)
    assert g["share_of_final_train"]["injection-resistance"] < g["share"]["injection-resistance"]


def test_eval_disjoint_excludes_rows_and_replay_holding_an_eval_user_message(tmp_path: Path) -> None:
    rows = _r2_rows({"daily-agent": 6})
    leak_prompt = rows[0]["messages"][-2]["content"]  # "please handle the <task> job"
    cases = write_jsonl(tmp_path / "eval-x.jsonl", [
        {"id": "c1", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": leak_prompt}]},
        {"id": "c2", "messages": [{"role": "user", "content": "yes"}]},  # too short to count
    ])
    users, info = build.read_eval_users([cases])
    assert info["user_messages"] == 1 and info["files"][0]["cases"] == 2
    prev = _dataset(tmp_path, "prev", [_r2("daily-agent", "gymoldone")]
                    + _gym_rows("old", 5, ["other"]))
    prev_rows = [json.loads(x) for x in (prev / "train.jsonl").read_text().splitlines() if x.strip()]
    prev_rows[0]["messages"][-2]["content"] = leak_prompt  # a replay row that holds the eval prompt
    (prev / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prev_rows))
    (tmp_path / "w").mkdir()
    b = make_builder(tmp_path / "w", train_sources=(GYM,), eval_task_frac=0.0, val_frac=0.0, category_cap=1.0,
                     replay_from=str(prev), replay_share=0.4)
    b.eval_users, b.eval_disjoint_info = users, info
    feed(b, write_jsonl(tmp_path / "r2.jsonl", rows), "extra")
    assert b.stats["excluded:eval_set_overlap"] == 1
    assert b.round2_report()["daily-agent"]["excluded"] == {"eval_set_overlap": 1}
    b.split()
    rep = b.add_replay()
    assert rep["dropped"].get("dropped:leak_eval_set") == 1
    g = b.gates()["eval_disjoint"]
    assert g["pass"] and g["excluded_input_rows"] == 1 and g["replay_dropped"] == 1
    with pytest.raises(ValueError):
        build.read_eval_users([write_jsonl(tmp_path / "empty.jsonl", [{"id": "x"}])])


def _args(*argv: str) -> Any:
    return build.parse_args(list(argv))


def test_area_cap_args_are_checked() -> None:
    base = ["build", "--out", "o", "--pack", "p", "--tokenizer-dir", "t"]
    assert build.area_cap_args(_args(*base, "--area-cap-for", "coding=0.3", "--area-cap-reason", "r")) == {"coding": 0.3}
    with pytest.raises(ValueError, match="unknown area"):
        build.area_cap_args(_args(*base, "--area-cap-for", "injection=0.2", "--area-cap-reason", "r"))
    with pytest.raises(ValueError, match="needs --area-cap-reason"):
        build.area_cap_args(_args(*base, "--area-cap-for", "coding=0.3"))
    with pytest.raises(ValueError):
        build.area_cap_args(_args(*base, "--area-cap-for", "coding=1.5", "--area-cap-reason", "r"))
    assert build.area_cap_args(_args(*base)) == {}


def test_cli_refuses_no_golden_with_traces(tmp_path: Path, capsys: Any) -> None:
    a = _args("build", "--out", str(tmp_path / "o"), "--pack", "p", "--tokenizer-dir", "t", "--no-golden",
              "--traces", "x.jsonl")
    assert build.cmd_build(a) == 2 and "--no-golden" in capsys.readouterr().err


def test_round2_recipe_command_carries_every_rule(tmp_path: Path, capsys: Any) -> None:
    out = str(tmp_path / "build-r2-max-derived")
    a = _args("round2", "--out", out, "--dry-run", "--print", "--judge-workdir", str(tmp_path / "nojudge"))
    assert build.cmd_round2(a) == 0
    cap = capsys.readouterr()
    cmd = cap.out
    for part in ("--no-golden", "--area-cap-for injection-resistance=0.2", "--area-cap-for daily-agent=0.3",
                 "--area-cap-for coding=0.3", "--category-cap 1", "--replay-share 0.4",
                 "--unjudged-verifiers deterministic,tool-success", "--judge-calibration",
                 "--allow-max-derived", "--max-derived-approval 'operator Q6b 2026-10-30'", "--exclude-sessions",
                 "--eval-holdout", "--eval-disjoint", "--dry-run", "--train-sources qwen38-27b-gym"):
        assert part in cmd, part
    assert "--verdicts" not in cmd and "judge_pending row is excluded" in cap.err
    # calibrated verdicts join once the judge-pending pass wrote them; extra build args pass through
    wd = tmp_path / "judge"
    wd.mkdir()
    (wd / "judge-sample.jsonl").write_text("")
    a = _args("round2", "--out", out, "--print", "--judge-workdir", str(wd), "--", "--limit", "50")
    argv, notes = build.round2_argv(a)
    assert argv[argv.index("--verdicts") + 1] == str(wd / "judge-sample.jsonl") and not notes
    assert argv[-2:] == ["--limit", "50"]
    # the recipe's command line parses as a build
    assert build.parse_args(argv).area_cap_for == ["injection-resistance=0.2", "daily-agent=0.3", "coding=0.3"]


def test_round2_plan_sizes_generation_per_area() -> None:
    def area(inp: int, pend: int, acc: int, before: int, kept: int, cap: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return ({"input": inp, "input_judge_pending": pend, "input_by_verifier": {"tool-success": inp - pend},
                 "accepted": acc, "final": {"train": kept, "val": 0}},
                {"before": before, "kept": kept, "cap": cap})
    spec = {"injection-resistance": area(1000, 200, 800, 700, 96, 0.2),
            "daily-agent": area(1100, 20, 1000, 900, 144, 0.3),
            "coding": area(250, 200, 40, 36, 36, 0.3),
            "error-recovery": area(300, 130, 150, 130, 130, None),
            "long-context": area(50, 15, 30, 26, 26, None),
            "tool-choice": area(27, 9, 18, 16, 16, None)}
    report = {"round2": {a: v[0] for a, v in spec.items()},
              "caps": {"area_caps": {"caps": CAPS, "train": {a: v[1] for a, v in spec.items()}}},
              "replay": {"added": 299}, "gates": {"area_caps": {"new_train_rows": 448}, "train_size": {"value": 747}},
              "splits": {"val": {"rows": 20}, "eval-tools": {"rows": 60}}}
    plan = build.round2_plan(report, 5000, 0.4)
    assert plan["target"]["new_train"] == 3000 and plan["judge_pending_backlog"] == 574
    need = plan["target"]["need"]
    assert need["injection-resistance"] == {"train_rows_at_cap": 600, "train_rows_short": 0, "passing_rows_short": 0}
    assert need["coding"]["train_rows_short"] == 900 - 36
    u = need["uncapped (error-recovery, long-context, tool-choice)"]
    assert u["train_rows_at_least"] == 600 and u["train_rows_short"] == 600 - 172
    assert plan["areas"]["coding"]["dropped_by_caps"] == 0 and plan["areas"]["daily-agent"]["dropped_by_caps"] == 756
