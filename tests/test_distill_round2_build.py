"""build.py for round 2: rows verified without the local judge (uncalibrated categories) and
replay of a previous dataset's train rows. Synthetic rows only; no model, no network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.test_distill_build import (  # noqa: F401 - helpers
    GYM,
    _load,
    GYM_MODEL,
    VERIFIED,
    build,
    feed,
    gym,
    make_builder,
    read_jsonl,
    word,
    write_jsonl,
)


def r2_record(verifier: str, *, category_calibrated: list[str] | None = None, det_pass: bool = True,
              checks: list[str] | None = None) -> dict[str, Any]:
    rec = {"verified": True, "tool_success": True, "sandboxed": True, "side_effects": False, "round": 2,
           "verifier": verifier, "verifiers": ["tool-success"] + (["deterministic"] if verifier == "deterministic" else []),
           "judge": {"applied": False, "reason": "category other is not calibrated for the local judge",
                     "calibrated_categories": category_calibrated if category_calibrated is not None
                     else ["coding", "sentinel/daemon"]},
           "r2": {"deterministic": {"checks": checks if checks is not None else
                                    (["injection"] if verifier == "deterministic" else []),
                                    "pass": det_pass, "failed": []}}}
    return rec


def _v(meta: dict[str, Any], **kw: Any) -> Any:
    return build.verification_of(meta, strict=True, judge_models=(GYM_MODEL,), **kw)


def test_an_unjudged_round2_row_trains_only_on_an_allowed_verifier_outside_calibrated_categories() -> None:
    ok = _v({"category": "other", "verification": r2_record("deterministic")},
            unjudged_verifiers=build.UNJUDGED_VERIFIERS)
    assert ok.decision is True and ok.basis == "r2-deterministic"
    ts = _v({"category": "memory/wiki", "verification": r2_record("tool-success")},
            unjudged_verifiers=build.UNJUDGED_VERIFIERS)
    assert ts.decision is True and ts.basis == "r2-tool-success"
    # without the allow (the old rule) a row with no judge never trains
    assert _v({"category": "other", "verification": r2_record("deterministic")}).basis == "no_judge"
    # a narrower allow refuses tool success alone
    assert _v({"category": "other", "verification": r2_record("tool-success")},
              unjudged_verifiers=("deterministic",)).basis == "unjudged_verifier_not_allowed"
    # the judge was skipped in a category it is calibrated for: fail closed
    assert _v({"category": "coding", "verification": r2_record("tool-success")},
              unjudged_verifiers=build.UNJUDGED_VERIFIERS).basis == "judge_skipped_in_calibrated_category"
    # "deterministic" needs its checks to have passed
    bad = r2_record("deterministic", det_pass=False)
    assert _v({"category": "other", "verification": bad},
              unjudged_verifiers=build.UNJUDGED_VERIFIERS).basis == "deterministic_not_passed"
    # a round-1 record cannot claim it
    old = dict(r2_record("tool-success"), round=1)
    assert _v({"category": "other", "verification": old},
              unjudged_verifiers=build.UNJUDGED_VERIFIERS).basis == "unjudged_verifier_not_allowed"
    # a judged record is read exactly as before
    assert _v({"category": "other", "verification": VERIFIED},
              unjudged_verifiers=build.UNJUDGED_VERIFIERS).basis == "judge-pass"


def test_unjudged_round2_gym_rows_enter_train_with_their_label(tmp_path: Path) -> None:
    rows = [gym("gym" + word(t), "bash", {"command": f"make {t}"}, category="other",
                verification=r2_record("deterministic")) for t in range(20)]
    b = make_builder(tmp_path, train_sources=(GYM,), eval_task_frac=0.0, val_frac=0.0, category_cap=1.0)
    feed(b, write_jsonl(tmp_path / "r2.jsonl", rows), "extra")
    b.split()
    assert len(b.final["train"]) == 20 and {e.label for e in b.final["train"]} == {"r2-deterministic"}
    (tmp_path / "n").mkdir()
    b2 = make_builder(tmp_path / "n", train_sources=(GYM,), eval_task_frac=0.0, val_frac=0.0, category_cap=1.0,
                      unjudged_verifiers=None)
    feed(b2, write_jsonl(tmp_path / "r2b.jsonl", rows), "extra")
    b2.split()
    assert not b2.final["train"] and b2.stats["excluded:verification_failed"] == 20
    assert b2.stats["verification:no_judge"] == 20


# ── replay ────────────────────────────────────────────────────────────────────


def _dataset(tmp_path: Path, name: str, rows: list[dict[str, Any]], **kw: Any) -> Path:
    out = tmp_path / name
    out.mkdir(parents=True)
    (tmp_path / f"{name}-work").mkdir(exist_ok=True)
    b = make_builder(tmp_path / f"{name}-work", train_sources=(GYM,), eval_task_frac=0.0, val_frac=0.0,
                     category_cap=1.0, **kw)
    feed(b, write_jsonl(tmp_path / f"{name}-in.jsonl", rows), "extra")
    info = b.split()
    gates = b.gates()
    written = b.write(out)
    report = b.report(info, gates, catalog="stubs", head_tokens={}, inputs={}, elapsed=0.0)
    (out / "report.json").write_text(json.dumps(report))
    (out / build.DATASET_MANIFEST).write_text(json.dumps(b.dataset_manifest(out, written, report,
                                                                            out / "report.json")))
    return out


def _gym_rows(prefix: str, n: int, cats: list[str]) -> list[dict[str, Any]]:
    out = []
    for t in range(n):
        task = prefix + word(t)
        out.append(gym(task, "bash", {"command": f"make {task}"}, category=cats[t % len(cats)]))
    return out


def test_replay_adds_a_stratified_seeded_share_and_never_leaks_into_val_or_eval(tmp_path: Path) -> None:
    prev_rows = _gym_rows("old", 60, ["coding", "coding", "other"])
    # one previous task comes back in the new build's own rows (and lands in val or eval-tools)
    shared = prev_rows[0]
    prev = _dataset(tmp_path, "prev", prev_rows)
    prev_train = read_jsonl(prev / "train.jsonl")
    assert len(prev_train) == 60
    new_rows = _gym_rows("new", 80, ["other", "memory/wiki"]) + [shared]
    (tmp_path / "w").mkdir()
    b = make_builder(tmp_path / "w", train_sources=(GYM,), eval_task_frac=0.2, val_frac=0.1,
                     holdout_max_group_share=0.5, category_cap=1.0, replay_from=str(prev), replay_share=0.4)
    feed(b, write_jsonl(tmp_path / "new.jsonl", new_rows), "extra")
    b.split()
    new_train = len(b.final["train"])
    info = b.add_replay()
    added = info["added"]
    assert added == min(info["eligible"], round(0.4 * new_train / 0.6))
    assert abs(info["share_of_train"] - added / (added + new_train)) < 1e-9
    # stratified: the replay keeps the previous dataset's coding:other mix (2:1) within one row
    cats = info["by_category"]
    assert abs(cats.get("coding", 0) - 2 * cats.get("other", 0)) <= 2
    gates = b.gates()
    assert gates["leakage"]["pass"], gates["leakage"]
    held = {e.task_id for k in ("val", "eval-tools") for e in b.final[k]}
    assert not held & {e.task_id for e in b.final["train"] if e.kind == "replay"}
    if any(e.task_id == build.trace_keys(shared).h_prompt for k in ("val", "eval-tools") for e in b.final[k]):
        assert sum(v for k, v in info["dropped"].items() if k.startswith("dropped:leak_")) >= 1
    # the same seed draws the same rows
    (tmp_path / "w2").mkdir()
    b2 = make_builder(tmp_path / "w2", train_sources=(GYM,), eval_task_frac=0.2, val_frac=0.1,
                      holdout_max_group_share=0.5, category_cap=1.0, replay_from=str(prev), replay_share=0.4)
    (tmp_path / "again").mkdir()
    feed(b2, write_jsonl(tmp_path / "again" / "new.jsonl", new_rows), "extra")
    b2.split()
    b2.add_replay()
    assert [e.id for e in b2.final["train"]] == [e.id for e in b.final["train"]]
    out = tmp_path / "out"
    out.mkdir()
    b.write(out)
    rows = read_jsonl(out / "train.jsonl")
    rep = [r for r in rows if r.get("replay_from") == "prev"]
    assert len(rep) == added and all(r["text"] and r["loss_spans_tok"] for r in rep)
    report = b.report({"plan": build.SplitPlan({}, 0, 0, 0), "train_caps": {}, "val_caps": {},
                       "train_class_caps": {}, "val_class_caps": {}, "held_rows": 0, "eval_sampled": 0,
                       "fail_used": 0, "fail_dropped_task_in_train_or_val": 0}, gates, catalog="stubs",
                      head_tokens={}, inputs={}, elapsed=0.0)
    assert report["replay"]["added"] == added and report["replay"]["new_train_rows"] == new_train


def test_max_derived_replay_needs_the_allow_and_makes_the_dataset_max_derived(tmp_path: Path) -> None:
    prev = _dataset(tmp_path, "prev", _gym_rows("old", 20, ["coding"]))
    # the previous dataset's rows as Max-derived ones under the Q6b approval
    rows = read_jsonl(prev / "train.jsonl")
    for r in rows:
        r.update(max_derived=True, model="qwen3.8-max", source="max-golden",
                 max_derived_approval="operator Q6b 2026-10-30")
    write_jsonl(prev / "train.jsonl", rows)
    new_rows = _gym_rows("new", 30, ["other"])
    (tmp_path / "w").mkdir()
    b = make_builder(tmp_path / "w", train_sources=(GYM,), eval_task_frac=0.0, val_frac=0.0, category_cap=1.0,
                     replay_from=str(prev))
    feed(b, write_jsonl(tmp_path / "new.jsonl", new_rows), "extra")
    b.split()
    info = b.add_replay()
    assert info["added"] == 0 and info["dropped"]["dropped:max_derived_needs_allow"] == 20
    (tmp_path / "w2").mkdir()
    b = make_builder(tmp_path / "w2", train_sources=(GYM,), eval_task_frac=0.0, val_frac=0.0, category_cap=1.0,
                     replay_from=str(prev), allow_max_derived=True,
                     max_derived_approval="operator Q6b 2026-10-30 (round 2)")
    feed(b, write_jsonl(tmp_path / "new2.jsonl", new_rows), "extra")
    b.split()
    info = b.add_replay()
    assert info["added"] == 20 and info["max_derived_added"] == 20  # 0.4 x 30 / 0.6 = 20
    assert b.train_val_max_derived() is True
    gate = b.policy_gate()
    assert gate["pass"] and gate["max_derived_covered_in_train_val"] == 20
    share = b.share()["by_split"]["train"]
    assert share["max"]["rows"] == 20 and share[GYM]["rows"] == 30


def test_cli_refuses_max_derived_replay_without_the_allow(tmp_path: Path, capsys: Any) -> None:
    prev = _dataset(tmp_path, "prev", _gym_rows("old", 5, ["coding"]))
    rows = read_jsonl(prev / "train.jsonl")
    for r in rows:
        r["max_derived"] = True
    write_jsonl(prev / "train.jsonl", rows)
    a = build.parse_args(["build", "--out", str(tmp_path / "o"), "--pack", str(tmp_path / "p.json"),
                          "--tokenizer-dir", str(tmp_path), "--replay-from", str(prev)])
    assert build.cmd_build(a) == 2
    assert "needs --allow-max-derived" in capsys.readouterr().err
    a = build.parse_args(["build", "--out", str(tmp_path / "o"), "--pack", str(tmp_path / "p.json"),
                          "--tokenizer-dir", str(tmp_path), "--replay-from", str(prev), "--replay-share", "1.5"])
    assert build.cmd_build(a) == 2


def test_allocate_is_proportional_and_capped() -> None:
    assert build.allocate({"a": 70, "b": 20, "c": 10}, 10) == {"a": 7, "b": 2, "c": 1}
    assert build.allocate({"a": 3, "b": 100}, 50) == {"a": 1, "b": 49}
    assert sum(build.allocate({"a": 1, "b": 1}, 5).values()) == 2


# ── judge_pending: only the calibrated judge decides ─────────────────────────

CAL_HOST, CAL_SHA, CAL_QUANT = "tower-4090", "b" * 64, "UD-Q4_K_XL"
def _calibration(tmp_path: Path) -> Path:
    rep = {"schema": "distill-judge-calibration/2", "view": "outcome", "rubric_version": "2",
           "decision": {"decision": "use-local-judge"},
           "by_category": [{"category": "coding", "apply": True}, {"category": "other", "apply": False}],
           "judge": {"models": {"qwen3.8-27b-ud-q4_k_xl": 9}, "hosts": {CAL_HOST: 9}, "quants": {CAL_QUANT: 9},
                     "sha256": [CAL_SHA]}}
    p = tmp_path / "calibration-report.json"
    p.write_text(json.dumps(rep))
    return p


def _pending(task: str, *, scope: str = "calibrated-pending", category: str = "coding",
             prose: bool = False) -> dict[str, Any]:
    rec = r2_record("tool-success")
    rec["judge"].update(pending=True)
    rec["judge_pending"] = True
    row = gym(task, "bash", {"command": f"make {task}"}, category=category, verification=rec, round=2,
              judge_scope=scope, judge_pending=scope == "calibrated-pending", run_id=f"run-{task}",
              call_result={"status": "success", "text": f"built {task}", "injected": False})
    if prose:
        row["messages"][-1] = {"role": "assistant", "content": f"Done: {task} is built and the tests pass."}
        row["_meta"].pop("tool_name")
        row["_meta"].pop("call_result")
    return row


def _verdict(item: dict[str, Any], verdict: str, **who: Any) -> dict[str, Any]:
    row = {"id": item["id"], "label": "pass", "category": item["category"], "source": item["source"],
           "view": "outcome", "rubric_version": "2", "provider": "local", "judge_key": "none",
           "model": "qwen3.8-27b-ud-q4_k_xl", "judge_route": "remote", "judge_host": CAL_HOST,
           "judge_quant": CAL_QUANT, "judge_sha256": CAL_SHA, "status": "ok", "verdict": verdict}
    row.update(who)
    return row


def _builder(tmp_path: Path, verdict_files: list[Path], cal: Path | None) -> Any:
    stats: Any = build.collections.Counter()
    checks: dict[str, Any] = {}
    judges: dict[str, Any] = {}
    aliases: dict[str, str] = {}
    verdicts = build.load_verdicts([str(p) for p in verdict_files], stats, checks, judges, aliases)
    s = build.Settings(train_range=(1, 100000), eval_min=0, min_presence=0, train_sources=(GYM,),
                       eval_task_frac=0.0, val_frac=0.0, category_cap=1.0,
                       calibrated_judge=build.calibrated_judge_of(cal) if cal else None)
    from tests.test_distill_build import make_pack, make_student
    b = build.Builder(student=make_student(), pack=make_pack(), catalog="stubs", settings=s, index={},
                      verdicts=verdicts, spool=tmp_path / "spool.jsonl", verdict_checks=checks,
                      verdict_judges=judges, verdict_aliases=aliases)
    b.stats.update(stats)
    return b


def test_judge_pending_rows_train_only_on_the_calibrated_judges_verdict(tmp_path: Path, capsys: Any) -> None:
    cal = _calibration(tmp_path)
    rows = [_pending("gym" + word(t)) for t in range(6)] + [_pending("prose", scope="not-applicable", prose=True)]
    traces = write_jsonl(tmp_path / "qwen38-27b-gym.r2.traces.jsonl", rows)
    work = tmp_path / "judge-pending"
    a = build.parse_args(["judge-pending", "--traces", str(traces), "--calibration", str(cal),
                          "--workdir", str(work)])
    assert build.cmd_judge_pending(a) == 0
    items = read_jsonl(work / "judge-queue.jsonl")
    meta = json.loads((work / "judge-queue.meta.json").read_text())
    assert meta["mode"] == "local-judge" and meta["calibration"]["judge_quants"] == [CAL_QUANT]
    assert len(items) == 6 and all(it["outcome"]["text"].startswith("built ") for it in items)
    assert all(it["source"]["file"] == traces.name and it["id"] == "r2p-" + it["source"]["h_exact"] for it in items)
    # judge.py renders the item in its outcome view, result included
    jd = _load("distill_judge", "judge.py")
    text, _cut = jd.render_prompt(items[0], view="outcome")
    assert "## What the tool returned" in text and items[0]["outcome"]["text"] in text
    # no verdict yet: the call rows wait, the prose row trains on its own checks
    (tmp_path / "a").mkdir()
    b = _builder(tmp_path / "a", [], cal)
    feed(b, traces, "extra")
    assert b.stats["excluded:judge_pending"] == 6
    assert [e.label for e in b.examples] == ["r2-tool-success"]
    # the calibrated judge's verdicts: 4 pass, 1 fail; one row judged by the IQ3 run judge (ignored)
    vs = [_verdict(it, "pass") for it in items[:4]] + [_verdict(items[4], "fail")] \
        + [_verdict(items[5], "pass", judge_quant="IQ3_XXS", judge_host="mini-host", judge_sha256="c" * 64)]
    vfile = write_jsonl(work / "judge-sample.jsonl", vs)
    (tmp_path / "b").mkdir()
    b = _builder(tmp_path / "b", [vfile], cal)
    feed(b, traces, "extra")
    labels = sorted(e.label for e in b.examples)
    assert labels.count("r2-calibrated-judge") == 4 and "r2-tool-success" in labels
    assert b.stats["excluded:judge_fail"] == 1 and b.stats["excluded:judge_pending"] == 1
    assert b.stats["judge_pending_uncalibrated_verdict_ignored"] == 1
    b.split()
    assert b.gates()["verdicts"]["pass"]
    # without --judge-calibration nothing is the calibrated judge: every call row waits
    (tmp_path / "c").mkdir()
    b = _builder(tmp_path / "c", [vfile], None)
    feed(b, traces, "extra")
    assert b.stats["excluded:judge_pending"] == 6
    # the verdicts still find their rows when the trace file is copied under another name (h_exact)
    (tmp_path / "inputs-r2").mkdir()
    copied = write_jsonl(tmp_path / "inputs-r2" / "qwen38-27b-gym.copy.jsonl", rows)
    (tmp_path / "d").mkdir()
    b = _builder(tmp_path / "d", [vfile], cal)
    feed(b, copied, "extra")
    assert sorted(e.label for e in b.examples).count("r2-calibrated-judge") == 4
    b.split()
    assert b.gates()["verdicts"]["unmatched"] == 0
    # a second export leaves out what the calibrated judge already rated
    a = build.parse_args(["judge-pending", "--traces", str(traces), "--calibration", str(cal),
                          "--workdir", str(work), "--done", str(vfile), "--force"])
    assert build.cmd_judge_pending(a) == 8
    capsys.readouterr()


def test_a_prose_scope_that_holds_a_call_is_refused(tmp_path: Path) -> None:
    row = _pending("sneaky", scope="not-applicable")  # a call, but labelled as outside the judge's reach
    (tmp_path / "a").mkdir()
    b = _builder(tmp_path / "a", [], _calibration(tmp_path))
    feed(b, write_jsonl(tmp_path / "x.jsonl", [row]), "extra")
    assert b.stats["excluded:judge_scope_mismatch"] == 1 and not b.examples


def test_outside_the_calibrated_domain_a_call_trains_on_its_deterministic_verifier_only(tmp_path: Path) -> None:
    cal = _calibration(tmp_path)

    def outside(task: str, *, injected: bool = True, verifier: str = "deterministic") -> dict[str, Any]:
        row = _pending(task, scope="outside-calibration-domain")
        row["_meta"].update(judge_skipped_reason="outside_calibration_domain", verifier="deterministic")
        row["_meta"]["call_result"].update(injected=injected, planted=False)
        rec = row["_meta"]["verification"]
        rec["verifier"] = verifier
        rec["r2"]["deterministic"] = {"checks": ["recovery"], "pass": True, "failed": []}
        return row

    rows = [outside("gyma"), outside("gymb", injected=False), outside("gymc", verifier="tool-success")]
    (tmp_path / "a").mkdir()
    b = _builder(tmp_path / "a", [], cal)
    feed(b, write_jsonl(tmp_path / "o.jsonl", rows), "extra")
    assert [e.label for e in b.examples] == ["r2-deterministic"]   # no calibrated verdict needed
    assert b.stats["excluded:judge_scope_mismatch"] == 1             # its own result was not injected
    assert b.stats["verification:outside_domain_needs_deterministic"] == 1
    # never queued for the calibrated judge
    a = build.parse_args(["judge-pending", "--traces", str(tmp_path / "o.jsonl"), "--calibration", str(cal),
                          "--workdir", str(tmp_path / "jp")])
    assert build.cmd_judge_pending(a) == 8


# ── round-2 Max teacher rows (Token Plan, operator 2026-11-06) ────────────────


def _max_r2(task: str, *, pending: bool, category: str = "coding") -> dict[str, Any]:
    """A teacher_gen --teacher max-token-plan round-2 row: Max-derived, its self-judge info only."""
    rec = r2_record("tool-success" if pending else "deterministic", checks=None if pending else ["recovery"])
    rec["judge"].update(info_only={"verdict": "pass", "kind": "self-judge", "provider": "max-token-plan",
                                   "served_model": "qwen3.8-max"})
    meta: dict[str, Any] = dict(source="max-tokenplan-gym", max_derived=True,
                                tags=["max-derived"], teacher="max-token-plan", round=2, run_id=f"run-{task}",
                                target_area="coding" if pending else "error-recovery",
                                call_result={"status": "success", "text": f"built {task}", "injected": False})
    if pending:
        rec["judge"].update(pending=True)
        rec["judge_pending"] = True
        meta.update(judge_scope="calibrated-pending", judge_pending=True)
    else:
        meta.update(judge_scope="uncalibrated-category")
    return gym(task, "bash", {"command": f"make {task}"}, model="qwen3.8-max", category=category,
               verification=rec, **meta)


def _max_builder(tmp_path: Path, verdict_files: list[Path], cal: Path, *, listed: bool = True) -> Any:
    b = _builder(tmp_path, verdict_files, cal)
    b.s.allow_max_derived, b.s.max_derived_approval = True, "operator Q6b 2026-10-30"
    b.s.unjudged_verifiers = build.UNJUDGED_VERIFIERS
    if listed:  # the round-2 recipe lists the round-2 Max source and model (operator 2026-11-06)
        b.s.train_sources = (GYM, *build.R2_RECIPE["max_train_sources"])
        b.s.train_models = (*build.DEFAULT_TRAIN_MODELS, *build.R2_RECIPE["max_train_models"])
    return b


def test_round2_max_rows_reach_the_recipe_and_wait_for_the_calibrated_judge(
        tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    cal = _calibration(tmp_path)
    pool = tmp_path / "pool"
    (pool / "max-derived").mkdir(parents=True)
    rows = [_max_r2("max" + word(t), pending=True) for t in range(3)] \
        + [_max_r2("maxrec", pending=False, category="other")]
    traces = write_jsonl(pool / "max-derived" / "max-tokenplan-gym.max-tokenplan-r2-x-max-derived.traces.jsonl", rows)
    # a round-1 Max run's pool file (no r2 in its run name) stays out of the recipe
    write_jsonl(pool / "max-derived" / "max-tokenplan-gym.max-tokenplan-max-derived.traces.jsonl", rows[:1])
    local = write_jsonl(pool / "qwen38-27b-gym.local-r2-x.traces.jsonl", [_pending("loc" + word(0))])
    globs = [str(pool / Path(p).relative_to("~/distill/phase2b/pool")) for p in build.R2_RECIPE["pool"]]
    monkeypatch.setitem(build.R2_RECIPE, "pool", globs)
    # the recipe reads both pools
    a = build.parse_args(["round2", "--out", str(tmp_path / "r2-max-derived"), "--print"])
    argv, _notes = build.round2_argv(a)
    extra = [argv[i + 1] for i, x in enumerate(argv) if x == "--extra-traces"]
    assert {Path(p).name for p in build.expand(extra)} == {traces.name, local.name}
    # judge-pending's default is the recipe's pool: the Max coding calls are queued with the local one
    work = tmp_path / "jp"
    jp = build.parse_args(["judge-pending", "--calibration", str(cal), "--workdir", str(work)])
    assert build.cmd_judge_pending(jp) == 0
    items = read_jsonl(work / "judge-queue.jsonl")
    assert sorted(it["meta"]["source"] for it in items) == ["max-tokenplan-gym"] * 3 + [GYM]
    capsys.readouterr()
    # no calibrated verdict yet: the Max calls wait (not refused as unverified), the recovery row trains
    (tmp_path / "a").mkdir()
    b = _max_builder(tmp_path / "a", [], cal)
    feed(b, traces, "extra")
    assert b.stats["excluded:judge_pending"] == 3
    assert not any(k.startswith("excluded:max_payg_gym") for k in b.stats)
    assert [(e.label, e.max_derived) for e in b.examples] == [("r2-deterministic", True)]
    # the calibrated judge rules: 2 pass, 1 fail
    mx = [it for it in items if it["meta"]["source"] == "max-tokenplan-gym"]
    vfile = write_jsonl(work / "judge-sample.jsonl",
                        [_verdict(mx[0], "pass"), _verdict(mx[1], "pass"), _verdict(mx[2], "fail")])
    (tmp_path / "b").mkdir()
    b = _max_builder(tmp_path / "b", [vfile], cal)
    feed(b, traces, "extra")
    assert sorted(e.label for e in b.examples) == ["r2-calibrated-judge"] * 2 + ["r2-deterministic"]
    assert all(e.max_derived and e.source == "max-tokenplan-gym" for e in b.examples)
    assert b.stats["excluded:judge_fail"] == 1
    # a self-judge verdict alone never decides a pending Max call
    (tmp_path / "c").mkdir()
    selfv = write_jsonl(tmp_path / "self.jsonl", [_verdict(mx[0], "pass", judge_quant="IQ3_XXS",
                                                           judge_host="mini-host", judge_sha256="c" * 64)])
    b = _max_builder(tmp_path / "c", [selfv], cal)
    feed(b, traces, "extra")
    assert b.stats["excluded:judge_pending"] == 3 and b.stats["judge_pending_uncalibrated_verdict_ignored"] == 1


# ── Q6b for round-2 Max rows (operator 2026-11-06): the train-source guard ──


def test_round2_max_rows_train_only_when_listed_and_round1_max_rows_are_unchanged(tmp_path: Path) -> None:
    cal = _calibration(tmp_path)
    r2 = [_max_r2("maxrec" + word(t), pending=False, category="other") for t in range(3)]
    r1 = gym("maxround1", "bash", {"command": "make maxround1"}, model="qwen3.8-max", source="max-tokenplan-gym",
             max_derived=True, tags=["max-derived"],
             verification={**VERIFIED, "judge": {"verdict": "pass", "provider": "max-token-plan",
                                                 "judge_key": "token-plan", "model": "qwen3.8-max"}})
    traces = write_jsonl(tmp_path / "max.jsonl", [*r2, r1])
    # the allow alone: round-2 Max rows are refused (no split at all), a round-1 Max gym row still trains
    (tmp_path / "a").mkdir()
    b = _max_builder(tmp_path / "a", [], cal, listed=False)
    feed(b, traces, "extra")
    assert b.stats["excluded:max_r2_not_listed"] == 3
    assert [(e.source, e.max_derived, e.train_ok) for e in b.examples] == [("max-tokenplan-gym", True, True)]
    # listed source but not the model (or the other way round): still refused
    for sources, models in (((GYM, "max-tokenplan-gym"), build.DEFAULT_TRAIN_MODELS),
                            ((GYM,), (*build.DEFAULT_TRAIN_MODELS, "qwen3.8-max*"))):
        (tmp_path / "x").mkdir(exist_ok=True)
        b = _max_builder(tmp_path / "x", [], cal, listed=False)
        b.s.train_sources, b.s.train_models = sources, models
        feed(b, traces, "extra")
        assert b.stats["excluded:max_r2_not_listed"] == 3
    # listed (the round-2 recipe): they train, Max-derived under the approval, and are covered
    (tmp_path / "b").mkdir()
    b = _max_builder(tmp_path / "b", [], cal)
    feed(b, traces, "extra")
    assert len(b.examples) == 4 and all(e.max_derived and e.train_ok and e.approval == "operator Q6b 2026-10-30"
                                        for e in b.examples)
    b.split()
    g = b.gates()
    assert g["data_policy"]["pass"] and g["data_policy"]["max_derived_uncovered_in_train_val"] == 0
    assert b.train_val_max_derived()
    # without the allow, listing changes nothing: a Max gym row needs the allow first
    (tmp_path / "c").mkdir()
    b = _max_builder(tmp_path / "c", [], cal)
    b.s.allow_max_derived, b.s.max_derived_approval = False, None
    feed(b, traces, "extra")
    assert b.stats["excluded:max_payg_gym_needs_allow"] == 4 and not b.examples


def test_cli_lists_round2_max_sources_only_under_the_allow_and_tags_the_output(
        tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    from tests.test_distill_build import R, make_pack, make_student
    monkeypatch.setattr(R.Student, "load", classmethod(lambda cls, d: make_student()))
    pack = tmp_path / "pack.json"
    pack.write_text(json.dumps(make_pack()))
    rows = [_max_r2("maxrec" + word(t), pending=False, category="other") for t in range(6)] \
        + [gym("loc" + word(t), "bash", {"command": f"make loc{t}"}) for t in range(6)]
    traces = write_jsonl(tmp_path / "pool.jsonl", rows)
    sessions = tmp_path / "exclude-sessions.txt"
    sessions.write_text("incident:s9\n")
    ref = "operator Q6b 2026-10-30; round-2 Max rows operator 2026-11-06"
    listed = ["--train-sources", GYM, "max-tokenplan-gym", "--train-models", "qwen3.8-27b*", "qwen3.8-max*"]
    allow = ["--allow-max-derived", "--max-derived-approval", ref, "--exclude-sessions", str(sessions)]

    def run(out: Path, *extra: str) -> int:
        return build.main(["build", "--no-golden", "--extra-traces", str(traces), "--pack", str(pack),
                           "--tokenizer-dir", "unused", "--out", str(out), "--dry-run", "--eval-min", "0",
                           "--train-range", "1", "1000", "--val-share-range", "0", "1", "--eval-task-frac", "0",
                           "--val-frac", "0", "--min-presence", "0", "--category-cap", "1", *extra])

    tagged, plain = tmp_path / "r2-max-derived", tmp_path / "r2"
    # refused without the allow: the guard stands, and says how it may be lifted
    for flag_args in (listed, ["--train-sources", GYM, "max-tokenplan-gym"],
                      ["--train-models", "qwen3.8-27b*", "qwen3.8-max*"]):
        assert run(tagged, *flag_args) == 2
        err = capsys.readouterr().err
        assert "may not name Max" in err and "--allow-max-derived" in err
    # the allow with an untagged output is refused before anything is read
    assert run(plain, *listed, *allow) == 2
    assert "max-derived" in capsys.readouterr().err and not plain.exists()
    # the Max key never judges, allow or not
    assert run(tagged, *listed, *allow, "--judge-models", "qwen3.8-max") == 2
    assert "--judge-models may not name Max" in capsys.readouterr().err
    # accepted with the allow: logged, Max rows train, every output tagged and private
    assert run(tagged, *listed, *allow) == 1  # INCOMPLETE: no spot check
    err = capsys.readouterr().err
    assert "ALLOW (Q6b): --train-sources lists Max-derived max-tokenplan-gym" in err
    report = json.loads((tagged / "report.json").read_text())
    assert report["max_derived"] is True and report["tags"] == ["max-derived"] and report["private"] is True
    assert report["splits"]["train"]["source"] == {"max-tokenplan-gym": 6, GYM: 6}
    assert report["settings"]["train_sources"] == [GYM, "max-tokenplan-gym"]
    dp = report["data_policy"]
    assert dp["sources"]["max-tokenplan-gym"]["listed"] is True
    assert dp["max_derived_allow"]["approval"] == ref and dp["max_derived_allow"]["covered_in_train_val"] == 6
    assert report["gates"]["data_policy"]["pass"]
    assert report["settings"]["teacher_model"] == build.TEACHER_MODEL  # the guards fail closed
    # the allow without listing: the round-2 Max rows are refused, the local rows train
    assert run(tagged, *allow) == 1
    capsys.readouterr()
    report = json.loads((tagged / "report.json").read_text())
    assert report["splits"]["train"]["source"] == {GYM: 6}
    assert report["counts"]["excluded:max_r2_not_listed"] == 6
    # the round-2 recipe lists them, under the extended approval
    a = build.parse_args(["round2", "--out", str(tagged), "--print"])
    argv, _ = build.round2_argv(a)
    src = argv[argv.index("--train-sources") + 1:argv.index("--train-models")]
    assert src == [GYM, "max-tokenplan-gym"]
    assert "qwen3.8-max*" in argv[argv.index("--train-models") + 1:argv.index("--unjudged-verifiers")]
    assert argv[argv.index("--max-derived-approval") + 1] == ref and "--allow-max-derived" in argv


def test_round2_max_coding_calls_stay_judge_pending_when_listed(tmp_path: Path) -> None:
    """Listing the round-2 Max source changes nothing for a calibrated-category call: it waits
    for the calibrated judge; a self-judge or a Max judge verdict never decides it."""
    cal = _calibration(tmp_path)
    rows = [_max_r2("max" + word(t), pending=True) for t in range(2)] \
        + [_max_r2("maxd", pending=True, category="sentinel/daemon")]
    traces = write_jsonl(tmp_path / "m.jsonl", rows)
    maxv = write_jsonl(tmp_path / "maxv.jsonl", [
        {"id": "x", "source": {"file": traces.name, "line": i}, "status": "ok", "verdict": "pass",
         "provider": "max-token-plan", "judge_key": "token-plan", "model": "qwen3.8-max"} for i in range(3)])
    b = _max_builder(tmp_path, [maxv], cal)
    feed(b, traces, "extra")
    assert b.stats["excluded:judge_pending"] == 3 and not b.examples
    assert b.stats["judge_pending_uncalibrated_verdict_ignored"] == 3


# ── verdicts on rows excluded before their verdict applied ──────────────────


def test_verdicts_on_excluded_rows_are_accounted_and_only_unread_traces_are_unmatched(tmp_path: Path) -> None:
    from tests.test_distill_build import corpus, make_builder, trace
    golden = corpus(tmp_path)  # 40 tasks x 3 git calls, 36 others, 1 duplicate (last-but-one), 1 other model
    lines = golden.read_text().splitlines()
    n = len(lines)
    verdicts = {f"{golden.name}:1": "pass",          # applied
                f"{golden.name}:{n - 1}": "fail",    # the duplicate row: excluded before its verdict
                f"{golden.name}:{n}": "pass",        # another model's row: non_teacher_model
                f"{golden.name}:{n + 5}": "pass",    # no such line in a file that was read
                "golden_traces_9_9.jsonl:3": "pass",  # a trace file that is not an input
                "0123456789abcdef": "fail"}          # a hash no row has
    b = make_builder(tmp_path, verdicts=verdicts, eval_task_frac=0.0, val_frac=0.0, category_cap=1.0)
    feed(b, golden)
    b.split()
    g = b.gates()["verdicts"]
    assert g["on_excluded_rows_by_reason"] == {"duplicate": 1, "non_teacher_model": 1}
    assert g["on_excluded_rows"] == 2
    assert g["unmatched"] == 3 and not g["pass"]
    assert g["unmatched_by_cause"] == {"hash_not_found": 1, "line_not_in_trace_file": 1,
                                       "trace_file_not_an_input": 1}
    # only verdicts that find a read row (applied or on an excluded row): the gate passes
    ok = {k: v for k, v in verdicts.items() if k.startswith(golden.name) and not k.endswith(f":{n + 5}")}
    (tmp_path / "b").mkdir()
    b = make_builder(tmp_path / "b", verdicts=ok, eval_task_frac=0.0, val_frac=0.0, category_cap=1.0)
    feed(b, golden)
    b.split()
    g = b.gates()["verdicts"]
    assert g["pass"] and g["unmatched"] == 0 and g["on_excluded_rows"] == 2
    # a verdict keyed by the row's hash is found on an excluded row too
    dup = build.Trace("x:1", "golden", "x", 1, trace("solo", "bash", {"command": "ls"}, model="other-model"))
    hx = build.trace_keys(dup.raw).h_exact
    (tmp_path / "c").mkdir()
    b = make_builder(tmp_path / "c", verdicts={hx: "fail"}, eval_task_frac=0.0, val_frac=0.0)
    b.add(dup)
    assert b.stats["verdicts_on_excluded:non_teacher_model"] == 1 and b.verdicts_used == {hx}


# ── --per-task-cap ───────────────────────────────────────────────────────────


def _many_calls(tmp_path: Path, tasks: int, calls: int) -> Path:
    from tests.test_distill_build import trace
    rows = [trace(f"req{word(t)}", "bash", {"command": f"step {word(c)} of {word(t)}"}, ts=1_790_000_000 + t * 100 + c)
            for t in range(tasks) for c in range(calls)]
    return write_jsonl(tmp_path / "golden_traces_7_7.jsonl", rows)


def test_per_task_cap_holds_golden_rows_per_request_and_leaves_gym_rows(tmp_path: Path) -> None:
    from tests.test_distill_build import allow_builder
    golden = _many_calls(tmp_path, 6, 12)
    gyms = write_jsonl(tmp_path / "gym.jsonl", [gym("g" + word(t), "bash", {"command": f"step {word(c)}"})
                                                 for t in range(3) for c in range(5)])
    b = allow_builder(tmp_path / "a", eval_task_frac=0.0, val_frac=0.0, category_cap=1.0, per_task_cap=4,
                      per_task_cap_reason="test")
    feed(b, golden)
    feed(b, gyms, "extra")
    info = b.split()
    per = collections_counter(e.task_id for e in b.final["train"] if e.kind == "golden")
    assert len(per) == 6 and set(per.values()) == {4}
    assert sum(1 for e in b.final["train"] if e.source == GYM) == 15  # scope golden: gym rows untouched
    ptc = info["per_task_cap"]["pool"]
    assert ptc["rows_in_scope"] == 72 and ptc["dropped"] == 48 and ptc["tasks_over_cap"] == 6
    assert ptc["rows_per_task_before"] == {"p50": 12, "p90": 12, "max": 12}
    assert sum(1 for e in b.examples if e.split == "per_task_cap") == 48
    g = b.gates()["per_task_cap"]
    assert g["pass"] and g["max_rows_per_task"] == 4 and g["tasks_in_scope"] == 6
    # the choice within a task is seeded: the same rows every time
    b2 = allow_builder(tmp_path / "b", eval_task_frac=0.0, val_frac=0.0, category_cap=1.0, per_task_cap=4,
                       per_task_cap_reason="test")
    feed(b2, golden)
    feed(b2, gyms, "extra")
    b2.split()
    assert sorted(e.id for e in b2.final["train"]) == sorted(e.id for e in b.final["train"])
    # the cap runs before the split: val's quota is taken from capped rows, so a request with many
    # calls cannot fill val and then be cut down to a sliver
    b4 = allow_builder(tmp_path / "d", eval_task_frac=0.0, val_frac=0.25, category_cap=1.0, per_task_cap=4,
                       per_task_cap_reason="test")
    feed(b4, _many_calls(tmp_path / "d", 12, 12))
    b4.split()
    tr, va = len(b4.final["train"]), len(b4.final["val"])
    assert tr + va == 48 and 0.15 <= va / 48 <= 0.35
    assert max(collections_counter(e.task_id for e in b4.final["val"]).values()) == 4
    # scope all caps the gym rows too, but never a row of an uncapped round-2 area
    rows = [gym("g" + word(t), "bash", {"command": f"step {word(c)}"}, target_area=area)
            for t, area in enumerate(["coding", "long-context"]) for c in range(5)]
    b3 = make_builder_r2(tmp_path / "c", per_task_cap=2, per_task_cap_scope="all", per_task_cap_reason="test",
                         area_caps={"coding": 0.9})
    feed(b3, write_jsonl(tmp_path / "gym2.jsonl", rows), "extra")
    b3.split()
    assert collections_counter(e.area for e in b3.final["train"]) == {"coding": 2, "long-context": 5}


def test_per_task_cap_holds_across_replay(tmp_path: Path) -> None:
    from tests.test_distill_build import allow_builder
    golden = _many_calls(tmp_path, 4, 6)
    # a previous dataset whose train holds 6 more rows of the same requests (max-golden), and 4 of others
    prev_rows = []
    for t in range(4):
        for c in range(6, 12):
            prev_rows.append({"task": f"req{word(t)}", "cmd": f"step {word(c)} of {word(t)}", "source": "max-golden"})
    for t in range(4):
        prev_rows.append({"task": f"other{word(t)}", "cmd": f"other {word(t)}", "source": "max-golden"})
    src = _replay_source(tmp_path, prev_rows)
    b = allow_builder(tmp_path / "a", eval_task_frac=0.0, val_frac=0.0, category_cap=1.0, per_task_cap=5,
                      per_task_cap_reason="test", replay_from=str(src), replay_share=0.5)
    feed(b, golden)
    b.split()
    b.add_replay()
    per = collections_counter(e.task_id for e in b.final["train"])
    assert max(per.values()) <= 5
    assert b.replay_info["dropped"]["dropped:per_task_cap"] == 24  # 4 requests already at 5: all 6 each
    assert b.gates()["per_task_cap"]["pass"]


def test_cli_per_task_cap_needs_a_reason(tmp_path: Path, capsys: Any) -> None:
    for extra in (["--per-task-cap", "8"], ["--per-task-cap", "0", "--per-task-cap-reason", "x"]):
        assert build.main(["build", "--no-golden", "--out", str(tmp_path / "o"), "--pack", "p.json",
                           "--tokenizer-dir", "unused", "--dry-run", *extra]) == 2
        assert "--per-task-cap" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def collections_counter(it: Any) -> dict[str, int]:
    import collections
    return dict(collections.Counter(it))


def make_builder_r2(root: Path, **kw: Any) -> Any:
    from tests.test_distill_build import make_builder
    root.mkdir(parents=True, exist_ok=True)
    return make_builder(root, train_sources=(GYM,), eval_task_frac=0.0, val_frac=0.0, category_cap=1.0, **kw)


def _replay_source(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    """A previous dataset whose train rows read like golden Max rows (source max-golden, Q6b)."""
    prev = _dataset(tmp_path, "prev", [gym(r["task"], "bash", {"command": r["cmd"]}) for r in rows])
    train = read_jsonl(prev / "train.jsonl")
    assert len(train) == len(rows)
    for r in train:
        r.update(max_derived=True, model="qwen3.8-max", source="max-golden",
                 max_derived_approval="operator Q6b 2026-10-30")
    write_jsonl(prev / "train.jsonl", train)
    return prev
