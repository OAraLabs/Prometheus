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
