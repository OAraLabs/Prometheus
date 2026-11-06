"""Qwen Flash as a judge on the Token Plan (operator, 2026-11-06).

judge.py ``--provider token-plan --judge-model qwen3.8-flash``: the Max judge's exact rubric and prompt,
calibrated against Max's EXISTING verdicts (``subset --which max-judged --sample``, ``calibrate
--judge-provider flash-token-plan``) without a Max call, then bulk judging of the items Max never judged
(``subset --which pending``), whose verdicts build.py applies. No real API and no real key: the
Token Plan host is re-pointed at a loopback fake, and HOME is under tmp_path.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_distill_judge import golden_item, max_row
from tests.test_distill_spend import FAIL_REPLY, PASS_REPLY, chat_body, write_queue
from tests import test_distill_token_plan as tpt
from tests.test_distill_token_plan import StepAPI, ToAPI, _tp_judge_args, build, judge, spend, tp_rows

#: the Token Plan tests' fixtures, used here by name: HOME under tmp_path with no ambient key, a fake
#: QWEN_API_KEY in its env file, the loopback API, and no connection beyond loopback
_home = tpt._home
plan = tpt.plan
api = tpt.api
no_outbound = tpt.no_outbound

FLASH = "qwen3.8-flash"
MAX = "qwen3.8-max"


def _src(tmp: Path, items: list[dict[str, Any]], rows: list[dict[str, Any]], name: str = "phase2-rejudge") -> Path:
    src = tmp / name
    write_queue(src, items, {"mode": "rejudge", "view": "outcome"})
    (src / judge.RESULTS_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return src


def _max_rows(items: list[dict[str, Any]], verdicts: dict[str, str]) -> list[dict[str, Any]]:
    out = []
    for it in items:
        if it["id"] in verdicts:
            row = max_row(it, verdicts[it["id"]])
            row.update(input_tokens=5000, output_tokens=110, cached_input_tokens=0)
            out.append(row)
    return out


def test_flash_judges_a_calibration_queue_with_the_max_prompt_and_its_own_rows(
        plan: Path, _home: Path, api: StepAPI) -> None:
    items = [golden_item(i) for i in range(6)]
    for it in items:
        it["outcome"] = {"status": "success", "text": f"{it['id']} ok"}
    src = _src(_home, items, _max_rows(items, {it["id"]: "pass" for it in items}))
    cal = _home / "flash-cal"
    assert judge.main(["--workdir", str(cal), "subset", "--from", str(src), "--which", "max-judged"]) == 0
    api.steps += [{"body": chat_body(PASS_REPLY, model=FLASH)} for _ in range(6)]
    assert judge.main(_tp_judge_args(_home, cal, "--judge-model", FLASH), opener=ToAPI(api)) == judge.EXIT_DONE
    rows = judge.read_jsonl(cal / judge.RESULTS_FILE)
    assert {(r["provider"], r["judge_key"], r["model"], r["phase"]) for r in rows} == {
        ("flash-token-plan", "token-plan", FLASH, judge.PHASE_FLASH_CAL)}
    assert all(r["verification"]["judge"]["provider"] == "flash-token-plan" for r in rows)
    # byte for byte the Max judge's system prompt, user prompt, temperature, thinking and max_tokens
    queued = {it["id"]: it for it in judge.read_jsonl(cal / judge.QUEUE_FILE)}
    for req in api.requests:
        b = req["body"]
        assert b["model"] == FLASH
        user = b["messages"][1]["content"]
        it = next(q for q in queued.values() if judge.render_prompt(q, view="outcome")[0] == user)
        want = judge.judge_payload(judge.system_prompt("outcome"), judge.render_prompt(it, view="outcome")[0],
                                   model=MAX, max_tokens=1024)
        assert b["messages"] == want["messages"]
        assert (b["temperature"], b["enable_thinking"], b["max_tokens"]) == (0, False, 1024)
    tp = tp_rows(_home)
    assert len(tp) == 6 and {r["model"] for r in tp} == {FLASH} and {r["phase"] for r in tp} == {"2-judge-flash-cal"}
    # the Max workdir's own rows are untouched
    assert len(judge.read_jsonl(src / judge.RESULTS_FILE)) == 6


def test_flash_never_judges_a_max_workdir_and_max_never_a_calibration_queue(
        plan: Path, _home: Path, api: StepAPI, capsys: pytest.CaptureFixture[str]) -> None:
    items = [golden_item(i) for i in range(3)]
    src = _src(_home, items, _max_rows(items, {items[0]["id"]: "pass"}))
    # Flash on the Max re-judge workdir: refused before any call
    assert judge.main(_tp_judge_args(_home, src, "--judge-model", FLASH), opener=ToAPI(api)) == judge.EXIT_FATAL
    assert "never a Max workdir" in capsys.readouterr().err
    plain = _home / "phase2"
    write_queue(plain, items)
    assert judge.main(_tp_judge_args(_home, plain, "--judge-model", FLASH), opener=ToAPI(api)) == judge.EXIT_FATAL
    cal = _home / "cal"
    assert judge.main(["--workdir", str(cal), "subset", "--from", str(src), "--which", "max-judged"]) == 0
    # Max on a calibration queue: refused (its verdicts are the reference)
    assert judge.main(_tp_judge_args(_home, cal), opener=ToAPI(api)) == judge.EXIT_FATAL
    assert "never Max" in capsys.readouterr().err
    # a model the Token Plan does not judge with
    assert judge.main(_tp_judge_args(_home, cal, "--judge-model", "qwen3.8-plus"), opener=ToAPI(api)) \
        == judge.EXIT_FATAL
    assert api.requests == []
    assert "qwen3.8-flash" in spend.TOKEN_PLAN_MODELS and "qwen3.8-plus" not in spend.TOKEN_PLAN_MODELS


def test_the_calibration_sample_is_stratified_by_category_and_verdict_and_weighted() -> None:
    items, verdicts = [], {}
    for cat, n_pass, n_fail in (("coding", 300, 80), ("memory/wiki", 40, 10), ("other", 20, 1),
                                ("sentinel/daemon", 23, 5)):
        for v, n in (("pass", n_pass), ("fail", n_fail)):
            for _ in range(n):
                it = golden_item(len(items), category=cat)
                items.append(it)
                verdicts[it["id"]] = v
    picked, cells = judge.calibration_sample([dict(it) for it in items], verdicts, 120, seed=7,
                                             min_per_category=30, min_per_verdict=8)
    assert len(picked) == 120 and len({it["id"] for it in picked}) == 120
    got = collections.Counter((it["category"], verdicts[it["id"]]) for it in picked)
    # the small categories whole; every verdict represented; fails at their floor where they have it
    assert got[("other", "pass")] == 20 and got[("other", "fail")] == 1
    assert got[("sentinel/daemon", "pass")] == 23 and got[("sentinel/daemon", "fail")] == 5
    assert got[("memory/wiki", "fail")] >= 8 and got[("coding", "fail")] >= 8
    assert sum(c["sampled"] for c in cells.values()) == 120 and sum(c["pool"] for c in cells.values()) == len(items)
    for it in picked:
        cell = cells[f"{it['category']}|{verdicts[it['id']]}"]
        assert it["stratum_weight"] == pytest.approx(cell["pool"] / cell["sampled"])
    again, _ = judge.calibration_sample([dict(it) for it in items], verdicts, 120, seed=7,
                                        min_per_category=30, min_per_verdict=8)
    assert sorted(it["id"] for it in again) == sorted(it["id"] for it in picked)


def test_calibrate_flash_reports_weighted_agreement_kappa_and_tokens_for_both_judges(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    items = [golden_item(i, category="coding" if i < 40 else "other") for i in range(60)]
    mx = {it["id"]: ("fail" if i % 5 == 0 else "pass") for i, it in enumerate(items)}
    src = _src(tmp_path, items, _max_rows(items, mx))
    cal = tmp_path / "flash-cal"
    assert judge.main(["--workdir", str(cal), "subset", "--from", str(src), "--which", "max-judged",
                       "--sample", "30", "--min-per-category", "10", "--min-per-verdict", "4"]) == 0
    meta = json.loads((cal / judge.QUEUE_META_FILE).read_text())
    assert meta["sample"]["n"] == 30 and meta["n_pass"] + meta["n_fail"] == 30
    queue = judge.read_jsonl(cal / judge.QUEUE_FILE)
    assert all(set(q["source"]) == {"calibration_of"} for q in queue)  # build.py can never apply these
    flip = {queue[0]["id"]}
    rows = [{"id": q["id"], "status": "ok", "view": "outcome", "rubric_version": judge.RUBRIC_VERSION,
             "provider": "flash-token-plan", "judge_key": "token-plan", "model": FLASH,
             "verdict": ("pass" if mx[q["id"]] == "fail" else "fail") if q["id"] in flip else mx[q["id"]],
             "input_tokens": 4000, "output_tokens": 90} for q in queue]
    # a local judge's row on the same item is not Flash's
    rows.append({**rows[1], "provider": "local", "model": "qwen3.8-27b", "verdict": "fail"})
    (cal / judge.RESULTS_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert judge.main(["--workdir", str(cal), "calibrate", "--judge-provider", "flash-token-plan",
                       "--threshold", "0.9", "--min-category-pairs", "10"]) == judge.EXIT_DONE
    text = capsys.readouterr().out
    rep = json.loads((cal / judge.CALIBRATION_FILE).read_text())
    assert rep["judge_provider"] == "flash-token-plan" and rep["pairs"] == 30 and rep["agreement"]["k"] == 29
    assert rep["tokens"]["judge"] == {"n": 30, "input": 4000.0, "output": 90.0, "cached_input": None,
                                      "usage_estimated": 0}
    assert rep["tokens"]["max"]["input"] == 5000.0 and rep["tokens"]["max"]["output"] == 110.0
    assert 0.9 < rep["agreement_weighted"] < 1.0
    assert all(c["agreement_weighted"] is not None for c in rep["by_category"])
    assert "Flash calibration against Max" in text and "DECISION: use Flash" in text
    # without the flag, the local judge's single row is what is compared: incomplete
    assert judge.main(["--workdir", str(cal), "calibrate"]) == judge.EXIT_INCOMPLETE
    capsys.readouterr()


def test_pending_queue_keeps_sources_excludes_areas_and_build_applies_flash_verdicts(
        plan: Path, _home: Path, api: StepAPI, capsys: pytest.CaptureFixture[str]) -> None:
    items = [golden_item(i) for i in range(8)]
    items[6]["meta"]["target_area"] = "injection-resistance"
    items[7]["meta"]["target_area"] = "daily-agent"
    items[5]["meta"]["target_area"] = "coding"
    rows = _max_rows(items, {items[0]["id"]: "pass", items[1]["id"]: "fail"})
    rows.append(max_row(items[2], "", status="unparseable"))
    src = _src(_home, items, rows)
    bulk = _home / "flash-bulk"
    assert judge.main(["--workdir", str(bulk), "subset", "--from", str(src), "--which", "pending",
                       "--exclude-area", "injection-resistance", "--exclude-area", "daily-agent"]) == 0
    capsys.readouterr()
    queue = judge.read_jsonl(bulk / judge.QUEUE_FILE)
    assert [q["id"] for q in queue] == [items[i]["id"] for i in (2, 3, 4, 5)]
    assert all(q["source"]["file"] == "golden_traces_1_1.jsonl" for q in queue)
    meta = json.loads((bulk / judge.QUEUE_META_FILE).read_text())
    assert meta["mode"] == judge.MODE_PENDING and meta["max_judged"] == 2
    assert meta["excluded_by_area"] == {"daily-agent": 1, "injection-resistance": 1}
    assert meta["max_attempted_no_verdict"] == {"coding": 1}
    # --sample and --exclude-area belong to their own modes
    assert judge.main(["--workdir", str(_home / "x"), "subset", "--from", str(src), "--which", "pending",
                       "--sample", "3"]) == judge.EXIT_ERROR
    assert judge.main(["--workdir", str(_home / "y"), "subset", "--from", str(src), "--which", "max-judged",
                       "--exclude-area", "coding"]) == judge.EXIT_ERROR
    # the local judge never judges a pending queue (it is not calibrated for this)
    assert judge.main(["--workdir", str(bulk), "run", "--provider", "local", "--local-url",
                       "http://127.0.0.1:9", "--no-live-probe"]) == judge.EXIT_FATAL
    api.steps += [{"body": chat_body(PASS_REPLY, model=FLASH)}, {"body": chat_body(FAIL_REPLY, model=FLASH)},
                  {"body": chat_body(PASS_REPLY, model=FLASH)}, {"body": chat_body(PASS_REPLY, model=FLASH)}]
    assert judge.main(_tp_judge_args(_home, bulk, "--judge-model", FLASH, "--concurrency", "1"),
                      opener=ToAPI(api)) == judge.EXIT_DONE
    out = judge.read_jsonl(bulk / judge.RESULTS_FILE)
    assert {r["phase"] for r in out} == {judge.PHASE_FLASH} and {r["provider"] for r in out} == {"flash-token-plan"}
    # build.py: Max's file first, Flash's last; Flash verdicts land on their traces
    stats: collections.Counter = collections.Counter()
    v = build.load_verdicts([str(src / judge.RESULTS_FILE), str(bulk / judge.RESULTS_FILE)], stats)
    want = {f"golden_traces_1_1.jsonl:{r['source']['line'] + 1}": r["verdict"] for r in out}
    assert {k: v[k] for k in want} == want and v["golden_traces_1_1.jsonl:1"] == "pass"
    assert stats["verdict_rows_max_judge"] == 0 and stats["verdict_rows_tokenplan_max_judge"] == 2
