"""Gym reports — baseline vs experiment, with honesty about sample size.

The n=3-is-thin lesson, encoded: no winner declarations below the sample
bar. The report SHOWS everything but only CALLS a result when both arms
clear MIN_RUNS_PER_ARM judged runs overall, and flags every per-category
cell under MIN_RUNS_PER_CELL.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from typing import Any

MIN_RUNS_PER_ARM = 30
MIN_RUNS_PER_CELL = 6


def _aggregate(rows: list[sqlite3.Row]) -> dict[str, Any]:
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"runs": 0, "passed": 0})
    by_task: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"runs": 0, "passed": 0, "category": "", "fail_reasons": []}
    )
    total = {"runs": 0, "passed": 0, "repairs": 0, "dropped": 0,
             "breaker_trips": 0, "feedback_retries": 0, "harness_errors": 0}
    for r in rows:
        total["runs"] += 1
        total["passed"] += r["success"]
        total["repairs"] += r["repairs"]
        total["dropped"] += r["dropped_malformed"]
        total["breaker_trips"] += r["breaker_tripped"]
        total["feedback_retries"] += r["feedback_retries"]
        total["harness_errors"] += 1 if r["error"] else 0
        by_cat[r["category"]]["runs"] += 1
        by_cat[r["category"]]["passed"] += r["success"]
        t = by_task[r["task_id"]]
        t["runs"] += 1
        t["passed"] += r["success"]
        t["category"] = r["category"]
        if r["fail_reasons"] and len(t["fail_reasons"]) < 3:
            t["fail_reasons"].append(r["fail_reasons"][:200])
    return {"total": total, "by_category": dict(by_cat), "by_task": dict(by_task)}


def _rate(passed: int, runs: int) -> str:
    return f"{passed}/{runs} ({passed / runs:.0%})" if runs else "0/0 (—)"


def render_report(
    series: str,
    experiment: str,
    rows: list[sqlite3.Row],
    baseline_rows: list[sqlite3.Row] | None = None,
    *,
    manifest_notes: str = "",
) -> str:
    exp = _aggregate(rows)
    base = _aggregate(baseline_rows) if baseline_rows else None

    lines: list[str] = [
        f"# Gym report — series `{series}`, experiment `{experiment}`",
        "",
    ]
    if manifest_notes:
        lines += [f"> {manifest_notes}", ""]

    t = exp["total"]
    lines += [
        "## Totals",
        "",
        f"- Runs: **{t['runs']}** — passed {_rate(t['passed'], t['runs'])}",
        f"- Adapter repairs: {t['repairs']} · malformed drops: {t['dropped']} · "
        f"breaker trips: {t['breaker_trips']} · loop feedback retries: {t['feedback_retries']}",
    ]
    if t["harness_errors"]:
        lines.append(f"- ⚠️ harness-level errors (timeouts/crashes): {t['harness_errors']}")
    lines.append("")

    if base:
        b = base["total"]
        lines += [
            "## Verdict",
            "",
        ]
        if t["runs"] < MIN_RUNS_PER_ARM or b["runs"] < MIN_RUNS_PER_ARM:
            lines.append(
                f"**INSUFFICIENT n** — arms have {b['runs']} vs {t['runs']} runs; "
                f"the bar for any verdict is {MIN_RUNS_PER_ARM} per arm. "
                f"Numbers below are descriptive only."
            )
        else:
            br = b["passed"] / b["runs"]
            er = t["passed"] / t["runs"]
            delta = er - br
            lines.append(
                f"Baseline {br:.0%} → experiment {er:.0%} "
                f"(**{delta:+.0%}** over {b['runs']}+{t['runs']} runs). "
                + (
                    "Within noise for samples this size — treat as a wash "
                    "unless per-category effects below are large and consistent."
                    if abs(delta) < 0.10
                    else "Effect size is large enough to act on at this n."
                )
            )
        lines.append("")

    lines += ["## By category", ""]
    header = "| category | " + ("baseline | experiment | Δ |" if base else "pass rate |")
    sep = "|---|" + ("---|---|---|" if base else "---|")
    lines += [header, sep]
    cats = sorted(set(exp["by_category"]) | set(base["by_category"] if base else {}))
    for cat in cats:
        e = exp["by_category"].get(cat, {"runs": 0, "passed": 0})
        thin = " ⚠️thin" if 0 < e["runs"] < MIN_RUNS_PER_CELL else ""
        if base:
            bcat = base["by_category"].get(cat, {"runs": 0, "passed": 0})
            bthin = " ⚠️thin" if 0 < bcat["runs"] < MIN_RUNS_PER_CELL else ""
            d = ""
            if bcat["runs"] and e["runs"]:
                d = f"{e['passed'] / e['runs'] - bcat['passed'] / bcat['runs']:+.0%}"
            lines.append(
                f"| {cat} | {_rate(bcat['passed'], bcat['runs'])}{bthin} "
                f"| {_rate(e['passed'], e['runs'])}{thin} | {d} |"
            )
        else:
            lines.append(f"| {cat} | {_rate(e['passed'], e['runs'])}{thin} |")
    lines.append("")

    lines += ["## By task", ""]
    if base:
        lines += ["| task | baseline | experiment |", "|---|---|---|"]
    else:
        lines += ["| task | pass rate | sample failure |", "|---|---|---|"]
    for task_id in sorted(exp["by_task"]):
        e = exp["by_task"][task_id]
        if base:
            bt = base["by_task"].get(task_id, {"runs": 0, "passed": 0})
            lines.append(
                f"| {task_id} | {_rate(bt['passed'], bt['runs'])} "
                f"| {_rate(e['passed'], e['runs'])} |"
            )
        else:
            sample = e["fail_reasons"][0] if e["fail_reasons"] else ""
            lines.append(
                f"| {task_id} | {_rate(e['passed'], e['runs'])} | {sample[:90]} |"
            )
    lines.append("")
    lines.append(
        f"_Sample bar: ≥{MIN_RUNS_PER_ARM} runs/arm for a verdict; "
        f"cells under {MIN_RUNS_PER_CELL} runs flagged thin. "
        f"Deterministic predicate scoring; no LLM judging._"
    )
    return "\n".join(lines)


def report_json(rows: list[sqlite3.Row]) -> str:
    agg = _aggregate(rows)
    # fail_reasons lists make this non-serializable-stable; truncate
    return json.dumps(agg, indent=2, default=str)
