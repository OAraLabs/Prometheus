#!/usr/bin/env python3
"""Run the model-ladder suite against one served model.

Usage (endpoints come from flags or env, never from this file):

    # a smoke run: the tasks tagged smoke: true, one run each
    uv run python scripts/ladder_run.py --provider ollama \\
        --base-url "$LADDER_BASE_URL" --model qwen2.5:7b-instruct \\
        --judge-base-url "$LADDER_JUDGE_BASE_URL" --judge-model <pinned judge> \\
        --smoke --run-label smoke-qwen2.5-7b

    # a rung of the ladder (WP-2.2 / WP-2.3): model match, quant and judge pin
    # come from gym/ladder/rungs.yaml
    uv run python scripts/ladder_run.py --rung r08b --provider llama_cpp \\
        --base-url "$LADDER_BASE_URL" --judge-base-url "$LADDER_JUDGE_BASE_URL"

    # regenerate the report / re-run the empty-field check from telemetry.db
    uv run python scripts/ladder_run.py --report-only --run-label smoke-qwen2.5-7b

Rows land in telemetry.db (default ~/.prometheus/ladder/telemetry.db — the
live daemon's telemetry.db only if you pass it). A markdown report goes to
gym/results/ladder/<run-label>.md; it names no hosts.

Exit codes: 0 = ran and every required field is populated somewhere;
1 = the empty-field check failed (or nothing was recorded); 2 = refused
before any task ran; 3 = aborted mid-run (dead endpoint, unclean sandbox) —
the rows recorded before the abort stand and are reported.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import logging
import os
import re
import sqlite3
import sys
from pathlib import Path

import yaml

from prometheus.gym.ladder.record import (
    check_empty_fields,
    forced_tier,
    load_rows,
    render_ladder_table,
    render_report,
    render_tier_sweep,
)
from prometheus.gym.ladder.runner import (
    DEFAULT_TELEMETRY_DB,
    Contestant,
    JudgePin,
    LadderAbort,
    LadderPreflightError,
    run_ladder,
)
from prometheus.gym.ladder.suite import CLASS_IDS, load_suite, select_tasks

REPO = Path(__file__).resolve().parents[1]
RUNGS = REPO / "gym" / "ladder" / "rungs.yaml"


def _csv(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def _label(model: str) -> str:
    slug = re.sub(r"[^a-z0-9.]+", "-", os.path.basename(model).lower()).strip("-")
    return f"{slug or 'model'}-{dt.datetime.now():%Y%m%d-%H%M%S}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Model-ladder runner (WP-2.1 suite)")
    ap.add_argument("--suite", default=str(REPO / "gym" / "ladder" / "v1"))
    ap.add_argument("--provider", default="llama_cpp",
                    choices=["llama_cpp", "ollama", "lm_studio", "vllm"])
    ap.add_argument("--base-url", default=os.environ.get("LADDER_BASE_URL"))
    ap.add_argument("--model", default="", help="blank = ask the backend")
    ap.add_argument("--quant", default="", help="declared quantization (the probe wins)")
    ap.add_argument("--rung", default=None, help="rung id from gym/ladder/rungs.yaml")
    ap.add_argument("--judge-base-url", default=os.environ.get("LADDER_JUDGE_BASE_URL"))
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--no-judge", action="store_true",
                    help="run without a judge: judged tasks are recorded UNSCORED")
    ap.add_argument("--classes", default=None, help=f"comma list from {','.join(CLASS_IDS)}")
    ap.add_argument("--tasks", default=None, help="comma list of task ids")
    ap.add_argument("--smoke", action="store_true", help="only tasks tagged smoke: true")
    ap.add_argument("--live-web", action="store_true",
                    help="include live-web tasks (reach public hosts; not reproducible)")
    ap.add_argument("--runs-per-task", type=int, default=1)
    ap.add_argument("--run-label", default=None)
    ap.add_argument("--telemetry-db", default=DEFAULT_TELEMETRY_DB)
    ap.add_argument("--workdir", default=None, help="sandbox root (default: the suite's)")
    ap.add_argument("--report", default=None, help="markdown path (default gym/results/ladder/<label>.md)")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--compare", default=None,
                    help="comma list of run labels: write the cross-model ladder table "
                         "(accuracy next to format-miss rate) and exit")
    ap.add_argument("--force-adapter-tier", default=None, choices=["off", "light", "full"],
                    help="tier-sweep arm: run the rung's model with the daemon's adapter for "
                         "THIS tier (needs --rung; rows are not filed under the rung)")
    ap.add_argument("--tier-report", default=None,
                    help="comma list of tier-sweep run labels: write the per-tier report and exit")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    suite = load_suite(args.suite)
    db = Path(os.path.expanduser(args.telemetry_db))

    if args.tier_report:
        labels = _csv(args.tier_report) or []
        conn = sqlite3.connect(str(db))
        swept = [r for label in labels for r in load_rows(conn, label)]
        conn.close()
        unforced = sorted({r["run_label"] for r in swept if not forced_tier(r)})
        if not swept or unforced:
            print(f"❌ {'no rows for ' + str(labels) if not swept else 'not tier-sweep labels: ' + str(unforced)}"
                  f" in {db}; no report written")
            return 1
        out = (Path(args.report) if args.report
               else REPO / "gym" / "results" / "ladder" / f"tier-sweep-{labels[0]}.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_tier_sweep(swept, class_order=list(CLASS_IDS)))
        print(f"📄 tier-sweep report → {out}")
        return 0

    if args.compare:
        labels = _csv(args.compare) or []
        conn = sqlite3.connect(str(db))
        by_label = {label: load_rows(conn, label) for label in labels}
        conn.close()
        empty = [label for label, rows in by_label.items() if not rows]
        if empty:
            print(f"❌ no rows for {empty} in {db}; no table written")
            return 1
        forced = [label for label, rows in by_label.items() if any(forced_tier(r) for r in rows)]
        if forced:
            print(f"❌ {forced} are tier-sweep runs (adapter tier forced) — they never go in the "
                  "rung table; use --tier-report")
            return 1
        out = (Path(args.report) if args.report
               else REPO / "gym" / "results" / "ladder" / f"ladder-{'-vs-'.join(labels)}.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_ladder_table(by_label, class_order=list(CLASS_IDS)))
        print(f"📄 ladder table → {out}")
        return 0

    if not args.report_only:
        rung_spec, judge_model = None, args.judge_model
        if args.force_adapter_tier and not args.rung:
            print("❌ --force-adapter-tier needs --rung: a tier sweep is a sweep OF a rung, and the "
                  "rung's checks prove the endpoint serves that rung's model")
            return 2
        if args.rung:
            rungs = yaml.safe_load(RUNGS.read_text())
            rung_spec = next((r for r in rungs["rungs"] if r["id"] == args.rung), None)
            if rung_spec is None:
                print(f"❌ unknown rung {args.rung!r}; known: {[r['id'] for r in rungs['rungs']]}")
                return 2
            pinned = rungs["judge"]["model"]
            if judge_model and judge_model != pinned:
                print(f"❌ rung runs use the suite's pinned judge {pinned!r}, not {judge_model!r} — "
                      "rungs graded by different judges are not comparable")
                return 2
            judge_model = pinned
            if args.quant and args.quant.upper() != str(rung_spec["quantization"]).upper():
                print(f"❌ rung {args.rung!r} is pinned to {rung_spec['quantization']}; "
                      f"--quant {args.quant} would file rows under the wrong rung")
                return 2
        if not args.base_url:
            print("❌ --base-url (or LADDER_BASE_URL) is required")
            return 2
        judge_pin = None
        if not args.no_judge:
            if not (judge_model and args.judge_base_url):
                print("❌ a pinned judge is required: --judge-model and --judge-base-url "
                      "(or LADDER_JUDGE_BASE_URL). Pass --no-judge to record judged tasks unscored.")
                return 2
            judge_pin = JudgePin(base_url=args.judge_base_url, model=judge_model)

        wanted_classes = _csv(args.classes) or []
        unknown = [c for c in wanted_classes if c not in suite.classes]
        if unknown:
            print(f"❌ unknown class(es) {unknown}; known: {list(suite.classes)}")
            return 2
        deferred = [c for c in wanted_classes if c in suite.deferred_classes]
        if deferred:
            print(f"❌ class(es) {deferred} are defined but deferred — they have no tasks yet")
            return 2
        known_ids = {t.id for t in suite.tasks}
        unknown_ids = [t for t in (_csv(args.tasks) or []) if t not in known_ids]
        if unknown_ids:
            print(f"❌ unknown task id(s) {unknown_ids}")
            return 2
        tasks = select_tasks(
            suite, classes=_csv(args.classes), task_ids=_csv(args.tasks),
            smoke=args.smoke, include_live_web=args.live_web,
        )
        contestant = Contestant(
            provider=args.provider, base_url=args.base_url, model=args.model,
            quantization=args.quant or (rung_spec or {}).get("quantization", ""),
        )
        label = args.run_label or _label(args.model or (rung_spec or {}).get("id", "model"))
        args.run_label = label
        if db.exists():
            conn = sqlite3.connect(str(db))
            try:
                existing = len(load_rows(conn, label))
            except sqlite3.OperationalError:
                existing = 0
            conn.close()
            if existing:
                print(f"❌ run label {label!r} already has {existing} row(s) in {db} — a report "
                      "would mix the two runs. Pick a new --run-label.")
                return 2
        print(f"🪜 ladder {suite.name} (sha {suite.sha256[:12]}) — run label {label}")
        aborted = None
        try:
            asyncio.run(run_ladder(
                suite, tasks, contestant, run_label=label, judge_pin=judge_pin,
                telemetry_db=db, runs_per_task=args.runs_per_task, workdir=args.workdir,
                rung=args.rung, expect_model_match=(rung_spec or {}).get("match"),
                expect_adapter_tier=(rung_spec or {}).get("adapter_tier"),
                strict_quant=rung_spec is not None,
                force_adapter_tier=args.force_adapter_tier,
            ))
        except LadderPreflightError as exc:
            print(f"\n❌ refused: {exc}")
            return 2
        except LadderAbort as exc:
            # The rows recorded so far stand; the report below says what ran.
            aborted = str(exc)
            print(f"\n❌ run aborted: {exc}")

    if not args.run_label:
        print("❌ --report-only needs --run-label")
        return 2
    if args.report_only:
        aborted = None
    rows = []
    if db.exists():
        conn = sqlite3.connect(str(db))
        try:
            rows = load_rows(conn, args.run_label)
        except sqlite3.OperationalError:
            rows = []
        conn.close()
    if not rows:
        # Never overwrite a report with nothing — a wrong --telemetry-db
        # would otherwise replace a committed report with "No rows recorded".
        print(f"❌ no rows for run label {args.run_label!r} in {db}; no report written")
        return 1
    report = render_report(rows, title=f"Model ladder — {args.run_label}",
                           class_order=list(CLASS_IDS))
    out = Path(args.report) if args.report else REPO / "gym" / "results" / "ladder" / f"{args.run_label}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"\n📄 report → {out}")
    empty = check_empty_fields(rows)
    if empty:
        print(f"❌ empty-field check FAILED — empty in every row: {', '.join(empty)}")
        return 1
    print(f"✅ empty-field check passed over {len(rows)} row(s)")
    return 3 if not args.report_only and aborted else 0


if __name__ == "__main__":
    sys.exit(main())
