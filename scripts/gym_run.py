#!/usr/bin/env python3
"""Run a gym experiment from a manifest.

Usage:
    uv run python scripts/gym_run.py --manifest gym/experiments/s1-exp0-baseline.yaml
    uv run python scripts/gym_run.py --manifest <m> --baseline exp0-baseline   # comparison report
    uv run python scripts/gym_run.py --report-only --manifest <m> --baseline exp0-baseline

Writes run rows to ~/.prometheus/data/gym.db and a markdown report to
gym/results/<series>/<experiment>.md (commit the results — the spec treats
them as series artifacts).
"""

import argparse
import asyncio
import sys
from pathlib import Path

from prometheus.__main__ import load_config
from prometheus.gym.manifest import load_manifest
from prometheus.gym.report import render_report
from prometheus.gym.runner import GYM_DB, run_experiment
from prometheus.gym.store import GymStore
from prometheus.gym.tasks import load_taskset


async def main() -> int:
    parser = argparse.ArgumentParser(description="Tool-calling gym runner")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--baseline",
        default=None,
        help="experiment name (same series) to compare against in the report",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="skip running; regenerate the report from stored rows",
    )
    parser.add_argument(
        "--harvest",
        action="store_true",
        help="capture repair-pairs induced by this run into the gym harvest DB "
             "(closeout follow-up #6 — the 1,000-pair path). Off by default so "
             "ordinary gym runs don't write training pairs.",
    )
    parser.add_argument(
        "--harvest-db",
        default=None,
        help="override the harvest DB path (default ~/.prometheus/data/gym-training.db)",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    taskset = load_taskset(manifest.taskset)
    store = GymStore(GYM_DB)

    print(f"🏋  Gym — series {manifest.series}, experiment {manifest.experiment}")
    print(f"    taskset {taskset.path} (sha {taskset.sha256[:12]}, {len(taskset.tasks)} tasks)")
    print(f"    variable: {manifest.variable_name} {manifest.variable_payload or ''}")
    print(f"    runs/task: {manifest.runs_per_task}")
    print("=" * 60)

    harvest_start = 0
    harvest_db = None
    if args.harvest:
        from prometheus.gym.harvest import configure_harvest, pair_total

        harvest_db = configure_harvest(args.harvest_db)
        harvest_start = pair_total()
        print(f"🌱 harvest ON — capturing induced repair-pairs → {harvest_db}")
        print(f"    starting pairs: {harvest_start}")

    if not args.report_only:
        existing = store.runs(manifest.series, manifest.experiment)
        if existing:
            print(
                f"⚠️  {len(existing)} stored runs already exist for this "
                f"(series, experiment) — they will be overwritten per "
                f"(task, run_idx) key."
            )
        config = load_config()
        totals = await run_experiment(manifest, taskset, config, store=store)
        print("=" * 60)
        print(f"Done: {totals['passed']}/{totals['runs']} runs passed")

    if args.harvest:
        from prometheus.gym.harvest import pair_total

        gained = pair_total() - harvest_start
        print(f"🌱 harvested {gained} new repair-pair(s) → {harvest_db}")

    rows = store.runs(manifest.series, manifest.experiment)
    if not rows:
        print("No stored rows — nothing to report.")
        return 1
    baseline_rows = (
        store.runs(manifest.series, args.baseline) if args.baseline else None
    )
    if args.baseline and not baseline_rows:
        print(f"⚠️  baseline {args.baseline!r} has no stored rows; raw report only")
        baseline_rows = None

    report = render_report(
        manifest.series,
        manifest.experiment,
        rows,
        baseline_rows,
        manifest_notes=manifest.notes,
    )
    out_dir = Path("gym/results") / manifest.series
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest.experiment}.md"
    out_path.write_text(report)
    print(f"\n📄 report → {out_path}\n")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
