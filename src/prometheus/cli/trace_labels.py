"""``oara trace-labels`` — label stored traces that predate the trace labels.

DRY RUN BY DEFAULT: counts what it would write, every database opened
read-only. ``--apply`` takes a WAL-safe backup of each database, then fills
``tool_ok`` and ``turn_outcome`` where they are NULL (telemetry/trace_labels.py).
It never changes a label that is already set and never touches another
column, so running it twice, or while the daemon runs, is safe.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

DESCRIPTION = (
    "Fill the trace labels (tool_ok, turn_outcome) on telemetry tool_calls and "
    "training_pairs rows written before they existed. Never rewrites a set label."
)


def add_trace_labels_subparser(subparsers: argparse._SubParsersAction) -> None:
    from prometheus.security.scrub_capture_stores import DEFAULT_TELEMETRY, DEFAULT_TRAINING

    p = subparsers.add_parser(
        "trace-labels",
        help="Backfill tool_ok / turn_outcome on stored traces (dry run unless --apply)",
        description=DESCRIPTION + " Dry run unless --apply.",
    )
    p.add_argument("--apply", action="store_true",
                   help="write the labels, after a backup (default: dry run, counts only)")
    p.add_argument("--telemetry", type=Path, default=DEFAULT_TELEMETRY)
    p.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    p.add_argument("--json", action="store_true", help="print the counts as JSON")


def _open(path: Path, *, apply: bool) -> sqlite3.Connection:
    if apply:
        conn = sqlite3.connect(str(path), timeout=30)
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def run_trace_labels(args: argparse.Namespace) -> int:
    """Exit 0, or 1 if a store could not be read or written."""
    from prometheus.learning.pair_capture import PairStore
    from prometheus.security.scrub_capture_stores import _backup
    from prometheus.telemetry import trace_labels
    from prometheus.telemetry.tracker import ToolCallTelemetry

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    now = time.time()
    reports: list[dict] = []
    failures = 0
    telemetry_path: Path = args.telemetry.expanduser()
    training_path: Path = args.training.expanduser()
    if args.apply:
        # Opening each store with this build adds the label columns (the same
        # additive migration the daemon runs at start).
        for path, opener in ((telemetry_path, ToolCallTelemetry), (training_path, PairStore)):
            if path.exists():
                opener(path).close()

    tel_conn: sqlite3.Connection | None = None
    if telemetry_path.exists():
        try:
            if args.apply:
                print(f"backup: {_backup(telemetry_path, stamp, 'pre-trace-labels')}")
            tel_conn = _open(telemetry_path, apply=args.apply)
            reports.append(trace_labels.backfill_telemetry(
                tel_conn, apply=args.apply, now=now).as_dict())
        except (sqlite3.Error, OSError, RuntimeError) as exc:
            failures += 1
            print(f"telemetry.db: FAILED: {type(exc).__name__}: {exc}")
    else:
        print(f"telemetry.db: not found at {telemetry_path} — training pairs will be 'unknown'")

    if training_path.exists():
        try:
            if args.apply:
                print(f"backup: {_backup(training_path, stamp, 'pre-trace-labels')}")
            train_conn = _open(training_path, apply=args.apply)
            try:
                reports.append(trace_labels.backfill_training(
                    train_conn, tel_conn, apply=args.apply, now=now).as_dict())
            finally:
                train_conn.close()
        except (sqlite3.Error, OSError, RuntimeError) as exc:
            failures += 1
            print(f"training.db: FAILED: {type(exc).__name__}: {exc}")
    if tel_conn is not None:
        tel_conn.close()

    if args.json:
        print(json.dumps({"apply": bool(args.apply), "stores": reports}, indent=2))
    else:
        verb = "labelled" if args.apply else "would label"
        print(f"=== trace labels — {'APPLY' if args.apply else 'DRY RUN'} ===")
        for r in reports:
            print(f"{r['store']}: {r['rows']} row(s) without labels")
            print(f"  tool_ok {verb}: {r['tool_ok_set']}")
            for k, v in r["turn_outcome"].items():
                print(f"  turn_outcome={k:<10} {verb}: {v}")
            if r["still_running"]:
                print(f"  turn still running (left NULL): {r['still_running']}")
        if not args.apply:
            print("re-run with --apply to write them (a backup of each database is taken first)")
    return 1 if failures else 0
