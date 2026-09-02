#!/usr/bin/env python3
"""Scrub secrets already sitting in the capture stores.

Capture-time redaction (security/log_redaction.redact_capture, wired into
telemetry.record / record_silent_failure, PairStore.add_pair and the
golden-trace export) keeps NEW rows clean. This is the one-off for the rows
and files written before it existed — the 2026-08-31 finding: a Telegram bot
token in telemetry.db, training.db and a trajectories/ export.

    python3 scripts/scrub_capture_stores.py            # DRY RUN: counts only, touches nothing
    python3 scripts/scrub_capture_stores.py --apply    # rewrite in place, after a backup

Dry run is the default and prints, per store and column, how many rows/lines
carry a redactable shape. --apply first copies each SQLite database with the
sqlite backup API (WAL-safe — a plain `cp` of a WAL database copies only
checkpointed pages) to <db>.pre-scrub-<stamp>, then UPDATEs the redacted
columns row by row; JSONL files are rewritten line by line via a temp file +
replace. Nothing is deleted. Exit 0 on success; non-zero if a store could not
be opened (said loudly, never skipped silently).

Never prints a matched value — only counts.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# Run from a checkout or an install: the package is what matters.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prometheus.security.log_redaction import redact_capture, redact_secrets  # noqa: E402

HOME = Path(os.environ.get("PROMETHEUS_HOME", "~")).expanduser()
DEFAULT_TELEMETRY = HOME / ".prometheus" / "telemetry.db"
DEFAULT_TRAINING = HOME / ".prometheus" / "data" / "training.db"
DEFAULT_TRAJECTORIES = HOME / ".prometheus" / "trajectories"

# (table, key column, text columns that may carry a secret)
TELEMETRY_TARGETS = (
    ("tool_calls", "id", ("error_detail", "raw_model_output", "parsed_tool_call")),
    ("silent_failures", "id", ("exception_msg", "traceback", "context", "response_body")),
)
TRAINING_TARGETS = (
    ("training_pairs", "id", ("context", "rejected", "chosen", "meta")),
)


def _backup(db: Path, stamp: str) -> Path:
    dest = db.with_name(f"{db.name}.pre-scrub-{stamp}")
    src = sqlite3.connect(str(db))
    dst = sqlite3.connect(str(dest))
    with dst:
        src.backup(dst)  # WAL-safe copy
    src.close()
    dst.close()
    return dest


def scrub_sqlite(db: Path, targets, *, apply: bool, stamp: str) -> dict[str, int]:
    """Return {"table.column": rows_changed}. Rewrites only when *apply*."""
    counts: dict[str, int] = {}
    if not db.exists():
        print(f"  {db}: not found (skipped — nothing to scrub)")
        return counts
    if apply:
        print(f"  backup → {_backup(db, stamp).name}")
    conn = sqlite3.connect(str(db))
    try:
        existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, key, cols in targets:
            if table not in existing:
                continue
            present = [c for c in cols if c in {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}]
            for col in present:
                changed = 0
                rows = conn.execute(f"SELECT {key}, {col} FROM {table} WHERE {col} IS NOT NULL").fetchall()
                for rid, val in rows:
                    if not isinstance(val, str):
                        continue
                    new = redact_secrets(val)
                    if new != val:
                        changed += 1
                        if apply:
                            conn.execute(f"UPDATE {table} SET {col} = ? WHERE {key} = ?", (new, rid))
                counts[f"{table}.{col}"] = changed
        if apply:
            conn.commit()
    finally:
        conn.close()
    return counts


def scrub_jsonl_dir(directory: Path, *, apply: bool) -> dict[str, int]:
    """Return {filename: lines_changed}. Rewrites via temp + replace when *apply*."""
    counts: dict[str, int] = {}
    if not directory.exists():
        print(f"  {directory}: not found (skipped — nothing to scrub)")
        return counts
    for path in sorted(directory.glob("*.jsonl")):
        changed = 0
        out_lines: list[str] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.rstrip("\n")
                if not stripped:
                    out_lines.append(line)
                    continue
                try:
                    obj = json.loads(stripped)
                    new_obj = redact_capture(obj)
                    new = json.dumps(new_obj, ensure_ascii=False)
                    if new_obj != obj:
                        changed += 1
                        out_lines.append(new + "\n")
                    else:
                        out_lines.append(line)
                except json.JSONDecodeError:
                    new = redact_secrets(stripped)  # not JSON — scrub as text, keep the line
                    if new != stripped:
                        changed += 1
                    out_lines.append(new + "\n")
        counts[path.name] = changed
        if apply and changed:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text("".join(out_lines), encoding="utf-8")
            os.replace(tmp, path)
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="rewrite in place (default: dry run, counts only)")
    ap.add_argument("--telemetry", type=Path, default=DEFAULT_TELEMETRY)
    ap.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    ap.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES)
    args = ap.parse_args(argv)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"=== capture-store scrub — {mode} ===")
    total = 0
    failures = 0
    for label, fn in (
        ("telemetry.db", lambda: scrub_sqlite(args.telemetry, TELEMETRY_TARGETS, apply=args.apply, stamp=stamp)),
        ("training.db", lambda: scrub_sqlite(args.training, TRAINING_TARGETS, apply=args.apply, stamp=stamp)),
        ("trajectories/", lambda: scrub_jsonl_dir(args.trajectories, apply=args.apply)),
    ):
        print(f"{label}:")
        try:
            counts = fn()
        except (sqlite3.Error, OSError) as e:
            failures += 1
            print(f"  FAILED to scrub {label}: {type(e).__name__}: {e}")
            continue
        for k, v in counts.items():
            print(f"  {k:<40} {v:>8} {'rewritten' if args.apply else 'would change'}")
            total += v
    print(f"=== {total} row(s)/line(s) {'rewritten' if args.apply else 'would change'}; {failures} store(s) failed ===")
    if not args.apply and total:
        print("re-run with --apply to rewrite (a backup of each database is taken first)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
