#!/usr/bin/env python3
"""Flag banked training pairs whose ``chosen`` side carries template markup.

FLAG, DO NOT DELETE. These rows are the only surviving evidence of a live
upstream leak — a model emitting chat-template control tokens into a string
argument, which then passed structure, syntax and type validation and executed.
Deleting them destroys the sample; flagging them keeps the evidence while
taking them out of the training path (``export_training_pairs.py`` skips any
row with ``meta.quality`` set).

Sets ``meta.quality = "template_markup"`` and leaves everything else intact.
Idempotent: re-running touches nothing already flagged.

Dry-run by default — pass --apply to write.

    uv run python scripts/flag_corrupt_pairs.py                    # all known DBs
    uv run python scripts/flag_corrupt_pairs.py --db path/to.db --apply
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prometheus.adapter import markup_guard  # noqa: E402
from prometheus.learning.pair_capture import (  # noqa: E402
    QUALITY_KEY,
    QUALITY_TEMPLATE_MARKUP,
)

DEFAULT_DBS = [
    "~/.prometheus/data/training.db",
    "~/.prometheus/data/gym-training.db",
    "~/.prometheus/data/gym-training-verify.db",
    "~/.prometheus/data/dictwrap-train.db",
]


def scan_db(path: Path, apply: bool) -> tuple[int, int, int]:
    """Return (scanned, corrupt, newly_flagged)."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, pair_source, tool_name, chosen, meta FROM training_pairs"
        ).fetchall()
    except sqlite3.OperationalError:
        print(f"  {path}: no training_pairs table — skipped")
        return 0, 0, 0

    corrupt = flagged = 0
    for r in rows:
        try:
            chosen = json.loads(r["chosen"] or "{}")
        except ValueError:
            continue
        found = markup_guard.scan_arguments(chosen.get("input"))
        if not found:
            continue
        corrupt += 1
        try:
            meta = json.loads(r["meta"] or "{}")
        except ValueError:
            meta = {}
        if meta.get(QUALITY_KEY) == QUALITY_TEMPLATE_MARKUP:
            continue  # already flagged — idempotent
        print(f"    {r['pair_source']}/{r['tool_name']}: "
              f"{markup_guard.describe(found)}")
        if apply:
            meta[QUALITY_KEY] = QUALITY_TEMPLATE_MARKUP
            meta["quality_detail"] = markup_guard.describe(found)
            conn.execute(
                "UPDATE training_pairs SET meta = ? WHERE id = ?",
                (json.dumps(meta, default=str), r["id"]),
            )
        flagged += 1
    if apply:
        conn.commit()
    conn.close()
    return len(rows), corrupt, flagged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", action="append", default=None,
                    help="training db path (repeatable); defaults to the known set")
    ap.add_argument("--apply", action="store_true",
                    help="write the flags (default is a dry run)")
    args = ap.parse_args()

    dbs = [Path(d).expanduser() for d in (args.db or DEFAULT_DBS)]
    total_scanned = total_corrupt = total_flagged = 0
    for db in dbs:
        if not db.exists():
            print(f"  {db}: absent — skipped")
            continue
        print(f"  {db}:")
        scanned, corrupt, flagged = scan_db(db, args.apply)
        total_scanned += scanned
        total_corrupt += corrupt
        total_flagged += flagged

    verb = "flagged" if args.apply else "would flag"
    print(f"\n  {total_scanned} pairs scanned / {total_corrupt} corrupt / "
          f"{total_flagged} {verb}")
    if not args.apply and total_flagged:
        print("  dry run — re-run with --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
