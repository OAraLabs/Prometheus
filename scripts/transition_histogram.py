#!/usr/bin/env python3
"""Bucket a training-pair DB by transition type and print the histogram.

    uv run python scripts/transition_histogram.py --db ~/.prometheus/data/gym-training.db

Reads each pair's stored rejected/chosen JSON, classifies the transition
(prometheus.learning.transition_taxonomy), and prints counts + share. Breadth
is the metric the harvest-breadth sprint optimizes: pairs-per-transition-type,
not pairs/run.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from prometheus.learning.transition_taxonomy import TRANSITION_TYPES, classify_transition


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    conn = sqlite3.connect(str(Path(args.db).expanduser()))
    rows = conn.execute("SELECT pair_source, rejected, chosen FROM training_pairs").fetchall()
    counts = {t: 0 for t in TRANSITION_TYPES}
    for src, rej_s, cho_s in rows:
        try:
            rej = json.loads(rej_s) if rej_s else None
            cho = json.loads(cho_s) if cho_s else None
        except (ValueError, TypeError):
            rej, cho = None, None
        counts[classify_transition(src, rej, cho)] += 1

    total = sum(counts.values())
    if args.json:
        print(json.dumps({"total": total, "by_transition": counts}, indent=2))
        return 0
    print(f"transition histogram — {total} pairs in {args.db}")
    for t in TRANSITION_TYPES:
        n = counts[t]
        share = (100 * n / total) if total else 0
        bar = "█" * round(share / 2)
        print(f"  {t:24s} {n:4d}  {share:5.1f}%  {bar}")
    distinct = sum(1 for t in TRANSITION_TYPES if counts[t] > 0)
    print(f"  → {distinct}/{len(TRANSITION_TYPES)} transition types present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
