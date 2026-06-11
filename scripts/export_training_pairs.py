#!/usr/bin/env python3
"""Export training pairs to standard DPO JSONL.

One line per pair:
  {"prompt": <context as compact string>, "chosen": <chosen call JSON>,
   "rejected": <rejected call JSON>, "meta": {source, tool, model, ts}}

cloud_golden rows (chosen-only, rejected NULL) are excluded from DPO export
by default — they pair later against local attempts; --include-golden emits
them with "rejected": null for KTO-style consumers.

Dedupe happens at INSERT time (UNIQUE context_hash); the export re-checks
(context, rejected) anyway so merged databases stay clean.

Usage:
    uv run python scripts/export_training_pairs.py --out pairs.jsonl
    uv run python scripts/export_training_pairs.py --since 2026-06-01 --out pairs.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

from prometheus.learning.pair_capture import PairStore


def _parse_since(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)  # unix timestamp
    except ValueError:
        return dt.datetime.fromisoformat(value).timestamp()


def main() -> int:
    ap = argparse.ArgumentParser(description="Export training pairs as DPO JSONL")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument("--since", default=None, help="ISO date or unix ts filter")
    ap.add_argument("--include-golden", action="store_true",
                    help="emit cloud_golden chosen-only rows (rejected: null)")
    ap.add_argument("--db", default=None, help="training db path override")
    args = ap.parse_args()

    store = PairStore(args.db) if args.db else PairStore()
    rows = store.rows_since(_parse_since(args.since))

    out = Path(args.out)
    seen: set[str] = set()
    written = skipped_golden = deduped = 0
    with out.open("w") as fh:
        for r in rows:
            if r["rejected"] is None and not args.include_golden:
                skipped_golden += 1
                continue
            key = hashlib.sha256(
                ((r["context"] or "") + "\x1e" + (r["rejected"] or "")).encode()
            ).hexdigest()
            if key in seen:
                deduped += 1
                continue
            seen.add(key)
            fh.write(json.dumps({
                "prompt": r["context"] or "",
                "chosen": r["chosen"],
                "rejected": r["rejected"],
                "meta": {
                    "source": r["pair_source"],
                    "tool": r["tool_name"],
                    "model": r["model_id"],
                    "ts": r["timestamp"],
                },
            }) + "\n")
            written += 1

    print(f"Wrote {written} pairs → {out}")
    if skipped_golden:
        print(f"  ({skipped_golden} cloud_golden chosen-only rows skipped; "
              f"--include-golden to emit)")
    if deduped:
        print(f"  ({deduped} duplicate (context, rejected) rows deduped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
