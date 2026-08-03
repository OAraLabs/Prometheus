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

from prometheus.adapter import markup_guard
from prometheus.learning.pair_capture import QUALITY_KEY, PairStore


def _parse_since(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)  # unix timestamp
    except ValueError:
        return dt.datetime.fromisoformat(value).timestamp()


def _is_unfit(row) -> bool:
    """True when this row's `chosen` side is not fit to train on.

    Checks the persisted `meta.quality` flag AND re-scans `chosen` for template
    markup, so an unflagged corrupt row cannot leak into a fine-tune just
    because the flagging script was never run against that DB.
    """
    try:
        meta = json.loads(row["meta"] or "{}")
    except ValueError:
        meta = {}
    if meta.get(QUALITY_KEY):
        return True
    try:
        chosen = json.loads(row["chosen"] or "{}")
    except ValueError:
        return False
    return bool(markup_guard.scan_arguments(chosen.get("input")))


def main() -> int:
    ap = argparse.ArgumentParser(description="Export training pairs as DPO JSONL")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument("--since", default=None, help="ISO date or unix ts filter")
    ap.add_argument("--include-golden", action="store_true",
                    help="emit cloud_golden chosen-only rows (rejected: null)")
    ap.add_argument("--include-flagged", action="store_true",
                    help="emit rows flagged unfit (meta.quality) — for inspection, "
                         "never for training")
    ap.add_argument("--db", default=None, help="training db path override")
    args = ap.parse_args()

    store = PairStore(args.db) if args.db else PairStore()
    rows = store.rows_since(_parse_since(args.since))

    out = Path(args.out)
    seen: set[str] = set()
    written = skipped_golden = deduped = skipped_quality = 0
    with out.open("w") as fh:
        for r in rows:
            if r["rejected"] is None and not args.include_golden:
                skipped_golden += 1
                continue
            # Quality gate. Two conditions, deliberately: the stored flag (set
            # by scripts/flag_corrupt_pairs.py) AND a live re-scan of `chosen`.
            # The flag alone would be a check someone has to remember to run;
            # re-scanning makes "no corrupt positive label reaches training" a
            # property of the export instead. Belt and braces on the one path
            # that feeds a fine-tune.
            if not args.include_flagged and _is_unfit(r):
                skipped_quality += 1
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
    if skipped_quality:
        print(f"  ({skipped_quality} rows skipped as unfit to train on — "
              f"chat-template markup in `chosen`; --include-flagged to inspect)")
    if deduped:
        print(f"  ({deduped} duplicate (context, rejected) rows deduped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
