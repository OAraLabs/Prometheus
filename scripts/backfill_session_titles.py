#!/usr/bin/env python3
"""One-off title backfill for sessions predating GRAFT-MOBILE-BRIDGE 7.

Piece 7's generation is absence-only and fires after a completed turn, so
sessions from before it stay untitled unless they happen to be used again —
clients fall back to a first-user-message snippet. This walks the conversation
store once and titles what qualifies, then it is done. A script run
deliberately, not a route and not a startup hook.

    python3 scripts/backfill_session_titles.py --dry-run          # read first
    python3 scripts/backfill_session_titles.py                    # then write

Safe to run against a live daemon: writes are single-row commits to the shared
lcm.db, calls are sequential (the model is shared with live turns — --delay
paces them further apart), already-titled sessions are skipped so a rerun or a
resume after an interrupt is a no-op, and a manual rename is never overwritten
(the write path is maybe_title_session, absence-only). A dry run generates the
titles for real but stores nothing — dry-run-then-real pays the model twice,
by design: "what it would be titled" cannot be known without asking.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Editable/source-checkout path, same shim as scripts/daemon.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None,
                        help="path to prometheus.yaml (default: repo config)")
    parser.add_argument("--dry-run", action="store_true",
                        help="generate and print titles without storing them")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap generation attempts this run (skips are free)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="seconds between model calls (default 1.0)")
    args = parser.parse_args()

    from prometheus.__main__ import create_provider, load_config
    from prometheus.engine.session_titles import backfill_titles
    from prometheus.memory.lcm_conversation_store import LCMConversationStore

    config = load_config(args.config)
    provider, model = create_provider(config.get("model", {}) or {})
    store = LCMConversationStore()  # honors PROMETHEUS_DATA_DIR, same as the daemon

    counts = asyncio.run(backfill_titles(
        store, provider, model,
        dry_run=args.dry_run, limit=args.limit, delay_seconds=args.delay,
    ))

    mode = "dry run" if args.dry_run else "run"
    print(f"\n{mode} complete: {counts}")
    remaining = counts["eligible"] - counts["would_title" if args.dry_run else "titled"] \
        - counts["failed"]
    if args.limit is not None and remaining > 0:
        print(f"{remaining} eligible session(s) untouched (--limit) — rerun to continue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
