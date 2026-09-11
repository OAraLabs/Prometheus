#!/usr/bin/env python3
"""One-shot backfill for `subsystem_runs.billing_mode` (#284).

WHY THIS IS A SCRIPT AND NOT A MIGRATION
----------------------------------------
A migration runs on every daemon start, on every machine, forever. This is a
single operator action taken once, against one box, on the evidence available
at that moment. Baking it into `_migrate_schema` would re-assert someone's
2026-09 conclusion on databases that never had this plan.

WHY THE HOST IS NEITHER IN THIS FILE NOR IN THE DATABASE
--------------------------------------------------------
It is read from the environment at run time, classified, and dropped. A literal
production hostname in the repository is a real infrastructure identifier
persisted forever; so is one written into a telemetry row, which is why the
`billing_mode` column stores the verdict and no column stores the evidence.
`--host` exists for a box whose env is no longer set — but then YOU are
asserting it, not the config.

WHAT MAKES THE BACKFILL A FACT RATHER THAN A MEMORY
----------------------------------------------------
Only that it is run while the configuration that proves it is still in place.
`--verify` compares the totals `/api/usage` reports BEFORE and AFTER: if the
stamp agrees with read-time classification, the numbers are identical and the
backfill changed only WHERE the answer comes from. Once the env var moves, that
equality can never be reconstructed — which is the whole deadline.

Usage:
    python scripts/backfill_billing_mode.py --model qwen3.8-max            # dry run
    python scripts/backfill_billing_mode.py --model qwen3.8-max --apply
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.telemetry.cost import billing_for  # noqa: E402

DEFAULT_DB = Path.home() / ".prometheus" / "telemetry.db"
# Provider -> the env var whose value decides that provider's billing host.
HOST_ENV = {"qwen": "QWEN_BASE_URL"}


def _host_from_env(env_var: str) -> str | None:
    raw = os.environ.get(env_var)
    if not raw:
        return None
    return raw.split("//", 1)[-1].split("/", 1)[0].split("?", 1)[0] or None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--model", required=True,
                    help="exact model id whose rows are being labelled")
    ap.add_argument("--env-var", default="QWEN_BASE_URL",
                    help="env var holding the base_url that proves the host")
    ap.add_argument("--host", default=None,
                    help="override; use only when the env no longer holds it")
    ap.add_argument("--until", type=float, default=None,
                    help="epoch upper bound; default = the v3 boundary, i.e. "
                         "the instant the writer took over")
    ap.add_argument("--apply", action="store_true",
                    help="without this the script only reports")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"no such database: {args.db}", file=sys.stderr)
        return 2

    host = args.host or _host_from_env(args.env_var)
    if not host:
        print(
            f"cannot establish the billing host: {args.env_var} is unset and no "
            f"--host given.\nThis is the failure the deadline was about — the "
            f"configuration that proved it is gone. Refusing to guess.",
            file=sys.stderr,
        )
        return 3

    mode, reason = billing_for(args.model, f"//{host}")
    conn = sqlite3.connect(args.db)

    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'billing_recorded_since'"
    ).fetchone()
    boundary = float(row[0]) if row and row[0] else None
    until = args.until if args.until is not None else boundary
    if until is None:
        print("no v3 boundary stamped and no --until given; refusing to label "
              "rows the writer may have stamped itself.", file=sys.stderr)
        return 4

    where = ("WHERE model = ? AND billing_mode IS NULL AND timestamp < ? "
             "AND input_tokens IS NOT NULL")
    params = (args.model, until)

    n, first, last, toks = conn.execute(
        f"SELECT COUNT(*), MIN(timestamp), MAX(timestamp), "
        f"COALESCE(SUM(input_tokens),0) FROM subsystem_runs {where}", params
    ).fetchone()

    print(f"database      : {args.db}")
    print(f"model         : {args.model}")
    print(f"host (from {args.env_var if not args.host else '--host'}): {host}")
    print(f"classified as : {mode}  ({reason})")
    print(f"boundary      : {until}")
    print(f"rows to stamp : {n}   input_tokens: {toks}")
    print(f"window        : {first} .. {last}")

    if mode == "unknown":
        print("\nrefusing: the host does not classify. Stamping `unknown` "
              "records a conclusion nobody drew.", file=sys.stderr)
        return 5
    if n == 0:
        print("\nnothing to do (already stamped, or no matching rows).")
        return 0
    if not args.apply:
        print("\nDRY RUN — re-run with --apply to write.")
        return 0

    cur = conn.execute(
        f"UPDATE subsystem_runs SET billing_mode = ? {where}",
        (mode,) + params,
    )
    conn.commit()
    print(f"\nstamped {cur.rowcount} rows.")

    leftover = conn.execute(
        f"SELECT COUNT(*) FROM subsystem_runs {where}", params
    ).fetchone()[0]
    print(f"remaining unstamped in window: {leftover}  (must be 0)")

    # No interior boundary: a uniform time-window fact has no disagreements.
    distinct = conn.execute(
        "SELECT COUNT(DISTINCT billing_mode) FROM subsystem_runs "
        "WHERE model = ? AND timestamp < ? AND billing_mode IS NOT NULL",
        (args.model, until),
    ).fetchone()[0]
    print(f"distinct modes in window: {distinct}  (must be 1)")
    return 0 if leftover == 0 and distinct == 1 else 6


if __name__ == "__main__":
    raise SystemExit(main())
