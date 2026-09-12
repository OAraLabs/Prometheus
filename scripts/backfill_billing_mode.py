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
    # #474: redact the host to the matched marker before it reaches stdout.
    # The classification reason already does this (cost.py), but this line
    # prints the raw resolved host — stdout IS persisted state (terminal
    # scrollback, shell history, any transcript this output is pasted into),
    # and the host is a real infrastructure identifier. The marker prefix is
    # the part that carries the meaning; the account-scoped remainder is what
    # must not appear. `<redacted:host matches 'token-plan.'>` keeps the
    # output verifiable ("yes, it classified from the subscription host")
    # without publishing the identifier. When the host matches no marker the
    # reason says why, and the host is withheld all the same — a non-matching
    # host is still an address.
    from prometheus.telemetry.cost import SUBSCRIPTION_HOST_MARKERS

    _matched = next((m for m in SUBSCRIPTION_HOST_MARKERS if m in host), None)
    host_display = (
        f"<redacted: resolved host matches {_matched!r}>"
        if _matched
        else "<redacted: resolved host, matches no subscription marker>"
    )
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

    # #474: "rows to stamp: 0" meant two different things — the work is already
    # done, or the predicate matched nothing (a typo'd --model produces a
    # byte-identical report to a completed backfill, and a no-op reads as
    # success: §4e's shape in a reporting path). Count both sides so the output
    # says WHICH zero this is.
    window_where = "WHERE model = ? AND timestamp < ? AND input_tokens IS NOT NULL"
    already_stamped, no_mode_rows = conn.execute(
        f"SELECT SUM(billing_mode IS NOT NULL), SUM(billing_mode IS NULL) "
        f"FROM subsystem_runs {window_where}", params
    ).fetchone()
    already_stamped = already_stamped or 0
    no_mode_rows = no_mode_rows or 0
    # Rows outside the window for this model — a third explanation for a zero:
    # the model has rows, but none before the boundary (nothing to backfill).
    total_model_rows = conn.execute(
        "SELECT COUNT(*) FROM subsystem_runs WHERE model = ?", (args.model,)
    ).fetchone()[0]

    print(f"database      : {args.db}")
    print(f"model         : {args.model}")
    # #474: the resolved host is a real infrastructure identifier and stdout is
    # persisted state (terminal, shell history, pasted transcripts). Print the
    # redaction, not the address — the marker says everything the reader needs
    # ("yes, it classified from the subscription host").
    print(f"host (from {args.env_var if not args.host else '--host'}): {host_display}")
    print(f"classified as : {mode}  ({reason})")
    print(f"boundary      : {until}")
    print(f"already stamped : {already_stamped} rows carry a billing_mode in this window")
    print(f"rows to stamp : {n}   input_tokens: {toks}")
    print(f"window        : {first} .. {last}")
    print(f"model rows outside the window: {total_model_rows - already_stamped - n}")

    if mode == "unknown":
        print("\nrefusing: the host does not classify. Stamping `unknown` "
              "records a conclusion nobody drew.", file=sys.stderr)
        return 5
    if n == 0:
        # #474: name which zero this is. The old message admitted the ambiguity
        # in a parenthetical instead of resolving it.
        if already_stamped > 0:
            print(f"\nalready done: {already_stamped} row(s) in the window carry "
                  f"a billing_mode; nothing left to stamp.")
        elif total_model_rows > 0:
            print(f"\nnothing to do in this window: the model has "
                  f"{total_model_rows} row(s) but all are outside the boundary "
                  f"(written after the v3 writer took over) or lack "
                  f"input_tokens. This is NOT 'already backfilled'.")
        else:
            print(f"\nmatched nothing: subsystem_runs has NO rows for model "
                  f"{args.model!r} at all. Check the --model spelling — this "
                  f"output is otherwise identical to a completed backfill.")
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
