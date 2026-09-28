"""``oara scrub`` — redact secrets already kept in the capture stores.

DRY RUN BY DEFAULT: counts per store and column, every database opened
read-only. ``--apply`` rewrites in place after a WAL-safe backup of each
database. Stop the daemon first. The work is
:mod:`prometheus.security.scrub_capture_stores`; this only wires it to the CLI,
so a pip or Homebrew install can run it (scripts/ ships in neither).
"""

from __future__ import annotations

import argparse


def add_scrub_subparser(subparsers: argparse._SubParsersAction) -> None:
    from prometheus.security.scrub_capture_stores import DESCRIPTION, add_arguments

    p = subparsers.add_parser(
        "scrub",
        help="Redact secrets already kept in telemetry, training, trajectory, "
             "LCM and memory stores (dry run unless --apply)",
        description=DESCRIPTION + " Dry run unless --apply; stop the daemon before --apply.",
    )
    add_arguments(p)


def run_scrub(args: argparse.Namespace) -> int:
    from prometheus.security.scrub_capture_stores import run

    return run(args)
