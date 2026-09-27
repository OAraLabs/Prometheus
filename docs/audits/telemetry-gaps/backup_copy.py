"""Copy a live SQLite store with the backup API, for the telemetry-gap scans.

    D=$(mktemp -d) && python3 backup_copy.py ~/.prometheus/telemetry.db "$D/telemetry.db"
    ... run the scans against "$D/telemetry.db" ...
    rm -rf "$D"

Why the backup API and not ``cp``: the live store is in WAL mode, and a plain
copy of the main file misses every page still in ``-wal`` (the torn-copy shape
``telemetry/tracker.py`` warns about). The source is opened READ-ONLY and holds
no transaction, so the daemon's writes are never blocked; the whole copy is one
backup step, taken from one read snapshot.

Refuses to spin: Python's ``Connection.backup`` retries BUSY/LOCKED forever, so
the progress callback gives up after ``--max-busy`` retries. The copy is created
0600 inside the caller's private directory, then checked with
``PRAGMA integrity_check``. Prints the copy's row counts and newest row time
(aggregates only).
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


class _TooBusy(RuntimeError):
    pass


# The step's status codes (named in the sqlite3 module from Python 3.11).
_BUSY = (getattr(sqlite3, "SQLITE_BUSY", 5), getattr(sqlite3, "SQLITE_LOCKED", 6))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--max-busy", type=int, default=50)
    args = ap.parse_args()

    if args.dest.exists():
        raise SystemExit(f"refusing to overwrite {args.dest}")
    if not args.source.is_file():
        raise SystemExit(f"missing: {args.source}")
    # The copy holds conversation-derived text: owner-only from the first byte.
    fd = os.open(args.dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)

    busy = 0

    def progress(status: int, remaining: int, total: int) -> None:
        nonlocal busy
        if status in _BUSY:
            busy += 1
            if busy > args.max_busy:
                raise _TooBusy(f"source stayed busy for {busy} retries")

    src = sqlite3.connect(f"file:{args.source}?mode=ro", uri=True)
    dst = sqlite3.connect(args.dest)
    try:
        src.backup(dst, pages=-1, progress=progress, sleep=0.1)
    except _TooBusy as exc:
        dst.close()
        args.dest.unlink(missing_ok=True)
        raise SystemExit(f"backup abandoned: {exc}") from None
    finally:
        src.close()
    ok = dst.execute("PRAGMA integrity_check").fetchone()[0]
    dst.close()
    if ok != "ok":
        args.dest.unlink(missing_ok=True)
        raise SystemExit(f"integrity_check failed on the copy: {ok[:200]}")

    con = C.open_ro(args.dest)
    print(f"copy ok ({args.dest.stat().st_size / 1e6:.1f} MB, integrity ok, busy retries {busy})")
    for (name,) in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ):
        n = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        cols = {r[1] for r in con.execute(f'PRAGMA table_info("{name}")')}
        newest = ""
        if "timestamp" in cols and name != "signal_events":
            newest = C.utc(con.execute(f'SELECT MAX(timestamp) FROM "{name}"').fetchone()[0])
        print(f"   {name:30} {n:8}  newest {newest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
