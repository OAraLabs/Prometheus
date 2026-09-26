"""Scan 4: the exporter's two effects apart, the collision debt, and the exports on disk.

Aggregates only. Run on the production host against a nightly snapshot dir:
    python3 - <snapshot dir> < lcm_dup_scan4.py
"""

import collections
import datetime as dt
import glob
import hashlib
import json
import os
import sqlite3
import sys

SNAPDIR = sys.argv[1]
con = sqlite3.connect(f"file:{SNAPDIR}/data/lcm.db?mode=ro&immutable=1", uri=True)
tel = sqlite3.connect(f"file:{SNAPDIR}/telemetry.db?mode=ro&immutable=1", uri=True)
out: dict[str, object] = {}

by_session = collections.defaultdict(list)
for row in con.execute(
    "SELECT rowid, session_id, turn_index, timestamp FROM lcm_messages ORDER BY rowid"
):
    by_session[row[1]].append(row)

affected = set()
for sid, rows in by_session.items():
    seen: set[int] = set()
    for row in rows:
        if row[2] in seen:
            affected.add(sid)
            break
        seen.add(row[2])

# Collision debt: how far the newest numbering still sits below the session max.
debt = [
    max(0, max(r[2] for r in by_session[sid]) - by_session[sid][-1][2]) for sid in affected
]
out["affected_sessions_whose_next_row_will_still_collide"] = sum(1 for d in debt if d > 0)
out["rows_until_numbering_clears_session_max: sorted"] = sorted(debt)
out["sessions_over_500_rows"] = sum(1 for rows in by_session.values() if len(rows) > 500)
out["affected_sessions_over_500_rows"] = sum(1 for s in affected if len(by_session[s]) > 500)

ROLES = {"system", "user", "assistant", "tool"}


def resolve(messages: list[tuple], ts: float) -> str:
    """The exporter's context rule, hashed: before ts, text only, no trailing assistant, last 12."""
    kept = [(m[0], m[1]) for m in messages if m[2] < ts and m[0] in ROLES and m[1]]
    while kept and kept[-1][0] == "assistant":
        kept.pop()
    return hashlib.sha256(json.dumps(kept[-12:]).encode()).hexdigest()


calls = tel.execute(
    "SELECT session_id, timestamp FROM tool_calls WHERE is_golden = 1 AND session_id IS NOT NULL"
).fetchall()
cache: dict[str, tuple[list, list, list]] = {}
counts: collections.Counter[str] = collections.Counter()
for sid, ts in calls:
    if sid not in by_session:
        counts["session not in lcm"] += 1
        continue
    if sid not in cache:
        q = "SELECT role, content, timestamp FROM lcm_messages WHERE session_id = ? ORDER BY "
        cache[sid] = (
            con.execute(q + "turn_index ASC LIMIT 500", (sid,)).fetchall(),
            con.execute(q + "turn_index ASC", (sid,)).fetchall(),
            con.execute(q + "rowid ASC", (sid,)).fetchall(),
        )
    real, no_limit, by_rowid = cache[sid]
    r, n, ch = resolve(real, ts), resolve(no_limit, ts), resolve(by_rowid, ts)
    group = "affected" if sid in affected else "unaffected"
    size = ">500 rows" if len(by_session[sid]) > 500 else "<=500 rows"
    if r == ch:
        kind = "same as rowid order"
    elif n == ch:
        kind = "differs ONLY because of LIMIT 500"
    elif r == n:
        kind = "differs because of turn_index order (LIMIT irrelevant)"
    else:
        kind = "differs from both causes"
    counts[f"{group} | {size} | {kind}"] += 1
out["golden_context_resolution"] = dict(sorted(counts.items()))

files = sorted(glob.glob(os.path.expanduser("~/.prometheus/trajectories/golden_traces_*.jsonl")))
out["golden_export_files"] = len(files)
if files:
    stamps = [int(os.path.basename(f).split("_")[2]) for f in files]
    out["golden_export_first_last_day"] = [
        dt.datetime.fromtimestamp(min(stamps)).strftime("%Y-%m-%d"),
        dt.datetime.fromtimestamp(max(stamps)).strftime("%Y-%m-%d"),
    ]
    total = 0
    for f in files:
        with open(f) as fh:
            total += sum(1 for _ in fh)
    out["golden_export_lines_total"] = total
print(json.dumps(out, indent=1))
