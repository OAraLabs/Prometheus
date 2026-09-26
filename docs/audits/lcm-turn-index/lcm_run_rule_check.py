"""Check the migration's run rule against the daemon's restarts. Aggregates only.

A drop in turn_index that lands in an unused gap of the current numbering run is
kept in that run (the mid-turn "ahead-set" shape). If such a drop ever straddled
a daemon restart, the rule would be merging two lifetimes. This counts both.

Run on the production host against a snapshot:
    python3 - <snapshot>/data/lcm.db < lcm_run_rule_check.py
"""

import bisect
import collections
import sqlite3
import subprocess
import sys

con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro&immutable=1", uri=True)
journal = subprocess.run(
    ["journalctl", "--user", "-u", "prometheus.service", "-o", "short-unix",
     "--no-pager", "--grep", "Started server process"],
    capture_output=True, text=True,
).stdout.splitlines()
starts = sorted(float(line.split()[0]) for line in journal if line[:1].isdigit())


def start_between(a: float, b: float) -> bool:
    i = bisect.bisect_right(starts, a)
    return i < len(starts) and starts[i] <= b


by_session = collections.defaultdict(list)
for row in con.execute(
    "SELECT rowid, session_id, turn_index, timestamp FROM lcm_messages"
    " ORDER BY session_id, rowid"
):
    by_session[row[1]].append(row)

counts: collections.Counter[str] = collections.Counter()
for rows in by_session.values():
    used: set[int] = set()
    prev = None
    for row in rows:
        ti = row[2]
        if ti in used or (used and ti < min(used)):
            used = set()
        elif prev is not None and ti < prev[2]:
            counts["descent kept in run (gap fill)"] += 1
            if start_between(prev[3], row[3]):
                counts["...with a daemon start between"] += 1
            else:
                counts["...no daemon start between"] += 1
            if starts and row[3] < starts[0]:
                counts["...before journal coverage"] += 1
        used.add(ti)
        prev = row
print(dict(counts))
