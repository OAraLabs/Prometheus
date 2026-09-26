"""Past restarts: what rehydrate restored then, and what this code restores now.

For every time a session was resumed after a daemon restart (a journal start
between two of its rows), the REAL ``SessionManager.rehydrate_if_cold`` of the
checkout on PYTHONPATH runs against the rows that existed before that restart.
They are served from memory by a stand-in store, so nothing is written. The
journal says what the daemon actually restored at the time.

Runs on the production host against a nightly snapshot. Aggregates only: no
content, no session ids.
    PYTHONPATH=<checkout>/src python3 rehydrate_restarts.py <snapshot>/data/lcm.db
"""

import bisect
import collections
import json
import logging
import re
import sqlite3
import statistics
import subprocess
import sys

from prometheus.engine.session import SessionManager
from prometheus.memory.lcm_types import MessagePart


class _Store:
    """The three reads rehydrate makes, over an in-memory list of rows."""

    def __init__(self, sid: str, rows: list[MessagePart]) -> None:
        self.sid, self.rows = sid, rows

    def list_sessions(self):
        return [{"session_id": self.sid}]

    def messages_page(self, *, limit, before=None, session_id=None, include_compacted=True):
        rows = [r for r in self.rows if before is None or r.row_id < before]
        page = rows[-limit:]
        return page, len(rows) > len(page)

    def next_turn_index(self, session_id):
        return max((r.turn_index for r in self.rows), default=-1) + 1


class _Engine:
    def __init__(self, store: _Store) -> None:
        self.conversation_store = store


class _Details(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.last = ""

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith("rehydrate: "):
            self.last = record.getMessage()


details = _Details()
logging.getLogger("prometheus.engine.session").addHandler(details)
logging.getLogger("prometheus.engine.session").setLevel(logging.INFO)


def rehydrate(sid: str, rows: list[MessagePart]):
    mgr = SessionManager()
    mgr.lcm_engine = _Engine(_Store(sid, rows))
    mgr.rehydrate_enabled = True
    details.last = ""
    restored = mgr.rehydrate_if_cold(sid)
    messages = mgr.get_or_create(sid).messages
    tokens = sum(max(1, len(m.content_json) // 4) for m in messages)
    shortened = re.search(r"(\d+) tool payload", details.last)
    left_out = re.search(r"(\d+) message\(s\) left out", details.last)
    n_short = int(shortened.group(1)) if shortened else 0
    n_left = int(left_out.group(1)) if left_out else 0
    if not restored:
        how = "nothing"
    elif "(window " in details.last:
        how = "window (as before)"
    elif n_left:
        how = "newest turn: request + newest rounds"
    elif n_short:
        how = "newest turn: tool payloads shortened"
    else:
        how = "newest turn: as it is"
    return {"restored": restored, "tokens": tokens if restored else 0, "how": how,
            "shortened": n_short, "left_out": n_left}


con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro&immutable=1", uri=True)
journal = subprocess.run(
    ["journalctl", "--user", "-u", "prometheus.service", "-o", "short-unix", "--no-pager",
     "--grep", r"Started server process|rehydrate: "],
    capture_output=True, text=True).stdout.splitlines()
starts, logged = [], collections.defaultdict(list)
for line in journal:
    try:
        t = float(line.split()[0])
    except (ValueError, IndexError):
        continue
    if "Started server process" in line:
        starts.append(t)
    m = re.search(r"rehydrate: (\S+) restored (\d+)", line)
    if m:
        logged[m.group(1)].append(t)
starts.sort()
rehydrate_on = min((t for v in logged.values() for t in v), default=float("inf")) - 600


def start_between(a, b):
    i = bisect.bisect_right(starts, a)
    return starts[i] if i < len(starts) and starts[i] <= b else None


rows_by_session = collections.defaultdict(list)
for r in con.execute(
    "SELECT rowid, session_id, turn_index, role, content, content_json, timestamp, provenance,"
    " is_trusted FROM lcm_messages ORDER BY rowid"
):
    rows_by_session[r[1]].append(MessagePart(
        row_id=r[0], session_id=r[1], turn_index=r[2], role=r[3], content=r[4],
        content_json=r[5], timestamp=r[6], provenance=r[7], is_trusted=bool(r[8])))

events = []
for sid, rows in rows_by_session.items():
    for i in range(1, len(rows)):
        st = start_between(rows[i - 1].timestamp, rows[i].timestamp)
        if st is None:
            continue
        first = rows[i]
        path = ("injected (inject_turn)" if first.role == "user" and first.provenance != "user"
                else "human message" if first.role == "user" else "assistant row first")
        events.append({
            "era": "rehydrate on" if st >= rehydrate_on else "before rehydrate existed",
            "path": path,
            "journal_restored": any(st <= t <= first.timestamp + 5 for t in logged.get(sid, ())),
            "now": rehydrate(sid, rows[:i]),
        })


def stats(values):
    v = sorted(values)
    return {"n": len(v), "min": v[0], "median": statistics.median(v), "max": v[-1]} if v else None


out = {"daemon_starts": len(starts), "resumes_after_a_restart": len(events)}
for era in ("rehydrate on", "before rehydrate existed"):
    ev = [e for e in events if e["era"] == era]
    blind = [e for e in ev if not e["journal_restored"]]
    out[era] = {
        "resumes": len(ev),
        "restored_at_the_time(journal)": len(ev) - len(blind),
        "restored_nothing_at_the_time": len(blind),
        "blind_by_path": dict(collections.Counter(e["path"] for e in blind)),
        "blind_by_what_this_code_does(how)": dict(collections.Counter(
            f"{e['path']} -> {e['now']['how']}" for e in blind)),
        "blind_now_restored": sum(1 for e in blind if e["now"]["restored"]),
        "blind_now_messages": stats([e["now"]["restored"] for e in blind if e["now"]["restored"]]),
        "blind_now_tokens": stats([e["now"]["tokens"] for e in blind if e["now"]["restored"]]),
        "blind_now_payloads_shortened": stats([e["now"]["shortened"] for e in blind if e["now"]["restored"]]),
        "blind_now_messages_left_out": stats([e["now"]["left_out"] for e in blind if e["now"]["restored"]]),
        "blind_now_by_path": {
            path: {
                "restored": sum(1 for e in grp if e["now"]["restored"]),
                "messages": stats([e["now"]["restored"] for e in grp if e["now"]["restored"]]),
                "tokens": stats([e["now"]["tokens"] for e in grp if e["now"]["restored"]]),
                "payloads_shortened": stats([e["now"]["shortened"] for e in grp if e["now"]["restored"]]),
                "messages_left_out": stats([e["now"]["left_out"] for e in grp if e["now"]["restored"]]),
            }
            for path, grp in sorted(
                {p: [e for e in blind if e["path"] == p] for p in {e["path"] for e in blind}}.items())
        },
    }
out["all_resumes_this_code(how)"] = dict(collections.Counter(e["now"]["how"] for e in events))
print(json.dumps(out, indent=1))
