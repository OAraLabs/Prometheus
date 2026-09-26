"""Scan 3: attribution by numbering event and period; impact on the exporter and
the compactor using SQLite's REAL tie order. Aggregates only."""
import bisect
import collections
import datetime as dt
import hashlib
import json
import re
import sqlite3
import subprocess
import sys

SNAPDIR = sys.argv[1]
LCM = f"file:{SNAPDIR}/data/lcm.db?mode=ro&immutable=1"
TEL = f"file:{SNAPDIR}/telemetry.db?mode=ro&immutable=1"
con = sqlite3.connect(LCM, uri=True)
con.row_factory = sqlite3.Row
out = {}


def day(ts):
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def T(s):
    return dt.datetime.fromisoformat(s).timestamp()


PERIODS = [
    ("A 05-27..06-26 (no journal)", 0, T("2026-06-27")),
    ("B 06-27..08-30 (no rehydrate)", T("2026-06-27"), T("2026-08-31")),
    ("C 08-31..09-11 (rehydrate on, trim unfixed)", T("2026-08-31"), T("2026-09-11 15:55")),
    ("D 09-11..09-24 (trim fixed, deployed)", T("2026-09-11 15:55"), 1e12),
]


def period(ts):
    for n, a, b in PERIODS:
        if a <= ts < b:
            return n


j = subprocess.run(["journalctl", "--user", "-u", "prometheus.service", "-o", "short-unix",
                    "--no-pager", "--grep", r"Started server process|rehydrate: "],
                   capture_output=True, text=True).stdout.splitlines()
starts, rh = [], collections.defaultdict(list)
for line in j:
    try:
        t = float(line.split()[0])
    except ValueError:
        continue
    if "Started server process" in line:
        starts.append(t)
    m = re.search(r"rehydrate: (\S+) restored (\d+)", line)
    if m:
        rh[m.group(1)].append(t)
starts.sort()


def start_in(a, b):
    i = bisect.bisect_right(starts, a)
    return starts[i] if i < len(starts) and starts[i] <= b else None


rows = [dict(r) for r in con.execute(
    "SELECT rowid AS rid, session_id AS sid, turn_index AS ti, role, timestamp AS ts, compacted,"
    " provenance AS prov FROM lcm_messages ORDER BY rowid")]
by_s = collections.defaultdict(list)
for x in rows:
    by_s[x["sid"]].append(x)

# ---- attribution: each collision row -> the most recent numbering event before it ----
attr = collections.Counter()
for sid, rs in by_s.items():
    used, gmax, event = set(), -1, "none"
    for i, x in enumerate(rs):
        if i:
            p = rs[i - 1]
            st = start_in(p["ts"], x["ts"])
            if st is not None and x["ti"] <= gmax:
                logged = any(st <= t <= x["ts"] for t in rh.get(sid, ()))
                if x["ti"] == 0:
                    event = "restart: numbered from 0 (cold; rehydrate absent/declined/not called)"
                elif x["ti"] == p["ti"] + 1:
                    event = ("restart: rehydrated, continued from WINDOW max (<= session max)"
                             if logged else
                             "restart: continued from last row, below session max (no rehydrate log)")
                else:
                    event = "restart: other value below session max"
            elif st is not None:
                event = "none"   # clean continuation above the session max
            elif x["ti"] < p["ti"] and x["ti"] in used:
                if x["ti"] == 0:
                    event = "same lifetime: numbered from 0 (/reset or /clear)"
                else:
                    event = "same lifetime: fell back to a used index >0 (trim plateau before #446)"
        if x["ti"] in used:
            k = event
            if i and x["ti"] == rs[i - 1]["ti"]:
                k = "same index as the row just before (rollback retry / double write)"
            attr[(period(x["ts"]), k)] += 1
        used.add(x["ti"])
        gmax = max(gmax, x["ti"])
out["collision_rows_by_period_and_event"] = {f"{a} || {b}": v for (a, b), v in sorted(attr.items())}
rp = collections.Counter()
for x in rows:
    rp[period(x["ts"])] += 1
out["rows_by_period"] = dict(rp)

# ---- weekly rate ----
wk = collections.defaultdict(lambda: [0, 0])
for sid, rs in by_s.items():
    used = set()
    for x in rs:
        w = dt.datetime.fromtimestamp(x["ts"]).strftime("%G-W%V")
        wk[w][0] += 1
        if x["ti"] in used:
            wk[w][1] += 1
        used.add(x["ti"])
out["weekly[rows,collisions]"] = dict(sorted(wk.items()))

# ---- what does SQLite actually do with ties? ----
out["plan_get_uncompacted"] = [r[3] for r in con.execute(
    "EXPLAIN QUERY PLAN SELECT * FROM lcm_messages WHERE session_id = ? AND compacted = 0"
    " ORDER BY turn_index ASC", ("x",))]
out["plan_get_messages"] = [r[3] for r in con.execute(
    "EXPLAIN QUERY PLAN SELECT * FROM lcm_messages WHERE session_id = ?"
    " ORDER BY turn_index ASC LIMIT ?", ("x", 500))]
out["plan_get_all"] = [r[3] for r in con.execute(
    "EXPLAIN QUERY PLAN SELECT * FROM lcm_messages WHERE session_id = ? ORDER BY turn_index ASC",
    ("x",))]
ties_rowid_order = ties_other = 0
interleave_sessions = 0
for sid in by_s:
    got = [r[0] for r in con.execute(
        "SELECT rowid, turn_index FROM lcm_messages WHERE session_id = ? ORDER BY turn_index ASC",
        (sid,))]
    tis = {x["rid"]: x["ti"] for x in by_s[sid]}
    grp = collections.defaultdict(list)
    for r in got:
        grp[tis[r]].append(r)
    for g in grp.values():
        if len(g) > 1:
            if g == sorted(g):
                ties_rowid_order += 1
            else:
                ties_other += 1
    if got != [x["rid"] for x in by_s[sid]]:
        interleave_sessions += 1
out["dup_groups_returned_in_rowid_order_by_real_query"] = ties_rowid_order
out["dup_groups_returned_in_other_order"] = ties_other
out["sessions_whose_ORDER_BY_turn_index_differs_from_rowid"] = interleave_sessions

# ---- compaction in production: interleaving inside summary batches ----
rid_of, ts_of = {}, {}
for r in con.execute("SELECT rowid, id, timestamp FROM lcm_messages"):
    rid_of[r[1]] = r[0]
    ts_of[r[1]] = r[2]
span_ok, span_bad = [], []
interleaved = 0
for r in con.execute("SELECT source_message_ids, created_at FROM lcm_summaries WHERE depth = 0"):
    ids = json.loads(r[0] or "[]")
    if not ids or any(i not in rid_of for i in ids):
        continue
    rids = [rid_of[i] for i in ids]
    span = (max(ts_of[i] for i in ids) - min(ts_of[i] for i in ids)) / 3600
    if rids == sorted(rids):
        span_ok.append(span)
    else:
        span_bad.append(span)
        # count direction changes: >1 means rows from two eras alternate
        ch = sum(1 for a, b in zip(rids, rids[1:]) if b < a)
        if ch > 1:
            interleaved += 1


def med(v):
    v = sorted(v)
    return round(v[len(v) // 2], 2) if v else None


out["depth0_in_order: n, median source span h"] = [len(span_ok), med(span_ok)]
out["depth0_out_of_order: n, median source span h"] = [len(span_bad), med(span_bad)]
out["depth0_out_of_order_with_alternating_eras(>1 backward step)"] = interleaved
out["depth0_out_of_order_by_created_day"] = dict(sorted(collections.Counter(
    day(r[1]) for r in con.execute(
        "SELECT source_message_ids, created_at FROM lcm_summaries WHERE depth = 0")
    if json.loads(r[0] or "[]") and all(i in rid_of for i in json.loads(r[0]))
    and [rid_of[i] for i in json.loads(r[0])] != sorted(rid_of[i] for i in json.loads(r[0]))
).items()))

# compaction inversion classification
inv_pure = inv_other = 0
for sid, rs in by_s.items():
    comp = [x for x in rs if x["compacted"]]
    if not comp:
        continue
    for u in rs:
        if u["compacted"]:
            continue
        newer_comp = [c for c in comp if c["rid"] > u["rid"]]
        if not newer_comp:
            continue
        if all(c["ti"] < u["ti"] for c in newer_comp):
            inv_pure += 1
        else:
            inv_other += 1
out["uncompacted_rows_older_than_a_compacted_row: all newer compacted rows have LOWER turn_index"] = inv_pure
out["...: some newer compacted row has higher turn_index (e.g. failed batch)"] = inv_other

# ---- golden-trace exporter: real query vs rowid order ----
tel = sqlite3.connect(TEL, uri=True)
calls = tel.execute(
    "SELECT session_id, timestamp FROM tool_calls WHERE is_golden = 1 AND session_id IS NOT NULL"
).fetchall()
affected = set()
for sid, rs in by_s.items():
    seen = set()
    for x in rs:
        if x["ti"] in seen:
            affected.add(sid)
            break
        seen.add(x["ti"])
ROLES = {"system", "user", "assistant", "tool"}


def resolve(msgs, ts):
    o = [(m[0], m[1]) for m in msgs if m[2] < ts and m[0] in ROLES and m[1]]
    while o and o[-1][0] == "assistant":
        o.pop()
    return o[-12:]


cache = {}
same = diff = none_real = none_rowid = in_aff = 0
for sid, ts in calls:
    if sid not in by_s:
        continue
    if sid not in affected:
        continue
    in_aff += 1
    if sid not in cache:
        real = con.execute("SELECT role, content, timestamp FROM lcm_messages WHERE session_id = ?"
                           " ORDER BY turn_index ASC LIMIT 500", (sid,)).fetchall()
        byrow = con.execute("SELECT role, content, timestamp FROM lcm_messages WHERE session_id = ?"
                            " ORDER BY rowid ASC", (sid,)).fetchall()
        cache[sid] = (real, byrow)
    real, byrow = cache[sid]
    a, b = resolve(real, ts), resolve(byrow, ts)
    ha = hashlib.sha256(json.dumps(a).encode()).hexdigest()
    hb = hashlib.sha256(json.dumps(b).encode()).hexdigest()
    if not a:
        none_real += 1
    if not b:
        none_rowid += 1
    if ha == hb:
        same += 1
    else:
        diff += 1
out["golden_calls_with_session"] = len(calls)
out["golden_calls_in_affected_sessions"] = in_aff
out["golden_context_same_as_chronological"] = same
out["golden_context_DIFFERENT_from_chronological"] = diff
out["golden_context_empty_with_real_query"] = none_real
out["golden_context_empty_chronological"] = none_rowid

print(json.dumps(out, indent=1, default=str))
