"""Aggregate-only scan of lcm_messages turn_index duplicates.

Runs ON THE MINI against a nightly snapshot opened immutable/read-only.
Never prints message content or session ids: hashes and counts only.
"""
import collections
import datetime as dt
import hashlib
import json
import re
import sqlite3
import subprocess
import sys

SNAP = sys.argv[1]
con = sqlite3.connect(f"file:{SNAP}?mode=ro&immutable=1", uri=True)
con.row_factory = sqlite3.Row
out = {}

cols = [r[1] for r in con.execute("PRAGMA table_info(lcm_messages)")]
out["columns"] = cols
out["indexes"] = [tuple(r) for r in con.execute(
    "SELECT name, sql FROM sqlite_master WHERE tbl_name='lcm_messages' AND type='index'")]


def day(ts):
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def prefix(sid):
    if sid.startswith("cli-"):
        return "cli-"
    if ":" in sid:
        return sid.split(":", 1)[0] + ":"
    return "(no prefix)"


rows = []
for r in con.execute(
    "SELECT rowid AS rid, session_id, turn_index, role, content, content_json,"
    " timestamp, compacted, provenance FROM lcm_messages ORDER BY rowid"
):
    h = hashlib.sha256(
        (r["role"] + "\0" + (r["content"] or "") + "\0" + (r["content_json"] or "")).encode()
    ).hexdigest()
    hc = hashlib.sha256((r["content"] or "").encode()).hexdigest()
    rows.append((r["rid"], r["session_id"], r["turn_index"], r["role"], r["timestamp"],
                 r["compacted"], r["provenance"], h, hc))

out["total_rows"] = len(rows)
by_s = collections.defaultdict(list)
for row in rows:
    by_s[row[1]].append(row)
out["total_sessions"] = len(by_s)
out["first_row_day"] = day(min(r[4] for r in rows))
out["last_row_day"] = day(max(r[4] for r in rows))

# timestamps vs rowid monotonic?
dec_ts = sum(1 for a, b in zip(rows, rows[1:]) if b[4] < a[4])
out["timestamp_decreases_along_rowid_global"] = dec_ts

# ---- daemon lifetimes from the journal (uvicorn "Started server process") ----
j = subprocess.run(
    ["journalctl", "--user", "-u", "prometheus.service", "-o", "short-unix", "--no-pager",
     "--grep", r"Started server process|rehydrate: "],
    capture_output=True, text=True).stdout.splitlines()
starts, rehydrates = [], []
rh_re = re.compile(r"rehydrate: (\S+) restored (\d+) message")
for line in j:
    try:
        t = float(line.split()[0])
    except ValueError:
        continue
    if "Started server process" in line:
        starts.append(t)
    m = rh_re.search(line)
    if m:
        rehydrates.append((t, m.group(1), int(m.group(2))))
starts.sort()
out["journal_daemon_starts"] = len(starts)
out["journal_first_start_day"] = day(starts[0]) if starts else None
out["journal_rehydrate_events"] = len(rehydrates)
out["journal_first_rehydrate_day"] = day(rehydrates[0][0]) if rehydrates else None
out["journal_rehydrate_by_day"] = dict(collections.Counter(day(t) for t, _, _ in rehydrates))
rh_by_sid = collections.defaultdict(list)
for t, sid, n in rehydrates:
    rh_by_sid[sid].append(t)


def start_between(a, b):
    """True if a daemon start happened in (a, b]."""
    import bisect
    i = bisect.bisect_right(starts, a)
    return i < len(starts) and starts[i] <= b


# ---- duplicates ----
collision_rows = []   # (row, earlier_row_same_index)
groups = collections.Counter()
for sid, rs in by_s.items():
    seen = {}
    for row in rs:
        groups[(sid, row[2])] += 1
        if row[2] in seen:
            collision_rows.append((row, seen[row[2]]))
        seen[row[2]] = row
dup_groups = {k: v for k, v in groups.items() if v > 1}
out["dup_groups"] = len(dup_groups)
out["rows_in_dup_groups"] = sum(dup_groups.values())
out["collision_rows"] = len(collision_rows)
out["group_size_hist"] = dict(sorted(collections.Counter(dup_groups.values()).items()))
aff = {k[0] for k in dup_groups}
out["affected_sessions"] = len(aff)
out["affected_rows_share_of_affected_sessions"] = (
    sum(dup_groups.values()), sum(len(by_s[s]) for s in aff))
out["collision_first_day"] = day(min(r[0][4] for r in collision_rows))
out["collision_last_day"] = day(max(r[0][4] for r in collision_rows))

# prefix breakdown
pref = collections.defaultdict(lambda: [0, 0, 0, 0])  # sessions, affected, rows, collisions
for sid, rs in by_s.items():
    p = prefix(sid)
    pref[p][0] += 1
    pref[p][2] += len(rs)
    if sid in aff:
        pref[p][1] += 1
for row, _ in collision_rows:
    pref[prefix(row[1])][3] += 1
out["by_prefix[sessions,affected,rows,collisions]"] = dict(pref)

# per-day rows and collisions
per_day = collections.defaultdict(lambda: [0, 0])
for row in rows:
    per_day[day(row[4])][0] += 1
for row, _ in collision_rows:
    per_day[day(row[4])][1] += 1
out["per_day[rows,collisions]"] = dict(sorted(per_day.items()))

# ---- shape ----
shape = collections.Counter()
gap = collections.Counter()
for row, prev in collision_rows:
    if row[7] == prev[7]:
        shape["identical(role,content,content_json)"] += 1
        g = row[4] - prev[4]
        gap["<1s" if g < 1 else "<60s" if g < 60 else "<1h" if g < 3600 else ">=1h"] += 1
    elif row[3] == prev[3] and row[8] == prev[8]:
        shape["same role+content, different content_json"] += 1
    elif row[3] == prev[3]:
        shape["same role, different content"] += 1
    else:
        shape["different role"] += 1
out["shape_vs_previous_same_index"] = dict(shape)
out["identical_gap"] = dict(gap)
# identical content ANYWHERE earlier in the session (not just same index)
first_rid_of_hash = {}
for row in rows:
    first_rid_of_hash.setdefault((row[1], row[7]), row[0])
anyid = sum(1 for row, _ in collision_rows if first_rid_of_hash[(row[1], row[7])] < row[0])
out["collision_rows_whose_exact_message_exists_earlier_in_session"] = anyid
# whole-group shape
gshape = collections.Counter()
for (sid, ti), n in dup_groups.items():
    hs = {r[7] for r in by_s[sid] if r[2] == ti}
    gshape["all identical" if len(hs) == 1 else "all distinct" if len(hs) == n else "mixed"] += 1
out["group_shape"] = dict(gshape)

# ---- cause attribution: find the "restart point" each collision descends from ----
cause = collections.Counter()
cause_by_day = collections.defaultdict(collections.Counter)
restart_points = collections.Counter()
restart_target = collections.Counter()
lifetime_first = collections.Counter()
for sid, rs in by_s.items():
    maxti = -1
    run_cause = None
    prev = None
    used_ti = set()
    for row in rs:
        ti = row[2]
        is_coll = ti in used_ti
        used_ti.add(ti)
        if prev is not None and ti < prev[2] and ti <= maxti:
            # a descent into already-used territory: classify the jump
            restarted = start_between(prev[4], row[4])
            rehyd = any(prev[4] < t <= row[4] for t in rh_by_sid.get(sid, ()))
            if restarted and ti == 0:
                c = "daemon restart -> numbering from 0 (cold, no rehydrate)"
            elif restarted and rehyd:
                c = "daemon restart + rehydrate -> next_turn_index <= session max"
            elif restarted:
                c = "daemon restart -> numbering from >0 (no rehydrate log line)"
            elif ti == 0:
                c = "same lifetime -> numbering from 0 (/reset, forget, or 2nd writer)"
            else:
                c = "same lifetime -> descent to >0"
            run_cause = c
            restart_points[c] += 1
            restart_target["0" if ti == 0 else "1-9" if ti < 10 else "10-49" if ti < 50 else ">=50"] += 1
        if prev is not None and ti == prev[2] and is_coll and row[7] != prev[7]:
            c2 = "same index as immediately previous row, different message"
            cause[c2] += 1
            cause_by_day[day(row[4])][c2] += 1
        elif prev is not None and ti == prev[2] and is_coll:
            c2 = "same index as immediately previous row, identical message"
            cause[c2] += 1
            cause_by_day[day(row[4])][c2] += 1
        elif is_coll:
            c2 = run_cause or "unattributed"
            cause[c2] += 1
            cause_by_day[day(row[4])][c2] += 1
        maxti = max(maxti, ti)
        prev = row
out["restart_points(descents into used indices)"] = dict(restart_points)
out["restart_target_value"] = dict(restart_target)
out["collision_rows_by_cause"] = dict(cause)
out["collision_rows_by_day_and_cause"] = {k: dict(v) for k, v in sorted(cause_by_day.items())}

# sessions ever decreasing along rowid, and decreases that do not collide (ahead-set shape)
dec = nondup_dec = 0
dec_sessions = set()
for sid, rs in by_s.items():
    used = set()
    for a, b in zip(rs, rs[1:]):
        used.add(a[2])
        if b[2] < a[2]:
            dec += 1
            dec_sessions.add(sid)
            if b[2] not in used:
                nondup_dec += 1
out["decreases_along_rowid"] = dec
out["decreases_into_unused_index(ahead-set shape)"] = nondup_dec
out["sessions_with_decrease"] = len(dec_sessions)

# ---- impact on compaction actually run in production ----
try:
    scols = [r[1] for r in con.execute("PRAGMA table_info(lcm_summaries)")]
    out["summary_columns"] = scols
    rid_of = {}
    for r in con.execute("SELECT rowid, id, session_id FROM lcm_messages"):
        rid_of[r[1]] = r[0]
    n = in_order = out_of_order = missing = multi_life = 0
    affected_sum_sessions = set()
    sums_in_aff = 0
    for r in con.execute(
        "SELECT session_id, source_message_ids, depth FROM lcm_summaries WHERE depth = 0"
    ):
        ids = json.loads(r[1] or "[]")
        if not ids:
            continue
        n += 1
        rids = [rid_of.get(i) for i in ids]
        if any(x is None for x in rids):
            missing += 1
            continue
        if rids == sorted(rids):
            in_order += 1
        else:
            out_of_order += 1
            affected_sum_sessions.add(r[0])
        if r[0] in aff:
            sums_in_aff += 1
    out["depth0_summaries"] = n
    out["depth0_summaries_sources_in_rowid_order"] = in_order
    out["depth0_summaries_sources_OUT_of_rowid_order"] = out_of_order
    out["depth0_summaries_with_missing_sources"] = missing
    out["sessions_with_out_of_order_summary"] = len(affected_sum_sessions)
    out["depth0_summaries_in_affected_sessions"] = sums_in_aff
    out["summaries_total"] = con.execute("SELECT COUNT(*) FROM lcm_summaries").fetchone()[0]
except sqlite3.Error as e:
    out["summary_error"] = str(e)

# compaction inversion: an uncompacted row older (rowid) than a compacted row, beyond the tail
inv_sessions = 0
inv_rows = 0
for sid, rs in by_s.items():
    comp = [r[0] for r in rs if r[5]]
    if not comp:
        continue
    last_comp = max(comp)
    older_uncomp = [r for r in rs if not r[5] and r[0] < last_comp]
    if older_uncomp:
        inv_sessions += 1
        inv_rows += len(older_uncomp)
out["sessions_with_compacted_rows"] = sum(1 for rs in by_s.values() if any(r[5] for r in rs))
out["compaction_inversion_sessions(uncompacted row older than a compacted one)"] = inv_sessions
out["compaction_inversion_rows"] = inv_rows

# would get_uncompacted_messages (ORDER BY turn_index) differ from rowid order right now?
differs = 0
fresh_differs = 0
for sid, rs in by_s.items():
    unc = [r for r in rs if not r[5]]
    if not unc:
        continue
    by_ti = sorted(unc, key=lambda r: r[2])  # stable: ties keep rowid order (best case)
    if [r[0] for r in by_ti] != [r[0] for r in unc]:
        differs += 1
        if {r[0] for r in by_ti[-32:]} != {r[0] for r in unc[-32:]}:
            fresh_differs += 1
out["sessions_where_turn_index_order_differs_from_rowid_now(uncompacted)"] = differs
out["...of_which_fresh_tail_32_selects_different_rows"] = fresh_differs

print(json.dumps(out, indent=1, default=str))
