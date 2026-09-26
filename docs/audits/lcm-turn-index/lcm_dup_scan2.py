"""Scan 2: why did numbering restart? Aggregates only, no content, no session ids.

Replays SessionManager.rehydrate_if_cold's decision (window 40 rows by rowid,
8,000-token newest-first budget, first clean human turn) against the rows that
existed just before each post-restart collision.
"""
import bisect
import collections
import datetime as dt
import json
import re
import sqlite3
import subprocess
import sys

SNAP = sys.argv[1]
con = sqlite3.connect(f"file:{SNAP}?mode=ro&immutable=1", uri=True)
con.row_factory = sqlite3.Row
out = {}
WINDOW, BUDGET = 40, 8_000
PROV = {"user", "cron", "task_supervisor", "orchestrator", "teacher_escalation", "compactor",
        "supervisor", "file_mutation_verifier"}


def day(ts):
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def prefix(sid):
    if sid.startswith("cli-"):
        return "cli-"
    return sid.split(":", 1)[0] + ":" if ":" in sid else "(no prefix)"


def clean_human(role, prov, content_json):
    if role != "user":
        return False
    if prov not in PROV or prov != "user":
        return False
    if not content_json:
        return True
    try:
        blocks = json.loads(content_json)
    except Exception:
        return True  # from_stored falls back to one TextBlock
    if not blocks:
        return True
    return all(isinstance(b, dict) and b.get("type") == "text" for b in blocks)


rows = []
for r in con.execute(
    "SELECT rowid AS rid, session_id, turn_index, role, timestamp, provenance,"
    " content_json, length(coalesce(content_json, content, '')) AS clen, content"
    " FROM lcm_messages ORDER BY rowid"
):
    # cost exactly as rehydrate computes it: len(content_json or content or "") // 4
    cj = r["content_json"]
    cost_src = cj if cj else (r["content"] or "")
    rows.append(dict(rid=r["rid"], sid=r["session_id"], ti=r["turn_index"], role=r["role"],
                     ts=r["timestamp"], prov=r["provenance"],
                     clean=clean_human(r["role"], r["provenance"], cj),
                     cost=max(1, len(cost_src) // 4),
                     empty_text=(r["content"] or "") == ""))
by_s = collections.defaultdict(list)
for x in rows:
    by_s[x["sid"]].append(x)

tomb = {r[0]: r[1] for r in con.execute("SELECT session_id, deleted_at FROM session_tombstones")}
out["tombstoned_sessions"] = len(tomb)

j = subprocess.run(
    ["journalctl", "--user", "-u", "prometheus.service", "-o", "short-unix", "--no-pager",
     "--grep", r"Started server process|rehydrate: |DELETE /api/sessions/"],
    capture_output=True, text=True).stdout.splitlines()
starts, rh, deletes = [], collections.defaultdict(list), []
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
    m = re.search(r"DELETE /api/sessions/([^ /?\"]+)", line)
    if m:
        deletes.append((t, m.group(1)))
starts.sort()
out["journal_DELETE_sessions_lines"] = len(deletes)


def start_in(a, b):
    i = bisect.bisect_right(starts, a)
    return starts[i] if i < len(starts) and starts[i] <= b else None


def simulate(rs_before):
    page = rs_before[-WINDOW:]
    kept, budget = [], BUDGET
    for p in reversed(page):
        if kept and budget - p["cost"] < 0:
            break
        budget -= p["cost"]
        kept.append(p)
    kept.reverse()
    start = next((i for i, p in enumerate(kept) if p["clean"]), None)
    if start is None:
        return "no clean human turn in window", len(kept), None
    kept = kept[start:]
    return "would restore", len(kept), max(p["ti"] for p in kept) + 1


# ---- per restart point (descent into an already-used index) ----
events = []
for sid, rs in by_s.items():
    used, maxti = set(), -1
    for i, x in enumerate(rs):
        if i and x["ti"] < rs[i - 1]["ti"]:
            prev = rs[i - 1]
            if x["ti"] not in used:
                kind = "ahead-set (mid-turn row persisted early) - no collision"
                events.append(dict(kind=kind, day=day(x["ts"]), pfx=prefix(sid)))
            else:
                st = start_in(prev["ts"], x["ts"])
                ev = dict(day=day(x["ts"]), pfx=prefix(sid), ti=x["ti"], maxti=maxti,
                          first_role=x["role"], first_prov=x["prov"],
                          after_rehydrate_feature=x["ts"] >= dt.datetime(2026, 8, 31).timestamp())
                # rows following this descent that collide (until next descent)
                n = 0
                u2 = set(used)
                for jj in range(i, len(rs)):
                    y = rs[jj]
                    if jj > i and y["ti"] < rs[jj - 1]["ti"]:
                        break
                    if y["ti"] in u2:
                        n += 1
                    u2.add(y["ti"])
                ev["collisions"] = n
                if st is not None:
                    ev["kind"] = "restart"
                    ev["rehydrate_logged"] = any(st <= t <= x["ts"] for t in rh.get(sid, ()))
                    verdict, nkept, nxt = simulate(rs[:i])
                    ev["sim"] = verdict
                    ev["sim_next_ok"] = None if nxt is None else (nxt > maxti)
                    # first row in the new lifetime for this session: which path wrote it?
                    ev["lifetime_first_is_clean_user"] = x["clean"]
                    ev["tombstoned"] = sid in tomb
                    ev["tomb_hides_at_restart"] = (sid in tomb and prev["ts"] <= tomb[sid])
                else:
                    ev["kind"] = "same lifetime"
                    ev["delete_logged_between"] = any(
                        prev["ts"] <= t <= x["ts"] for t, _ in deletes)
                    ev["gap_s"] = x["ts"] - prev["ts"]
                events.append(ev)
        used.add(x["ti"])
        maxti = max(maxti, x["ti"])


def tally(evs, keys):
    c = collections.Counter()
    w = collections.Counter()
    for e in evs:
        k = tuple(str(e.get(k)) for k in keys)
        c[k] += 1
        w[k] += e.get("collisions", 0)
    return {" | ".join(k): [c[k], w[k]] for k in sorted(c)}


out["events_by_kind[points,collision_rows]"] = tally(events, ["kind"])
rs_ev = [e for e in events if e.get("kind") == "restart"]
out["restart_events_by_era_target_sim[points,rows]"] = tally(
    rs_ev, ["after_rehydrate_feature", "ti", "rehydrate_logged", "sim", "sim_next_ok"])
out["restart_events_after_0831_by_prefix_firstrow[points,rows]"] = tally(
    [e for e in rs_ev if e["after_rehydrate_feature"]],
    ["pfx", "first_role", "first_prov", "lifetime_first_is_clean_user", "sim", "tomb_hides_at_restart"])
out["restart_events_after_0831_by_day[points,rows]"] = tally(
    [e for e in rs_ev if e["after_rehydrate_feature"]], ["day", "pfx", "sim"])
sl = [e for e in events if e.get("kind") == "same lifetime"]
out["same_lifetime_by_target[points,rows]"] = tally(
    sl, ["pfx", "first_prov", "delete_logged_between"])
out["same_lifetime_target_bucket"] = tally(
    [dict(e, tb=("0" if e["ti"] == 0 else "50" if e["ti"] == 50 else "other")) for e in sl],
    ["tb", "day"])

# ---- lifetimes: how did each session's first row after a restart number itself? ----
life = collections.Counter()
life_rows = collections.Counter()
for sid, rs in by_s.items():
    maxti = -1
    last_ts = None
    for x in rs:
        if last_ts is not None and start_in(last_ts, x["ts"]) is not None:
            era = "after 08-31" if x["ts"] >= dt.datetime(2026, 8, 31).timestamp() else "before 08-31"
            if x["ti"] == maxti + 1:
                k = "continued (max+1)"
            elif x["ti"] > maxti + 1:
                k = "jumped above max"
            elif x["ti"] == 0:
                k = "from 0"
            else:
                k = "below max, not 0"
            life[(era, k)] += 1
        maxti = max(maxti, x["ti"])
        last_ts = x["ts"]
out["session_first_row_after_a_daemon_restart"] = {" | ".join(k): v for k, v in sorted(life.items())}

# ---- shape refinement ----
shape = collections.Counter()
by_day_identical = collections.Counter()
for sid, rs in by_s.items():
    seen = {}
    for x in rs:
        if x["ti"] in seen:
            p = seen[x["ti"]]
            shape[("same role" if p["role"] == x["role"] else "different role",
                   "both empty text" if p["empty_text"] and x["empty_text"] else "text present")] += 1
        seen[x["ti"]] = x
out["collision_shape_refined"] = {" | ".join(k): v for k, v in shape.items()}

print(json.dumps(out, indent=1, default=str))
