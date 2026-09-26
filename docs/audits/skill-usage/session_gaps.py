"""Tool calls with no session id: how many, why, and from which surface.

    python3 session_gaps.py <snapshot-dir>

``tool_calls.session_id`` appears when a daemon first runs the code that adds
it (the first non-null row). Rows before that have no session by
construction. After it, ``engine/agent_loop.py`` still writes NULL in three
cases, which this scan separates:

- **failure paths**: every ``telemetry.record(...)`` except the main success
  path omits ``session_id`` (validation_failed, permission_denied,
  hook_blocked, tool_exception, no_registry, …);
- **ephemeral sessions**: the success path passes ``None`` on purpose and
  also nulls ``parsed_tool_call`` / ``raw_model_output``;
- **runs with no session at all**: a success row that kept its parsed call.

Each post-column NULL row is attributed to a surface through its nearest
neighbour in the same table: the closest row (either side, within 5 min)
with the same model and a session id, else the latest ``loop_round`` row with
the same model within 10 min. Aggregates only.
"""

from __future__ import annotations

import bisect
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def main() -> None:
    tel = C.telemetry(Path(sys.argv[1]))
    rows = list(tel.execute(
        "SELECT rowid AS rid, timestamp, model, tool_name, success, error_type, session_id,"
        " parsed_tool_call IS NOT NULL AS has_parsed, is_golden FROM tool_calls ORDER BY rowid"))
    start = min(r["timestamp"] for r in rows if r["session_id"] is not None)
    calls = [r for r in rows if r["tool_name"] != "_loop_transition"]
    pseudo = [r for r in rows if r["tool_name"] == "_loop_transition"]
    before = [r for r in calls if r["timestamp"] < start]
    after = [r for r in calls if r["timestamp"] >= start]
    gap = [r for r in after if r["session_id"] is None]
    print(f"== session_id column: first non-null row {C.dt.datetime.fromtimestamp(start, C.dt.UTC):%Y-%m-%d %H:%M} UTC")
    print(f"   tool calls (no _loop_transition): {len(calls)}; with no session id: "
          f"{sum(r['session_id'] is None for r in calls)}")
    print(f"   before the column existed: {len(before)} (all NULL by construction; "
          f"{sum(r['is_golden'] for r in before)} golden)")
    print(f"   since the column exists:   {len(after)}, of which NULL: {len(gap)} "
          f"({len(gap) / max(len(after), 1):.1%})")
    print(f"   _loop_transition pseudo-rows since then with NULL session: "
          f"{sum(1 for r in pseudo if r['timestamp'] >= start and r['session_id'] is None)} of "
          f"{sum(1 for r in pseudo if r['timestamp'] >= start)}")

    # error_type values written by the main execution path, which DOES pass a
    # session id; every other writer in the tool path omits it.
    main_path = {None, "nonzero_exit", "tool_error"}

    def why(r) -> str:
        if r["error_type"] == "lucky_guess":
            return "lucky_guess marker (a deferred tool called by name; the call itself is its own row)"
        if r["error_type"] not in main_path:
            return f"failure path ({r['error_type']})"
        if not r["has_parsed"]:
            return "main path, ephemeral-shaped (no parsed call)"
        return "main path, the run had no session id"

    print("== why the post-column rows are NULL")
    for k, v in collections.Counter(why(r) for r in gap).most_common():
        print(f"   {v:5}  {k}")

    # Attribute each NULL row to a surface via its neighbours.
    by_model = collections.defaultdict(list)
    for r in after:
        if r["session_id"] is not None:
            by_model[r["model"]].append((r["timestamp"], r["session_id"]))
    for m in by_model:
        by_model[m].sort()
    rounds = collections.defaultdict(list)
    for r in tel.execute("SELECT timestamp, model, session_id FROM subsystem_runs WHERE subsystem='agent_loop'"
                         " AND operation='loop_round' AND session_id IS NOT NULL ORDER BY timestamp"):
        rounds[r["model"]].append((r["timestamp"], r["session_id"]))

    def attribute(r) -> str:
        lst = by_model.get(r["model"], [])
        i = bisect.bisect_left(lst, (r["timestamp"], ""))
        near = [lst[j] for j in (i - 1, i) if 0 <= j < len(lst) and abs(lst[j][0] - r["timestamp"]) <= 300]
        if near:
            return C.surface(min(near, key=lambda x: abs(x[0] - r["timestamp"]))[1])
        rl = rounds.get(r["model"], [])
        k = bisect.bisect_right(rl, (r["timestamp"], "￿")) - 1
        if k >= 0 and r["timestamp"] - rl[k][0] <= 600:
            return C.surface(rl[k][1]) + " (via loop_round)"
        return "unattributed"

    print("== post-column NULL rows by attributed surface / provider")
    tab = collections.Counter((attribute(r), C.provider_of(C.model_label(r["model"]))) for r in gap)
    for k, v in tab.most_common():
        print(f"   {v:5}  {k[0]:52} {k[1]}")
    print("== post-column NULL rows: attributed surface class x why")
    cross = collections.Counter()
    for r in gap:
        a = attribute(r)
        cls = ("user surfaces" if a.startswith("user:") else "evals / harness" if a.startswith("evals")
               else "other")
        w = why(r)
        cross[(cls, "failure path" if w.startswith("failure") else w.split(" (")[0])] += 1
    for k, v in sorted(cross.items()):
        print(f"   {v:5}  {k[0]:16} {k[1]}")
    print("== post-column NULL rows by ISO week (NULL / all calls that week)")
    wk_all = collections.Counter(C.week(r["timestamp"]) for r in after)
    wk_gap = collections.Counter(C.week(r["timestamp"]) for r in gap)
    print("   " + ", ".join(f"{w}: {wk_gap[w]}/{wk_all[w]}" for w in sorted(wk_all)))
    lucky = collections.Counter(r["tool_name"] for r in calls if r["error_type"] == "lucky_guess")
    print(f"== lucky_guess markers, all time: {sum(lucky.values())} across {len(lucky)} tools; "
          f"skill {lucky.get('skill', 0)}, tool_search {lucky.get('tool_search', 0)}; top {lucky.most_common(6)}")
    print("== of them, skill / tool_search:",
          sum(r["tool_name"] == "skill" for r in gap), "/", sum(r["tool_name"] == "tool_search" for r in gap))


if __name__ == "__main__":
    main()
