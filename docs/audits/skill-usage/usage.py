"""Real skill usage: skill and tool_search calls across ALL telemetry tool calls, plus LCM.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=~/prometheus-deploy/src \\
        python3 usage.py <snapshot-dir>

Telemetry ``tool_calls`` (every executed or attempted call, golden or not) is
the primary source; ``_loop_transition`` pseudo-rows are excluded. LCM gives
what telemetry lacks: the tool_search RESULT text (did it list a skill?) and
the human turns. Aggregates only.
"""

from __future__ import annotations

import bisect
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402
from catalog import load_catalog, registry_view  # noqa: E402

SKILL_ENTRY = re.compile(r'"type"\s*:\s*"skill"')


def main() -> None:
    snap = Path(sys.argv[1])
    tel = C.telemetry(snap)
    cat = load_catalog(snap)
    reg = registry_view(cat)
    archived = {s["name"] for s in cat if not s["served"]} | {s["stem"] for s in cat if not s["served"]}

    rows = list(tel.execute(
        "SELECT timestamp, model, served_model, tool_name, success, error_type, error_detail,"
        " parsed_tool_call, raw_model_output, session_id, is_golden FROM tool_calls"
        " WHERE tool_name != '_loop_transition'"
    ))
    print(f"== telemetry tool_calls: {len(rows)} calls, "
          f"{C.day(min(r['timestamp'] for r in rows))} .. {C.day(max(r['timestamp'] for r in rows))}")

    def label(r):
        return C.model_label(r["model"], r["served_model"])

    print("== by provider / model: calls, tool_search, skill")
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        k = (C.provider_of(label(r)), label(r))
        agg[k][0] += 1
        agg[k][1] += r["tool_name"] == "tool_search"
        agg[k][2] += r["tool_name"] == "skill"
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f"  {k[0]:32} {k[1]:44} calls={v[0]:5} tool_search={v[1]:4} skill={v[2]}")
    prov = collections.defaultdict(lambda: [0, 0, 0])
    for k, v in agg.items():
        for i in range(3):
            prov[k[0]][i] += v[i]
    print("  -- by provider:", {p: v for p, v in sorted(prov.items(), key=lambda kv: -kv[1][0])})

    print("== by surface: calls, tool_search, skill")
    start = C.session_column_start(tel)
    sur = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        s = C.surface(r["session_id"], r["timestamp"], start)
        sur[s][0] += 1
        sur[s][1] += r["tool_name"] == "tool_search"
        sur[s][2] += r["tool_name"] == "skill"
    for k, v in sorted(sur.items(), key=lambda kv: -kv[1][0]):
        print(f"  {k:44} calls={v[0]:5} tool_search={v[1]:4} skill={v[2]}")

    print("== by ISO week: calls, tool_search, skill  (all surfaces | tool_search on user surfaces)")
    wk = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        w = C.week(r["timestamp"])
        wk[w][0] += 1
        wk[w][1] += r["tool_name"] == "tool_search"
        wk[w][2] += r["tool_name"] == "skill"
        wk[w][3] += r["tool_name"] == "tool_search" and C.is_user_surface(r["session_id"])
    for w in sorted(wk):
        v = wk[w]
        print(f"  {w}  calls={v[0]:5} tool_search={v[1]:3} skill={v[2]} | user-surface tool_search={v[3]}")

    print("== tool_search inputs (telemetry parsed_tool_call)")
    acts = collections.Counter()
    for r in rows:
        if r["tool_name"] != "tool_search":
            continue
        inp = C.parsed_input(r["parsed_tool_call"])
        if inp is None:
            acts["no parsed call recorded"] += 1
            continue
        q = str(inp.get("query", ""))
        kind = "empty query" if not q.strip() else (
            "query mentions 'skill'" if "skill" in q.lower() else "other query")
        acts[(inp.get("action", "search"), kind)] += 1
    for k, v in acts.most_common():
        print(f"  {k}: {v}")
    ok = collections.Counter((r["success"], r["error_type"]) for r in rows if r["tool_name"] == "tool_search")
    print("  outcome:", dict(ok))

    print("== skill calls")
    sk = [r for r in rows if r["tool_name"] == "skill"]
    for r in sk:
        inp = C.parsed_input(r["parsed_tool_call"]) or {}
        name = str(inp.get("name", ""))
        where = ("builtin" if name in reg and reg[name]["source"] == "builtin" else
                 "user" if name in reg and reg[name]["source"] == "user" else
                 "auto" if name in reg else "archived auto" if name in archived else "not in catalog")
        nf = "Skill not found" in (r["error_detail"] or "")
        print(f"  {C.day(r['timestamp'])} {label(r):32} {C.surface(r['session_id'], r['timestamp'], start):44} "
              f"success={r['success']} not_found={nf} golden={r['is_golden']} resolves_to={where}")
    odd = collections.Counter(r["tool_name"] for r in rows
                              if "skill" in r["tool_name"].lower() and r["tool_name"] != "skill")
    blank = sum(1 for r in rows if not r["tool_name"].strip()
                and "skill" in (r["raw_model_output"] or "").lower())
    print(f"  other tool names containing 'skill': {dict(odd) or 0}; "
          f"blank-name calls whose raw output mentions 'skill': {blank}")

    # ---------------- LCM: results and turns ----------------
    con = C.lcm(snap)
    uses: dict[tuple[str, str], dict] = {}
    results = collections.Counter()
    current_turn: dict[str, int] = {}
    skill_turns: set[tuple[str, int]] = set()
    turn_count = collections.Counter()
    for row in C.lcm_rows(con):
        sid = row["session_id"]
        if C.human_text(row["role"], row["provenance"], row["content"], row["content_json"]):
            current_turn[sid] = row["rid"]
            turn_count[C.surface(sid)] += 1
            continue
        for b in C.blocks(row["content_json"]):
            if b.get("type") == "tool_use" and b.get("name") in ("tool_search", "skill"):
                uses[(sid, str(b.get("id")))] = {"name": b.get("name"), "input": b.get("input") or {}}
                if b.get("name") == "skill" and sid in current_turn:
                    skill_turns.add((sid, current_turn[sid]))
            elif b.get("type") == "tool_result":
                u = uses.get((sid, str(b.get("tool_use_id"))))
                if not u:
                    continue
                content = b.get("content")
                text = content if isinstance(content, str) else json.dumps(content)
                if u["name"] == "tool_search":
                    act = u["input"].get("action", "search")
                    n = len(SKILL_ENTRY.findall(text))
                    results[(act, "error" if b.get("is_error") else ("≥1 skill listed" if n else "no skill listed"))] += 1
                    results[("skill entries total", act)] += n
                    try:
                        json.loads(content if isinstance(content, str) else "")
                    except (json.JSONDecodeError, TypeError):
                        results[("result is not whole JSON (truncated or list)", act)] += 1
                else:
                    results[("skill result", "error" if b.get("is_error") else "ok",
                             "Skill not found" in text)] += 1
    n_ts = sum(1 for u in uses.values() if u["name"] == "tool_search")
    n_sk = sum(1 for u in uses.values() if u["name"] == "skill")
    print(f"== LCM: tool_search uses={n_ts}, skill uses={n_sk}")
    for k, v in sorted(results.items(), key=lambda kv: str(kv[0])):
        print(f"  {k}: {v}")
    print("== LCM human turns by surface, and turns with >=1 skill load")
    for k, v in turn_count.most_common():
        print(f"  {k:44} turns={v}")
    user_turns = sum(v for k, v in turn_count.items() if k.startswith("user:"))
    print(f"  user-surface turns: {user_turns}; turns with a skill load: {len(skill_turns)}")

    # Telemetry runs: one round-0 row per model-call sequence (06-12 onward).
    r0 = list(tel.execute(
        "SELECT timestamp, session_id FROM subsystem_runs WHERE subsystem='agent_loop'"
        " AND operation='loop_round' AND round_index=0"))
    first = min(r["timestamp"] for r in r0)
    skill_runs = set()
    ts_sorted = sorted(r["timestamp"] for r in r0)
    for r in sk:
        i = bisect.bisect_right(ts_sorted, r["timestamp"])
        if i:
            skill_runs.add(ts_sorted[i - 1])
    user_r0 = sum(1 for r in r0 if C.is_user_surface(r["session_id"]) or r["session_id"] is None)
    print(f"== telemetry runs (round-0 rows since {C.day(first)}): {len(r0)} "
          f"(user surfaces or pre-session-column: {user_r0}); runs containing a skill call: {len(skill_runs)}")


if __name__ == "__main__":
    main()
