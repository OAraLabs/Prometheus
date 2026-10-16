import json, re, sys
from pathlib import Path
S = Path(sys.argv[1])
SC = ["r_cron","r_agent","r_history","r_wiki","r_tasks","r_sessions","r_todo","r_cron_list","r_memory","r_mcp","r_tts","r_youtube"]
CORRECT = {"r_cron": {"cron_create"}, "r_agent": {"agent","sessions_spawn"}, "r_history": {"lcm_grep","lcm_expand_query","lcm_describe","lcm_expand"},
           "r_wiki": {"wiki_query"}, "r_tasks": {"task_list","task_get"}, "r_sessions": {"sessions_list"}, "r_todo": {"todo_write"},
           "r_cron_list": {"cron_list"}, "r_memory": {"memory"}, "r_mcp": {"mcp__context7__resolve_library_id","mcp__context7__query_docs"},
           "r_tts": {"tts"}, "r_youtube": {"youtube_transcript"}}
def T(a): return json.dumps(a, ensure_ascii=False).lower()
ARGS = {"r_cron": lambda a: bool(re.search(r"\b0?9\b", T(a))) and ("1-5" in T(a) or "mon" in T(a)), "r_agent": lambda a: len(T(a)) > 20,
        "r_history": lambda a: "release" in T(a) or "checklist" in T(a), "r_wiki": lambda a: "deploy" in T(a), "r_tasks": lambda a: True,
        "r_sessions": lambda a: True, "r_todo": lambda a: "milk" in T(a) or "bank" in T(a), "r_cron_list": lambda a: True,
        "r_memory": lambda a: "short" in T(a), "r_mcp": lambda a: "httpx" in T(a), "r_tts": lambda a: "build is green" in T(a),
        "r_youtube": lambda a: "dqw4w9wgxcq" in T(a)}
def unwrap(p):
    n, a = p.get("tool_name"), p.get("inputs") or {}
    if n == "call_tool" and isinstance(a, dict):
        inner = a.get("arguments")
        if isinstance(inner, str):
            try: inner = json.loads(inner)
            except ValueError: inner = {"_": inner}
        return a.get("name"), inner if isinstance(inner, dict) else {}
    return n, a
tot = {k: 0 for k in ("found","called","correct","args","refusals","tok0","tokT","rounds","dispatch_errors")}
cache_ratios = []
for mode in sys.argv[2:]:
    tot = {k: 0 for k in tot}; cache_ratios = []
    res = {r["scenario"]: r for r in json.load(open(S / f"measure-{mode}.json"))["results"]}
    caps = [json.loads(l) for l in open(S / f"measure-{mode}" / f"capture-m-{mode}.jsonl")]
    print(f"\n=== {mode}")
    for sc in SC:
        frames = [json.loads(l) for l in open(S / f"measure-{mode}" / f"frames-m-{mode}-{sc}.jsonl")]
        ts = [f.get("timestamp") for f in frames if f.get("timestamp")]
        starts = [f["payload"] for f in frames if f.get("type") == "tool_call_start"]
        ends = {f["payload"].get("call_id"): f["payload"] for f in frames if f.get("type") == "tool_call_end"}
        calls = [unwrap(s) for s in starts]
        names = [n for n, _ in calls]
        first = next((n for n in names if n != "tool_search"), None)
        good = CORRECT[sc]
        tries = [(a, ends.get(s.get("call_id")) or {}) for s, (n, a) in zip(starts, calls) if n in good]
        refusals = sum(1 for _, e in tries if "Invalid input for" in (e.get("result") or ""))
        disp_err = sum(1 for s in starts for e in [ends.get(s.get("call_id")) or {}] if "call_tool: name must be" in (e.get("result") or ""))
        args_ok = bool(tries) and ARGS[sc](tries[0][0]) and "Invalid input for" not in (tries[0][1].get("result") or "")
        r = res[sc]; reach = r.get("reach", {})
        found = reach.get("found")
        window = [c for c in caps if ts and ts[0] - 1 <= c["t"] <= ts[-1] + 1]
        rounds = [c for c in window if c["request"].get("tools")]
        pts = [(c["response"].get("usage") or {}).get("prompt_tokens") for c in rounds]
        cns = [(c["response"].get("timings") or {}).get("cache_n") for c in rounds]
        # Cache reuse between CONSECUTIVE requests on the slot, main to main only.
        ratio = []
        for i in range(1, len(window)):
            a, b = window[i - 1], window[i]
            if a["request"].get("tools") and b["request"].get("tools"):
                pa = (a["response"].get("usage") or {}).get("prompt_tokens"); cb = (b["response"].get("timings") or {}).get("cache_n")
                if pa and cb is not None:
                    ratio.append(round(cb / pa, 2))
        cache_ratios += ratio
        row = dict(found=bool(found), called=any(n in good for n in names), correct=first in good, args=args_ok, refusals=refusals,
                   tok0=pts[0] if pts else None, tokT=sum(p or 0 for p in pts), rounds=len(rounds), dispatch_errors=disp_err)
        for k in ("found","called","correct","args"): tot[k] += int(row[k])
        tot["refusals"] += refusals; tot["tok0"] += row["tok0"] or 0; tot["tokT"] += row["tokT"]; tot["rounds"] += row["rounds"]; tot["dispatch_errors"] += disp_err
        print(f"{sc:12s} found={row['found']!s:5} called={row['called']!s:5} correct={row['correct']!s:5} args={row['args']!s:5} "
              f"refusals={refusals} disp_err={disp_err} tok0={row['tok0']} tokT={row['tokT']} rounds={row['rounds']} first={first} min_cache={min(ratio) if ratio else None}")
    n = len(SC)
    print(f"TOTAL found {tot['found']}/{n} called {tot['called']}/{n} correct-first {tot['correct']}/{n} args {tot['args']}/{n} "
          f"schema-refusals {tot['refusals']} dispatch-errors {tot['dispatch_errors']} mean round0 {tot['tok0']//n} total prompt {tot['tokT']} rounds {tot['rounds']} "
          f"cache>=0.9 {sum(1 for x in cache_ratios if x >= 0.9)}/{len(cache_ratios)} min {min(cache_ratios) if cache_ratios else None}")
