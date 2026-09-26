"""Would a recorded tool_search call have listed a skill? Replay with the daemon's own scorer.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=~/prometheus-deploy/src \\
        python3 tool_search_replay.py <snapshot-dir> <tools.json>

Telemetry keeps each tool_search INPUT (``parsed_tool_call``) but not its
result; LCM keeps results only for the sessions it stores. This replays every
recorded query through ``ToolSearchTool._score_tool`` / ``_score_skill`` and
the same stable sort and top-5 cut as ``_handle_search``, against

- the tools: ``tools.json`` (``dump_tool_catalog.py``) plus any MCP tool
  schemas telemetry recorded, and
- the skills the registry held at the call's time (born before it, not yet
  archived), in ``list_skills()`` order.

The LCM results are the check: the replay must agree with them. Aggregates only.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402
from catalog import load_catalog  # noqa: E402


def _tier(score: float) -> str:
    if score < 2.0:
        return "skill name contains the query"
    if score == 2.0:
        return "skill description contains the whole query"
    if score < 3.0:
        return "some query words in name/description"
    return "edit distance to the name only (no word matched)"


def main() -> None:
    from prometheus.tools.tool_search import ToolSearchTool

    snap, tools_path = Path(sys.argv[1]), Path(sys.argv[2])
    tel = C.telemetry(snap)
    tools = [SimpleNamespace(**t) for t in json.loads(tools_path.read_text())]
    known = {t.name for t in tools}
    for row in tel.execute("SELECT DISTINCT tool_schema FROM tool_calls WHERE tool_name LIKE 'mcp__%'"
                           " AND tool_schema IS NOT NULL"):
        s = json.loads(row["tool_schema"])
        if s.get("name") not in known:
            tools.append(SimpleNamespace(name=s["name"], description=s.get("description", "")))
            known.add(s["name"])
    cat = load_catalog(snap)

    def skills_at(ts: float | None):
        live = []
        for s in cat:
            if ts is not None and s["born"] > ts:
                continue
            if not s["served"] and (ts is None or s["archived_at"] is None or s["archived_at"] <= ts):
                continue  # archived (today), or archived with no record: treat as gone
            live.append(SimpleNamespace(name=s["name"], description=s["description"]))
        return sorted({s.name: s for s in live}.values(), key=lambda s: s.name)

    def replay(query: str, action: str, ts: float | None):
        if action == "select":
            if any(t.name == query for t in tools):
                return 0, None
            return (1, 1) if any(s.name.lower() == query.lower() or s.name == query.title()
                                 for s in skills_at(ts)) else (0, None)
        if not query.strip():
            return (len(skills_at(ts)) > 0) * 1, None
        q = query.lower()
        scored = [(ToolSearchTool._score_tool(t, q), "tool") for t in tools]
        scored += [(ToolSearchTool._score_skill(s, q), "skill") for s in skills_at(ts)]
        scored.sort(key=lambda x: x[0])
        top = scored[:5]
        listed = [sc for sc, kind in top if kind == "skill"]
        return len(listed), (min(listed) if listed else None)

    # 1. Check the replay against LCM's recorded results.
    con = C.lcm(snap)
    uses = {}
    agree = collections.Counter()
    for row in C.lcm_rows(con):
        for b in C.blocks(row["content_json"]):
            if b.get("type") == "tool_use" and b.get("name") == "tool_search":
                uses[(row["session_id"], str(b.get("id")))] = (b.get("input") or {}, row["timestamp"])
            elif b.get("type") == "tool_result" and (row["session_id"], str(b.get("tool_use_id"))) in uses:
                inp, ts = uses[(row["session_id"], str(b.get("tool_use_id")))]
                if b.get("is_error"):
                    agree["error result (skipped)"] += 1
                    continue
                content = b.get("content")
                text = content if isinstance(content, str) else json.dumps(content)
                actual = text.count('"type": "skill"')
                n, _ = replay(str(inp.get("query", "")), inp.get("action", "search"), ts)
                try:
                    json.loads(content if isinstance(content, str) else "")
                    whole = "whole JSON"
                except (json.JSONDecodeError, TypeError):
                    whole = "cut/not JSON"
                agree[(inp.get("action", "search"), whole,
                       "recorded has skill" if actual else "recorded has none",
                       "replay has skill" if n else "replay has none")] += 1
    print("== replay vs LCM-recorded results")
    for k, v in sorted(agree.items(), key=str):
        print(f"  {k}: {v}")

    # 2. Every telemetry query.
    res = collections.Counter()
    best = collections.Counter()
    for row in tel.execute("SELECT timestamp, session_id, parsed_tool_call FROM tool_calls"
                           " WHERE tool_name='tool_search'"):
        inp = C.parsed_input(row["parsed_tool_call"])
        cls = C.surface(row["session_id"])
        cls = "evals" if cls.startswith("evals") else "user surfaces" if cls.startswith("user:") else (
            "pre-2026-08-15" if cls.startswith("none") else "test/coding")
        if inp is None:
            res[(cls, "no input recorded")] += 1
            continue
        act = inp.get("action", "search")
        n, bs = replay(str(inp.get("query", "")), act, row["timestamp"])
        res[(cls, act, "lists >=1 skill" if n else "lists no skill")] += 1
        n_now, _ = replay(str(inp.get("query", "")), act, None)
        res[(cls, act, "with TODAY's catalog: lists >=1 skill" if n_now else "with TODAY's catalog: none")] += 1
        if act == "search" and bs is not None:
            best[(cls, _tier(bs))] += 1
    print("== replay over every recorded tool_search input")
    for k, v in sorted(res.items(), key=str):
        print(f"  {k}: {v}")
    print("== why the best LISTED skill made the top 5 (search calls that listed one)")
    for k, v in sorted(best.items(), key=str):
        print(f"  {k}: {v}")
    print(f"tools ranked: {len(tools)}; skills today: {len(skills_at(None))}")


if __name__ == "__main__":
    main()
