"""tool_search on every recorded query: skills listed before and after the relevance gate.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<checkout with the gate>/src \\
        python3 tool_search_gate_before_after.py <snapshot-dir> <tools.json>

"Before" is the replay ``tool_search_replay.py`` validated against LCM's
recorded results (14 of 14 whole results reproduced). "After" is the same
ranking with the gate the C1 change adds: a skill enters the top 5 only when
``skill_match_score(query, name, description) >= SKILL_MIN_MATCH``. Tool
ranking is identical in both. Aggregates only.
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


def main() -> None:
    from prometheus.tools.tool_search import SKILL_MIN_MATCH, ToolSearchTool, skill_match_score

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

    def skills_at(ts: float):
        live = [SimpleNamespace(name=s["name"], description=s["description"]) for s in cat
                if s["born"] <= ts and (s["served"] or (s["archived_at"] or 0) > ts)]
        return sorted({s.name: s for s in live}.values(), key=lambda s: s.name)

    def listed(query: str, ts: float, gate: bool) -> int:
        q = query.lower()
        scored = [(ToolSearchTool._score_tool(t, q), "tool") for t in tools]
        for s in skills_at(ts):
            if gate and skill_match_score(query, s.name, s.description) < SKILL_MIN_MATCH:
                continue
            scored.append((ToolSearchTool._score_skill(s, q), "skill"))
        scored.sort(key=lambda x: x[0])
        return sum(1 for _, kind in scored[:5] if kind == "skill")

    start = C.session_column_start(tel)
    table = collections.Counter()
    for row in tel.execute("SELECT timestamp, session_id, parsed_tool_call FROM tool_calls"
                           " WHERE tool_name='tool_search'"):
        inp = C.parsed_input(row["parsed_tool_call"])
        if inp is None or inp.get("action", "search") != "search" or not str(inp.get("query", "")).strip():
            continue
        cls = C.surface(row["session_id"], row["timestamp"], start)
        cls = ("evals/benchmarks" if cls.startswith("evals") else "user surfaces"
               if cls.startswith("user:") else "no session id" if cls.startswith("no session")
               else "test/coding")
        q = str(inp["query"])
        before, after = listed(q, row["timestamp"], False), listed(q, row["timestamp"], True)
        table[(cls, "searches")] += 1
        table[(cls, "before: lists >=1 skill")] += before > 0
        table[(cls, "after: lists >=1 skill")] += after > 0
        table[(cls, "skill entries before")] += before
        table[(cls, "skill entries after")] += after
    print(f"== recorded searches (non-empty queries), SKILL_MIN_MATCH = {SKILL_MIN_MATCH}")
    for cls in ("user surfaces", "no session id", "evals/benchmarks", "test/coding"):
        n = table[(cls, "searches")]
        if n:
            print(f"   {cls:16} searches {n:3} | listing a skill: before {table[(cls, 'before: lists >=1 skill')]:3}"
                  f" → after {table[(cls, 'after: lists >=1 skill')]:3} | skill entries: before "
                  f"{table[(cls, 'skill entries before')]:3} → after {table[(cls, 'skill entries after')]:3}")


if __name__ == "__main__":
    main()
