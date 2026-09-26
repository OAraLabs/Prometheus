"""Was the ``skill`` tool (and ``tool_search``) in front of the model? Per run, from telemetry.

    python3 advertised.py <snapshot-dir>

``subsystem_runs`` rows ``agent_loop/tool_advertisement`` (one per run since
2026-07-31) record ``deferred_active``, ``advertised`` and ``registered_total``
but not the tool names, so the set is inferred from the run's shape:

- deferred active -> exactly ``tools.deferred_loading.always_loaded`` (+ MCP
  tools when ``mcp_always_deferred`` is false). On the mini that list holds
  ``tool_search`` and not ``skill``.
- ``advertised == registered_total`` on the main registry -> the full catalog,
  which holds both.
- a 7-tool "registry direct" run -> a restricted registry (evals / coding).

Qwen 3.8 Max needs a second step. Before #462 (cd1b798, 2026-09-11) the
advertisement was resolved BEFORE routing, so a session routed to the cloud
model could be served with the LOCAL model's deferred set. Each run's round-0
``loop_round`` row names the model that served it; it is paired with the
latest advertisement row of the same session (or of the pre-#458 ``web``
namespace) within 30 minutes. Aggregates only.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def shape(d: dict) -> str:
    if d.get("deferred_active"):
        # 13 = the mini's always_loaded list; 15 = that + the two context7 MCP
        # tools. Smaller sets are older lists or a profile filter (research = 3),
        # whose contents the row does not record.
        ts = "tool_search yes" if d.get("advertised") in (13, 15) else "tool_search unknown"
        return f"deferred set ({d.get('advertised')} tools: {ts}, skill no)"
    if d.get("advertised") == d.get("registered_total") and (d.get("registered_total") or 0) > 20:
        return f"full catalog ({d.get('advertised')} tools: tool_search yes, skill yes)"
    return f"restricted registry ({d.get('advertised')} of {d.get('registered_total')})"


def main() -> None:
    tel = C.telemetry(Path(sys.argv[1]))
    ads = list(tel.execute(
        "SELECT timestamp, session_id, model, summary_json FROM subsystem_runs"
        " WHERE subsystem='agent_loop' AND operation='tool_advertisement' ORDER BY timestamp"))
    print(f"== {len(ads)} runs with an advertisement row, "
          f"{C.day(ads[0]['timestamp'])} .. {C.day(ads[-1]['timestamp'])}")
    tab = collections.Counter()
    for r in ads:
        d = json.loads(r["summary_json"] or "{}")
        label = C.model_label(r["model"])
        tab[(C.provider_of(label), C.surface(r["session_id"]).split(" (")[0], shape(d))] += 1
    for k, v in sorted(tab.items(), key=lambda kv: -kv[1]):
        print(f"  {v:5}  {k[0]:32} {k[1]:24} {k[2]}")
    main_runs = {k: v for k, v in tab.items() if not k[2].startswith("restricted")}
    skill_adv = sum(v for k, v in main_runs.items() if "skill yes" in k[2])
    ts_yes = sum(v for k, v in main_runs.items() if "tool_search yes" in k[2])
    print(f"  main-registry runs: {sum(main_runs.values())}; skill advertised on {skill_adv}; "
          f"tool_search advertised on {ts_yes} (unknown on {sum(main_runs.values()) - ts_yes})")

    print("== Qwen 3.8 Max: the advertisement each served run actually got")
    ev = list(tel.execute(
        "SELECT timestamp, session_id, operation, model, summary_json FROM subsystem_runs"
        " WHERE subsystem='agent_loop' AND (operation='tool_advertisement'"
        " OR (operation='loop_round' AND round_index=0)) ORDER BY timestamp"))
    last: dict[str | None, tuple] = {}
    runs = collections.Counter()
    weekly = collections.Counter()
    for r in ev:
        if r["operation"] == "tool_advertisement":
            last[r["session_id"]] = (r["timestamp"], r["model"], json.loads(r["summary_json"] or "{}"))
            continue
        ad = last.pop(r["session_id"], None)
        if ad is None or r["timestamp"] - ad[0] > 1800:
            web = last.get("web")
            ad = last.pop("web") if web and r["timestamp"] - web[0] <= 1800 else None
        if r["model"] != "qwen3.8-max":
            continue
        key = "no advertisement row found" if ad is None else (
            f"advertised for {C.model_label(ad[1])}: {shape(ad[2])}")
        runs[key] += 1
        weekly[(C.week(r["timestamp"]), "skill yes" in key)] += 1
    total = sum(runs.values())
    for k, v in runs.most_common():
        print(f"  {v:4}  {k}")
    print(f"  total served runs {total}; tool_search advertised on "
          f"{sum(v for k, v in runs.items() if 'tool_search yes' in k)}; skill advertised on "
          f"{sum(v for k, v in runs.items() if 'skill yes' in k)}")
    print("  by week (skill advertised / not):",
          {w: (weekly[(w, True)], weekly[(w, False)]) for w in sorted({w for w, _ in weekly})})


if __name__ == "__main__":
    main()
