#!/usr/bin/env python3
"""Verified teacher rows per hour, by target area and category (stdlib only; read-only).

    python3 gen_stats.py [--runs '~/distill/phase2b/runs/*'] [--hours 24] [--json] [--now EPOCH]

Reads every generation run directory (``teacher_gen.py --out``: the local runs, the
remote-teacher runs, any Max runs) through its ledger(s) ``*.tasks.jsonl``: a task
event with status ``kept`` or ``released`` contributes its ``rows`` (verified rows
written to the traces) at its ``ts``. Relaxed, rejected, skipped and errored tasks
add no rows. Torn last lines are skipped. Nothing is written.

Per run and over all runs, for the last ``--hours`` (default 24) and since the run's
first event:

* ``rows``, ``rows_per_hour`` (rows / the window's wall-clock hours; since start: from
  the first event to now) and ``rows_per_active_hour`` (rows / hours that saw any task
  finish: generation runs in windows, so wall-clock rates understate a window's pace);
* ``by_area``: round-2 rows by target area (``injection-resistance``, ``error-recovery``,
  ``long-context``, ``daily-agent``, ``coding``); a round-1 task is ``round-1``;
* ``by_category``: rows by category (chat-mode, memory/wiki, sentinel/daemon, other,
  coding);
* ``tasks``: finished tasks by status and the keep rate (kept / attempted);
* ``keep_rate_by_area``: kept / attempted per round-2 area;
* ``tool_choice`` (headline, operator 2026-11-01 ~23:00 ET): over the attempted tool-choice
  tasks, how many made a first call other than the reference tool (``wrong_tool``) or no
  call (``no_call``), and ``wrong_tool_rate`` = (wrong + no call) / tasks with a verdict
  (``teacher_gen.py``'s ``verification.r2.toolchoice``).

The daily 09:00 heartbeat calls ``gen_stats.py --json`` (or the text form, a few lines).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

DEFAULT_RUNS = "~/distill/phase2b/runs/*"
VERIFIED = frozenset({"kept", "released"})
ATTEMPTED = frozenset({"kept", "released", "held", "relaxed", "rejected"})


def read_events(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in data.split("\n"):
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue  # a torn last line (the writer is mid-append)
        if isinstance(ev, dict) and isinstance(ev.get("ts"), (int, float)):
            out.append(ev)
    return out


def run_dirs(patterns: Iterable[str]) -> list[Path]:
    dirs: set[Path] = set()
    for pat in patterns:
        for p in glob.glob(os.path.expanduser(pat)):
            path = Path(p)
            if path.is_dir() and any(path.glob("*.tasks.jsonl")):
                dirs.add(path)
    return sorted(dirs)


def _released_rows(events: list[dict[str, Any]]) -> dict[str, int]:
    """A ``released`` event carries no rows of its own: they are its task's held rows."""
    held: dict[str, int] = {}
    for ev in events:
        if ev.get("status") == "held":
            held[str(ev.get("task_id"))] = int(ev.get("rows") or 0)
    return held


def tally(events: list[dict[str, Any]], since: float, now: float) -> dict[str, Any]:
    held = _released_rows(events)
    rows = 0
    by_area: collections.Counter = collections.Counter()
    by_cat: collections.Counter = collections.Counter()
    status: collections.Counter = collections.Counter()
    hours: set[int] = set()
    area_tried: collections.Counter = collections.Counter()
    area_kept: collections.Counter = collections.Counter()
    tc: collections.Counter = collections.Counter()
    for ev in events:
        ts = float(ev["ts"])
        if ts < since or ts > now:
            continue
        st = str(ev.get("status"))
        if st in ATTEMPTED or st in ("skipped", "error"):
            status[st] += 1
        if st in ATTEMPTED:
            hours.add(int(ts // 3600))
            area = str(ev.get("area") or "round-1")
            area_tried[area] += 1
            area_kept[area] += st in VERIFIED or st == "held"
            v = (((ev.get("verification") or {}).get("r2") or {}).get("toolchoice")
                 if isinstance(ev.get("verification"), dict) else None)
            if isinstance(v, dict) and "right" in v:
                tc["tasks"] += 1
                tc["right"] += bool(v["right"])
                tc["no_call"] += v.get("first") is None
                tc["wrong_tool"] += (not v["right"]) and v.get("first") is not None
                tc["wrong_neighbour"] += bool(v.get("neighbour"))
        if st not in VERIFIED:
            continue
        n = int(ev.get("rows") or 0) if st == "kept" else held.get(str(ev.get("task_id")), 0)
        rows += n
        by_area[str(ev.get("area") or "round-1")] += n
        by_cat[str(ev.get("domain") or "unknown")] += n
    span_h = max((now - since) / 3600.0, 1e-9)
    attempted = sum(status[s] for s in ATTEMPTED)
    kept = status["kept"] + status["released"]
    return {"rows": rows, "hours": round(span_h, 2), "rows_per_hour": round(rows / span_h, 2),
            "active_hours": len(hours),
            "rows_per_active_hour": round(rows / len(hours), 2) if hours else None,
            "by_area": dict(by_area.most_common()), "by_category": dict(by_cat.most_common()),
            "by_area_per_hour": {k: round(v / span_h, 2) for k, v in by_area.most_common()},
            "by_category_per_hour": {k: round(v / span_h, 2) for k, v in by_cat.most_common()},
            "tasks": dict(status.most_common()),
            "keep_rate": round(kept / attempted, 3) if attempted else None,
            "keep_rate_by_area": {a: {"kept": area_kept[a], "tried": n, "rate": round(area_kept[a] / n, 3)}
                                  for a, n in area_tried.most_common()},
            "tool_choice": {**{k: tc[k] for k in ("tasks", "right", "wrong_tool", "no_call", "wrong_neighbour")},
                            "wrong_tool_rate": round((tc["wrong_tool"] + tc["no_call"]) / tc["tasks"], 3)
                            if tc["tasks"] else None}}


def stats(patterns: Iterable[str], hours: float = 24.0, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    runs: dict[str, Any] = {}
    every: list[dict[str, Any]] = []
    for d in run_dirs(patterns):
        events = [ev for f in sorted(d.glob("*.tasks.jsonl")) for ev in read_events(f)]
        if not events:
            continue
        start = min(float(ev["ts"]) for ev in events)
        runs[d.name] = {"first_event": start, "last_event": max(float(ev["ts"]) for ev in events),
                        "last_hours": tally(events, now - hours * 3600, now),
                        "since_start": tally(events, start, now)}
        every += events
    out: dict[str, Any] = {"schema": "distill-gen-stats/1", "now": now, "hours": hours, "runs": runs}
    if every:
        start = min(float(ev["ts"]) for ev in every)
        out["all"] = {"first_event": start, "last_hours": tally(every, now - hours * 3600, now),
                      "since_start": tally(every, start, now)}
    return out


def _fmt(d: dict[str, Any], key: str) -> str:
    items = d.get(key) or {}
    return ", ".join(f"{k} {v}" for k, v in items.items()) or "-"


def render(out: dict[str, Any]) -> str:
    lines = []
    h = out["hours"]
    for name, r in list(out["runs"].items()) + ([("ALL", out["all"])] if "all" in out else []):
        last, tot = r["last_hours"], r["since_start"]
        lines.append(f"{name}: last {h:g}h {last['rows']} rows ({last['rows_per_hour']}/h, "
                     f"{last['rows_per_active_hour'] or 0}/active h, keep {last['keep_rate']}); "
                     f"since start {tot['rows']} rows ({tot['rows_per_hour']}/h over {tot['hours']}h)")
        lines.append(f"  last {h:g}h per hour by area: {_fmt(last, 'by_area_per_hour')}")
        lines.append(f"  last {h:g}h per hour by category: {_fmt(last, 'by_category_per_hour')}")
        lines.append(f"  since start by area: {_fmt(tot, 'by_area')}; by category: {_fmt(tot, 'by_category')}")
        tc = last.get("tool_choice") or {}
        if tc.get("tasks"):
            lines.append(f"  last {h:g}h tool choice: wrong-tool rate {tc['wrong_tool_rate']:.1%} "
                         f"({tc['wrong_tool']} wrong, {tc['no_call']} no call of {tc['tasks']})")
    return "\n".join(lines) or "no generation runs found"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gen_stats.py", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--runs", action="append", help=f"run directory glob (repeatable; default {DEFAULT_RUNS})")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--now", type=float, help="epoch seconds (tests)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    out = stats(a.runs or [DEFAULT_RUNS], a.hours, a.now)
    print(json.dumps(out, indent=1, sort_keys=True) if a.json else render(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
