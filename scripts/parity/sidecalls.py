"""Which model calls are the turn's own, and which run beside it.

A scenario's TURN is a strict sequence: every round's request, in order. Beside
it the daemon makes calls that no turn waits on, or whose timing a turn does
not decide:

* ``title``: the session title (engine/session_titles.py), fire-and-forget
  after a turn's ``chat_done``. Turn 2's ``chat_done`` can schedule a second
  title task while turn 1's is still in flight (the titler checks for an
  existing title before asking and marks nothing in flight), so the same
  request goes out once or twice.
* ``summary``: a lone user message with no tools: the compactor's span
  summaries, LCM summaries, and the boot-time thinking-suppression warm-up.

Both were shown to vary between two runs of UNCHANGED code (0.9.12, audit
2026-10-16): every ``model_switch`` re-recording attempt that day was refused
for a repeated title request, and ``compaction`` replays put the title's
``subsystem_runs`` row before or after the turn's ``tool_advertisement`` row.
So side calls are matched by KIND and content, in any order and any repeat
count, and the rows they write are compared as a set; the turn's requests and
rows stay strict.
"""

from __future__ import annotations

import json
import re
from typing import Any

MAIN = "main"
TITLE = "title"
SUMMARY = "summary"
SIDE_KINDS = frozenset({TITLE, SUMMARY})

#: The titler's system prompt (engine/session_titles.py generate_title);
#: tests/test_parity_sidecalls.py fails if the source stops saying it.
TITLE_SYSTEM = "You name conversations. Respond with only the title."

#: subsystem_runs rows written by side calls: compared as a set.
SIDE_SUBSYSTEMS = frozenset({"session_titles"})


def _texts(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
    return ""


def request_kind(req: Any) -> str:
    """``title``, ``summary`` or ``main`` for one normalized completion request."""
    if not isinstance(req, dict):
        return MAIN
    msgs = req.get("messages") or []
    systems = [_texts(m.get("content")) for m in msgs if isinstance(m, dict) and m.get("role") == "system"]
    if _texts(req.get("system")):
        systems.append(_texts(req.get("system")))
    if any(s.strip() == TITLE_SYSTEM for s in systems):
        return TITLE
    if not req.get("tools") and not systems and len(msgs) == 1 and (msgs[0] or {}).get("role") == "user":
        return SUMMARY
    return MAIN


def _split(path: str, table: str, dump: Any) -> tuple[Any, list[list] | None]:
    """(*dump* without the side calls' rows, those rows or None). The side rows'
    ``id`` is neutralized: nothing refers to a title run, and a repeat has its own."""
    if not (isinstance(dump, dict) and table == "subsystem_runs" and path.endswith("telemetry.db")):
        return dump, None
    cols = dump.get("columns") or []
    if "subsystem" not in cols:
        return dump, None
    at = cols.index("subsystem")
    at_id = cols.index("id") if "id" in cols else None
    main, side = [], {}
    for r in dump.get("rows") or []:
        if r[at] not in SIDE_SUBSYSTEMS:
            main.append(r)
            continue
        r = list(r)
        if at_id is not None:
            r[at_id] = "<side-run>"
        side[json.dumps(r, sort_keys=True, ensure_ascii=False, default=str)] = r
    return {**dump, "rows": main}, [side[k] for k in sorted(side)]


_ORDINAL = re.compile(r"<([A-Za-z][A-Za-z0-9_]*):([0-9]+)>")


def split_stores(stores: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict]]:
    """Normalized stores with the side calls' rows taken out, and those rows
    by ``"<path>:<table>"`` (``{"columns", "rows"}``).

    The normalizer numbers placeholders (``<uuid:N>``) in order of first
    appearance, so a title row that moves shifts the number of every id after
    it. Both parts are renumbered in one fixed walk with the side rows last: a
    relabeling, so the turn's rows are equal here exactly when they are equal
    up to the side calls' order and repeats.
    """
    main: dict[str, Any] = {}
    side: dict[str, dict] = {}
    for path, dump in stores.items():
        if isinstance(dump, dict) and isinstance(dump.get("sqlite"), dict):
            tables = {}
            for table, d in dump["sqlite"].items():
                tables[table], rows = _split(path, table, d)
                if rows:
                    side[f"{path}:{table}"] = {"columns": d.get("columns") or [], "rows": rows}
            dump = {**dump, "sqlite": tables}
        main[path] = dump
    # "main" sorts before "side": the turn's placeholders are numbered first.
    text = json.dumps({"main": main, "side": side}, sort_keys=True, ensure_ascii=False, default=str)
    seen: dict[str, dict[str, int]] = {}

    def relabel(m: re.Match) -> str:
        table = seen.setdefault(m.group(1), {})
        return f"<{m.group(1)}:{table.setdefault(m.group(2), len(table) + 1)}>"

    out = json.loads(_ORDINAL.sub(relabel, text))
    return out["main"], out["side"]


def side_diff(recorded: dict | None, replayed: dict | None) -> list[str]:
    """What is wrong with a replay's side-call rows, or nothing.

    Every replayed row must be one the recording wrote (a side call is answered
    from a recorded exchange, so its row is that exchange's row), and every
    kind (subsystem, operation) recorded must be written at least once. A
    recording that asked the same title twice, and got two different answers,
    is satisfied by either one.
    """
    cols = (recorded or replayed or {}).get("columns") or []
    rec = [dict(zip(cols, r)) for r in (recorded or {}).get("rows") or []]
    rep = [dict(zip(cols, r)) for r in (replayed or {}).get("rows") or []]

    def kind(r: dict) -> tuple:
        return (r.get("subsystem"), r.get("operation"))

    out = [f"side-call row the recording never wrote: {json.dumps(r, sort_keys=True, default=str)[:400]}"
           for r in rep if r not in rec]
    out += [f"side call recorded but never made: {k}" for k in sorted({kind(r) for r in rec} - {kind(r) for r in rep},
                                                                       key=str)]
    return out
