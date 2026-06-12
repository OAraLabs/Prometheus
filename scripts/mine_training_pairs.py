#!/usr/bin/env python3
"""Mine historical self-correction pairs out of telemetry.db.

D4 finding: the adapter repair machinery has zero recorded events in all
history, but `parsed_tool_call` is stored on BOTH failed and successful
rows — so failed-call → later-success-on-the-same-tool arcs are minable
retroactively. This backfills the flywheel before live capture accrues.

What counts as a minable failure (calling errors, not execution outcomes):
  - error_type = 'input_validation'  (pydantic rejected the args)
  - error_type = 'tool_error' whose detail matches the mode-misuse shape
    ("'command' is required for local_bash tasks", …) — D2's class
Bash rows with plain nonzero exits (grep no-match, missing CLIs, timeouts)
are EXCLUDED: those calls were well-formed (D3).

A pair = (failed parsed_tool_call, first successful parsed_tool_call of the
same tool within --window seconds). Context is telemetry-only (historical
rows carry no session linkage — limitation noted in the row's context.kind).

Usage:
    uv run python scripts/mine_training_pairs.py            # dry-run report
    uv run python scripts/mine_training_pairs.py --commit   # write pairs
    uv run python scripts/mine_training_pairs.py --window 900 --commit
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

from prometheus.learning.pair_capture import PairStore

TELEMETRY_DB = Path("~/.prometheus/telemetry.db").expanduser()
MODE_MISUSE_RE = re.compile(r"'\w+' (?:is|are) required")


def _parse_call(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not data.get("name"):
        return None
    return {"name": data["name"], "input": data.get("input", {})}


def mine(window_s: float, db_path: Path = TELEMETRY_DB) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    failures = conn.execute(
        """
        SELECT id, timestamp, model, tool_name, error_type, error_detail,
               parsed_tool_call
          FROM tool_calls
         WHERE success = 0
           AND parsed_tool_call IS NOT NULL
           AND tool_name NOT IN ('', '_loop_transition', '_malformed')
           AND (
                error_type = 'input_validation'
             OR (error_type = 'tool_error' AND error_detail IS NOT NULL)
           )
         ORDER BY timestamp
        """
    ).fetchall()

    pairs: list[dict] = []
    for f in failures:
        if f["error_type"] == "tool_error" and not MODE_MISUSE_RE.search(
            f["error_detail"] or ""
        ):
            continue  # well-formed call, execution-level outcome — not minable
        rejected = _parse_call(f["parsed_tool_call"])
        if rejected is None:
            continue
        success = conn.execute(
            """
            SELECT timestamp, model, parsed_tool_call FROM tool_calls
             WHERE tool_name = ? AND success = 1
               AND timestamp > ? AND timestamp <= ?
               AND parsed_tool_call IS NOT NULL
             ORDER BY timestamp LIMIT 1
            """,
            (f["tool_name"], f["timestamp"], f["timestamp"] + window_s),
        ).fetchone()
        if success is None:
            continue
        chosen = _parse_call(success["parsed_tool_call"])
        if chosen is None or chosen == rejected:
            continue
        pairs.append({
            "pair_source": "self_correction",
            "model_id": f["model"],
            "tool_name": f["tool_name"],
            "context": {
                "kind": "telemetry_only",
                "failed_at": f["timestamp"],
                "error_type": f["error_type"],
                "error_feedback": (f["error_detail"] or "")[:500],
            },
            "rejected": rejected,
            "chosen": chosen,
            "meta": {
                "mined_from": "telemetry.db",
                "gap_s": round(success["timestamp"] - f["timestamp"], 1),
            },
        })

    pairs.extend(_mine_tool_as_shell(conn, window_s))
    conn.close()
    return pairs


# Builtin tool names a model might run as shell commands (D3: task_list,
# sessions_list, task_create observed live as `bash: … command not found`).
_TOOLISH_CMD_RE = re.compile(
    r"\b(task_list|task_create|task_get|task_stop|sessions_list|sessions_send|"
    r"cron_list|cron_create|web_search|tool_search|lcm_grep)\b"
)


def _mine_tool_as_shell(conn: sqlite3.Connection, window_s: float) -> list[dict]:
    """Cross-tool pairs: bash invoked a TOOL NAME as a shell command (exit
    127 'command not found'), then the real tool succeeded shortly after."""
    out: list[dict] = []
    failures = conn.execute(
        """
        SELECT timestamp, model, error_detail, parsed_tool_call
          FROM tool_calls
         WHERE tool_name = 'bash' AND success = 0
           AND error_detail LIKE '%command not found%'
           AND parsed_tool_call IS NOT NULL
         ORDER BY timestamp
        """
    ).fetchall()
    for f in failures:
        rejected = _parse_call(f["parsed_tool_call"])
        if rejected is None:
            continue
        m = _TOOLISH_CMD_RE.search(str(rejected["input"].get("command", "")))
        if not m:
            continue
        target_tool = m.group(1)
        success = conn.execute(
            """
            SELECT timestamp, parsed_tool_call FROM tool_calls
             WHERE tool_name = ? AND success = 1
               AND timestamp > ? AND timestamp <= ?
               AND parsed_tool_call IS NOT NULL
             ORDER BY timestamp LIMIT 1
            """,
            (target_tool, f["timestamp"], f["timestamp"] + window_s),
        ).fetchone()
        if success is None:
            continue
        chosen = _parse_call(success["parsed_tool_call"])
        if chosen is None:
            continue
        out.append({
            "pair_source": "self_correction",
            "model_id": f["model"],
            "tool_name": target_tool,
            "context": {
                "kind": "telemetry_only",
                "failed_at": f["timestamp"],
                "error_type": "tool_as_shell",
                "error_feedback": (f["error_detail"] or "")[:300],
            },
            "rejected": rejected,
            "chosen": chosen,
            "meta": {
                "mined_from": "telemetry.db",
                "mined_pattern": "tool_as_shell",
                "gap_s": round(success["timestamp"] - f["timestamp"], 1),
            },
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Mine self-correction pairs from telemetry history")
    ap.add_argument("--window", type=float, default=600.0, help="max seconds between failure and success")
    ap.add_argument("--commit", action="store_true", help="write pairs to training.db (default: dry-run)")
    ap.add_argument("--db", default=str(TELEMETRY_DB), help="telemetry db path")
    args = ap.parse_args()

    pairs = mine(args.window, Path(args.db).expanduser())

    by_tool: dict[str, int] = {}
    for p in pairs:
        by_tool[p["tool_name"]] = by_tool.get(p["tool_name"], 0) + 1
    print(f"Minable self-correction pairs (window {args.window:.0f}s): {len(pairs)}")
    for tool, n in sorted(by_tool.items(), key=lambda kv: -kv[1]):
        print(f"  {tool}: {n}")

    if not args.commit:
        print("\nDry run — pass --commit to write to training.db")
        if pairs:
            print("\nSample pair:")
            print(json.dumps(pairs[0], indent=2, default=str)[:800])
        return 0

    store = PairStore()
    written = sum(
        store.add_pair(
            pair_source=p["pair_source"],
            model_id=p["model_id"],
            tool_name=p["tool_name"],
            context=p["context"],
            rejected=p["rejected"],
            chosen=p["chosen"],
            meta=p["meta"],
        )
        for p in pairs
    )
    print(f"\nWritten: {written} new pairs ({len(pairs) - written} deduped)")
    print(json.dumps(store.stats(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
