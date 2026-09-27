"""Rows affected by each telemetry gap (docs/audits/TELEMETRY-GAPS.md), per window.

    python3 gaps.py <copy of telemetry.db> [--lcm <copy of lcm.db>] [--anchor EPOCH]

Run on the mini against backup-API copies (``backup_copy.py``). Every gap is a
row selector plus the class of each row (surface, provider/model); the scan
prints how many rows each selector matches in the last 14 and 30 days before
``--anchor`` (default: now), split by surface class and by provider/model, and
the all-time count where that says something the windows do not. Aggregates
only: no session ids, inputs, outputs or error text.

Surface of a row: its own session id when it has one; otherwise the class of
its nearest neighbour (``_common.Attributor``, the skill-usage audit's rule).
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import sqlite3
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

# error_type values written by the failure-path writers in engine/agent_loop.py
# (and malformed_empty by the providers' shared parser). None of them passes a
# session id; the main execution path writes None / nonzero_exit / tool_error.
FAILURE_TYPES = (
    "hook_blocked", "no_registry", "validation_failed", "unknown_tool", "template_markup",
    "input_validation", "permission_denied", "tool_timeout", "tool_exception", "malformed_empty",
)
MAIN_PATH_TYPES = (None, "nonzero_exit", "tool_error")
# Failure writers that record the call's input (parsed_tool_call) and those that do not.
WRITES_INPUT = ("validation_failed", "template_markup", "input_validation")
DROPS_INPUT = ("permission_denied", "hook_blocked", "tool_timeout", "tool_exception",
               "unknown_tool", "no_registry")
# Where a successful round's NULL cache count comes from, by provider. Evidence:
# wire_shapes.py over the recorded parity exchanges (llama.cpp, Ollama,
# Anthropic) and the parser's documented shape (qwen, xai).
CACHE_VERDICT = {
    "ollama (local)": "correct NULL: the wire carries no cache fields",
    "llama.cpp (local)": "defect: on the wire, provider never parses it, envelope drops it",
    "anthropic (cloud)": "defect: parsed, then dropped by the envelope",
    "qwen (cloud)": "defect: parsed (documented shape), then dropped by the envelope",
    "xai (cloud)": "defect: parsed (documented shape), then dropped by the envelope",
}


class Row:
    __slots__ = ("ts", "model", "surface", "extra")

    def __init__(self, ts: float, model: str | None, surface: str, extra: str = "") -> None:
        self.ts, self.model, self.surface, self.extra = ts, model, surface, extra


def report(gid: str, title: str, rows: list[Row], anchor: float, *,
           all_time: bool = False, by_extra: str | None = None) -> None:
    print(f"\n== {gid}: {title}")
    wc = C.window_counts((r.ts for r in rows), anchor)
    line = f"   rows: 14d {wc[14]:,}   30d {wc[30]:,}"
    if all_time:
        line += f"   all time {len(rows):,}"
    print(line)
    if not rows:
        return

    def split(key: Callable[[Row], str], header: str) -> None:
        c: collections.Counter = collections.Counter()
        for r in rows:
            for d in C.WINDOWS:
                if C.in_window(r.ts, anchor, d):
                    c[key(r), d] += 1
        keys = sorted({k for k, _ in c}, key=lambda k: (-c[k, 30], k))
        if keys:
            C.table([(k, f"{c[k, 14]:,}", f"{c[k, 30]:,}") for k in keys], (header, "14d", "30d"))

    split(lambda r: r.surface, "surface")
    split(lambda r: f"{C.provider_of(r.model)} / {C.model_label(r.model)}", "provider / model")
    if by_extra:
        split(lambda r: r.extra, by_extra)


def tool_rows(tel: sqlite3.Connection, where: str, params: Iterable = (), *,
              attribute: C.Attributor, extra: Callable[[sqlite3.Row], str] | None = None) -> list[Row]:
    out = []
    for r in tel.execute(f"SELECT timestamp, model, session_id, error_type, tool_name FROM tool_calls WHERE {where}",
                         tuple(params)):
        sid = r["session_id"]
        surf = C.surface(sid) if sid is not None else attribute(r["timestamp"], r["model"])
        out.append(Row(r["timestamp"], r["model"], surf, extra(r) if extra else ""))
    return out


def run_rows(tel: sqlite3.Connection, where: str, params: Iterable = (), *,
             attribute: C.Attributor, extra: Callable[[sqlite3.Row], str] | None = None) -> list[Row]:
    out = []
    for r in tel.execute(
        "SELECT timestamp, model, session_id, summary_json, billing_mode, outcome, operation"
        f" FROM subsystem_runs WHERE {where}", tuple(params)
    ):
        sid = r["session_id"]
        if sid == "web":
            surf = "routing namespace 'web' (not a conversation)"
        elif sid is not None:
            surf = C.surface(sid)
        else:
            surf = attribute(r["timestamp"], r["model"])
        out.append(Row(r["timestamp"], r["model"], surf, extra(r) if extra else ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("telemetry", type=Path)
    ap.add_argument("--lcm", type=Path)
    ap.add_argument("--anchor")
    ap.add_argument("--also-anchor", action="append", default=[], metavar="EPOCH",
                    help="extra anchor(s) for the 14-day cache count, e.g. a nightly snapshot's time")
    args = ap.parse_args()
    anchor = C.anchor_arg(args.anchor)
    tel = C.open_ro(args.telemetry)
    att = C.Attributor(tel)
    col_start = tel.execute(
        "SELECT MIN(timestamp) FROM tool_calls WHERE session_id IS NOT NULL").fetchone()[0]
    newest_call = tel.execute("SELECT MAX(timestamp) FROM tool_calls").fetchone()[0]
    newest_run = tel.execute("SELECT MAX(timestamp) FROM subsystem_runs").fetchone()[0]
    print(f"anchor {C.utc(anchor)}; windows 14d from {C.utc(anchor - 14 * C.DAY)}, "
          f"30d from {C.utc(anchor - 30 * C.DAY)}")
    print(f"newest tool call {C.utc(newest_call)}; newest subsystem_runs row {C.utc(newest_run)}; "
          f"tool_calls.session_id first written {C.utc(col_start)}")

    ph = ",".join("?" * len(FAILURE_TYPES))
    # ── A. session attribution ──────────────────────────────────────────
    t1 = tool_rows(tel, f"error_type IN ({ph}) AND session_id IS NULL AND timestamp >= ?",
                   (*FAILURE_TYPES, col_start), attribute=att, extra=lambda r: r["error_type"])
    report("T1", "failure-path tool_calls rows with no session id (since the column)", t1, anchor,
           all_time=True, by_extra="error_type")
    with_sid = tel.execute(f"SELECT COUNT(*) FROM tool_calls WHERE error_type IN ({ph}) AND session_id IS NOT NULL",
                           FAILURE_TYPES).fetchone()[0]
    print(f"   failure-path rows that DO carry a session id, all time: {with_sid}")

    t2 = tool_rows(tel, "error_type = 'lucky_guess'", attribute=att, extra=lambda r: r["tool_name"])
    report("T2", "lucky_guess marker rows (success=1, no session id; a second row for one call)", t2, anchor,
           all_time=True)
    for d in C.WINDOWS:
        succ = tel.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE success = 1 AND tool_name != '_loop_transition'"
            " AND timestamp BETWEEN ? AND ?", (anchor - d * C.DAY, anchor)).fetchone()[0]
        marks = sum(1 for r in t2 if C.in_window(r.ts, anchor, d))
        print(f"   {d}d: {marks} of {succ:,} 'successful calls' in every tool_calls reader are markers")

    t3 = tool_rows(tel, "tool_name = '_loop_transition' AND session_id IS NULL AND timestamp >= ?",
                   (col_start,), attribute=att, extra=lambda r: r["error_type"] or "tool_success")
    report("T3", "_loop_transition rows with no session id (since the column)", t3, anchor,
           all_time=True, by_extra="reason")

    ph2 = ",".join("?" * (len(MAIN_PATH_TYPES) - 1))
    main_null = list(tel.execute(
        "SELECT timestamp, model, tool_schema IS NULL AS no_schema FROM tool_calls WHERE session_id IS NULL"
        f" AND tool_name != '_loop_transition' AND (error_type IS NULL OR error_type IN ({ph2}))"
        " AND timestamp >= ?", (*MAIN_PATH_TYPES[1:], col_start)))
    older_build = [r for r in main_null if r["no_schema"]]
    current = [r for r in main_null if not r["no_schema"]]
    print("\n== T4: runs with no session id at all")
    print(f"   main-path tool_calls rows with no session id since the column: {len(main_null)}")
    print(f"     written by a build older than #209 (no tool_schema, which today's main path always fills): "
          f"{len(older_build)}, on {sorted({C.utc(r['timestamp'])[:10] for r in older_build})}")
    print(f"     written by current code: {len(current)}; windows {C.window_counts((r['timestamp'] for r in current), anchor)}")
    t4 = run_rows(tel, "subsystem = 'agent_loop' AND operation IN ('loop_round', 'tool_advertisement')"
                  " AND session_id IS NULL", attribute=att,
                  extra=lambda r: (r["operation"] + ": " + str((json.loads(r["summary_json"] or "{}")
                                                               ).get("source", "")))[:80])
    report("T4", "agent_loop rows (rounds, advertisements) with no session id", t4, anchor,
           all_time=True, by_extra="operation: advertisement source")

    # A microcompact row is written at the top of a round, before that round's
    # model call; the round's own loop_round row (same round_index, written when
    # the call returns) carries the run's real session. A row filed under "web"
    # whose round ran under a real session is the defect, certainly. One whose
    # round is ALSO under "web" is ambiguous: a conversation a client really named
    # "web" (LCM has one), or a round from before #458 (merged 2026-09-11), when
    # loop_round rows on the web path were filed under "web" too. So the defect
    # count is a floor.
    rounds_by_idx: dict[int, list[tuple[float, str | None]]] = collections.defaultdict(list)
    for r in tel.execute("SELECT timestamp, round_index, session_id FROM subsystem_runs WHERE"
                         " subsystem='agent_loop' AND operation='loop_round' ORDER BY timestamp"):
        rounds_by_idx[r["round_index"]].append((r["timestamp"], r["session_id"]))

    def filed_under(r: sqlite3.Row) -> str:
        sid = r["session_id"]
        if sid != "web":
            return "its own conversation"
        lst = rounds_by_idx.get(r["round_index"], [])
        i = bisect.bisect_left(lst, (r["timestamp"], ""))
        nxt = next((s for t, s in lst[i:i + 20] if t - r["timestamp"] <= 600), None)
        if nxt == "web":
            return "'web', and its round is too (ambiguous: see above)"
        return "'web', but its round ran under a real session (defect)"

    t5 = []
    for r in tel.execute("SELECT timestamp, model, session_id, round_index FROM subsystem_runs"
                         " WHERE subsystem = 'agent_loop' AND operation = 'microcompact'"):
        t5.append(Row(r["timestamp"], r["model"], C.surface(r["session_id"]) if r["session_id"] != "web"
                      else "routing namespace 'web'", filed_under(r)))
    report("T5", "microcompact rows, by the session id they were filed under", t5, anchor, all_time=True,
           by_extra="filed under")
    print(f"   all time, filed under 'web' with a real round: "
          f"{sum(1 for r in t5 if r.extra.endswith('(defect)'))}")

    # ── B. tokens and cost ──────────────────────────────────────────────
    ever = tel.execute("SELECT COUNT(*) FROM subsystem_runs WHERE cached_input_tokens IS NOT NULL"
                       " OR cache_write_tokens IS NOT NULL").fetchone()[0]
    first_round = tel.execute("SELECT MIN(timestamp) FROM subsystem_runs WHERE subsystem='agent_loop'"
                              " AND operation='loop_round'").fetchone()[0]
    t6 = run_rows(tel, "subsystem = 'agent_loop' AND operation = 'loop_round' AND outcome = 'success'"
                  " AND cached_input_tokens IS NULL", attribute=att,
                  extra=lambda r: CACHE_VERDICT.get(C.provider_of(r["model"]), "unclassified provider"))
    print(f"\n   rows with ANY cache count, whole table, all time: {ever}; first loop_round row {C.utc(first_round)}")
    report("T6", "successful rounds with no cache count (loop_round, cached_input_tokens NULL)", t6, anchor,
           by_extra="why NULL")

    t7 = [r for r in t6 if C.provider_of(r.model) == "llama.cpp (local)"]
    report("T7", "llama.cpp rounds whose cache count is on the wire but never parsed", t7, anchor)

    t8 = run_rows(tel, "subsystem = 'agent_loop' AND operation = 'loop_round'", attribute=att)
    t8 = [r for r in t8 if C.provider_of(r.model) == "anthropic (cloud)"]
    report("T8", "Anthropic rounds (input_tokens excludes cache reads and writes)", t8, anchor, all_time=True)

    bound = tel.execute("SELECT value FROM schema_meta WHERE key = 'billing_recorded_since'").fetchone()
    bound_ts = float(bound[0]) if bound else 0.0
    t9 = run_rows(tel, "billing_mode = 'unknown' AND timestamp >= ?", (bound_ts,), attribute=att,
                  extra=lambda r: f"{r['operation']}")
    print(f"\n   billing_mode recorded since {C.utc(bound_ts)}")
    report("T9", "rows stamped billing 'unknown' (since the stamp exists)", t9, anchor, all_time=True,
           by_extra="operation")

    t10 = run_rows(tel, "subsystem = 'context_compactor' AND input_tokens IS NULL", attribute=att,
                   extra=lambda r: f"{r['operation']} ({r['outcome']})")
    report("T10", "context-compactor model calls with no token counts and no session column", t10, anchor,
           all_time=True, by_extra="operation")

    print("\n== T11: turn-path model calls that write no telemetry row")
    for tool in ("vision",):
        n = [r["timestamp"] for r in tel.execute("SELECT timestamp FROM tool_calls WHERE tool_name = ?", (tool,))]
        print(f"   `{tool}` tool calls (one model call each): {C.window_counts(n, anchor)}, all time {len(n)}")
    if args.lcm:
        lcm = C.open_ro(args.lcm)
        summ = [r[0] for r in lcm.execute("SELECT created_at FROM lcm_summaries")]
        titles = [r[0] for r in lcm.execute("SELECT updated_at FROM session_titles WHERE updated_at IS NOT NULL")]
        print(f"   LCM summaries written (one summarizer call each): {C.window_counts(summ, anchor)}, all time {len(summ)}")
        print(f"   session titles last written (generated or renamed; an upper bound on title calls): "
              f"{C.window_counts(titles, anchor)}, all time {len(titles)}")
    else:
        print("   (pass --lcm for the LCM summary and session-title counts)")

    # ── C. model identity ───────────────────────────────────────────────
    t12 = tool_rows(tel, "tool_name != '_loop_transition' AND served_model IS NULL AND timestamp >= ?",
                    (col_start,), attribute=att)
    t12 = [r for r in t12 if C.provider_of(r.model) not in ("llama.cpp (local)",)]
    report("T12", "tool_calls rows from providers that never report served_model", t12, anchor)
    llama_null = tel.execute(
        "SELECT COUNT(*) FROM tool_calls WHERE tool_name != '_loop_transition' AND served_model IS NULL"
        " AND timestamp >= ?", (anchor - 30 * C.DAY,)).fetchone()[0] - sum(
        1 for r in t12 if C.in_window(r.ts, anchor, 30))
    print(f"   llama.cpp rows with served_model NULL in 30d (the only provider that reports it): {llama_null}")

    t13 = run_rows(tel, "subsystem = 'agent_loop' AND operation = 'loop_round' AND (model IS NULL OR model = '')",
                   attribute=att)
    report("T13", "loop_round rows with a blank model name and no served model", t13, anchor, all_time=True)

    crossed = 0
    for r in tel.execute("SELECT model, served_model FROM tool_calls WHERE served_model IS NOT NULL"):
        if "cloud" in C.provider_of(r["model"]) and C.is_path_model(r["served_model"]):
            crossed += 1
    print("\n== T14: fallback-served rounds recorded under the model that failed")
    print(f"   tool_calls rows naming a cloud model but served by a local file, all time: {crossed}")

    t15 = tool_rows(tel, f"error_type IN ({','.join('?' * len(DROPS_INPUT))}) AND parsed_tool_call IS NULL",
                    DROPS_INPUT, attribute=att, extra=lambda r: r["error_type"])
    report("T15", "failure rows that drop the call's input (parsed_tool_call NULL)", t15, anchor,
           all_time=True, by_extra="error_type")
    kept = tel.execute(f"SELECT COUNT(*) FROM tool_calls WHERE error_type IN ({','.join('?' * len(WRITES_INPUT))})"
                       " AND parsed_tool_call IS NOT NULL", WRITES_INPUT).fetchone()[0]
    print(f"   for contrast, failure rows from the writers that keep it: {kept:,} with an input")

    # ── the two numbers in the brief ────────────────────────────────────
    print("\n== reconciliations")
    calls = tel.execute("SELECT COUNT(*), SUM(session_id IS NULL) FROM tool_calls WHERE tool_name != '_loop_transition'"
                        " AND timestamp >= ?", (col_start,)).fetchone()
    print(f"   tool calls since the column: {calls[0]:,}, with no session id: {calls[1]:,} "
          f"(= T1 {len(t1)} + T2 {sum(1 for r in t2 if r.ts >= col_start)} + T4 main-path {len(main_null)})")
    g = tel.execute("SELECT COUNT(*), SUM(session_id IS NULL), SUM(session_id IS NULL AND timestamp < ?)"
                    " FROM tool_calls WHERE is_golden = 1", (col_start,)).fetchone()
    print(f"   golden rows: {g[0]:,}; with no session id: {g[1]:,}, of which older than the column: {g[2]:,}")
    anchors = [("anchor", anchor), ("newest loop_round", tel.execute(
        "SELECT MAX(timestamp) FROM subsystem_runs WHERE operation = 'loop_round'").fetchone()[0])]
    anchors += [("extra anchor", float(a)) for a in args.also_anchor]
    for label, a in anchors:
        n = tel.execute("SELECT COUNT(*) FROM subsystem_runs WHERE subsystem = 'agent_loop' AND operation = 'loop_round'"
                        " AND outcome = 'success' AND cached_input_tokens IS NULL AND timestamp BETWEEN ? AND ?",
                        (a - 14 * C.DAY, a)).fetchone()[0]
        print(f"   successful rounds with no cache count, 14 days before the {label} ({C.utc(a)}): {n:,}")


if __name__ == "__main__":
    main()
