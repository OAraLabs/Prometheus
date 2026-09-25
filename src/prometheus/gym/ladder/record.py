"""Recording ladder runs in telemetry.db, and reading them back.

One ladder run leaves three kinds of rows in the SAME telemetry database,
all carrying the run's ``session_id``:

* ``tool_calls`` — one per tool call, written by the real agent loop
  (success, retries, adapter repairs, served model);
* ``subsystem_runs`` with ``subsystem='agent_loop'`` — one per model round,
  written by the loop's LLMCallEnvelope (input/output tokens);
* ``subsystem_runs`` with ``subsystem='model_ladder'`` — the run's summary,
  written here: ``operation`` = task class, ``outcome`` = verdict,
  ``duration_ms`` = wall time, ``input_tokens``/``output_tokens`` = the sum
  over rounds, ``model``, ``session_id``; everything the schema has no column
  for (quantization, adapter tier, tool-call success, repairs, rounds, ...)
  goes in ``summary_json``. See docs/MODEL-LADDER.md, "Where each field lives".

The schema is not changed. The summary row is written through
``ToolCallTelemetry.record_run`` — which swallows write errors by design — so
every write is READ BACK, and a row that did not land raises.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from typing import Any

from prometheus.telemetry.tracker import (
    NON_CALL_FAILURE_TYPES,
    POLICY_ERROR_TYPES,
    SYNTHETIC_TOOL_NAME,
)

SUBSYSTEM = "model_ladder"

# The WP-2.1 field list. `check_empty_fields` asserts none of these is empty
# in EVERY row of a run — a field empty everywhere is a pipeline gap, not a
# property of the tasks.
REQUIRED_FIELDS: tuple[str, ...] = (
    "task_class",
    "model",
    "quantization",
    "adapter_tier",
    "adapter_strictness",
    "success",
    "tool_call_success",
    "repairs",
    "rounds",
    "input_tokens",
    "output_tokens",
    "duration_ms",
)

_EMPTY = (None, "", "unknown", "unreported")


class LadderRecordError(RuntimeError):
    """A summary row did not land in telemetry.db."""


def harvest_run_metrics(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    window: tuple[float, float] | None = None,
    transcript: Any = None,
) -> dict[str, Any]:
    """Tool-call and round metrics for one run, from the pipeline's own rows.

    Some loop paths write ``tool_calls`` rows with NO session_id — measured:
    every permission-denied call. Those are attributed to this run when they
    fall inside its time *window* (runs are sequential, and the default
    ladder DB has no other writer), and counted in ``tool_calls_unattributed``
    so a reader knows which rows were joined by time rather than by id.
    ``window=None`` turns that attribution off (the runner does, when the
    target is the live daemon's telemetry.db, which has other writers).

    Two things the telemetry rows do not see come from the *transcript*:
    calls the loop's repeat guard answered ``BLOCKED`` (it writes no row for
    them — counted here as failed calls), and adapter repairs (``repairs`` is
    the number of calls whose executed form differed from what the model
    emitted; the loop records per-call repair counts only on some paths, kept
    as ``repair_ops_telemetry``).

    ``tool_call_success`` follows telemetry's denominator rule: policy denials
    and non-zero exits are well-formed calls, so they are counted but kept out
    of the denominator (see NON_CALL_FAILURE_TYPES). NULL when the run made no
    tool calls — undefined, not zero.

    Tokens: a provider that does not report usage leaves 0 on every round.
    A real round always has input tokens, so an all-zero sum is recorded as
    NULL with ``tokens_source='unreported'`` — never as a measured 0.
    """
    calls = conn.execute(
        "SELECT success, retries, repairs, error_type, served_model FROM tool_calls "
        "WHERE session_id = ? AND tool_name != ?",
        (session_id, SYNTHETIC_TOOL_NAME),
    ).fetchall()
    unattributed = []
    if window is not None:
        unattributed = conn.execute(
            "SELECT success, retries, repairs, error_type, served_model FROM tool_calls "
            "WHERE (session_id IS NULL OR session_id = '') AND tool_name != ? "
            "AND timestamp >= ? AND timestamp <= ?",
            (SYNTHETIC_TOOL_NAME, window[0], window[1]),
        ).fetchall()
        calls = list(calls) + list(unattributed)
    blocked = 0
    repaired = 0
    if transcript is not None:
        blocked = sum(
            1 for e in transcript.tool_events
            if e.is_error and str(e.result_content or "").startswith("BLOCKED: this exact ")
        )
        repaired = sum(1 for e in transcript.tool_events if e.repaired)
    n = len(calls) + blocked
    ok = sum(1 for c in calls if c[0])
    excluded = sum(1 for c in calls if not c[0] and c[3] in NON_CALL_FAILURE_TYPES)
    denom = n - excluded
    served = sorted({c[4] for c in calls if c[4]})

    # operation='loop_round' only: the loop also files per-turn rows under
    # subsystem 'agent_loop' (tool_advertisement, breaker trip/halt, ...)
    # that are not model calls.
    rounds = conn.execute(
        "SELECT input_tokens, output_tokens, thinking FROM subsystem_runs "
        "WHERE subsystem = 'agent_loop' AND operation = 'loop_round' AND session_id = ?",
        (session_id,),
    ).fetchall()
    tin = sum(r[0] or 0 for r in rounds)
    tout = sum(r[1] or 0 for r in rounds)
    reported = tin > 0
    thinking = {r[2] for r in rounds if r[2] is not None}
    return {
        "tool_calls": n,
        "tool_calls_ok": ok,
        "tool_calls_excluded": excluded,
        "tool_calls_denied": sum(1 for c in calls if c[3] in POLICY_ERROR_TYPES),
        "tool_calls_unattributed": len(unattributed),
        "tool_call_success": round(ok / denom, 4) if denom > 0 else None,
        "tool_calls_blocked": blocked,
        "repairs": repaired if transcript is not None else sum(c[2] or 0 for c in calls),
        "repair_ops_telemetry": sum(c[2] or 0 for c in calls),
        "retries": sum(c[1] or 0 for c in calls),
        "dropped_malformed": sum(1 for c in calls if c[3] == "malformed_empty"),
        "served_models": served,
        "rounds": len(rounds) or None,
        "input_tokens": tin if reported else None,
        "output_tokens": tout if reported else None,
        "tokens_source": "provider" if reported else "unreported",
        "thinking": (thinking.pop() == 1) if len(thinking) == 1 else None,
    }


def final_round_used_reasoning_fallback(
    conn: sqlite3.Connection, session_id: str, window: tuple[float, float]
) -> bool:
    """Did the run's LAST model round answer with its reasoning channel?

    When a thinking model spends its whole output budget reasoning, the
    llama.cpp provider returns the unfinished reasoning as the reply and files
    a ``silent_failures`` row (``used_reasoning_fallback: true``) through the
    telemetry handle the ladder installs. That row has no session id, so it is
    placed by time: the final round is the interval between the run's last two
    ``loop_round`` rows (or the run's start).
    """
    rounds = [r[0] for r in conn.execute(
        "SELECT timestamp FROM subsystem_runs WHERE subsystem = 'agent_loop' "
        "AND operation = 'loop_round' AND session_id = ? ORDER BY timestamp",
        (session_id,),
    )]
    if not rounds:
        return False
    lo = rounds[-2] if len(rounds) > 1 else window[0]
    hi = rounds[-1]
    for (ctx,) in conn.execute(
        "SELECT context FROM silent_failures WHERE timestamp > ? AND timestamp <= ?",
        (lo, hi),
    ):
        try:
            if json.loads(ctx or "{}").get("used_reasoning_fallback") is True:
                return True
        except (TypeError, ValueError):
            continue
    return False


def outcome_for(verdict: str) -> str:
    """subsystem_runs.outcome for a verdict: ``partial`` = the model finished
    but its answer could not be read (a format miss — never ``failed``);
    ``skipped`` = no verdict (unscored, error)."""
    return {"pass": "success", "fail": "failed", "format_miss": "partial"}.get(verdict, "skipped")


def record_summary(tel: Any, summary: dict[str, Any]) -> None:
    """Write the run's summary row and prove it landed."""
    # Tokens stay OUT of this row's input/output_tokens columns: the run's
    # loop_round rows already carry them, and usage_rollup (/api/usage) sums
    # every subsystem_runs row with tokens — a copy here would count each run
    # twice. The run's totals are in summary_json.
    tel.record_run(
        SUBSYSTEM,
        summary["task_class"],
        outcome_for(summary["verdict"]),
        duration_ms=summary["duration_ms"],
        summary=summary,
        session_id=summary["session_id"],
        model=summary["model"],
        thinking=summary.get("thinking"),
    )
    row = tel._conn.execute(
        "SELECT summary_json FROM subsystem_runs WHERE subsystem = ? AND session_id = ?",
        (SUBSYSTEM, summary["session_id"]),
    ).fetchone()
    if row is None or not row[0]:
        raise LadderRecordError(
            f"ladder summary for {summary['session_id']} did not land in "
            f"{tel.db_path} — refusing to continue a run that is not being recorded"
        )


def load_rows(conn: sqlite3.Connection, run_label: str) -> list[dict[str, Any]]:
    """Every summary row of one run label, flattened: columns + summary_json."""
    out = []
    for r in conn.execute(
        "SELECT timestamp, operation, outcome, duration_ms, input_tokens, "
        "output_tokens, session_id, model, node_id, summary_json FROM subsystem_runs "
        "WHERE subsystem = ? ORDER BY timestamp",
        (SUBSYSTEM,),
    ):
        summary = json.loads(r[9] or "{}")
        if summary.get("run_label") != run_label:
            continue
        summary.setdefault("node_id", r[8])
        summary["_columns"] = {
            "operation": r[1], "outcome": r[2], "duration_ms": r[3],
            "input_tokens": r[4], "output_tokens": r[5], "session_id": r[6],
            "model": r[7], "node_id": r[8],
        }
        out.append(summary)
    return out


def check_empty_fields(rows: list[dict[str, Any]]) -> list[str]:
    """Required fields that are empty in EVERY row. [] = the check passes.

    Empty per row is fine where the field is undefined for that task (a Q&A
    answered without tools has no tool-call success). Empty in every row is a
    recording gap and fails the check.
    """
    if not rows:
        return list(REQUIRED_FIELDS)
    return [f for f in REQUIRED_FIELDS if all(r.get(f) in _EMPTY for r in rows)]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _mean(xs: list[Any]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def _line_mark(ok: bool | None) -> str:
    return {True: "✓", False: "✗"}.get(ok, "—")  # type: ignore[arg-type]


def _fmt(x: Any, spec: str = ".0f", none: str = "—") -> str:
    return none if x is None else format(x, spec)


VERDICTS = ("pass", "fail", "format_miss", "unscored", "error")


def accuracy(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """(passes, decided) — decided = pass + fail. Format misses, unscored and
    errors are not answers the model got wrong, so they are not in it."""
    passed = sum(1 for r in rows if r["verdict"] == "pass")
    return passed, passed + sum(1 for r in rows if r["verdict"] == "fail")


def format_misses(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """(replies without an ANSWER line, replies that were asked for one)."""
    asked = [r for r in rows if r.get("answer_format_ok") is not None]
    return sum(1 for r in asked if r["answer_format_ok"] is False), len(asked)


def _ratio(n: int, d: int) -> str:
    return f"{n}/{d}" if d else "—"


def _class_row(cid: str, rs: list[dict[str, Any]]) -> str:
    counts = {v: sum(1 for r in rs if r["verdict"] == v) for v in VERDICTS}
    passed, decided = accuracy(rs)
    lo, hi = wilson(passed, decided)
    miss, asked = format_misses(rs)
    calls = sum(r.get("tool_calls") or 0 for r in rs)
    ok = sum(r.get("tool_calls_ok") or 0 for r in rs)
    excl = sum(r.get("tool_calls_excluded") or 0 for r in rs)
    tin, tout = _mean([r.get("input_tokens") for r in rs]), _mean([r.get("output_tokens") for r in rs])
    return (
        f"| {cid} | {len(rs)} | {counts['pass']} | {counts['fail']} | {counts['format_miss']} "
        f"| {counts['unscored']} | {counts['error']} | "
        + (f"{passed}/{decided} ({lo:.2f}–{hi:.2f})" if decided else "—")
        + f" | {_ratio(miss, asked)} | {_ratio(ok, calls - excl)} "
        f"| {sum(r.get('repairs') or 0 for r in rs)} "
        f"| {_fmt(_mean([r.get('rounds') for r in rs]), '.1f')} "
        f"| {_fmt(tin)} / {_fmt(tout)} "
        f"| {_fmt(_mean([r['duration_ms'] / 1000 for r in rs]), '.1f')} |"
    )


def render_ladder_table(
    rows_by_label: dict[str, list[dict[str, Any]]], *, class_order: list[str]
) -> str:
    """The cross-model table: each model's accuracy NEXT TO its format-miss
    rate, overall and per class — so a small model that answers right but
    skips the answer line is not read as failing those tasks."""
    classes = [c for c in class_order
               if any(r["task_class"] == c for rs in rows_by_label.values() for r in rs)]
    head = ("| model | quant | tier | accuracy (95% CI) | answer line missing | "
            + " | ".join(f"{c} acc / miss" for c in classes) + " |")
    sep = "|---|---|---|---|---:|" + "---|" * len(classes)
    out = ["# Model ladder", "",
           "Accuracy = pass ÷ (pass + fail). *answer line missing* = replies without their "
           "`ANSWER:` line ÷ replies asked for one; those are scored where the bare last line "
           "still gives the answer and otherwise recorded as format misses, never as wrong.",
           "", head, sep]
    for label, rs in rows_by_label.items():
        if not rs:
            continue
        first = rs[0]
        p, d = accuracy(rs)
        lo, hi = wilson(p, d)
        m, a = format_misses(rs)
        cells = []
        for c in classes:
            cr = [r for r in rs if r["task_class"] == c]
            cp, cd = accuracy(cr)
            cm, ca = format_misses(cr)
            cells.append(f"{_ratio(cp, cd)} / {_ratio(cm, ca)}")
        out.append(
            f"| {first.get('model')} (`{label}`) | {first.get('quantization')} "
            f"| {first.get('adapter_tier')} | "
            + (f"{p}/{d} ({lo:.2f}–{hi:.2f})" if d else "—")
            + f" | {_ratio(m, a)} | " + " | ".join(cells) + " |"
        )
    return "\n".join(out) + "\n"


def render_report(rows: list[dict[str, Any]], *, title: str, class_order: list[str]) -> str:
    """Markdown for one run label. Names no hosts or URLs — safe to commit."""
    if not rows:
        return f"# {title}\n\nNo rows recorded.\n"
    first = rows[0]
    judge = next((r["judge"] for r in rows if r.get("judge")), None)
    judge_prov = (judge or {}).get("provenance") or {}
    lines = [
        f"# {title}",
        "",
        f"- suite: `{first['suite']}` (sha `{first['suite_sha'][:12]}`), run label `{first['run_label']}`",
        f"- model: `{first['model']}` via `{first['provider']}`"
        + (f", served as `{', '.join(first['served_models'])}`" if first.get("served_models") else ""),
        f"- quantization: `{first.get('quantization')}` ({first.get('quantization_source')})",
        f"- adapter: tier `{first.get('adapter_tier')}`, base strictness `{first.get('adapter_strictness')}`",
        f"- KV cache: k={first.get('kv_cache', {}).get('k') or 'unknown'} "
        f"v={first.get('kv_cache', {}).get('v') or 'unknown'} "
        f"({first.get('kv_cache', {}).get('source')})",
        f"- judge: `{judge_prov.get('model') or first.get('judge_model') or 'none'}`"
        f" (pinned: {judge_prov.get('pinned', first.get('judge_model') is not None)})",
        f"- runs: {len(rows)}; live web: {'on' if first.get('live_web') else 'off'}",
        "",
        "## By class",
        "",
        "Accuracy counts only runs with a verdict on the answer (pass ÷ pass + fail). A reply",
        "without its `ANSWER:` line is a **format miss**: scored anyway when its bare last line",
        "still gives the answer (and counted in *answer line missing*), otherwise recorded as a",
        "`format_miss` verdict — never as a wrong answer.",
        "",
        "| class | runs | pass | fail | format miss | unscored | error | accuracy (95% CI) | answer line missing | tool-call success | repairs | rounds (mean) | tokens in / out (mean) | time s (mean) |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|",
    ]
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_class[r["task_class"]].append(r)
    ordered = [c for c in class_order if c in by_class] + sorted(set(by_class) - set(class_order))
    for cid in ordered + ["**all**"]:
        rs = rows if cid == "**all**" else by_class[cid]
        lines.append(_class_row(cid, rs))
    lines += [
        "",
        "## By difficulty (pass / decided)",
        "",
        "| class | easy | medium | hard |",
        "|---|---:|---:|---:|",
    ]
    for cid in [c for c in class_order if c in by_class] + sorted(set(by_class) - set(class_order)):
        cells = []
        for level in ("easy", "medium", "hard"):
            rs = [r for r in by_class[cid] if r.get("difficulty") == level]
            level_decided = [r for r in rs if r["verdict"] in ("pass", "fail")]
            level_passed = sum(1 for r in level_decided if r["verdict"] == "pass")
            cells.append(f"{level_passed}/{len(level_decided)}" if rs else "—")
        lines.append(f"| {cid} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Runs",
        "",
        "| task | class | difficulty | verdict | answer line | decided by | rounds | tool calls ok/all | repairs | tokens in / out | time s | why (if not pass) |",
        "|---|---|---|---|:---:|---|---:|---:|---:|---|---:|---|",
    ]
    from prometheus.gym.ladder.runner import redact

    for r in rows:
        why = (redact("; ".join(r.get("fail_reasons") or [])) or "")[:140]
        why = why.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {r['task_id']} | {r['task_class']} | {r.get('difficulty') or '—'} "
            f"| {r['verdict']} | {_line_mark(r.get('answer_format_ok'))} | {r['verdict_source']} "
            f"| {_fmt(r.get('rounds'))} | {r.get('tool_calls_ok', 0)}/{r.get('tool_calls', 0)} "
            f"| {r.get('repairs', 0)} | {_fmt(r.get('input_tokens'))} / {_fmt(r.get('output_tokens'))} "
            f"| {r['duration_ms'] / 1000:.1f} | {why} |"
        )
    empty = check_empty_fields(rows)
    lines += [
        "",
        "## Empty-field check",
        "",
        "Required fields: " + ", ".join(f"`{f}`" for f in REQUIRED_FIELDS) + ".",
        "",
        ("PASS — every required field is populated in at least one row."
         if not empty else
         "FAIL — empty in every row: " + ", ".join(f"`{f}`" for f in empty) + "."),
        "",
    ]
    per_field = []
    for f in REQUIRED_FIELDS:
        filled = sum(1 for r in rows if r.get(f) not in _EMPTY)
        per_field.append(f"| `{f}` | {filled}/{len(rows)} |")
    lines += ["| field | rows populated |", "|---|---:|", *per_field, ""]
    return "\n".join(lines)
