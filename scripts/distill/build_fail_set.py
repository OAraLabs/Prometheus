#!/usr/bin/env python3
"""Rebuild the teacher's FAILED tool calls as golden-shape traces (distill Phase 2, fail side).

    nice -n 19 ionice -c3 python3 build_fail_set.py --make-lcm-snapshot
    nice -n 19 ionice -c3 python3 build_fail_set.py \\
        [--telemetry ~/distill/snapshot/telemetry.db] [--lcm ~/distill/snapshot/lcm.db] \\
        [--out ~/distill/phase2/fail_traces.jsonl] [--model qwen3.8-max]

Runs on the host that holds the stores, with Prometheus importable (the deployed
tree on ``PYTHONPATH``). Training data never leaves that host.

The harvested corpus holds only successful calls: a golden row requires that the
tool succeeded. The fail side of the Phase 2 judge check exists only as
``telemetry.db`` ``tool_calls`` rows with ``success = 0`` and no exported
context. This script gives each one the same treatment the exporter gives a
golden row, through the exporter's own code:

* the input half comes from ``sentinel.golden_trace_exporter.lcm_context_resolver``
  (same scan depth, shared-surface and group-member refusals, strictly-before
  rule, trailing-assistant trim, 12-message cap);
* the example is shaped by ``ToolCallTelemetry._build_example`` and redacted with
  the same ``redact_capture`` the exporter applies before writing.

The only additions are under ``_meta``: ``label: "fail"``, the telemetry row it
came from and the recorded failure (``error_type``, ``error_detail``,
``result_summary``; all redacted at capture and again here).

Read-only by construction. Both stores are opened ``mode=ro`` with
``query_only`` on, and they must be SNAPSHOTS: a path inside the live
Prometheus config directory is refused. ``--make-lcm-snapshot`` copies the live
``lcm.db`` with the SQLite backup API from a read-only connection (no write, no
checkpoint, no lock on the source) and turns the copy's journal mode to DELETE
so later read-only opens need no side files. Its destination (``--lcm``) is
checked before anything is written: one inside the live config directory, or
the source itself, is refused. The LCM store object is built
without its constructor, which would apply the schema (a write).

Outputs (mode 0600): ``fail_traces.jsonl`` and ``fail_traces.summary.json``.
The summary holds counts only (rows read, rebuilt, skipped by reason) and is the
only file meant to be copied off the host.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

TEACHER_MODEL = "qwen3.8-max"
DEFAULT_SNAPSHOT_DIR = Path("~/distill/snapshot")
DEFAULT_OUT = Path("~/distill/phase2/fail_traces.jsonl")
# Never a training row: the loop's own bookkeeping (see the Phase 1 audit).
LOOP_TRANSITION_TOOL = "_loop_transition"
# Longest recorded error text carried into _meta. The judge only needs the gist.
FAILURE_TEXT_CHARS = 2000

# Columns read per failed row: the exporter's own golden columns, then the
# failure facts. Kept in this order so the head maps onto _GOLDEN_COLUMNS.
_EXTRA_COLUMNS = ("error_type", "error_detail", "result_summary", "turn_id", "retries")


class FailSetError(RuntimeError):
    """A refusal: the run would read a live store or has nothing to read."""


# ---------------------------------------------------------------------------
# Read-only stores
# ---------------------------------------------------------------------------


def live_config_dir() -> Path:
    """Where the live Prometheus stores are, asked of Prometheus itself (creates nothing)."""
    try:
        from prometheus.config.paths import config_dir_path

        return Path(config_dir_path()).expanduser().resolve()
    except Exception:  # noqa: BLE001 - an older tree: fall back to the documented default
        return Path(os.environ.get("PROMETHEUS_CONFIG_DIR") or "~/.prometheus").expanduser().resolve()


def refuse_live(path: Path, *, live_dir: Path | None = None) -> Path:
    """Return *path* resolved, or refuse when it points inside the live config directory."""
    resolved = Path(path).expanduser().resolve()
    live = live_dir if live_dir is not None else live_config_dir()
    if resolved == live or live in resolved.parents:
        raise FailSetError(
            f"{resolved} is a live store under {live}; point this at a snapshot "
            "(see --make-lcm-snapshot)"
        )
    if not resolved.is_file():
        raise FailSetError(f"{resolved} does not exist")
    return resolved


def open_ro(path: Path) -> sqlite3.Connection:
    """A read-only connection: ``mode=ro`` on the URI, and ``query_only`` as a second lock."""
    conn = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def refuse_snapshot_target(src: Path, dst: Path, *, live_dir: Path | None = None) -> Path:
    """Return *dst* resolved, or refuse when writing it could replace a live store.

    Checked before anything is written: the copy goes to ``<dst>.tmp`` and is
    renamed over *dst*, so a *dst* inside the live config directory (or the
    source itself) would swap a new file in under the running daemon.
    """
    src = Path(src).expanduser().resolve()
    target = Path(dst).expanduser().resolve()
    live = Path(live_dir).expanduser().resolve() if live_dir is not None else live_config_dir()
    if target == src or (target.exists() and src.exists() and os.path.samefile(target, src)):
        raise FailSetError(f"{target} is the database being copied; give the snapshot its own path")
    if target == live or live in target.parents:
        raise FailSetError(
            f"{target} is under the live config directory {live}; snapshots go elsewhere "
            f"(default {DEFAULT_SNAPSHOT_DIR})"
        )
    return target


def snapshot_sqlite(src: Path, dst: Path, *, live_dir: Path | None = None) -> int:
    """Copy *src* to *dst* with the backup API from a read-only connection.

    The source database and its WAL are never written, checkpointed or locked
    for writing; the copy includes what is still only in the WAL. (Reading a
    WAL database needs its ``-shm`` index: a live daemon already has one open,
    and SQLite creates it, with an empty ``-wal``, when nothing does.) The copy
    is written beside *dst* and renamed into place, then switched to journal
    mode DELETE so a ``mode=ro`` open of it needs no side files. A *dst* inside
    the live config directory, or the source itself, is refused before any write.
    Returns the snapshot's size in bytes.
    """
    src = Path(src).expanduser().resolve()
    dst = refuse_snapshot_target(src, dst, live_dir=live_dir)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.unlink(missing_ok=True)
    source = open_ro(src)
    target = sqlite3.connect(str(tmp))
    try:
        source.backup(target, pages=4096, sleep=0.05)
        target.execute("PRAGMA journal_mode = DELETE")
        target.commit()
    finally:
        target.close()
        source.close()
    os.chmod(tmp, 0o600)
    tmp.replace(dst)
    return dst.stat().st_size


def readonly_lcm_store(path: Path) -> Any:
    """An ``LCMConversationStore`` over a read-only connection.

    Built without ``__init__`` on purpose: the constructor applies the schema
    and migrations, which are writes. Only the read path the resolver uses
    (``messages_before`` → ``_row_to_message``) is exercised.
    """
    from prometheus.memory.lcm_conversation_store import LCMConversationStore

    store = LCMConversationStore.__new__(LCMConversationStore)
    conn = open_ro(path)
    conn.row_factory = sqlite3.Row
    store._db_path = Path(path)
    store._conn = conn
    return store


# ---------------------------------------------------------------------------
# Rows and examples
# ---------------------------------------------------------------------------


def failed_rows(conn: sqlite3.Connection, model: str) -> list[dict[str, Any]]:
    """Every failed call *model* made, oldest first, as golden-row dicts plus failure facts."""
    from prometheus.telemetry.tracker import _GOLDEN_COLUMNS

    cols = (*_GOLDEN_COLUMNS, *_EXTRA_COLUMNS)
    rows = conn.execute(
        f"SELECT {', '.join(cols)} FROM tool_calls"  # noqa: S608 - column names are constants
        " WHERE model = ? AND success = 0 AND tool_name <> ? ORDER BY rowid",
        (model, LOOP_TRANSITION_TOOL),
    ).fetchall()
    return [dict(zip(cols, row)) for row in rows]


def call_problem(row: dict[str, Any]) -> str | None:
    """Why ``_build_example`` would refuse the CALL half, or None when it is usable."""
    raw = row.get("parsed_tool_call")
    if not raw:
        return "no_parsed_call"
    try:
        call = json.loads(raw)
        json.dumps(call.get("input", {}))
    except (TypeError, ValueError, AttributeError):
        return "unparseable_call"
    return None


def context_problem(row: dict[str, Any], store: Any) -> str:
    """Why the resolver returned no context. Diagnosis only; the resolver decides.

    Mirrors the resolver's own refusals in its order, then splits "nothing came
    back" into no rows at all versus rows that all drop out (tool rows with no
    flat text, or only the call's own assistant preamble).
    """
    from prometheus.sentinel.golden_trace_exporter import CONTEXT_SCAN_ROWS, SHARED_SURFACE_IDS

    session_id = row.get("session_id")
    if not session_id or row.get("timestamp") is None:
        return "no_session_id"
    if session_id in SHARED_SURFACE_IDS:
        return "shared_surface_session"
    try:
        messages = store.messages_before(session_id, row["timestamp"], limit=CONTEXT_SCAN_ROWS)
    except Exception:  # noqa: BLE001 - the resolver treats a failed read the same way
        return "lcm_read_error"
    if not messages:
        return "no_conversation_rows_before_call"
    if any(getattr(m, "provenance", "user") == "member" for m in messages):
        return "group_member_conversation"
    return "no_usable_context"


def _clip(text: Any, limit: int = FAILURE_TEXT_CHARS) -> str | None:
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= limit else text[:limit] + " [truncated]"


def build_fail_example(row: dict[str, Any], resolver: Any) -> tuple[dict[str, Any] | None, list]:
    """``(example, context)`` for one failed row, through the exporter's own code.

    The example is None when ``_build_example`` refuses it (no call, no context).
    """
    from prometheus.security.log_redaction import redact_capture
    from prometheus.telemetry.tracker import ToolCallTelemetry

    try:
        context = resolver(row) or []
    except Exception:  # noqa: BLE001 - the exporter's writer swallows the same way
        context = []
    example = ToolCallTelemetry._build_example(row, context)
    if example is None:
        return None, context
    example["_meta"].update({
        "label": "fail",
        "source": "telemetry-failed-call",
        "telemetry_rowid": row.get("rowid"),
        "turn_id": row.get("turn_id"),
        "failure": {
            "error_type": row.get("error_type"),
            "error_detail": _clip(row.get("error_detail")),
            "result_summary": _clip(row.get("result_summary")),
        },
    })
    # The exporter redacts the whole example right before writing it: the
    # resolver adds LCM text that was never redacted at capture.
    return redact_capture(example), context


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return ordered[k]


def build_fail_set(
    telemetry_path: Path,
    lcm_path: Path,
    out_path: Path,
    *,
    model: str = TEACHER_MODEL,
    live_dir: Path | None = None,
) -> dict[str, Any]:
    """Rebuild every failed *model* call from the two snapshots. Returns the summary."""
    from prometheus.sentinel.golden_trace_exporter import lcm_context_resolver

    telemetry_path = refuse_live(telemetry_path, live_dir=live_dir)
    lcm_path = refuse_live(lcm_path, live_dir=live_dir)
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tconn = open_ro(telemetry_path)
    try:
        rows = failed_rows(tconn, model)
    finally:
        tconn.close()
    store = readonly_lcm_store(lcm_path)
    resolver = lcm_context_resolver(store)

    skipped: collections.Counter[str] = collections.Counter()
    by_tool: collections.Counter[str] = collections.Counter()
    by_error: collections.Counter[str] = collections.Counter()
    ctx_lens: list[int] = []
    sessions: set[str] = set()
    written = 0
    tmp = out_path.with_name(out_path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            os.chmod(tmp, 0o600)
            for row in rows:
                problem = call_problem(row)
                if problem:
                    skipped[problem] += 1
                    continue
                example, context = build_fail_example(row, resolver)
                if example is None:
                    skipped[context_problem(row, store) if not context else "exporter_refused"] += 1
                    continue
                fh.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1
                by_tool[row["tool_name"]] += 1
                by_error[row.get("error_type") or "none recorded"] += 1
                ctx_lens.append(len(context))
                sessions.add(row["session_id"])
        tmp.replace(out_path)
    finally:
        tmp.unlink(missing_ok=True)
        store._conn.close()

    summary: dict[str, Any] = {
        "schema": "distill-fail-set-summary/1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": model,
        "failed_rows": len(rows),
        "rebuilt": written,
        "skipped": sum(skipped.values()),
        "skipped_by_reason": dict(skipped.most_common()),
        "rebuilt_sessions": len(sessions),
        "rebuilt_by_tool": dict(by_tool.most_common()),
        "rebuilt_by_error_type": dict(by_error.most_common()),
        "context_messages": {
            "p50": _percentile(ctx_lens, 50), "p90": _percentile(ctx_lens, 90),
            "max": max(ctx_lens, default=0),
        },
    }
    summary_path = out_path.with_name(out_path.stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    os.chmod(summary_path, 0o600)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--telemetry", type=Path, default=DEFAULT_SNAPSHOT_DIR / "telemetry.db",
                    help="telemetry.db SNAPSHOT (default: %(default)s)")
    ap.add_argument("--lcm", type=Path, default=DEFAULT_SNAPSHOT_DIR / "lcm.db",
                    help="lcm.db SNAPSHOT (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output JSONL (default: %(default)s)")
    ap.add_argument("--model", default=TEACHER_MODEL, help="teacher model name (default: %(default)s)")
    ap.add_argument("--make-lcm-snapshot", action="store_true",
                    help="first copy the live lcm.db into --lcm (backup API, read-only source)")
    ap.add_argument("--live-lcm", type=Path, default=None,
                    help="live lcm.db to snapshot (default: Prometheus's own data path)")
    args = ap.parse_args(argv)

    try:
        if args.make_lcm_snapshot:
            src = args.live_lcm
            if src is None:
                from prometheus.config.paths import config_dir_path

                src = Path(config_dir_path()) / "data" / "lcm.db"
            size = snapshot_sqlite(src, args.lcm)
            print(f"lcm snapshot: {size} bytes written to {Path(args.lcm).expanduser()}")
        summary = build_fail_set(args.telemetry, args.lcm, args.out, model=args.model)
    except FailSetError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
