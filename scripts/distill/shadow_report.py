#!/usr/bin/env python3
"""Per-category agreement between the primary and the shadow model, from shadow mirroring's rows.

    uv run python scripts/distill/shadow_report.py                       # <data dir>/shadow
    uv run python scripts/distill/shadow_report.py --dir data/distill/shadow --since 2026-11-01 \\
        --json-out data/distill/shadow/summary.json --markdown-out data/distill/shadow/summary.md
    uv run python scripts/distill/shadow_report.py --judge mypackage.judges:equivalent

Reads the ``shadow-<UTC date>.jsonl`` files the daemon writes while ``shadow.enabled`` is on
(``src/prometheus/engine/shadow.py``, rows of schema ``shadow-row/1``), prints a markdown table per
request category, and lists the categories whose agreement reaches ``--threshold``. Read-only: it
calls no model and changes no routing.

Agreement, per row
------------------
* A row the PRIMARY failed is not comparable (there is nothing to agree with); it is only counted.
* A row the SHADOW failed (an error or a timeout) counts as a DISAGREEMENT: a student that did not
  answer did not answer the same way. ``--exclude-shadow-errors`` drops those rows instead, to tell
  a box that was down from a model that is wrong.
* The primary called tools: agree = the same tools in the same order AND equivalent arguments. The
  argument verdict is the row's own ``args_equivalent``, computed by the daemon from the full
  arguments (a row keeps large arguments only as a size and a hash).
* The primary answered: agree = the shadow answered too (no tool call) AND the judge says the two
  answers are equivalent.

The judge is pluggable: ``--judge module:function`` names a callable
``judge(primary_text, shadow_text, row) -> bool``. The default is normalized exact match
(``prometheus.engine.shadow.normalize_answer_text``: case, whitespace, markup and a trailing full stop
ignored), because the runbook's model judge cannot be called on the current key
(``docs/sprints/DISTILL-OPEN-QUESTIONS.md`` #6). It is strict, so the answer agreement it reports is
a floor: two right answers worded differently count as a disagreement.

A category is "primary-equivalent" when its agreement is at least ``--threshold`` (0.95) over at
least ``--min-rows`` comparable rows (30): fewer rows than that cannot tell 95% from 80%. Those are
the candidates to route locally later (the runbook calls them "Max-equivalent" when the primary is
the teacher). The Wilson 95% interval is printed beside each rate.

Exit status: 0 on a report (whatever it says), 2 when the input or the judge cannot be read.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from prometheus.engine.shadow import CATEGORIES, ROW_SCHEMA, normalize_answer_text
from prometheus.gym.ladder.record import wilson

REPORT_SCHEMA = "shadow-report/1"
DEFAULT_THRESHOLD = 0.95
DEFAULT_MIN_ROWS = 30

Judge = Callable[[str, str, dict[str, Any]], bool]


def normalized_match(primary_text: str, shadow_text: str, row: dict[str, Any]) -> bool:
    """The default judge: the two answers are the same text once normalized."""
    return normalize_answer_text(primary_text) == normalize_answer_text(shadow_text)


def load_judge(spec: str | None) -> tuple[Judge, str]:
    """``module:function`` -> the callable, and the name the report records for it."""
    if not spec:
        return normalized_match, "normalized-match"
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"--judge must be module:function, got {spec!r}")
    judge = getattr(importlib.import_module(module_name), attr)
    if not callable(judge):
        raise ValueError(f"--judge {spec!r} is not callable")
    return judge, spec


# ── reading rows ───────────────────────────────────────────────────────────────────────────


def row_files(paths: Iterable[str | Path]) -> list[Path]:
    """Every row file under the given files and directories, in name (= date) order."""
    out: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            out.extend(sorted(p.glob("shadow-*.jsonl")))
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"no such file or directory: {p}")
    return out


def read_rows(files: Iterable[Path], *, since: str | None = None, until: str | None = None,
              primary_model: str | None = None, shadow_model: str | None = None,
              tally: Counter[str] | None = None) -> list[dict[str, Any]]:
    """Rows from ``files``, filtered. A torn or foreign line is counted in ``tally``, never fatal.

    ``since`` / ``until`` are inclusive UTC dates (YYYY-MM-DD) compared with the row's ``ts``.
    """
    tally = tally if tally is not None else Counter()
    rows: list[dict[str, Any]] = []
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    tally["malformed_lines"] += 1
                    continue
                if (not isinstance(row, dict) or row.get("schema") != ROW_SCHEMA
                        or not isinstance(row.get("primary"), dict)
                        or not isinstance(row.get("shadow"), dict)):
                    tally["foreign_rows"] += 1
                    continue
                day = str(row.get("ts", ""))[:10]
                if (since and day < since) or (until and day > until):
                    tally["outside_dates"] += 1
                    continue
                if primary_model and row["primary"].get("model") != primary_model:
                    tally["other_primary_model"] += 1
                    continue
                if shadow_model and row["shadow"].get("model") != shadow_model:
                    tally["other_shadow_model"] += 1
                    continue
                rows.append(row)
    return rows


# ── agreement ──────────────────────────────────────────────────────────────────────────────


def assess(row: dict[str, Any], judge: Judge, *, exclude_shadow_errors: bool = False) -> dict[str, Any]:
    """One row's verdict: is it comparable, what kind of round was it, and did the two agree?"""
    primary, shadow = row["primary"], row["shadow"]
    out: dict[str, Any] = {"comparable": False, "kind": None, "agree": None,
                           "same_tool": None, "args_equivalent": None, "answer_equivalent": None}
    if primary.get("error"):
        out["excluded"] = "primary_error"
        return out
    shadow_failed = bool(shadow.get("error"))
    if shadow_failed and exclude_shadow_errors:
        out["excluded"] = "shadow_error"
        return out
    out["comparable"] = True
    if primary.get("tools"):
        out["kind"] = "tool"
        same = not shadow_failed and row.get("same_tool") is True
        args = same and row.get("args_equivalent") is True
        out.update(same_tool=same, args_equivalent=args, agree=args)
    else:
        out["kind"] = "answer"
        answered = not shadow_failed and not shadow.get("tools")
        equivalent = answered and bool(judge(str(primary.get("text") or ""),
                                             str(shadow.get("text") or ""), row))
        out.update(same_tool=answered, answer_equivalent=equivalent, agree=equivalent)
    return out


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": [round(lo, 4), round(hi, 4)]}


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 1) if values else None


def _category_stats(rows: list[dict[str, Any]], verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [v for v in verdicts if v["comparable"]]
    tool = [v for v in comparable if v["kind"] == "tool"]
    answer = [v for v in comparable if v["kind"] == "answer"]
    ok = [r for r in rows if not r["primary"].get("error") and not r["shadow"].get("error")]
    return {
        "rows": len(rows),
        "primary_errors": sum(1 for r in rows if r["primary"].get("error")),
        "shadow_errors": sum(1 for r in rows if r["shadow"].get("error")),
        "comparable": len(comparable),
        "agreement": _rate(sum(1 for v in comparable if v["agree"]), len(comparable)),
        "same_tool": _rate(sum(1 for v in tool if v["same_tool"]), len(tool)),
        "args_equivalent": _rate(sum(1 for v in tool if v["args_equivalent"]), len(tool)),
        "answer_equivalent": _rate(sum(1 for v in answer if v["answer_equivalent"]), len(answer)),
        "primary_latency_ms_p50": _median([float(r["primary"].get("latency_ms") or 0) for r in ok]),
        "shadow_latency_ms_p50": _median([float(r["shadow"].get("latency_ms") or 0) for r in ok]),
    }


def summarize(rows: list[dict[str, Any]], judge: Judge, *, judge_name: str = "normalized-match",
              threshold: float = DEFAULT_THRESHOLD, min_rows: int = DEFAULT_MIN_ROWS,
              exclude_shadow_errors: bool = False) -> dict[str, Any]:
    """The report as data. Pure, so a heartbeat or a test can call it without the CLI."""
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_cat.setdefault(str(row.get("category") or "other"), []).append(row)
    order = [c for c in CATEGORIES if c in by_cat] + sorted(c for c in by_cat if c not in CATEGORIES)

    categories: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    all_verdicts: list[dict[str, Any]] = []
    for cat in order:
        verdicts = [assess(r, judge, exclude_shadow_errors=exclude_shadow_errors) for r in by_cat[cat]]
        stats = _category_stats(by_cat[cat], verdicts)
        rate = stats["agreement"]["rate"]
        stats["equivalent"] = bool(stats["comparable"] >= min_rows and rate is not None
                                   and rate >= threshold)
        categories[cat] = stats
        all_rows += by_cat[cat]
        all_verdicts += verdicts

    return {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge": judge_name,
        "threshold": threshold,
        "min_rows": min_rows,
        "exclude_shadow_errors": exclude_shadow_errors,
        "rows": len(rows),
        "first_ts": min((str(r.get("ts")) for r in rows), default=None),
        "last_ts": max((str(r.get("ts")) for r in rows), default=None),
        "primary_models": dict(Counter(str(r["primary"].get("model")) for r in rows)),
        "shadow_models": dict(Counter(str(r["shadow"].get("model")) for r in rows)),
        "categories": categories,
        "overall": _category_stats(all_rows, all_verdicts),
        "equivalent_categories": [c for c, s in categories.items() if s["equivalent"]],
    }


# ── rendering ──────────────────────────────────────────────────────────────────────────────


def _pct(r: dict[str, Any]) -> str:
    if not r["n"]:
        return "–"
    lo, hi = r["ci95"]
    return f"{r['rate']:.1%} ({r['k']}/{r['n']}; {lo:.0%}–{hi:.0%})"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Shadow agreement",
        "",
        f"{summary['rows']} rows, {summary['first_ts'] or '–'} to {summary['last_ts'] or '–'}. "
        f"Primary: {', '.join(f'{m} ({n})' for m, n in summary['primary_models'].items()) or '–'}. "
        f"Shadow: {', '.join(f'{m} ({n})' for m, n in summary['shadow_models'].items()) or '–'}.",
        f"Judge: `{summary['judge']}`. Shadow errors "
        f"{'excluded' if summary['exclude_shadow_errors'] else 'count as disagreement'}.",
        "",
        "| category | rows | comparable | agreement | same tool | args equivalent | answer equivalent "
        "| primary errors | shadow errors | p50 ms primary / shadow |",
        "|---|---:|---:|---|---|---|---|---:|---:|---|",
    ]
    for name, s in [*summary["categories"].items(), ("**all**", summary["overall"])]:
        lat = f"{s['primary_latency_ms_p50'] or '–'} / {s['shadow_latency_ms_p50'] or '–'}"
        lines.append(
            f"| {name} | {s['rows']} | {s['comparable']} | {_pct(s['agreement'])} "
            f"| {_pct(s['same_tool'])} | {_pct(s['args_equivalent'])} | {_pct(s['answer_equivalent'])} "
            f"| {s['primary_errors']} | {s['shadow_errors']} | {lat} |")
    lines.append("")
    equivalent = summary["equivalent_categories"]
    lines.append(
        f"**Primary-equivalent (agreement >= {summary['threshold']:.0%} over >= {summary['min_rows']} "
        f"comparable rows):** {', '.join(equivalent) if equivalent else 'none yet'}.")
    short = [c for c, s in summary["categories"].items() if s["comparable"] < summary["min_rows"]]
    if short:
        lines.append(f"Too few comparable rows to judge: {', '.join(short)}.")
    return "\n".join(lines) + "\n"


# ── CLI ────────────────────────────────────────────────────────────────────────────────────


def _default_dir() -> Path:
    from prometheus.config.paths import data_dir_path

    return data_dir_path() / "shadow"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    ap.add_argument("--dir", action="append", dest="paths", metavar="PATH",
                    help="a row file or a directory of shadow-*.jsonl (repeatable; default "
                         "<data dir>/shadow)")
    ap.add_argument("--since", help="first UTC date to include, YYYY-MM-DD")
    ap.add_argument("--until", help="last UTC date to include, YYYY-MM-DD")
    ap.add_argument("--primary-model", help="only rows whose primary was this model")
    ap.add_argument("--shadow-model", help="only rows whose shadow was this model")
    ap.add_argument("--judge", help="answer judge as module:function (default: normalized match)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    ap.add_argument("--exclude-shadow-errors", action="store_true",
                    help="drop rows the shadow failed instead of counting them as disagreements")
    ap.add_argument("--json-out", help="also write the summary as JSON here")
    ap.add_argument("--markdown-out", help="also write the markdown table here")
    args = ap.parse_args(argv)

    for flag in ("since", "until"):
        value = getattr(args, flag)
        if value:
            try:
                datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                print(f"error: --{flag} must be YYYY-MM-DD, got {value!r}", file=sys.stderr)
                return 2
    try:
        judge, judge_name = load_judge(args.judge)
    except (ImportError, AttributeError, ValueError) as exc:
        print(f"error: cannot load the judge: {exc}", file=sys.stderr)
        return 2
    try:
        files = row_files(args.paths or [_default_dir()])
        tally: Counter[str] = Counter()
        rows = read_rows(files, since=args.since, until=args.until,
                         primary_model=args.primary_model, shadow_model=args.shadow_model,
                         tally=tally)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = summarize(rows, judge, judge_name=judge_name, threshold=args.threshold,
                        min_rows=args.min_rows, exclude_shadow_errors=args.exclude_shadow_errors)
    summary["files"] = [str(f) for f in files]
    summary["skipped"] = dict(tally)
    md = render_markdown(summary)
    if tally:
        md += "Skipped: " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items())) + ".\n"
    print(md, end="")
    if args.markdown_out:
        Path(args.markdown_out).expanduser().write_text(md, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).expanduser().write_text(json.dumps(summary, indent=2) + "\n",
                                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
