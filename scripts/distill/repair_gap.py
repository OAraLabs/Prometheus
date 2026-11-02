#!/usr/bin/env python3
"""The adapter-repair gap on eval-tools: raw scores vs scores after the live repair.

    PYTHONPATH=src python scripts/distill/repair_gap.py \\
        --results <dir>/eval-tools.jsonl --cases <evalsets>/eval-tools.jsonl \\
        --summary <dir>/eval-tools.summary.json [--json-out gap.json]

``score.py`` scores each reply's calls RAW: the calls the live loop reads off
the reply (llama-server's ``tool_calls``, else the adapter's text extraction),
before the adapter touches them. The daemon then runs every call through
``ModelAdapter.validate_and_repair`` before it executes it. This script scores
the same stored replies twice, with score.py's own verdict code and metric
definitions (task_success, tool_name_accuracy, arg_validity, arg_exactness):

raw          the calls exactly as score.py reads them. Every case's outcome
             and the four metrics must equal ``--summary`` (the score.py
             summary of the same rows), or the script exits 1: that is the
             check that both columns are scored the way score.py scores.
post-repair  each call after what the daemon's adapter at tier light does to
             it (``agent_loop``: ``adapter.validate_and_repair(name, input,
             registry)``). A call the adapter refuses (unknown name with no
             match within 3 edits, or an input that is not an object) keeps
             its raw form, which score.py already scores as invalid. A call
             whose arguments did not read as a JSON object never reaches the
             adapter (the provider names it unreadable and the loop refuses
             it), so it is left as it is too.

The adapter is ``ModelAdapter(tier="light")``, a fresh one per case (a cold
daemon: adaptive strictness needs 10 calls of one tool before it can raise
that tool's strictness, so a single case never moves it). At tier light the
validator's strictness is NONE, so the repairs that can apply are the
optional-null drop (every call), and, on a call that fails the invariants, a
fuzzy tool-name match (Levenshtein <= 3) and JSON extraction from a string
input. Coercion and unknown-parameter stripping are MEDIUM+ and never run
here. ``adapter.unwrap_dict_args`` (the loop's dict unwrap) is off by default
and not applied.

The registry is the case's own tools (every eval-tools case offers the same
tool list). Each tool's input model validates by the tool's JSON schema
(``jsonschema``, required) and raises pydantic's ``ValidationError`` with each
error at its top-level argument, which is all the adapter reads off a model
(``_drop_optional_nulls`` reads the refused locations and the schema's
``properties`` / ``required``).

Not modelled (reported, not applied): llama-server ``tool_calls`` entries with
an empty name, which the live provider drops before the loop sees them.

The report holds counts and case ids only, never reply or call text.
Exit codes: 0 ok, 1 the raw recompute differs from the summary, 2 bad input.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_SCORE_PATH = Path(__file__).resolve().with_name("score.py")


def _load_score() -> Any:
    """score.py as one module, shared with whoever loaded it first (the tests
    load it as ``distill_score``)."""
    mod = sys.modules.get("distill_score")
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location("distill_score", _SCORE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["distill_score"] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load_score()

METRICS = ("task_success", "tool_name_accuracy", "arg_validity", "arg_exactness")
VERDICT_KEYS = ("task_success", "tool_name_correct", "arg_valid", "arg_exact")
REPORT_SCHEMA = "distill-repair-gap/1"


# ---------------------------------------------------------------------------
# The registry the adapter sees, over a case's tools
# ---------------------------------------------------------------------------


class SchemaInputModel:
    """A tool input model that validates by the tool's JSON schema, the way
    the adapter reads a pydantic model: ``model_validate`` raises pydantic's
    ``ValidationError`` (each error at its top-level argument) and
    ``model_json_schema`` returns the schema."""

    def __init__(self, schema: dict[str, Any]) -> None:
        if S._jsonschema is None:
            raise RuntimeError("repair_gap needs the jsonschema package (the mcp extra)")
        self._schema = schema
        self._validator = S._jsonschema.validators.validator_for(schema)(schema)

    def model_json_schema(self) -> dict[str, Any]:
        return self._schema

    def model_validate(self, data: Any) -> Any:
        from pydantic_core import InitErrorDetails, PydanticCustomError, ValidationError

        errors = list(self._validator.iter_errors(data))
        if not errors:
            return data
        details: list[InitErrorDetails] = []
        for e in errors:
            path = list(e.absolute_path)
            details.append(InitErrorDetails(
                type=PydanticCustomError("json_schema", "{reason}", {"reason": e.message}),
                loc=tuple(path[:1]),
                input=data,
            ))
        raise ValidationError.from_exception_data("arguments", details)


class _RepairTool:
    def __init__(self, name: str, description: str, schema: dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.input_model = SchemaInputModel(schema)
        self.example_call = None


class RepairRegistry:
    """What ``ModelAdapter.validate_and_repair`` reads off a ``ToolRegistry``:
    ``get(name)``, ``list_tools()`` (the fuzzy match and the error text) and
    each tool's ``input_model``."""

    def __init__(self, tools: list[dict[str, Any]]) -> None:
        self._tools: dict[str, _RepairTool] = {}
        for t in tools:
            fn = t.get("function", t)
            name = S.tool_name(t)
            self._tools[name] = _RepairTool(name, fn.get("description") or "", S.tool_parameters(t))

    def get(self, name: str) -> _RepairTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[_RepairTool]:
        return list(self._tools.values())


def fresh_adapter() -> Any:
    """A cold daemon's adapter for a local model (tier light)."""
    from prometheus.adapter import ModelAdapter

    return ModelAdapter(tier="light")


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------


def repair_calls(case: Any, calls: list[dict[str, Any]],
                 adapter: Any | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The calls as the daemon would execute them, and one event per call:
    ``outcome`` is unchanged | repaired | refused (the adapter raised) |
    unreadable (arguments not an object: the loop refuses it before the
    adapter) | empty_name (the provider drops it; kept as is here), plus the
    repair ``kinds`` and whether the name changed."""
    from prometheus.adapter.validator import repair_kinds

    adapter = adapter or fresh_adapter()
    registry = RepairRegistry(case.tools)
    out: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for c in calls:
        ev: dict[str, Any] = {"outcome": "unchanged", "kinds": [], "name_changed": False}
        events.append(ev)
        if not c["from_text"] and not str(c["name"]).strip():
            ev["outcome"] = "empty_name"
            out.append(c)
            continue
        if not isinstance(c["arguments"], dict):
            ev["outcome"] = "unreadable"
            out.append(c)
            continue
        try:
            name, args, log = adapter.validate_and_repair(
                c["name"], copy.deepcopy(c["arguments"]), registry)
        except ValueError:
            ev["outcome"] = "refused"
            out.append(c)
            continue
        if not log and name == c["name"] and S.strict_eq(args, c["arguments"]):
            out.append(c)
            continue
        ev.update(outcome="repaired", kinds=repair_kinds(log), name_changed=name != c["name"])
        out.append({**c, "name": name, "arguments": args,
                    "arguments_raw": json.dumps(args, ensure_ascii=False), "parse_error": None})
    return out, events


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def tool_metrics(vs: list[dict[str, Any]]) -> dict[str, Any]:
    """score.summarize's eval-tools metrics over these verdicts."""
    valid = [v for v in vs if "arg_valid" in v]
    exact = [v for v in vs if "arg_exact" in v]
    return {
        "task_success": S.rate(sum(v["task_success"] for v in vs), len(vs)),
        "tool_name_accuracy": S.rate(sum(v["tool_name_correct"] for v in vs), len(vs)),
        "arg_validity": S.rate(sum(v["arg_valid"] for v in valid), len(valid)),
        "arg_exactness": S.rate(sum(v["arg_exact"] for v in exact), len(exact)),
    }


def _kr(r: dict[str, Any]) -> dict[str, Any]:
    return {"k": r["k"], "n": r["n"], "rate": r["rate"]}


def repair_gap(cases: list[Any], rows: dict[tuple[str, str], dict[str, Any]],
               summary: dict[str, Any] | None, *, set_name: str = "eval-tools") -> dict[str, Any]:
    group = [c for c in cases if c.set == set_name]
    raw_vs: list[dict[str, Any]] = []
    rep_vs: list[dict[str, Any]] = []
    errors = missing = 0
    kind_calls: Counter[str] = Counter()
    outcome_calls: Counter[str] = Counter()
    first_call: Counter[str] = Counter()
    cases_repaired: list[str] = []
    changed: dict[str, dict[str, list[str]]] = {k: {"fixed": [], "broken": []} for k in VERDICT_KEYS}
    raw_outcomes: dict[str, bool] = {}
    for c in group:
        r = rows.get(c.key)
        if r is None or not S.row_matches(c, r):
            missing += 1
            continue
        if r.get("status") != "ok":
            errors += 1
            continue
        message = r.get("message") or {}
        calls, from_text = S.reply_calls(c, message)
        raw = S.score_case(c, r)  # score.py's own read: the check below holds it to the summary
        fixed_calls, events = repair_calls(c, calls)
        rep = S.score_case(c, r, (fixed_calls, from_text))
        raw_vs.append(raw)
        rep_vs.append(rep)
        raw_outcomes[c.id] = bool(raw["passed"])
        for ev in events:
            outcome_calls[ev["outcome"]] += 1
            kind_calls.update(ev["kinds"])
        if events:
            first_call[events[0]["outcome"]] += 1
        if any(ev["outcome"] == "repaired" for ev in events):
            cases_repaired.append(c.id)
        for k in VERDICT_KEYS:
            a, b = raw.get(k), rep.get(k)
            if a is not None and b is not None and bool(a) != bool(b):
                changed[k]["fixed" if b else "broken"].append(c.id)
            elif (a is None) != (b is None):  # entered or left a denominator
                changed[k].setdefault("denominator", []).append(c.id)

    raw_m, rep_m = tool_metrics(raw_vs), tool_metrics(rep_vs)
    check: dict[str, Any] = {"summary_given": summary is not None}
    if summary is not None:
        entry = (summary.get("sets") or {}).get(set_name) or {}
        sm = entry.get("metrics") or {}
        diffs = [m for m in METRICS
                 if (sm.get(m) or {}).get("k") != raw_m[m]["k"]
                 or (sm.get(m) or {}).get("n") != raw_m[m]["n"]]
        outcome_diffs = sorted(i for i in set(raw_outcomes) | set(entry.get("outcomes") or {})
                               if raw_outcomes.get(i) != (entry.get("outcomes") or {}).get(i))
        counts = {"errors": errors, "missing": missing, "scored": len(raw_vs)}
        count_diffs = [k for k, v in counts.items() if entry.get(k) != v]
        check.update(
            matches=not diffs and not outcome_diffs and not count_diffs,
            metric_diffs=diffs, outcome_diffs=outcome_diffs[:20],
            count_diffs=count_diffs,
            summary_scorer_version=summary.get("scorer_version"),
            summary_schema_validator=summary.get("schema_validator"),
        )
    gap = {m: (None if raw_m[m]["rate"] is None or rep_m[m]["rate"] is None
               else round(100 * (rep_m[m]["rate"] - raw_m[m]["rate"]), 2)) for m in METRICS}
    return {
        "schema": REPORT_SCHEMA, "set": set_name,
        "label": (summary or {}).get("label"),
        "scorer_version": S.SCORER_VERSION, "schema_validator": S.schema_validator(),
        "adapter": {"tier": "light", "fresh_per_case": True, "registry": "case tools"},
        "cases": len(group), "scored": len(raw_vs), "errors": errors, "missing": missing,
        "raw": {m: _kr(raw_m[m]) for m in METRICS},
        "post_repair": {m: _kr(rep_m[m]) for m in METRICS},
        "gap_pts": gap,
        "changed": {k: {kk: sorted(vv) for kk, vv in v.items()} for k, v in changed.items()},
        "cases_with_a_repair": sorted(cases_repaired),
        "calls_by_outcome": dict(outcome_calls),
        "first_call_by_outcome": dict(first_call),
        "repair_kinds": dict(kind_calls),
        "raw_check": check,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _cell(r: dict[str, Any]) -> str:
    return f"{r['k']}/{r['n']} ({100 * r['rate']:.1f}%)" if r["rate"] is not None else f"{r['k']}/{r['n']}"


def render(report: dict[str, Any]) -> str:
    lines = [f"{report['label'] or '?'}  ({report['set']}: {report['scored']} scored, "
             f"{report['errors']} errors, {report['missing']} missing)",
             f"  {'metric':<20} {'raw':<20} {'post-repair':<20} gap   fixed broken"]
    keymap = dict(zip(METRICS, VERDICT_KEYS))
    for m in METRICS:
        ch = report["changed"][keymap[m]]
        g = report["gap_pts"][m]
        lines.append(f"  {m:<20} {_cell(report['raw'][m]):<20} {_cell(report['post_repair'][m]):<20} "
                     f"{'n/a' if g is None else f'{g:+.1f}':<5} {len(ch['fixed']):>5} {len(ch['broken']):>6}")
    chk = report["raw_check"]
    lines.append(f"  cases with a repair: {len(report['cases_with_a_repair'])}; calls by outcome "
                 f"{report['calls_by_outcome']}; first call {report['first_call_by_outcome']}; "
                 f"repair kinds {report['repair_kinds'] or '{}'}")
    if chk.get("summary_given"):
        lines.append("  raw recompute == summary: " + ("yes" if chk["matches"] else
                     f"NO (metrics {chk['metric_diffs']}, outcomes {chk['outcome_diffs']}, "
                     f"counts {chk['count_diffs']})"))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--results", required=True, help="score.py result JSONL")
    ap.add_argument("--cases", action="append", required=True, help="case JSONL (repeatable)")
    ap.add_argument("--summary", help="score.py summary of the same rows (the raw check)")
    ap.add_argument("--set", default="eval-tools", choices=list(S.TOOL_SETS))
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)
    if S._jsonschema is None:
        print("error: repair_gap needs the jsonschema package (the mcp extra)", file=sys.stderr)
        return 2
    try:
        cases = S.load_cases(args.cases)
        rows, labels = S.latest_rows(Path(args.results).expanduser())
        summary = (json.loads(Path(args.summary).expanduser().read_text(encoding="utf-8"))
                   if args.summary else None)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if len(labels) > 1:
        print(f"error: {args.results} mixes labels {sorted(labels)}", file=sys.stderr)
        return 2
    report = repair_gap(cases, rows, summary, set_name=args.set)
    if report["label"] is None and labels:
        report["label"] = next(iter(labels))
    if args.json_out:
        Path(args.json_out).expanduser().write_text(json.dumps(report, indent=2) + "\n",
                                                    encoding="utf-8")
    print(render(report), end="")
    chk = report["raw_check"]
    return 1 if chk.get("summary_given") and not chk["matches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
