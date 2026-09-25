"""Turn a run into comparable observables, and diff them against a recording.

The diff is organised by what a reviewer asks first — did the TOOL CALLS
change, did a GATE DECISION change, did a CHECKPOINT, a MEMORY WRITE, a
TELEMETRY ROW or the FINAL REPLY change — but it is computed over the whole
normalized dump, so a write to a store no category names still fails the run
(it lands under "other").

Model REQUESTS are compared too, by the model server: a request that differs
from the recording is a divergence even when every persisted byte matches,
because it means the daemon asked the model something else and the recorded
answer no longer answers it.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from parity.normalize import normalize_observables
from parity.runner import RunOutput

# Which stores feed which category. Matched on the dump's path suffix + table.
CATEGORIES: list[tuple[str, str, str | None]] = [
    ("tool_calls", "telemetry.db", "tool_calls"),
    ("gate_decisions", "audit.db", "permission_audit"),
    ("gate_decisions", "permission_audit.jsonl", None),
    ("checkpoints", "checkpoints.db", None),
    ("memory", "lcm.db", None),
    ("memory", "memory.db", None),
    ("memory", "training.db", None),
    ("memory", "MEMORY.md", None),
    ("memory", "USER.md", None),
    ("telemetry", "telemetry.db", None),
]


def categorize(path: str, table: str | None) -> str:
    if "/checkpoints/" in path:           # the store and its content-addressed blobs
        return "checkpoints"
    if path.startswith(("ws/", "cwd/")):  # the files the agent's tools act on
        return "workspace_files"
    for cat, suffix, tbl in CATEGORIES:
        if path.endswith(suffix) and (tbl is None or tbl == table):
            return cat
    if "/wiki/" in path:
        return "memory"
    return "other"


def expected_from(out: RunOutput, root: Path) -> dict:
    return normalize_observables({"steps": out.steps, "stores": out.stores}, root)


@dataclass
class CompareResult:
    actual: dict
    diffs: dict[str, list[str]] = field(default_factory=dict)   # category -> lines
    request_mismatches: list[dict] = field(default_factory=list)
    extra_requests: int = 0
    missing_requests: int = 0
    harness_errors: list[str] = field(default_factory=list)
    step_failures: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        if self.harness_errors:
            return 2
        if self.diffs or self.request_mismatches or self.extra_requests or self.missing_requests:
            return 1
        return 0


def _lines(obj: Any) -> list[str]:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False, default=str).splitlines()


def _leaves(a: Any, b: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    """Every differing leaf, by path — so a one-line change inside a 30 kB
    system prompt is reported as that line, not as two identical prefixes."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[tuple[str, Any, Any]] = []
        for k in sorted(set(a) | set(b), key=str):
            if a.get(k, _MISSING) != b.get(k, _MISSING):
                out += _leaves(a.get(k, _MISSING), b.get(k, _MISSING), f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                out += _leaves(x, y, f"{path}[{i}]")
        return out
    return [(path, a, b)]


class _Missing:
    def __repr__(self) -> str:
        return "<absent>"


_MISSING = _Missing()


def _show(v: Any) -> str:
    if isinstance(v, str) or v is _MISSING:
        return repr(v)
    return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)


def _udiff(a: Any, b: Any, label: str, context: int = 1, limit: int = 60) -> list[str]:
    out: list[str] = []
    for path, x, y in _leaves(a, b):
        if isinstance(x, list) and isinstance(y, list):
            # Rows added or removed: show them, one JSON line per field.
            body = list(difflib.unified_diff(_lines(x), _lines(y), "recorded", "replayed",
                                             n=context, lineterm=""))[2:]
            out.append(f"{label}{path}: ({len(x)} -> {len(y)} items)")
            out += ["  " + ln[:240] for ln in body]
        elif isinstance(x, str) and isinstance(y, str) and ("\n" in x or "\n" in y):
            body = list(difflib.unified_diff(x.splitlines(), y.splitlines(), "recorded",
                                             "replayed", n=context, lineterm=""))[2:]
            out.append(f"{label}{path}: (multi-line text)")
            out += ["  " + ln[:240] for ln in body]
        else:
            out.append(f"{label}{path}:")
            out.append(f"  - recorded: {_show(x)[:240]}")
            out.append(f"  + replayed: {_show(y)[:240]}")
    if len(out) > limit:
        out = out[:limit] + [f"... ({len(out) - limit} more diff lines)"]
    return out


def _keyed(table: dict | None) -> Any:
    """Rows as {column: value} so a diff path names the column, not an index."""
    if not table:
        return table
    cols = table.get("columns", [])
    return {"rows": [dict(zip(cols, r)) for r in table.get("rows", [])]}


def compare(out: RunOutput, expected: dict, root: Path) -> CompareResult:
    actual = normalize_observables({"steps": out.steps, "stores": out.stores}, root)
    res = CompareResult(actual=actual, harness_errors=list(out.errors),
                        step_failures=list(out.step_failures))

    # Requests (divergence at the provider boundary).
    for s in out.served:
        if s.recorded_index is None:
            res.extra_requests += 1
            res.diffs.setdefault("model_requests", []).append(
                f"arrival #{s.index} ({s.upstream}): no recorded answer left for this request")
        elif not s.matched:
            rec = out.exchanges[s.recorded_index].request
            res.request_mismatches.append({"arrival": s.index, "recorded": s.recorded_index})
            res.diffs.setdefault("model_requests", []).extend(
                [f"arrival #{s.index} ({s.upstream}) vs recorded #{s.recorded_index}:"]
                + _udiff(rec, s.request, f"request#{s.recorded_index}", limit=80))
    res.missing_requests = len(out.unconsumed)

    # Steps: final replies are their own category; everything else is "steps".
    exp_steps, act_steps = expected.get("steps", []), actual.get("steps", [])
    for i in range(max(len(exp_steps), len(act_steps))):
        e = exp_steps[i] if i < len(exp_steps) else None
        a = act_steps[i] if i < len(act_steps) else None
        if e == a:
            continue
        cat = "final_reply" if (e or a or {}).get("op") == "chat" else "steps"
        if cat == "final_reply" and e and a and e.get("reply") == a.get("reply"):
            cat = "steps"
        res.diffs.setdefault(cat, []).extend(_udiff(e, a, f"step[{i}]"))

    # Stores, per table / file.
    exp_st, act_st = expected.get("stores", {}), actual.get("stores", {})
    for path in sorted(set(exp_st) | set(act_st)):
        e, a = exp_st.get(path), act_st.get(path)
        if e == a:
            continue
        if isinstance(e, dict) and isinstance(a, dict) and "sqlite" in e and "sqlite" in a:
            for table in sorted(set(e["sqlite"]) | set(a["sqlite"])):
                te, ta = e["sqlite"].get(table), a["sqlite"].get(table)
                if te != ta:
                    res.diffs.setdefault(categorize(path, table), []).extend(
                        _udiff(_keyed(te), _keyed(ta), f"{path}:{table}"))
        else:
            res.diffs.setdefault(categorize(path, None), []).extend(_udiff(e, a, path))
    return res


def render_report(name: str, res: CompareResult) -> str:
    if res.exit_code == 0:
        return f"[replay] {name}: PARITY (0 diffs, all model requests matched)"
    lines = [f"[replay] {name}: {'HARNESS ERROR' if res.exit_code == 2 else 'DIFF'}"]
    for err in res.harness_errors:
        lines.append(f"  harness error: {err}")
    for fail in res.step_failures:
        lines.append(f"  step failed: {fail}")
    if res.request_mismatches:
        lines.append(f"  model requests that differ from the recording: {len(res.request_mismatches)}")
    if res.extra_requests:
        lines.append(f"  model requests the recording has no answer for: {res.extra_requests}")
    if res.missing_requests:
        lines.append(f"  recorded model requests the daemon never made: {res.missing_requests}")
    for cat in ("model_requests", "tool_calls", "gate_decisions", "checkpoints",
                "workspace_files", "memory", "telemetry", "final_reply", "steps", "other"):
        if cat in res.diffs:
            lines.append(f"  -- {cat} --")
            lines.extend("    " + ln for ln in res.diffs[cat])
    return "\n".join(lines)
