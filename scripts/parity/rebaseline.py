"""Re-derive expected files from a replay of the COMMITTED exchanges — no model.

WHEN IT APPLIES. A change that alters only what the daemon RECORDS (a
telemetry column filled, a row added) and not one model request. Every request
then still matches its committed exchange, so the committed answers are still
the answers, and a replay against them is a faithful recording of the new
code: no 4090, no hosted key. A change that alters any request is not that
kind of change and is re-recorded against real models (``record``).

THE RULES (Will, 2026-09-26, WP-X.21):

* it is its own explicit mode, ``rebaseline-from-exchanges`` — never a side
  effect of ``replay``;
* it refuses — exits non-zero and writes NOTHING — when any model request
  differs from its committed exchange, has no committed answer, or a committed
  request is never made;
* it writes only ``*.expected.json``; the traces (the inputs) are never touched;
* the PR that uses it shows the per-column diff, attributed to each change. This
  mode prints that diff; ``--dry-run`` prints it and writes nothing, so a bundle
  can be run one change at a time.

It also refuses, because each would bake something untrue into a golden:

* a harness error, or a daemon step that failed;
* a scenario whose own ``require`` check fails on the new run — it would no
  longer exercise its subject;
* observables that differ between replays of the same tree — a value that is not
  stable cannot be a baseline.

All-or-nothing: the command judges every scenario before it writes any file.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from parity.compare import compare, render_report
from parity.runner import RunOutput

MODE = "rebaseline-from-exchanges"


@dataclass
class Verdict:
    name: str
    changed: bool                          # the observables differ from the committed expected file
    refusals: list[str] = field(default_factory=list)
    harness_error: bool = False
    report: str = ""                       # the per-category, per-column diff (or "unchanged")
    expected: dict | None = None           # what to write: set only when changed AND not refused
    categories: list[str] = field(default_factory=list)


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def judge(name: str, runs: list[RunOutput], expected: dict, root: Path,
          *, require: Callable[[Any], list[str]] | None = None) -> Verdict:
    """Decide whether *runs* (replays of one scenario on one tree) may replace
    its expected file. Pure: reads nothing, writes nothing."""
    if not runs:
        raise ValueError("judge needs at least one replay")
    results = [compare(out, expected, root) for out in runs]
    first = results[0]
    seen: dict[str, list[int]] = {}      # refusal -> the replays it happened in
    for i, res in enumerate(results, 1):
        found = [f"harness error: {err}" for err in res.harness_errors]
        found += [f"step failed: {fail}" for fail in res.step_failures]
        if res.request_mismatches:
            found.append(f"{len(res.request_mismatches)} model request(s) differ from the "
                         f"committed exchanges — re-record instead")
        if res.extra_requests:
            found.append(f"{res.extra_requests} model request(s) have no committed answer "
                         f"— re-record instead")
        if res.missing_requests:
            found.append(f"{res.missing_requests} committed model request(s) were never made "
                         f"— re-record instead")
        for msg in found:
            seen.setdefault(msg, []).append(i)
    refusals = [msg if len(results) == 1 else
                f"{msg} [replay {', '.join(map(str, idx))} of {len(results)}]"
                for msg, idx in seen.items()]
    if len({_digest(r.actual) for r in results}) > 1:
        refusals.append(f"the observables differ between the {len(results)} replays of this "
                        f"tree — not stable enough to be a baseline")
    if require is not None and not refusals:
        refusals += [f"the scenario's own check fails on the new run: {p}"
                     for p in require(runs[0].evidence())]
    changed = bool(first.diffs)
    head = ("REFUSED — nothing is written" if refusals
            else "CHANGED — this diff becomes the expected file" if changed else "unchanged")
    # render_report's first line is the replay verdict; the body is the
    # per-category, per-column diff (recorded = the committed expected file).
    body = render_report(name, first).splitlines()[1:]
    return Verdict(
        name=name, changed=changed, refusals=refusals,
        harness_error=any(res.harness_errors for res in results),
        report="\n".join([f"[{MODE}] {name}: {head}", *body]),
        expected=first.actual if (changed and not refusals) else None,
        categories=sorted(first.diffs),
    )
