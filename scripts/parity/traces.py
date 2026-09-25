"""Parity trace files: what was recorded, and what it must replay to.

Two files per scenario under ``tests/fixtures/parity/``:

``<name>.trace.json``
    The INPUTS: the exact config, files, steps, and every model exchange
    (normalized request + verbatim response). A replay reads its steps from
    here, not from ``scenarios.py``, so editing a scenario without
    re-recording cannot silently desynchronise the two.

``<name>.expected.json``
    The OUTPUTS the replay must reproduce: normalized step results and the
    normalized dump of every store the daemon wrote. Kept apart from the
    trace so a re-baseline shows up as its own reviewable diff.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from parity.model_server import Exchange
from parity.scenarios import Scenario

FORMAT = 1


def trace_dir(src_root: Path) -> Path:
    return src_root / "tests" / "fixtures" / "parity"


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def write_trace(src_root: Path, scenario: Scenario, *, config_text: str,
                exchanges: list[Exchange], meta: dict) -> Path:
    path = trace_dir(src_root) / f"{scenario.name}.trace.json"
    _dump(path, {
        "format": FORMAT,
        "scenario": scenario.name,
        "covers": scenario.covers,
        "recorded": meta,
        "config": config_text,
        "files": scenario.files,
        "git_repos": list(scenario.git_repos),
        "steps": scenario.steps,
        "exchanges": [ex.to_json() for ex in exchanges],
    })
    return path


def write_expected(src_root: Path, name: str, expected: dict) -> Path:
    path = trace_dir(src_root) / f"{name}.expected.json"
    _dump(path, expected)
    return path


def load_trace(src_root: Path, name: str) -> dict:
    path = trace_dir(src_root) / f"{name}.trace.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ValueError(f"{path}: trace format {data.get('format')} != {FORMAT}")
    return data


def load_expected(src_root: Path, name: str) -> dict:
    return json.loads((trace_dir(src_root) / f"{name}.expected.json").read_text(encoding="utf-8"))


def scenario_from_trace(trace: dict, base: Scenario | None) -> Scenario:
    """The scenario as RECORDED; ``require`` comes from code (it is a check, not an input)."""
    return Scenario(
        name=trace["scenario"], covers=trace["covers"], steps=trace["steps"],
        files=trace["files"], git_repos=tuple(trace["git_repos"]),
        config={}, require=base.require if base else None,
    )


def exchanges_from_trace(trace: dict) -> list[Exchange]:
    return [Exchange.from_json(d) for d in trace["exchanges"]]


def available(src_root: Path) -> list[str]:
    return sorted(p.name[: -len(".trace.json")]
                  for p in trace_dir(src_root).glob("*.trace.json"))


def renormalize_trace(src_root: Path, name: str) -> None:
    """Rewrite a trace's stored requests under the current request rules."""
    path = trace_dir(src_root) / f"{name}.trace.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["exchanges"] = [Exchange.from_json(d).to_json() for d in data["exchanges"]]
    _dump(path, data)
