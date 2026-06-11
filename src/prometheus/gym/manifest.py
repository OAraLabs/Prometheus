"""Experiment manifests — exactly ONE variable changed vs baseline.

The runner refuses a manifest that changes more than one variable: that rule
is what keeps an experiment interpretable. ``variable: none`` is the baseline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Variables the v1 runner knows how to apply. Reserved names are accepted in
# the schema but REFUSED at load with a clear message (sampling needs a
# provider-level override; schema_lowering is Workstream C's plug-in point).
IMPLEMENTED_VARIABLES = frozenset({
    "none",              # baseline — no change
    "example_call",      # {tool: str, example: dict} → example appended to system prompt
    "tool_description",  # {tool: str, text: str} → tool.description override
    "system_prompt",     # {text: str} → replaces the task set's system prompt
})
RESERVED_VARIABLES = frozenset({"sampling", "schema_lowering"})


@dataclass
class ExperimentManifest:
    series: str
    experiment: str
    taskset: str               # path relative to repo root
    runs_per_task: int
    variable_name: str         # one of IMPLEMENTED_VARIABLES
    variable_payload: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    sha256: str = ""
    path: str = ""


def load_manifest(path: str | Path) -> ExperimentManifest:
    p = Path(path)
    raw_bytes = p.read_bytes()
    data = yaml.safe_load(raw_bytes)
    for key in ("series", "experiment", "taskset", "variable"):
        if key not in data:
            raise ValueError(f"{p}: manifest missing required key {key!r}")

    variable = data["variable"]
    if isinstance(variable, str):
        variable = {variable: None}
    if not isinstance(variable, dict):
        raise ValueError(f"{p}: 'variable' must be a mapping or the string 'none'")

    # ── The single-variable rule ──────────────────────────────────────
    declared = [k for k, v in variable.items() if k != "none" and v is not None]
    if "none" in variable and declared:
        raise ValueError(
            f"{p}: manifest declares 'none' AND {declared} — pick one"
        )
    if len(declared) > 1:
        raise ValueError(
            f"{p}: REFUSED — manifest changes {len(declared)} variables "
            f"({sorted(declared)}); an experiment changes exactly one"
        )

    if not declared:
        name, payload = "none", {}
    else:
        name = declared[0]
        payload = variable[name] or {}
        if name in RESERVED_VARIABLES:
            raise ValueError(
                f"{p}: variable {name!r} is reserved but not implemented in "
                f"the v1 runner — implemented: {sorted(IMPLEMENTED_VARIABLES)}"
            )
        if name not in IMPLEMENTED_VARIABLES:
            raise ValueError(
                f"{p}: unknown variable {name!r}; implemented: "
                f"{sorted(IMPLEMENTED_VARIABLES)}"
            )

    # Payload sanity per variable
    if name == "example_call":
        if "tool" not in payload or "example" not in payload:
            raise ValueError(f"{p}: example_call needs {{tool, example}}")
    if name == "tool_description":
        if "tool" not in payload or "text" not in payload:
            raise ValueError(f"{p}: tool_description needs {{tool, text}}")
    if name == "system_prompt" and "text" not in payload:
        raise ValueError(f"{p}: system_prompt needs {{text}}")

    runs = int(data.get("runs_per_task", 3))
    if runs < 1:
        raise ValueError(f"{p}: runs_per_task must be ≥ 1")

    return ExperimentManifest(
        series=str(data["series"]),
        experiment=str(data["experiment"]),
        taskset=str(data["taskset"]),
        runs_per_task=runs,
        variable_name=name,
        variable_payload=payload,
        notes=data.get("notes", ""),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        path=str(p),
    )
