"""Conservative dict-wrap unwrapping — Phase 4, reshaped by the evidence.

The original premise (nested schemas defeat the model → flatten them) was
refuted in diagnostics: all 49 builtin input schemas are flat (depth 1). The
observed pathology is the INVERSE — the model invents nesting against flat
schemas:

    sessions_list  {"status": {"status": null}}        (param self-wrap)
    task_create    {"prompt": {"description": ..., ...}}  (top-level wrap)
    browser        {"action": {"navigate": ...}}        (param wrap)
    download_file  {"destination": {"/path": ...}}      (param wrap)

So instead of lowering our schemas, the adapter unwraps the model's phantom
nesting — conservative by construction:

  * runs ONLY after the original input failed pydantic validation;
  * accepts a transform ONLY if the transformed input passes validation;
  * never touches a call that already validates;
  * refuses (returns None) rather than approximating.

Two deterministic transforms, tried in order:

  1. Param self-unwrap — for each declared param whose value is a single-key
     dict keyed by the param's own name, replace it with the inner value:
     ``{"status": {"status": X}} → {"status": X}``.
  2. Top-level promote — the whole argument object is a single-key dict
     whose value is a dict of plausible arguments:
     ``{"prompt": {"description": ..., "prompt": ...}} → inner dict``
     (only when the inner dict's keys overlap the schema's properties).

Every accepted unwrap is logged loudly and surfaces in the repair log (so
telemetry `repairs` counts it) — and becomes a training pair upstream.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def _validates(tool: Any, candidate: dict[str, Any]) -> bool:
    try:
        tool.input_model.model_validate(candidate)
        return True
    except Exception:
        return False


def try_unwrap_arguments(
    tool: Any,
    tool_input: Any,
) -> tuple[dict[str, Any], list[str]] | None:
    """Attempt conservative unwrapping of dict-wrapped arguments.

    Returns ``(unwrapped_input, unwrap_log)`` where the unwrapped input is
    GUARANTEED to pass the tool's pydantic validation, or ``None`` when no
    conservative transform fixes the call.
    """
    if not isinstance(tool_input, dict):
        return None
    if _validates(tool, tool_input):
        return None  # never touch a call that already validates

    try:
        schema = tool.input_model.model_json_schema()
        properties = set(schema.get("properties", {}))
    except Exception:
        return None

    # ── 1. Param self-unwrap ─────────────────────────────────────────
    unwrapped: dict[str, Any] = {}
    notes: list[str] = []
    for key, value in tool_input.items():
        if (
            key in properties
            and isinstance(value, dict)
            and len(value) == 1
            and next(iter(value)) == key
        ):
            unwrapped[key] = value[key]
            notes.append(f"unwrapped self-keyed dict for param {key!r}")
        else:
            unwrapped[key] = value
    if notes and _validates(tool, unwrapped):
        log.warning(
            "Adapter unwrapped dict-wrapped argument(s) for %s: %s",
            getattr(tool, "name", "?"), "; ".join(notes),
        )
        return unwrapped, notes

    # ── 2. Top-level promote ─────────────────────────────────────────
    if len(tool_input) == 1:
        ((key, inner),) = tool_input.items()
        if (
            isinstance(inner, dict)
            and inner
            and properties
            and set(inner) & properties
        ):
            if _validates(tool, inner):
                note = f"promoted inner dict wrapped under {key!r} to arguments"
                log.warning(
                    "Adapter unwrapped top-level wrap for %s: %s",
                    getattr(tool, "name", "?"), note,
                )
                return dict(inner), [note]

    return None
