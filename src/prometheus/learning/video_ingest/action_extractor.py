"""Action extraction from vision digests.

Ported from skillforge-engine ``core/action_extractor.py``. Kept: the
tooltip-marker filter (Google-Sheets-style hint text that vision models
misread as typed input) and the ``_normalize_target`` cell-reference
canonicalization. Dropped: the psutil memory/progress logging (a plain
log line every 10 frames instead) and the cursor-event fallback branch —
there are no cursor events in the video-ingestion path.
``parameterize_actions`` was extended to read the typed value out of the
action description (first quoted fragment) because vision-derived
actions carry no keylogger ``typed_text``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

_TOOLTIP_MARKERS = (
    "type '@'",
    "type '@date'",
    'type "@"',
    'type "@date"',
    "smart chip",
    "date picker",
    "to open date",
    "to insert a people",
    "type here to search",
    "press enter to search",
    "search or type",
)


def _is_tooltip_action(action_type: str, target: str, description: str) -> bool:
    """Return True if this action is a UI tooltip/hint string, not real user input.

    Google Sheets (and similar apps) display helper text INSIDE empty
    cells, e.g. "Type '@' then a name to insert a people smart chip".
    Vision models read these as typed input despite the formula bar being
    empty. These are never real user actions and must be dropped.
    """
    if action_type not in ("type", "select"):
        return False
    combined = (target + " " + description).lower()
    return any(marker in combined for marker in _TOOLTIP_MARKERS)


def _normalize_target(action_type: str, target: str, description: str) -> str:
    """Normalize an action target to a canonical form before dedup.

    For TYPE actions the vision model inconsistently sets ``target`` to
    the typed value ("Cat", "145"), the column header ("Amount"), or the
    cell reference ("A1", "B3"). Extract the canonical cell reference
    from the description when present:

        "User types 'Cat' into cell A1"  -> A1
        "User types '145' into cell B3"  -> B3
        "typing in cell C5"              -> C5
    """
    if action_type not in ("type", "select"):
        return target  # Only normalize TYPE/SELECT; CLICK targets are fine as-is

    m = re.search(r"(?:into|in|cell)\s+([A-Z]+\d+)", description, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # No cell ref — return original target (handles form fields like "Job Name")
    return target


def extract_actions(digests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract a sequence of discrete user actions from vision digests.

    Consumes the digest shape produced by ``vision_digest.digest_keyframes``:
    ``{keyframe_index, vision: {actions: [{type, target, description}],
    context: {...}}, narration}``.
    """
    actions: list[dict[str, Any]] = []
    total = len(digests)

    for idx, digest in enumerate(digests, 1):
        if idx % 10 == 0 or idx == total:
            log.info("Action extraction: frame %d/%d", idx, total)

        vision = digest.get("vision")
        kf_index = digest.get("keyframe_index", 0)
        if not vision or "actions" not in vision:
            continue

        for va in vision["actions"]:
            raw_type = va.get("type", "unknown")
            raw_target = va.get("target", "")
            description = va.get("description", "")

            # Drop tooltip/hint text the model mistook for user input.
            if _is_tooltip_action(raw_type, raw_target, description):
                log.debug("[kf=%s] Dropped tooltip action: %.60s", kf_index, description)
                continue

            # Normalize target to the canonical cell ref BEFORE any
            # downstream pass keys on `target`.
            normalized_target = _normalize_target(raw_type, raw_target, description)

            actions.append({
                "step": len(actions) + 1,
                "keyframe": kf_index,
                "type": raw_type,
                "target": normalized_target,
                "description": description,
                "context": vision.get("context", {}),
            })

    # Re-number steps sequentially
    for i, action in enumerate(actions):
        action["step"] = i + 1

    return actions


def _extract_typed_value(description: str) -> str:
    """Best-effort typed value from a vision description (quoted fragment)."""
    m = re.search(r"['\"]([^'\"]{1,80})['\"]", description)
    return m.group(1) if m else ""


def parameterize_actions(
    actions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Identify typed values that should become skill parameters.

    Each TYPE action with a recoverable value gets a ``parameter`` /
    ``original_value`` annotation and a matching entry in the returned
    parameters list.
    """
    parameters: list[dict[str, Any]] = []
    param_counter = 0

    for action in actions:
        if action.get("type") != "type":
            continue
        value = action.get("typed_text") or _extract_typed_value(action.get("description", ""))
        if not value:
            continue

        param_name = f"input_{param_counter}"
        param_counter += 1

        parameters.append({
            "name": param_name,
            "type": "string",
            "description": action.get("description", ""),
            "default": value,
            "source_step": action.get("step", 0),
        })

        action["parameter"] = param_name
        action["original_value"] = value

    return actions, parameters
