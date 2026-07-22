"""Funnel adapter — vision actions into the live recorder's action shape.

The live recorder (DOM path) and the video-ingestion pipeline (vision
path) share one funnel: ``live_recorder.quality_gate.gate_actions``
followed by ``live_recorder.synthesizer.build_skill_content``. This
module maps the vision pipeline's lowercase action dicts
(``{type, target, description, context, parameter?, original_value?}``)
into the funnel's uppercase action shape (``{action_type, target,
description, url, page_title, is_parameter, parameter_name, value,
field_type, timestamp}``) and its parameter shape.

New for Prometheus (no SkillForge ancestor) — the original vision
pipeline had its own synthesizer; here both recording paths converge on
the shared one.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_TYPE_MAP = {
    "click": "CLICK",
    "type": "TYPE",
    "fill_field": "FILL_FIELD",
    "navigate": "NAVIGATE",
    "select": "SELECT",
    "toggle": "TOGGLE",
    "submit": "SUBMIT",
}


def to_funnel_action(action: dict[str, Any]) -> dict[str, Any]:
    """Map one vision action into the live recorder funnel shape."""
    raw_type = str(action.get("type") or "unknown").lower()
    target = str(action.get("target") or "")
    description = str(action.get("description") or f"Perform {raw_type} action")
    context = action.get("context") or {}

    action_type = _TYPE_MAP.get(raw_type, "UNKNOWN")
    if action_type == "CLICK" and (
        "button" in target.lower() or "button" in description.lower()
    ):
        action_type = "CLICK_BUTTON"

    funnel: dict[str, Any] = {
        "action_type": action_type,
        "target": target,
        "description": description,
        "url": str(context.get("url") or ""),
        "page_title": str(context.get("page_title") or context.get("page_or_view") or ""),
        "is_parameter": False,
        "parameter_name": None,
        "value": "",
        "field_type": "",
        "timestamp": action.get("keyframe", 0),
    }

    if action_type in ("TYPE", "FILL_FIELD"):
        funnel["field_type"] = "text"
        param_name = action.get("parameter")
        if param_name:
            funnel["is_parameter"] = True
            funnel["parameter_name"] = param_name
            funnel["value"] = str(action.get("original_value") or "")

    return funnel


def to_funnel_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map extracted vision actions into funnel actions."""
    mapped = [to_funnel_action(a) for a in actions]
    log.info("Funnel mapping: %d vision actions -> %d funnel actions", len(actions), len(mapped))
    return mapped


def to_funnel_parameters(parameters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map ``parameterize_actions`` output to the funnel parameter shape.

    ``{name, type, default, description}`` becomes
    ``{name, type, default_value, description}`` with the vision
    pipeline's generic ``string`` type mapped to the funnel's ``text``.
    """
    mapped = []
    for p in parameters:
        p_type = p.get("type") or "text"
        mapped.append({
            "name": p.get("name", ""),
            "type": "text" if p_type == "string" else p_type,
            "default_value": p.get("default", ""),
            "description": p.get("description", ""),
        })
    return mapped
