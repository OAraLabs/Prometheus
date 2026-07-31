"""Deterministic event-to-action mapping for the live recorder.

Converts grouped browser events into structured workflow actions.
No model needed — pure deterministic logic based on DOM data.

Ported from skillforge-engine core/event_to_actions.py.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def get_element_label(element: dict[str, Any]) -> str:
    """Extract a human-readable label for an element.

    Priority: ariaLabel > closestLabel > placeholder > name > id > textContent.
    """
    label = (
        element.get("ariaLabel")
        or element.get("closestLabel")
        or element.get("placeholder")
        or element.get("name")
        or element.get("id")
        or element.get("textContent", "")[:50]
    )
    return label.strip()


def map_click_event(event: dict[str, Any]) -> dict[str, Any]:
    """Map a click event to a workflow action."""
    element = event.get("element") or {}
    tag = element.get("tagName", "").upper()
    element_type = element.get("type", "").lower()
    text_content = element.get("textContent", "").strip()
    label = get_element_label(element)

    if tag == "BUTTON" or (tag == "INPUT" and element_type in ("button", "submit")):
        return {
            "action_type": "CLICK_BUTTON",
            "target": text_content or label,
            "element_type": "button",
            "css_selector": element.get("cssSelector"),
            "description": f"Click the '{text_content or label}' button",
        }

    if tag == "A":
        href = element.get("href", "")
        return {
            "action_type": "NAVIGATE",
            "target": text_content or label,
            "url": event.get("toUrl") or href,
            "description": f"Click '{text_content or label}' link",
        }

    if tag == "INPUT" and element_type in ("checkbox", "radio"):
        checked = element.get("checked", False)
        return {
            "action_type": "TOGGLE",
            "target": label,
            "value": checked,
            "element_type": element_type,
            "description": f"{'Check' if checked else 'Uncheck'} '{label}'",
        }

    if tag == "SELECT":
        selected_value = element.get("value", "")
        return {
            "action_type": "SELECT",
            "target": label,
            "value": selected_value,
            "description": f"Select '{selected_value}' from '{label}' dropdown",
        }

    return {
        "action_type": "CLICK",
        "target": text_content or label or tag,
        "element_type": tag.lower(),
        "css_selector": element.get("cssSelector"),
        "description": f"Click {text_content or label or tag}",
    }


def map_input_event(event: dict[str, Any]) -> dict[str, Any]:
    """Map an input/change event to a workflow action."""
    element = event.get("element") or {}
    label = get_element_label(element)
    value = event.get("inputValue") or element.get("value", "")
    field_type = element.get("type", "text").lower()

    # Password field — value arrives pre-masked from the extension
    if field_type == "password":
        return {
            "action_type": "TYPE",
            "target": label,
            "value": "••••••",
            "field_type": field_type,
            "is_parameter": True,
            "parameter_name": element.get("name") or element.get("id") or "password",
            "description": f"Enter password in '{label}' field",
        }

    if field_type == "email":
        return {
            "action_type": "TYPE",
            "target": label,
            "value": value,
            "field_type": field_type,
            "is_parameter": True,
            "parameter_name": element.get("name") or element.get("id") or "email",
            "description": f"Type '{value}' in the '{label}' field",
        }

    return {
        "action_type": "TYPE",
        "target": label,
        "value": value,
        "field_type": field_type,
        "is_parameter": bool(value),
        "parameter_name": element.get("name") or element.get("id"),
        "description": f"Type '{value}' in the '{label}' field",
    }


def map_navigation_event(event: dict[str, Any]) -> dict[str, Any]:
    """Map a page navigation event to a workflow action."""
    return {
        "action_type": "NAVIGATE",
        "from_url": event.get("fromUrl", ""),
        "to_url": event.get("toUrl", ""),
        "page_title": event.get("pageTitle", ""),
        "description": f"Navigate to {event.get('pageTitle') or event.get('toUrl', '')}",
    }


def map_submit_event(event: dict[str, Any]) -> dict[str, Any]:
    """Map a form submit event to a workflow action."""
    element = event.get("element") or {}
    return {
        "action_type": "SUBMIT",
        "target": "form",
        "form_action": element.get("formAction", ""),
        "description": "Submit form",
    }


def map_fill_field_action(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Map a grouped click + input into a single fill-field action."""
    if len(events) < 2:
        return map_event_to_action(events[0])

    input_event = events[1]
    element = input_event.get("element") or {}
    label = get_element_label(element)
    value = input_event.get("inputValue") or element.get("value", "")
    field_type = element.get("type", "text").lower()
    shown_value = value if field_type != "password" else "••••••"

    return {
        "action_type": "FILL_FIELD",
        "target": label,
        "value": shown_value,
        "field_type": field_type,
        "is_parameter": bool(value) and field_type != "password",
        "parameter_name": element.get("name") or element.get("id"),
        "description": f"Fill '{label}' field with '{shown_value}'",
    }


def map_event_to_action(event: dict[str, Any]) -> dict[str, Any]:
    """Map a single event to a workflow action (deterministic)."""
    event_type = event.get("type", "")

    if event_type in ("click", "dblclick"):
        return map_click_event(event)
    if event_type in ("input", "change"):
        return map_input_event(event)
    if event_type == "navigation":
        return map_navigation_event(event)
    if event_type == "submit":
        return map_submit_event(event)

    log.warning("Unknown event type: %s", event_type)
    return {
        "action_type": "UNKNOWN",
        "event_type": event_type,
        "description": f"Perform {event_type} action",
    }


def convert_action_group_to_action(action_group: dict[str, Any]) -> dict[str, Any]:
    """Convert an action group (from event_processor) to a workflow action."""
    events = action_group.get("events", [])
    screenshot = action_group.get("screenshot")
    timestamp = action_group.get("timestamp", 0)

    if not events:
        return {
            "action_type": "UNKNOWN",
            "description": "Unknown action",
            "timestamp": timestamp,
        }

    if len(events) == 2 and events[0].get("type") == "click" and events[1].get("type") in ("input", "change"):
        action = map_fill_field_action(events)
    else:
        # Single event, or use the last event of the group (has final state)
        action = map_event_to_action(events[-1])

    action["timestamp"] = timestamp
    action["elapsed_ms"] = events[0].get("elapsedMs", 0)
    action["url"] = events[0].get("url", "")
    action["page_title"] = events[0].get("pageTitle", "")

    if screenshot:
        action["screenshot"] = screenshot

    return action


def extract_parameters(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract workflow parameters from actions.

    Any field where the user typed a value becomes a potential parameter.
    """
    parameters = []

    for action in actions:
        if action.get("is_parameter"):
            param_name = action.get("parameter_name") or f"param_{len(parameters)}"
            parameters.append({
                "name": param_name,
                "type": action.get("field_type", "text"),
                "default_value": action.get("value", ""),
                "description": f"Value for {action.get('target', param_name)}",
            })

    return parameters


def events_to_actions(processed_events: dict[str, Any]) -> dict[str, Any]:
    """Convert processed event groups to workflow actions.

    Args:
        processed_events: Output from :func:`event_processor.process_events`.

    Returns:
        Dict with workflow actions and detected parameters.
    """
    if not processed_events.get("success"):
        return processed_events

    action_groups = processed_events.get("actions", [])
    actions = [convert_action_group_to_action(group) for group in action_groups]
    parameters = extract_parameters(actions)

    log.info(
        "Converted %d action groups -> %d workflow actions (%d parameters)",
        len(action_groups), len(actions), len(parameters),
    )

    return {
        "success": True,
        "actions": actions,
        "parameters": parameters,
        "action_count": len(actions),
        "parameter_count": len(parameters),
    }
