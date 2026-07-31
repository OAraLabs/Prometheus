"""Event processor for the live recorder.

Filters, deduplicates, and groups browser events into logical workflow
actions. Pure deterministic logic — no model calls, no I/O.

Ported from skillforge-engine core/event_processor.py.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def filter_noise_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove noise events that don't contribute to workflow understanding.

    Filters out scroll and focus events, plus repeated rapid clicks on the
    same element (within 1 second).
    """
    filtered: list[dict[str, Any]] = []
    last_click_element = None
    last_click_time = 0

    for event in events:
        event_type = event.get("type")
        timestamp = event.get("timestamp", 0)

        if event_type == "scroll":
            continue

        # Focus events are implied by clicks/inputs
        if event_type == "focus":
            continue

        if event_type in ("click", "dblclick"):
            element = event.get("element") or {}
            css_selector = element.get("cssSelector", "")

            if css_selector == last_click_element and (timestamp - last_click_time) < 1000:
                log.debug("Filtered duplicate click on %s", css_selector)
                continue

            last_click_element = css_selector
            last_click_time = timestamp

        filtered.append(event)

    log.info("Event filtering: %d -> %d events", len(events), len(filtered))
    return filtered


def merge_sequential_inputs(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge sequential input events on the same field into one input action.

    Example: 10 keydown events typing "john@email.com" become the single
    final input event that carries the full value.
    """
    merged: list[dict[str, Any]] = []
    i = 0

    while i < len(events):
        event = events[i]
        event_type = event.get("type")

        if event_type in ("input", "change", "keydown"):
            element = event.get("element") or {}
            css_selector = element.get("cssSelector", "")

            field_inputs = [event]
            j = i + 1

            while j < len(events):
                next_event = events[j]
                next_type = next_event.get("type")
                next_element = next_event.get("element") or {}
                next_selector = next_element.get("cssSelector", "")

                if next_type in ("input", "change", "keydown") and next_selector == css_selector:
                    field_inputs.append(next_event)
                    j += 1
                else:
                    break

            if len(field_inputs) > 1:
                # The last event in the sequence has the final value
                final_event = field_inputs[-1]
                log.debug("Merged %d input events on %s", len(field_inputs), css_selector)
                merged.append(final_event)
                i = j
            else:
                merged.append(event)
                i += 1
        else:
            merged.append(event)
            i += 1

    log.info("Input merging: %d -> %d events", len(events), len(merged))
    return merged


def group_related_actions(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group related events into logical actions.

    A click on a field followed by typing into it (within 2 seconds)
    becomes a single "fill field" group.
    """
    groups: list[list[dict[str, Any]]] = []
    i = 0

    while i < len(events):
        event = events[i]
        event_type = event.get("type")

        if event_type == "click" and i + 1 < len(events):
            next_event = events[i + 1]
            next_type = next_event.get("type")

            if next_type in ("input", "change"):
                time_diff = next_event.get("timestamp", 0) - event.get("timestamp", 0)
                if time_diff < 2000:
                    groups.append([event, next_event])
                    log.debug("Grouped click + input into fill field action")
                    i += 2
                    continue

        groups.append([event])
        i += 1

    log.info("Action grouping: %d events -> %d action groups", len(events), len(groups))
    return groups


def attach_screenshots(
    action_groups: list[list[dict[str, Any]]],
    screenshots: dict[int, str],
) -> list[dict[str, Any]]:
    """Attach the recorded screenshot to each action group.

    The extension replaces each event's captured image with a
    ``screenshotIndex`` referencing the ``screenshot_N`` multipart part;
    ``screenshots`` maps that index to a stored file path. A group is
    tagged with the screenshot of its first event that carries one.
    """
    actions = []

    for group in action_groups:
        screenshot = None
        for event in group:
            idx = event.get("screenshotIndex")
            if idx is not None and idx in screenshots:
                screenshot = screenshots[idx]
                break

        actions.append({
            "events": group,
            "timestamp": group[0].get("timestamp", 0),
            "screenshot": screenshot,
        })

    return actions


def process_events(
    events: list[dict[str, Any]],
    screenshots: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Main event processing pipeline.

    Args:
        events: Captured events from the extension (already JSON-decoded).
        screenshots: Optional dict of screenshot index -> stored path.

    Returns:
        Processed action groups ready for :func:`events_to_actions`.
    """
    if not isinstance(events, list):
        return {"success": False, "error": "Events must be an array"}

    log.info("Processing %d raw events", len(events))

    filtered = filter_noise_events(events)
    merged = merge_sequential_inputs(filtered)
    grouped = group_related_actions(merged)

    if screenshots:
        actions = attach_screenshots(grouped, screenshots)
    else:
        actions = [
            {"events": group, "timestamp": group[0].get("timestamp", 0)}
            for group in grouped
        ]

    return {
        "success": True,
        "raw_event_count": len(events),
        "filtered_event_count": len(filtered),
        "merged_event_count": len(merged),
        "action_count": len(actions),
        "actions": actions,
    }
