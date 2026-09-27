"""SKILL.md synthesis for the live recorder.

Generates Prometheus-format skill markdown (frontmatter ``name:`` /
``description:``, ``## When to use`` / ``## Steps`` / ``## Notes``
sections) from structured workflow actions. Template-based — no model
required. Returns content; persistence goes through
``SkillCreator.persist_skill_content()`` like every other
machine-generated skill.

Ported from skillforge-engine core/live_synthesizer.py; the output format
was rewritten to match the skills/auto/ conventions that the loader,
curator, and Beacon already understand.

Every recorded value — a typed value, a label, a page title, a URL, a
selector, a parameter's name and type, a step's description (the vision
path's comes from a model) — is rendered by :func:`_code`: JSON-encoded
inside inline code. A recording can put anything in those strings, and they
used to be pasted into the markdown as they were, so a typed value holding a
line break and ```` ```python ```` opened a real code block in a skill that
auto-persists without review (WP-X.42). Rendered this way a value is one
line, and nothing in it can open a fence, a heading, a list item or any
other structure.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger(__name__)


@dataclass
class LiveSkillDraft:
    """A synthesized skill, ready for the quality gate and persistence."""

    name: str
    title: str
    description: str
    content: str
    step_count: int
    parameter_count: int


# Line breaks json.dumps leaves literal with ensure_ascii=False; ``str.splitlines``
# (and so the skill loader) splits on them all the same.
_LITERAL_LINE_BREAKS = {"\x85": "\\u0085", "\u2028": "\\u2028", "\u2029": "\\u2029"}


def _code(value: object) -> str:
    """A recorded value as inline code that cannot open any markdown structure.

    JSON-encoded, so a line break, a quote or a control character becomes an
    escape and the value stays on one line; a backtick (which would close the
    code span) and the line breaks JSON keeps literal are escaped as well.
    Inside a code span nothing else is interpreted.
    """
    text = json.dumps("" if value is None else str(value), ensure_ascii=False)
    text = text.replace("`", "\\u0060")
    for char, escaped in _LITERAL_LINE_BREAKS.items():
        text = text.replace(char, escaped)
    return f"`{text}`"


def _infer_app(start_url: str) -> str:
    """Derive an app name from the recording's start URL.

    Kept to the characters a domain label may hold: the name lands in the
    title heading and the frontmatter ``description:``.
    """
    try:
        domain = urlparse(start_url).netloc
    except ValueError:
        domain = ""
    domain = domain.replace("www.", "")
    label = re.sub(r"[^A-Za-z0-9-]", "", domain.split(".")[0]) if domain else ""
    if not label:
        return "Web"
    return label.title()


def infer_workflow_title(actions: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    """Infer a workflow title from actions and metadata."""
    app = _infer_app(metadata.get("start_url", ""))

    action_types = [a.get("action_type", "") for a in actions]

    if "SUBMIT" in action_types or any(
        "form" in a.get("description", "").lower() for a in actions[:3]
    ):
        verb = "Complete Form"
    elif any(t in ("TYPE", "FILL_FIELD") for t in action_types):
        verb = "Enter Data"
    elif "NAVIGATE" in action_types[:3]:
        verb = "Navigate"
    else:
        verb = "Workflow"

    return f"{app} - {verb}"


def generate_workflow_summary(actions: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    """Generate a one-line workflow summary from actions."""
    action_counts: dict[str, int] = {}
    for action in actions:
        action_type = action.get("action_type", "UNKNOWN")
        action_counts[action_type] = action_counts.get(action_type, 0) + 1

    parts = []
    field_count = action_counts.get("TYPE", 0) + action_counts.get("FILL_FIELD", 0)
    if field_count:
        parts.append(f"fill {field_count} field(s)")
    if action_counts.get("CLICK_BUTTON"):
        parts.append(f"click {action_counts['CLICK_BUTTON']} button(s)")
    if action_counts.get("SELECT"):
        parts.append(f"make {action_counts['SELECT']} selection(s)")
    if action_counts.get("NAVIGATE"):
        parts.append("navigate between pages")

    summary = ", ".join(parts) if parts else "complete a workflow"
    app = _infer_app(metadata.get("start_url", ""))
    return f"Recorded {len(actions)}-step {app} workflow to {summary}"


def _slug_for(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower().strip())
    return slug.strip("-")[:64].rstrip("-") or "recorded-workflow"


def action_to_markdown_step(action: dict[str, Any], step_num: int) -> str:
    """Convert a workflow action to a markdown step."""
    description = action.get("description", "Perform action")
    action_type = action.get("action_type", "")

    details = []

    if action_type in ("TYPE", "FILL_FIELD"):
        if action.get("field_type") == "password":
            details.append("**Security**: use secure credential storage, never a literal value")
        elif action.get("is_parameter"):
            details.append(f"**Parameter**: {_code(action.get('parameter_name') or 'value')}")
    elif action_type == "NAVIGATE":
        url = action.get("to_url") or action.get("url", "")
        if url:
            details.append(f"**URL**: {_code(url)}")
    elif action_type in ("CLICK_BUTTON", "CLICK"):
        css_selector = action.get("css_selector", "")
        if css_selector:
            details.append(f"**Selector**: {_code(css_selector)}")

    step_md = f"{step_num}. {_code(description)}"
    if details:
        step_md += "\n   - " + "\n   - ".join(details)
    return step_md


def build_skill_content(
    actions: list[dict[str, Any]],
    parameters: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> LiveSkillDraft:
    """Build Prometheus-format SKILL.md content from workflow actions.

    Args:
        actions: Workflow actions from :func:`events_to_actions`.
        parameters: Detected parameters.
        metadata: Recording metadata (start_url, duration_seconds, ...).

    Returns:
        A :class:`LiveSkillDraft` whose ``content`` is ready for
        ``SkillCreator.persist_skill_content()``.
    """
    title = infer_workflow_title(actions, metadata)
    name = _slug_for(title)
    description = generate_workflow_summary(actions, metadata)
    start_url = metadata.get("start_url", "")
    duration = int(metadata.get("duration_seconds") or 0)

    lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
        "---",
        "",
        f"# {title}",
        "",
        "## When to use",
        "Repeat the recorded browser workflow on "
        + (_code(start_url) if start_url else "the target application") + ".",
        "",
    ]

    if parameters:
        lines.append("## Parameters")
        lines.append("")
        for param in parameters:
            param_name = param.get("name", "unknown")
            param_type = param.get("type", "text")
            default_value = param.get("default_value", "")
            entry = f"- {_code(param_name)} ({_code(param_type)})"
            if default_value and param_type != "password":
                entry += f" — example: {_code(default_value)}"
            lines.append(entry)
            if param.get("description"):
                lines.append(f"  {_code(param['description'])}")
        lines.append("")

    lines.append("## Steps")
    lines.append("")
    for i, action in enumerate(actions, 1):
        lines.append(action_to_markdown_step(action, i))
    lines.append("")

    lines.append("## Notes")
    lines.append("- Captured with the live recorder browser extension (deterministic DOM trace, no vision model).")
    if start_url:
        lines.append(f"- Recording started at {_code(start_url)}")
    if duration:
        lines.append(f"- Original run took ~{duration} seconds over {len(actions)} steps.")
    lines.append("- Selectors reflect the UI at recording time; re-verify them if the application changes.")
    lines.append("")

    content = "\n".join(lines)
    log.info("Synthesized skill '%s' (%d steps, %d parameters)", name, len(actions), len(parameters))

    return LiveSkillDraft(
        name=name,
        title=title,
        description=description,
        content=content,
        step_count=len(actions),
        parameter_count=len(parameters),
    )
