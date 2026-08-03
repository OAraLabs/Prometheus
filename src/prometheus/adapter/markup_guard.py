"""Chat-template markup guard — reject leaked control tokens before dispatch.

THREE VALIDATORS, NONE VALIDATING CONTENT. Every tool call already passes three
checks, and not one of them asks what a string *says*:

  * **GBNF** constrains JSON *structure* — a string literal may contain any
    bytes, so the grammar cannot object to what is inside the quotes.
  * **The provider's ``json.loads``** validates *syntax* — faithfully, which
    means it faithfully carries garbage through.
  * **Pydantic** validates *type* — "is this a ``str``?" — and a string of
    chat-template markup is, unarguably, a ``str``.

So the markup satisfies all three and executes. It did: 13 of the 2,591 tool
calls with a recorded ``parsed_tool_call`` carried it, and **every one has
success=1** — including a live ``wiki_compile`` on a Telegram session whose
``entity_name`` was ``{"}}<tool_call|><|tool_response>call:[]}``. The flywheel
is only where it became visible; the values had already run.

THE PATTERN IS DERIVED FROM OBSERVED OUTPUT, NOT INVENTED. Gemma does not emit
well-formed ``<|name|>`` tokens — it emits half-formed ones with the pipe on
either side. Across the corrupt corpus the shapes were ``<|tool_response>``,
``<tool_call|>``, ``<|tool_call>``, ``<channel|>``, ``<|channel>``,
``<|thought|>`` and ``<|"|>``. A conventional ``<\\|[a-z_]+\\|>`` would have
matched **none of the eight single-piped ones**. Hence: an angle-bracketed span
with a pipe adjacent to the opening ``<`` or the closing ``>``.

THE THRESHOLD IS MEASURED, NOT GUESSED. Rejecting on a single marker
over-rejects: a real, successful ``bash`` call on 2026-07-24 carried the Python
regex ``r'<([a-zA-Z0-9_:-]+)(?:\\s|>)'``, which legitimately contains ``|>``.
Requiring **two** markers scored **zero** false positives across all 2,591
parsed calls on record while still catching all 13 corrupt ones — the minimum
observed in a corrupt value is exactly 2 — and it leaves searching for the
tokens themselves working, which is a real operation someone performs while
investigating exactly this bug. Over-rejection on the dispatch path is its own
outage, so the line is drawn where the evidence put it.

Deliberately NOT a sanitizer: a repaired-in-place value is a value nobody saw.
The call is rejected, the model is told precisely what was wrong, and the
existing retry loop does the work — which also turns the corrupt attempt into
the *rejected* side of a training pair instead of the *chosen* side.
"""

from __future__ import annotations

import re
from typing import Any

# An angle-bracketed span whose pipe sits against either bracket. The {0,64}
# bound keeps a stray '<' ... '|>' pages apart in a long string from being
# spliced into one spurious "marker".
MARKUP_RE = re.compile(r"<\|[^<>]{0,64}>|<[^<>]{0,64}\|>")

# Markers required *within a single string argument* before it is rejected.
# See the module docstring: 1 over-rejects on real traffic, 2 measured clean.
REJECT_AT = 2


def find_markup(value: Any) -> list[str]:
    """Return every template-marker occurrence in ``value`` (empty if not a str)."""
    if not isinstance(value, str):
        return []
    return MARKUP_RE.findall(value)


def scan_arguments(tool_input: Any) -> dict[str, list[str]]:
    """Map argument path -> markers, for string leaves at or over the threshold.

    Walks nested dicts/lists so a corrupt value nested inside a structured
    argument is not missed. The threshold applies **per string leaf**, not to
    the call as a whole: two unrelated single-marker arguments are two
    legitimate strings, not one corrupt call.
    """
    found: dict[str, list[str]] = {}
    _walk(tool_input, "", found)
    return found


def _walk(node: Any, path: str, out: dict[str, list[str]]) -> None:
    if isinstance(node, str):
        hits = MARKUP_RE.findall(node)
        if len(hits) >= REJECT_AT:
            out[path or "<argument>"] = hits
    elif isinstance(node, dict):
        for key, value in node.items():
            _walk(value, f"{path}.{key}" if path else str(key), out)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _walk(value, f"{path}[{index}]", out)


def describe(found: dict[str, list[str]]) -> str:
    """One-line summary of a scan result, for telemetry and logs."""
    return "; ".join(
        f"{path}: {len(markers)} markers {sorted(set(markers))[:3]}"
        for path, markers in sorted(found.items())
    )


def rejection_message(tool_name: str, found: dict[str, list[str]]) -> str:
    """Specific, actionable feedback — the model must know what to change.

    Names the offending argument and shows the actual markers, in the same
    spirit as the structured schema errors: a generic "try again" teaches
    nothing and burns a round.
    """
    parts = []
    for path, markers in sorted(found.items()):
        sample = ", ".join(repr(m) for m in sorted(set(markers))[:3])
        parts.append(f"argument {path!r} contains {len(markers)} of them ({sample})")
    return (
        f"Rejected call to {tool_name}: chat-template control tokens leaked into "
        f"the arguments — " + "; ".join(parts) + ". These are decoding artifacts, "
        "not text. Re-issue the call with only the literal value you intended for "
        "that argument, with no <|...|> markers."
    )
