"""The arguments an operator is shown when asked to approve a tool call.

WHY THIS EXISTS
---------------
The approval prompt named the tool and the reason and **nothing else**
(``agent_loop`` passed only ``(tool_name, decision.reason)``). For a file
write that was survivable, because the reason carries the path. For a desktop
action it is not: ``computer_type_text requires confirmation`` tells the
operator neither what is being typed nor where, and an approval given on that
basis is not consent to anything in particular.

Ruled 2026-09-18 (MCP consent survey) and promoted to a prerequisite on
2026-09-19: **the approval must show the arguments being approved.**

THREE DECISIONS THIS FILE MAKES, EACH FOR A STATED REASON
---------------------------------------------------------
1. **Redaction reuses the audit redactor, deliberately.** ``AuditLogger``
   already carries two layers — by NAME (``token``/``secret``/``api_key``…)
   and by SHAPE (URL userinfo, query values, opaque UPPER_SNAKE assignments) —
   and the second layer exists precisely because the first is incomplete by
   construction. A second, parallel redactor here would drift from it, and
   the drift would be invisible until something leaked. One redactor.

   ⚠ It is NOT sufficient on its own and must not be described as if it were.
   An argument can carry a secret that neither layer matches — a bare
   passphrase in a ``text`` field has no name beside it and no distinctive
   shape. That is a real residual risk of showing arguments at all, and the
   answer is that the operator seeing their own machine's prompt is the
   intended reader. It is recorded here rather than left for someone to
   discover.

2. **Truncation is per-value AND overall.** A single huge argument must not
   push the actual decision off a phone screen, and a call with forty
   arguments must not either. Both limits are stated as constants rather than
   buried in a format string.

3. **The ``…`` is never silent.** A truncated value says how much was cut.
   "Approve this text" where the text is quietly the first 80 of 4000
   characters is the same defect as a narrow prompt for a wide grant.
"""

from __future__ import annotations

from typing import Any

from prometheus.permissions.audit import AuditLogger

#: Longest single rendered value before it is cut. Chosen against the audit
#: summary (200) and reason (500) limits already in the system: an argument is
#: shown alongside a reason, so it gets less room than the reason does.
MAX_VALUE_CHARS = 160

#: Longest whole rendered block. A Telegram message and a Beacon card both
#: have to stay glanceable; past this the operator is scrolling rather than
#: deciding.
MAX_TOTAL_CHARS = 700

#: Arguments never worth a line in a consent prompt: machine plumbing the
#: operator cannot act on. Kept SHORT and justified — an exclusion list is how
#: the thing you needed to see goes missing, so nothing is added here without
#: a reason that survives "what if that was the dangerous part?".
_NOISE_KEYS: frozenset[str] = frozenset({
    "session",        # driver session handle, opaque
    "snapshot_id",    # snapshot binding — validated in code, not by a human
    "element_token",  # opaque per-snapshot handle, unreadable by design
})


def redact_arguments(
    arguments: dict[str, Any] | None,
    *,
    max_value: int = MAX_VALUE_CHARS,
    max_total: int = MAX_TOTAL_CHARS,
) -> dict[str, str] | None:
    """Render tool arguments for human consent: redacted, truncated, ordered.

    Returns ``None`` when there is nothing to show, so a caller can tell
    "no arguments" from "arguments that rendered empty".
    """
    if not arguments:
        return None

    out: dict[str, str] = {}
    total = 0
    for key in sorted(arguments):
        if key in _NOISE_KEYS or key.startswith("_"):
            continue
        rendered = _render_one(arguments[key], max_value)
        if rendered is None:
            continue
        # Redact through the audit layers — one redactor for the whole system.
        # Applied to "key=value" so the BY-NAME layer can see the name, which
        # is the half that catches `token`, `api_key`, `password`.
        masked = AuditLogger.redact(f"{key}={rendered}")
        masked = masked[len(key) + 1:] if masked.startswith(f"{key}=") else masked
        if total + len(masked) > max_total:
            out["…"] = f"({len(arguments) - len(out)} more argument(s) not shown)"
            break
        out[key] = masked
        total += len(masked)
    return out or None


def _render_one(value: Any, max_value: int) -> str | None:
    """One argument value as a string, or None to omit it entirely."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        inner = ", ".join(str(v) for v in value)
        return _truncate(f"[{inner}]", max_value)
    if isinstance(value, dict):
        if not value:
            return None
        return _truncate(str(value), max_value)
    text = str(value)
    if not text.strip():
        return None
    # Newlines collapse: a multi-line value must not turn one prompt line into
    # thirty, and a leading blank line must not make the value look empty.
    text = " ⏎ ".join(line for line in text.splitlines() if line.strip())
    return _truncate(text, max_value)


def _truncate(text: str, limit: int) -> str:
    """Cut to *limit*, and SAY how much was cut. Never a silent ellipsis."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (+{len(text) - limit} more chars)"


def format_arguments(rendered: dict[str, str] | None) -> list[str]:
    """Prompt lines for a rendered argument map. Empty list when there is none.

    One line per argument rather than a single joined blob: the operator is
    scanning for the one value that decides the answer, and a wall of
    ``k=v, k=v, k=v`` is the shape that gets skimmed.
    """
    if not rendered:
        return []
    return [f"  {key}: {value}" for key, value in rendered.items()]
