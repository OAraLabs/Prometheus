"""Honest async-promise validator — behavioral companion to managed tasks.

Pure and surface-agnostic. A gateway calls this at turn finalization with the
outgoing text and THIS TURN's new messages. If the text promises a later
self-notification tied to completion but no qualifying ``task_create`` was
registered this turn, it returns a one-sentence system-note correction — so the
model cannot make a promise it has no mechanism to keep — else ``None``.

CONSERVATIVE detection (per design): fires only on future-tense self-
notification BOUND to completion. It must NOT fire on conditionals
("...if I find anything"), reversed direction ("let me know if..."), or
within-turn progress ("I'll keep you posted as I work through this"). The
patterns live in ``PROMISE_PATTERNS`` as a single module-level constant for
easy tuning.
"""

from __future__ import annotations

import re
from typing import Any

# task_create on_complete values that actually deliver a later notification /
# re-engagement. (task_create has no "none" option today, but we read the value
# so the rule stays correct if one is ever added.)
_NOTIFYING_ON_COMPLETE = {"notify", "reengage", "both"}

# Conservative promise detection. Each pattern requires, in order:
#   1. future-tense first person — "I'll" / "I will"
#   2. a SELF-notification verb aimed at the user — "let you know" or
#      "{notify,ping,message,update,tell} you"
#   3. a completion connector — "when" / "once" / "after"
#   4. a completion anchor in the SAME clause — done / finished / ready / ...
# Requirements (3)+(4) are what reject conditionals ("if I find anything"),
# reversed direction ("let me know if..."), and within-turn progress ("keep you
# posted as I work"). ``[^.?!\n]*`` keeps the match inside one clause so a
# completion word in a later sentence cannot satisfy an earlier promise.
PROMISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bI(?:'ll|\s+will)\s+"
        r"(?:let\s+you\s+know|(?:notify|ping|message|update|tell)\s+you)\s+"
        r"(?:when|once|after)\b[^.?!\n]*\b"
        r"(?:done|finished|finish|finishes|complete|completes|completed|"
        r"ready|wraps?\s+up|wrapped\s+up)\b",
        re.IGNORECASE,
    ),
)

# One sentence, clearly a delimited system note (NOT posing as the model's own
# continuation), pointing at the honest alternative.
CORRECTION = (
    "(System note: no tracked task was started, so nothing will auto-notify you "
    "— ask me to check, or I can start a tracked task.)"
)


def _promises_async_notification(text: str) -> bool:
    return any(pat.search(text) for pat in PROMISE_PATTERNS)


def _has_qualifying_task(new_messages: list[Any]) -> bool:
    """True iff a ``task_create`` tool call with a notifying ``on_complete``
    appears in this turn's new messages (reads the ToolUseBlock's input args)."""
    for msg in new_messages or ():
        for block in getattr(msg, "tool_uses", None) or ():
            if getattr(block, "name", None) != "task_create":
                continue
            args = getattr(block, "input", None) or {}
            if args.get("on_complete", "notify") in _NOTIFYING_ON_COMPLETE:
                return True
    return False


def check_async_promise(text: str, new_messages: list[Any]) -> str | None:
    """Return a correction note iff *text* makes an async-completion promise AND
    no qualifying ``task_create`` appears in *new_messages*; else ``None``.

    Pure: no I/O, no globals. The caller decides whether/how to apply it.
    """
    if not text or not _promises_async_notification(text):
        return None
    if _has_qualifying_task(new_messages):
        return None
    return CORRECTION


def evaluate_and_record(
    text: str,
    new_messages: list[Any],
    *,
    provenance: str,
    session_id: str | None = None,
    telemetry: Any | None = None,
) -> str | None:
    """Gateway call-site convenience around :func:`check_async_promise`.

    - Gates on real user turns: re-engagement turns (``provenance != "user"``)
      are never corrected — those are the agent acting on a task result, not a
      user-facing promise.
    - Records a telemetry event on EVERY detection (so the lie-rate is
      measurable even though we also append the correction).
    - Returns the correction string to append, or ``None``.

    ``telemetry`` is injected (not imported) so this stays surface-agnostic and
    unit-testable; ``None`` disables recording.
    """
    if provenance != "user":
        return None
    correction = check_async_promise(text, new_messages)
    if correction is None:
        return None
    if telemetry is not None:
        try:
            telemetry.record_signal_event(
                signal_type="async_promise_uncovered",
                payload={"session_id": session_id},
                source_subsystem="honesty_validator",
            )
        except Exception:  # noqa: BLE001 — telemetry must never break a reply
            pass
    return correction
