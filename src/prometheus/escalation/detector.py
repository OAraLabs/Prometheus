"""Tier-1 deterministic failure detector for agent turns.

Pure, regex/deterministic only — no LLM judging (Tier-2 self-eval is
explicitly out of scope per SPRINT-TEACHER-ESCALATION.md). Consumed by:

- the teacher-escalation trigger (``escalation.teacher``), and
- the bake-off harness (``BAKEOFF-harness.md``), which imports this module
  standalone from a branch checkout.

DEPENDENCY POLICY: stdlib only. No ``prometheus.*`` imports, ever — the
bake-off loads this file by path outside any installed package.

Input contract — ``detect_failure(tool_results, final_reply)``:

``tool_results`` is this turn's tool activity, in call order, as dicts in
the repo's trace shape (see ``learning/skill_creator.py::_format_trace``)
plus an error flag:

    {"tool_name": str, "arguments": dict|str, "result": str, "is_error": bool}

``is_error`` is optional; when absent, an error-shaped ``result`` text is
used as a fallback (anchored, so informative results like grep's
"(no output)" never count as errors). ``final_reply`` is the turn's final
assistant text as delivered to the user.

Provenance: clean-room reimplementation inspired by the Odysseus project's
teacher-escalation design (MIT). Design knowledge only; no source copied.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Tunables (module constants so the bake-off and tests can reference them)
# ---------------------------------------------------------------------------

# Clarification stall: a reply this short that is only a question back at the
# user, with zero tool activity, on what was presumably a concrete instruction.
# 80 chars is tuned so terse stalls ("Which file do you mean?") trigger while
# substantive clarifying questions that offer interpretations (the legitimate
# case on a genuinely ambiguous request) run longer and do not.
CLARIFICATION_STALL_MAX_CHARS = 80

# Repetition: the same tool called with byte-identical canonical args this
# many times in one turn. Three matches the captured collapse arcs — by the
# third identical attempt the model is looping, not exploring.
REPETITION_THRESHOLD = 3

# ---------------------------------------------------------------------------
# Pattern tables — one comment per pattern naming the REAL failure it catches.
# ---------------------------------------------------------------------------

CAPABILITY_DENIAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # "I don't have a tool for that": the model denies tooling it actually has
    # (or hallucinates absence after schema confusion). Observed in the
    # post-collapse replies of the D1 family (FINDINGS-TOOLCALLING-2026-06-10)
    # where the model gave up on calling anything.
    ("no_tool", re.compile(
        r"\bI\s+(?:don't|do\s+not|dont)\s+have\s+(?:a|the|any)\s+tool\b", re.I)),
    # "no such tool/action" / "unknown tool/action": the model surfaced a
    # registry rejection to the user verbatim instead of recovering with a
    # real tool name — the user gets an error message as the "answer".
    ("unknown_tool", re.compile(
        r"\b(?:no\s+such|unknown)\s+(?:tool|action)\b", re.I)),
    # "I am unable to access/run/...": capability denial. Verb-constrained on
    # purpose — honest negative outcomes ("I was unable to find any matches")
    # are a correct answer, not a failure, and must not match.
    ("unable_to_act", re.compile(
        r"\bI\s*(?:am|'m)\s+unable\s+to\s+"
        r"(?:access|execute|run|perform|do|use|call|invoke|complete)\b", re.I)),
    # "I cannot/can't execute ...": same denial class, same verb constraint.
    ("cannot_act", re.compile(
        r"\bI\s+can(?:not|'t)\s+"
        r"(?:access|execute|run|perform|use|call|invoke)\b", re.I)),
    # "I don't have access/permission/the ability to ...": denial of standing
    # capability (vs a one-shot tool error) — the model believes the harness
    # can't do the thing at all, so it will fail this whole task class.
    ("no_access", re.compile(
        r"\bI\s+(?:don't|do\s+not)\s+have\s+"
        r"(?:access|permission|the\s+ability)\s+to\b", re.I)),
    # The agent loop's own terminal reply when the circuit breaker gives up:
    # "Circuit breaker tripped: ... The model cannot produce valid tool calls
    # for this request." (engine/agent_loop.py:902,1062). If the user sees
    # this, the turn failed by definition — the captured 2026-06-10 collapse
    # arc ("can you hear me now?") ended exactly here. Highest-precision
    # signal in the set.
    ("breaker_terminal", re.compile(r"\bCircuit\s+breaker\s+tripped\b", re.I)),
]

# Retry-plan acknowledgments: if the reply admits the error AND commits to a
# next step, the model is still driving — an error result alone is not a
# failed turn. (The unrecovered-error signal requires the ABSENCE of these.)
RETRY_ACK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:let\s+me|I(?:'ll|\s+will)|going\s+to)\s+(?:re)?try\b", re.I),
    re.compile(r"\btry(?:ing)?\s+(?:again|a\s+different|another)\b", re.I),
    re.compile(r"\bretry(?:ing)?\b", re.I),
    re.compile(r"\b(?:instead|alternative(?:ly)?|different\s+approach)\b", re.I),
    re.compile(r"\bswitch(?:ing)?\s+to\b", re.I),
    re.compile(r"\b(?:let\s+me|I(?:'ll|\s+will))\s+(?:use|check|look|read|fix|attempt)\b", re.I),
]

# Error-shaped result text, used ONLY when a result dict carries no explicit
# ``is_error`` flag (e.g. transcripts recorded outside the live loop).
# Anchored at the start for error/exception/traceback so mid-text mentions
# ("0 errors found") and informative results ("(no output)") don't count;
# timeouts match anywhere because tools phrase them freely.
ERROR_TEXT_PATTERN = re.compile(
    r"(?:\A\s*(?:error|exception|traceback)\b)|(?:\btime(?:d\s+out|out)\b)",
    re.I,
)

# Stable signal identifiers (these are what land in telemetry payloads).
SIGNAL_UNRECOVERED_ERROR = "unrecovered_tool_error"
SIGNAL_CAPABILITY_DENIAL = "capability_denial"
SIGNAL_CLARIFICATION_STALL = "clarification_stall"
SIGNAL_REPETITION = "repetition"
SIGNAL_EMPTY_REPLY = "empty_reply_after_tools"


@dataclass
class FailureVerdict:
    """Outcome of Tier-1 failure detection for one agent turn."""

    failed: bool
    reasons: list[str] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)


def _canonical_args(arguments: object) -> str:
    """Canonicalize call arguments so dict ordering doesn't defeat the
    repetition signal."""
    if isinstance(arguments, (dict, list)):
        try:
            return json.dumps(arguments, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return repr(arguments)
    return str(arguments)


def _is_error_result(tr: dict) -> bool:
    """True if a tool-result dict represents an error/timeout."""
    if "is_error" in tr:
        return bool(tr["is_error"])
    return bool(ERROR_TEXT_PATTERN.search(str(tr.get("result", ""))[:2000]))


def _acknowledges_retry(reply: str) -> bool:
    return any(p.search(reply) for p in RETRY_ACK_PATTERNS)


def detect_failure(tool_results: list, final_reply: str) -> FailureVerdict:
    """Classify one completed agent turn as failed or not.

    Pure: no I/O, no globals mutated. Multiple signals may fire on one turn;
    all are reported. ``tool_results`` dicts follow the module-docstring
    contract; unknown keys are ignored.
    """
    reasons: list[str] = []
    matched: list[str] = []
    reply = (final_reply or "").strip()

    # Signal: empty/whitespace reply after tool activity. The turn did work
    # and then said nothing — the captured collapse arcs end this way when
    # the breaker doesn't produce its terminal message.
    if tool_results and not reply:
        reasons.append(
            f"empty reply after {len(tool_results)} tool call(s)")
        matched.append(SIGNAL_EMPTY_REPLY)

    # Signal: capability denial (incl. the loop's breaker-terminal reply).
    for name, pattern in CAPABILITY_DENIAL_PATTERNS:
        m = pattern.search(reply)
        if m:
            reasons.append(f"capability denial in reply: {m.group(0)!r}")
            matched.append(f"{SIGNAL_CAPABILITY_DENIAL}:{name}")

    # Signal: clarification stall — the reply is nothing but a terse question
    # back at the user, with zero tool activity.
    if (
        not tool_results
        and reply
        and len(reply) <= CLARIFICATION_STALL_MAX_CHARS
        and reply.endswith("?")
    ):
        reasons.append(
            f"clarification stall: {len(reply)}-char question-only reply "
            f"with no tool calls")
        matched.append(SIGNAL_CLARIFICATION_STALL)

    # Signal: repetition — same tool, byte-identical canonical args, >= N
    # times in one turn. The model is looping, not exploring.
    counts: dict[tuple[str, str], int] = {}
    for tr in tool_results or ():
        key = (str(tr.get("tool_name", "")), _canonical_args(tr.get("arguments")))
        counts[key] = counts.get(key, 0) + 1
    for (tool_name, _args), n in counts.items():
        if n >= REPETITION_THRESHOLD:
            reasons.append(
                f"tool {tool_name!r} called with identical args {n}x in one turn")
            matched.append(SIGNAL_REPETITION)
            break  # one repetition reason is enough; keep reasons readable

    # Signal: unrecovered tool error — the turn ENDED on an error/timeout and
    # the reply commits to no retry plan. An error mid-turn that the model
    # recovered from (or ends acknowledging with a next step) is normal agent
    # behavior and must not fire.
    if tool_results:
        last = tool_results[-1]
        if _is_error_result(last) and not _acknowledges_retry(reply):
            reasons.append(
                f"turn ended on an error result from {last.get('tool_name', '?')!r} "
                f"with no retry plan in the reply")
            matched.append(SIGNAL_UNRECOVERED_ERROR)

    return FailureVerdict(failed=bool(reasons), reasons=reasons,
                          matched_patterns=matched)
