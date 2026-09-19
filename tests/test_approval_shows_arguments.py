"""The operator must see the arguments they are approving.

THE DEFECT
----------
The approval prompt named the tool and the reason and nothing else —
``agent_loop`` passed only ``(tool_name, decision.reason)``. For a file write
that survived, because the reason carries the path. For a desktop action it
does not: ``computer_type_text requires confirmation`` tells the operator
neither what is being typed nor where. An approval given on that basis is not
consent to anything in particular.

Ruled 2026-09-18 in the MCP consent survey and promoted to a prerequisite on
2026-09-19, because for ``type_text`` the arguments ARE the decision.

WHAT THESE TESTS HOLD
---------------------
That the arguments arrive, that they are scrubbed before they arrive, that
scrubbing happens ONCE at the point of capture (so no surface can forget), and
that truncation never hides how much it cut.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.permissions.approval_queue import (
    ApprovalQueue, PendingAction,
)
from prometheus.permissions.argument_view import (
    MAX_VALUE_CHARS, format_arguments, redact_arguments,
)


# ── REDACTION ───────────────────────────────────────────────────────────────

def test_a_named_secret_is_masked():
    out = redact_arguments({"api_key": "abcdef123456789", "app": "firefox"})
    assert "abcdef123456789" not in str(out)
    assert out["app"] == "firefox"


def test_a_bearer_token_is_masked():
    out = redact_arguments({"text": "Bearer sk-live-abcdefghijklmnop"})
    assert "sk-live-abcdefghijklmnop" not in str(out)


def test_a_tokenised_url_is_masked_whatever_the_parameter_is_called():
    """The SHAPE layer — the half that catches the name nobody thought of."""
    out = redact_arguments({"text": "https://x.test/cb?sig=QQQQQQQQQQQQQQQQ"})
    assert "QQQQQQQQQQQQQQQQ" not in str(out)


def test_redaction_is_the_audit_redactor_not_a_second_one():
    """One redactor. A parallel copy would drift with nothing to notice."""
    import inspect

    from prometheus.permissions import argument_view

    src = inspect.getsource(argument_view)
    assert "AuditLogger.redact" in src, (
        "argument_view stopped using the audit redactor — a second redaction "
        "implementation will drift from the first and the drift is invisible"
    )
    assert "re.compile" not in src, (
        "argument_view grew its own redaction patterns; use AuditLogger's"
    )


# ── TRUNCATION ──────────────────────────────────────────────────────────────

def test_a_long_value_is_cut_and_says_how_much():
    out = redact_arguments({"text": "x" * 500})
    assert len(out["text"]) < 500
    assert "more chars" in out["text"], (
        f"a truncated value did not say how much was hidden: {out['text']!r} — "
        f"approving 'this text' where the text is quietly the first 160 of 500 "
        f"characters is the same defect as a narrow prompt for a wide grant"
    )


def test_truncation_respects_the_stated_limit():
    out = redact_arguments({"text": "y" * 1000})
    head = out["text"].split("…")[0]
    assert len(head) <= MAX_VALUE_CHARS


def test_many_arguments_do_not_push_the_decision_off_the_screen():
    out = redact_arguments({f"k{i}": "v" * 100 for i in range(50)})
    assert "…" in out, "a 50-argument call rendered without any bound"


def test_newlines_do_not_explode_one_line_into_thirty():
    out = redact_arguments({"text": "a\nb\nc\nd"})
    assert "\n" not in out["text"]


# ── WHAT IS SHOWN, AND WHAT IS NOT ──────────────────────────────────────────

def test_opaque_plumbing_is_omitted():
    """A human cannot act on an element token; it is noise in a consent
    prompt. The exclusion list is short and justified for exactly that
    reason — an exclusion list is how the thing you needed goes missing."""
    out = redact_arguments({
        "element_token": "tok-abc", "snapshot_id": "s1", "session": "sess",
        "text": "hello",
    })
    assert out == {"text": "hello"}


def test_the_payload_is_never_omitted():
    out = redact_arguments({"text": "transfer 500 to account 9"})
    assert "transfer 500" in out["text"]


def test_no_arguments_is_distinguishable_from_empty_ones():
    assert redact_arguments(None) is None
    assert redact_arguments({}) is None
    assert redact_arguments({"element_token": "t"}) is None


def test_format_produces_one_line_per_argument():
    lines = format_arguments({"text": "hi", "app": "firefox"})
    assert len(lines) == 2
    assert all(ln.startswith("  ") for ln in lines)


# ── THE WIRE AND THE PROMPT ─────────────────────────────────────────────────

def _queue() -> ApprovalQueue:
    return ApprovalQueue(telegram_adapter=None, default_chat_id=None,
                         timeout_seconds=1)


def test_arguments_are_scrubbed_at_capture_not_per_surface():
    """One scrub, at construction. A surface cannot forget what it never had."""
    queue = _queue()

    async def go():
        task = asyncio.ensure_future(queue.request_approval(
            "computer_type_text", "reason",
            arguments={"text": "hello", "api_key": "abcdef123456789"},
        ))
        await asyncio.sleep(0.05)
        action = next(iter(queue.pending.values()))
        return action

    action = asyncio.run(go())
    assert action.arguments is not None
    assert "abcdef123456789" not in str(action.arguments), (
        "the RAW arguments were stored on the pending action; a surface "
        "rendering them would echo the secret"
    )
    assert action.arguments["text"] == "hello"


def test_the_beacon_wire_carries_the_scrubbed_arguments():
    queue = _queue()
    action = PendingAction(
        request_id="r1", tool_name="computer_type_text", description="d",
        arguments={"text": "hello"},
    )
    wire = queue.serialize_pending(action)
    assert "arguments" in wire, (
        "Beacon's approval card cannot show the arguments because they are "
        "not on the wire"
    )
    assert wire["arguments"] == {"text": "hello"}


def test_the_prompt_puts_the_arguments_above_the_answer_options():
    """A prompt that offers the answers before the thing being decided
    invites the reflex answer."""
    import re

    from prometheus.permissions import approval_queue

    src = approval_queue.__file__
    text = open(src).read()
    with_idx = text.index('lines.append("With:")')
    approve_idx = text.index('"/approve — approve this ONCE')
    assert with_idx < approve_idx, (
        "the arguments are rendered below the /approve line"
    )
