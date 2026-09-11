"""Capture what `classify_turn_error` actually received, not a guess (#321).

WHY A HARNESS AND NOT A FIX
----------------------------
The fallback's terminal-kind predicate cannot express "retryable in seconds"
versus "retryable in three days" — both are `KIND_RATE_LIMIT`. Whether that is
a live defect depends on how one particular provider presents an exhausted
quota on the wire, and guessing that is the error mode the issue exists to
avoid.

WHAT IS ALREADY KNOWN, FROM REAL EVENTS
----------------------------------------
Three exhaustion failures on 2026-08-26, 08-29 and 08-31 logged exactly:

    httpx.HTTPStatusError: Client error '429 Too Many Requests' for url '...'

So the status is a plain **429**, not a 403 — which rules out "the classifier
already calls this billing" on status alone. The body was never read on that
path (fixed separately), and the headers were never logged at all.

The durable row for those same three events was
`{"exception_type": "HTTPStatusError"}` — the class name and nothing more.

WHAT THIS ADDS
---------------
Status and body were already captured. These tests pin the three that were not:
the rate/quota HEADERS, the classifier's CONCLUSION, and a durable row that
carries both. The point is stated in
`test_the_row_distinguishes_a_transient_429_from_an_exhausted_quota`: two
responses that are identical in status must produce different rows.
"""

from __future__ import annotations

import pytest

httpx = pytest.importorskip("httpx")

from prometheus.api.turn_errors import (  # noqa: E402
    QUOTA_HEADERS,
    quota_headers,
    redact_url,
)
from prometheus.learning.llm_envelope import _failure_summary  # noqa: E402


def _error(status: int, body: str, headers: dict[str, str] | None = None):
    """An httpx.HTTPStatusError shaped like a real provider rejection."""
    request = httpx.Request("POST", "https://provider.example.invalid/v1/chat/completions")
    response = httpx.Response(status, content=body.encode(), headers=headers or {},
                              request=request)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


# ── headers ──────────────────────────────────────────────────────────────────


def test_quota_headers_are_captured():
    exc = _error(429, "{}", {"retry-after": "259200",
                             "x-ratelimit-remaining-tokens": "0"})
    got = quota_headers(exc.response)
    assert got["retry-after"] == "259200"
    assert got["x-ratelimit-remaining-tokens"] == "0"


def test_the_header_capture_is_an_allowlist_not_a_dump():
    """A raw header dump into a log trades a blind spot for a credential leak."""
    exc = _error(429, "{}", {
        "retry-after": "20",
        "authorization": "Bearer SUPERSECRET",
        "set-cookie": "session=SUPERSECRET",
        "x-internal-routing-key": "SUPERSECRET",
    })
    got = quota_headers(exc.response)
    assert got == {"retry-after": "20"}
    assert "SUPERSECRET" not in repr(got)
    assert "authorization" not in QUOTA_HEADERS


def test_header_capture_never_raises_on_a_non_response():
    assert quota_headers(None) == {}
    assert quota_headers(object()) == {}


def test_a_logged_url_never_carries_its_query_string():
    """Gemini puts the API key in `?key=`; this value goes to a log."""
    out = redact_url("https://host.example.invalid/v1/chat?key=SUPERSECRET&x=1")
    assert "SUPERSECRET" not in out
    assert out.startswith("https://host.example.invalid/v1/chat")


# ── the durable row ──────────────────────────────────────────────────────────


def test_the_failure_row_carries_more_than_a_class_name():
    """The three real events left only `{"exception_type": "HTTPStatusError"}`."""
    exc = _error(429, '{"error":{"message":"rate limit"}}', {"retry-after": "20"})
    row = _failure_summary(exc)
    assert row["exception_type"] == "HTTPStatusError"
    assert row["status"] == 429
    assert row["kind"], "the classifier's conclusion is the decision; record it"
    assert row["quota_headers"]["retry-after"] == "20"


def test_the_row_distinguishes_a_transient_429_from_an_exhausted_quota():
    """THE POINT OF THE HARNESS.

    Identical status. The rows must not be identical, or Monday's event tells us
    nothing we did not already know.
    """
    transient = _failure_summary(
        _error(429, '{"error":{"message":"Too many requests, slow down"}}',
               {"retry-after": "20", "x-ratelimit-remaining-tokens": "48000"})
    )
    exhausted = _failure_summary(
        _error(429, '{"error":{"message":"You exceeded your current quota",'
                    '"type":"insufficient_quota"}}',
               {"retry-after": "259200", "x-ratelimit-remaining-tokens": "0"})
    )

    assert transient["status"] == exhausted["status"] == 429
    assert transient != exhausted, (
        "two failures that need opposite responses produced identical rows"
    )
    # The headers alone separate them, before anyone argues about body text.
    assert transient["quota_headers"]["retry-after"] == "20"
    assert exhausted["quota_headers"]["retry-after"] == "259200"
    assert exhausted["quota_headers"]["x-ratelimit-remaining-tokens"] == "0"


def test_the_row_is_still_written_when_classification_is_impossible():
    """A diagnostic that can fail the write it decorates is worse than none."""
    row = _failure_summary(RuntimeError("no response attached"))
    # `kind: unknown` is kept — "the classifier could not tell" is a real
    # finding. The placeholder provider is not: `classify_turn_error` never
    # returns an empty provider, it returns the literal "the model provider" so
    # its user-facing hint reads as a sentence. Storing that would be noise
    # shaped like data.
    assert row == {"exception_type": "RuntimeError", "kind": "unknown"}
    assert "provider" not in row


def test_an_exhausted_quota_body_is_already_classified_terminal():
    """Documents the OTHER half, so Monday's capture is read correctly.

    If the provider's body matches `_BILLING_MARKERS`, the classifier already
    calls it billing and the fallback already fires — there is no bug left to
    fix. This test states that precondition so the capture is interpreted
    against it rather than re-litigated.
    """
    from prometheus.api.turn_errors import KIND_BILLING, classify_turn_error
    from prometheus.engine.fallback import is_terminal

    exc = _error(429, '{"error":{"type":"insufficient_quota"}}')
    detail = classify_turn_error(exc)
    assert detail["kind"] == KIND_BILLING
    assert is_terminal(detail["kind"]) is True


def test_a_bare_429_with_no_marker_is_still_only_a_rate_limit():
    """And this is the case that would be a real defect. Stated, not assumed."""
    from prometheus.api.turn_errors import KIND_RATE_LIMIT, classify_turn_error
    from prometheus.engine.fallback import is_terminal

    exc = _error(429, '{"error":{"message":"Requests rate limit exceeded"}}')
    detail = classify_turn_error(exc)
    assert detail["kind"] == KIND_RATE_LIMIT
    assert is_terminal(detail["kind"]) is False
