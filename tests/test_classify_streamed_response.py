"""classify_turn_error must work on STREAMED responses — which is all of them.

Every provider call in this daemon streams (`stream_message` -> `_call_once` ->
`raise_for_status()` on a response whose body was never read). Reading `.text` on such a response
raises `httpx.ResponseNotRead`, and `getattr(response, "text", "")` does NOT absorb it — getattr's
default only covers AttributeError. The exception escaped to this module's defensive handler,
which returns KIND_UNKNOWN.

So the module whose entire purpose is turning a provider failure into an actionable kind answered
"unknown" for essentially every real failure. It looked correct in tests because a hand-built
`httpx.Response(..., text=...)` IS readable — the fixture was the wrong shape, not the code.

Found when a deliberately-401'd provider failed to trigger the fallback: `kind='unknown'
terminal=False`.
"""

from __future__ import annotations

import httpx
import pytest

from prometheus.api.turn_errors import (
    KIND_AUTH,
    KIND_BILLING,
    KIND_PROVIDER_ERROR,
    KIND_RATE_LIMIT,
    classify_turn_error,
)

URL = "http://127.0.0.1:8005/api/v1/chat/completions"


def _streamed(status: int, body: bytes = b"nope") -> httpx.HTTPStatusError:
    """A response shaped like the ones providers actually raise: never read."""
    req = httpx.Request("POST", URL)
    resp = httpx.Response(status, request=req, stream=httpx.ByteStream(body))
    return httpx.HTTPStatusError(f"{status}", request=req, response=resp)


def test_the_body_really_is_unreadable_on_these():
    """Guard the premise: if httpx ever makes .text safe, this file's reason for existing moved."""
    exc = _streamed(401)
    with pytest.raises(httpx.ResponseNotRead):
        _ = exc.response.text


@pytest.mark.parametrize(
    "status,expected",
    [(401, KIND_AUTH), (403, KIND_AUTH), (429, KIND_RATE_LIMIT), (500, KIND_PROVIDER_ERROR),
     (502, KIND_PROVIDER_ERROR), (402, KIND_BILLING)],
)
def test_a_streamed_failure_classifies_by_status(status, expected):
    result = classify_turn_error(_streamed(status))
    assert result["kind"] == expected, f"{status} on a streamed response gave {result['kind']!r}"
    assert result["status"] == status, "the status must survive even with no body"


def test_a_readable_body_still_sharpens_billing_detection():
    """Losing the body must DEGRADE the answer, never discard it — when it is there, use it."""
    req = httpx.Request("POST", URL)
    resp = httpx.Response(403, request=req, text="Your credit balance is too low")
    result = classify_turn_error(httpx.HTTPStatusError("403", request=req, response=resp))
    assert result["kind"] == KIND_BILLING, "a readable billing body still outranks plain auth"


def test_a_streamed_403_without_a_body_is_auth_not_billing():
    """The body is what distinguishes them; with none, the status-only answer must still be useful."""
    assert classify_turn_error(_streamed(403))["kind"] == KIND_AUTH
