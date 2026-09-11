"""An exhausted quota must not be classified as a retryable rate limit.

THE DEFECT
----------
`OpenAICompatProvider._call_once` opened a streaming response and called
`response.raise_for_status()` without reading the body first.

On a streamed response the content has not been fetched, so `response.text`
raises `httpx.ResponseNotRead`. `classify_turn_error` catches that and degrades
to an empty body — deliberately, and it documents why: "the body is optional:
it only sharpens BILLING detection". With no body, a 429 can only be read as a
rate limit.

But the body is the ONLY reliable billing signal. Providers disagree on the
status code (Anthropic 400, others 402/429), which is exactly why
`_BILLING_MARKERS` looks for "insufficient_quota" and friends in the text.

Measured before the fix, on a real 429 whose streamed body says
`insufficient_quota`:

    response.text  : raises ResponseNotRead — the body was never read
    classified kind: rate_limit
    is_terminal    : False      <- fallback fires only when True

So an account that cannot pay was retried with backoff, forever, and never
degraded to the fallback model. `anthropic.py` already did the right thing
(`await response.aread()` before `raise_for_status()`); this path did not.

WHY THE FIXTURE IS BUILT THE WAY IT IS
---------------------------------------
The body must arrive as a STREAM and be left unread. A first attempt used a
plain `httpx.MockTransport` handler returning `httpx.Response(429,
content=...)`, which pre-loads the content — `response.text` worked, the kind
came back "billing", and the defect was invisible. A fixture that cannot
exhibit the defect measures nothing about it, so these use
`httpx.AsyncByteStream`.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.api.turn_errors import (  # noqa: E402
    KIND_BILLING,
    KIND_RATE_LIMIT,
    classify_turn_error,
)
from prometheus.engine.fallback import is_terminal  # noqa: E402
from prometheus.engine.messages import ConversationMessage, TextBlock  # noqa: E402
from prometheus.providers.base import ApiMessageRequest  # noqa: E402
from prometheus.providers.openai_compat import OpenAICompatProvider  # noqa: E402

QUOTA_BODY = (
    b'{"error":{"message":"You exceeded your current quota",'
    b'"type":"insufficient_quota","code":"insufficient_quota"}}'
)
PLAIN_RATE_LIMIT_BODY = (
    b'{"error":{"message":"Rate limit reached for gpt-x",'
    b'"type":"requests","code":"rate_limit_exceeded"}}'
)


def _streamed(body: bytes):
    """A response whose body is a STREAM — unread until someone reads it."""

    class _Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield body

    return _Stream()


@pytest.fixture
def failing_with(monkeypatch):
    def _install(status: int, body: bytes):
        async def handler(request):
            return httpx.Response(
                status, stream=_streamed(body),
                headers={"content-type": "application/json"},
            )

        real = httpx.AsyncClient

        class _Patched(real):
            def __init__(self, *a, **k):
                k["transport"] = httpx.MockTransport(handler)
                super().__init__(*a, **k)

        monkeypatch.setattr(httpx, "AsyncClient", _Patched)

        provider = OpenAICompatProvider(
            api_key="sk-FAKE-not-a-key", model="gpt-x",
            base_url="https://api.example.invalid/v1",
        )
        request = ApiMessageRequest(
            model="gpt-x",
            messages=[ConversationMessage(role="user", content=[TextBlock(text="hi")])],
            max_tokens=32,
        )

        async def run():
            async for _ in provider._call_once(request):
                pass

        with pytest.raises(httpx.HTTPStatusError) as caught:
            asyncio.run(run())
        return caught.value

    return _install


def test_the_error_body_is_read_before_raising(failing_with):
    """The root cause, asserted directly.

    `response.text` raising ResponseNotRead is the whole mechanism; every
    downstream misclassification follows from it.
    """
    exc = failing_with(429, QUOTA_BODY)
    assert exc.response.text, (
        "the response body is still unread — `.text` is empty or raises, which "
        "is what erases the billing signal"
    )
    assert "insufficient_quota" in exc.response.text


def test_an_exhausted_quota_is_terminal_and_falls_back(failing_with):
    """The consequence the defect actually had."""
    exc = failing_with(429, QUOTA_BODY)
    kind = classify_turn_error(exc).get("kind")

    assert kind == KIND_BILLING, (
        f"an exhausted quota classified as {kind!r}; with the body unread a 429 "
        f"can only look like a rate limit, so it is retried forever and the "
        f"fallback never fires"
    )
    assert is_terminal(kind) is True


def test_a_genuine_rate_limit_is_still_a_rate_limit(failing_with):
    """The fix must not make every 429 terminal.

    Without this, `kind = KIND_BILLING` unconditionally would satisfy the test
    above while abandoning a provider that was about to recover — which the
    fallback module's own docstring calls out as the failure to avoid.
    """
    exc = failing_with(429, PLAIN_RATE_LIMIT_BODY)
    kind = classify_turn_error(exc).get("kind")

    assert kind == KIND_RATE_LIMIT, f"a plain rate limit classified as {kind!r}"
    assert is_terminal(kind) is False


@pytest.mark.parametrize("status", [400, 402, 403])
def test_billing_markers_are_honoured_on_the_other_statuses_too(failing_with, status):
    """Providers disagree on the code; the body is what settles it."""
    exc = failing_with(status, QUOTA_BODY)
    assert classify_turn_error(exc).get("kind") == KIND_BILLING


def test_a_server_error_body_is_still_read(failing_with):
    """Reading applies to every 4xx/5xx, not only the one that motivated it."""
    exc = failing_with(500, b'{"error":{"message":"internal"}}')
    assert "internal" in exc.response.text
