"""A mid-stream error must not be reported as a completed message.

THE DEFECT
----------
The Messages API can report a failure PART-WAY THROUGH a successful response:
HTTP 200, headers sent, some deltas delivered, then

    {"type": "error", "error": {"type": "overloaded_error", ...}}

`_call_once`'s event chain had no branch for `error`, so the event fell through
and the loop simply ended. Execution continued to the bottom of the method,
which unconditionally yields `ApiMessageCompleteEvent` — announcing a COMPLETE
message built from whatever partial content had arrived.

Measured before the fix, with an `overloaded_error` arriving after one text
delta:

    exception raised : NONE
    events yielded   : ['ApiTextDeltaEvent', 'ApiMessageCompleteEvent']
    stop_reason      : None
    text             : ''

Empty, because the content block had never been closed — `content_block_stop`
is what appends it. So the caller was handed a successful, empty reply. It
cannot distinguish that from a model that chose to say nothing.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.engine.messages import ConversationMessage, TextBlock  # noqa: E402
from prometheus.providers.anthropic import (  # noqa: E402
    _RETRYABLE_STATUS_CODES,
    AnthropicProvider,
    AnthropicStreamError,
)
from prometheus.providers.base import (  # noqa: E402
    ApiMessageCompleteEvent,
    ApiMessageRequest,
)
from prometheus.providers.retry import _status_of  # noqa: E402

MESSAGE_START = 'data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}'
BLOCK_START = 'data: {"type":"content_block_start","content_block":{"type":"text","text":""}}'
TEXT_DELTA = ('data: {"type":"content_block_delta",'
              '"delta":{"type":"text_delta","text":"The answer is "}}')
BLOCK_STOP = 'data: {"type":"content_block_stop"}'


def _error_event(kind: str = "overloaded_error", msg: str = "Overloaded") -> str:
    return f'data: {{"type":"error","error":{{"type":"{kind}","message":"{msg}"}}}}'


@pytest.fixture
def stream_of(monkeypatch):
    """Drive the real `_call_once` over a scripted SSE body on a 200 response."""

    def _install(lines: list[str]):
        class _Response:
            status_code = 200
            text = ""

            async def aread(self):
                return b""

            def raise_for_status(self):
                return None

            async def aiter_lines(self):
                for line in lines:
                    yield line

        class _StreamCtx:
            async def __aenter__(self):
                return _Response()

            async def __aexit__(self, *exc):
                return False

        class _Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            def stream(self, *a, **k):
                return _StreamCtx()

        monkeypatch.setattr(httpx, "AsyncClient", _Client)

        provider = AnthropicProvider(api_key="sk-ant-FAKE-not-a-key", model="claude-opus-5")
        request = ApiMessageRequest(
            model="claude-opus-5",
            messages=[ConversationMessage(role="user", content=[TextBlock(text="hi")])],
            max_tokens=64,
        )

        async def collect():
            events = []
            async for event in provider._call_once(request):
                events.append(event)
            return events

        return collect

    return _install


def test_a_mid_stream_error_raises_instead_of_completing(stream_of):
    collect = stream_of([MESSAGE_START, BLOCK_START, TEXT_DELTA, _error_event()])

    with pytest.raises(AnthropicStreamError) as caught:
        asyncio.run(collect())

    assert "overloaded_error" in str(caught.value)
    assert caught.value.error_type == "overloaded_error"


def test_no_complete_message_event_is_emitted_after_an_error(stream_of):
    """The precise defect: a COMPLETE message announced for a failed stream."""
    collect = stream_of([MESSAGE_START, BLOCK_START, TEXT_DELTA, _error_event()])

    events = []

    async def run():
        async for event in AnthropicProvider(
            api_key="sk-ant-FAKE-not-a-key", model="claude-opus-5"
        )._call_once(
            ApiMessageRequest(
                model="claude-opus-5",
                messages=[ConversationMessage(role="user", content=[TextBlock(text="hi")])],
                max_tokens=64,
            )
        ):
            events.append(event)

    with pytest.raises(AnthropicStreamError):
        asyncio.run(run())

    completions = [e for e in events if isinstance(e, ApiMessageCompleteEvent)]
    assert not completions, (
        "a complete-message event was emitted for a stream that errored; its "
        f"content was {completions[0].message.content!r}"
    )


def test_a_clean_stream_still_completes(stream_of):
    """The fix must not turn a good stream into an error.

    Without this, `raise` on every event would pass the two tests above.
    """
    collect = stream_of([
        MESSAGE_START, BLOCK_START, TEXT_DELTA, BLOCK_STOP,
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        '"usage":{"output_tokens":5}}',
    ])
    events = asyncio.run(collect())

    completions = [e for e in events if isinstance(e, ApiMessageCompleteEvent)]
    assert len(completions) == 1, [type(e).__name__ for e in events]
    assert completions[0].stop_reason == "end_turn"
    text = "".join(
        b.text for b in completions[0].message.content if hasattr(b, "text")
    )
    assert text == "The answer is "


@pytest.mark.parametrize("kind,status", [
    ("overloaded_error", 529),
    ("rate_limit_error", 429),
    ("api_error", 500),
    ("invalid_request_error", 400),
])
def test_the_error_carries_the_status_the_retry_loop_reads(stream_of, kind, status):
    """One policy for both shapes of the same condition.

    `retry._status_of` is what decides retryability. Without a status, an
    overload delivered mid-stream would be classified differently from the
    identical overload delivered as a 529, purely because of where in the
    response it appeared.
    """
    collect = stream_of([MESSAGE_START, _error_event(kind, "boom")])

    with pytest.raises(AnthropicStreamError) as caught:
        asyncio.run(collect())

    assert _status_of(caught.value) == status
    assert (status in _RETRYABLE_STATUS_CODES) == (kind != "invalid_request_error"), (
        "the mapped status disagrees with the provider's own retry policy"
    )


def test_an_unrecognised_error_type_surfaces_and_is_not_retryable(stream_of):
    """A new error type must fail loudly, not be retried on a guess.

    Retrying something nobody has classified is how one hard failure becomes
    four hard failures.
    """
    collect = stream_of([MESSAGE_START, _error_event("brand_new_error", "?")])

    with pytest.raises(AnthropicStreamError) as caught:
        asyncio.run(collect())

    assert _status_of(caught.value) is None
    assert _status_of(caught.value) not in _RETRYABLE_STATUS_CODES
    assert "brand_new_error" in str(caught.value)
