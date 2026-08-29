"""A retry must not replay over text the consumer already received (issue #293).

Every provider retries by calling ``_call_once`` again. That is correct BEFORE the first event
and wrong after it: whatever was already yielded is already on the user's screen, so a second
attempt appends a second — possibly contradictory — answer to the first.

Reproduced before the fix as ``"The answer is 4" + "The answer is 7"`` arriving as one reply.

Both directions are tested for every provider, because over-fixing is as bad as the bug: killing
retries for a failure that happened BEFORE any output would trade a rare wrong answer for a
common avoidable failure.
"""

from __future__ import annotations

import httpx
import pytest

from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.providers.anthropic import AnthropicProvider
from prometheus.providers.base import ApiMessageRequest, ApiTextDeltaEvent
from prometheus.providers.llama_cpp import LlamaCppProvider
from prometheus.providers.ollama import OllamaProvider
from prometheus.providers.openai_compat import OpenAICompatProvider
from prometheus.providers.stub import StubProvider


def _providers():
    return {
        "openai_compat": OpenAICompatProvider(base_url="http://unused", api_key="k", model="m"),
        "anthropic": AnthropicProvider(api_key="k", model="m"),
        "llama_cpp": LlamaCppProvider(base_url="http://unused"),
        "ollama": OllamaProvider(base_url="http://unused"),
        "stub": StubProvider(base_url="http://unused"),
    }


PROVIDER_IDS = sorted(_providers())


def _request():
    return ApiMessageRequest(
        model="m",
        messages=[ConversationMessage(role="user", content=[TextBlock(text="2+2?")])],
    )


async def _drain(provider):
    seen: list[str] = []
    error: BaseException | None = None
    try:
        async for ev in provider.stream_message(_request()):
            if isinstance(ev, ApiTextDeltaEvent):
                seen.append(ev.text)
    except BaseException as exc:  # noqa: BLE001 — the raise is the assertion
        error = exc
    return seen, error


@pytest.mark.parametrize("name", PROVIDER_IDS)
@pytest.mark.asyncio
async def test_a_failure_AFTER_output_does_not_retry(name):
    """The bug: the consumer read both answers concatenated."""
    provider = _providers()[name]
    attempts = {"n": 0}

    async def call_once(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            yield ApiTextDeltaEvent(text="The answer is 4")
            raise httpx.TimeoutException("read timeout mid-stream")
        yield ApiTextDeltaEvent(text="The answer is 7")

    provider._call_once = call_once
    seen, error = await _drain(provider)

    assert attempts["n"] == 1, f"{name} retried after emitting output"
    assert seen == ["The answer is 4"], f"{name} replayed: {''.join(seen)!r}"
    assert isinstance(error, httpx.TimeoutException), f"{name} swallowed the failure"


@pytest.mark.parametrize("name", PROVIDER_IDS)
@pytest.mark.asyncio
async def test_a_failure_BEFORE_any_output_still_retries(name):
    """The guard against over-fixing. A retryable failure with nothing yet emitted is exactly
    what retries exist for, and must keep working."""
    provider = _providers()[name]
    attempts = {"n": 0}

    async def call_once(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("refused before any byte")
            yield  # pragma: no cover — makes this an async generator
        yield ApiTextDeltaEvent(text="recovered")

    provider._call_once = call_once
    seen, error = await _drain(provider)

    assert attempts["n"] == 2, f"{name} did NOT retry a pre-output failure (resilience lost)"
    assert seen == ["recovered"], f"{name} lost the successful retry: {seen}"
    assert error is None, f"{name} raised despite a successful retry: {error!r}"


@pytest.mark.parametrize("name", PROVIDER_IDS)
@pytest.mark.asyncio
async def test_a_NON_retryable_failure_before_output_still_raises_at_once(name):
    """401 is not retryable and never was — the fix must not have made it one."""
    provider = _providers()[name]
    attempts = {"n": 0}

    class Resp:
        status_code = 401

    class AuthError(Exception):
        response = Resp()

    async def call_once(request):
        attempts["n"] += 1
        raise AuthError("expired key")
        yield  # pragma: no cover

    provider._call_once = call_once
    seen, error = await _drain(provider)

    assert attempts["n"] == 1, f"{name} retried a 401"
    assert seen == []
    assert isinstance(error, AuthError)
