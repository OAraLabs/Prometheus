"""One streaming-retry loop, shared by every provider.

It existed five times, byte-for-byte, in openai_compat / anthropic / llama_cpp / ollama / stub.
That is why the mid-stream replay bug (#293) was five bugs and its fix was five identical edits
with five identical comments. The sixth copy is the one worth preventing.

WHAT IS SHARED IS THE LOOP. WHAT IS NOT SHARED IS THE POLICY.

`retryable_status` is passed in per provider rather than folded into one constant, because the two
sets genuinely disagree: anthropic includes 529 (its own "overloaded" code) and the others do not.
Merging them would silently change three providers' retry behaviour, and a refactor is the wrong
place to make a policy decision — the whole point of extracting this is that behaviour is
IDENTICAL afterwards. If 529 should be universal (or Anthropic-only), that is its own change with
its own reasoning; see #295.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import AsyncIterator, Awaitable, Callable, Iterable

import httpx

log = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_DELAY = 1.0
MAX_DELAY = 30.0

# Transport failures carry no HTTP status; these are the ones worth another attempt.
_RETRYABLE_EXCEPTIONS = (httpx.ConnectError, httpx.TimeoutException, ConnectionError)


def _status_of(exc: BaseException) -> int | None:
    """The HTTP status an exception carries, or None for a transport-level failure."""
    status = getattr(exc, "status_code", None)
    if status is None and hasattr(exc, "response"):
        status = getattr(exc.response, "status_code", None)
    return status if isinstance(status, int) else None


async def stream_with_retry(
    call_once: Callable[[], AsyncIterator],
    *,
    retryable_status: Iterable[int],
    label: str,
    max_retries: int = MAX_RETRIES,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> AsyncIterator:
    """Stream `call_once()`, retrying with exponential backoff and jitter.

    `call_once` is a zero-argument factory returning a FRESH async iterator per attempt — not an
    iterator, which could only be consumed once.

    NEVER retries once output has been yielded (#293). A retry re-runs the request from scratch,
    and whatever already reached the consumer is already on their screen, so replaying over it
    appends a second, possibly contradictory answer to the first. Retries stay fully available for
    the case they exist for: a failure before the first event.

    `sleep` is injectable so tests can assert backoff without spending it.
    """
    statuses = frozenset(retryable_status)
    last_error: BaseException | None = None

    for attempt in range(max_retries + 1):
        # Whether THIS attempt has handed anything to the consumer yet.
        emitted = False
        try:
            async for event in call_once():
                emitted = True
                yield event
            return
        except GeneratorExit:
            # The consumer stopped iterating — a protocol signal, not a failure to retry.
            raise
        except Exception as exc:  # noqa: BLE001 — classified below, then retried or re-raised
            last_error = exc
            status = _status_of(exc)
            retryable = status in statuses if status is not None else isinstance(exc, _RETRYABLE_EXCEPTIONS)
            if emitted or attempt >= max_retries or not retryable:
                raise
            delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
            delay += random.uniform(0, delay * 0.25)
            log.warning(
                "%s request failed (attempt %d/%d), retrying in %.1fs: %s",
                label, attempt + 1, max_retries + 1, delay, exc,
            )
            await sleep(delay)

    # Unreachable in practice: the loop either returns, raises, or exhausts attempts via the
    # `attempt >= max_retries` branch above. Kept so a future edit cannot fall out silently.
    if last_error is not None:
        raise last_error
