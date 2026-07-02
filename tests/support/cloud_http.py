"""IGNITION Piece 2 — the cloud HTTP boundary recorder.

A registered double for ``httpx.AsyncClient`` that captures the EXACT
serialized request (url + json body) the real cloud providers send, and
returns a benign empty SSE stream (200, no events) so ``stream_message``
completes. Acceptance tests assert on ``captured[...]["json"]`` — the
serialized request, not a spy on our own code.

Registered as ``cloud_http.RecordingAsyncClient`` — acceptance tests must name
it in ``allow_doubles``.
"""

from __future__ import annotations

from typing import Any

from tests.support.doubles import register_double


class _EmptyStreamResponse:
    status_code = 200
    text = ""

    def raise_for_status(self) -> None:
        return None

    async def aread(self) -> bytes:
        return b""

    async def aiter_lines(self):
        if False:
            yield ""


class _StreamCM:
    async def __aenter__(self) -> _EmptyStreamResponse:
        return _EmptyStreamResponse()

    async def __aexit__(self, *exc: Any) -> bool:
        return False


@register_double(
    "cloud_http.RecordingAsyncClient",
    replaces="httpx.AsyncClient (cloud provider HTTP boundary)",
)
class RecordingAsyncClient:
    """Captures every (method, url, json) triple; class-level so tests read it
    after the provider's ``async with httpx.AsyncClient(...)`` block closes."""

    captured: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> "RecordingAsyncClient":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def stream(self, method: str, url: str, *, json: Any = None, headers: Any = None) -> _StreamCM:
        type(self).captured.append({"method": method, "url": url, "json": json})
        return _StreamCM()

    @classmethod
    def reset(cls) -> None:
        cls.captured = []
