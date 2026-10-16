"""web_search says what went wrong, and tries once more before giving up.

On 2026-10-15 a session on the live daemon (0.9.10) fired three searches and
got "web_search failed: " three times, with nothing after the colon. httpx
raised errors whose message is empty (a connection the server closed reads as
``ReadError('')``) and the tool printed only the message. The failures took
60-125 ms against 600-900 ms for a good search, the same thing happened 8 times
out of 8 on 10-11, and on a scratch daemon the next try a second later worked.

Pinned: the type is always named, the message too when there is one; one retry
after a pause; a retry that works returns the results; two failures report the
last one and how many tries were made.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin import web_search as ws
from prometheus.tools.builtin.web_search import WebSearchInput, WebSearchTool

RESULTS = (
    '<a class="result__a" href="https://example.org/a">Alpha</a>'
    '<a class="result__snippet" href="https://example.org/a">first</a>'
)


def _serve(monkeypatch, steps: list) -> list[httpx.Request]:  # noqa: ANN001
    """Answer the n-th request with ``steps[n]``: an exception to raise, or a status."""
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        step = steps[min(len(seen), len(steps)) - 1]
        if isinstance(step, Exception):
            raise step
        return httpx.Response(step, text=RESULTS if step == 200 else "nope", request=req)

    real = httpx.AsyncClient

    def client(*args, **kwargs):  # noqa: ANN002, ANN003
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setattr(ws, "_RETRY_DELAY_SECONDS", 0.0)
    return seen


def _search(tmp_path):  # noqa: ANN001, ANN202
    return asyncio.run(WebSearchTool().execute(
        WebSearchInput(query="whatsapp cloud api"), ToolExecutionContext(cwd=tmp_path)))


def test_a_dropped_connection_is_retried_once_and_the_retry_s_results_return(tmp_path, monkeypatch):
    seen = _serve(monkeypatch, [httpx.ReadError(""), 200])
    result = _search(tmp_path)
    assert not result.is_error, result.output
    assert "Alpha" in result.output and "https://example.org/a" in result.output
    assert len(seen) == 2


def test_an_empty_message_still_names_the_error(tmp_path, monkeypatch):
    seen = _serve(monkeypatch, [httpx.ReadError(""), httpx.ReadError("")])
    result = _search(tmp_path)
    assert result.is_error
    assert result.output == "web_search failed: ReadError (tried 2 times)"
    assert len(seen) == 2, "one retry, no more"


def test_a_message_is_kept_after_the_type(tmp_path, monkeypatch):
    _serve(monkeypatch, [httpx.ConnectError("boom"), httpx.ConnectError("boom")])
    assert _search(tmp_path).output == "web_search failed: ConnectError: boom (tried 2 times)"


def test_an_http_error_status_is_retried_and_named(tmp_path, monkeypatch):
    seen = _serve(monkeypatch, [503, 403])
    result = _search(tmp_path)
    assert result.is_error
    assert result.output.startswith("web_search failed: HTTPStatusError: Client error '403")
    assert len(seen) == 2


def test_a_server_error_then_a_good_answer_returns_results(tmp_path, monkeypatch):
    _serve(monkeypatch, [503, 200])
    assert not _search(tmp_path).is_error


def test_a_first_try_that_works_is_not_repeated(tmp_path, monkeypatch):
    seen = _serve(monkeypatch, [200])
    assert not _search(tmp_path).is_error
    assert len(seen) == 1


@pytest.mark.parametrize("exc,text", [
    (httpx.ReadError(""), "ReadError"),
    (httpx.ReadTimeout("   "), "ReadTimeout"),
    (httpx.ConnectError("[Errno 111] refused"), "ConnectError: [Errno 111] refused"),
    (None, "no response"),
])
def test_describe(exc, text):  # noqa: ANN001
    assert ws._describe(exc) == text
