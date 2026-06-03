"""Tests for prometheus.jobs.daily_briefing (branch fix/cron-briefing-tz).

Uses stubbed web_search / message / provider so nothing hits the network or
~/.prometheus. Async functions are driven with asyncio.run() to avoid any
pytest-asyncio mode coupling.
"""

from __future__ import annotations

import asyncio

import pytest

import prometheus.jobs.daily_briefing as jb
from prometheus.jobs.daily_briefing import (
    BriefingError,
    resolve_chat_id,
    resolve_telegram_token,
    run_briefing,
)
from prometheus.providers.base import ApiTextDeltaEvent
from prometheus.tools.base import ToolResult
from prometheus.tools.builtin.message import MessagePlatform


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #


class _StubProvider:
    """Yields the given text as streamed deltas (no agent loop involved)."""

    def __init__(self, text: str = "Headlines:\n- A\n- B\nMarkets:\n- C") -> None:
        self._text = text

    async def stream_message(self, request):  # noqa: ANN001
        # Provider-agnostic: emit text deltas; the summarizer accumulates them.
        mid = len(self._text) // 2
        yield ApiTextDeltaEvent(text=self._text[:mid])
        yield ApiTextDeltaEvent(text=self._text[mid:])


class _EmptyProvider:
    """An async generator that yields nothing -> empty briefing."""

    async def stream_message(self, request):  # noqa: ANN001
        if False:  # pragma: no cover - makes this an async generator
            yield


class _StubWebSearch:
    def __init__(self, *, is_error: bool = False) -> None:
        self._is_error = is_error
        self.calls: list = []

    async def execute(self, arguments, context):  # noqa: ANN001
        self.calls.append(arguments)
        if self._is_error:
            return ToolResult(output="network down", is_error=True)
        return ToolResult(output=f"results for: {arguments.query}")


class _StubMessage:
    def __init__(self) -> None:
        self.calls: list = []

    async def execute(self, arguments, context):  # noqa: ANN001
        self.calls.append(arguments)
        return ToolResult(output="sent")


# --------------------------------------------------------------------------- #
# run_briefing pipeline
# --------------------------------------------------------------------------- #


def test_run_briefing_success_sends_once_to_chat() -> None:
    ws = _StubWebSearch()
    msg = _StubMessage()
    out = asyncio.run(
        run_briefing(
            provider=_StubProvider(),
            model="gemma4-26b",
            chat_id="55512345",
            web_search=ws,
            message=msg,
        )
    )
    assert out.strip()                      # non-empty briefing
    assert len(ws.calls) == len(jb.SEARCH_QUERIES)
    assert len(msg.calls) == 1              # exactly one send
    sent = msg.calls[0]
    assert sent.platform == MessagePlatform.telegram
    assert sent.recipient == "55512345"     # right chat target
    assert sent.content == out


def test_run_briefing_raises_when_web_search_fails() -> None:
    ws = _StubWebSearch(is_error=True)
    msg = _StubMessage()
    with pytest.raises(BriefingError, match="web_search failed"):
        asyncio.run(
            run_briefing(
                provider=_StubProvider(),
                model="m",
                chat_id="1",
                web_search=ws,
                message=msg,
            )
        )
    assert msg.calls == []                   # never reached the send stage


def test_run_briefing_raises_on_empty_completion() -> None:
    ws = _StubWebSearch()
    msg = _StubMessage()
    with pytest.raises(BriefingError, match="empty briefing"):
        asyncio.run(
            run_briefing(
                provider=_EmptyProvider(),
                model="m",
                chat_id="1",
                web_search=ws,
                message=msg,
            )
        )
    assert msg.calls == []


def test_run_briefing_raises_when_send_fails() -> None:
    class _FailingMessage(_StubMessage):
        async def execute(self, arguments, context):  # noqa: ANN001
            self.calls.append(arguments)
            return ToolResult(output="telegram 400", is_error=True)

    msg = _FailingMessage()
    with pytest.raises(BriefingError, match="Telegram send failed"):
        asyncio.run(
            run_briefing(
                provider=_StubProvider(),
                model="m",
                chat_id="1",
                web_search=_StubWebSearch(),
                message=msg,
            )
        )


# --------------------------------------------------------------------------- #
# Credential resolution (fail-loud)
# --------------------------------------------------------------------------- #


def test_resolve_chat_id_precedence(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    # briefing_chat_id wins
    assert resolve_chat_id(
        {"gateway": {"briefing_chat_id": 999, "allowed_chat_ids": [111]}}
    ) == "999"
    # falls back to allowed_chat_ids[0] (the daemon's source)
    assert resolve_chat_id({"gateway": {"allowed_chat_ids": [111, 222]}}) == "111"
    # then TELEGRAM_CHAT_ID env
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "777")
    assert resolve_chat_id({"gateway": {}}) == "777"


def test_resolve_chat_id_fails_loud_when_unconfigured(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with pytest.raises(BriefingError, match="No Telegram chat target"):
        resolve_chat_id({"gateway": {"allowed_chat_ids": []}})


def test_resolve_token(monkeypatch) -> None:
    monkeypatch.delenv("PROMETHEUS_TELEGRAM_TOKEN", raising=False)
    assert resolve_telegram_token({"gateway": {"telegram_token": "abc"}}) == "abc"
    monkeypatch.setenv("PROMETHEUS_TELEGRAM_TOKEN", "envtok")
    assert resolve_telegram_token({"gateway": {}}) == "envtok"
    monkeypatch.delenv("PROMETHEUS_TELEGRAM_TOKEN", raising=False)
    with pytest.raises(BriefingError, match="No Telegram bot token"):
        resolve_telegram_token({"gateway": {}})


# --------------------------------------------------------------------------- #
# main() exit-code translation
# --------------------------------------------------------------------------- #


def test_main_returns_zero_on_success(monkeypatch) -> None:
    async def _ok() -> None:
        return None

    monkeypatch.setattr(jb, "_main_async", _ok)
    assert jb.main() == 0


def test_main_returns_nonzero_on_failure(monkeypatch) -> None:
    async def _boom() -> None:
        raise BriefingError("web_search failed: network down")

    monkeypatch.setattr(jb, "_main_async", _boom)
    assert jb.main() == 1
