"""Regression: Telegram network timeouts from PlatformConfig must reach the
PTB request object.

PTB's built-in defaults are 5s connect/read/write. Photo downloads that took
>5s timed out with 'Failed to download photo: Timed out' because
PlatformConfig's 30s values were never wired into the Application builder.
"""

from __future__ import annotations

import pytest

pytest.importorskip("telegram")

from telegram.ext import Application  # noqa: E402

from prometheus.gateway.config import PlatformConfig  # noqa: E402
from prometheus.gateway.telegram import TelegramAdapter  # noqa: E402


def _adapter(read: float = 30.0, connect: float = 30.0, write: float = 30.0):
    gw = TelegramAdapter.__new__(TelegramAdapter)
    gw.config = PlatformConfig(
        platform="telegram",  # type: ignore[arg-type]
        token="123:dummy",
        read_timeout=read,
        connect_timeout=connect,
        write_timeout=write,
    )
    return gw


def test_timeouts_applied_to_bot_request():
    gw = _adapter()
    builder = Application.builder().token(gw.config.token)
    app = gw._configure_network(builder).build()

    req = app.bot.request
    assert req.read_timeout == 30.0
    # The combined httpx timeout carries connect/write too.
    httpx_timeout = req._client._timeout
    assert httpx_timeout.connect == 30.0
    assert httpx_timeout.write == 30.0


def test_custom_timeout_values_flow_through():
    gw = _adapter(read=12.5, connect=3.5, write=7.5)
    builder = Application.builder().token(gw.config.token)
    app = gw._configure_network(builder).build()

    req = app.bot.request
    assert req.read_timeout == 12.5
    httpx_timeout = req._client._timeout
    assert httpx_timeout.connect == 3.5
    assert httpx_timeout.write == 7.5
