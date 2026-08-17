"""Gateway liveness — the 2026-08-15 Telegram outage.

Two defects, both invisible in the logs at the time:

1. ``drop_pending_updates=True`` meant every restart destroyed whatever the
   user sent while the daemon was down. Telegram holds updates for 24h; the
   daemon was throwing that grace away on a routine deploy.

2. ``running`` is self-reported. It stayed True for 45 minutes while Telegram
   answered every call with 502, because nothing ever asked Telegram. PTB
   could not raise the alarm either: httpx's 5s read_timeout fired before the
   ~15.7s 502 landed, so every failure surfaced as ``TimedOut`` — the one
   branch of network_retry_loop that skips ``on_err_cb`` and logs at DEBUG.

The fix is a real probe, and these tests pin both halves.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.heartbeat import Heartbeat
from prometheus.gateway.telegram import TelegramAdapter
from prometheus.tools.base import ToolRegistry


def _adapter(prometheus_config: dict | None = None) -> TelegramAdapter:
    return TelegramAdapter(
        config=PlatformConfig(platform=Platform.TELEGRAM, token="test-token"),
        agent_loop=MagicMock(),
        tool_registry=ToolRegistry(),
        prometheus_config=prometheus_config if prometheus_config is not None else {},
    )


def _mock_app() -> MagicMock:
    """A telegram Application stub with the awaited methods made awaitable."""
    app = MagicMock()
    app.initialize = AsyncMock()
    app.start = AsyncMock()
    app.stop = AsyncMock()
    app.shutdown = AsyncMock()
    app.updater.start_polling = AsyncMock()
    app.updater.stop = AsyncMock()
    app.bot.set_my_commands = AsyncMock()
    app.bot.get_me = AsyncMock()
    app.bot.send_message = AsyncMock()
    return app


# ---------------------------------------------------------------------------
# 1. The message-loss regression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_polling_keeps_the_pending_backlog(monkeypatch):
    """drop_pending_updates MUST be False — this is the data-loss pin.

    If this flips back to True, every daemon restart silently destroys the
    messages sent while it was down, with no log line to show for it.
    """
    app = _mock_app()
    builder = MagicMock()
    # ⚠ PTB's builder is FLUENT — every configuration method returns self,
    # verified against the real class, not assumed. The double must model
    # that for EVERY setter, not just token().
    #
    # It modelled only token() until #224 wired PlatformConfig timeouts in as
    # `builder = self._configure_network(builder)`. That reassignment handed
    # `builder` an auto-created child mock, so `build()` returned a different
    # object than the one configured below and `await app.initialize()` blew
    # up on a bare MagicMock — a test failure caused entirely by the double,
    # with both production changes correct. Two PRs that merge-tree clean and
    # still collide, through a mock.
    for _setter in ("token", "connect_timeout", "read_timeout",
                    "write_timeout", "proxy"):
        getattr(builder, _setter).return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr(
        "prometheus.gateway.telegram.Application.builder", lambda: builder
    )

    # probe off, so this test asserts on polling alone and leaks no task
    adapter = _adapter({"gateway": {"telegram": {"probe_interval_seconds": 0}}})
    await adapter.start()

    app.updater.start_polling.assert_awaited_once()
    kwargs = app.updater.start_polling.await_args.kwargs
    assert kwargs["drop_pending_updates"] is False, (
        "dropping pending updates destroys messages sent while the daemon "
        "was down — Telegram already holds them for 24h"
    )
    assert adapter._probe_task is None


# ---------------------------------------------------------------------------
# 2. Reachability is probed, not assumed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reachable_is_none_before_any_probe():
    adapter = _adapter()
    assert adapter.reachable is None, "unknown must not masquerade as healthy"


@pytest.mark.asyncio
async def test_probe_success_marks_reachable():
    adapter = _adapter()
    adapter._app = _mock_app()

    assert await adapter._probe_reachability() is True
    assert adapter.reachable is True
    assert adapter._last_reachable_at is not None
    assert adapter._last_probe_error is None


@pytest.mark.asyncio
async def test_probe_failure_marks_unreachable_while_running_stays_true(caplog):
    """The exact 2026-08-15 shape: started, self-reporting fine, actually dark."""
    adapter = _adapter()
    adapter._app = _mock_app()
    adapter._app.bot.get_me = AsyncMock(side_effect=TimeoutError("Timed out"))
    adapter._running = True  # start() said so and nothing can clear it

    with caplog.at_level(logging.ERROR):
        assert await adapter._probe_reachability() is False

    assert adapter.running is True, "running is self-reported — unchanged"
    assert adapter.reachable is False, "...but reachability tells the truth"
    assert "UNREACHABLE" in caplog.text
    assert "TimeoutError" in adapter._last_probe_error


@pytest.mark.asyncio
async def test_outage_logs_once_not_every_probe(caplog):
    """A 45-minute outage must not mean 45 minutes of identical ERROR lines."""
    adapter = _adapter()
    adapter._app = _mock_app()
    adapter._app.bot.get_me = AsyncMock(side_effect=TimeoutError("Timed out"))

    with caplog.at_level(logging.ERROR):
        for _ in range(5):
            await adapter._probe_reachability()

    assert caplog.text.count("UNREACHABLE") == 1


@pytest.mark.asyncio
async def test_recovery_is_announced(caplog):
    adapter = _adapter()
    adapter._app = _mock_app()
    adapter._app.bot.get_me = AsyncMock(side_effect=TimeoutError("Timed out"))
    await adapter._probe_reachability()
    assert adapter.reachable is False

    adapter._app.bot.get_me = AsyncMock()
    with caplog.at_level(logging.INFO):
        await adapter._probe_reachability()

    assert adapter.reachable is True
    assert adapter._last_probe_error is None
    assert "reachable again" in caplog.text


@pytest.mark.asyncio
async def test_probe_loop_survives_a_crashing_probe(caplog):
    """A probe that raises must not kill the loop — that would re-blind us."""
    adapter = _adapter()
    calls = []

    async def boom() -> bool:
        calls.append(1)
        raise RuntimeError("probe exploded")

    adapter._probe_reachability = boom
    task = asyncio.create_task(adapter._reachability_loop(0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(calls) > 1, "loop stopped after the first crash"


# ---------------------------------------------------------------------------
# 3. Config + surfaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg,expected",
    [
        ({}, 60.0),
        ({"gateway": {"telegram": {"probe_interval_seconds": 15}}}, 15.0),
        ({"gateway": {"telegram": {"probe_interval_seconds": 0}}}, 0.0),
        ({"gateway": {"telegram": {"probe_interval_seconds": "nonsense"}}}, 60.0),
        ({"gateway": {"telegram": None}}, 60.0),
    ],
)
def test_probe_interval_resolution(cfg, expected):
    assert _adapter(cfg)._probe_interval_seconds() == expected


def test_health_snapshot_shape():
    adapter = _adapter()
    snap = adapter.health_snapshot()
    assert snap["platform"] == "telegram"
    assert snap["running"] is False
    assert snap["reachable"] is None
    assert snap["last_error"] is None


@pytest.mark.asyncio
async def test_stop_cancels_the_probe_task():
    adapter = _adapter()
    adapter._app = _mock_app()
    adapter._running = True
    adapter._probe_task = asyncio.create_task(adapter._reachability_loop(3600))
    await asyncio.sleep(0)

    await adapter.stop()

    assert adapter._probe_task is None


# ---------------------------------------------------------------------------
# 4. Heartbeat surfaces it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_reports_gateway_reachable():
    gateway = MagicMock()
    gateway.running = True
    gateway.reachable = False
    gateway.platform.value = "telegram"

    status = await Heartbeat(gateway=gateway).check()

    assert status["gateway_running"] is True
    assert status["gateway_reachable"] is False


@pytest.mark.asyncio
async def test_heartbeat_reports_none_for_adapters_without_a_probe():
    """A Slack/Discord adapter with no probe must report unknown, not True."""
    gateway = MagicMock(spec=["running", "platform"])
    gateway.running = True
    gateway.platform.value = "slack"

    status = await Heartbeat(gateway=gateway).check()

    assert status["gateway_reachable"] is None


@pytest.mark.asyncio
async def test_heartbeat_without_a_gateway_reports_none():
    status = await Heartbeat(gateway=None).check()
    assert status["gateway_running"] is None
    assert status["gateway_reachable"] is None


# ---------------------------------------------------------------------------
# 5. /api/status
# ---------------------------------------------------------------------------


def _client(gateway_adapter=None):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.web.server import create_app

    app = create_app({}, boot_sha="deadbeef")
    app.state.gateway_adapter = gateway_adapter
    return TestClient(app)


def test_status_reports_unwired_gateway():
    body = _client().get("/api/status").json()
    assert body["gateway"] == {"wired": False}


def test_status_reports_a_dark_gateway():
    adapter = _adapter()
    adapter._running = True
    adapter._reachable = False
    adapter._last_probe_error = "TimedOut: Timed out"

    body = _client(adapter).get("/api/status").json()

    assert body["gateway"]["wired"] is True
    assert body["gateway"]["running"] is True
    assert body["gateway"]["reachable"] is False
    assert body["gateway"]["last_error"] == "TimedOut: Timed out"


def test_status_falls_back_for_adapters_without_a_snapshot():
    gateway = MagicMock(spec=["running", "reachable"])
    gateway.running = True
    gateway.reachable = None

    body = _client(gateway).get("/api/status").json()

    assert body["gateway"] == {"wired": True, "running": True, "reachable": None}
