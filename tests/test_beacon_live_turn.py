"""Tests for Beacon live-turn signals: Telegram emit + WS mapping."""

from __future__ import annotations

import json
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# ActivitySignal
# ---------------------------------------------------------------------------

from prometheus.sentinel.signals import ActivitySignal


def test_activity_signal_fields():
    sig = ActivitySignal(
        kind="turn_completed",
        source="telegram",
        payload={"session_id": "sess-1"},
    )
    assert sig.kind == "turn_completed"
    assert sig.source == "telegram"
    assert sig.payload == {"session_id": "sess-1"}
    assert sig.timestamp is not None


# ---------------------------------------------------------------------------
# TelegramAdapter._emit_turn_completed
# ---------------------------------------------------------------------------

class TestTelegramTurnCompleted:
    """_emit_turn_completed emits an ActivitySignal with the right fields."""

    @pytest.mark.asyncio
    async def test_emits_turn_completed(self):
        from prometheus.gateway.telegram import TelegramAdapter

        # Bypass __init__ entirely.
        adapter = TelegramAdapter.__new__(TelegramAdapter)

        bus = MagicMock()
        bus.emit = AsyncMock()
        adapter._signal_bus = bus

        await adapter._emit_turn_completed(
            session_id="sess-1",
            provenance="user",
            model_name="qwen3.8-27b",
            model_provider="llama-cpp",
            message_count=5,
        )

        bus.emit.assert_awaited_once()
        sig = bus.emit.await_args[0][0]
        assert isinstance(sig, ActivitySignal)
        assert sig.kind == "turn_completed"
        assert sig.source == "telegram"
        assert sig.payload["session_id"] == "sess-1"
        assert sig.payload["provenance"] == "user"
        assert sig.payload["model"] == "qwen3.8-27b"
        assert sig.payload["model_provider"] == "llama-cpp"
        assert sig.payload["message_count"] == 5

    @pytest.mark.asyncio
    async def test_no_bus_is_noop(self):
        from prometheus.gateway.telegram import TelegramAdapter

        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter._signal_bus = None

        # Must not raise.
        await adapter._emit_turn_completed(
            session_id="sess-1",
            provenance="user",
            model_name="m",
            model_provider="p",
            message_count=1,
        )

    @pytest.mark.asyncio
    async def test_bus_exception_is_swallowed(self):
        from prometheus.gateway.telegram import TelegramAdapter

        adapter = TelegramAdapter.__new__(TelegramAdapter)
        bus = MagicMock()
        bus.emit = AsyncMock(side_effect=RuntimeError("bus down"))
        adapter._signal_bus = bus

        # Must not raise.
        await adapter._emit_turn_completed(
            session_id="sess-1",
            provenance="user",
            model_name="m",
            model_provider="p",
            message_count=1,
        )


# ---------------------------------------------------------------------------
# WebSocketBridge signal→event mapping
# ---------------------------------------------------------------------------

class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[Any] = []

    async def send(self, text: str) -> None:
        self.sent.append(json.loads(text))

    def close(self, *args: Any, **kwargs: Any) -> None:
        pass


class TestWSBridgeTurnCompleted:
    """turn_completed signals map to a dedicated event type."""

    @staticmethod
    def _bridge() -> Any:
        from prometheus.web.ws_server import WebSocketBridge

        # Bypass __init__; only set the attributes the handler touches.
        bridge = WebSocketBridge.__new__(WebSocketBridge)
        bridge._clients = set()
        bridge._subscribers = {}
        bridge._subscriptions = {}
        bridge._ws_filters = {}  # broadcast() consults the Piece-4 filter
        bridge._last_ids = {}
        bridge._log = MagicMock()
        return bridge

    @pytest.mark.asyncio
    async def test_turn_completed_maps_to_dedicated_event(self):
        sig = ActivitySignal(
            kind="turn_completed",
            source="telegram",
            payload={
                "session_id": "sess-1",
                "provenance": "user",
                "model": "qwen3.8-27b",
                "model_provider": "llama-cpp",
                "message_count": 5,
            },
        )

        bridge = self._bridge()
        ws = _FakeWS()
        bridge._clients.add(ws)
        await bridge._on_signal(sig)

        events = ws.sent
        assert len(events) == 1
        assert events[0]["type"] == "turn_completed"
        assert events[0]["payload"]["session_id"] == "sess-1"
        assert events[0]["payload"]["model"] == "qwen3.8-27b"
        assert events[0]["payload"]["model_provider"] == "llama-cpp"
        assert events[0]["payload"]["message_count"] == 5

    @pytest.mark.asyncio
    async def test_other_kinds_still_activity(self):
        """Kinds that aren't turn_completed still map to the generic
        activity event, unchanged."""
        sig = ActivitySignal(
            kind="tool_call",
            source="daemon",
            payload={"tool": "bash"},
        )

        bridge = self._bridge()
        ws = _FakeWS()
        bridge._clients.add(ws)
        await bridge._on_signal(sig)

        events = ws.sent
        assert len(events) == 1
        assert events[0]["type"] == "sentinel_signal"
        assert events[0]["payload"]["kind"] == "tool_call"
