"""Telegram typing indicator spans the whole turn (feat/turn-progress-and-errors).

``send_chat_action`` was fired ONCE at dispatch, but Telegram clears "typing…"
after ~5 seconds — so any turn longer than that (a local-model turn is
routinely minutes) looked dead in the client even while tools were running.
The indicator now refreshes on a sub-5s cadence for the life of the turn.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("telegram")

import prometheus.gateway.telegram as tg  # noqa: E402
from prometheus.gateway.telegram import TelegramAdapter  # noqa: E402


class _Bot:
    def __init__(self) -> None:
        self.actions: list = []

    async def send_chat_action(self, chat_id, action):
        self.actions.append((chat_id, action))


class _App:
    def __init__(self) -> None:
        self.bot = _Bot()


def _gateway() -> TelegramAdapter:
    """A gateway instance without running __init__ — these tests exercise the
    typing helpers in isolation, not gateway construction."""
    gw = TelegramAdapter.__new__(TelegramAdapter)
    gw._app = _App()
    return gw


@pytest.fixture
def fast_typing(monkeypatch):
    """Shrink the refresh cadence so tests are fast and deterministic.

    Opt-in, NOT autouse: the cadence test below must observe the real shipped
    value, which a global override would mask.
    """
    monkeypatch.setattr(tg, "TYPING_REFRESH_SECONDS", 0.02)


def test_refresh_cadence_is_under_telegrams_expiry():
    # Telegram expires a chat action at ~5s; refreshing at or above that would
    # let the indicator visibly flicker off mid-turn. Deliberately does NOT
    # take fast_typing — this asserts the value we actually ship.
    assert tg.TYPING_REFRESH_SECONDS < 5.0


@pytest.mark.asyncio
async def test_typing_is_re_asserted_while_the_turn_runs(fast_typing):
    gw = _gateway()
    async with gw._typing(123):
        # Simulate a long turn: several refresh intervals of real work.
        await asyncio.sleep(0.25)
    actions = gw._app.bot.actions
    # Fired repeatedly, not once — the whole point of the change.
    assert len(actions) >= 3, f"expected repeated typing actions, got {len(actions)}"
    assert all(chat_id == 123 for chat_id, _ in actions)


@pytest.mark.asyncio
async def test_typing_stops_when_the_turn_ends(fast_typing):
    gw = _gateway()
    async with gw._typing(7):
        await asyncio.sleep(0.12)
    settled = len(gw._app.bot.actions)
    await asyncio.sleep(0.2)  # several more intervals
    assert len(gw._app.bot.actions) == settled, "typing kept firing after the turn"


@pytest.mark.asyncio
async def test_typing_failures_never_break_the_turn(fast_typing):
    gw = _gateway()

    async def boom(chat_id, action):
        raise RuntimeError("telegram is down")

    gw._app.bot.send_chat_action = boom
    # The turn body must still run and complete normally.
    ran = False
    async with gw._typing(1):
        await asyncio.sleep(0.05)
        ran = True
    assert ran


@pytest.mark.asyncio
async def test_no_app_is_a_clean_noop():
    gw = TelegramAdapter.__new__(TelegramAdapter)
    gw._app = None
    async with gw._typing(1):
        pass  # must not raise
