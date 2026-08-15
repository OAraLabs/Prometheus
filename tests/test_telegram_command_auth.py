"""Slash commands must obey allowed_chat_ids.

THE DEFECT (found 2026-08-14, pre-existing): `chat_allowed()` was enforced in
exactly one place — `on_message`, reached only via the text handler, which is
registered as ``filters.TEXT & ~filters.COMMAND``. That ``~COMMAND`` meant
slash commands never passed through the check at all. All 51 were callable by
any chat that found the bot, including `/gate off` (suppresses approval
prompts globally) and `/approve always` (persists a trust grant).

These tests assert BOTH directions, because an auth guard that only refuses is
half a control (Standing-Principles §2c): the allowed chat must still get
through, or the fix locks the operator out of their own bot.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.ext import ApplicationHandlerStop

from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.telegram import TelegramAdapter
from prometheus.tools.base import ToolRegistry

# The real operator chat on this deployment. Hard-coded on purpose: the
# lock-out direction is the one that must never regress silently.
OPERATOR_CHAT = 8139235390
STRANGER_CHAT = 999000111


def _adapter(allowed: list[int] | None) -> TelegramAdapter:
    cfg = PlatformConfig(
        platform=Platform.TELEGRAM,
        token="test-token",
        allowed_chat_ids=allowed if allowed is not None else [],
    )
    agent_loop = AsyncMock()
    agent_loop._model_router = None
    return TelegramAdapter(
        config=cfg,
        agent_loop=agent_loop,
        tool_registry=ToolRegistry(),
        model_name="test-model-v1",
        model_provider="llama_cpp",
    )


def _update(chat_id: int, text: str = "/gate off"):
    upd = MagicMock()
    upd.effective_chat = MagicMock(id=chat_id)
    upd.effective_user = MagicMock(id=4242)
    upd.message = MagicMock(text=text)
    return upd


class TestCommandAuthorization:

    @pytest.mark.asyncio
    async def test_allowed_chat_passes_through(self):
        """THE LOCK-OUT DIRECTION. Must return normally — raising here would
        block the operator from their own bot."""
        a = _adapter([OPERATOR_CHAT])
        await a._authorize_update(_update(OPERATOR_CHAT), MagicMock())

    @pytest.mark.asyncio
    async def test_unauthorized_chat_is_stopped(self):
        a = _adapter([OPERATOR_CHAT])
        with pytest.raises(ApplicationHandlerStop):
            await a._authorize_update(_update(STRANGER_CHAT), MagicMock())

    @pytest.mark.asyncio
    async def test_gate_off_from_a_stranger_is_stopped(self):
        """The command that motivated this: /gate off suppresses approval
        prompts globally."""
        a = _adapter([OPERATOR_CHAT])
        with pytest.raises(ApplicationHandlerStop):
            await a._authorize_update(
                _update(STRANGER_CHAT, "/gate off"), MagicMock()
            )

    @pytest.mark.asyncio
    async def test_approve_always_from_a_stranger_is_stopped(self):
        a = _adapter([OPERATOR_CHAT])
        with pytest.raises(ApplicationHandlerStop):
            await a._authorize_update(
                _update(STRANGER_CHAT, "/approve always abc123"), MagicMock()
            )

    @pytest.mark.asyncio
    async def test_empty_allowlist_now_permits_NOBODY(self):
        """SUPERSEDED, deliberately.

        #202 scoped itself to "fix command auth, do not tighten a deployment
        that never opted in", and this test pinned that decision. The decision
        has been reversed by ruling: an empty allowlist meant "allow every
        chat", and the adjacent gateway.telegram_enabled defaulted ON at the
        behaviour site, so the two together made a hand-trimmed config a
        public bot. Absence is not permission.

        The rename is the point — the old NAME asserted the old semantics as
        loudly as the body did."""
        a = _adapter([])
        with pytest.raises(ApplicationHandlerStop):
            await a._authorize_update(_update(STRANGER_CHAT), MagicMock())

    @pytest.mark.asyncio
    async def test_update_without_a_chat_passes(self):
        """Poll answers and similar carry no chat to authorize; every handler
        already returns early when effective_chat is None."""
        a = _adapter([OPERATOR_CHAT])
        upd = MagicMock()
        upd.effective_chat = None
        await a._authorize_update(upd, MagicMock())

    @pytest.mark.asyncio
    async def test_non_text_update_from_a_stranger_is_stopped(self):
        """Media updates have no .text — the guard must not crash building
        its log line and fail open."""
        a = _adapter([OPERATOR_CHAT])
        upd = MagicMock()
        upd.effective_chat = MagicMock(id=STRANGER_CHAT)
        upd.effective_user = MagicMock(id=1)
        upd.message = None
        with pytest.raises(ApplicationHandlerStop):
            await a._authorize_update(upd, MagicMock())


class TestGuardIsRegisteredFirst:
    """A guard that exists but is not registered — or is registered in the
    wrong group — is Standing-Principles §2e. Assert the wiring, not just the
    function."""

    def test_registered_in_a_lower_group_than_the_commands(self):
        import inspect

        src = inspect.getsource(TelegramAdapter)
        assert "TypeHandler(Update, self._authorize_update), group=-1" in src, (
            "the authorization TypeHandler must be registered in group -1 so "
            "it runs before every command handler"
        )

    def test_guard_appears_before_the_first_command_registration(self):
        import inspect

        src = inspect.getsource(TelegramAdapter)
        guard = src.index("self._authorize_update), group=-1")
        first_cmd = src.index('CommandHandler("start"')
        assert guard < first_cmd
