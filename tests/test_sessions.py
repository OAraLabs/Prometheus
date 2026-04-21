"""Tests for gateway-agnostic conversation session management (GRAFT-THREAD)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock, ToolResultBlock
from prometheus.engine.session import ChatSession, SessionManager, MAX_SESSION_MESSAGES
from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.platform_base import MessageEvent, SendResult
from prometheus.tools.base import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_msg(text: str) -> ConversationMessage:
    return ConversationMessage.from_user_text(text)


def _assistant_msg(text: str) -> ConversationMessage:
    return ConversationMessage(role="assistant", content=[TextBlock(text=text)])


def _make_run_result(
    user_text: str, assistant_text: str, *, with_tool_call: bool = False
) -> MagicMock:
    """Build a mock RunResult with realistic messages list."""
    messages = [_user_msg(user_text)]
    if with_tool_call:
        messages.append(
            ConversationMessage(
                role="assistant",
                content=[ToolUseBlock(name="test_tool", input={"q": "x"})],
            )
        )
        messages.append(
            ConversationMessage(
                role="user",
                content=[ToolResultBlock(tool_use_id="toolu_abc", content="result")],
            )
        )
    messages.append(_assistant_msg(assistant_text))

    result = MagicMock()
    result.text = assistant_text
    result.messages = messages
    return result


# ===================================================================
# ChatSession unit tests
# ===================================================================


class TestChatSession:
    def test_add_user_message(self):
        s = ChatSession("test:1")
        s.add_user_message("hello")
        assert len(s.get_messages()) == 1
        assert s.get_messages()[0].role == "user"
        assert s.get_messages()[0].text == "hello"

    def test_add_result_messages(self):
        s = ChatSession("test:1")
        s.add_user_message("hi")
        pre_len = len(s.get_messages())  # 1

        result_msgs = [_user_msg("hi"), _assistant_msg("hello back")]
        s.add_result_messages(result_msgs, pre_len)

        assert len(s.get_messages()) == 2
        assert s.get_messages()[1].role == "assistant"
        assert s.get_messages()[1].text == "hello back"

    def test_add_result_messages_with_tool_calls(self):
        s = ChatSession("test:1")
        s.add_user_message("do something")
        pre_len = len(s.get_messages())

        result = _make_run_result("do something", "done", with_tool_call=True)
        s.add_result_messages(result.messages, pre_len)

        # user + assistant(tool_use) + user(tool_result) + assistant(final)
        assert len(s.get_messages()) == 4
        assert s.get_messages()[-1].text == "done"

    def test_rollback_last(self):
        s = ChatSession("test:1")
        s.add_user_message("hello")
        s.add_user_message("world")
        s.rollback_last()
        assert len(s.get_messages()) == 1
        assert s.get_messages()[0].text == "hello"

    def test_rollback_empty_is_safe(self):
        s = ChatSession("test:1")
        s.rollback_last()  # should not raise
        assert len(s.get_messages()) == 0

    def test_clear(self):
        s = ChatSession("test:1")
        s.add_user_message("a")
        s.add_user_message("b")
        s.clear()
        assert len(s.get_messages()) == 0

    def test_trim_enforces_limit(self):
        s = ChatSession("test:1")
        for i in range(60):
            s.add_user_message(f"msg {i}")
        s.trim(50)
        assert len(s.get_messages()) == 50
        # Should keep the last 50 messages
        assert s.get_messages()[0].text == "msg 10"
        assert s.get_messages()[-1].text == "msg 59"

    def test_trim_noop_when_under_limit(self):
        s = ChatSession("test:1")
        s.add_user_message("only one")
        s.trim(50)
        assert len(s.get_messages()) == 1

    def test_get_messages_returns_list(self):
        s = ChatSession("test:1")
        msgs = s.get_messages()
        assert isinstance(msgs, list)
        assert len(msgs) == 0


# ===================================================================
# SessionManager unit tests
# ===================================================================


class TestSessionManager:
    def test_get_or_create_returns_same_session(self):
        sm = SessionManager()
        s1 = sm.get_or_create("telegram:123")
        s2 = sm.get_or_create("telegram:123")
        assert s1 is s2

    def test_different_ids_different_sessions(self):
        sm = SessionManager()
        s1 = sm.get_or_create("telegram:123")
        s2 = sm.get_or_create("telegram:456")
        assert s1 is not s2

    def test_clear_resets_history(self):
        sm = SessionManager()
        s = sm.get_or_create("test:1")
        s.add_user_message("hello")
        sm.clear("test:1")
        assert len(s.get_messages()) == 0

    def test_clear_nonexistent_is_safe(self):
        sm = SessionManager()
        sm.clear("doesnt:exist")  # should not raise

    def test_remove_deletes_session(self):
        sm = SessionManager()
        s1 = sm.get_or_create("test:1")
        s1.add_user_message("hello")
        sm.remove("test:1")
        s2 = sm.get_or_create("test:1")
        assert s2 is not s1
        assert len(s2.get_messages()) == 0

    def test_remove_nonexistent_is_safe(self):
        sm = SessionManager()
        sm.remove("doesnt:exist")  # should not raise

    def test_cross_platform_isolation(self):
        sm = SessionManager()
        tg = sm.get_or_create("telegram:123")
        sl = sm.get_or_create("slack:123")
        tg.add_user_message("from telegram")
        assert len(sl.get_messages()) == 0
        assert tg is not sl


# ===================================================================
# Telegram adapter integration tests
# ===================================================================


def _make_telegram_adapter(agent_loop=None, session_manager=None):
    from prometheus.gateway.telegram import TelegramAdapter

    config = PlatformConfig(platform=Platform.TELEGRAM, token="test")
    if agent_loop is None:
        agent_loop = AsyncMock()
    adapter = TelegramAdapter(
        config=config,
        agent_loop=agent_loop,
        tool_registry=ToolRegistry(),
        session_manager=session_manager,
    )
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id=1))
    return adapter


def _make_event(chat_id: int = 123, text: str = "hello") -> MessageEvent:
    return MessageEvent(
        chat_id=chat_id,
        user_id=456,
        text=text,
        message_id=1,
        platform=Platform.TELEGRAM,
    )


class TestTelegramSessionIntegration:
    @pytest.mark.asyncio
    async def test_dispatch_accumulates_history(self):
        """Two messages to same chat — second call gets full history."""
        sm = SessionManager()
        result1 = _make_run_result("hello", "hi there")
        result2 = _make_run_result("how are you", "I'm good")
        agent_loop = AsyncMock()
        agent_loop.run_async = AsyncMock(side_effect=[result1, result2])

        adapter = _make_telegram_adapter(agent_loop=agent_loop, session_manager=sm)

        await adapter.on_message(_make_event(text="hello"))
        await adapter.on_message(_make_event(text="how are you"))

        # Second call should have received messages from both turns
        second_call = agent_loop.run_async.call_args_list[1]
        messages = second_call.kwargs["messages"]
        texts = [m.text for m in messages if m.text]
        assert "hello" in texts
        assert "hi there" in texts
        assert "how are you" in texts

    @pytest.mark.asyncio
    async def test_different_chats_isolated(self):
        sm = SessionManager()
        result = _make_run_result("msg", "reply")
        agent_loop = AsyncMock()
        agent_loop.run_async = AsyncMock(return_value=result)

        adapter = _make_telegram_adapter(agent_loop=agent_loop, session_manager=sm)

        await adapter.on_message(_make_event(chat_id=100, text="from chat 100"))
        await adapter.on_message(_make_event(chat_id=200, text="from chat 200"))

        s100 = sm.get_or_create("telegram:100")
        s200 = sm.get_or_create("telegram:200")
        texts_100 = [m.text for m in s100.get_messages() if m.text]
        texts_200 = [m.text for m in s200.get_messages() if m.text]
        assert "from chat 100" in texts_100
        assert "from chat 200" not in texts_100
        assert "from chat 200" in texts_200

    @pytest.mark.asyncio
    async def test_error_rolls_back_user_message(self):
        sm = SessionManager()
        agent_loop = AsyncMock()
        agent_loop.run_async = AsyncMock(side_effect=RuntimeError("model down"))

        adapter = _make_telegram_adapter(agent_loop=agent_loop, session_manager=sm)
        await adapter.on_message(_make_event(text="should be rolled back"))

        session = sm.get_or_create("telegram:123")
        assert len(session.get_messages()) == 0

    @pytest.mark.asyncio
    async def test_reset_clears_session(self):
        sm = SessionManager()
        result = _make_run_result("hi", "hello")
        agent_loop = AsyncMock()
        agent_loop.run_async = AsyncMock(return_value=result)

        adapter = _make_telegram_adapter(agent_loop=agent_loop, session_manager=sm)
        await adapter.on_message(_make_event(chat_id=123, text="hi"))

        session = sm.get_or_create("telegram:123")
        assert len(session.get_messages()) > 0

        # Simulate /reset
        update = MagicMock()
        update.effective_chat = MagicMock()
        update.effective_chat.id = 123
        await adapter._cmd_reset(update, MagicMock())

        assert len(session.get_messages()) == 0

    @pytest.mark.asyncio
    async def test_tool_calls_preserved_in_session(self):
        sm = SessionManager()
        result = _make_run_result("do it", "done", with_tool_call=True)
        agent_loop = AsyncMock()
        agent_loop.run_async = AsyncMock(return_value=result)

        adapter = _make_telegram_adapter(agent_loop=agent_loop, session_manager=sm)
        await adapter.on_message(_make_event(text="do it"))

        session = sm.get_or_create("telegram:123")
        # user + assistant(tool_use) + user(tool_result) + assistant(final) = 4
        assert len(session.get_messages()) == 4

    @pytest.mark.asyncio
    async def test_session_limit_enforced(self):
        sm = SessionManager()
        agent_loop = AsyncMock()

        # Pre-fill session close to limit
        session = sm.get_or_create("telegram:123")
        for i in range(MAX_SESSION_MESSAGES):
            session.add_user_message(f"old msg {i}")

        result = _make_run_result("new msg", "new reply")
        agent_loop.run_async = AsyncMock(return_value=result)

        adapter = _make_telegram_adapter(agent_loop=agent_loop, session_manager=sm)
        await adapter.on_message(_make_event(text="new msg"))

        assert len(session.get_messages()) <= MAX_SESSION_MESSAGES


# ===================================================================
# Slack adapter integration tests
# ===================================================================


def _make_slack_adapter(agent_loop=None, session_manager=None):
    from prometheus.gateway.slack import SlackAdapter

    config = PlatformConfig(
        platform=Platform.SLACK,
        token="xoxb-test",
        app_token="xapp-test",
    )
    if agent_loop is None:
        agent_loop = AsyncMock()
    adapter = SlackAdapter(
        config=config,
        agent_loop=agent_loop,
        tool_registry=ToolRegistry(),
        session_manager=session_manager,
    )
    adapter._add_reaction = AsyncMock()
    adapter._remove_reaction = AsyncMock()
    return adapter


class TestSlackSessionIntegration:
    @pytest.mark.asyncio
    async def test_dispatch_accumulates_history(self):
        sm = SessionManager()
        result1 = _make_run_result("hello", "hi there")
        result2 = _make_run_result("how are you", "I'm good")
        agent_loop = AsyncMock()
        agent_loop.run_async = AsyncMock(side_effect=[result1, result2])

        adapter = _make_slack_adapter(agent_loop=agent_loop, session_manager=sm)
        say = AsyncMock()

        await adapter._dispatch_to_agent("C123", "U456", "hello", "ts1", None, say)
        await adapter._dispatch_to_agent("C123", "U456", "how are you", "ts2", None, say)

        second_call = agent_loop.run_async.call_args_list[1]
        messages = second_call.kwargs["messages"]
        texts = [m.text for m in messages if m.text]
        assert "hello" in texts
        assert "hi there" in texts
        assert "how are you" in texts

    @pytest.mark.asyncio
    async def test_error_rolls_back_user_message(self):
        sm = SessionManager()
        agent_loop = AsyncMock()
        agent_loop.run_async = AsyncMock(side_effect=RuntimeError("model down"))

        adapter = _make_slack_adapter(agent_loop=agent_loop, session_manager=sm)
        say = AsyncMock()

        await adapter._dispatch_to_agent("C123", "U456", "fail", "ts1", None, say)

        session = sm.get_or_create("slack:C123")
        assert len(session.get_messages()) == 0

    @pytest.mark.asyncio
    async def test_reset_clears_session(self):
        sm = SessionManager()
        session = sm.get_or_create("slack:C123")
        session.add_user_message("hi")

        adapter = _make_slack_adapter(session_manager=sm)
        body = {"channel_id": "C123"}
        await adapter._slash_reset(ack=AsyncMock(), respond=AsyncMock(), body=body)

        assert len(session.get_messages()) == 0


# ===================================================================
# Cross-gateway isolation
# ===================================================================


class TestCrossGatewayIsolation:
    @pytest.mark.asyncio
    async def test_telegram_and_slack_same_id_isolated(self):
        """Telegram chat 123 and Slack channel 123 must NOT share state."""
        sm = SessionManager()
        result = _make_run_result("msg", "reply")

        tg_loop = AsyncMock()
        tg_loop.run_async = AsyncMock(return_value=result)
        sl_loop = AsyncMock()
        sl_loop.run_async = AsyncMock(return_value=result)

        tg = _make_telegram_adapter(agent_loop=tg_loop, session_manager=sm)
        sl = _make_slack_adapter(agent_loop=sl_loop, session_manager=sm)

        await tg.on_message(_make_event(chat_id=123, text="telegram msg"))
        say = AsyncMock()
        await sl._dispatch_to_agent("123", "U1", "slack msg", "ts1", None, say)

        tg_session = sm.get_or_create("telegram:123")
        sl_session = sm.get_or_create("slack:123")

        tg_texts = [m.text for m in tg_session.get_messages() if m.text]
        sl_texts = [m.text for m in sl_session.get_messages() if m.text]

        assert "telegram msg" in tg_texts
        assert "telegram msg" not in sl_texts
        assert "slack msg" in sl_texts
        assert "slack msg" not in tg_texts
