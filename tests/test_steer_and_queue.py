"""SPRINT-2 WS1 — /steer + /queue functional tests.

Tests follow the Sprint 4 A5 functional-wiring pattern: assert side effects
actually occurred (the captured ``system_prompt`` contains the steer text,
the registry has the new tool, the queue has the right items) rather than
structural shape (a method exists, a slot is defined).

Spec'd in ``docs/sprints/...`` (Sprint 2). Hermes-attributed adaptation —
see ``src/prometheus/engine/session.py`` module docstring for the diff
against ``run_agent.py:AIAgent.steer``.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock

import pytest

from prometheus.engine.agent_loop import AgentLoop, LoopContext, run_loop
from prometheus.engine.messages import (
    ConversationMessage, TextBlock, ToolResultBlock, ToolUseBlock,
)
from prometheus.engine.session import ChatSession
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent, ApiMessageRequest, ApiTextDeltaEvent,
    ModelProvider,
)


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Provider stubs — capture the request so we can assert on it.
# ---------------------------------------------------------------------------


def _text_complete(text: str) -> list:
    msg = ConversationMessage(role="assistant", content=[TextBlock(text=text)])
    return [
        ApiTextDeltaEvent(text=text),
        ApiMessageCompleteEvent(
            message=msg,
            usage=UsageSnapshot(input_tokens=10, output_tokens=5),
            stop_reason="stop",
        ),
    ]


def _tool_complete(tool_id: str, name: str, tool_input: dict) -> list:
    msg = ConversationMessage(
        role="assistant",
        content=[ToolUseBlock(id=tool_id, name=name, input=tool_input)],
    )
    return [
        ApiMessageCompleteEvent(
            message=msg,
            usage=UsageSnapshot(input_tokens=10, output_tokens=10),
            stop_reason="tool_calls",
        )
    ]


class CapturingProvider(ModelProvider):
    """Mock provider that records every ``ApiMessageRequest`` it receives.

    Returns scripted responses in sequence. Used to assert what the loop
    actually sent to the model — specifically what ``system_prompt`` value
    each iteration carried.
    """

    def __init__(self, responses: list[list]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.captured_system_prompts: list[str] = []
        self.captured_requests: list[ApiMessageRequest] = []

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator:
        self.captured_requests.append(request)
        self.captured_system_prompts.append(request.system_prompt or "")
        events = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        for event in events:
            yield event


# ---------------------------------------------------------------------------
# Session-state queues — ChatSession contract
# ---------------------------------------------------------------------------


class TestChatSessionQueues:
    def test_enqueue_steer_appends_and_strips(self):
        s = ChatSession("t1")
        assert s.enqueue_steer("  hello  ") is True
        assert s.enqueue_steer("") is False  # empty → no-op
        assert s.queued_steers == ["hello"]

    def test_drain_steers_concatenates_with_blank_lines(self):
        s = ChatSession("t1")
        s.enqueue_steer("first")
        s.enqueue_steer("second")
        out = s.drain_steers()
        assert out == "first\n\nsecond"
        assert s.queued_steers == []  # cleared

    def test_drain_steers_returns_none_when_empty(self):
        s = ChatSession("t1")
        assert s.drain_steers() is None

    def test_enqueue_drain_prompt_is_fifo(self):
        s = ChatSession("t1")
        s.enqueue_prompt("first")
        s.enqueue_prompt("second")
        assert s.drain_prompt() == "first"
        assert s.drain_prompt() == "second"
        assert s.drain_prompt() is None

    def test_clear_steers_returns_count(self):
        s = ChatSession("t1")
        s.enqueue_steer("a")
        s.enqueue_steer("b")
        assert s.clear_steers() == 2
        assert s.queued_steers == []


# ---------------------------------------------------------------------------
# Run-loop integration — steer drains into system_prompt
# ---------------------------------------------------------------------------


class TestSteerAppendsToNextModelCall:
    @pytest.mark.asyncio
    async def test_steer_appears_in_next_system_prompt(self):
        """One queued steer + agent makes one model call → the system_prompt
        for that call contains the [STEER FROM USER, mid-turn] addendum."""
        session = ChatSession("t1")
        session.enqueue_steer("focus on Ubuntu, skip Mac")

        provider = CapturingProvider([_text_complete("Acknowledged.")])
        ctx = LoopContext(
            provider=provider,
            model="qwen-test",
            system_prompt="You are Prometheus.",
            max_tokens=1024,
            session_state=session,
        )
        messages = [ConversationMessage.from_user_text("write a setup guide")]
        async for _ in run_loop(ctx, messages):
            pass

        assert len(provider.captured_system_prompts) == 1
        sp = provider.captured_system_prompts[0]
        assert "[STEER FROM USER, mid-turn]" in sp, (
            f"Expected STEER addendum in system_prompt. Got: {sp!r}"
        )
        assert "focus on Ubuntu, skip Mac" in sp
        # The original system prompt is preserved.
        assert sp.startswith("You are Prometheus.")
        # And the queue is cleared after the drain.
        assert session.queued_steers == []

    @pytest.mark.asyncio
    async def test_multiple_steers_concatenate_in_one_addendum(self):
        """Hermes parity: multiple steers before drain combine."""
        session = ChatSession("t1")
        session.enqueue_steer("first guidance")
        session.enqueue_steer("second guidance")
        session.enqueue_steer("third guidance")

        provider = CapturingProvider([_text_complete("Done.")])
        ctx = LoopContext(
            provider=provider, model="qwen-test",
            system_prompt="sys.", max_tokens=1024, session_state=session,
        )
        messages = [ConversationMessage.from_user_text("hi")]
        async for _ in run_loop(ctx, messages):
            pass

        sp = provider.captured_system_prompts[0]
        for phrase in ("first guidance", "second guidance", "third guidance"):
            assert phrase in sp, f"Missing {phrase!r} in {sp!r}"

    @pytest.mark.asyncio
    async def test_no_steer_no_addendum(self):
        """Empty queue → system_prompt is unchanged from the original."""
        session = ChatSession("t1")
        provider = CapturingProvider([_text_complete("Done.")])
        ctx = LoopContext(
            provider=provider, model="qwen-test",
            system_prompt="You are Prometheus.", max_tokens=1024,
            session_state=session,
        )
        messages = [ConversationMessage.from_user_text("hi")]
        async for _ in run_loop(ctx, messages):
            pass

        assert provider.captured_system_prompts == ["You are Prometheus."]


# ---------------------------------------------------------------------------
# Steer preserves the tool-calling cycle
# ---------------------------------------------------------------------------


class _EchoToolRegistry:
    """Minimal tool registry that mimics what dispatch_tool_calls needs."""
    def __init__(self):
        self._tool = _EchoTool()
    def get(self, name): return self._tool if name == "echo" else None
    def get_tool(self, name): return self.get(name)
    def list_tools(self): return [self._tool]
    def list_schemas(self): return [{"name": "echo", "input_schema": {}}]


class _EchoTool:
    name = "echo"
    description = "echo"
    class input_model:
        @staticmethod
        def model_validate(d):
            return _Args(d)
    def is_read_only(self, parsed): return True
    async def execute(self, parsed, ctx):
        from prometheus.tools.base import ToolResult
        return ToolResult(output=f"echoed: {parsed.payload}")


class _Args:
    def __init__(self, d): self.payload = d.get("payload", "")


class TestSteerDoesNotBreakToolCycle:
    @pytest.mark.asyncio
    async def test_steer_arrives_after_tool_call_not_before(self):
        """Sequence:
          turn 1 — model calls a tool (no steer queued yet)
          user enqueues steer between turn 1 and turn 2
          turn 2 — model is called with the steer in system_prompt
        The tool-calling cycle is preserved: the loop didn't restart and
        the steer didn't insert as a user-role message."""
        session = ChatSession("t1")

        # Scripted: tool call → text completion
        provider = CapturingProvider([
            _tool_complete("c1", "echo", {"payload": "ping"}),
            _text_complete("All done."),
        ])

        # Inject a steer between the two model calls. We can't pause
        # mid-loop, so use a CapturingProvider subclass that enqueues
        # the steer right after recording the first request.
        original_stream = provider.stream_message

        async def stream_with_steer_after_first(request):
            # On the FIRST call, enqueue the steer mid-stream so it's
            # picked up by the SECOND model call only.
            if provider._call_count == 0:
                session.enqueue_steer("focus on lowercase")
            async for ev in original_stream(request):
                yield ev

        provider.stream_message = stream_with_steer_after_first  # type: ignore

        ctx = LoopContext(
            provider=provider, model="qwen-test",
            system_prompt="sys.", max_tokens=1024,
            tool_registry=_EchoToolRegistry(),
            session_state=session,
        )
        messages = [ConversationMessage.from_user_text("call echo with payload=ping")]
        async for _ in run_loop(ctx, messages):
            pass

        # Two model calls happened (turn 1 = tool call, turn 2 = final).
        assert len(provider.captured_system_prompts) == 2
        # Turn 1: no steer addendum.
        assert "STEER FROM USER" not in provider.captured_system_prompts[0]
        # Turn 2: steer addendum present.
        assert "STEER FROM USER" in provider.captured_system_prompts[1]
        assert "focus on lowercase" in provider.captured_system_prompts[1]

        # Conversation messages have the expected shape: user → assistant
        # (tool_use) → user (tool_result) → assistant (text). No
        # role="system" message in the conversation itself; the steer
        # lives entirely in the per-call system_prompt.
        roles = [m.role for m in messages]
        assert roles[0] == "user"   # original prompt
        assert roles[1] == "assistant"  # tool call
        assert roles[2] == "user"   # tool result (Prometheus convention)
        assert roles[3] == "assistant"  # final text
        for m in messages:
            assert m.role in ("user", "assistant"), (
                f"Conversation contains an unexpected role {m.role!r}; "
                f"steer should NOT inject as a conversation message."
            )


# ---------------------------------------------------------------------------
# /unqueue + /clear-steers + status surface
# ---------------------------------------------------------------------------


class TestQueueCancellation:
    def test_unqueue_pops_last_in(self):
        s = ChatSession("t1")
        s.enqueue_prompt("first")
        s.enqueue_prompt("second")
        # /unqueue drops the most recently queued (last-in).
        s.queued_prompts.pop()
        assert s.queued_prompts == ["first"]

    def test_clear_steers_empties_queue(self):
        s = ChatSession("t1")
        s.enqueue_steer("a")
        s.enqueue_steer("b")
        assert s.clear_steers() == 2
        assert s.drain_steers() is None


# ---------------------------------------------------------------------------
# Gateway-level: /queue fires as next turn
# ---------------------------------------------------------------------------


class TestQueueFiresAsNextTurn:
    """The Telegram gateway, after a turn ends, drains queued_prompts and
    dispatches the next one. We exercise this via the gateway's
    ``_dispatch_to_agent`` method directly to avoid spinning up the full
    Telegram adapter (real polling, real bot)."""

    @pytest.mark.asyncio
    async def test_queued_prompt_dispatches_as_next_user_turn(self):
        from prometheus.gateway.platform_base import (
            MessageEvent, Platform, MessageType,
        )
        from prometheus.gateway.telegram import TelegramAdapter
        from prometheus.engine.session import SessionManager
        from prometheus.engine.agent_loop import RunResult

        # Stub AgentLoop: records every user_message it was asked to run on
        # and returns an empty RunResult so the gateway moves on.
        seen_user_messages: list[str] = []

        class _StubAgentLoop:
            async def run_async(self, **kw):
                # Capture the LAST user message from the inbound messages
                # list (the gateway always appends the user message before
                # invoking run_async).
                msgs = kw.get("messages") or []
                if msgs:
                    seen_user_messages.append(msgs[-1].text)
                return RunResult(
                    text="ok",
                    messages=list(msgs),
                    usage=UsageSnapshot(),
                    turns=1,
                )

        # Build a TelegramAdapter without actually starting it.
        sm = SessionManager()
        chat_id = 99
        session_key = f"telegram:{chat_id}"
        session = sm.get_or_create(session_key)
        session.enqueue_prompt("follow-up task")

        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter.session_manager = sm
        adapter.agent_loop = _StubAgentLoop()
        adapter.system_prompt = "sys"
        # tool_registry only needs list_schemas() in this path.
        adapter.tool_registry = type(
            "_R", (), {"list_schemas": lambda self: []},
        )()
        adapter._app = None
        adapter.send = AsyncMock()

        event = MessageEvent(
            chat_id=chat_id, user_id=1, text="initial task",
            message_id=42, platform=Platform.TELEGRAM,
        )

        await adapter._dispatch_to_agent(event)

        # Two turns dispatched: the initial inbound message, then the
        # drained queued prompt.
        assert seen_user_messages == ["initial task", "follow-up task"], (
            f"Expected the queued prompt to fire as a second turn. "
            f"Got dispatched user messages: {seen_user_messages}"
        )
        # Queue is empty after drain.
        assert session.queued_prompts == []
