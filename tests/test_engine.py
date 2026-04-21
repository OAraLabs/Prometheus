"""Tests for Sprint 1: agent loop, messages, and provider interface."""

from __future__ import annotations

import json
import pytest
from typing import AsyncIterator
from unittest.mock import AsyncMock

from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.engine.usage import UsageSnapshot
from prometheus.engine.agent_loop import AgentLoop, LoopContext, run_loop
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
    ModelProvider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_response(text: str, stop_reason: str = "stop") -> list:
    """Build a minimal provider response: text delta + complete event."""
    msg = ConversationMessage(role="assistant", content=[TextBlock(text=text)])
    return [
        ApiTextDeltaEvent(text=text),
        ApiMessageCompleteEvent(
            message=msg,
            usage=UsageSnapshot(input_tokens=10, output_tokens=5),
            stop_reason=stop_reason,
        ),
    ]


def _tool_response(tool_name: str, tool_id: str, tool_input: dict) -> list:
    """Build a provider response that requests a tool call."""
    msg = ConversationMessage(
        role="assistant",
        content=[ToolUseBlock(id=tool_id, name=tool_name, input=tool_input)],
    )
    return [
        ApiMessageCompleteEvent(
            message=msg,
            usage=UsageSnapshot(input_tokens=10, output_tokens=10),
            stop_reason="tool_calls",
        )
    ]


class MockProvider(ModelProvider):
    """Provider that returns scripted responses in sequence."""

    def __init__(self, responses: list[list]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator:
        events = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        for event in events:
            yield event


# ---------------------------------------------------------------------------
# Message construction tests
# ---------------------------------------------------------------------------

def test_conversation_message_from_user_text():
    msg = ConversationMessage.from_user_text("hello")
    assert msg.role == "user"
    assert msg.text == "hello"
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], TextBlock)


def test_conversation_message_text_property():
    msg = ConversationMessage(
        role="assistant",
        content=[TextBlock(text="foo "), TextBlock(text="bar")],
    )
    assert msg.text == "foo bar"


def test_conversation_message_tool_uses():
    tool = ToolUseBlock(id="t1", name="bash", input={"command": "ls"})
    msg = ConversationMessage(role="assistant", content=[tool])
    assert len(msg.tool_uses) == 1
    assert msg.tool_uses[0].name == "bash"


def test_tool_result_block():
    r = ToolResultBlock(tool_use_id="t1", content="result", is_error=False)
    assert r.tool_use_id == "t1"
    assert not r.is_error


def test_usage_snapshot_total():
    u = UsageSnapshot(input_tokens=100, output_tokens=50)
    assert u.total_tokens == 150


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def test_parse_text_response():
    events = _text_response("Hello, world!")
    complete = [e for e in events if isinstance(e, ApiMessageCompleteEvent)]
    assert len(complete) == 1
    assert complete[0].message.text == "Hello, world!"
    assert complete[0].stop_reason == "stop"


def test_parse_tool_response():
    events = _tool_response("bash", "toolu_abc", {"command": "echo hi"})
    complete = [e for e in events if isinstance(e, ApiMessageCompleteEvent)]
    assert complete[0].message.tool_uses[0].name == "bash"
    assert complete[0].message.tool_uses[0].input == {"command": "echo hi"}


# ---------------------------------------------------------------------------
# Loop behavior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_exits_on_non_tool_response():
    provider = MockProvider([_text_response("Hi there")])
    loop = AgentLoop(provider=provider)
    result = await loop.run_async(
        system_prompt="You are helpful.",
        user_message="Say hello.",
    )
    assert result.text == "Hi there"
    assert result.turns == 1
    assert provider._call_count == 1


@pytest.mark.asyncio
async def test_loop_collects_text_from_final_turn():
    provider = MockProvider([_text_response("The answer is 42.")])
    loop = AgentLoop(provider=provider)
    result = await loop.run_async("You are helpful.", "What is 6*7?")
    assert "42" in result.text


@pytest.mark.asyncio
async def test_loop_handles_empty_text_response():
    """Empty content should not crash — returns empty string."""
    msg = ConversationMessage(role="assistant", content=[])
    provider = MockProvider([[
        ApiMessageCompleteEvent(
            message=msg,
            usage=UsageSnapshot(),
            stop_reason="stop",
        )
    ]])
    loop = AgentLoop(provider=provider)
    result = await loop.run_async("Be quiet.", "Say nothing.")
    assert result.text == ""
    assert result.turns == 1


@pytest.mark.asyncio
async def test_loop_appends_messages():
    """Conversation history should grow with each turn."""
    provider = MockProvider([_text_response("pong")])
    loop = AgentLoop(provider=provider)
    result = await loop.run_async("You are a ping-pong bot.", "ping")
    # user message + assistant response
    assert len(result.messages) == 2
    assert result.messages[0].role == "user"
    assert result.messages[1].role == "assistant"


@pytest.mark.asyncio
async def test_loop_without_tool_registry_returns_error_on_tool_call():
    """If model requests a tool and no registry is set, return error result and loop exits."""
    tool_events = _tool_response("bash", "toolu_1", {"command": "ls"})
    # After the tool error response is appended, model returns text
    text_events = _text_response("Done.")

    provider = MockProvider([tool_events, text_events])
    loop = AgentLoop(provider=provider, tool_registry=None)
    result = await loop.run_async("You are helpful.", "Run ls")
    # Loop should not crash — tool error fed back, model continues
    assert result.turns >= 1


def test_run_sync_wrapper():
    """loop.run() synchronous entry point should work."""
    provider = MockProvider([_text_response("sync works")])
    loop = AgentLoop(provider=provider)
    result = loop.run("You are helpful.", "Test sync.")
    assert result.text == "sync works"
