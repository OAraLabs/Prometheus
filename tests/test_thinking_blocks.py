"""#333 — thinking blocks persist and round-trip (Beacon B2's daemon half).

One wire vocabulary for every provider: the Anthropic thinking-block shape.
The pins that matter:

- the Anthropic stream's thinking/redacted_thinking blocks survive into the
  final ConversationMessage (they were silently dropped at assembly);
- OpenAI-style ``reasoning_content`` normalizes into the SAME block;
- round-trip correctness both ways: Anthropic gets signed thinking back
  (it requires it in multi-turn tool use) and never an unsigned block; the
  OpenAI wire NEVER sees thinking — and the builder's loud
  UnsupportedContentBlock else-branch would have raised on every request
  containing a thinking turn had the skip not been explicit;
- flat text excludes thinking (it is not the reply), content_json carries
  it (that is what Beacon renders), from_stored brings it back.

SSE harness mirrors tests/test_cloud_providers.py's fake-httpx pattern.
"""

from __future__ import annotations

import json

import pytest

from prometheus.engine.messages import (
    ConversationMessage,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from prometheus.providers.anthropic import _build_anthropic_messages
from prometheus.providers.stub import (
    _build_openai_messages,
    _parse_assistant_message,
)


class _FakeSSEResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.status_code = 200

    async def __aenter__(self) -> "_FakeSSEResponse":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeHttpxClient:
    def __init__(self, *args: object, lines: list[str] | None = None,
                 **kw: object) -> None:
        self._lines = lines or []

    async def __aenter__(self) -> "_FakeHttpxClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def stream(self, method: str, url: str, **kwargs: object) -> _FakeSSEResponse:
        return _FakeSSEResponse(self._lines)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}"


def _thinking_msg() -> ConversationMessage:
    return ConversationMessage(role="assistant", content=[
        ThinkingBlock(thinking="let me consider", signature="sig-abc"),
        ThinkingBlock(thinking="normalized span, no signature"),
        RedactedThinkingBlock(data="opaque-bytes"),
        TextBlock(text="the answer"),
        ToolUseBlock(id="t1", name="bash", input={"command": "ls"}),
    ])


class TestMessageModel:
    def test_flat_text_excludes_thinking_content_json_carries_it(self) -> None:
        msg = _thinking_msg()
        assert msg.text == "the answer"
        stored = json.loads(msg.content_json)
        assert [b["type"] for b in stored] == [
            "thinking", "thinking", "redacted_thinking", "text", "tool_use",
        ]
        assert stored[0]["signature"] == "sig-abc"

    def test_from_stored_round_trips(self) -> None:
        msg = _thinking_msg()
        back = ConversationMessage.from_stored(
            role="assistant", content=msg.text, content_json=msg.content_json,
        )
        assert type(back.content[0]).__name__ == "ThinkingBlock"
        assert back.content[0].signature == "sig-abc"
        assert type(back.content[2]).__name__ == "RedactedThinkingBlock"


class TestAnthropicRoundTrip:
    def test_signed_thinking_rides_back_unsigned_does_not(self) -> None:
        wire = _build_anthropic_messages([_thinking_msg()])
        blocks = wire[0]["content"]
        kinds = [b["type"] for b in blocks]
        # Signed thinking + redacted preserved (Anthropic requires both in
        # multi-turn tool use); the UNSIGNED normalized span is dropped —
        # the API rejects unsigned thinking.
        assert kinds == ["thinking", "redacted_thinking", "text", "tool_use"]
        assert blocks[0]["signature"] == "sig-abc"
        assert blocks[1]["data"] == "opaque-bytes"

    @pytest.mark.asyncio
    async def test_stream_assembles_thinking_blocks(self, monkeypatch) -> None:
        import httpx

        from prometheus.providers.anthropic import AnthropicProvider
        from prometheus.providers.base import (
            ApiMessageCompleteEvent, ApiMessageRequest, ApiTextDeltaEvent,
        )

        lines = [
            _sse({"type": "message_start",
                  "message": {"usage": {"input_tokens": 5}}}),
            _sse({"type": "content_block_start", "index": 0,
                  "content_block": {"type": "thinking", "thinking": ""}}),
            _sse({"type": "content_block_delta", "index": 0,
                  "delta": {"type": "thinking_delta", "thinking": "hmm, "}}),
            _sse({"type": "content_block_delta", "index": 0,
                  "delta": {"type": "thinking_delta", "thinking": "greppable"}}),
            _sse({"type": "content_block_delta", "index": 0,
                  "delta": {"type": "signature_delta", "signature": "sig-1"}}),
            _sse({"type": "content_block_stop", "index": 0}),
            _sse({"type": "content_block_start", "index": 1,
                  "content_block": {"type": "redacted_thinking",
                                    "data": "hidden"}}),
            _sse({"type": "content_block_stop", "index": 1}),
            _sse({"type": "content_block_start", "index": 2,
                  "content_block": {"type": "text", "text": ""}}),
            _sse({"type": "content_block_delta", "index": 2,
                  "delta": {"type": "text_delta", "text": "the reply"}}),
            _sse({"type": "content_block_stop", "index": 2}),
            _sse({"type": "message_delta",
                  "delta": {"stop_reason": "end_turn"},
                  "usage": {"output_tokens": 7}}),
        ]
        monkeypatch.setattr(
            httpx, "AsyncClient",
            lambda *a, **kw: _FakeHttpxClient(lines=lines),
        )
        provider = AnthropicProvider(api_key="test-key")
        request = ApiMessageRequest(
            model="claude-fable-5", max_tokens=64,
            messages=[ConversationMessage.from_user_text("why?")],
        )
        deltas: list[str] = []
        final: ApiMessageCompleteEvent | None = None
        async for event in provider.stream_message(request):
            if isinstance(event, ApiTextDeltaEvent):
                deltas.append(event.text)
            elif isinstance(event, ApiMessageCompleteEvent):
                final = event

        assert final is not None
        kinds = [type(b).__name__ for b in final.message.content]
        assert kinds == ["ThinkingBlock", "RedactedThinkingBlock", "TextBlock"]
        assert final.message.content[0].thinking == "hmm, greppable"
        assert final.message.content[0].signature == "sig-1"
        assert final.message.content[1].data == "hidden"
        # Thinking never streamed as visible text.
        assert deltas == ["the reply"]


class TestOpenAiWire:
    def test_thinking_is_deliberately_skipped_not_raised(self) -> None:
        # THE landmine: the builder's else-branch raises
        # UnsupportedContentBlock on unknown blocks — without the explicit
        # skip, every request whose history contains a thinking turn dies.
        from prometheus.providers.base import ApiMessageRequest

        request = ApiMessageRequest(
            model="qwen", max_tokens=64, messages=[_thinking_msg()],
        )
        wire = _build_openai_messages(request)
        entry = wire[-1]
        assert entry["role"] == "assistant"
        assert entry["content"] == "the answer"
        assert "thinking" not in json.dumps(wire)

    def test_reasoning_content_normalizes_into_a_thinking_block(self) -> None:
        msg, dropped = _parse_assistant_message({
            "message": {
                "content": "final answer",
                "reasoning_content": "chain of thought here",
                "tool_calls": None,
            }
        })
        assert dropped == 0
        kinds = [type(b).__name__ for b in msg.content]
        assert kinds == ["ThinkingBlock", "TextBlock"]
        assert msg.content[0].thinking == "chain of thought here"
        assert msg.content[0].signature is None

    @pytest.mark.parametrize("empty", [None, "", "   "])
    def test_empty_reasoning_is_omitted(self, empty) -> None:
        msg, _ = _parse_assistant_message({
            "message": {"content": "x", "reasoning_content": empty,
                        "tool_calls": None}
        })
        assert [type(b).__name__ for b in msg.content] == ["TextBlock"]
