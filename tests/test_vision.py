"""Tests for Sprint 15 GRAFT: VisionTool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

import pytest

from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiStreamEvent,
    ApiTextDeltaEvent,
    ModelProvider,
)
from prometheus.tools.builtin.vision import VisionTool, VisionInput, _detect_mime
from prometheus.tools.base import ToolExecutionContext, ToolResult


class TestVisionTool:

    def test_detect_mime_jpeg(self):
        assert _detect_mime(b"\xff\xd8\xff\xe0") == "image/jpeg"

    def test_detect_mime_png(self):
        assert _detect_mime(b"\x89PNG\r\n\x1a\n") == "image/png"

    def test_detect_mime_gif(self):
        assert _detect_mime(b"GIF89a") == "image/gif"

    def test_detect_mime_unknown(self):
        assert _detect_mime(b"random") == "image/jpeg"

    def test_file_not_found(self):
        tool = VisionTool()
        result = asyncio.run(
            tool.execute(
                VisionInput(image_path="/nonexistent/image.jpg"),
                ToolExecutionContext(cwd=Path.cwd()),
            )
        )
        assert result.is_error
        assert "not found" in result.output.lower()

    def test_no_provider(self, tmp_path):
        # Create a minimal JPEG file
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        tool = VisionTool()
        result = asyncio.run(
            tool.execute(
                VisionInput(image_path=str(img)),
                ToolExecutionContext(cwd=Path.cwd(), metadata={}),
            )
        )
        assert result.is_error
        assert "provider" in result.output.lower()

    def test_is_read_only(self):
        tool = VisionTool()
        assert tool.is_read_only(VisionInput(image_path="/x.jpg"))


# ---------------------------------------------------------------------------
# The streaming path: the description is the model's text ONCE
# ---------------------------------------------------------------------------
#
# Every provider streams ApiTextDeltaEvent chunks and then ONE terminal
# ApiMessageCompleteEvent that carries the same text again as the whole
# message. VisionTool appended both, so every description arrived twice, back
# to back with no separator ("...may indicate:\n- User needs toThis is a
# screenshot of..."), the second copy cut at the same max_tokens boundary as
# the first. A Beacon upload to a non-vision model puts that description in the
# user's own message ("[Image: <desc>]"), so the user saw the wall of text
# repeated in their bubble and the model was handed it twice.
#
# The scripted provider streams the way llama.cpp / openai_compat / anthropic
# do. A provider stub that yielded only the complete event would pass the old
# code, which is why test_complete_event_alone_still_describes pins the other
# shape separately.

class _ScriptedProvider(ModelProvider):
    def __init__(self, events: list[ApiStreamEvent]) -> None:
        self._events = events

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        for event in self._events:
            yield event


def _complete(text: str) -> ApiMessageCompleteEvent:
    return ApiMessageCompleteEvent(
        message=ConversationMessage(role="assistant", content=[TextBlock(text=text)]),
        usage=UsageSnapshot(input_tokens=10, output_tokens=5),
        stop_reason="length",
    )


def _describe(tmp_path, provider: ModelProvider) -> ToolResult:
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return asyncio.run(
        VisionTool().execute(
            VisionInput(image_path=str(img)),
            ToolExecutionContext(cwd=Path.cwd(), metadata={"provider": provider}),
        )
    )


class TestVisionDescribesOnce:

    def test_streamed_description_is_not_doubled(self, tmp_path):
        full = "This is a screenshot of a chat window. The browser pane on the right is empty."
        provider = _ScriptedProvider(
            [
                ApiTextDeltaEvent(text="This is a screenshot of a chat window. "),
                ApiTextDeltaEvent(text="The browser pane on the right is empty."),
                _complete(full),
            ]
        )
        result = _describe(tmp_path, provider)
        assert not result.is_error
        assert result.output == full
        assert result.output.count("This is a screenshot") == 1

    def test_complete_event_alone_still_describes(self, tmp_path):
        # A provider that yields no deltas must still produce a description.
        provider = _ScriptedProvider([_complete("A red square.")])
        result = _describe(tmp_path, provider)
        assert not result.is_error
        assert result.output == "A red square."
