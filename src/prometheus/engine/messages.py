# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/engine/messages.py
# License: MIT
# Modified: renamed imports (openharness → prometheus); removed Anthropic-specific
#           assistant_message_from_api() — provider now handles response parsing

"""Conversation message models used by the query engine."""

from __future__ import annotations

from typing import Any, Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    """Plain text content."""

    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    """A request from the model to execute a named tool."""

    type: Literal["tool_use"] = "tool_use"
    id: str = Field(default_factory=lambda: f"toolu_{uuid4().hex}")
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    """Tool result content sent back to the model."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = Annotated[TextBlock | ToolUseBlock | ToolResultBlock, Field(discriminator="type")]


class ConversationMessage(BaseModel):
    """A single assistant or user message."""

    role: Literal["user", "assistant"]
    content: list[ContentBlock] = Field(default_factory=list)

    @classmethod
    def from_user_text(cls, text: str) -> "ConversationMessage":
        """Construct a user message from raw text."""
        return cls(role="user", content=[TextBlock(text=text)])

    @property
    def text(self) -> str:
        """Return concatenated text blocks."""
        return "".join(
            block.text for block in self.content if isinstance(block, TextBlock)
        )

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        """Return all tool calls contained in the message."""
        return [block for block in self.content if isinstance(block, ToolUseBlock)]

    def to_openai_param(self) -> dict[str, Any]:
        """Convert the message into OpenAI-compatible message params."""
        blocks = []
        tool_calls = []

        for block in self.content:
            if isinstance(block, TextBlock):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolUseBlock):
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": __import__("json").dumps(block.input),
                    },
                })
            elif isinstance(block, ToolResultBlock):
                # Tool results are separate messages in OpenAI format
                blocks.append({"type": "text", "text": block.content})

        param: dict[str, Any] = {"role": self.role}
        if self.role == "assistant" and tool_calls:
            param["tool_calls"] = tool_calls
            if blocks:
                param["content"] = " ".join(b["text"] for b in blocks if b.get("type") == "text")
        else:
            param["content"] = " ".join(b["text"] for b in blocks if b.get("type") == "text") or ""

        return param

    def to_api_param(self) -> dict[str, Any]:
        """Convert the message into provider wire format (OpenAI-compatible)."""
        return self.to_openai_param()


def serialize_content_block(block: ContentBlock) -> dict[str, Any]:
    """Convert a local content block into the provider wire format."""
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}

    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }

    return {
        "type": "tool_result",
        "tool_use_id": block.tool_use_id,
        "content": block.content,
        "is_error": block.is_error,
    }
