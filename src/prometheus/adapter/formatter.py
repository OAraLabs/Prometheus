"""Model-specific prompt formatting for tool calling.

Each formatter knows how to:
  - format_tools(tools)                           → model-specific tool schemas
  - format_system_prompt(base, tools, context)    → augmented system prompt
  - parse_tool_calls(raw_response)                → list[ToolUseBlock]

Formatters are applied *before* sending to the LLM so each provider gets
the representation it needs.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from prometheus.engine.messages import ToolUseBlock


class ModelPromptFormatter(ABC):
    """Base class for model-specific prompt formatting."""

    @abstractmethod
    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transform tool schemas into the model's preferred format."""

    @abstractmethod
    def format_system_prompt(
        self,
        base_prompt: str,
        tools: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Build the full system prompt, injecting tool descriptions if needed."""

    @abstractmethod
    def parse_tool_calls(self, raw_response: str) -> list[ToolUseBlock]:
        """Parse tool calls embedded in raw model text output."""


# ---------------------------------------------------------------------------
# AnthropicFormatter — passthrough for the Anthropic API
# ---------------------------------------------------------------------------

class AnthropicFormatter(ModelPromptFormatter):
    """Passthrough formatter — Anthropic's API handles tool formatting natively."""

    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return tools  # Already Anthropic format

    def format_system_prompt(
        self,
        base_prompt: str,
        tools: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> str:
        return base_prompt  # Anthropic handles tool injection

    def parse_tool_calls(self, raw_response: str) -> list[ToolUseBlock]:
        return []  # Anthropic API returns structured tool_use blocks directly


class PassthroughFormatter(ModelPromptFormatter):
    """For cloud API models that handle tool formatting natively.

    Used by: OpenAI, Gemini, xAI — all support native function calling.
    The provider converts to wire format; the formatter does nothing.
    """

    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return tools

    def format_system_prompt(
        self,
        base_prompt: str,
        tools: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> str:
        return base_prompt

    def parse_tool_calls(self, raw_response: str) -> list[ToolUseBlock]:
        return []  # Cloud APIs return structured tool calls directly


# ---------------------------------------------------------------------------
# QwenFormatter — OpenAI-compatible with explicit examples
# ---------------------------------------------------------------------------

_QWEN_TOOL_CALLING_EXAMPLE = """
When you want to use a tool, respond ONLY with a JSON object in this exact format:
```json
{"name": "<tool_name>", "arguments": {"<param>": "<value>"}}
```
Do not include any text before or after the JSON. After receiving the tool result, you can continue your response.
""".strip()


class QwenFormatter(ModelPromptFormatter):
    """Formatter for Qwen and similar OpenAI-compatible models.

    Injects explicit tool-calling instructions and an example into the system
    prompt to improve reliability with open weights models.
    """

    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic-format tools to OpenAI function-calling format."""
        result = []
        for t in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", t.get("parameters", {})),
                },
            })
        return result

    def format_system_prompt(
        self,
        base_prompt: str,
        tools: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> str:
        if not tools:
            return base_prompt

        tool_list = "\n".join(
            f"- {t['name']}: {t.get('description', '')}" for t in tools
        )
        return (
            f"{base_prompt}\n\n"
            f"You have access to these tools:\n{tool_list}\n\n"
            f"{_QWEN_TOOL_CALLING_EXAMPLE}"
        )

    def parse_tool_calls(self, raw_response: str) -> list[ToolUseBlock]:
        """Extract tool calls from Qwen's text output.

        Handles:
        - Clean JSON: {"name": "...", "arguments": {...}}
        - JSON in markdown: ```json {...} ```
        - Multiple calls separated by newlines
        """
        results: list[ToolUseBlock] = []

        # Find all ```json ... ``` blocks
        for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", raw_response, re.DOTALL):
            block = _parse_tool_call_json(m.group(1))
            if block:
                results.append(block)

        if results:
            return results

        # Whole response is a JSON object
        stripped = raw_response.strip()
        if stripped.startswith("{"):
            block = _parse_tool_call_json(stripped)
            if block:
                return [block]

        # Any line that is itself a JSON object
        for line in raw_response.splitlines():
            line = line.strip()
            if line.startswith("{"):
                block = _parse_tool_call_json(line)
                if block:
                    results.append(block)

        return results


# ---------------------------------------------------------------------------
# GemmaFormatter — Google's native function-calling format
# ---------------------------------------------------------------------------

_GEMMA_TOOL_EXAMPLE = """
To call a function, emit exactly this format (no other text on the same line):
<tool_call>{"name": "<function_name>", "arguments": {"key": "value"}}</tool_call>
After receiving <tool_response>...</tool_response>, continue your answer normally.
""".strip()


class GemmaFormatter(ModelPromptFormatter):
    """Formatter for Gemma models using Google's native function-calling tokens."""

    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert to Gemma's function declaration format."""
        result = []
        for t in tools:
            result.append({
                "function_declarations": [{
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", t.get("parameters", {})),
                }]
            })
        return result

    def format_system_prompt(
        self,
        base_prompt: str,
        tools: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> str:
        if not tools:
            return base_prompt

        tool_list = "\n".join(
            f"- {t['name']}: {t.get('description', '')}" for t in tools
        )
        return (
            f"{base_prompt}\n\n"
            f"Available functions:\n{tool_list}\n\n"
            f"{_GEMMA_TOOL_EXAMPLE}"
        )

    def parse_tool_calls(self, raw_response: str) -> list[ToolUseBlock]:
        """Parse <tool_call>...</tool_call> tags from Gemma output."""
        results: list[ToolUseBlock] = []
        for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", raw_response, re.DOTALL):
            block = _parse_tool_call_json(m.group(1))
            if block:
                results.append(block)
        return results


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _parse_tool_call_json(text: str) -> ToolUseBlock | None:
    """Try to parse a JSON string as a tool call. Returns None on failure."""
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    name = data.get("name") or data.get("function") or data.get("tool")
    if not name or not isinstance(name, str):
        return None
    args = data.get("arguments") or data.get("parameters") or data.get("args") or {}
    if not isinstance(args, dict):
        return None
    return ToolUseBlock(
        id=f"toolu_{uuid4().hex[:12]}",
        name=name,
        input=args,
    )
