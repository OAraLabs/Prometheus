# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/tools/base.py
# License: MIT
# Modified: renamed imports (openharness → prometheus);
#           added to_openai_schema() for llama.cpp / OpenAI-compatible function calling;
#           added list_schemas() alias and list_schemas_for_task() for dynamic tool loading

"""Tool abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel


@dataclass
class ToolExecutionContext:
    """Shared execution context for tool invocations."""

    cwd: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """Normalized tool execution result."""

    output: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Base class for all Prometheus tools."""

    name: str
    description: str
    input_model: type[BaseModel]
    example_call: dict[str, Any] | None = None  # Static example for error feedback

    @abstractmethod
    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        """Execute the tool."""

    def is_read_only(self, arguments: BaseModel) -> bool:
        """Return whether the invocation is read-only."""
        del arguments
        return False

    def to_api_schema(self) -> dict[str, Any]:
        """Return the tool schema in Anthropic Messages API format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }

    def to_openai_schema(self) -> dict[str, Any]:
        """Return the tool schema in OpenAI function-calling format (for llama.cpp)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


class ToolRegistry:
    """Map tool names to implementations."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Return a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def to_api_schema(self) -> list[dict[str, Any]]:
        """Return all tool schemas in Anthropic API format."""
        return [tool.to_api_schema() for tool in self._tools.values()]

    def list_schemas(self) -> list[dict[str, Any]]:
        """Alias for to_api_schema() — returns all tool schemas in Anthropic format."""
        return self.to_api_schema()

    def to_openai_schemas(self) -> list[dict[str, Any]]:
        """Return all tool schemas in OpenAI function-calling format."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def schemas_for_names(self, names: list[str]) -> list[dict[str, Any]]:
        """Return tool schemas for the given tool names (preserving order)."""
        return [
            self._tools[n].to_api_schema()
            for n in names
            if n in self._tools
        ]

    def list_schemas_for_task(self, task_description: str) -> list[dict[str, Any]]:
        """Return tool schemas relevant to the given task description.

        Uses simple keyword matching against tool names and descriptions.
        Falls back to all schemas if no match is found.
        """
        words = set(task_description.lower().split())
        matched: list[BaseTool] = []
        for tool in self._tools.values():
            haystack = f"{tool.name} {tool.description}".lower()
            if any(word in haystack for word in words):
                matched.append(tool)
        if not matched:
            return self.to_api_schema()
        return [tool.to_api_schema() for tool in matched]
