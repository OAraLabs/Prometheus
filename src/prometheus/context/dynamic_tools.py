"""DynamicToolLoader — task-adaptive tool selection for Sprint 4.

Reduces the tool list sent to the model based on the current task:
  - Core tools are always included (bash, read_file, write_file).
  - Task-based: keyword matching adds relevant tools.
  - On-demand: if model requests an unknown tool, load it from the registry.

Usage:
    loader = DynamicToolLoader(registry)
    schemas = loader.active_schemas(task_description="read config and grep for errors")
    # → includes bash, read_file, write_file, grep (keyword matched)
"""

from __future__ import annotations

from typing import Any

from prometheus.tools.base import ToolRegistry

# Tools always included regardless of task
CORE_TOOLS: frozenset[str] = frozenset({"bash", "read_file", "write_file"})

# Keyword → additional tool names to include
_KEYWORD_TOOL_MAP: dict[str, list[str]] = {
    "grep": ["grep"],
    "search": ["grep", "web_search"],
    "find": ["grep", "glob"],
    "glob": ["glob"],
    "pattern": ["glob"],
    "edit": ["edit_file"],
    "modify": ["edit_file"],
    "replace": ["edit_file"],
    "patch": ["edit_file"],
    "list": ["glob"],
    "files": ["glob"],
    # Web tools
    "web": ["web_search", "web_fetch"],
    "url": ["web_fetch"],
    "fetch": ["web_fetch"],
    "browse": ["browser"],
    "navigate": ["browser"],
    "website": ["web_fetch", "browser"],
    # Messaging
    "message": ["message"],
    "send": ["message"],
    "discord": ["message"],
    "slack": ["message"],
    "telegram": ["message"],
    # Audio / TTS
    "speak": ["tts"],
    "voice": ["tts"],
    "audio": ["tts"],
    "speech": ["tts"],
    # Dashboard / visualization
    "dashboard": ["dashboard"],
    "html": ["dashboard"],
    "visuali": ["dashboard"],
    "serve": ["dashboard"],
    # Notebooks
    "notebook": ["notebook_edit"],
    "jupyter": ["notebook_edit"],
    "ipynb": ["notebook_edit"],
    # Sessions
    "session": ["sessions_list", "sessions_send", "sessions_spawn"],
    "agent": ["sessions_list", "sessions_spawn"],
    # User interaction
    "ask": ["ask_user"],
    "clarify": ["ask_user"],
    "question": ["ask_user"],
}


class DynamicToolLoader:
    """Select an appropriate subset of tools for a given task.

    Args:
        registry: Populated ToolRegistry (all available tools).
        deferred_config: Optional config dict from tools.deferred_loading.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        deferred_config: dict[str, Any] | None = None,
    ) -> None:
        self._registry = registry
        self._deferred = deferred_config or {}
        self._deferred_enabled = self._deferred.get("enabled", False)
        self._always_loaded: frozenset[str] = frozenset(
            self._deferred.get("always_loaded", [])
        ) if self._deferred_enabled else frozenset()

    def active_schemas(
        self,
        task_description: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return tool schemas appropriate for *task_description*.

        Always includes CORE_TOOLS.  Additional tools are added by
        keyword-matching *task_description* against _KEYWORD_TOOL_MAP.
        Falls back to all registered tools if no task description is given.

        Args:
            task_description: Free-text description of the current task.

        Returns:
            List of tool schemas in Anthropic API format.
        """
        # Deferred loading: only include always_loaded tools in the prompt
        if self._deferred_enabled:
            return self._deferred_schemas()

        if task_description is None:
            return self._registry.to_api_schema()

        selected: set[str] = set(CORE_TOOLS)
        words = set(task_description.lower().split())

        for keyword, tools in _KEYWORD_TOOL_MAP.items():
            if keyword in words:
                selected.update(tools)

        schemas: list[dict[str, Any]] = []
        for tool in self._registry.list_tools():
            if tool.name in selected:
                schemas.append(tool.to_api_schema())

        # If nothing extra matched (only core), return all to avoid over-pruning
        if not schemas:
            return self._registry.to_api_schema()

        return schemas

    def on_demand(self, tool_name: str) -> dict[str, Any] | None:
        """Return the schema for *tool_name* if it exists in the registry.

        Called when the model requests a tool not in the active schema set.

        Args:
            tool_name: Tool name the model is trying to call.

        Returns:
            Tool schema dict, or None if the tool is not registered.
        """
        tool = self._registry.get(tool_name)
        if tool is None:
            return None
        return tool.to_api_schema()

    def _deferred_schemas(self) -> list[dict[str, Any]]:
        """Return only schemas for always_loaded tools (deferred mode)."""
        schemas = []
        for tool in self._registry.list_tools():
            if tool.name in self._always_loaded:
                schemas.append(tool.to_api_schema())
        return schemas

    @property
    def deferred_count(self) -> int:
        """Number of tools deferred (not in prompt) when deferred loading is on."""
        if not self._deferred_enabled:
            return 0
        total = len(self._registry.list_tools())
        loaded = len(self._always_loaded)
        return max(0, total - loaded)

    def all_schemas(self) -> list[dict[str, Any]]:
        """Return schemas for every registered tool."""
        return self._registry.to_api_schema()
