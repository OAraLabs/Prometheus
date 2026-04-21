"""Deferred tool loading via search and select."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Inline Levenshtein (avoids circular imports from adapter/validator)
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------

class ToolSearchInput(BaseModel):
    """Arguments for the tool_search tool."""

    query: str = Field(description="Search query: tool name, description keywords, or exact name for select")
    action: Literal["search", "select"] = Field(
        default="search",
        description="'search' to fuzzy-match tools, 'select' to load a specific tool by exact name",
    )


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

class ToolSearchTool(BaseTool):
    """Search for available tools and skills by name or description.

    Use 'search' to find tools or skills matching a query, or 'select'
    to load a specific tool/skill by exact name.
    """

    name = "tool_search"
    description = (
        "Search for available tools and skills by name or description. "
        "Use 'search' to find tools or skills matching a query, or 'select' to "
        "load a specific tool by exact name. Use the skill tool to load a skill's instructions."
    )
    input_model = ToolSearchInput

    def __init__(self) -> None:
        self._registry: ToolRegistry | None = None
        self._skill_registry: Any | None = None

    def set_registry(self, registry: ToolRegistry) -> None:
        """Inject the tool registry (called after construction)."""
        self._registry = registry

    def set_skill_registry(self, skill_registry: Any) -> None:
        """Inject the skill registry (called after construction)."""
        self._skill_registry = skill_registry

    def is_read_only(self, arguments: BaseModel) -> bool:
        del arguments
        return True

    async def execute(
        self,
        arguments: ToolSearchInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        if self._registry is None:
            return ToolResult(
                output="ToolSearchTool has no registry configured. Call set_registry() first.",
                is_error=True,
            )

        if arguments.action == "select":
            return self._handle_select(arguments.query)
        return self._handle_search(arguments.query)

    # -- action handlers ----------------------------------------------------

    def _handle_select(self, name: str) -> ToolResult:
        """Exact-name lookup; return the full schema for one tool or skill."""
        assert self._registry is not None
        tool = self._registry.get(name)
        if tool is not None:
            result = tool.to_api_schema()
            result["type"] = "tool"
            return ToolResult(output=json.dumps(result, indent=2))

        # Check skills
        if self._skill_registry is not None:
            skill = self._skill_registry.get(name)
            if skill is not None:
                return ToolResult(output=json.dumps({
                    "type": "skill",
                    "name": skill.name,
                    "description": skill.description,
                    "source": skill.source,
                    "path": skill.path,
                    "hint": f'Use skill(name="{skill.name}") to load full instructions.',
                }, indent=2))

        available = sorted(t.name for t in self._registry.list_tools())
        if self._skill_registry is not None:
            available.extend(f"[skill] {s.name}" for s in self._skill_registry.list_skills())
        return ToolResult(
            output=json.dumps({"error": f"No tool or skill named '{name}'.", "available": available}, indent=2),
            is_error=True,
        )

    def _handle_search(self, query: str) -> ToolResult:
        """Fuzzy search across tool names, descriptions, AND skill names/descriptions."""
        assert self._registry is not None
        tools = self._registry.list_tools()

        # Empty query: return all tool + skill names
        if not query.strip():
            names = sorted(t.name for t in tools)
            result: dict[str, Any] = {"tools": names}
            if self._skill_registry is not None:
                result["skills"] = sorted(s.name for s in self._skill_registry.list_skills())
            return ToolResult(output=json.dumps(result, indent=2))

        query_lower = query.lower()
        scored: list[tuple[float, str, dict[str, Any]]] = []  # (score, type, entry)

        # Score tools
        for tool in tools:
            score = self._score_tool(tool, query_lower)
            entry = tool.to_api_schema()
            entry["type"] = "tool"
            entry["match_score"] = round(score, 3)
            scored.append((score, "tool", entry))

        # Score skills
        if self._skill_registry is not None:
            for skill in self._skill_registry.list_skills():
                score = self._score_skill(skill, query_lower)
                entry = {
                    "type": "skill",
                    "name": skill.name,
                    "description": skill.description,
                    "source": skill.source,
                    "match_score": round(score, 3),
                }
                scored.append((score, "skill", entry))

        scored.sort(key=lambda t: t[0])
        results = [entry for _, _, entry in scored[:5]]
        return ToolResult(output=json.dumps(results, indent=2))

    # -- scoring ------------------------------------------------------------

    @staticmethod
    def _score_tool(tool: BaseTool, query_lower: str) -> float:
        """Score a tool against a query (lower is better).

        Combines substring matching and Levenshtein distance for fuzzy
        matching.  Exact substring hits in the name get the best score.
        """
        name_lower = tool.name.lower()
        desc_lower = tool.description.lower()

        # Exact name match
        if query_lower == name_lower:
            return 0.0

        # Substring in name
        if query_lower in name_lower:
            return 1.0

        # Substring in description
        if query_lower in desc_lower:
            return 2.0

        # Check if any query word is a substring of name or description
        words = query_lower.split()
        word_hits = sum(
            1 for w in words if w in name_lower or w in desc_lower
        )
        if word_hits > 0:
            return 3.0 - (word_hits / max(len(words), 1))

        # Levenshtein against the tool name (normalized)
        dist = _levenshtein(query_lower, name_lower)
        max_len = max(len(query_lower), len(name_lower), 1)
        normalized = dist / max_len
        return 4.0 + normalized

    @staticmethod
    def _score_skill(skill: Any, query_lower: str) -> float:
        """Score a skill against a query (lower is better). Same tiers as _score_tool."""
        name_lower = skill.name.lower()
        desc_lower = skill.description.lower()

        if query_lower == name_lower:
            return 0.0
        if query_lower in name_lower:
            return 1.0
        if query_lower in desc_lower:
            return 2.0

        words = query_lower.split()
        word_hits = sum(1 for w in words if w in name_lower or w in desc_lower)
        if word_hits > 0:
            return 3.0 - (word_hits / max(len(words), 1))

        dist = _levenshtein(query_lower, name_lower)
        max_len = max(len(query_lower), len(name_lower), 1)
        return 4.0 + dist / max_len
