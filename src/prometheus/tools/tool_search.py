"""Deferred tool loading via search and select."""

from __future__ import annotations

import json
import re
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
# Skill relevance
# ---------------------------------------------------------------------------

#: A skill is listed only at or above this match score (see skill_match_score).
#: Tools and skills share one top-5, and when a query matched no text anywhere
#: the ranking fell back to edit distance against the NAME — with ~134 skill
#: names against ~55 tool names that filled results with arbitrary skills (20
#: of the 25 user searches that listed one, docs/audits/SKILL-USAGE.md §1).
#: Tools keep their old ranking; only a skill must now actually match.
SKILL_MIN_MATCH = 0.5

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "into", "what", "how",
    "can", "you", "your", "are", "was", "were", "has", "have", "had", "not",
    "but", "all", "any", "its", "our", "out", "via", "per", "who", "why",
    "when", "where", "which", "will", "would", "should", "could", "about",
    "there", "their", "them", "then", "than", "these", "those", "just", "also",
    "some", "more", "most", "very", "only", "over", "such", "each", "both",
    "other",
})


def skill_match_score(query: str, name: str, description: str) -> float:
    """How well *query* matches a skill's name and description, 0.0–1.0.

    1.0 when the whole query appears in the name or description as a phrase
    (starting and ending on word boundaries). Otherwise the share of the
    query's words of 3+ letters, stopwords excluded, that equal a word of the
    name/description or begin one when the query word has 4+ letters
    ("deploy" matches "deploy" and "deployment"; "log" does not match
    "catalog"). Edit distance plays no part.
    """
    q = query.lower().strip()
    if not q:
        return 0.0
    text = f"{name} {description}".lower()
    if re.search(r"(?<![a-z0-9])" + re.escape(q) + r"(?![a-z0-9])", text):
        return 1.0
    words = [w for w in _WORD.findall(q) if len(w) >= 3 and w not in _STOPWORDS]
    if not words:
        return 0.0
    tokens = set(_WORD.findall(text))
    hits = sum(
        1 for w in words
        if w in tokens or (len(w) >= 4 and any(t.startswith(w) for t in tokens))
    )
    return hits / len(words)


def _name_words(text: str) -> list[str]:
    """*text*'s words as a tool name spells them (``cron_create`` -> cron, create)."""
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]


def _contains_run(words: list[str], run: list[str]) -> bool:
    n = len(run)
    return any(words[i:i + n] == run for i in range(len(words) - n + 1))


def find_me_score(query_lower: str, queries: tuple[str, ...]) -> float | None:
    """How well *query_lower* matches a tool's ``find_me`` queries (lower is
    better), or None when it does not: the query itself 0.25, one inside the
    other 0.5, else at least 60% of a declared query's words (3+ letters,
    stopwords out, a 4+ letter word may begin one) between 1.0 and 1.2."""
    best: float | None = None
    for declared in queries:
        d = declared.lower().strip()
        if not d:
            continue
        if query_lower.strip() == d:
            score = 0.25
        elif len(query_lower.strip()) >= 4 and (d in query_lower or query_lower.strip() in d):
            score = 0.5
        else:
            share = skill_match_score(d, "", query_lower)
            if share < 0.6:
                continue
            score = 1.5 - share / 2
        best = score if best is None else min(best, score)
    return best


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
    find_me = ("find a tool by description", "load a deferred tool")

    def __init__(self) -> None:
        self._registry: ToolRegistry | None = None
        self._skill_registry: Any | None = None
        # tools.deferred_loading.search_mcp — set by mcp.bootstrap at boot
        # (the config and this instance never meet anywhere else). True =
        # MCP tools appear in search results like any other tool.
        self.include_mcp: bool = True

    def set_registry(self, registry: ToolRegistry) -> None:
        """Inject the tool registry (called after construction)."""
        self._registry = registry

    def set_skill_registry(self, skill_registry: Any) -> None:
        """Inject the skill registry (called after construction)."""
        self._skill_registry = skill_registry

    def get_skill_registry(self) -> Any | None:
        """Return the skill registry (for hot-reload after install)."""
        return self._skill_registry

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
        """Name lookup; return the full schema for one tool or skill.

        An exact name, else a near miss ("select cron_create", "cron create",
        "agent explorer"): the one tool whose name, as words, appears whole in
        the query (the longest when names nest, as ``grep`` in ``lcm_grep``).
        Two candidates resolve nothing. A miss lists the 5 closest names."""
        assert self._registry is not None
        tool = self._registry.get(name)
        if tool is not None:
            return self._tool_schema(tool)

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

        # After both exact lookups, so a skill named like a tool stays the skill.
        near = self._near_miss(name)
        if near is not None:
            return self._tool_schema(near, resolved_from=name)

        return ToolResult(
            output=json.dumps({
                "error": f"No tool or skill named '{name}'.",
                "closest": self._closest(name),
                "hint": "Select one of these by its exact name, or use action 'search' with a description.",
            }, indent=2),
            is_error=True,
        )

    @staticmethod
    def _tool_schema(tool: BaseTool, resolved_from: str | None = None) -> ToolResult:
        result = tool.to_api_schema()
        result["type"] = "tool"
        if resolved_from is not None:
            result["resolved_from"] = resolved_from
        return ToolResult(output=json.dumps(result, indent=2))

    def _near_miss(self, query: str) -> BaseTool | None:
        """The one tool a near-miss *query* names (see :meth:`_handle_select`)."""
        assert self._registry is not None
        words = _name_words(query)
        if not words:
            return None
        hits = [(len(run), t) for t in self._registry.list_tools()
                if (run := _name_words(t.name)) and _contains_run(words, run)]
        if not hits:
            return None
        longest = max(n for n, _ in hits)
        best = [t for n, t in hits if n == longest]
        return best[0] if len(best) == 1 else None

    def _closest(self, query: str, limit: int = 5) -> list[str]:
        """The *limit* names nearest *query*: what a miss lists instead of every
        tool and skill (all of them came to ~1,700 tokens)."""
        assert self._registry is not None
        q = query.lower()
        scored = [(self._score_tool(t, q), t.name) for t in self._registry.list_tools()]
        if self._skill_registry is not None:
            scored += [(self._score_skill(s, q), f"[skill] {s.name}") for s in self._skill_registry.list_skills()
                       if skill_match_score(query, s.name, s.description) >= SKILL_MIN_MATCH]
        return [n for _, n in sorted(scored)[:limit]]

    def _handle_search(self, query: str) -> ToolResult:
        """Fuzzy search across tool names, descriptions, AND skill names/descriptions."""
        assert self._registry is not None
        tools = self._registry.list_tools()
        # tools.deferred_loading.search_mcp (reader wired by mcp.bootstrap,
        # which is the only place the tools config and this instance meet):
        # false hides MCP tools from search results — select-by-exact-name
        # still works, because an operator naming a tool is explicit in a
        # way a fuzzy match is not.
        if not self.include_mcp:
            tools = [t for t in tools if not t.name.startswith("mcp__")]

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

        # Score skills — only those that actually match the query (SKILL_MIN_MATCH);
        # edit distance alone never lists one.
        if self._skill_registry is not None:
            for skill in self._skill_registry.list_skills():
                if skill_match_score(query, skill.name, skill.description) < SKILL_MIN_MATCH:
                    continue
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

        # One of the plain-language requests the tool declares it answers.
        declared = find_me_score(query_lower, tuple(getattr(tool, "find_me", ()) or ()))
        if declared is not None and declared < 1.0:
            return declared

        # Substring in name
        if query_lower in name_lower:
            return 1.0

        if declared is not None:
            return declared

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
