"""Naming-convention guard for the default builtin tool registry.

``AgentTool.name`` was ``"Agent"`` — the only capitalized name among the 49
tools ``create_tool_registry`` registers by default, while both the README's
builtin-tools list and the oara.ai page documented it as ``agent``. The
advertised interface and the real schema disagreed, which is the same defect
class as the ``GrepTool.example_call`` fix in PR #134: the reader being misled
is the *model*, on every call.

The failure was asymmetric across adapter tiers, which is why it stayed
invisible:

  * tier ``full``/``light`` (local models — the daemon's DEFAULT path; the live
    ``model.provider: llama_cpp`` config resolves to ``light``) —
    ``ToolRegistry.get`` is a plain case-sensitive dict lookup, so
    ``get("agent")`` misses and the validator's ``_fuzzy_match_tool_name``
    lowercases both sides, matching ``Agent`` at Levenshtein distance 0. The
    call is *repaired* rather than dispatched, and ``capture_pair`` banks it as
    a ``levenshtein_repair`` training pair — teaching the flywheel the model
    erred when the harness was inconsistent.
  * tier ``off`` (Anthropic + any ``ProviderRegistry.is_cloud`` provider — the
    on-demand cloud routes) — ``validate_and_repair`` returns early, so there
    is no fuzzy net at all. A model that emitted ``agent`` got a hard
    ``Unknown tool: agent``.

So the same name defect self-corrected on the tier the daemon defaults to and
hard-failed on the tier it routes to on demand — which is precisely why
neither surfaced.

This test pins the convention for every default-registered tool so a future
tool cannot reintroduce a name the docs and the model disagree about.

MCP tools (``mcp__{server}__{tool}``) are registered dynamically from remote
server manifests, not by ``create_tool_registry``, so they are out of scope
here — their names are not ours to choose.
"""

from __future__ import annotations

import re

import pytest

from prometheus.__main__ import create_tool_registry


pytestmark = pytest.mark.integration


# Lowercase snake_case: leading letter, then letters/digits/underscores.
TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class TestToolNameConvention:
    """Every default-registered tool name is lowercase snake_case."""

    def test_all_registered_names_match_convention(self) -> None:
        registry = create_tool_registry({}, None)
        names = [t.name for t in registry.list_tools()]

        assert names, "create_tool_registry registered no tools"

        offenders = sorted(n for n in names if not TOOL_NAME_RE.match(n))
        assert not offenders, (
            f"Tool names must match {TOOL_NAME_RE.pattern} (lowercase "
            f"snake_case) — the README's builtin-tools list and the model's "
            f"tool schema have to agree. Offenders: {offenders}"
        )

    def test_agent_tool_is_lowercase(self) -> None:
        """Regression pin for the specific tool that broke the convention.

        Asserted against the live registry rather than the class attribute so
        it also covers the registration path the model actually sees.
        """
        registry = create_tool_registry({}, None)

        assert registry.get("agent") is not None, (
            "The subagent-spawning tool must register as 'agent' — the name "
            "both the README and the oara.ai page document."
        )
        assert registry.get("Agent") is None, (
            "'Agent' resurfaced. ToolRegistry.get is case-sensitive, so this "
            "name only resolves via adapter fuzzy-repair on local tiers and "
            "hard-fails on tier 'off' (Anthropic/cloud)."
        )

    def test_lookup_is_exact_not_fuzzy(self) -> None:
        """The registry must not paper over case drift itself.

        If ``get`` were ever made case-insensitive, the convention guard above
        would still pass while the model kept emitting a name the schema never
        advertised. Pin the exact-match contract so the fix stays at the name.
        """
        registry = create_tool_registry({}, None)

        assert registry.get("AGENT") is None
        assert registry.get("Bash") is None, (
            "ToolRegistry.get resolved a miscased name — lookup became "
            "case-insensitive, which hides naming drift instead of failing it."
        )
