"""The daemon must pass max_tool_iterations_cloud into LoopContext.

Live bug (2026-07-31): a grok-4.5 turn stopped with "Tool iteration limit
reached (26/25)" while ``max_tool_iterations_cloud: 50`` sat correctly in the
config. The config was right; ``daemon.py`` simply never forwarded it, so
``LoopContext.max_tool_iterations_cloud`` stayed None and
``_effective_max_tool_iterations`` fell back to the LOCAL cap for every
provider — cloud runs were silently capped at the small-window limit.

``__main__.py`` (the CLI path) forwarded it correctly, which is why this
never showed up there. Both entry points are asserted here so they can't
drift apart again.
"""

from __future__ import annotations

import ast
from pathlib import Path

from prometheus.engine.agent_loop import LoopContext, _effective_max_tool_iterations

SRC = Path(__file__).resolve().parents[1] / "src" / "prometheus"


def _kwargs_for(module_path: Path, callee: str) -> set[str]:
    """Keyword names passed to every ``callee(...)`` call in a module.

    Both AgentLoop(...) and LoopContext(...) end up owning a LoopContext, so
    the cap has to be forwarded at whichever one the entry point constructs.
    """
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if name == callee:
            names.update(kw.arg for kw in node.keywords if kw.arg)
    return names


def test_daemon_agentloop_forwards_the_cloud_cap():
    """The daemon builds its loop via AgentLoop(...) — the construction that
    was dropping the cap."""
    kwargs = _kwargs_for(SRC / "daemon.py", "AgentLoop")
    assert "max_tool_iterations" in kwargs, "sanity: the local cap is forwarded"
    assert "max_tool_iterations_cloud" in kwargs, (
        "daemon.py must forward max_tool_iterations_cloud — without it the "
        "LoopContext default (None) silently caps cloud runs at the local limit"
    )


def test_cli_entrypoint_also_forwards_it():
    """__main__.py builds a LoopContext directly; it already forwarded the cap,
    and must keep doing so — the two entry points must not drift apart."""
    kwargs = _kwargs_for(SRC / "__main__.py", "LoopContext")
    assert "max_tool_iterations_cloud" in kwargs


def test_agentloop_accepts_the_kwarg():
    """Guard the forward itself: passing an unknown kwarg would crash the
    daemon at startup rather than fail a test."""
    import inspect

    from prometheus.engine.agent_loop import AgentLoop

    assert "max_tool_iterations_cloud" in inspect.signature(AgentLoop.__init__).parameters


class _Adapter:
    def __init__(self, tier: str) -> None:
        self.tier = tier


def test_cloud_tier_uses_the_cloud_cap():
    ctx = LoopContext(
        provider=object(), model="grok-4.5", system_prompt="", max_tokens=8,
        max_tool_iterations=25, max_tool_iterations_cloud=50,
        adapter=_Adapter("off"),
    )
    assert _effective_max_tool_iterations(ctx) == 50


def test_local_tier_keeps_the_local_cap():
    ctx = LoopContext(
        provider=object(), model="qwen", system_prompt="", max_tokens=8,
        max_tool_iterations=25, max_tool_iterations_cloud=50,
        adapter=_Adapter("strict"),
    )
    assert _effective_max_tool_iterations(ctx) == 25


def test_unset_cloud_cap_shares_the_local_limit():
    """Backward-compatible: None means 'no separate cloud limit'."""
    ctx = LoopContext(
        provider=object(), model="grok-4.5", system_prompt="", max_tokens=8,
        max_tool_iterations=25, max_tool_iterations_cloud=None,
        adapter=_Adapter("off"),
    )
    assert _effective_max_tool_iterations(ctx) == 25


def test_the_regression_shape_exactly():
    """The live failure: cloud provider + configured cloud cap, but the field
    never populated → capped at 25 instead of 50."""
    broken = LoopContext(
        provider=object(), model="grok-4.5", system_prompt="", max_tokens=8,
        max_tool_iterations=25,  # cloud cap NOT passed — the daemon.py bug
        adapter=_Adapter("off"),
    )
    assert _effective_max_tool_iterations(broken) == 25, "reproduces the bug"

    fixed = LoopContext(
        provider=object(), model="grok-4.5", system_prompt="", max_tokens=8,
        max_tool_iterations=25, max_tool_iterations_cloud=50,
        adapter=_Adapter("off"),
    )
    assert _effective_max_tool_iterations(fixed) == 50, "and the fix"
