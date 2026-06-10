#!/usr/bin/env python3
"""Validate whether deferred tool loading is SAFE for the configured model.

Why this exists: ``tools.deferred_loading.enabled`` shrinks the prompt to the
``always_loaded`` core tools (~8) and makes the model reach the other ~40 via
``tool_search``. That saves ~7.5k tokens/call — but only works if the model
actually DISCOVERS tools it can't see. ``scripts/smoke_test_tool_calling.py``
can't test this (it builds a 7-tool registry where everything is always-loaded),
so this script builds the daemon's FULL registry with deferred ON and asks the
live model to do tasks that each need a NON-core (deferred) tool.

Pass bar: reaches a deferred tool in >= 2/3 tasks. Exit 0 = safe to flip, 1 = not.

Finding 2026-06-10 (gemma4-26b, tier light): 0/3 — the model gives up
("I have no web search tool"), falls back to bash for the wrong job, or thrashes
to the turn limit, even WITH an explicit "use tool_search" system-prompt nudge.
Deferred loading is therefore OFF for this model. Re-run after a model change.

Usage:
    uv run python scripts/validate_deferred_loading.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.__main__ import (
    load_config, create_tool_registry, create_security_gate, create_adapter,
)
from prometheus.providers.registry import ProviderRegistry
from prometheus.context.dynamic_tools import DynamicToolLoader
from prometheus.engine import AgentLoop
from prometheus.telemetry.tracker import ToolCallTelemetry

# Each task needs a DEFERRED (non always-loaded) tool, phrased as a real task —
# NOT "search for a tool X" — so we measure spontaneous discovery.
TASKS = [
    ("web_search", "Search the web for what the latest stable version of Python is."),
    ("cron", "List the cron jobs currently scheduled on this system using your tools."),
    ("memory/lcm", "Use your tools to recall anything stored about my projects."),
]


async def main() -> int:
    cfg = load_config()
    model_cfg = cfg["model"]
    sec_cfg = cfg.get("security", {})
    provider = ProviderRegistry.create(model_cfg)
    gate = create_security_gate(sec_cfg)
    registry = create_tool_registry(sec_cfg, security_gate=gate)
    adapter = create_adapter(model_cfg, cfg.get("adapter"))

    deferred_cfg = dict(cfg.get("tools", {}).get("deferred_loading") or {})
    deferred_cfg["enabled"] = True  # force ON for the test regardless of on-disk flag
    loader = DynamicToolLoader(registry, deferred_cfg)
    in_prompt = {s["name"] for s in loader.active_schemas()}
    print(f"Registry: {len(registry.list_tools())} tools | deferred ON → "
          f"{len(in_prompt)} in prompt | tier: {adapter.tier}\n")

    tel = ToolCallTelemetry()
    reached_n = 0
    for label, task in TASKS:
        loop = AgentLoop(
            provider=provider, model=model_cfg.get("model", ""),
            tool_registry=registry, adapter=adapter, permission_checker=gate,
            telemetry=tel, tool_loader=loader, max_turns=12,
        )
        print(f"━━━ {label}: {task}")
        try:
            r = await loop.run_async(
                system_prompt="You are Prometheus. Use your tools to accomplish the task.",
                user_message=task, session_id="cli",
            )
            used = [t["tool_name"] for t in loop._tool_trace]
            non_core = [t for t in used if t not in in_prompt and t != "tool_search"]
            reached = bool(non_core)
            reached_n += int(reached)
            print(f"   tools: {used} | tool_search={'tool_search' in used} | "
                  f"reached deferred={reached} {non_core}")
            print(f"   answer: {(r.text or '')[:140].strip()}\n")
        except Exception as e:
            print(f"   ERROR: {type(e).__name__}: {e}\n")

    print("=" * 60)
    print(f"Reached a deferred tool in {reached_n}/{len(TASKS)} tasks.")
    print("→ SAFE to enable deferred loading." if reached_n >= 2
          else "→ NOT safe: keep tools.deferred_loading.enabled = false.")
    return 0 if reached_n >= 2 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
