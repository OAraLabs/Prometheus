#!/usr/bin/env python3
"""Live acceptance smoke for the repair-pair flywheel (Workstream A).

Induces a known-repairable bad call ('task_lists', Levenshtein distance 1
from the registered 'task_list') through the REAL configured adapter
validate/repair + loop dispatch + pair-capture path and asserts a
levenshtein_repair pair lands in training.db with non-empty context.

Two induction modes:

1. Model-driven (default first attempt): prompt the live model to emit the
   misnamed call. The post-invariants model usually REFUSES the bait
   (observed 2026-06-12: 3 runs x 2 prompts, 6 coherent refusals naming the
   correct tool) — good name discipline, useless for induction.
2. Injected (``--inject``, also the automatic fallback when the model
   declines): fabricate the misnamed tool call and feed it straight to
   ``_execute_tool_call`` — the same entry the live loop uses — with the
   same real adapter/registry/telemetry, mirroring
   tests/test_training_pairs.py::test_levenshtein_repair_produces_pair.
   The repair, dispatch, and capture are all genuine; only the emission is
   synthetic. Injected pairs carry session_id "smoke:inject" in their
   context so they are filterable in training data.

    uv run python scripts/smoke_pair_capture.py [--inject] [--use-default-db]

Exit 0 = pair captured (the report says by which mode); 1 = no pair even
via injection, i.e. a real capture-plumbing failure.
"""

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from prometheus.__main__ import create_adapter, create_security_gate, load_config
from prometheus.engine import AgentLoop
from prometheus.engine.agent_loop import LoopContext, _execute_tool_call
from prometheus.learning import pair_capture
from prometheus.providers.registry import ProviderRegistry
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import ToolRegistry
from prometheus.tools.builtin import BashTool
from prometheus.tools.builtin.task_list import TaskListTool


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--use-default-db", action="store_true",
        help="write to the real ~/.prometheus/data/training.db so /pairs "
             "and /api/pairs reflect the captured pair (acceptance mode)",
    )
    ap.add_argument(
        "--inject", action="store_true",
        help="skip the model-driven attempt and inject the misnamed call "
             "directly (model-independent induction)",
    )
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="pair-smoke-"))
    if args.use_default_db:
        pair_capture.configure({})
    else:
        pair_capture.configure({"db_path": str(tmp / "training.db")})
    store = pair_capture.get_store()

    config = load_config()
    provider = ProviderRegistry.create(config["model"])
    registry = ToolRegistry()
    registry.register(BashTool(workspace=str(tmp)))
    registry.register(TaskListTool())
    adapter = create_adapter(config["model"], config.get("adapter"))
    if hasattr(provider, "set_grammar"):
        grammar = adapter.generate_grammar(registry)
        if grammar:
            provider.set_grammar(grammar)
    gate = create_security_gate(config.get("security", {}))
    model_name = config["model"].get("model", "unknown")
    telemetry = ToolCallTelemetry(db_path=tmp / "tel.db")

    mode = None

    if not args.inject:
        loop = AgentLoop(
            provider=provider,
            model=model_name,
            tool_registry=registry,
            adapter=adapter,
            permission_checker=gate,
            telemetry=telemetry,
        )
        # The misnamed-tool prompt: 'task_lists' is Levenshtein distance 1
        # from task_list → adapter fuzzy repair fires (reachable since
        # 511ed2f) → a levenshtein_repair pair must land. The model may
        # instead refuse / emit the correct name — then no repair happens
        # and we fall through to injection below.
        prompts = [
            "Call the tool named exactly 'task_lists' to show background tasks.",
            "Use your 'task_lists' tool (that exact name) to list tasks, then say DONE.",
        ]
        for prompt in prompts:
            print(f"▶ {prompt}")
            result = await loop.run_async(
                system_prompt="You are a tool-using assistant. Use tools when asked.",
                user_message=prompt,
                session_id="system",
            )
            print(f"  model: {getattr(result, 'text', '')[:120]!r}")
            if store.rows_since():
                mode = "model-driven"
                break
        if mode is None:
            print("  model declined the misnamed call — falling back to injection")

    if mode is None:
        # Model-independent induction: same adapter, registry, telemetry,
        # and dispatch entry as the live loop; only the emission is forged.
        ctx = LoopContext(
            provider=provider,
            model=model_name,
            system_prompt="",
            max_tokens=256,
            tool_registry=registry,
            adapter=adapter,
            permission_checker=gate,
            telemetry=telemetry,
            session_id="smoke:inject",
        )
        print("▶ injecting ToolUseBlock(name='task_lists', input={})")
        block = await _execute_tool_call(ctx, "task_lists", "smoke-inject-1", {})
        print(f"  result: is_error={getattr(block, 'is_error', None)}")
        if store.rows_since():
            mode = "injected"

    rows = store.rows_since()
    print(f"\npairs captured: {len(rows)}")
    for r in rows:
        ctx_json = json.loads(r["context"] or "{}")
        print(f"  [{r['pair_source']}] {json.loads(r['rejected'])['name']!r} → "
              f"{json.loads(r['chosen'])['name']!r} ctx.keys={sorted(ctx_json)}")
        if not ctx_json:
            print("  ❌ EMPTY CONTEXT")
            return 1
    if not rows:
        print("❌ no pair captured even via injection — capture plumbing is broken")
        return 1
    print(f"✅ live pair capture acceptance PASSED ({mode})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
