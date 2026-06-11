#!/usr/bin/env python3
"""Live acceptance smoke for the repair-pair flywheel (Workstream A).

Induces a known-repairable bad call against the REAL model through the REAL
pipeline and asserts a pair lands in a throwaway training.db with non-empty
context. Mirrors the gym/smoke pipeline construction.

    uv run python scripts/smoke_pair_capture.py
Exit 0 = pair captured; 1 = not.
"""

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from prometheus.__main__ import create_adapter, create_security_gate, load_config
from prometheus.engine import AgentLoop
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

    loop = AgentLoop(
        provider=provider,
        model=config["model"].get("model", "unknown"),
        tool_registry=registry,
        adapter=adapter,
        permission_checker=create_security_gate(config.get("security", {})),
        telemetry=ToolCallTelemetry(db_path=tmp / "tel.db"),
    )

    # The misnamed-tool prompt: 'task_lists' is Levenshtein distance 1 from
    # task_list → adapter fuzzy repair fires (reachable since 511ed2f) → a
    # levenshtein_repair pair must land. If the model instead emits the
    # CORRECT name spontaneously, no repair happens — retry a nudgier prompt.
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
        rows = store.rows_since()
        if rows:
            break

    rows = store.rows_since()
    print(f"\npairs captured: {len(rows)}")
    for r in rows:
        ctx = json.loads(r["context"] or "{}")
        print(f"  [{r['pair_source']}] {json.loads(r['rejected'])['name']!r} → "
              f"{json.loads(r['chosen'])['name']!r} ctx.keys={sorted(ctx)}")
        if not ctx:
            print("  ❌ EMPTY CONTEXT")
            return 1
    if not rows:
        print("❌ no pair captured (did the model call the misnamed tool?)")
        return 1
    print("✅ live pair capture acceptance PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
