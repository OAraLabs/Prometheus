"""``python -m prometheus --worker``: run one brief as ONE turn, then exit.

How the daemon's spawned workers run (``sessions_spawn``, ``task_create`` with
``type=local_agent``). The interactive REPL they used to be read stdin one
line per turn and never exited: in the 2026-10-09 incident (#704) a 59-line
brief became 59 turns on the same over-full history, and the worker then sat
at ``>>>`` for an hour.

A worker reads its whole brief to EOF (the manager closes stdin after writing
it), runs it as a single user turn, prints what a ``--once`` run prints, and
ends with one result line (:mod:`prometheus.tasks.worker_result`). Exit 0 when
the turn finished, 1 when it did not — a stop the LOOP chose (the #356 context
pre-flight, the iteration cap) counts as not finished, with that reason.

Source: novel code for Prometheus, 2026-10-10.
"""

from __future__ import annotations

import sys
from typing import Any

from prometheus.engine.agent_loop import LoopContext, MaxTurnsExhausted, run_loop
from prometheus.engine.messages import ConversationMessage
from prometheus.engine.stream_events import (
    AssistantTextDelta,
    AssistantTurnComplete,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from prometheus.tasks.worker_result import format_result


def _window(context: LoopContext) -> int | None:
    compactor = context.compactor
    if compactor is None or not hasattr(compactor, "limit_for"):
        return None
    try:
        return int(compactor.limit_for(context.model, backend=context.backend))
    except Exception:  # noqa: BLE001 — an unsized window is reported as unknown
        return None


async def run_worker_turn(
    context: LoopContext, brief: str, *, route: str | None = None,
) -> dict[str, Any]:
    """Run *brief* as one turn; return the result (see worker_result)."""
    from prometheus.api.turn_errors import classify_turn_error

    messages = [ConversationMessage.from_user_text(brief)]
    status, reason = "ok", None
    response_text = ""
    try:
        async for event, _usage in run_loop(context, messages, surface="cli"):
            if isinstance(event, AssistantTextDelta):
                print(event.text, end="", flush=True)
                response_text += event.text
            elif isinstance(event, ToolExecutionStarted):
                print(f"\n[tool] {event.tool_name}...", end="", flush=True)
            elif isinstance(event, ToolExecutionCompleted):
                print(f" {'error' if event.is_error else 'ok'}", flush=True)
            elif isinstance(event, AssistantTurnComplete):
                if event.message.text and not response_text:
                    print(event.message.text, end="", flush=True)
                    response_text = event.message.text
                stop = getattr(event, "stop_reason", None)
                if stop:
                    status, reason = "failed", str(stop)
    except MaxTurnsExhausted as cap:
        # #774: a worker has no next turn to continue in, so "keep it" means
        # saying honestly why it stopped — not the generic error kind.
        print(f"\n[stopped] reached the limit of {cap.max_turns} model rounds "
              f"(model.max_turns)", flush=True)
        status, reason = "failed", "max_turns_exhausted"
    except Exception as exc:  # noqa: BLE001 — the turn failed; say how, then exit
        kind = str(classify_turn_error(exc).get("kind") or "error")
        print(f"\n[error] {kind}: {exc}", flush=True)
        status, reason = "failed", kind
    print()
    provider = getattr(context.provider, "provider_name", None) or type(context.provider).__name__
    return {
        "status": status, "reason": reason, "route": route, "model": context.model,
        "provider": str(provider), "window": _window(context),
    }


async def run_worker(context: LoopContext, *, route: str | None) -> int:
    """Read the brief from stdin to EOF, run it, print the result line."""
    from prometheus.cli.worker_approval import set_worker_token, split_token_line

    # The manager's first line is this worker's approval token, never part of
    # the brief and never in the environment (tasks/worker_auth.py).
    token, brief = split_token_line(sys.stdin.read())
    if token:
        set_worker_token(token)
    if not brief.strip():
        result: dict[str, Any] = {
            "status": "failed", "reason": "empty_brief", "route": route,
            "model": context.model, "provider": None, "window": None,
        }
    else:
        result = await run_worker_turn(context, brief, route=route)
    print(format_result(result), flush=True)
    return 0 if result["status"] == "ok" else 1
