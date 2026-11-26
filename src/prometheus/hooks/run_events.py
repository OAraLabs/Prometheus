"""Run-level events — hooks/1.1 (D17), the blackboard protocol's event group.

Seven events about RUNS (a whole worker execution), beside hooks/1.0's
events about turns. Additive minor version per docs/contracts/hooks.md
section 12: new event names, observe-only for pillars, no capability
change on any existing event. The contract: docs/contracts/blackboard.md
section 8.

What ships here (off by default, ``blackboard.run_events.enabled``):

- the seven event names on :class:`HookEvent` (additive; the registry's
  per-event lists stay empty unless an operator hook subscribes);
- :class:`RunEventEmitter`, wired to the BackgroundTaskManager's existing
  terminal task signals. A coding run (``session_id`` ``coding:<id>``)
  resolving emits ``run_finished`` (exit 0) or ``run_failed`` (anything
  else) through the SAME HookExecutor operator hooks already use — one
  payload, one env-var contract ($ARGUMENTS / PROMETHEUS_HOOK_PAYLOAD).

What does NOT ship here (named, not faked):

- ``run_started`` at launch: the launch site (POST /api/code) does not
  yet carry mission/item facts; the emitter exposes ``started()`` for
  that wiring, and nothing calls it until the roadmap pieces exist.
- ``worker_escalated``, ``worker_context_pressure``: their facts live in
  the agent loop's turn path; wiring them is the loop-side slice (B8).
- ``merge_result``, ``mission_amended``: the merge tool and the Beacon
  amendment route do not exist yet (one PR each, blackboard.md §12).
- the pillar transport: no pillar seam is merged (wp35-instinct-shadow
  is unreachable from this machine). When it lands, these events ride
  it observe-only; until then an operator hook may subscribe, and with
  no subscriber the emitter builds no payload and fires nothing.

A pillar's ACTIONS (start a run, request a merge) go through the daemon
API under the gate (D18), never through a hook response. Every event
here is observe-only: the executor's result is discarded, exactly as
for post_tool_use.

Source: novel code for Prometheus, 2026-11-26.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

#: The config gate. Absent section, absent key, or a non-true value all
#: mean OFF — same literal-true-only shape as acp.gate.acp_enabled.
def run_events_enabled(config: Any) -> bool:
    section = (config or {}).get("blackboard") if isinstance(config, dict) else None
    if not isinstance(section, dict):
        return False
    events = section.get("run_events")
    if not isinstance(events, dict):
        return False
    return events.get("enabled", False) is True


#: The coding-run session prefix the task manager already stamps.
_CODING_PREFIX = "coding:"


class RunEventEmitter:
    """Translates terminal coding-run task signals into run-level events.

    Subscribes to ``task_completed`` / ``task_failed`` on the SignalBus
    (the same signals the completion handler and the live-stream tailer
    read). Only coding runs (``session_id`` starting ``coding:``) are
    translated today; every other task is ignored. With no hook
    subscribed to the event, ``execute`` is a registry lookup that finds
    an empty list — no payload is built beyond the facts already on the
    signal, and nothing fires.

    Never raises into the bus: a failed emission is logged and dropped.
    The run's own outcome does not depend on its event.
    """

    def __init__(self, hook_executor: Any, *, bus: Any = None) -> None:
        self._executor = hook_executor
        self._bus = bus

    def subscribe(self, bus: Any) -> None:
        """Attach to the bus's terminal task signals (idempotent per bus)."""
        if bus is None or bus is self._bus:
            return
        self._bus = bus
        bus.subscribe("task_completed", self._on_terminal)
        bus.subscribe("task_failed", self._on_terminal)

    async def _on_terminal(self, signal: Any) -> None:
        try:
            payload = getattr(signal, "payload", None) or {}
            session_id = str(payload.get("session_id") or "")
            if not session_id.startswith(_CODING_PREFIX):
                return
            if self._executor is None:
                return
            from prometheus.hooks.events import HookEvent

            status = str(payload.get("status") or "")
            ok = status == "completed" and payload.get("return_code") == 0
            event = HookEvent.RUN_FINISHED if ok else HookEvent.RUN_FAILED
            facts = {
                "run_id": session_id[len(_CODING_PREFIX):],
                "session_id": session_id,
                "status": status,
                "return_code": payload.get("return_code"),
                "error": payload.get("error"),
                # The branch-local acceptance record is the run's own
                # report (acceptance_exit etc.); the tail carries it when
                # the report parsed. Named, not re-derived.
                "worker_result": payload.get("worker_result"),
                "output_tail_ref": payload.get("artifact_path"),
            }
            await self._executor.execute(event, {
                "event": event.value,
                "facts": facts,
                "content": {},
            })
        except Exception:  # noqa: BLE001 - an event never decides a run's outcome
            log.warning("run-event emission failed", exc_info=True)

    async def started(self, *, run_id: str, facts: dict[str, Any]) -> None:
        """``run_started`` — for the launch-site wiring (not called yet).

        Kept explicit so the launch PR has one place to call; nothing in
        this PR invokes it, and the docstring says so rather than
        pretending the event fires.
        """
        if self._executor is None:
            return
        from prometheus.hooks.events import HookEvent

        try:
            await self._executor.execute(HookEvent.RUN_STARTED, {
                "event": HookEvent.RUN_STARTED.value,
                "facts": {"run_id": run_id, **facts},
                "content": {},
            })
        except Exception:  # noqa: BLE001
            log.warning("run_started emission failed", exc_info=True)