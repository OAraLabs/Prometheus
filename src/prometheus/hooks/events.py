# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/hooks/events.py
# License: MIT
# Modified: renamed imports (openharness → prometheus)

"""Hook event names supported by Prometheus."""

from __future__ import annotations

from enum import Enum


class HookEvent(str, Enum):
    """Events that can trigger hooks."""

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    # Run-level events — hooks/1.1 (D17), additive per docs/contracts/hooks.md
    # section 12. Observe-only for pillars; the contract is
    # docs/contracts/blackboard.md section 8. Names exist so a subscription
    # can name them; emission sites land with their own PRs (run_events.py
    # ships the terminal coding-run translation, off by default).
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"
    WORKER_ESCALATED = "worker_escalated"
    WORKER_CONTEXT_PRESSURE = "worker_context_pressure"
    MERGE_RESULT = "merge_result"
    MISSION_AMENDED = "mission_amended"
