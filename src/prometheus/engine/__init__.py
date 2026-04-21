# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/engine/__init__.py
# License: MIT
# Modified: renamed imports (openharness → prometheus), QueryEngine → AgentLoop

"""Engine package exports.

AgentLoop and RunResult are lazy-loaded to break a circular import:
  providers.base → engine.messages → engine.__init__ → engine.agent_loop → providers.base
"""

from __future__ import annotations

from prometheus.engine.session import ChatSession, SessionManager
from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.engine.stream_events import (
    AssistantTextDelta,
    AssistantTurnComplete,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from prometheus.engine.usage import UsageSnapshot

__all__ = [
    "AgentLoop",
    "ChatSession",
    "RunResult",
    "SessionManager",
    "AssistantTextDelta",
    "AssistantTurnComplete",
    "ConversationMessage",
    "TextBlock",
    "ToolExecutionCompleted",
    "ToolExecutionStarted",
    "ToolResultBlock",
    "ToolUseBlock",
    "UsageSnapshot",
]


def __getattr__(name: str):
    if name in ("AgentLoop", "RunResult"):
        from prometheus.engine.agent_loop import AgentLoop, RunResult
        globals()["AgentLoop"] = AgentLoop
        globals()["RunResult"] = RunResult
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
