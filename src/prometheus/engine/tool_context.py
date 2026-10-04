"""Is this code running inside a tool call? One flag, set by the agent loop.

WHY THIS EXISTS (computer-use v1.1, the door — W3)
---------------------------------------------------
Only a PERSON may start a desktop task. Credentials answer that question for
every route a request can arrive by, but not for code already inside the
process: a tool a model called can reach any module-level object, and the
task runner is one. So the agent loop marks the span of every
``tool.execute`` with this flag, and the door refuses to start while it is
set (``computer.door.InToolContext``).

It is a ``ContextVar``, so it follows the call and nothing else: tasks a tool
spawns inherit it (``asyncio`` copies the context at task creation), while a
Telegram update or a REST request handled concurrently — a different context
— does not see it.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Iterator

#: The name of the tool whose ``execute`` is running, or None.
TOOL_EXECUTION: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "prometheus_tool_execution", default=None)


def in_tool_execution() -> bool:
    """True inside any tool call the agent loop is running."""
    return TOOL_EXECUTION.get() is not None


def current_tool() -> str | None:
    return TOOL_EXECUTION.get()


@contextlib.contextmanager
def tool_execution(tool_name: str) -> Iterator[None]:
    """Mark the enclosed span as a tool call. Always unset on the way out."""
    token = TOOL_EXECUTION.set(tool_name or "?")
    try:
        yield
    finally:
        TOOL_EXECUTION.reset(token)
