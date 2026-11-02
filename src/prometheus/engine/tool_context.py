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

#: The conversation the agent run in progress is recorded under: set by the agent loop around a run, None on an
#: ephemeral turn and outside any run. It lives here, not in ``agent_loop``, so the approval queue can read it
#: without importing the loop. Two readers: the loop's telemetry rows (descriptive), and the approval queue,
#: which stamps every pending request with the session that raised it so a SCOPED device is shown, and may answer,
#: only the tool calls of its own sessions (``web/route_access.py``). The read-only calls of a round run under
#: ``gather`` in tasks that copy the context, so they see it too.
RUN_SESSION: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "prometheus_run_session", default=None)


#: Gateway trust (0.9.17): who asked for the turn in progress. (teaches, author): ``teaches`` is False when a group
#: member who is not the owner wrote the turn's request, so nothing in it may teach (memory, skills, GEPA, training
#: data); ``author`` names them. Set by the agent loop around a run; (True, None) outside one.
TURN_TRUST: contextvars.ContextVar[tuple[bool, str | None]] = contextvars.ContextVar(
    "prometheus_turn_trust", default=(True, None))


#: Headers a chat request handed named MCP servers for this turn only: {server: {Header: value}}. Set by the route
#: around the turn (``turn_mcp_headers``), read by the MCP adapter at each call; None outside such a turn. A
#: ContextVar so it follows the turn and its tool tasks and nothing else: a concurrent turn never sees it. The
#: values are credentials: nothing logs, stores or shows them.
TURN_MCP_HEADERS: contextvars.ContextVar[dict[str, dict[str, str]] | None] = contextvars.ContextVar(
    "prometheus_turn_mcp_headers", default=None)


@contextlib.contextmanager
def turn_mcp_headers(headers: dict[str, dict[str, str]] | None) -> Iterator[None]:
    """Make *headers* the current turn's MCP headers for the span of the block."""
    token = TURN_MCP_HEADERS.set(headers or None)
    try:
        yield
    finally:
        try:
            TURN_MCP_HEADERS.reset(token)
        except ValueError:
            # Closed from another context (a streamed reply's generator finalised
            # elsewhere): clear it there rather than leave it set.
            TURN_MCP_HEADERS.set(None)


def turn_trust_of(messages: list) -> tuple[bool, str | None]:
    """(teaches, author) for a turn: from its request, the latest user message a person wrote (an owner's,
    provenance "user", or a group member's, "member"). Tool results and injected turns are skipped: they are not
    the request."""
    for msg in reversed(messages or []):
        if getattr(msg, "role", "") != "user":
            continue
        if any(getattr(block, "type", "") == "tool_result" for block in getattr(msg, "content", []) or []):
            continue
        provenance = getattr(msg, "provenance", "user")
        if provenance == "member":
            return False, getattr(msg, "author", None)
        if provenance == "user":
            return True, getattr(msg, "author", None)
    return True, None


def turn_teaches() -> bool:
    """May what this turn produces teach (memory, skills, GEPA, training data)?"""
    return TURN_TRUST.get()[0]


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
