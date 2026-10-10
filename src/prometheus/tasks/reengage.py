"""Deliver a task's re-engagement turn on the conversation's OWN surface.

WHY (2026-10-09, #704)
----------------------
Re-engagement was wired to Telegram's ``inject_turn`` and nothing else. A
Beacon conversation's re-engagement ran through Telegram (which "cannot
deliver" a non-Telegram reply, by its own docstring) and, with no Telegram
adapter configured, was skipped outright. Worse, the task recorded the web
path's shared ``"web"`` namespace as its creator, so the turn ran in a
session nobody was reading.

:class:`ReengageRouter` is the one ``inject_turn`` the completion handler is
given. A ``telegram:*`` session goes to Telegram; anything else goes to the
web bridge, which records the turn and shows it live in Beacon; Telegram's
path is the last resort, as before (the turn runs and persists). The daemon
sets the Telegram half at boot and the web launcher sets the bridge half when
it builds the bridge, which is later. Every turn it delivers is the caller's
— ``task_supervisor``, untrusted — never the user's.

Source: novel code for Prometheus, 2026-10-10.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

log = logging.getLogger(__name__)

Inject = Callable[..., Awaitable[str]]


def conversation_of(metadata: Mapping[str, Any] | None) -> str | None:
    """The conversation a tool call belongs to: the TURN's own session.

    ``session_id`` in tool metadata is the routing namespace, which on the
    web path is the literal ``"web"`` shared by every Beacon conversation;
    ``effective_session_id`` is the conversation itself.
    """
    meta = metadata or {}
    return meta.get("effective_session_id") or meta.get("session_id") or None


class ReengageRouter:
    """``inject_turn(session_id, content, *, provenance, is_trusted)``, routed."""

    def __init__(self, *, telegram: Inject | None = None) -> None:
        self.telegram = telegram
        self.bridge: Inject | None = None

    def set_bridge(self, inject: Inject | None) -> None:
        self.bridge = inject

    async def __call__(
        self, session_id: str, content: str, *,
        provenance: str = "task_supervisor", is_trusted: bool = False,
    ) -> str:
        if session_id.startswith("telegram:") and self.telegram is not None:
            target, surface = self.telegram, "telegram"
        elif self.bridge is not None:
            target, surface = self.bridge, "web"
        elif self.telegram is not None:
            target, surface = self.telegram, "telegram"
        else:
            raise RuntimeError(
                f"no surface can deliver a re-engagement to {session_id!r} "
                "(no web bridge, no Telegram)")
        log.info("re-engaging %s through %s", session_id, surface)
        return await target(session_id, content, provenance=provenance, is_trusted=is_trusted)


_ROUTER: ReengageRouter | None = None


def get_reengage_router() -> ReengageRouter:
    """The process-wide router (daemon and web launcher share it)."""
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = ReengageRouter()
    return _ROUTER
