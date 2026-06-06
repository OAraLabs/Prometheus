"""Kanban board (projects + stories) — daemon-backed store for Beacon Desktop's
Board view.

A small SQLite store plus the REST routes in ``prometheus.web.server`` expose a
projects/stories board that can be shared across clients pointed at this daemon.
"""

from __future__ import annotations

from prometheus.kanban.store import (
    STORY_PRIORITIES,
    STORY_STATUSES,
    KanbanStore,
    get_kanban_store,
)

__all__ = [
    "KanbanStore",
    "get_kanban_store",
    "STORY_STATUSES",
    "STORY_PRIORITIES",
]
