"""In-memory hook registry (replaces OpenHarness hooks/loader.py for Sprint 2).

loader.py / hot_reload.py are deferred to Sprint 5.
"""

from __future__ import annotations

from prometheus.hooks.events import HookEvent
from prometheus.hooks.schemas import HookDefinition


class HookRegistry:
    """Simple in-memory map of events → hook definitions."""

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[HookDefinition]] = {
            event: [] for event in HookEvent
        }

    def add(self, event: HookEvent, hook: HookDefinition) -> None:
        """Register a hook for an event."""
        self._hooks[event].append(hook)

    def get(self, event: HookEvent) -> list[HookDefinition]:
        """Return all hooks registered for an event."""
        return list(self._hooks.get(event, []))

    def clear(self, event: HookEvent | None = None) -> None:
        """Remove all hooks, optionally scoped to a single event."""
        if event is None:
            for key in self._hooks:
                self._hooks[key] = []
        else:
            self._hooks[event] = []
