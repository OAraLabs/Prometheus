"""Hook loader — builds a HookRegistry from YAML config.

Donor pattern: HKUDS/OpenHarness src/openharness/hooks/loader.py (MIT).
Adapted for Prometheus: reads from prometheus.yaml hooks section.
"""

from __future__ import annotations

import logging
from typing import Any

from prometheus.hooks.events import HookEvent
from prometheus.hooks.registry import HookRegistry
from prometheus.hooks.schemas import (
    AgentHookDefinition,
    CommandHookDefinition,
    HookDefinition,
    HttpHookDefinition,
    PromptHookDefinition,
)

logger = logging.getLogger(__name__)

_TYPE_MAP: dict[str, type[HookDefinition]] = {
    "command": CommandHookDefinition,
    "http": HttpHookDefinition,
    "prompt": PromptHookDefinition,
    "agent": AgentHookDefinition,
}


def load_hook_registry(hooks_config: dict[str, list[dict[str, Any]]]) -> HookRegistry:
    """Build a HookRegistry from a hooks config dict.

    Expected format (from prometheus.yaml ``hooks:`` section)::

        hooks:
          pre_tool_use:
            - type: command
              command: "echo checking $ARGUMENTS"
              block_on_failure: false
          post_tool_use:
            - type: http
              url: "http://localhost:9090/hook"

    Args:
        hooks_config: mapping of event name → list of hook definition dicts.

    Returns:
        Populated HookRegistry.
    """
    registry = HookRegistry()
    for event_name, hook_defs in hooks_config.items():
        try:
            event = HookEvent(event_name)
        except ValueError:
            logger.warning("Unknown hook event: %s — skipping", event_name)
            continue

        for raw in hook_defs:
            hook_type = raw.get("type", "")
            cls = _TYPE_MAP.get(hook_type)
            if cls is None:
                logger.warning("Unknown hook type: %s — skipping", hook_type)
                continue
            try:
                hook = cls(**raw)
                registry.add(event, hook)
            except Exception as exc:
                logger.warning("Failed to load hook %s: %s", raw, exc)

    return registry
