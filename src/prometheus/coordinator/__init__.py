# Source: OpenHarness (HKUDS/OpenHarness) coordinator + novel Prometheus code
# Original: src/openharness/coordinator/__init__.py
# License: MIT
# Modified: renamed imports (openharness → prometheus), added SubagentSpawner,
#           HealthMonitor, adapted AgentDefinition/TeamRecord for Prometheus

"""Coordinator package — multi-agent coordination for Sprint 8."""

from __future__ import annotations

from prometheus.coordinator.agent_definitions import (
    AgentDefinition,
    get_all_agent_definitions,
    get_agent_definition,
)
from prometheus.coordinator.coordinator_mode import (
    TeamRecord,
    TeamRegistry,
    get_team_registry,
    is_coordinator_mode,
)
from prometheus.coordinator.subagent import SubagentSpawner, SubagentResult
from prometheus.coordinator.health import HealthMonitor, HealthStatus
from prometheus.coordinator.divergence import (
    DivergenceDetector,
    DivergenceResult,
    GoalTracker,
    Checkpoint,
    CheckpointStore,
)

__all__ = [
    "AgentDefinition",
    "get_all_agent_definitions",
    "get_agent_definition",
    "TeamRecord",
    "TeamRegistry",
    "get_team_registry",
    "is_coordinator_mode",
    "SubagentSpawner",
    "SubagentResult",
    "HealthMonitor",
    "HealthStatus",
    "DivergenceDetector",
    "DivergenceResult",
    "GoalTracker",
    "Checkpoint",
    "CheckpointStore",
]
