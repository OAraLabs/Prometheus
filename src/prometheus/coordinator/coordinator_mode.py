# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/coordinator/coordinator_mode.py
# License: MIT
# Modified: renamed imports (openharness → prometheus), adapted to use Prometheus's
#           AgentLoop for subagent spawning, simplified team management

"""Multi-agent coordination mode — team management and task routing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from prometheus.coordinator.agent_definitions import AgentDefinition, get_agent_definition

logger = logging.getLogger(__name__)


@dataclass
class TeamRecord:
    """A named team of agents working together."""

    name: str
    description: str = ""
    agents: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class TeamRegistry:
    """Manages teams of agents."""

    def __init__(self) -> None:
        self._teams: dict[str, TeamRecord] = {}

    def create_team(
        self,
        name: str,
        description: str = "",
        agents: list[str] | None = None,
    ) -> TeamRecord:
        """Create and register a team."""
        team = TeamRecord(name=name, description=description, agents=agents or [])
        self._teams[name] = team
        return team

    def get_team(self, name: str) -> TeamRecord | None:
        """Look up a team by name."""
        return self._teams.get(name)

    def list_teams(self) -> list[TeamRecord]:
        """Return all registered teams."""
        return list(self._teams.values())

    def add_agent_to_team(self, team_name: str, agent_name: str) -> bool:
        """Add an agent to an existing team. Returns False if team not found."""
        team = self._teams.get(team_name)
        if team is None:
            return False
        if agent_name not in team.agents:
            team.agents.append(agent_name)
        return True

    def remove_agent_from_team(self, team_name: str, agent_name: str) -> bool:
        """Remove an agent from a team. Returns False if team not found."""
        team = self._teams.get(team_name)
        if team is None:
            return False
        if agent_name in team.agents:
            team.agents.remove(agent_name)
        return True


# Module-level singleton
_team_registry: TeamRegistry | None = None


def get_team_registry() -> TeamRegistry:
    """Return the global TeamRegistry singleton."""
    global _team_registry
    if _team_registry is None:
        _team_registry = TeamRegistry()
    return _team_registry


def is_coordinator_mode(agent_count: int = 0) -> bool:
    """Return True if the system has multiple agents active (coordinator mode)."""
    return agent_count > 1


def get_coordinator_system_prompt(team: TeamRecord | None = None) -> str:
    """Build a system prompt for a coordinator agent managing a team."""
    base = (
        "You are a coordinator agent. You break tasks into subtasks and "
        "delegate them to specialized subagents using the Agent tool.\n\n"
        "Guidelines:\n"
        "- Decompose complex tasks into independent subtasks\n"
        "- Choose the right agent type for each subtask\n"
        "- Provide clear, complete prompts to subagents\n"
        "- Synthesize results from subagents into a coherent response\n"
    )
    if team:
        base += f"\nTeam: {team.name} — {team.description}\n"
        base += f"Available agents: {', '.join(team.agents)}\n"
    return base
