# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/coordinator/agent_definitions.py
# License: MIT
# Modified: renamed imports (openharness → prometheus), stripped to essentials
#           for Prometheus's local-model multi-agent use case

"""Agent type definitions for multi-agent coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentDefinition:
    """Configuration for a named agent type."""

    name: str
    description: str
    system_prompt: str = ""
    tools: list[str] = field(default_factory=list)
    model: str = ""
    read_only: bool = False
    max_turns: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)


# Built-in agent definitions for Prometheus
_BUILTIN_AGENTS: dict[str, AgentDefinition] = {
    "general-purpose": AgentDefinition(
        name="general-purpose",
        description="General-purpose agent for multi-step tasks.",
        system_prompt="You are a helpful assistant that can use tools to accomplish tasks.",
        tools=["Bash", "FileRead", "FileWrite", "FileEdit", "Glob", "Grep"],
    ),
    "explorer": AgentDefinition(
        name="explorer",
        description="Fast read-only agent for codebase exploration.",
        system_prompt="You explore codebases. Read files, search, answer questions. Do not modify anything.",
        tools=["FileRead", "Glob", "Grep"],
        read_only=True,
        max_turns=25,
    ),
    "planner": AgentDefinition(
        name="planner",
        description="Architect agent for designing implementation plans.",
        system_prompt="You design implementation plans. Explore the codebase, identify files, plan changes.",
        tools=["FileRead", "Glob", "Grep"],
        read_only=True,
        max_turns=30,
    ),
    "worker": AgentDefinition(
        name="worker",
        description="Implementation-focused agent for writing code.",
        system_prompt="You implement code changes as instructed. Write clean, tested code.",
        tools=["Bash", "FileRead", "FileWrite", "FileEdit", "Glob", "Grep"],
        max_turns=100,
    ),
    "verification": AgentDefinition(
        name="verification",
        description="Verification agent — runs tests and checks correctness.",
        system_prompt="You verify implementations. Run tests, check outputs, report issues.",
        tools=["Bash", "FileRead", "Glob", "Grep"],
        read_only=True,
        max_turns=25,
    ),
}


def get_all_agent_definitions() -> dict[str, AgentDefinition]:
    """Return all built-in agent definitions."""
    return dict(_BUILTIN_AGENTS)


def get_agent_definition(name: str) -> AgentDefinition | None:
    """Return a single agent definition by name."""
    return _BUILTIN_AGENTS.get(name)


def register_agent_definition(defn: AgentDefinition) -> None:
    """Register a custom agent definition."""
    _BUILTIN_AGENTS[defn.name] = defn
