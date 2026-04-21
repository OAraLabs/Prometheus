"""Agent profiles — configurable presets that control which bootstrap files,
tools, and subsystems load for a given session.

Builtin profiles are hardcoded. Custom profiles are loaded from YAML files
in ``~/.prometheus/profiles/``. Custom profiles with the same name as a
builtin override it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from prometheus.config.paths import get_config_dir

log = logging.getLogger(__name__)

_PROFILES_DIR = "profiles"


@dataclass
class AgentProfile:
    """A named configuration preset controlling context loading."""

    name: str
    description: str = ""
    bootstrap_files: list[str] = field(default_factory=lambda: ["SOUL.md", "AGENTS.md", "ANATOMY.md"])
    tools: list[str] | None = None          # None = all tools
    exclude_tools: list[str] = field(default_factory=list)
    subsystems: dict[str, bool] = field(default_factory=dict)
    max_tool_schemas: int | None = None


# ------------------------------------------------------------------
# Builtin profiles
# ------------------------------------------------------------------

_BUILTINS: dict[str, AgentProfile] = {
    "full": AgentProfile(
        name="full",
        description="All capabilities enabled. Default for Telegram assistant mode.",
        bootstrap_files=["SOUL.md", "AGENTS.md", "ANATOMY.md"],
        tools=None,
        exclude_tools=[],
        subsystems={"sentinel": True, "wiki": True, "cron": True, "learning": True},
    ),
    "coder": AgentProfile(
        name="coder",
        description="Focused coding. Lean context, fast tool calls.",
        bootstrap_files=["SOUL.md"],
        tools=[
            "bash", "file_read", "file_write", "file_edit", "grep", "glob",
            "todo_write", "task_create", "agent", "lsp",
        ],
        exclude_tools=[],
        subsystems={"sentinel": False, "wiki": False, "cron": False, "learning": False},
    ),
    "research": AgentProfile(
        name="research",
        description="Knowledge retrieval and synthesis. No file mutations.",
        bootstrap_files=["SOUL.md"],
        tools=[
            "wiki_query", "wiki_compile", "lcm_grep", "lcm_expand",
            "lcm_describe", "lcm_expand_query", "file_read", "grep", "glob",
        ],
        exclude_tools=[],
        subsystems={"sentinel": False, "wiki": True, "cron": False, "learning": False},
    ),
    "assistant": AgentProfile(
        name="assistant",
        description="Conversational assistant. Memory-rich, tool-light.",
        bootstrap_files=["SOUL.md", "AGENTS.md"],
        tools=[
            "wiki_query", "lcm_grep", "file_read", "bash", "cron_list",
            "sentinel_status", "todo_write",
        ],
        exclude_tools=[],
        subsystems={"sentinel": True, "wiki": True, "cron": True, "learning": True},
    ),
    "minimal": AgentProfile(
        name="minimal",
        description="Maximum context for conversation. Almost no tool overhead.",
        bootstrap_files=["SOUL.md"],
        tools=["bash", "file_read"],
        exclude_tools=[],
        subsystems={"sentinel": False, "wiki": False, "cron": False, "learning": False},
    ),
}


# ------------------------------------------------------------------
# ProfileStore
# ------------------------------------------------------------------


class ProfileStore:
    """Load builtin and custom profiles."""

    def __init__(self, custom_dir: Path | None = None) -> None:
        self._profiles: dict[str, AgentProfile] = dict(_BUILTINS)
        self._custom_dir = custom_dir or (get_config_dir() / _PROFILES_DIR)
        self._custom_dir.mkdir(parents=True, exist_ok=True)
        self._load_custom_profiles()

    def get(self, name: str) -> AgentProfile | None:
        return self._profiles.get(name)

    def list_profiles(self) -> list[AgentProfile]:
        return sorted(self._profiles.values(), key=lambda p: p.name)

    def names(self) -> list[str]:
        return sorted(self._profiles.keys())

    def _load_custom_profiles(self) -> None:
        for path in self._custom_dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or "name" not in data:
                    continue
                profile = AgentProfile(
                    name=data["name"],
                    description=data.get("description", ""),
                    bootstrap_files=data.get("bootstrap_files", ["SOUL.md", "AGENTS.md", "ANATOMY.md"]),
                    tools=data.get("tools"),
                    exclude_tools=data.get("exclude_tools", []),
                    subsystems=data.get("subsystems", {}),
                    max_tool_schemas=data.get("max_tool_schemas"),
                )
                self._profiles[profile.name] = profile
            except Exception:
                log.warning("Failed to load custom profile: %s", path)


def get_profile_store() -> ProfileStore:
    """Return a ProfileStore using the default config directory."""
    return ProfileStore()


def filter_tools_by_profile(
    all_schemas: list[dict],
    profile: AgentProfile,
) -> list[dict]:
    """Filter a list of tool schemas according to *profile*.

    If ``profile.tools`` is None, all schemas are included (minus excludes).
    Otherwise only tools named in ``profile.tools`` are kept, then excludes
    are applied.
    """
    if profile.tools is not None:
        allowed = set(profile.tools)
        schemas = [s for s in all_schemas if s.get("name") in allowed]
    else:
        schemas = list(all_schemas)

    if profile.exclude_tools:
        excluded = set(profile.exclude_tools)
        schemas = [s for s in schemas if s.get("name") not in excluded]

    if profile.max_tool_schemas is not None:
        schemas = schemas[: profile.max_tool_schemas]

    return schemas
