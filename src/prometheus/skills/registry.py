"""Skill registry."""

from __future__ import annotations

from prometheus.skills.types import SkillDefinition


class SkillRegistry:
    """Store loaded skills by name."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition) -> None:
        """Register one skill."""
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillDefinition | None:
        """Return a skill by name, case-insensitive fallback."""
        return (
            self._skills.get(name)
            or self._skills.get(name.lower())
            or self._skills.get(name.title())
        )

    def list_skills(self) -> list[SkillDefinition]:
        """Return all skills sorted by name."""
        return sorted(self._skills.values(), key=lambda s: s.name)
