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

    def reload_user_skills(self) -> int:
        """Re-scan ~/.prometheus/skills/ and merge new/updated entries.

        Used after a Printing Press install so the freshly-copied
        SKILL.md is available in the same conversation — no daemon
        restart required. Returns the number of skills added or
        replaced. Existing skills with no on-disk match are kept
        (this is purely additive).
        """
        # Lazy import — registry.py is imported by loader.py, avoid a cycle.
        from prometheus.skills.loader import load_user_skills

        added = 0
        for skill in load_user_skills():
            existing = self._skills.get(skill.name)
            if existing is None or existing.path != skill.path or \
                    existing.content != skill.content:
                self._skills[skill.name] = skill
                added += 1
        return added
