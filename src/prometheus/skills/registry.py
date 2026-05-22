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

        Idempotency: if two ``.md`` files declare the same ``name:`` in
        their frontmatter (e.g. ``foo.md`` and a Curator-generated
        ``foo.bak-<ts>.md``), the loader returns both as separate
        entries. We dedupe by name *before* comparing against existing
        state — last entry wins, matching ``load_skill_registry``'s
        ``register`` semantics — so a second back-to-back call with no
        on-disk changes always returns 0. Without dedupe, the two entries
        would flip the registry's bound path/content on each call and
        every reload would report N > 0.
        """
        # Lazy import — registry.py is imported by loader.py, avoid a cycle.
        from prometheus.skills.loader import load_user_skills

        # Dedupe by name, last-write-wins (mirrors load_skill_registry).
        latest: dict[str, SkillDefinition] = {}
        for skill in load_user_skills():
            latest[skill.name] = skill

        added = 0
        for name, skill in latest.items():
            existing = self._skills.get(name)
            if existing is None or existing.path != skill.path or \
                    existing.content != skill.content:
                self._skills[name] = skill
                added += 1
        return added
