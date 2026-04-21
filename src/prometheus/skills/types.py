"""Skill data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillDefinition:
    """A loaded skill."""

    name: str
    description: str
    content: str
    source: str  # "builtin" | "user" | "plugin"
    path: str | None = None
