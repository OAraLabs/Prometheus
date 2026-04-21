"""Skill loading from builtin and user directories."""

from __future__ import annotations

from pathlib import Path

from prometheus.config.paths import get_config_dir
from prometheus.skills.registry import SkillRegistry
from prometheus.skills.types import SkillDefinition

_BUILTIN_SKILLS_DIR = Path(__file__).parent / "builtin"


def get_user_skills_dir() -> Path:
    """Return the user skills directory (~/.prometheus/skills/)."""
    path = get_config_dir() / "skills"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_builtin_skills() -> list[SkillDefinition]:
    """Return skills bundled with Prometheus."""
    skills: list[SkillDefinition] = []
    if not _BUILTIN_SKILLS_DIR.exists():
        return skills
    for path in sorted(_BUILTIN_SKILLS_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        name, description = _parse_skill_markdown(path.stem, content)
        skills.append(
            SkillDefinition(
                name=name,
                description=description,
                content=content,
                source="builtin",
                path=str(path),
            )
        )
    return skills


def load_user_skills() -> list[SkillDefinition]:
    """Load markdown skills from the user config directory.

    Scans both ``~/.prometheus/skills/*.md`` and
    ``~/.prometheus/skills/auto/*.md`` (auto-generated skills).
    """
    skills: list[SkillDefinition] = []
    user_dir = get_user_skills_dir()
    # Collect from top-level and auto/ subdirectory
    paths = sorted(user_dir.glob("*.md"))
    auto_dir = user_dir / "auto"
    if auto_dir.is_dir():
        paths.extend(sorted(auto_dir.glob("*.md")))
    for path in paths:
        content = path.read_text(encoding="utf-8")
        name, description = _parse_skill_markdown(path.stem, content)
        source = "auto" if "auto" in path.parts else "user"
        skills.append(
            SkillDefinition(
                name=name,
                description=description,
                content=content,
                source=source,
                path=str(path),
            )
        )
    return skills


def load_skill_registry(cwd: str | Path | None = None) -> SkillRegistry:
    """Load builtin and user-defined skills into a registry."""
    del cwd  # reserved for future plugin loading
    registry = SkillRegistry()
    for skill in get_builtin_skills():
        registry.register(skill)
    for skill in load_user_skills():
        registry.register(skill)
    return registry


def _parse_skill_markdown(default_name: str, content: str) -> tuple[str, str]:
    """Extract name and description from a skill markdown file.

    Checks YAML frontmatter (--- ... ---) first, then falls back to
    the first heading and first paragraph.
    """
    name = default_name
    description = ""
    lines = content.splitlines()

    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                for fm_line in lines[1:i]:
                    fm = fm_line.strip()
                    if fm.startswith("name:"):
                        val = fm[5:].strip().strip("'\"")
                        if val:
                            name = val
                    elif fm.startswith("description:"):
                        val = fm[12:].strip().strip("'\"")
                        if val:
                            description = val
                break

    if not description:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# "):
                if name == default_name:
                    name = stripped[2:].strip() or default_name
                continue
            if stripped and not stripped.startswith("---") and not stripped.startswith("#"):
                description = stripped[:200]
                break

    if not description:
        description = f"Skill: {name}"
    return name, description
