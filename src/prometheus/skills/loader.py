"""Skill loading from builtin and user directories."""

from __future__ import annotations

from pathlib import Path

import yaml

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
                core=_skill_is_core(content),
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
                core=_skill_is_core(content),
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


def skills_for_prompt() -> list[dict[str, str]] | None:
    """Build the ``skills`` argument for the system-prompt assembler.

    Returns ``[{"name", "description", "core"}, ...]`` for every loaded
    skill, or ``None`` when there are none (or loading fails) so callers
    can pass the result straight through. ``core`` is True for skills
    whose frontmatter opts into the always-visible prompt tier
    (``tier: core`` or ``autoload: true``); the assembler renders those
    name+description lines inline and keeps the rest tool_search-only.
    Additive and fail-open: a broken skill file must never block prompt
    assembly, so returning ``None`` simply skips the skills section.
    """
    try:
        registry = load_skill_registry()
    except Exception:
        return None
    skills = registry.list_skills()
    if not skills:
        return None
    return [
        {"name": s.name, "description": s.description, "core": s.core}
        for s in skills
    ]


def _skill_is_core(content: str) -> bool:
    """Return True when a skill opts into the always-visible prompt tier.

    Recognised frontmatter keys (either one is enough): ``tier: core`` or
    ``autoload: true``. Skills without the flag stay reachable through
    tool_search only — they are never lost, just not pre-rendered.
    Fail-open False: malformed or absent frontmatter never breaks prompt
    assembly and simply keeps the skill out of the core tier.
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return False
    block = None
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            block = chr(10).join(lines[1:i])  # frontmatter block
            break
    if block is None:
        return False
    try:
        data = yaml.safe_load(block)
    except yaml.YAMLError:
        data = None
    if isinstance(data, dict):
        tier = str(data.get("tier", "")).strip().lower()
        if tier == "core":
            return True
        autoload = data.get("autoload", False)
        return autoload is True or str(autoload).strip().lower() in ("true", "yes", "1")
    # Fallback tolerant scan for frontmatter that is not valid YAML.
    for fm_line in block.splitlines():
        fm = fm_line.strip().lower()
        if fm.startswith("tier:") and fm[5:].strip().strip("'\"") == "core":
            return True
        if fm.startswith("autoload:") and fm[9:].strip().strip("'\"").lower() in (
            "true", "yes", "1",
        ):
            return True
    return False


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
                fm_name, fm_description = _parse_frontmatter("\n".join(lines[1:i]))
                if fm_name:
                    name = fm_name
                if fm_description:
                    description = fm_description
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


def _parse_frontmatter(block: str) -> tuple[str, str]:
    """Return ``(name, description)`` parsed from a YAML frontmatter block.

    Parses with a real YAML loader so block scalars (``description: >`` or
    ``description: |``) and quoted or multi-line values resolve to their full
    text instead of the bare ``>``/``|`` indicator. The previous hand-rolled
    scan captured only the same-line remainder, turning a folded description
    into the literal ``">"``.

    Falls back to a tolerant line scan when the block is not valid YAML — many
    hand-written descriptions contain an unquoted ``": "`` that PyYAML rejects —
    so the loader stays as lenient as before for everything except the folded
    case it now fixes.
    """
    try:
        data = yaml.safe_load(block)
    except yaml.YAMLError:
        data = None

    if isinstance(data, dict):
        raw_name = data.get("name")
        raw_description = data.get("description")
        name = str(raw_name).strip() if isinstance(raw_name, (str, int, float)) else ""
        description = (
            str(raw_description).strip()
            if isinstance(raw_description, (str, int, float))
            else ""
        )
        return name, description

    # Fallback: tolerant line scan for frontmatter that is not valid YAML.
    name = ""
    description = ""
    block_scalar_indicators = {">", "|", ">-", "|-", ">+", "|+"}
    for fm_line in block.splitlines():
        fm = fm_line.strip()
        if fm.startswith("name:"):
            val = fm[5:].strip().strip("'\"")
            if val:
                name = val
        elif fm.startswith("description:"):
            val = fm[12:].strip().strip("'\"")
            if val and val not in block_scalar_indicators:
                description = val
    return name, description
