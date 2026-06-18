"""Tests for the skills module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from prometheus.skills.loader import (
    _parse_skill_markdown,
    get_builtin_skills,
    load_skill_registry,
    load_user_skills,
)
from prometheus.skills.registry import SkillRegistry
from prometheus.skills.types import SkillDefinition


# ---------------------------------------------------------------------------
# _parse_skill_markdown
# ---------------------------------------------------------------------------


def test_parse_yaml_frontmatter():
    content = "---\nname: my-skill\ndescription: Does something useful.\n---\n# My Skill\nBody text."
    name, description = _parse_skill_markdown("default", content)
    assert name == "my-skill"
    assert description == "Does something useful."


def test_parse_markdown_heading_fallback():
    content = "# Awesome Skill\n\nThis is what it does."
    name, description = _parse_skill_markdown("default", content)
    assert name == "Awesome Skill"
    assert description == "This is what it does."


def test_parse_default_name_fallback():
    content = "Just some text with no heading."
    name, description = _parse_skill_markdown("my-file", content)
    assert name == "my-file"
    assert description == "Just some text with no heading."


def test_parse_empty_content():
    name, description = _parse_skill_markdown("empty", "")
    assert name == "empty"
    assert description == "Skill: empty"


def test_parse_folded_scalar_description():
    # Regression (SPRINT MEMORY-3): a YAML folded block scalar (`>`) spanning
    # multiple lines must resolve to the joined text, not the literal ">".
    content = (
        "---\n"
        "name: folded-skill\n"
        "description: >\n"
        "  First line of the description\n"
        "  continues onto a second line.\n"
        "---\n"
        "# Folded Skill\n"
        "Body.\n"
    )
    name, description = _parse_skill_markdown("default", content)
    assert name == "folded-skill"
    assert description == "First line of the description continues onto a second line."
    assert ">" not in description


def test_parse_literal_scalar_description():
    # A literal block scalar (`|`) preserves the newline between lines.
    content = (
        "---\n"
        "name: literal-skill\n"
        "description: |\n"
        "  Line one.\n"
        "  Line two.\n"
        "---\n"
    )
    name, description = _parse_skill_markdown("default", content)
    assert name == "literal-skill"
    assert description == "Line one.\nLine two."


def test_parse_frontmatter_non_yaml_falls_back():
    # An unquoted ": " makes the value invalid YAML; the tolerant line-scan
    # fallback must still recover name + description (no regression).
    content = "---\nname: colon-skill\ndescription: Use this: for special cases\n---\n"
    name, description = _parse_skill_markdown("default", content)
    assert name == "colon-skill"
    assert description == "Use this: for special cases"


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------


def test_registry_register_and_get():
    skill = SkillDefinition(name="commit", description="Git commits", content="...", source="builtin")
    reg = SkillRegistry()
    reg.register(skill)
    assert reg.get("commit") is skill
    assert reg.get("COMMIT") is skill  # case-insensitive fallback


def test_registry_list_sorted():
    reg = SkillRegistry()
    for name in ["plan", "commit", "debug"]:
        reg.register(SkillDefinition(name=name, description="", content="", source="builtin"))
    names = [s.name for s in reg.list_skills()]
    assert names == sorted(names)


def test_registry_get_missing():
    reg = SkillRegistry()
    assert reg.get("nonexistent") is None


# ---------------------------------------------------------------------------
# Builtin skills
# ---------------------------------------------------------------------------


def test_builtin_skills_loaded():
    skills = get_builtin_skills()
    names = {s.name for s in skills}
    assert "commit" in names
    assert "debug" in names
    assert "plan" in names
    for skill in skills:
        assert skill.source == "builtin"
        assert skill.content


def test_builtin_skill_has_description():
    skills = {s.name: s for s in get_builtin_skills()}
    for name in ("commit", "debug", "plan"):
        assert skills[name].description, f"{name} should have a description"


# ---------------------------------------------------------------------------
# load_user_skills + load_skill_registry
# ---------------------------------------------------------------------------


def test_load_user_skills_from_directory():
    with tempfile.TemporaryDirectory() as tmp:
        skill_file = Path(tmp) / "my_custom.md"
        skill_file.write_text(
            "---\nname: custom\ndescription: My custom skill.\n---\nDo stuff.\n",
            encoding="utf-8",
        )

        import unittest.mock as mock
        with mock.patch(
            "prometheus.skills.loader.get_user_skills_dir", return_value=Path(tmp)
        ):
            skills = load_user_skills()

    assert len(skills) == 1
    assert skills[0].name == "custom"
    assert skills[0].source == "user"


def test_load_user_skill_with_folded_description():
    # End-to-end (spec side-effect): a skill whose `description:` is a folded
    # block scalar over two lines loads with the full joined description.
    with tempfile.TemporaryDirectory() as tmp:
        skill_file = Path(tmp) / "folded.md"
        skill_file.write_text(
            "---\nname: folded\ndescription: >\n"
            "  A folded description that\n  spans two lines.\n---\nBody.\n",
            encoding="utf-8",
        )

        import unittest.mock as mock
        with mock.patch(
            "prometheus.skills.loader.get_user_skills_dir", return_value=Path(tmp)
        ):
            skills = load_user_skills()

    assert len(skills) == 1
    assert skills[0].description == "A folded description that spans two lines."


def test_load_skill_registry_includes_builtins():
    registry = load_skill_registry()
    names = {s.name for s in registry.list_skills()}
    assert "commit" in names
    assert "debug" in names
    assert "plan" in names
