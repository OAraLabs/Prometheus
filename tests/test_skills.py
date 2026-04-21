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


def test_load_skill_registry_includes_builtins():
    registry = load_skill_registry()
    names = {s.name for s in registry.list_skills()}
    assert "commit" in names
    assert "debug" in names
    assert "plan" in names
