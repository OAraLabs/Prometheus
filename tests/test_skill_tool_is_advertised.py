"""The ``skill`` tool is advertised by default, and says what a skill is (option C2).

The skill-usage audit (docs/audits/SKILL-USAGE.md §3) found ``skill`` outside
the deferred set. 822 of 973 main-registry runs were told to "use the skill
tool" with no such tool in their list. That was every local run since
2026-08-03, plus 167 of 287 Qwen 3.8 Max runs, which were served with the local
set before #462. The model had only a one-line description ("Read a builtin or
user-defined skill by name.") that never said what a skill is or when to load
one.

This changes the advertised tool list, so it is trace-changing by design.
"""

from __future__ import annotations

from prometheus.config.shipped_defaults import SHIPPED_ALWAYS_LOADED
from prometheus.context.dynamic_tools import DynamicToolLoader
from prometheus.tools.base import ToolRegistry
from prometheus.tools.builtin.cron_list import CronListTool
from prometheus.tools.builtin.skill import SkillTool
from tests.support.advertisement import template_always_loaded


def test_skill_is_in_the_shipped_always_loaded():
    assert "skill" in SHIPPED_ALWAYS_LOADED


def test_the_template_ships_it_too():
    assert "skill" in template_always_loaded()


def test_a_deferred_run_is_offered_the_skill_tool():
    registry = ToolRegistry()
    registry.register(SkillTool())
    registry.register(CronListTool())
    loader = DynamicToolLoader(registry, None)  # no config: the shipped set
    names = [s["name"] for s in loader.schemas_for_run(True)]
    assert "skill" in names
    assert "cron_list" not in names  # still deferred: the set is not "everything"


def test_the_description_says_what_a_skill_is_and_when_to_load_one():
    d = SkillTool.description
    assert "instructions" in d.lower()  # what a skill is
    assert "before" in d.lower()        # when to load one
    assert "tool_search" in d           # how to find one that is not listed
    # It rides every request once advertised: keep it to a few lines.
    assert len(d) <= 400


def test_the_prompt_and_the_tool_agree_on_the_name():
    from prometheus.context.prompt_assembler import build_runtime_system_prompt

    prompt = build_runtime_system_prompt(
        cwd=".", config={"bootstrap": {"load_soul": False, "load_agents": False},
                         "anatomy": {"include_in_system_prompt": False}},
        memory_content="(none)",
        skills=[{"name": "x", "description": "d", "core": False}],
    )
    assert f"use the {SkillTool.name} tool" in prompt
