"""tool_search lists a skill only on a real name/description match.

Tools and skills share one top-5 ranked by substring tiers and, failing any
text match, by Levenshtein distance to the NAME. With ~134 skill names against
~55 tool names, a query that matched nothing filled the list with arbitrary
skills: in the audit's replay, 20 of the 25 user-surface searches that listed
a skill did so on edit distance alone (docs/audits/SKILL-USAGE.md §1).

The rule now (skills only; tool ranking is unchanged): a skill is listed when
the whole query appears in its name or description, or when its match score —
the share of the query's words of 3+ letters (stopwords excluded) that match a
whole word, or the start of a word of 4+ letters, in its name or description —
is at least ``SKILL_MIN_MATCH`` (0.5). Edit distance alone never lists one.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from prometheus.skills.registry import SkillRegistry
from prometheus.skills.types import SkillDefinition
from prometheus.tools.base import ToolExecutionContext, ToolRegistry
from prometheus.tools.builtin.bash import BashTool
from prometheus.tools.builtin.grep import GrepTool
from prometheus.tools.tool_search import (
    SKILL_MIN_MATCH,
    ToolSearchInput,
    ToolSearchTool,
    skill_match_score,
)


def _tool() -> ToolSearchTool:
    tools = ToolRegistry()
    tools.register(BashTool())
    tools.register(GrepTool())
    skills = SkillRegistry()
    for name, desc in (
        ("docker-deploy", "Deploy containers with Docker and docker-compose"),
        ("invoice-archive-locate", "Locate the archive folder inside the ledger"),
        ("catalog-audit", "Audit a product catalog for missing fields"),
    ):
        skills.register(SkillDefinition(name=name, description=desc, content="#", source="auto"))
    t = ToolSearchTool()
    t.set_registry(tools)
    t.set_skill_registry(skills)
    return t


def _search(query: str) -> list[dict]:
    result = asyncio.run(_tool().execute(ToolSearchInput(query=query),
                                         ToolExecutionContext(cwd=Path.cwd())))
    return json.loads(result.output)


def _skills(entries: list[dict]) -> list[str]:
    return [e["name"] for e in entries if e.get("type") == "skill"]


def test_the_stated_threshold():
    assert SKILL_MIN_MATCH == 0.5


class TestEditDistanceAloneListsNoSkill:
    def test_a_query_that_matches_nothing_lists_no_skill(self):
        # Before: 2 tools + 3 skills all fit in the top 5, so all 3 skills were listed.
        assert _skills(_search("zqxv")) == []

    def test_a_near_miss_spelling_of_a_skill_name_is_not_a_match(self):
        # Two edits from "docker deploy": close by Levenshtein, no word matches.
        assert _skills(_search("dokcer dploy")) == []

    def test_tools_still_rank_by_edit_distance(self):
        names = [e["name"] for e in _search("bsh")]
        assert "bash" in names


class TestRealMatchesStillList:
    def test_the_whole_query_in_a_name(self):
        assert _skills(_search("docker")) == ["docker-deploy"]

    def test_the_whole_query_in_a_description(self):
        assert "docker-deploy" in _skills(_search("docker-compose"))

    def test_half_the_meaningful_words(self):
        # deploy (word start of "deploy"), containers (whole word); "please" misses: 2/3.
        assert _skills(_search("deploy my containers please")) == ["docker-deploy"]

    def test_below_half_is_not_enough(self):
        # locate matches, "files" and "fast" do not: 1/3.
        assert "invoice-archive-locate" not in _skills(_search("locate files fast"))


class TestWhatDoesNotCountAsAMatch:
    def test_stopwords_and_short_words_are_ignored(self):
        assert _skills(_search("the of to and a")) == []
        assert _skills(_search("find the thing")) == []

    def test_a_query_word_inside_a_longer_word_does_not_count(self):
        # "log" is inside "catalog", but it is not a word or a word start.
        assert skill_match_score("log", "catalog-audit", "Audit a product catalog") == 0.0

    def test_a_word_start_needs_four_letters(self):
        assert skill_match_score("dep", "docker-deploy", "Deploy containers") == 0.0
        assert skill_match_score("depl", "docker-deploy", "Deploy containers") == 1.0


def test_select_and_empty_query_are_unchanged():
    t = _tool()
    ctx = ToolExecutionContext(cwd=Path.cwd())
    sel = json.loads(asyncio.run(t.execute(ToolSearchInput(query="catalog-audit", action="select"),
                                           ctx)).output)
    assert sel["type"] == "skill"
    listing = json.loads(asyncio.run(t.execute(ToolSearchInput(query=""), ctx)).output)
    assert listing["skills"] == ["catalog-audit", "docker-deploy", "invoice-archive-locate"]


@pytest.mark.parametrize("query,expected", [
    ("docker", 1.0),                 # whole query in the name
    ("archive ledger", 1.0),         # both words whole
    ("ledger stuff", 0.5),           # one of two
    ("", 0.0),
])
def test_match_score_values(query, expected):
    assert skill_match_score(query, "invoice-archive-locate docker",
                             "Locate the archive folder inside the ledger") == expected
