"""tool_search finds what the model means (audit 2026-10-16, fix 4).

* ``select`` resolves a near miss to the one tool whose name, as words,
  appears whole in it ("select cron_create", "cron create", "agent explorer");
  the longest when names nest (``grep`` inside ``lcm_grep``); two candidates
  resolve nothing.
* A miss lists the 5 closest names, not every tool and skill.
* Each tool declares one or two ``find_me`` requests and search ranks on them;
  they never reach the schema the model is sent.

Every registered tool, through the checkout's registry:
tests/test_local_tool_reachability.py (Choice).
"""

from __future__ import annotations

import json

import pytest

from prometheus.tools.base import ToolRegistry
from prometheus.tools.tool_search import ToolSearchTool, find_me_score


@pytest.fixture(scope="module")
def search():
    from prometheus.__main__ import create_tool_registry

    reg = create_tool_registry({})
    tool = next(t for t in reg.list_tools() if t.name == "tool_search")
    tool.set_registry(reg)
    return tool


def _select(search, query):
    res = search._handle_select(query)
    return res, json.loads(res.output)


@pytest.mark.parametrize("query,name", [
    ("select cron_create", "cron_create"),
    ("cron create", "cron_create"),
    ("cron-create", "cron_create"),
    ("agent explorer", "agent"),
    ("the lcm_grep tool", "lcm_grep"),     # grep is a tool too: the longer name wins
    ("web_search please", "web_search"),
])
def test_a_near_miss_resolves_to_the_one_tool_it_names(search, query, name):
    res, body = _select(search, query)
    assert not res.is_error and body["name"] == name
    assert body["resolved_from"] == query


def test_an_exact_name_is_not_marked_resolved(search):
    res, body = _select(search, "cron_create")
    assert body["name"] == "cron_create" and "resolved_from" not in body


@pytest.mark.parametrize("query", ["read_file or write_file", "edit", "something unrelated"])
def test_two_candidates_or_none_resolve_nothing(search, query):
    res, body = _select(search, query)
    assert res.is_error and "resolved_from" not in body


def test_a_skill_named_like_a_tool_stays_the_skill():
    from types import SimpleNamespace

    from prometheus.__main__ import create_tool_registry

    reg = create_tool_registry({})
    tool = ToolSearchTool()
    tool.set_registry(reg)
    skill = SimpleNamespace(name="web_search_tips", description="tips", source="user", path="/x")
    tool.set_skill_registry(SimpleNamespace(get=lambda n: skill if n == skill.name else None,
                                            list_skills=lambda: [skill]))
    body = json.loads(tool._handle_select("web_search_tips").output)
    assert body["type"] == "skill" and body["name"] == "web_search_tips"


def test_a_miss_lists_the_five_closest_names(search):
    res, body = _select(search, "cron_creat")
    assert res.is_error
    assert len(body["closest"]) == 5 and body["closest"][0].startswith("cron")
    assert "available" not in body


def test_search_ranks_a_tool_by_the_requests_it_declares(search):
    names = [e["name"] for e in json.loads(search._handle_search("schedule a recurring job").output)]
    assert names[0] == "cron_create"
    names = [e["name"] for e in json.loads(search._handle_search("please schedule a recurring job for me").output)]
    assert names[0] == "cron_create"


def test_find_me_never_reaches_the_schema(search):
    reg: ToolRegistry = search._registry
    for tool in reg.list_tools():
        assert tool.find_me, f"{tool.name} declares no find_me"
        assert "find_me" not in json.dumps(tool.to_api_schema())


def test_find_me_scores():
    declared = ("schedule a recurring job", "create a cron job")
    assert find_me_score("schedule a recurring job", declared) == 0.25
    assert find_me_score("schedule a recurring job daily", declared) == 0.5
    assert 1.0 <= find_me_score("recurring job schedule", declared) <= 1.2
    assert find_me_score("read a file", declared) is None
