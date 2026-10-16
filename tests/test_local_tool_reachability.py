"""Tool reachability on a local model, for EVERY registered tool.

Parametrized over the registry of the checkout under test (collected by asking
it, so a new or changed tool is covered without editing this file), plus the
gate's MCP tool. Three properties:

1. Awareness: the local prompt says how many tools are not loaded and names
   them (by category), generated from the registry. A registered tool in
   neither the loaded set nor that summary is a tool the model cannot know.
2. Access: through the real daemon and the strict llama.cpp stand-in,
   tool_search finds it, its schema comes back, it is called in the same
   turn, it really runs, and the client receives its result. The daemon runs
   in ``autonomous`` mode with outbound HTTP pointed at a dead port, inside
   a temp sandbox, so "runs" means the tool's own code ran.
3. Choice: each tool declares one or two plain-language queries next to its
   definition (``find_me`` on the tool class); each must rank the tool in
   tool_search's top 3, and near-miss names ("select cron_create", "cron
   create", "agent explorer") must resolve to it with select.

What 0.9.12 fails is ``xfail(strict=True)``: Access fails for every deferred
tool, Awareness for every deferred tool, Choice for every tool (no tool
declares queries and select is exact-name only). Read another release with
``LOCAL_GATE_CHECKOUT=<checkout> ... --runxfail``.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess

import pytest

from tests.support.gate_daemon import MCP_TOOLS, build_and_boot, checkout_under_test
from tests.support.scripted_llama import ScriptedLlamaServer, want_then_answer
from tests.support.served_template import recorded_props

pytestmark = pytest.mark.integration

_INVENTORY_CODE = r'''
import json
from prometheus.__main__ import create_tool_registry
from prometheus.tools.tool_search import ToolSearchTool

def near_misses(name):
    words = name.replace("__", " ").replace("_", " ").strip()
    cands = {f"select {name}", words, name.replace("_", "-"), f"{name} tool", f"{name} explorer" if name == "agent" else f"{name} please"}
    return sorted(c for c in cands if c != name)

reg = create_tool_registry({})
# The gate's config always configures an MCP server, and the daemon then also
# registers mcp_status (mcp/bootstrap.py); the factory alone does not.
try:
    from prometheus.tools.builtin.mcp_status import McpStatusTool
    reg.register(McpStatusTool(None))
except Exception:
    pass
search = next((t for t in reg.list_tools() if t.name == "tool_search"), None) or ToolSearchTool()
search.set_registry(reg)
out = []
for tool in reg.list_tools():
    schema = tool.to_api_schema()
    queries = [q for q in (getattr(tool, "find_me", ()) or ()) if isinstance(q, str)]
    ranks = {}
    for q in queries:
        names = [e.get("name") for e in json.loads(search._handle_search(q).output)]
        ranks[q] = names.index(tool.name) + 1 if tool.name in names else None
    near = {}
    for miss in near_misses(tool.name):
        res = search._handle_select(miss)
        try:
            body = json.loads(res.output)
        except ValueError:
            body = {}
        near[miss] = (not res.is_error) and isinstance(body, dict) and body.get("name") == tool.name
    out.append({"name": tool.name, "schema": schema.get("input_schema") or {}, "find_me": queries,
                "ranks": ranks, "near": near})
print(json.dumps(out))
'''


def _inventory() -> list[dict]:
    root, python = checkout_under_test()
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONSAFEPATH": "1"}
    out = subprocess.run([python, "-c", _INVENTORY_CODE], env=env, capture_output=True, text=True, timeout=300,
                         cwd=str(root))
    if out.returncode != 0:
        raise RuntimeError(f"could not list the registry of {root}:\n{out.stderr[-2000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


INVENTORY = {t["name"]: t for t in _inventory()}
for _t in MCP_TOOLS:
    INVENTORY["mcp__docs__" + _t["name"]] = {"name": "mcp__docs__" + _t["name"], "schema": _t["inputSchema"],
                                            "find_me": [], "ranks": {}, "near": {}}
TOOLS = sorted(INVENTORY)


# ---------------------------------------------------------------- arguments


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


_PATHY = re.compile(r"(^|_)(path|file|dir|directory|root|cwd|notebook|output|dest)", re.I)


def _resolve(spec: dict, defs: dict) -> dict:
    """Follow ``$ref`` into ``$defs`` and take the first non-null branch of
    ``anyOf``/``oneOf``/``allOf``, so an enum behind a pydantic Enum is seen."""
    for _ in range(5):
        ref = spec.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            spec = {**defs.get(ref.rsplit("/", 1)[-1], {}), **{k: v for k, v in spec.items() if k != "$ref"}}
            continue
        for key in ("anyOf", "oneOf", "allOf"):
            branches = [b for b in spec.get(key) or [] if b.get("type") != "null"]
            if branches:
                spec = {**{k: v for k, v in spec.items() if k != key}, **branches[0]}
                break
        else:
            return spec
    return spec


def _value(name: str, spec: dict, tool: str):
    if "const" in spec:
        return spec["const"]
    if "enum" in spec and spec["enum"]:
        return spec["enum"][0]
    kind = spec.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "string")
    if kind in ("integer", "number"):
        return _free_port() if "port" in name else 1
    if kind == "boolean":
        return False
    if kind == "array":
        return []
    if kind == "object":
        return {}
    if _PATHY.search(name):
        return "{workspace}/gate-" + tool + ".txt"
    if name in ("command", "cmd"):
        return "echo gate"
    if "url" in name:
        return "http://127.0.0.1:9/"
    if name == "schedule":
        return "0 9 * * 1-5"
    if name in ("session_id", "target_session"):
        return "beacon:gate-reach"
    return "gate"


def args_for(tool: str) -> dict:
    schema = INVENTORY[tool]["schema"] or {}
    props = schema.get("properties") or {}
    defs = schema.get("$defs") or {}
    return {k: _value(k, _resolve(props.get(k) or {}, defs), tool) for k in schema.get("required") or []}


# ---------------------------------------------------------------- the daemon


@pytest.fixture(scope="module")
def reach(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("reach")
    with ScriptedLlamaServer(props=recorded_props(), workspace=str(tmp / "workspace")) as model:
        daemon = build_and_boot(tmp, model.url, extra={"security.permission_mode": "autonomous"})
        try:
            yield daemon, model
        finally:
            daemon.stop()


@pytest.fixture(scope="module")
def round_zero(reach):
    """A plain local turn's first request: what the model is told."""
    daemon, model = reach
    daemon.turn("[gate:hello] Hi.")
    return next(e for e in model.of("hello") if e.status == 200)


#: Results the LOOP writes instead of running the tool (agent_loop.py's own
#: ToolResultBlocks and the adapter's retry prompt). A tool that ran and failed
#: ("Tool X raised an exception", its own error text) did run.
_REFUSED = ("Unknown tool:", "Invalid input for ", "Permission denied for ", "pre_tool_use hook blocked",
            "No tool registry configured", "chat-template markup", "NOT EXECUTED",
            "Your tool call for ", "Giving up on ")


def _deferred_summary(system: str) -> str:
    """The prompt's statement of the tools not loaded: from the sentence that
    gives their number to the next top-level heading ("" when there is none)."""
    found = re.search(r"(\d+)\s+(?:more|other|additional|deferred|unloaded)\s+tools", system)
    if not found:
        return ""
    end = system.find("\n# ", found.end())
    return system[found.start(): end if end >= 0 else len(system)]


# ---------------------------------------------------------------- 1. awareness


@pytest.mark.xfail(strict=True, reason="awareness on 0.9.12: the local prompt says nothing about the deferred tools")
def test_awareness_the_prompt_states_how_many_tools_are_deferred(round_zero):
    deferred = [t for t in TOOLS if t not in round_zero.offered]
    found = re.search(r"(\d+)\s+(?:more|other|additional|deferred|unloaded)\s+tools", round_zero.system)
    assert found and int(found.group(1)) == len(deferred), (
        f"{len(deferred)} tools are deferred; the prompt states {found.group(0) if found else 'no count'}")


@pytest.mark.parametrize("tool", TOOLS)
def test_awareness_every_tool_is_loaded_or_named(round_zero, tool, request):
    if tool not in round_zero.offered:
        request.applymarker(pytest.mark.xfail(strict=True, reason=(
            "awareness on 0.9.12: no deferred summary (a deferred name elsewhere in the prompt, like the "
            "base prompt's lcm_grep, does not tell the model the tool exists but must be loaded)")))
    summary = _deferred_summary(round_zero.system)
    named = re.search(rf"(?<![\w-]){re.escape(tool)}(?![\w-])", summary) is not None
    assert tool in round_zero.offered or named, (
        f"tool {tool} is registered but neither loaded nor named in the deferred summary")


# ---------------------------------------------------------------- 2. access


def test_access_the_daemon_registers_nothing_this_suite_misses(reach):
    daemon, model = reach
    model.scripts["inventory"] = want_then_answer("tool_search", {"query": "", "action": "search"})
    turn = daemon.turn("[gate:inventory] List everything.")
    result = next((p.get("result", "") for p in turn.of("tool_call_end") if p.get("tool_name") == "tool_search"), "")
    try:
        listed = set(json.loads(result).get("tools", []))
    except ValueError:
        pytest.fail(f"tool_search with an empty query did not list the tools: {result[:300]!r}")
    assert listed - set(TOOLS) == set(), "the daemon registers tools the suite does not cover"


@pytest.mark.parametrize("tool", TOOLS)
def test_access_tool_is_found_called_run_and_its_result_reaches_the_client(reach, round_zero, tool, request):
    daemon, model = reach
    deferred = tool not in round_zero.offered
    if deferred:
        request.applymarker(pytest.mark.xfail(strict=True, reason=(
            "access on 0.9.12: a deferred tool never joins the run's tools, so the server cannot call it")))
    key = "reach_" + re.sub(r"[^a-z0-9_]", "_", tool.lower())
    model.scripts[key] = want_then_answer(tool, args_for(tool))
    turn = daemon.turn(f"[gate:{key}] Use the tool you need for this.", timeout=180)
    ends = turn.of("tool_call_end")
    if deferred:
        searched = [p for p in ends if p.get("tool_name") == "tool_search"]
        assert any(f'"{tool}"' in (p.get("result") or "") for p in searched), (
            f"{tool}: tool_search never returned its schema to the turn")
    assert tool in turn.tools_started, f"{tool}: found but never called (tools started: {turn.tools_started})"
    mine = [p for p in ends if p.get("tool_name") == tool]
    assert mine, f"{tool}: called, but the client received no result"
    result = mine[-1].get("result") or ""
    assert not any(r in result[:300] for r in _REFUSED), f"{tool}: dispatched but not run: {result[:200]!r}"
    assert turn.selects() <= 1, f"{tool}: {turn.selects()} tool_search selects in one turn"


# ---------------------------------------------------------------- 3. choice


@pytest.mark.parametrize("tool", [t for t in TOOLS if not t.startswith("mcp__")])
@pytest.mark.xfail(strict=True, reason="choice on 0.9.12: no tool declares find_me queries")
def test_choice_tool_declares_one_or_two_find_me_queries(tool):
    queries = INVENTORY[tool]["find_me"]
    assert 1 <= len(queries) <= 2, f"tool {tool} declares {len(queries)} find_me queries next to its definition"


@pytest.mark.parametrize("tool", [t for t in TOOLS if not t.startswith("mcp__")])
@pytest.mark.xfail(strict=True, reason="choice on 0.9.12: no tool declares find_me queries to rank")
def test_choice_tool_is_findable_by_its_own_description(tool):
    ranks = INVENTORY[tool]["ranks"]
    assert ranks, f"tool {tool} not findable by its own description: it declares no find_me queries"
    misses = {q: r for q, r in ranks.items() if r is None or r > 3}
    assert not misses, f"tool {tool} not findable by its own description: {misses} (rank in tool_search, top 3 needed)"


@pytest.mark.parametrize("tool", [t for t in TOOLS if not t.startswith("mcp__")])
@pytest.mark.xfail(strict=True, reason="choice on 0.9.12: select is exact-name only")
def test_choice_near_miss_names_resolve_to_the_tool(tool):
    near = INVENTORY[tool]["near"]
    unresolved = sorted(m for m, ok in near.items() if not ok)
    assert not unresolved, f"tool {tool}: near-miss names that do not resolve with select: {unresolved}"
