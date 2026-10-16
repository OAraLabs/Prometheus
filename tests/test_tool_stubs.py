"""Every tool callable on a local run: full schemas and stubs (audit 2026-10-16, fix 3).

The far side, through the real daemon and a stand-in that admits only offered
names and declared argument names: tests/test_local_gate_integration.py (a, b)
and tests/test_local_tool_reachability.py (access, awareness).
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest
from pydantic import BaseModel

from prometheus.adapter import ModelAdapter
from prometheus.context.dynamic_tools import DynamicToolLoader
from prometheus.context.tool_stubs import (
    DEFAULT_CATALOG,
    FULL_SCHEMA_TOOLS,
    MAX_SCHEMA_RETRIES,
    MAX_STUB_ENUM,
    catalog,
    first_line,
    minimal_example,
    refusal,
    sanitize,
    skeleton,
)
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.stream_events import ToolExecutionCompleted
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult


@pytest.fixture(scope="module")
def registry():
    from prometheus.__main__ import create_tool_registry

    return create_tool_registry({})


def _props(schema: dict) -> dict:
    return schema["input_schema"].get("properties") or {}


def test_every_tool_keeps_every_argument(registry):
    """A property NAMED like a schema keyword (glob's "pattern") is an argument:
    the grammar admits only declared names, so dropping one makes it unsendable."""
    for tool in registry.list_tools():
        raw = set(tool.input_model.model_json_schema().get("properties") or {})
        sent = set(_props(skeleton(tool.to_api_schema())))
        assert sent == raw, f"{tool.name}: stub lost {sorted(raw - sent)}"
        assert set(sanitize(tool.input_model.model_json_schema()).get("properties") or {}) == raw


def test_a_stub_is_name_one_line_and_typed_optional_sorted_properties(registry):
    stub = skeleton(registry.get("cron_create").to_api_schema())
    assert stub["name"] == "cron_create"
    assert "\n" not in stub["description"] and len(stub["description"]) <= 120
    props = _props(stub)
    assert list(props) == sorted(props)
    assert "required" not in stub["input_schema"]
    assert all(set(p) <= {"type", "enum"} for p in props.values())


def test_a_short_enum_is_listed_and_a_long_one_is_not():
    short = {"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {
        "mode": {"type": "string", "enum": ["messages", "summaries", "both"]},
        "size": {"type": "string", "enum": [f"s{i}" for i in range(MAX_STUB_ENUM + 1)]}}}}
    props = _props(skeleton(short))
    assert props["mode"] == {"type": "string", "enum": ["messages", "summaries", "both"]}
    assert props["size"] == {"type": "string"}


def test_the_catalog_is_sorted_full_for_the_ambient_tools_and_stubs_for_the_rest(registry):
    block, undescribed = catalog(registry.to_api_schema())
    names = [s["name"] for s in block]
    assert names == sorted(names) and set(names) == {t.name for t in registry.list_tools()}
    for s in block:
        if s["name"] in FULL_SCHEMA_TOOLS:
            assert s["name"] not in undescribed
            assert s["input_schema"] == sanitize(registry.get(s["name"]).input_model.model_json_schema())
        else:
            assert s["name"] in undescribed
    assert {"agent", "ask_user"} <= set(FULL_SCHEMA_TOOLS)


def test_the_catalog_is_byte_identical_whatever_the_registration_order(registry):
    schemas = registry.to_api_schema()
    a = json.dumps(catalog(schemas)[0])
    b = json.dumps(catalog(list(reversed(schemas)))[0])
    assert a == b


def test_sanitize_inlines_refs_and_collapses_optional():
    schema = {"$defs": {"Mode": {"type": "string", "enum": ["a", "b"]}},
              "properties": {"mode": {"$ref": "#/$defs/Mode"},
                             "root": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Root"},
                             "pattern": {"type": "string", "pattern": "^x"}},
              "type": "object", "title": "In"}
    out = sanitize(schema)
    assert "$defs" not in out and "title" not in out
    assert out["properties"]["mode"] == {"type": "string", "enum": ["a", "b"]}
    assert out["properties"]["root"] == {"type": "string"}
    assert out["properties"]["pattern"] == {"type": "string"}


def test_first_line():
    assert first_line("Do a thing. Then more.") == "Do a thing."
    assert len(first_line("x" * 300)) == 120


def test_refusal_carries_the_schema_and_a_minimal_call_then_stops(registry):
    schema = registry.get("cron_create").input_model.model_json_schema()
    first = refusal("cron_create", schema, "name: Field required", 1)
    assert "Its full schema:" in first and "A minimal valid call:" in first
    assert json.loads(first.split("A minimal valid call: ", 1)[1].split("\n")[0]).keys() == set(schema["required"])
    last = refusal("cron_create", schema, "name: Field required", MAX_SCHEMA_RETRIES + 1)
    assert "Not retried again" in last and "full schema" not in last


def test_minimal_example_uses_an_enums_first_value():
    assert minimal_example({"properties": {"m": {"enum": ["x", "y"]}}, "required": ["m"]}) == {"m": "x"}


def test_offered_for_run_reports_what_the_loop_sends(registry):
    pinned = {"always_loaded": ["bash", "memory", "vault_search"]}
    full, stubs = DynamicToolLoader(registry, pinned).offered_for_run(True)
    assert set(full) == set(FULL_SCHEMA_TOOLS) & {t.name for t in registry.list_tools()}
    assert "memory" in stubs and "vault_search" in stubs        # always_loaded decides nothing here
    assert len(full) + len(stubs) == len(registry.list_tools())
    names, none = DynamicToolLoader(registry, {**pinned, "local_catalog": "names"}).offered_for_run(True)
    assert sorted(names) == ["bash", "memory", "vault_search"] and none == []
    every, _ = DynamicToolLoader(registry, pinned).offered_for_run(False)
    assert len(every) == len(registry.list_tools())          # a non-deferred (cloud) run: every schema


def test_the_default_catalog_and_a_typo(caplog):
    reg = ToolRegistry()
    assert DynamicToolLoader(reg, {}).catalog_mode == DEFAULT_CATALOG == "stubs"
    assert DynamicToolLoader(reg, {"local_catalog": "names"}).catalog_mode == "names"
    with caplog.at_level(logging.WARNING):
        assert DynamicToolLoader(reg, {"local_catalog": "stub"}).catalog_mode == "stubs"
    assert any("local_catalog" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------- through the loop


class _CityInput(BaseModel):
    city: str


class _Weather(BaseTool):
    name = "weather"
    description = "Weather for a city."
    input_model = _CityInput

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output=f"sunny in {arguments.city}")


class _Empty(BaseModel):
    pass


def _plain(n: str) -> BaseTool:
    class _T(BaseTool):
        name = n
        description = f"{n} tool"
        input_model = _Empty

        async def execute(self, arguments, context):  # noqa: ANN001
            return ToolResult(output="ok")

    return _T()


class _Script(ModelProvider):
    def __init__(self, calls: list[tuple[str, dict]]) -> None:
        self.calls, self.seen = list(calls), []

    async def stream_message(self, request):  # noqa: ANN001
        self.seen.append([t["name"] for t in request.tools])
        c = self.calls.pop(0) if self.calls else None
        content = [ToolUseBlock(id=f"t{len(self.seen)}", name=c[0], input=c[1])] if c else [TextBlock(text="done")]
        yield ApiMessageCompleteEvent(message=ConversationMessage(role="assistant", content=content),
                                      usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop")


def _run(calls: list[tuple[str, dict]], mode: str = "stubs") -> tuple[_Script, list[tuple[str, bool, str]]]:
    reg = ToolRegistry()
    for t in (_plain("bash"), _plain("read_file"), _Weather()):
        reg.register(t)
    loader = DynamicToolLoader(reg, {"enabled": "auto", "always_loaded": ["bash", "read_file"], "local_catalog": mode})
    prov = _Script(calls)
    ctx = LoopContext(provider=prov, model="m", system_prompt="", max_tokens=128, tool_registry=reg,
                      tool_loader=loader, adapter=ModelAdapter(tier="light"))

    async def go():
        out = []
        async for ev, _ in run_loop(ctx, [ConversationMessage.from_user_text("weather?")]):
            if isinstance(ev, ToolExecutionCompleted):
                out.append((ev.tool_name, ev.is_error, ev.output))
        return out

    return prov, asyncio.run(go())


def test_a_local_run_offers_every_tool_sorted():
    prov, _ = _run([])
    assert prov.seen[0] == ["bash", "read_file", "weather"]


def test_names_mode_keeps_the_old_catalog():
    prov, _ = _run([], mode="names")
    assert prov.seen[0] == ["bash", "read_file"] or "weather" not in prov.seen[0]


def test_a_stub_call_with_wrong_arguments_gets_the_schema_then_runs():
    _, results = _run([("weather", {}), ("weather", {"city": "Oslo"})])
    (name, err, out), (name2, err2, out2) = results
    assert name == "weather" and err and "Its full schema:" in out and '"city": "<city>"' in out
    assert "For further information" not in out          # pydantic's link is not sent
    assert (name2, err2, out2) == ("weather", False, "sunny in Oslo")


def test_refusals_stop_after_the_cap():
    # Three DIFFERENT wrong calls: an identical one is stopped earlier by the repeat guard.
    _, results = _run([("weather", {}), ("weather", {"city": None}), ("weather", {"city": ["x"]})])
    assert len(results) == MAX_SCHEMA_RETRIES + 1
    outs = [o for _, _, o in results]
    assert all("Its full schema:" in o for o in outs[:MAX_SCHEMA_RETRIES])
    assert "Not retried again" in outs[MAX_SCHEMA_RETRIES]
