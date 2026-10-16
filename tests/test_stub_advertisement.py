"""What a run offers under local_catalog "stubs", the default (audit 2026-10-16, fix 3).

The names-mode versions of these guards stay, pinned to ``local_catalog:
"names"``: tests/test_advertisement_after_routing.py,
tests/test_deferred_tier_aware.py, tests/test_lucky_guess_marker.py,
tests/test_fallback_attribution.py, tests/test_telemetry_records_what_was_sent.py.
Here, the same guards for the default catalog:

* a local turn offers every tool: full schemas for the ambient set, stubs for
  the rest, sorted; a cloud turn offers every tool with its full schema;
* the catalog is decided AFTER routing (#462): a context that boots cloud and
  routes to a local backend gets the local catalog, and the telemetry row names
  the routed model and how many tools went as stubs (#458: under the turn's
  session);
* a config flip lands on the NEXT run, and the catalog is frozen across rounds;
* a local round never records a lucky guess: every tool is offered.
"""

from __future__ import annotations

import asyncio
import sqlite3
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from prometheus.context.dynamic_tools import DynamicToolLoader
from prometheus.context.tool_stubs import FULL_SCHEMA_TOOLS, full, skeleton
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

NAMES = ("bash", "read_file", "write_file", "image_generate", "tts")
AMBIENT = {n for n in NAMES if n in FULL_SCHEMA_TOOLS}        # bash, read_file, write_file
STUBBED = set(NAMES) - AMBIENT                                # image_generate, tts


class _In(BaseModel):
    text: str = ""


def _tool(tool_name: str) -> BaseTool:
    class _T(BaseTool):
        name = tool_name
        description = f"{tool_name} test tool. Longer explanation that a stub leaves out."
        input_model = _In

        async def execute(self, arguments, context):  # noqa: ANN001
            return ToolResult(output="ok", is_error=False)

    return _T()


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    for n in NAMES:
        reg.register(_tool(n))
    return reg


def _adapter(tier: str):
    from prometheus.adapter import ModelAdapter

    return ModelAdapter(tier=tier)


class _Provider(ModelProvider):
    """Round 1 calls *first_call* (when given), then answers. Records every request."""

    def __init__(self, first_call: str | None = None, on_round=None) -> None:  # noqa: ANN001
        self.first_call, self.on_round, self.requests = first_call, on_round, []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        if self.on_round is not None:
            self.on_round(len(self.requests))
        if self.first_call and len(self.requests) == 1:
            content = [ToolUseBlock(id="t1", name=self.first_call, input={})]
        else:
            content = [TextBlock(text="done")]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop",
        )


class _Telemetry:
    def __init__(self) -> None:
        self.runs: list[dict] = []

    def record_run(self, **kw) -> None:
        self.runs.append(kw)

    def record(self, **kw) -> None:
        pass

    def row(self) -> dict:
        (r,) = [r for r in self.runs if r.get("operation") == "tool_advertisement"]
        return r


def _router_to_local():
    class _Decision:
        provider = None
        adapter = _adapter("strict")
        model_name = "Qwen3.8-27B.gguf"
        provider_name = "llama_cpp"
        reason = "user_override"
        backend = "4090"

    return SimpleNamespace(route=lambda message, context=None: _Decision(),
                           get_override_for_session=lambda sid: None)


def _ctx(registry, provider, *, adapter, router=None, telemetry=None, loader=None, model="stub"):
    return LoopContext(
        provider=provider, model=model, system_prompt=f"- Model: {model}", max_tokens=128,
        tool_registry=registry,
        tool_loader=loader or DynamicToolLoader(registry, {"enabled": "auto",
                                                           "always_loaded": ["bash", "read_file"]}),
        adapter=adapter, telemetry=telemetry, session_id="web", model_router=router,
    )


def _run(ctx, session: str = "telegram:42") -> None:
    async def drain() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")], session_id=session):
            pass

    asyncio.run(drain())


def _by_name(request) -> dict:
    return {t["name"]: t for t in request.tools}


def _assert_local_catalog(request, registry) -> None:
    tools = _by_name(request)
    assert [t["name"] for t in request.tools] == sorted(NAMES), "every tool, sorted"
    for name in AMBIENT:
        assert tools[name] == full(registry.get(name).to_api_schema()), f"{name} must carry its full schema"
    for name in STUBBED:
        assert tools[name] == skeleton(registry.get(name).to_api_schema()), f"{name} must be a stub"


# ---------------------------------------------------------------- local and cloud


def test_a_local_turn_offers_full_schemas_for_the_ambient_set_and_stubs_for_the_rest(registry):
    """Stubs-mode version of test_call_site_advertises_deferred_set_on_local and
    test_unrouted_turn_behaves_as_before."""
    tel = _Telemetry()
    ctx = _ctx(registry, _Provider(), adapter=_adapter("strict"), telemetry=tel)
    _run(ctx)
    _assert_local_catalog(ctx.provider.requests[0], registry)
    s = tel.row()["summary"]
    assert (s["deferred_active"], s["advertised"], s["stubs"], s["catalog"]) == (True, 5, 2, "stubs")
    assert tel.row()["session_id"] == "telegram:42"


def test_a_cloud_turn_offers_every_full_schema(registry):
    """Stubs-mode version of test_call_site_advertises_full_catalog_on_cloud and
    test_tool_advertisement_row_full_catalog_case: no stub on a cloud turn."""
    tel = _Telemetry()
    ctx = _ctx(registry, _Provider(), adapter=_adapter("off"), telemetry=tel)
    _run(ctx)
    sent = ctx.provider.requests[0].tools
    assert {t["name"] for t in sent} == set(NAMES)
    assert all(t["input_schema"] == registry.get(t["name"]).to_api_schema()["input_schema"] for t in sent)
    s = tel.row()["summary"]
    assert (s["deferred_active"], s["advertised"], s["stubs"], s["catalog"]) == (False, 5, 0, "full")


# ---------------------------------------------------------------- after routing (#462, #458)


def test_the_catalog_is_decided_after_routing(registry):
    """Stubs-mode version of test_routed_local_turn_advertises_the_deferred_set:
    boot cloud (tier off), route to a local backend: the local catalog."""
    ctx = _ctx(registry, _Provider(), adapter=_adapter("off"), router=_router_to_local(),
               model="claude-opus-4-6")
    _run(ctx)
    _assert_local_catalog(ctx.provider.requests[0], registry)


def test_the_row_names_the_routed_model_and_its_stubs(registry):
    """Stubs-mode version of test_advertisement_row_names_the_post_routing_model."""
    tel = _Telemetry()
    ctx = _ctx(registry, _Provider(), adapter=_adapter("off"), router=_router_to_local(),
               telemetry=tel, model="claude-opus-4-6")
    _run(ctx)
    row = tel.row()
    assert row["model"] == "Qwen3.8-27B.gguf"
    assert row["session_id"] == "telegram:42"
    assert (row["summary"]["deferred_active"], row["summary"]["stubs"]) == (True, 2)


# ---------------------------------------------------------------- frozen per run


def test_the_next_run_picks_up_a_flip(registry):
    """Stubs-mode version of test_next_run_picks_up_the_flip: stubs, then (with
    deferral switched off between runs) every full schema."""
    loader = DynamicToolLoader(registry, {"enabled": "auto"})
    p1 = _Provider()
    _run(_ctx(registry, p1, adapter=_adapter("strict"), loader=loader))
    _assert_local_catalog(p1.requests[0], registry)
    loader.set_configured(False)
    p2 = _Provider()
    _run(_ctx(registry, p2, adapter=_adapter("strict"), loader=loader))
    assert all(t["input_schema"] == registry.get(t["name"]).to_api_schema()["input_schema"]
               for t in p2.requests[0].tools)


def test_the_catalog_is_frozen_across_rounds_even_if_config_flips(registry):
    """Stubs-mode version of test_advertised_set_is_frozen_across_rounds_even_if_config_flips."""
    loader = DynamicToolLoader(registry, {"enabled": "auto"})
    provider = _Provider(first_call="bash", on_round=lambda n: loader.set_configured(False if n == 1 else "auto"))
    _run(_ctx(registry, provider, adapter=_adapter("strict"), loader=loader))
    assert len(provider.requests) >= 2
    assert all(r.tools == provider.requests[0].tools for r in provider.requests[1:])


# ---------------------------------------------------------------- lucky guesses


def test_a_local_round_never_records_a_lucky_guess(registry, tmp_path):
    """Stubs-mode counterpart of the lucky-guess tests (pinned to names mode):
    every tool is offered, so a call to one that is not ambient (a stub) is an
    offered tool, not a guess."""
    from prometheus.telemetry.tracker import ToolCallTelemetry

    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    ctx = _ctx(registry, _Provider(first_call="image_generate"), adapter=_adapter("strict"), telemetry=tel)
    _run(ctx, session="desktop:stubs")
    tel.close()
    con = sqlite3.connect(tmp_path / "telemetry.db")
    calls = con.execute("SELECT tool_name FROM tool_calls WHERE tool_name = 'image_generate'").fetchall()
    guesses = con.execute(
        "SELECT COUNT(*) FROM subsystem_runs WHERE subsystem = 'agent_loop' AND operation = 'lucky_guess'"
    ).fetchone()[0]
    assert calls, "precondition: the stubbed tool was called"
    assert guesses == 0


# ---------------------------------------------------------------- what a round sent


def test_a_round_that_sends_another_set_records_it(registry, tmp_path):
    """Stubs-mode version of the test of the same name in
    tests/test_telemetry_records_what_was_sent.py: an adapter that narrows the
    catalog, and the round's tool_sets row names what it sent."""
    import json

    from prometheus.adapter import ModelAdapter
    from prometheus.telemetry.tracker import ToolCallTelemetry
    from prometheus.telemetry.writer import tool_set_row

    class _Narrowing(ModelAdapter):
        def format_request(self, system_prompt, tools):  # noqa: ANN001
            return system_prompt, [t for t in tools if t.get("name") == "bash"]

    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    provider = _Provider(first_call="bash")
    _run(_ctx(registry, provider, adapter=_Narrowing(tier="strict"), telemetry=tel), session="desktop:sent")
    tel.close()
    con = sqlite3.connect(tmp_path / "telemetry.db")
    con.row_factory = sqlite3.Row
    assert [t["name"] for t in provider.requests[0].tools] == ["bash"]
    sent = tool_set_row(["bash"])["tool_set_hash"]
    sets = {r["tool_set_hash"]: json.loads(r["tool_names"]) for r in con.execute("SELECT * FROM tool_sets")}
    assert sets[sent] == ["bash"]
    hashes = [r["tool_set_hash"] for r in con.execute("SELECT tool_set_hash FROM responses ORDER BY id")]
    assert hashes and all(h == sent for h in hashes), (hashes, sets)
