"""Telemetry records the tools a round SENT (audit 2026-10-16, fix 7).

The lucky-guess marker used to recompute "was this tool advertised?" from the
loader and the adapter at call time. A run whose tools were chosen while it was
routed to a cloud model had been sent every tool; when the adapter changed
under it (a cloud→local fallback, or #462's advertise-before-routing), the
recomputation said "deferred" and the call was marked a lucky guess: 25 of the
28 recorded guesses were tools the round had offered. ``tool_sets`` likewise
hashed the run-start list once, whatever a round sent.

Pinned here, through ``run_loop``:

* a call to a tool the round offered writes no marker, even when the
  adapter the call runs under would have deferred it;
* a call to a tool the round did not offer writes one;
* a round that sends a set other than the turn's gets its own ``tool_sets``
  row, and its ``responses`` row names that set.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

from pydantic import BaseModel

from prometheus.context.dynamic_tools import DynamicToolLoader
from prometheus.engine.agent_loop import LoopContext, _offered_names, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.telemetry.writer import tool_set_row
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult


class _EmptyInput(BaseModel):
    pass


def _tool(tool_name: str) -> BaseTool:
    class _T(BaseTool):
        name = tool_name
        description = f"{tool_name} test tool"
        input_model = _EmptyInput

        async def execute(self, arguments, context):  # noqa: ANN001
            return ToolResult(output="ok", is_error=False)

    return _T()


class _Provider(ModelProvider):
    """Round 1 calls *tool* (first running *on_call*); round 2 answers in prose."""

    def __init__(self, tool: str, on_call=None) -> None:  # noqa: ANN001
        self.tool, self.on_call, self.calls = tool, on_call, 0
        self.sent: list[list[str]] = []

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        self.sent.append(sorted(t.get("name") or t.get("function", {}).get("name") for t in request.tools))
        if self.calls == 1:
            if self.on_call is not None:
                self.on_call()
            content = [ToolUseBlock(id="t1", name=self.tool, input={})]
        else:
            content = [TextBlock(text="done")]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop",
        )


def _run(tmp_path, *, tool: str, adapter, flip_deferral: bool = False):  # noqa: ANN001
    registry = ToolRegistry()
    for name in ("bash", "read_file", "image_generate"):
        registry.register(_tool(name))
    # local_catalog "names": the 0.9.12 catalog, whose always_loaded set this test is about (under the default "stubs" every tool is offered: tests/test_tool_stubs.py).
    loader = DynamicToolLoader(registry, {"enabled": "auto", "always_loaded": ["bash", "read_file"],
                                          "local_catalog": "names"})
    answered = {"yes": False}
    if flip_deferral:
        # Deferral resolves OFF when the run starts (every tool sent) and ON
        # once the model has answered: the recomputation the marker used to do
        # then disagrees with what the round sent, as it did after #462 and a
        # cloud->local fallback.
        def resolve(adapter=None):  # noqa: ANN001
            return (True, "later: on") if answered["yes"] else (False, "start: off")

        loader.resolve_deferred = resolve  # type: ignore[method-assign]
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)

    def on_call() -> None:
        answered["yes"] = True

    provider = _Provider(tool, on_call=on_call)
    ctx = LoopContext(provider=provider, model="stub-model", system_prompt="", max_tokens=128,
                      tool_registry=registry, tool_loader=loader, adapter=adapter, telemetry=tel)

    async def drain() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")], session_id="desktop:sent"):
            pass

    asyncio.run(drain())
    tel.close()
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    return con, provider


def _markers(con) -> list[str]:
    return [json.loads(r["summary_json"])["tool"] for r in con.execute(
        "SELECT summary_json FROM subsystem_runs WHERE subsystem = 'agent_loop' AND operation = 'lucky_guess'")]


def test_a_tool_the_round_offered_is_no_lucky_guess_whatever_a_recomputation_says(tmp_path):
    from prometheus.adapter import ModelAdapter

    con, provider = _run(tmp_path, tool="image_generate", adapter=ModelAdapter(tier="light"), flip_deferral=True)
    assert "image_generate" in provider.sent[0], "precondition: the round offered the tool"
    assert _markers(con) == []


def test_a_tool_the_round_did_not_offer_is_a_lucky_guess(tmp_path):
    from prometheus.adapter import ModelAdapter

    con, provider = _run(tmp_path, tool="image_generate", adapter=ModelAdapter(tier="light"))
    assert "image_generate" not in provider.sent[0], "precondition: the round deferred the tool"
    assert _markers(con) == ["image_generate"]


def test_a_round_that_sends_another_set_records_it(tmp_path):
    from prometheus.adapter import ModelAdapter

    class _Narrowing(ModelAdapter):
        """Formats the run's tools down to bash only, as a tier may."""

        def format_request(self, system_prompt, tools):  # noqa: ANN001
            return system_prompt, [t for t in tools if (t.get("name") or t.get("function", {}).get("name")) == "bash"]

    con, provider = _run(tmp_path, tool="bash", adapter=_Narrowing(tier="light"))
    assert provider.sent[0] == ["bash"], "precondition: the round sent only bash"
    sets = {r["tool_set_hash"]: json.loads(r["tool_names"]) for r in con.execute("SELECT * FROM tool_sets")}
    sent_hash = tool_set_row(["bash"])["tool_set_hash"]
    assert sets[sent_hash] == ["bash"]
    hashes = [r["tool_set_hash"] for r in con.execute("SELECT tool_set_hash FROM responses ORDER BY id")]
    assert hashes and all(h == sent_hash for h in hashes), (hashes, sets)


def test_offered_names():
    tools = [{"name": "bash"}, {"function": {"name": "grep"}}]
    assert _offered_names(tools, tools, True) == {"bash", "grep"}
    assert _offered_names(tools, [], True) == {"bash", "grep"}          # tier full: in the prompt
    assert _offered_names(tools, tools, False) == frozenset()          # a no-tools round
