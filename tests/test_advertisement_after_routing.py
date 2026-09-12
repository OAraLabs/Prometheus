"""#462 + #458 — the advertisement must describe the model actually called,
filed under the session the turn actually ran as.

#462: the deferral decision and the profile filter read ``context.adapter`` /
``context.model``, and the ModelRouter REASSIGNS both when a session routes to
a named backend. Resolving the advertisement before routing shipped the cloud
model's "deferral disabled" catalog into a local 32k window — a 34,020-token
request that 400'd before any inference (two-message turn, no history to
compact). These tests pin the ordering by behaviour: a context that BOOTS
cloud-tier but ROUTES to a local backend must advertise the DEFERRED set, and
the telemetry row must name the post-routing model.

#458: the advertisement row was keyed on ``context.session_id`` — the shared
web context's routing namespace, not this turn's session. A row under the
wrong key is indistinguishable from a row never written: querying
subsystem_runs for the failed turn returned the failure and no advertisement
row at all. Pinned here: the row carries the run-scoped session id.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from prometheus.context.dynamic_tools import DynamicToolLoader
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
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


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    for n in ("bash", "read_file", "write_file", "image_generate", "tts"):
        reg.register(_tool(n))
    return reg


def _adapter(tier: str):
    from prometheus.adapter import ModelAdapter

    return ModelAdapter(tier=tier)


class _QuietProvider(ModelProvider):
    """One plain-text reply; records what was advertised."""

    def __init__(self) -> None:
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant", content=[TextBlock(text="ok")]
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


class _RecordingTelemetry:
    def __init__(self) -> None:
        self.runs: list[dict] = []

    def record_run(self, **kw) -> None:
        self.runs.append(kw)

    def record(self, **kw) -> None:
        pass


def _router_to_local(registry) -> object:
    """A router whose decision swaps in a LOCAL-tier adapter and model — the
    /4090-style named-backend override that triggered the live 400."""

    class _Decision:
        provider = None
        adapter = _adapter("strict")  # local tier → deferral engages
        model_name = "Qwen3.8-27B.gguf"
        provider_name = "llama_cpp"
        reason = "user_override"
        backend = "4090"

    return SimpleNamespace(
        route=lambda message, context=None: _Decision(),
        get_override_for_session=lambda sid: None,
    )


def _ctx(registry, *, ctx_session, router, telemetry, adapter):
    loader = DynamicToolLoader(
        registry, {"enabled": "auto", "always_loaded": ["bash", "read_file"]}
    )
    return LoopContext(
        provider=_QuietProvider(),
        model="claude-opus-4-6",          # boots as a CLOUD model
        system_prompt="- Model: claude-opus-4-6",
        max_tokens=128,
        tool_registry=registry,
        tool_loader=loader,
        adapter=adapter,                   # boots as a CLOUD adapter (tier off)
        telemetry=telemetry,
        session_id=ctx_session,            # shared web namespace
        model_router=router,
    )


async def _drain(ctx, *, turn_session):
    async for _ in run_loop(
        ctx,
        [ConversationMessage.from_user_text("go")],
        session_id=turn_session,
    ):
        pass


def test_routed_local_turn_advertises_the_deferred_set(registry):
    """#462, the live failure: boot cloud (tier off → full catalog), route to
    a local backend (tier strict → deferral). Before the fix the advertisement
    was resolved BEFORE routing, so all 5 schemas shipped into the small window
    and the row said 'auto → disabled (cloud provider)' while a local model
    was called. After the fix the catalog must be the DEFERRED set."""
    tel = _RecordingTelemetry()
    ctx = _ctx(
        registry,
        ctx_session="web",
        router=_router_to_local(registry),
        telemetry=tel,
        adapter=_adapter("off"),  # cloud tier at boot
    )
    asyncio.run(_drain(ctx, turn_session="telegram:42"))

    advertised = {t["name"] for t in ctx.provider.requests[0].tools}
    assert advertised == {"bash", "read_file"}, (
        f"the routed-local turn advertised {sorted(advertised)} — the "
        "advertisement must be resolved AFTER the router swapped in the local "
        "adapter, or the deferral decision describes the wrong model"
    )


def test_advertisement_row_names_the_post_routing_model(registry):
    """#462, telemetry half: the row records the model ACTUALLY CALLED, not
    the one the context booted with."""
    tel = _RecordingTelemetry()
    ctx = _ctx(
        registry,
        ctx_session="web",
        router=_router_to_local(registry),
        telemetry=tel,
        adapter=_adapter("off"),
    )
    asyncio.run(_drain(ctx, turn_session="telegram:42"))

    rows = [r for r in tel.runs if r.get("operation") == "tool_advertisement"]
    assert len(rows) == 1
    assert rows[0]["model"] == "Qwen3.8-27B.gguf", (
        "the row must describe the routed model — a pre-routing write records "
        "the cloud model that was never called"
    )
    assert rows[0]["summary"]["deferred_active"] is True
    assert rows[0]["summary"]["advertised"] == 2


def test_advertisement_row_is_filed_under_the_turn_session(registry):
    """#458: keyed by effective_session_id, not the shared context's. On the
    web path context.session_id is the routing namespace ('web'); a row filed
    there is unqueryable for the turn it describes."""
    tel = _RecordingTelemetry()
    ctx = _ctx(
        registry,
        ctx_session="web",
        router=None,
        telemetry=tel,
        adapter=_adapter("strict"),
    )
    asyncio.run(_drain(ctx, turn_session="telegram:42"))

    rows = [r for r in tel.runs if r.get("operation") == "tool_advertisement"]
    assert len(rows) == 1
    assert rows[0]["session_id"] == "telegram:42", (
        f"row was filed under {rows[0]['session_id']!r} — the turn ran as "
        "'telegram:42'; a row under the wrong key reads as never written"
    )


def test_unrouted_turn_behaves_as_before(registry):
    """Control: with no router the advertisement is unchanged by the move —
    the pre-existing local-boot behaviour (deferred on strict tier) holds."""
    tel = _RecordingTelemetry()
    ctx = _ctx(
        registry,
        ctx_session="telegram:42",
        router=None,
        telemetry=tel,
        adapter=_adapter("strict"),
    )
    asyncio.run(_drain(ctx, turn_session="telegram:42"))

    advertised = {t["name"] for t in ctx.provider.requests[0].tools}
    assert advertised == {"bash", "read_file"}
    rows = [r for r in tel.runs if r.get("operation") == "tool_advertisement"]
    assert rows[0]["session_id"] == "telegram:42"
    assert rows[0]["summary"]["deferred_active"] is True
