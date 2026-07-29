"""Tier-aware deferred tool loading (feat/deferred-tools-tier-aware).

Context from the EMBERFALL baseline: all 49 tool schemas shipped every round
(~9.6k tokens, 60.7% of round 0) while the task used 3 of them — and the
existing deferred-loading machinery could not help, because run_loop called
``active_schemas()`` with no arguments, which can't see the adapter and so
can't resolve anything tier-aware.

Properties pinned here:

* tri-state parsing (true / false / "auto", default "auto", garbage → "auto")
* auto resolves per provider tier: cloud ("off") → full catalog, local
  (any real tier) → deferred; unknown provenance → full catalog
* explicit true/false beats auto in BOTH directions
* the run_loop call-site fix actually shrinks what is advertised on the wire
* the advertised set is FROZEN across rounds — even if config flips mid-run
  (the #120 prefix-mutation bug class)
* the per-run tool_advertisement telemetry row makes the A/B measurable
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from prometheus.context.dynamic_tools import DynamicToolLoader, _normalize_enabled
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
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


def _loader(registry, enabled, always=("bash", "read_file")) -> DynamicToolLoader:
    return DynamicToolLoader(
        registry, {"enabled": enabled, "always_loaded": list(always)}
    )


def _adapter(tier: str):
    return SimpleNamespace(tier=tier)


# ---------------------------------------------------------------------------
# Tri-state parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (True, True),
        (False, False),
        ("auto", "auto"),
        ("AUTO", "auto"),
        ("true", True),     # hand-edited yaml strings happen
        ("false", False),
        (None, "auto"),
        ("bananas", "auto"),  # typo must not silently pick a fixed state
        (3, "auto"),
    ],
)
def test_normalize_enabled(raw, expected):
    assert _normalize_enabled(raw) == expected


def test_missing_config_defaults_to_auto(registry):
    assert DynamicToolLoader(registry).configured_mode == "auto"


# ---------------------------------------------------------------------------
# Auto resolution per provider
# ---------------------------------------------------------------------------


def test_auto_disables_on_cloud(registry):
    enabled, source = _loader(registry, "auto").resolve_deferred(_adapter("off"))
    assert enabled is False
    assert "auto" in source and "cloud" in source


@pytest.mark.parametrize("tier", ["strict", "light", "full"])
def test_auto_enables_on_local_tiers(registry, tier):
    enabled, source = _loader(registry, "auto").resolve_deferred(_adapter(tier))
    assert enabled is True
    assert "auto" in source and "local" in source


def test_auto_without_adapter_advertises_everything(registry):
    # Unknown provenance: full schemas are the compatible status quo — an
    # optimization must not fire blind.
    enabled, source = _loader(registry, "auto").resolve_deferred(None)
    assert enabled is False
    assert "unknown" in source


# ---------------------------------------------------------------------------
# Explicit beats auto
# ---------------------------------------------------------------------------


def test_explicit_true_wins_even_on_cloud(registry):
    enabled, source = _loader(registry, True).resolve_deferred(_adapter("off"))
    assert enabled is True
    assert source == "explicitly enabled"


def test_explicit_false_wins_even_on_local(registry):
    enabled, source = _loader(registry, False).resolve_deferred(_adapter("strict"))
    assert enabled is False
    assert source == "explicitly disabled"


def test_set_configured_round_trips(registry):
    loader = _loader(registry, False)
    loader.set_configured("auto")
    assert loader.configured_mode == "auto"
    assert loader.resolve_deferred(_adapter("strict"))[0] is True
    loader.set_configured(False)
    assert loader.resolve_deferred(_adapter("strict"))[0] is False


def test_schemas_for_run_shapes(registry):
    loader = _loader(registry, "auto")
    assert {s["name"] for s in loader.schemas_for_run(True)} == {"bash", "read_file"}
    assert len(loader.schemas_for_run(False)) == 5


# ---------------------------------------------------------------------------
# The call-site fix, end to end through run_loop
# ---------------------------------------------------------------------------


class _CapturingProvider(ModelProvider):
    """Round 1: emits a tool call (forcing a second round). Round 2+: plain
    text (ending the turn). Records every request's tools list."""

    def __init__(self) -> None:
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        if len(self.requests) == 1:
            content = [ToolUseBlock(id="t1", name="bash", input={})]
        else:
            content = [TextBlock(text="done")]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


def _real_adapter(tier: str):
    """A REAL ModelAdapter (run_loop drives extract_tool_calls/format_request
    on it) — SimpleNamespace is only enough for resolve_deferred() itself."""
    from prometheus.adapter import ModelAdapter

    return ModelAdapter(tier=tier)


def _ctx(provider, registry, loader, adapter=None, telemetry=None) -> LoopContext:
    return LoopContext(
        provider=provider,
        model="stub",
        system_prompt="",
        max_tokens=128,
        tool_registry=registry,
        tool_loader=loader,
        adapter=adapter,
        telemetry=telemetry,
    )


async def _drain(agen) -> None:
    async for _ in agen:
        pass


def _run(ctx, text="go") -> None:
    asyncio.run(_drain(run_loop(ctx, [ConversationMessage.from_user_text(text)])))


def test_call_site_advertises_deferred_set_on_local(registry):
    provider = _CapturingProvider()
    ctx = _ctx(provider, registry, _loader(registry, "auto"), adapter=_real_adapter("strict"))
    _run(ctx)
    assert provider.requests, "no request captured"
    names = {t["name"] for t in provider.requests[0].tools}
    assert names == {"bash", "read_file"}, (
        "run_loop must advertise the deferred set — before this fix it called "
        "active_schemas() argument-less and shipped every registered tool"
    )


def test_call_site_advertises_full_catalog_on_cloud(registry):
    provider = _CapturingProvider()
    ctx = _ctx(provider, registry, _loader(registry, "auto"), adapter=_real_adapter("off"))
    _run(ctx)
    assert len(provider.requests[0].tools) == 5


def test_advertised_set_is_frozen_across_rounds_even_if_config_flips(registry):
    """#120 bug class: the catalog must be byte-stable for the whole run. The
    provider flips the config between rounds — the second round's request must
    still carry the round-0 set. (The flip lands on the NEXT run.)"""
    loader = _loader(registry, "auto")

    class _FlippingProvider(_CapturingProvider):
        async def stream_message(self, request):  # noqa: ANN001
            loader.set_configured(False if len(self.requests) == 0 else "auto")
            async for ev in super().stream_message(request):
                yield ev

    provider = _FlippingProvider()
    ctx = _ctx(provider, registry, loader, adapter=_real_adapter("strict"))
    _run(ctx)
    assert len(provider.requests) >= 2, "harness must produce a multi-round run"
    first = [t["name"] for t in provider.requests[0].tools]
    for i, req in enumerate(provider.requests[1:], start=2):
        assert [t["name"] for t in req.tools] == first, (
            f"round {i} advertised a different tool set than round 1 — "
            "mid-run catalog mutation invalidates the cached prompt prefix"
        )


def test_next_run_picks_up_the_flip(registry):
    loader = _loader(registry, "auto")
    p1 = _CapturingProvider()
    _run(_ctx(p1, registry, loader, adapter=_real_adapter("strict")))
    assert {t["name"] for t in p1.requests[0].tools} == {"bash", "read_file"}

    loader.set_configured(False)  # the Beacon toggle writes between runs
    p2 = _CapturingProvider()
    _run(_ctx(p2, registry, loader, adapter=_real_adapter("strict")))
    assert len(p2.requests[0].tools) == 5


# ---------------------------------------------------------------------------
# Telemetry: the A/B is measurable from the DB
# ---------------------------------------------------------------------------


class _RecordingTelemetry:
    def __init__(self) -> None:
        self.runs: list[dict] = []

    def record_run(self, **kw) -> None:
        self.runs.append(kw)

    def record(self, **kw) -> None:  # tool-call rows — not under test
        pass


def test_tool_advertisement_row_is_recorded(registry):
    tel = _RecordingTelemetry()
    provider = _CapturingProvider()
    ctx = _ctx(provider, registry, _loader(registry, "auto"),
               adapter=_real_adapter("strict"), telemetry=tel)
    _run(ctx)
    ads = [r for r in tel.runs if r.get("operation") == "tool_advertisement"]
    assert len(ads) == 1, "exactly one advertisement row per run"
    s = ads[0]["summary"]
    assert s["deferred_active"] is True
    assert s["advertised"] == 2
    assert s["registered_total"] == 5
    assert "auto" in s["source"]


def test_tool_advertisement_row_full_catalog_case(registry):
    tel = _RecordingTelemetry()
    provider = _CapturingProvider()
    ctx = _ctx(provider, registry, _loader(registry, "auto"),
               adapter=_real_adapter("off"), telemetry=tel)
    _run(ctx)
    s = [r for r in tel.runs if r.get("operation") == "tool_advertisement"][0]["summary"]
    assert s["deferred_active"] is False
    assert s["advertised"] == 5
