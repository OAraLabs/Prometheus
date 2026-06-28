"""Per-message Agent|Chat mode (Sprint B / Piece 2).

THE MERGE GATE: with NO mode passed, every chat turn is byte-identical to today's
always-agentic path (tools offered, grammar not suppressed). Also proves: chat mode offers
ZERO tools (empty schema + grammar suppressed → structurally tool-free at EVERY tier),
concurrent turns on the SHARED loop_context don't cross-talk (mode is a per-call param, not
shared state), and the send-path validation (absent → agent, malformed → clean 400).
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ApiMessageRequest, ModelProvider
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult


class _EmptyInput(BaseModel):
    pass


class _NoopTool(BaseTool):
    name = "noop"
    description = "no-op"
    input_model = _EmptyInput

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output="ok", is_error=False)


class _CapturingProvider(ModelProvider):
    """Records each ApiMessageRequest, then ends the turn with a plain-text reply
    (no tool_uses → run_loop returns after one round)."""

    def __init__(self) -> None:
        self.requests: list[ApiMessageRequest] = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text="ok")]),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


def _ctx(provider: _CapturingProvider) -> LoopContext:
    reg = ToolRegistry()
    reg.register(_NoopTool())
    return LoopContext(provider=provider, model="stub", system_prompt="", max_tokens=128, tool_registry=reg)


async def _drain(agen) -> None:
    async for _ in agen:
        pass


def test_merge_gate_default_no_mode_is_byte_identical_agent():
    # PRIMARY ACCEPTANCE: no mode passed → tools offered (non-empty) and grammar NOT
    # suppressed — identical to today's always-agentic request.
    prov = _CapturingProvider()
    asyncio.run(_drain(run_loop(_ctx(prov), [ConversationMessage.from_user_text("hi")])))
    req = prov.requests[0]
    assert req.tools, "default (no-mode) turn must still offer tools — byte-identical agent path"
    assert req.suppress_tools is False, "default turn must NOT suppress the grammar"


def test_chat_mode_offers_zero_tools_and_suppresses_grammar():
    prov = _CapturingProvider()
    asyncio.run(_drain(run_loop(_ctx(prov), [ConversationMessage.from_user_text("hi")], mode="chat")))
    req = prov.requests[0]
    assert req.tools == [], "chat mode must offer no tools (empty schema)"
    assert req.suppress_tools is True, "chat mode must suppress the tool-calling grammar (structural)"


def test_unknown_mode_falls_back_to_agent_never_silently_drops_tools():
    # run_loop treats ONLY 'chat' as no-tools; an unrecognized value is the safe agent default.
    prov = _CapturingProvider()
    asyncio.run(_drain(run_loop(_ctx(prov), [ConversationMessage.from_user_text("hi")], mode="bogus")))
    req = prov.requests[0]
    assert req.tools and req.suppress_tools is False


def test_concurrency_no_crosstalk_on_shared_context():
    # SAME shared LoopContext, two concurrent run_loops — one agent, one chat. Because mode is a
    # per-call PARAMETER (never stored on the shared context), the turns can't leak into each other.
    prov = _CapturingProvider()
    ctx = _ctx(prov)

    async def _both():
        await asyncio.gather(
            _drain(run_loop(ctx, [ConversationMessage.from_user_text("a")], mode="agent")),
            _drain(run_loop(ctx, [ConversationMessage.from_user_text("c")], mode="chat")),
        )

    asyncio.run(_both())
    assert len(prov.requests) == 2
    assert sorted(bool(r.tools) for r in prov.requests) == [False, True]
    assert sorted(r.suppress_tools for r in prov.requests) == [False, True]


def test_llama_cpp_grammar_kept_by_default_but_dropped_on_suppress_tools():
    from prometheus.providers.llama_cpp import LlamaCppProvider

    prov = LlamaCppProvider(grammar='root ::= "x"')
    base = dict(model="m", messages=[ConversationMessage.from_user_text("hi")], tools=[])
    # default (suppress_tools=False) + empty tools → grammar applied, exactly as today
    p_default = prov._build_request_payload(ApiMessageRequest(suppress_tools=False, **base))
    assert "grammar" in p_default, "default empty-tools request must still carry the GBNF grammar"
    # chat (suppress_tools=True) → grammar dropped → structurally tool-free at the local tier
    p_chat = prov._build_request_payload(ApiMessageRequest(suppress_tools=True, **base))
    assert "grammar" not in p_chat, "suppress_tools must drop the grammar (local-tier structural fix)"


# ── send-path validation (REST edge) ──
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402


def _client_capturing_mode():
    captured: dict = {}

    class _Bridge:
        async def dispatch_user_message(self, session_id, content, client_msg_id=None, mode="agent"):
            captured["mode"] = mode

    app = create_app({})
    app.state.ws_bridge = _Bridge()
    return TestClient(app), captured


def test_rest_absent_mode_dispatches_as_agent():
    c, captured = _client_capturing_mode()
    r = c.post("/api/chat/send", json={"session_id": "s", "message": "hi"})
    assert r.status_code == 200 and captured["mode"] == "agent"


def test_rest_chat_mode_threads_through():
    c, captured = _client_capturing_mode()
    r = c.post("/api/chat/send", json={"session_id": "s", "message": "hi", "mode": "chat"})
    assert r.status_code == 200 and captured["mode"] == "chat"


def test_rest_malformed_mode_is_400():
    c, _ = _client_capturing_mode()
    r = c.post("/api/chat/send", json={"session_id": "s", "message": "hi", "mode": "bogus"})
    assert r.status_code == 400
