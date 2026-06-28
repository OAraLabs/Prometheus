"""Shared fixtures + helpers for the force-search spec (tests/test_force_search.py).

Per the brief: fixtures (spy_provider / cloud_spy_provider / rest_client / ws_client) are
pytest fixtures injected as params; make_request / run_turn / run_turn_async / SEARCH_TOOL are
module-level helpers the spec imports as bare globals.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from prometheus.adapter.enforcer import StructuredOutputEnforcer
from prometheus.api.tool_choice import AUTO, NONE, REQUIRED, forced_tool_name, normalize_tool_choice
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

SEARCH_TOOL = "web_search"
_OTHER_TOOL = "noop_tool"

# Anthropic-format schemas the enforcer turns into the GBNF grammar (auto / required / specific).
_GRAMMAR_SCHEMAS = [
    {"name": SEARCH_TOOL, "input_schema": {"type": "object", "properties": {}}},
    {"name": _OTHER_TOOL, "input_schema": {"type": "object", "properties": {}}},
]


class _EmptyInput(BaseModel):
    pass


class _WebSearchStub(BaseTool):
    name = SEARCH_TOOL
    description = "stub web search"
    input_model = _EmptyInput

    async def execute(self, arguments, context):  # noqa: ANN001 — never called (run_turn breaks pre-dispatch)
        return ToolResult(output="(stub)", is_error=False)


class _NoopStub(BaseTool):
    name = _OTHER_TOOL
    description = "stub noop"
    input_model = _EmptyInput

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output="(stub)", is_error=False)


def _make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_WebSearchStub())
    reg.register(_NoopStub())
    return reg


class _Record:
    """What a spy exposes per call: the request's directive + the provider's per-call
    decisions (selected grammar for local; native tool_choice + synthetic prefill for cloud)."""

    def __init__(self, request, *, selected_grammar=None, native_tool_choice=None, synthetic_prefill=None):
        self.tool_choice = request.tool_choice
        self.tool_schema = list(request.tools)  # the offered tools (alias for assertions)
        self.suppress_tools = request.suppress_tools
        self.selected_grammar = selected_grammar
        self.native_tool_choice = native_tool_choice
        self.synthetic_prefill = synthetic_prefill


def _blocks(msg) -> list[dict]:
    """ConversationMessage content -> the {"type": ...} block dicts the spec inspects."""
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    out: list[dict] = []
    for b in content:
        cls = type(b).__name__
        if cls == "ToolUseBlock" or getattr(b, "type", None) == "tool_use":
            out.append({"type": "tool_use", "name": getattr(b, "name", None), "input": getattr(b, "input", {})})
        else:
            out.append({"type": "text", "text": getattr(b, "text", "")})
    return out


# ── Provider doubles ──────────────────────────────────────────────────────────


class SpyLlamaProvider(ModelProvider):
    """LlamaCppProvider double: records each per-call request, SELECTS a GBNF grammar from the
    immutable boot grammar per tool_choice (auto->boot identity, none->None, required->prose
    dropped, {tool}->single-alternative), and is directive-compliant when scripted."""

    def __init__(self) -> None:
        self._enforcer = StructuredOutputEnforcer()
        self.boot_grammar = self._enforcer.generate_grammar(_GRAMMAR_SCHEMAS)  # immutable "auto"
        self._grammar_cache: dict = {}
        self.requests: list[_Record] = []
        self._script: list[str] = []
        # A SINGLE shared LoopContext, like the daemon's — proves per-call selection.
        self.shared_loop_context = LoopContext(
            provider=self, model="stub", system_prompt="", max_tokens=256, tool_registry=_make_registry()
        )

    def script(self, items) -> None:
        self._script = list(items)

    def grammar_admits_tool(self, grammar, tool_name) -> bool:
        return self._enforcer.grammar_admits_tool(grammar, tool_name)

    @property
    def last_request(self) -> _Record:
        return self.requests[-1]

    @property
    def last_grammar(self):
        return self.requests[-1].selected_grammar

    def _select_grammar(self, tool_choice):
        if tool_choice == AUTO:
            return self.boot_grammar  # identity — the unmodified boot grammar
        if tool_choice == NONE:
            return None  # dropped (suppress_tools / chat)
        forced = forced_tool_name(tool_choice)
        key = REQUIRED if tool_choice == REQUIRED else f"tool:{forced}"
        if key not in self._grammar_cache:  # cached, immutable, per-directive
            if tool_choice == REQUIRED:
                self._grammar_cache[key] = self._enforcer.generate_grammar(_GRAMMAR_SCHEMAS, require_tool_use=True)
            else:
                self._grammar_cache[key] = self._enforcer.generate_grammar(_GRAMMAR_SCHEMAS, only_tool=forced)
        return self._grammar_cache[key]

    async def stream_message(self, request):  # noqa: ANN001
        tc = request.tool_choice
        self.requests.append(_Record(request, selected_grammar=self._select_grammar(tc)))
        # Directive-compliant: a scripted turn with tools allowed yields a tool_use; otherwise
        # (no script, or tools disallowed) a plain-text turn. run_turn breaks after this turn.
        item = self._script.pop(0) if self._script else None
        if item is not None and tc != NONE:
            tool_name = forced_tool_name(tc) or SEARCH_TOOL
            msg = ConversationMessage(role="assistant", content=[ToolUseBlock(name=tool_name, input={})])
        else:
            msg = ConversationMessage(role="assistant", content=[TextBlock(text="ok")])
        yield ApiMessageCompleteEvent(
            message=msg, usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop"
        )


class SpyCloudProvider(ModelProvider):
    """Cloud provider double: maps tool_choice -> native tool_choice param, with a synthetic
    tool_use prefill ONLY when native `required` is unsupported (the gated fallback)."""

    def __init__(self) -> None:
        self.requests: list[_Record] = []
        self.supports_required = True
        self.shared_loop_context = LoopContext(
            provider=self, model="stub", system_prompt="", max_tokens=256, tool_registry=_make_registry()
        )

    @property
    def last_request(self) -> _Record:
        return self.requests[-1]

    @property
    def last_tool_choice(self):
        return self.requests[-1].native_tool_choice

    async def stream_message(self, request):  # noqa: ANN001
        tc = request.tool_choice
        native = tc  # passthrough: auto/none/required/{tool} map straight to the native param
        prefill = None
        if tc == REQUIRED and not self.supports_required:
            prefill = {"type": "tool_use", "synthetic": True}  # gated fallback only
        self.requests.append(_Record(request, native_tool_choice=native, synthetic_prefill=prefill))
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text="ok")]),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


# ── Inbound payload + turn driver ──────────────────────────────────────────────


def make_request(*, mode=None, tool_choice=None, tools=None, text="hi") -> dict:
    """Build the INBOUND payload a WS/REST client would send. The single seam encoding the
    field representation (mode is sugar; tool_choice is the lever)."""
    payload: dict = {"session_id": "spec", "message": text}
    if mode is not None:
        payload["mode"] = mode
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if tools is not None:
        payload["tools"] = tools
    return payload


def _resolve(payload, valid_names):
    mode = payload.get("mode") or "agent"
    tc_raw = payload.get("tool_choice")
    tc = normalize_tool_choice(tc_raw, valid_names) if tc_raw is not None else None
    return mode, tc


async def run_turn_async(provider, payload) -> list[dict]:
    """Drive ONE turn through run_loop with `provider` injected (per-call tool_choice resolved
    the same way the send path does). Returns the assistant turn's content blocks — captured
    from the first AssistantTurnComplete, then we stop (before any tool dispatch)."""
    ctx = provider.shared_loop_context
    valid = {t.name for t in ctx.tool_registry.list_tools()}
    mode, tc = _resolve(payload, valid)
    msgs = [ConversationMessage.from_user_text(payload.get("message", "hi"))]
    blocks: list[dict] | None = None
    try:
        async for event, _usage in run_loop(ctx, msgs, mode=mode, tool_choice=tc):
            if type(event).__name__ == "AssistantTurnComplete":
                blocks = _blocks(event.message)
                break
    except Exception:
        pass
    return blocks if blocks is not None else []


def run_turn(provider, payload) -> list[dict]:
    return asyncio.run(run_turn_async(provider, payload))


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def spy_provider() -> SpyLlamaProvider:
    return SpyLlamaProvider()


@pytest.fixture
def cloud_spy_provider() -> SpyCloudProvider:
    return SpyCloudProvider()


class _BridgeDouble:
    """Minimal ws_bridge for the REST edge: exposes loop_context.tool_registry (for tool-name
    validation) + an inert dispatch."""

    def __init__(self) -> None:
        self.loop_context = SimpleNamespace(tool_registry=_make_registry())

    async def dispatch_user_message(self, session_id, content, client_msg_id=None, mode="agent", tool_choice=None):
        return None


class _RestClient:
    def __init__(self) -> None:
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from prometheus.web.server import create_app

        app = create_app({})
        app.state.ws_bridge = _BridgeDouble()
        self._tc = TestClient(app)

    async def send_chat(self, payload):
        r = self._tc.post("/api/chat/send", json=payload)
        return SimpleNamespace(status=r.status_code, body=r.json() if r.content else None)


class _WsClient:
    def __init__(self) -> None:
        from prometheus.web.ws_server import WebSocketBridge

        self._bridge = WebSocketBridge(
            loop_context=SimpleNamespace(tool_registry=_make_registry()),
            agent_state_ref={"state": "idle"},
            config={},
        )

    async def send_message(self, payload):
        captured: list[dict] = []

        class _FakeWs:
            async def send(self, data):
                captured.append(json.loads(data))

        raw = json.dumps({"type": "send_message", "payload": payload})
        await self._bridge._handle_client_message(_FakeWs(), raw)
        for frame in captured:
            if frame.get("type") == "error":
                return frame
        return captured[0] if captured else {}


@pytest.fixture
def rest_client() -> _RestClient:
    return _RestClient()


@pytest.fixture
def ws_client() -> _WsClient:
    return _WsClient()
