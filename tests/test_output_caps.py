"""Each request's output cap comes from the model serving it, not one flat 4096.

``max_tokens=4096`` was hard-coded where the daemon and the CLI build their
loop, so every route sent 4096: qwen3.8-max (131,072 allowed), Claude Opus 5
(128K, with its default thinking spending the same budget) and the local 27B
alike. A long file in a write call did not fit, and the reply was cut inside
the call (#685). The per-route ``max_tokens`` key reached the provider and
was then ignored, because the request always carried 4096.

Pinned here: the resolution order (per-model override, the route's
``max_tokens``, the built-in table, the llama.cpp cap, the default), the clamp
to each model's documented maximum, the room a MEASURED local window keeps for
the conversation, the request a real turn sends, the cap the compactor plans
around, the truncation error naming the cap actually sent, and every entry
point passing the caps in. Ported from the unpushed feat/per-model-output-caps
branch onto 0.9.12.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import AgentLoop, LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage
from prometheus.providers.anthropic import AnthropicProvider
from prometheus.providers.llama_cpp import LlamaCppProvider
from prometheus.providers.openai_compat import OpenAICompatProvider
from prometheus.providers.registry import ProviderRegistry
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

SRC = Path(__file__).resolve().parents[1] / "src" / "prometheus"


def _caps(config: dict | None = None):  # noqa: ANN202
    from prometheus.providers.output_caps import OutputCaps

    return OutputCaps.from_config(config or {})


def _limits():  # noqa: ANN202
    from prometheus.providers.output_caps import LOCAL_OUTPUT_CAP, MODEL_OUTPUT_LIMITS

    return MODEL_OUTPUT_LIMITS, LOCAL_OUTPUT_CAP


# ── the resolver ─────────────────────────────────────────────────────────────

CLOUD = SimpleNamespace(provider_name="qwen", route_max_tokens=None)
LLAMA = SimpleNamespace(provider_name="llama_cpp", route_max_tokens=None)
OLLAMA = SimpleNamespace(provider_name="ollama", route_max_tokens=None)


@pytest.mark.parametrize("model,cap", [
    ("qwen3.8-max", 16384), ("qwen3.8-flash", 16384), ("grok-4.5", 16384),
    ("claude-haiku-4-5", 16384),
    # Thinking on by default and inside max_tokens: twice the room.
    ("claude-sonnet-5", 32768), ("claude-opus-5", 32768),
    ("deepseek-flash", 32768), ("gemini-2.5-pro", 32768),
])
def test_a_known_model_gets_its_cap(model, cap):
    assert _caps().resolve(model, provider=CLOUD, default=4096) == (cap, "table")


def test_a_llama_cpp_backend_gets_the_local_cap_whatever_its_model_is_called():
    _table, local = _limits()
    assert local == 8192
    for name in ("", "qwen3.8-27b", "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"):
        assert _caps().resolve(name, provider=LLAMA, default=4096) == (8192, "local")


def test_an_unknown_model_keeps_the_default_on_ollama_and_in_the_cloud():
    assert _caps().resolve("qwen3.5:9b", provider=OLLAMA, default=4096) == (4096, "default")
    assert _caps().resolve("some-new-model", provider=CLOUD, default=4096) == (4096, "default")


def test_the_order_is_override_then_route_then_table():
    caps = _caps({"context": {"model_overrides": {"qwen3.8-max": {"max_output_tokens": 20000}}}})
    routed = SimpleNamespace(provider_name="qwen", route_max_tokens=12000)
    assert caps.resolve("qwen3.8-max", provider=routed, default=4096) == (20000, "model_override")
    assert _caps().resolve("qwen3.8-max", provider=routed, default=4096) == (12000, "route")
    assert _caps().resolve("qwen3.8-max", provider=CLOUD, default=4096) == (16384, "table")


def test_no_setting_asks_for_more_than_the_model_allows(caplog):
    caps = _caps({"context": {"model_overrides": {"qwen3.8-max": {"max_output_tokens": 500000}}}})
    with caplog.at_level(logging.WARNING):
        assert caps.resolve("qwen3.8-max", provider=CLOUD, default=4096) == (
            131072, "model_override+clamped")
    assert "above the model's maximum" in caplog.text
    routed = SimpleNamespace(provider_name="gemini", route_max_tokens=200000)
    assert _caps().resolve("gemini-2.5-pro", provider=routed, default=4096) == (
        65536, "route+clamped")
    # No published maximum: an operator's number stands.
    caps = _caps({"context": {"model_overrides": {"grok-4.5": {"max_output_tokens": 200000}}}})
    assert caps.resolve("grok-4.5", provider=CLOUD, default=4096) == (200000, "model_override")


def test_every_table_cap_fits_every_documented_maximum():
    """A fallback served by another listed model never gets a cap it refuses."""
    table, _local = _limits()
    caps = [limit.cap for limit in table.values()]
    maxima = [limit.maximum for limit in table.values() if limit.maximum is not None]
    assert max(caps) <= min(maxima)


def test_overrides_are_read_beside_effective_limit_and_bad_values_are_skipped(caplog):
    config = {"context": {"model_overrides": {
        "claude-sonnet-4-5": {"effective_limit": 200000},     # a window only
        "qwen3.8-max": {"effective_limit": 1000000, "max_output_tokens": 24000},
        "glm-5.3": {"max_output_tokens": 0},
        "kimi-k2.6": {"max_output_tokens": "lots"},
        "gpt-6-astra": {"max_output_tokens": True},
    }}}
    with caplog.at_level(logging.WARNING):
        caps = _caps(config)
    assert dict(caps.model_overrides) == {"qwen3.8-max": 24000}
    assert caplog.text.count("not a positive integer") == 3


# ── the per-route max_tokens key is live ─────────────────────────────────────

def test_a_route_max_tokens_reaches_the_provider_and_is_absent_when_unset():
    with_key = ProviderRegistry.create({"provider": "qwen", "api_key": "test-key",
                                        "model": "qwen3.8-max", "max_tokens": 12000})
    without = ProviderRegistry.create({"provider": "qwen", "api_key": "test-key",
                                       "model": "qwen3.8-max"})
    assert with_key.route_max_tokens == 12000
    assert without.route_max_tokens is None


# ── what a real turn sends ───────────────────────────────────────────────────

def _serve(monkeypatch, bodies: list[str]) -> list[dict]:
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        body = bodies[min(len(seen), len(bodies)) - 1]
        return httpx.Response(200, content=body.encode(),
                              headers={"content-type": "text/event-stream"})

    real = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return seen


def _openai_sse(*, text: str = "", call: tuple[str, str, str] | None = None,
                finish: str = "stop") -> str:
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1790000000,
            "model": "served"}
    deltas: list[dict] = [{"role": "assistant", "content": text or None}]
    if call is not None:
        call_id, name, args = call
        deltas.append({"tool_calls": [{"index": 0, "id": call_id, "type": "function",
                                       "function": {"name": name, "arguments": args}}]})
    chunks = [{**base, "choices": [{"index": 0, "delta": d, "finish_reason": None}]}
              for d in deltas]
    chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def _anthropic_sse(text: str) -> str:
    events = [
        {"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant",
                                              "model": "served", "content": [],
                                              "usage": {"input_tokens": 1, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 3}},
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def _sent_max_tokens(provider, model: str, output_caps, *, tool=None) -> LoopContext:  # noqa: ANN001
    registry = ToolRegistry()
    if tool is not None:
        registry.register(tool)
    ctx = LoopContext(provider=provider, model=model, system_prompt="You are a test.",
                      max_tokens=4096, tool_registry=registry, output_caps=output_caps)

    async def drain() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("hi")]):
            pass

    asyncio.run(drain())
    return ctx


@pytest.mark.parametrize("model,expected", [("qwen3.8-max", 16384), ("some-new-model", 4096)])
def test_a_cloud_turn_asks_for_its_models_cap(monkeypatch, model, expected):
    seen = _serve(monkeypatch, [_openai_sse(text="ok")])
    provider = OpenAICompatProvider(base_url="http://unit.test/v1", api_key="test-key",
                                    model=model, provider_name="qwen")
    _sent_max_tokens(provider, model, _caps())
    assert [s["max_tokens"] for s in seen] == [expected]


def test_an_opus_5_turn_asks_for_room_for_its_thinking(monkeypatch):
    seen = _serve(monkeypatch, [_anthropic_sse("ok")])
    _sent_max_tokens(AnthropicProvider(api_key="test-key", base_url="http://unit.test/v1"),
                     "claude-opus-5", _caps())
    assert [s["max_tokens"] for s in seen] == [32768]


def test_a_llama_cpp_turn_asks_for_the_local_cap(monkeypatch):
    seen = _serve(monkeypatch, [_openai_sse(text="ok")])
    _sent_max_tokens(LlamaCppProvider(base_url="http://unit.test"), "local-27b", _caps())
    assert [s["max_tokens"] for s in seen] == [8192]


def test_a_route_max_tokens_reaches_the_request(monkeypatch):
    seen = _serve(monkeypatch, [_openai_sse(text="ok")])
    provider = ProviderRegistry.create({"provider": "qwen", "api_key": "test-key",
                                        "base_url": "http://unit.test/v1",
                                        "model": "qwen3.8-max", "max_tokens": 12000})
    _sent_max_tokens(provider, "qwen3.8-max", _caps())
    assert [s["max_tokens"] for s in seen] == [12000]


def test_a_context_without_caps_sends_its_own_max_tokens(monkeypatch):
    """Harnesses and tests that build a LoopContext themselves are unchanged."""
    seen = _serve(monkeypatch, [_openai_sse(text="ok")])
    provider = OpenAICompatProvider(base_url="http://unit.test/v1", api_key="test-key",
                                    model="qwen3.8-max", provider_name="qwen")
    _sent_max_tokens(provider, "qwen3.8-max", None)
    assert [s["max_tokens"] for s in seen] == [4096]


def test_a_fallback_request_asks_for_the_fallback_models_cap():
    from prometheus.engine.agent_loop import _output_cap

    primary = SimpleNamespace(provider_name="llama_cpp", route_max_tokens=None)
    fallback = SimpleNamespace(model="qwen3.8-max",
                               provider=SimpleNamespace(provider_name="qwen",
                                                        route_max_tokens=None))
    ctx = SimpleNamespace(output_caps=_caps(), max_tokens=4096, model="local-27b",
                          provider=primary, fallback=fallback, compactor=None)
    assert _output_cap(ctx, "local-27b") == 8192
    assert _output_cap(ctx, "qwen3.8-max") == 16384


class _WriteInput(BaseModel):
    path: str
    content: str


class _Write(BaseTool):
    name = "write"
    description = "write a file"
    input_model = _WriteInput

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output="written", is_error=False)


def test_a_cut_call_is_told_the_cap_that_was_actually_sent(tmp_path, monkeypatch):
    seen = _serve(monkeypatch, [
        _openai_sse(call=("call_1", "write", '{"path": "a.md", "content": "# A\\nline tw'),
                    finish="length"),
        _openai_sse(text="I will write it in parts."),
    ])
    provider = OpenAICompatProvider(base_url="http://unit.test/v1", api_key="test-key",
                                    model="qwen3.8-max", provider_name="qwen")
    registry = ToolRegistry()
    registry.register(_Write())
    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    ctx = LoopContext(provider=provider, model="qwen3.8-max", system_prompt="You are a test.",
                      max_tokens=4096, tool_registry=registry, telemetry=tel,
                      output_caps=_caps())

    async def drain() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("write a")]):
            pass

    asyncio.run(drain())
    tel.close()
    [result] = [m["content"] for m in seen[1]["messages"] if m.get("role") == "tool"]
    assert "(16384 tokens)" in result, result
    [detail] = [r[0] for r in sqlite3.connect(tmp_path / "telemetry.db").execute(
        "SELECT error_detail FROM tool_calls WHERE error_type = 'truncated_at_output_limit'")]
    assert "max_tokens=16384" in detail


# ── a measured local window keeps most of its room ───────────────────────────

@pytest.mark.parametrize("window,cap", [(32768, 8192), (16384, 4096), (8192, 2048), (2048, 1024)])
def test_a_measured_window_caps_the_reply_at_a_quarter_of_it(window, cap):
    assert _caps().resolve("local-27b", provider=LLAMA, default=4096,
                           window=window, window_measured=True)[0] == cap


def test_the_room_rule_applies_to_every_source_but_only_on_a_measured_window():
    caps = _caps({"context": {"model_overrides": {"qwen3.8-max": {"max_output_tokens": 20000}}}})
    assert caps.resolve("qwen3.8-max", provider=CLOUD, default=4096,
                        window=16384, window_measured=True) == (4096, "model_override+room")
    # A cloud window is a configured floor, not a measurement: no clamp.
    assert caps.resolve("qwen3.8-max", provider=CLOUD, default=4096,
                        window=16384, window_measured=False) == (20000, "model_override")
    # A window nobody knows (no compactor): no clamp.
    assert _caps().resolve("local-27b", provider=LLAMA, default=4096,
                           window=0, window_measured=True) == (8192, "local")


class _Compactor:
    """Reports a measured window and records the max_tokens it is asked to plan for."""

    def __init__(self, window: int) -> None:
        self.window = window
        self.planned: list[int] = []

    def limit_for(self, model, backend=None):  # noqa: ANN001
        return self.window

    def is_measured(self, model, backend=None):  # noqa: ANN001
        return True

    def estimate_total(self, system_prompt, messages, tools_chars=0):  # noqa: ANN001
        return 100

    async def apply(self, messages, *, max_tokens=None, **kw):  # noqa: ANN001
        self.planned.append(max_tokens)
        return messages


def test_a_small_local_window_sends_and_plans_around_a_quarter_of_it(monkeypatch):
    from prometheus.engine.agent_loop import _output_cap

    ctx = SimpleNamespace(output_caps=_caps(), max_tokens=4096, model="local-27b",
                          provider=LLAMA, fallback=None, backend=None, compactor=_Compactor(16384))
    assert _output_cap(ctx, "local-27b") == 4096
    ctx.compactor = _Compactor(65536)
    assert _output_cap(ctx, "local-27b") == 8192, "a large window still gets the local cap, no more"


def test_the_compactor_plans_around_the_cap_the_request_sends(monkeypatch):
    """#773's projected-overflow rule counts the request's max_tokens as room the
    next round needs: it must be the cap actually sent, not a flat 4096."""
    seen = _serve(monkeypatch, [_openai_sse(text="ok")])
    compactor = _Compactor(16384)
    ctx = LoopContext(provider=LlamaCppProvider(base_url="http://unit.test"), model="local-27b",
                      system_prompt="You are a test.", max_tokens=4096,
                      tool_registry=ToolRegistry(), output_caps=_caps(), compactor=compactor)

    async def drain() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("hi")]):
            pass

    asyncio.run(drain())
    assert [s["max_tokens"] for s in seen] == [4096], "a quarter of the 16384 window"
    assert compactor.planned == [4096]


# ── every entry point passes the caps in ─────────────────────────────────────

def _calls_missing(module: Path, callee: str, kwarg: str) -> list[int]:
    """Line numbers of the ``callee(...)`` calls in ``module`` without ``kwarg``."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    missing = []
    found = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if name == callee:
            found += 1
            if kwarg not in {kw.arg for kw in node.keywords}:
                missing.append(node.lineno)
    assert found, f"premise: {module.name} calls {callee}(...)"
    return missing


@pytest.mark.parametrize("module,callee", [
    ("daemon.py", "AgentLoop"), ("daemon.py", "LoopContext"),
    ("__main__.py", "LoopContext"), ("__main__.py", "CodingSession"),
])
def test_every_entry_point_passes_the_caps(module, callee):
    assert _calls_missing(SRC / module, callee, "output_caps") == []


def test_agent_loop_hands_the_caps_to_each_run(monkeypatch):
    seen = _serve(monkeypatch, [_openai_sse(text="ok")])
    provider = OpenAICompatProvider(base_url="http://unit.test/v1", api_key="test-key",
                                    model="qwen3.8-max", provider_name="qwen")
    loop = AgentLoop(provider=provider, model="qwen3.8-max", output_caps=_caps())

    asyncio.run(loop.run_async("You are a test.", "hi"))
    assert [s["max_tokens"] for s in seen] == [16384]
