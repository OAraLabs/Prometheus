"""A reply the output limit cuts off in the middle of a tool call never runs that call.

Large tool calls (a whole file in a write call's ``content``) arrived with
empty arguments. The reply hit ``max_tokens`` (4096 by default) partway through
the call's arguments, so the provider was left holding a JSON prefix that does
not parse, and every provider replaced it with ``{}`` without a word. The tool
then refused an empty call, and the model never learned why: it was told its
arguments were missing, not that it had run out of room, so it had no reason
to send anything smaller.

Pinned here, at both layers:

1. Each provider with the ``{}`` fallback (the shared OpenAI-shape parser that
   the compatible clouds, llama.cpp, Ollama and the stub use, and the Anthropic
   parser) names the cut call on ``truncated_tool_calls`` when the stream
   stopped at the output limit (``finish_reason: length``, ``stop_reason:
   max_tokens``) with that call's arguments unreadable. Only the call in
   progress at the cut: a complete call before it, a reply that ended normally,
   and a last call whose arguments did parse are not named.
2. ``run_loop`` does not run a named call. The model gets an error result for
   it that says the call was cut at the output limit and asks for smaller
   pieces; a complete call in the same reply still runs; and telemetry counts
   the cut call (``error_type = 'truncated_at_output_limit'``).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import httpx
import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage
from prometheus.providers.anthropic import AnthropicProvider
from prometheus.providers.base import ApiMessageCompleteEvent, ApiMessageRequest
from prometheus.providers.llama_cpp import LlamaCppProvider
from prometheus.providers.ollama import OllamaProvider
from prometheus.providers.openai_compat import OpenAICompatProvider
from prometheus.providers.stub import StubProvider
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

# The arguments of a write call, cut mid-string the way a length stop leaves
# them: a JSON prefix, streamed in fragments.
CUT_ARGS = ['{"path": "notes.md", ', '"content": "# Notes\\n\\nline one\\nline tw']
WHOLE_ARGS = ['{"q": ', '"gears"}']


# ── a server that answers each POST with the next scripted stream ───────────

def _serve(monkeypatch, bodies: list[str]) -> list[dict]:
    """Route every httpx.AsyncClient to a server answering the n-th chat POST
    with ``bodies[n]`` (the last one again past the end); returns the request
    bodies it received. Ollama's capability probe gets a 404 (unknown)."""
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/show":
            return httpx.Response(404, json={"error": "not found"})
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


def _openai_sse(calls: list[tuple[str, str, list[str]]], finish: str, *, text: str = "") -> str:
    """An OpenAI-shape stream: ``calls`` is (id, name, argument fragments) per
    call, in stream order; ``finish`` is the final chunk's finish_reason."""
    base: dict = {"id": "chatcmpl-1", "object": "chat.completion.chunk",
                  "created": 1790000000, "model": "served-model"}
    deltas: list[dict] = [{"role": "assistant", "content": text or None}]
    for index, (call_id, name, fragments) in enumerate(calls):
        deltas.append({"tool_calls": [{"index": index, "id": call_id, "type": "function",
                                       "function": {"name": name, "arguments": ""}}]})
        deltas += [{"tool_calls": [{"index": index, "function": {"arguments": f}}]}
                   for f in fragments]
    chunks = [{**base, "choices": [{"index": 0, "delta": d, "finish_reason": None}]}
              for d in deltas]
    chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    chunks.append({**base, "choices": [],
                   "usage": {"prompt_tokens": 40, "completion_tokens": 16, "total_tokens": 56}})
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def _openai_prose(text: str) -> str:
    return _openai_sse([], "stop", text=text)


def _anthropic_sse(calls: list[tuple[str, str, list[str]]], stop: str, *, text: str = "") -> str:
    """An Anthropic Messages stream: an optional text block, then one tool_use
    block per call; every block is closed, as the API closes them at a cut."""
    events: list[dict] = [{"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "served-model",
        "content": [], "stop_reason": None, "usage": {"input_tokens": 40, "output_tokens": 1}}}]
    index = 0
    if text:
        events += [
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": text}},
            {"type": "content_block_stop", "index": 0},
        ]
        index = 1
    for call_id, name, fragments in calls:
        events.append({"type": "content_block_start", "index": index, "content_block": {
            "type": "tool_use", "id": call_id, "name": name, "input": {}}})
        events += [{"type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": f}} for f in fragments]
        events.append({"type": "content_block_stop", "index": index})
        index += 1
    events.append({"type": "message_delta", "delta": {"stop_reason": stop},
                   "usage": {"output_tokens": 16}})
    events.append({"type": "message_stop"})
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def _anthropic_prose(text: str) -> str:
    return _anthropic_sse([], "end_turn", text=text)


# ── layer 1: each provider names the cut call ────────────────────────────────

_OPENAI_SHAPE = {
    "openai_compat": lambda: OpenAICompatProvider(
        base_url="http://unit.test/v1", api_key="test-key", model="m", provider_name="qwen"),
    "llama_cpp": lambda: LlamaCppProvider(base_url="http://unit.test"),
    "ollama": lambda: OllamaProvider(base_url="http://unit.test"),
    "stub": lambda: StubProvider(base_url="http://unit.test"),
}


async def _complete(provider) -> ApiMessageCompleteEvent:
    done = None
    request = ApiMessageRequest(model="m", messages=[ConversationMessage.from_user_text("q")],
                                max_tokens=16)
    async for ev in provider.stream_message(request):
        if isinstance(ev, ApiMessageCompleteEvent):
            done = ev
    assert done is not None, "the stream ended without a completion event"
    return done


@pytest.mark.parametrize("name", sorted(_OPENAI_SHAPE))
def test_an_openai_shape_provider_names_the_call_a_length_stop_cut(monkeypatch, name):
    _serve(monkeypatch, [_openai_sse(
        [("call_1", "lookup", WHOLE_ARGS), ("call_2", "write", CUT_ARGS)], "length")])
    done = asyncio.run(_complete(_OPENAI_SHAPE[name]()))
    assert done.stop_reason == "length"
    assert [t.id for t in done.message.tool_uses] == ["call_1", "call_2"]
    assert done.message.tool_uses[0].input == {"q": "gears"}
    # Nothing of the cut call's arguments can be read; the block keeps the
    # empty input, and the event says why.
    assert done.message.tool_uses[1].input == {}
    assert done.truncated_tool_calls == ("call_2",), (
        "the call the length stop cut must be named, and only that one"
    )


def test_the_anthropic_provider_names_the_call_a_max_tokens_stop_cut(monkeypatch):
    _serve(monkeypatch, [_anthropic_sse(
        [("toolu_1", "lookup", WHOLE_ARGS), ("toolu_2", "write", CUT_ARGS)], "max_tokens",
        text="Writing the notes now.")])
    done = asyncio.run(_complete(AnthropicProvider(api_key="test-key",
                                                   base_url="http://unit.test/v1")))
    assert done.stop_reason == "max_tokens"
    assert [t.id for t in done.message.tool_uses] == ["toolu_1", "toolu_2"]
    assert done.message.tool_uses[0].input == {"q": "gears"}
    assert done.message.tool_uses[1].input == {}
    assert done.truncated_tool_calls == ("toolu_2",)


@pytest.mark.parametrize("name", sorted(_OPENAI_SHAPE))
def test_a_cut_before_any_arguments_is_a_cut(monkeypatch, name):
    """Empty arguments are a complete no-argument call only when the reply
    ended normally; at a length stop the call never got its ``{}``."""
    _serve(monkeypatch, [_openai_sse([("call_1", "write", [])], "length")])
    done = asyncio.run(_complete(_OPENAI_SHAPE[name]()))
    assert done.truncated_tool_calls == ("call_1",)


def test_an_anthropic_cut_before_any_arguments_is_a_cut(monkeypatch):
    _serve(monkeypatch, [_anthropic_sse([("toolu_1", "write", [])], "max_tokens")])
    done = asyncio.run(_complete(AnthropicProvider(api_key="test-key",
                                                   base_url="http://unit.test/v1")))
    assert done.truncated_tool_calls == ("toolu_1",)


@pytest.mark.parametrize("name", sorted(_OPENAI_SHAPE))
def test_a_reply_that_ended_normally_names_nothing(monkeypatch, name):
    """Unreadable arguments in a reply the model ended itself are a malformed
    call, not a cut one; this change leaves that case as it was."""
    _serve(monkeypatch, [_openai_sse([("call_1", "write", CUT_ARGS)], "tool_calls")])
    done = asyncio.run(_complete(_OPENAI_SHAPE[name]()))
    assert done.truncated_tool_calls == ()


@pytest.mark.parametrize("name", sorted(_OPENAI_SHAPE))
def test_a_length_stop_after_the_last_call_closed_names_nothing(monkeypatch, name):
    _serve(monkeypatch, [_openai_sse([("call_1", "lookup", WHOLE_ARGS)], "length")])
    done = asyncio.run(_complete(_OPENAI_SHAPE[name]()))
    assert done.message.tool_uses[0].input == {"q": "gears"}
    assert done.truncated_tool_calls == ()


def test_an_anthropic_reply_that_ended_normally_names_nothing(monkeypatch):
    _serve(monkeypatch, [_anthropic_sse([("toolu_1", "lookup", WHOLE_ARGS)], "max_tokens")])
    done = asyncio.run(_complete(AnthropicProvider(api_key="test-key",
                                                   base_url="http://unit.test/v1")))
    assert done.truncated_tool_calls == ()
    _serve(monkeypatch, [_anthropic_sse([("toolu_1", "write", CUT_ARGS)], "tool_use")])
    done = asyncio.run(_complete(AnthropicProvider(api_key="test-key",
                                                   base_url="http://unit.test/v1")))
    assert done.truncated_tool_calls == ()


# ── layer 2: the loop never runs the cut call, and says why ──────────────────

class _LookupInput(BaseModel):
    q: str


class _WriteInput(BaseModel):
    path: str
    content: str


class _Lookup(BaseTool):
    name = "lookup"
    description = "look a count up"
    input_model = _LookupInput

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(self, arguments, context):  # noqa: ANN001
        self.calls.append(arguments.model_dump())
        return ToolResult(output="3", is_error=False)


class _Write(BaseTool):
    name = "write"
    description = "write a file"
    input_model = _WriteInput

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(self, arguments, context):  # noqa: ANN001
        self.calls.append(arguments.model_dump())
        return ToolResult(output="written", is_error=False)


def _turn(tmp_path, provider, model: str) -> tuple[_Lookup, _Write, list[sqlite3.Row]]:
    """One turn with ``lookup`` and ``write`` registered. Returns both tools
    and the tool rows telemetry wrote."""
    lookup, write = _Lookup(), _Write()
    registry = ToolRegistry()
    registry.register(lookup)
    registry.register(write)
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    ctx = LoopContext(provider=provider, model=model, system_prompt="You are a test.",
                      max_tokens=256, tool_registry=registry, telemetry=tel)

    async def drain() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("write my notes")],
                                session_id="desktop:truncated-call"):
            pass

    asyncio.run(drain())
    tel.close()
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT tool_name, success, error_type, error_detail, tool_use_id"
                       " FROM tool_calls WHERE tool_name != '_loop_transition'"
                       " ORDER BY id").fetchall()
    return lookup, write, rows


def _assert_says_cut(content: str) -> None:
    assert "output limit" in content, content
    assert "256" in content, "the result names the limit the reply hit"
    assert "not run" in content, content
    assert "smaller" in content, "the result asks for the content in smaller pieces"


def test_the_loop_answers_a_cut_call_with_a_truncation_error_and_never_runs_it(
        tmp_path, monkeypatch):
    seen = _serve(monkeypatch, [
        _openai_sse([("call_1", "lookup", WHOLE_ARGS), ("call_2", "write", CUT_ARGS)], "length"),
        _openai_prose("I will write it in parts."),
    ])
    provider = OpenAICompatProvider(base_url="http://unit.test/v1", api_key="test-key",
                                    model="qwen3.8-max", provider_name="qwen")
    lookup, write, rows = _turn(tmp_path, provider, "qwen3.8-max")

    assert write.calls == [], "a call the output limit cut must never run"
    assert lookup.calls == [{"q": "gears"}], "a complete call in the same reply still runs"

    assert len(seen) == 2, "the model is asked again, with the results"
    results = {m["tool_call_id"]: m["content"] for m in seen[1]["messages"]
               if m.get("role") == "tool"}
    assert results["call_1"] == "3"
    _assert_says_cut(results["call_2"])

    assert [(r["tool_name"], r["success"], r["error_type"], r["tool_use_id"]) for r in rows] == [
        ("lookup", 1, None, "call_1"),
        ("write", 0, "truncated_at_output_limit", "call_2"),
    ]
    assert "length" in rows[1]["error_detail"]


def test_the_loop_answers_an_anthropic_cut_call_the_same_way(tmp_path, monkeypatch):
    seen = _serve(monkeypatch, [
        _anthropic_sse([("toolu_2", "write", CUT_ARGS)], "max_tokens", text="Writing it."),
        _anthropic_prose("I will write it in parts."),
    ])
    provider = AnthropicProvider(api_key="test-key", base_url="http://unit.test/v1")
    _lookup, write, rows = _turn(tmp_path, provider, "claude-haiku-4-5")

    assert write.calls == []
    assert len(seen) == 2
    [result] = [b for m in seen[1]["messages"] if isinstance(m["content"], list)
                for b in m["content"] if b.get("type") == "tool_result"]
    assert result["tool_use_id"] == "toolu_2"
    assert result["is_error"] is True
    _assert_says_cut(result["content"])
    assert [(r["tool_name"], r["success"], r["error_type"]) for r in rows] == [
        ("write", 0, "truncated_at_output_limit"),
    ]
