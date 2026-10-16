"""A tool call whose arguments cannot be read never runs; the model is told why.

Ported from #685 onto 0.9.12, with two cases that PR did not cover. On
2026-10-15 a qwen3.8-max turn on the live 0.9.10 daemon wrote a report into
``write_file``: five replies in a row were cut at 4096 output tokens, but
DashScope reported ``finish_reason: tool_calls``, not ``length``, so a check on
the stop reason alone sees nothing. Output tokens at the cap now count as a
stop at the limit. And unreadable arguments in a reply that ended normally are
not run on ``{}`` either: the model is told they were not valid JSON.

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
   parser) names every call whose arguments do not read as a JSON object on
   ``unreadable_tool_calls``, and the last call when the output limit stopped
   the reply before any of its arguments arrived. A complete call is not named.
2. ``run_loop`` does not run a named call. When the reply ended at the output
   limit (by its stop reason, or by output tokens at the cap) the result says
   the call was cut off and asks for smaller pieces; otherwise it says the
   arguments were not valid JSON. A complete call in the same reply still
   runs, telemetry counts the refused call, and the repeat guard never sees its
   ``{}`` (two cut calls in a row are two cut calls, not one "BLOCKED" repeat).
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


def _openai_sse(calls: list[tuple[str, str, list[str]]], finish: str, *, text: str = "",
                completion_tokens: int = 16) -> str:
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
                   "usage": {"prompt_tokens": 40, "completion_tokens": completion_tokens,
                             "total_tokens": 40 + completion_tokens}})
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
    assert done.unreadable_tool_calls == ("call_2",), (
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
    assert done.unreadable_tool_calls == ("toolu_2",)


@pytest.mark.parametrize("name", sorted(_OPENAI_SHAPE))
def test_a_cut_before_any_arguments_is_a_cut(monkeypatch, name):
    """Empty arguments are a complete no-argument call only when the reply
    ended normally; at a length stop the call never got its ``{}``."""
    _serve(monkeypatch, [_openai_sse([("call_1", "write", [])], "length")])
    done = asyncio.run(_complete(_OPENAI_SHAPE[name]()))
    assert done.unreadable_tool_calls == ("call_1",)


def test_an_anthropic_cut_before_any_arguments_is_a_cut(monkeypatch):
    _serve(monkeypatch, [_anthropic_sse([("toolu_1", "write", [])], "max_tokens")])
    done = asyncio.run(_complete(AnthropicProvider(api_key="test-key",
                                                   base_url="http://unit.test/v1")))
    assert done.unreadable_tool_calls == ("toolu_1",)


@pytest.mark.parametrize("name", sorted(_OPENAI_SHAPE))
def test_unreadable_arguments_are_named_whatever_the_stop(monkeypatch, name):
    """The qwen3.8-max case: cut at the cap, but the server said tool_calls."""
    _serve(monkeypatch, [_openai_sse([("call_1", "write", CUT_ARGS)], "tool_calls")])
    done = asyncio.run(_complete(_OPENAI_SHAPE[name]()))
    assert done.stop_reason == "tool_calls"
    assert done.message.tool_uses[0].input == {}
    assert done.unreadable_tool_calls == ("call_1",)


@pytest.mark.parametrize("name", sorted(_OPENAI_SHAPE))
def test_a_no_argument_call_in_a_reply_that_ended_normally_names_nothing(monkeypatch, name):
    _serve(monkeypatch, [_openai_sse([("call_1", "lookup", [])], "tool_calls")])
    done = asyncio.run(_complete(_OPENAI_SHAPE[name]()))
    assert done.unreadable_tool_calls == ()


@pytest.mark.parametrize("name", sorted(_OPENAI_SHAPE))
def test_a_length_stop_after_the_last_call_closed_names_nothing(monkeypatch, name):
    _serve(monkeypatch, [_openai_sse([("call_1", "lookup", WHOLE_ARGS)], "length")])
    done = asyncio.run(_complete(_OPENAI_SHAPE[name]()))
    assert done.message.tool_uses[0].input == {"q": "gears"}
    assert done.unreadable_tool_calls == ()


def test_an_anthropic_reply_that_ended_normally_names_nothing(monkeypatch):
    _serve(monkeypatch, [_anthropic_sse([("toolu_1", "lookup", WHOLE_ARGS)], "max_tokens")])
    done = asyncio.run(_complete(AnthropicProvider(api_key="test-key",
                                                   base_url="http://unit.test/v1")))
    assert done.unreadable_tool_calls == ()
    _serve(monkeypatch, [_anthropic_sse([("toolu_1", "lookup", [])], "tool_use")])
    done = asyncio.run(_complete(AnthropicProvider(api_key="test-key",
                                                   base_url="http://unit.test/v1")))
    assert done.unreadable_tool_calls == ()


def test_anthropic_unreadable_arguments_are_named_whatever_the_stop(monkeypatch):
    _serve(monkeypatch, [_anthropic_sse([("toolu_1", "write", CUT_ARGS)], "tool_use")])
    done = asyncio.run(_complete(AnthropicProvider(api_key="test-key",
                                                   base_url="http://unit.test/v1")))
    assert done.unreadable_tool_calls == ("toolu_1",)


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

    by_call = {r["tool_use_id"]: r for r in rows}
    assert sorted(by_call) == ["call_1", "call_2"], "one row per call, the cut one included"
    assert (by_call["call_1"]["tool_name"], by_call["call_1"]["success"]) == ("lookup", 1)
    cut = by_call["call_2"]
    assert (cut["tool_name"], cut["success"], cut["error_type"]) == (
        "write", 0, "truncated_at_output_limit")
    assert "stop_reason=length" in cut["error_detail"]
    assert "max_tokens=256" in cut["error_detail"]


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


# ── the two cases #685 missed ────────────────────────────────────────────────

def test_output_tokens_at_the_cap_count_as_a_cut_when_the_server_says_tool_calls(
        tmp_path, monkeypatch):
    """2026-10-15, live: qwen3.8-max, five write_file replies of 4111-4225 output
    tokens at a 4096 cap, every one with finish_reason tool_calls."""
    seen = _serve(monkeypatch, [
        _openai_sse([("call_1", "write", CUT_ARGS)], "tool_calls", completion_tokens=300),
        _openai_sse([("call_2", "write", CUT_ARGS)], "tool_calls", completion_tokens=300),
        _openai_prose("I will write it in parts."),
    ])
    provider = OpenAICompatProvider(base_url="http://unit.test/v1", api_key="test-key",
                                    model="qwen3.8-max", provider_name="qwen")
    _lookup, write, rows = _turn(tmp_path, provider, "qwen3.8-max")

    assert write.calls == []
    assert len(seen) == 3
    for request, call_id in ((seen[1], "call_1"), (seen[2], "call_2")):
        results = {m["tool_call_id"]: m["content"] for m in request["messages"]
                   if m.get("role") == "tool"}
        _assert_says_cut(results[call_id])
        assert "Field required" not in results[call_id]
        assert "BLOCKED" not in results[call_id], "a second cut call is not a repeat of the first {}"
    assert [(r["error_type"], r["error_detail"]) for r in rows] == [
        ("truncated_at_output_limit", "stop_reason=tool_calls max_tokens=256 output_tokens=300"),
    ] * 2


def test_invalid_json_in_a_reply_that_ended_normally_is_not_run_and_says_so(
        tmp_path, monkeypatch):
    seen = _serve(monkeypatch, [
        _openai_sse([("call_1", "write", ['{"path": "a.md", "content": "x" "y"}'])], "tool_calls"),
        _openai_prose("Retrying with valid JSON."),
    ])
    provider = OpenAICompatProvider(base_url="http://unit.test/v1", api_key="test-key",
                                    model="qwen3.8-max", provider_name="qwen")
    _lookup, write, rows = _turn(tmp_path, provider, "qwen3.8-max")

    assert write.calls == []
    [result] = [m["content"] for m in seen[1]["messages"] if m.get("role") == "tool"]
    assert "not valid JSON" in result and "not run" in result, result
    assert "output limit" not in result, "a call that was sent whole is not called cut off"
    assert [(r["tool_name"], r["success"], r["error_type"]) for r in rows] == [
        ("write", 0, "unreadable_arguments"),
    ]


@pytest.mark.parametrize("stop,out,cap,expected", [
    ("length", 10, 4096, True), ("max_tokens", 10, 4096, True),
    ("tool_calls", 4111, 4096, True), ("tool_calls", 4096, 4096, True),
    ("tool_calls", 4095, 4096, False), ("stop", 10, 4096, False), (None, 0, 0, False),
])
def test_cut_at_output_limit(stop, out, cap, expected):
    from prometheus.engine.agent_loop import cut_at_output_limit

    assert cut_at_output_limit(stop, out, cap) is expected
