"""A tool call served by a cloud provider records what the provider says served it
(WP-X.21 T12a, docs/audits/TELEMETRY-GAPS.md).

``tool_calls.served_model`` exists so a row names what actually answered,
beside the name that was asked for (``model``). Only llama.cpp's parser read
it. The OpenAI-compatible parser (qwen, xai and the other compatible clouds)
and the Anthropic parser ignored the model each stream names. On the mini that
left the column NULL on 4,967 tool calls in the last 30 days (qwen3.8-max
4,746, qwen3.8-flash 196, grok-4.5 25), so the fine-tuning corpus cannot say
which snapshot of a cloud alias produced a golden example. The committed
``hosted_route`` trace shows the case exactly: the request asks for
``claude-haiku-4-5`` and the stream's ``message_start`` names a dated id.

Pinned here: each parser reads the model its stream names, over the recorded
streams in the committed parity traces; a real turn through each provider
writes it to the tool row, beside the requested name, which is unchanged, as
is the request; a stream that names no model leaves the column NULL.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage
from prometheus.providers.anthropic import AnthropicProvider
from prometheus.providers.base import ApiMessageCompleteEvent, ApiMessageRequest
from prometheus.providers.openai_compat import OpenAICompatProvider
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

TRACES = Path(__file__).resolve().parent / "fixtures" / "parity"


# ── a server that answers each POST with the next scripted stream ───────────

def _serve(monkeypatch, bodies: list[str]) -> list[dict]:
    """Route every httpx.AsyncClient to a server answering the n-th POST with
    ``bodies[n]`` (the last one again past the end); returns the request
    bodies it received."""
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


def _events(body: str) -> list[dict]:
    out = []
    for line in body.splitlines():
        if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]"):
            out.append(json.loads(line[6:]))
    return out


def _recorded_streams(trace: str, path: str) -> list[tuple[dict, str]]:
    """(request, response body) of every streamed completion in a committed trace."""
    data = json.loads((TRACES / f"{trace}.trace.json").read_text(encoding="utf-8"))
    return [(ex["request"], ex["body"]) for ex in data["exchanges"]
            if ex["method"] == "POST" and ex["path"] == path and "data: " in (ex["body"] or "")]


def _qwen() -> OpenAICompatProvider:
    return OpenAICompatProvider(base_url="http://unit.test/v1", api_key="test-key",
                                model="qwen3.8-max", provider_name="qwen")


def _anthropic() -> AnthropicProvider:
    return AnthropicProvider(api_key="test-key", base_url="http://unit.test/v1")


async def _complete(provider, model: str) -> ApiMessageCompleteEvent:
    done = None
    request = ApiMessageRequest(model=model, messages=[ConversationMessage.from_user_text("q")],
                                max_tokens=16)
    async for ev in provider.stream_message(request):
        if isinstance(ev, ApiMessageCompleteEvent):
            done = ev
    assert done is not None, "the stream ended without a completion event"
    return done


# ── the parsers, over recorded streams ──────────────────────────────────────

def test_the_openai_compatible_parser_reads_the_model_a_recorded_stream_names(monkeypatch):
    """The OpenAI chat-completions wire, as the parity recording captured it
    (llama.cpp speaks the same wire the compatible clouds do)."""
    streams = _recorded_streams("tool_calls", "/v1/chat/completions")
    assert streams, "premise: the trace holds streamed completions"
    for request, body in streams:
        named = {e.get("model") for e in _events(body)} - {None}
        assert len(named) == 1, "premise: the recorded stream names one model"
        _serve(monkeypatch, [body])
        done = asyncio.run(_complete(_qwen(), request["model"]))
        assert done.served_model == named.pop()


def test_the_anthropic_parser_reads_the_model_a_recorded_message_start_names(monkeypatch):
    streams = _recorded_streams("hosted_route", "/v1/messages")
    assert streams, "premise: the trace holds the hosted turn"
    for request, body in streams:
        [start] = [e for e in _events(body) if e.get("type") == "message_start"]
        _serve(monkeypatch, [body])
        done = asyncio.run(_complete(_anthropic(), request["model"]))
        assert done.served_model == start["message"]["model"]


# ── the rows a real turn writes ─────────────────────────────────────────────

class _Input(BaseModel):
    q: str


class _Lookup(BaseTool):
    name = "lookup"
    description = "look a count up"
    input_model = _Input

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output="3", is_error=False)


def _turn(tmp_path, provider, model: str) -> sqlite3.Row:
    """One turn: the model calls ``lookup``, then answers. Returns the tool row."""
    registry = ToolRegistry()
    registry.register(_Lookup())
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    ctx = LoopContext(provider=provider, model=model, system_prompt="You are a test.",
                      max_tokens=256, tool_registry=registry, telemetry=tel)

    async def drain() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("how many gears?")],
                                session_id="desktop:t12a"):
            pass

    asyncio.run(drain())
    tel.close()
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    [row] = con.execute("SELECT tool_name, success, model, served_model FROM tool_calls"
                        " WHERE tool_name != '_loop_transition'").fetchall()
    return row


def _openai_sse(served: str | None, *, tool: bool) -> str:
    """A compatible cloud's stream, in the chunk shape the recordings show."""
    base: dict = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1790000000}
    if served is not None:
        base["model"] = served
    if tool:
        deltas = [{"role": "assistant", "content": None, "tool_calls": [
                      {"index": 0, "id": "call_1", "type": "function",
                       "function": {"name": "lookup", "arguments": ""}}]},
                  {"tool_calls": [{"index": 0, "function": {"arguments": '{"q": "gears"}'}}]}]
    else:
        deltas = [{"role": "assistant", "content": "There are 3 gears."}]
    chunks = [{**base, "choices": [{"index": 0, "delta": d, "finish_reason": None}]} for d in deltas]
    chunks.append({**base, "choices": [{"index": 0, "delta": {},
                                        "finish_reason": "tool_calls" if tool else "stop"}]})
    chunks.append({**base, "choices": [],
                   "usage": {"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49}})
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


@pytest.mark.parametrize("provider_name,requested", [("qwen", "qwen3.8-max"), ("xai", "grok-4.5")])
def test_a_compatible_clouds_tool_call_records_the_served_model_beside_the_requested_one(
        tmp_path, monkeypatch, provider_name, requested):
    served = f"{requested}-snapshot-a"  # synthetic: an alias answered by a snapshot
    seen = _serve(monkeypatch, [_openai_sse(served, tool=True), _openai_sse(served, tool=False)])
    provider = OpenAICompatProvider(base_url="http://unit.test/v1", api_key="test-key",
                                    model=requested, provider_name=provider_name)
    row = _turn(tmp_path, provider, requested)
    assert (row["tool_name"], row["success"]) == ("lookup", 1)
    assert row["served_model"] == served, "the stream named what served the call"
    assert row["model"] == requested, "the requested name is kept, not overwritten"
    assert [s["model"] for s in seen] == [requested, requested], "the request is unchanged"


def _anthropic_tool_sse(served: str) -> str:
    """A tool_use turn, in the event sequence the recorded hosted stream shows."""
    events = [
        {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": served,
            "content": [], "stop_reason": None, "usage": {"input_tokens": 40, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": '{"q": "gears"}'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def test_an_anthropic_tool_call_records_the_dated_model_that_served_the_alias(tmp_path, monkeypatch):
    # The recorded hosted turn: what it asked for, what answered, and its reply.
    [(request, answer)] = _recorded_streams("hosted_route", "/v1/messages")
    requested = request["model"]
    served = next(e for e in _events(answer) if e["type"] == "message_start")["message"]["model"]
    seen = _serve(monkeypatch, [_anthropic_tool_sse(served), answer])
    row = _turn(tmp_path, _anthropic(), requested)
    assert (row["tool_name"], row["success"]) == ("lookup", 1)
    assert (row["model"], row["served_model"]) == (requested, served)
    assert [s["model"] for s in seen] == [requested, requested], "the request is unchanged"


def test_a_stream_that_names_no_model_leaves_the_column_null(tmp_path, monkeypatch):
    """None, never the requested name dressed up as an answer, and never ''."""
    _serve(monkeypatch, [_openai_sse(None, tool=True), _openai_sse(None, tool=False)])
    row = _turn(tmp_path, _qwen(), "qwen3.8-max")
    assert (row["tool_name"], row["model"], row["served_model"]) == ("lookup", "qwen3.8-max", None)
