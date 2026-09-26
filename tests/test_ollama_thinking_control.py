"""Ollama thinking control (WP-X.35).

Ollama turns thinking ON for every model whose /api/show capabilities list
``thinking`` unless the request says otherwise, and its OpenAI-compatible
route streams the thought as ``delta.reasoning`` — which the provider used to
drop. Driven through the REAL ``stream_message`` against a fake Ollama that
answers /api/show and /v1/chat/completions the way 0.23.0 does (and the 0.34
chunk shape where it differs).

The byte pins were captured from origin/main's provider (5cb7af0) with httpx
0.28.1: a model that does not list ``thinking`` — or whose capabilities cannot
be read — must get exactly those bytes.
"""

from __future__ import annotations

import json

import httpx
import pytest

from prometheus.engine.messages import ConversationMessage, TextBlock, ThinkingBlock
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
)
from prometheus.providers.ollama import OllamaProvider

QWEN25 = "qwen2.5:7b-instruct"          # the parity goldens' Ollama model
QWEN35 = "qwen3.5:9b"                   # the router's fallback: a thinking model
GPT_OSS = "gpt-oss:20b"

# origin/main's exact request bodies (see the module docstring).
MAIN_BYTES = {
    'plain': (
        b'{"model":"qwen2.5:7b-instruct","messages":[{"role":"system","content":"You are terse."},{"role":"user","content":"two"}],"stream":true,"max_tokens":64,"stream_options":{"include_usage":true}}'
    ),
    'tools': (
        b'{"model":"qwen2.5:7b-instruct","messages":[{"role":"user","content":"read a.txt"}],"stream":true,"max_tokens":128,"stream_options":{"include_usage":true},"tools":[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]}'
    ),
    'json_grammar': (
        b'{"model":"qwen2.5:7b-instruct","messages":[{"role":"user","content":"x"}],"stream":true,"max_tokens":8,"stream_options":{"include_usage":true},"format":"json","grammar":"root ::= \\"x\\""}'
    ),
}

TOOL = {"name": "read_file", "description": "Read a file",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                         "required": ["path"]}}


def _request(case: str = "plain", model: str = QWEN25, **kw) -> ApiMessageRequest:
    base = {
        "plain": dict(messages=[ConversationMessage.from_user_text("two")], max_tokens=64,
                      system_prompt="You are terse."),
        "tools": dict(messages=[ConversationMessage.from_user_text("read a.txt")],
                      max_tokens=128, tools=[TOOL]),
        "json_grammar": dict(messages=[ConversationMessage.from_user_text("x")], max_tokens=8),
    }[case]
    return ApiMessageRequest(model=model, **{**base, **kw})


def _provider(case: str = "plain") -> OllamaProvider:
    if case == "json_grammar":
        return OllamaProvider(base_url="http://unit.test:1", force_json=True,
                              grammar='root ::= "x"')
    return OllamaProvider(base_url="http://unit.test:1")


# ---------------------------------------------------------------------------
# A fake Ollama 0.23.0
# ---------------------------------------------------------------------------

def _chunk(delta: dict, finish=None) -> dict:
    return {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": "m",
            "system_fingerprint": "fp_ollama",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def sse(*, reasoning: str = "", content: str = "", finish: str = "stop",
        usage: tuple[int, int] = (30, 5), new_shape: bool = False,
        tool_call: dict | None = None) -> bytes:
    """One streamed completion as Ollama sends it: thought chunks first
    (content "" on 0.23.0, content omitted and role only first on 0.34),
    then content, then the finish chunk, the usage chunk and [DONE]."""
    chunks = []
    for i, word in enumerate(reasoning.split(" ") if reasoning else []):
        piece = word if i == 0 else " " + word
        if new_shape:
            delta = {"reasoning": piece}
            if i == 0:
                delta["role"] = "assistant"
        else:
            delta = {"role": "assistant", "content": "", "reasoning": piece}
        chunks.append(_chunk(delta))
    for i, word in enumerate(content.split(" ") if content else []):
        piece = word if i == 0 else " " + word
        chunks.append(_chunk({"content": piece} if new_shape
                             else {"role": "assistant", "content": piece}))
    if tool_call:
        chunks.append(_chunk({"role": "assistant", "content": "", "tool_calls": [tool_call]}))
    chunks.append(_chunk({} if new_shape else {"role": "assistant", "content": ""}, finish))
    chunks.append({"choices": [], "usage": {"prompt_tokens": usage[0],
                                            "completion_tokens": usage[1],
                                            "total_tokens": sum(usage)}})
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


class FakeOllama:
    """Routes by path. ``show`` maps a model to its /api/show answer: a dict
    (the JSON body), or an int (that HTTP status). ``completions`` is the queue
    of SSE bodies /v1/chat/completions answers with, in order."""

    def __init__(self, show: dict | None = None, completions: list[bytes] | None = None):
        self.show = show or {}
        self.completions = list(completions or [sse(content="two")])
        self.show_calls: list[bytes] = []
        self.chat_calls: list[bytes] = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/show":
            self.show_calls.append(req.content)
            answer = self.show.get(json.loads(req.content).get("model"), 404)
            if isinstance(answer, int):
                return httpx.Response(answer, json={"error": "not found"})
            if isinstance(answer, bytes):
                return httpx.Response(200, content=answer)
            return httpx.Response(200, json=answer)
        assert req.url.path == "/v1/chat/completions", req.url.path
        self.chat_calls.append(req.content)
        body = self.completions.pop(0)
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    @property
    def chat_bodies(self) -> list[dict]:
        return [json.loads(b) for b in self.chat_calls]


def show(*caps: str, family: str = "qwen35") -> dict:
    return {"capabilities": list(caps), "details": {"family": family}}


NO_THINKING = show("completion", "tools", family="qwen2")
THINKING = show("completion", "tools", "thinking")


@pytest.fixture
def fake(monkeypatch):
    server = FakeOllama()
    real = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(server.handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return server


class _RecordingTelemetry:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_silent_failure(self, subsystem, operation, exc, context=None):  # noqa: ANN001
        self.calls.append({"subsystem": subsystem, "operation": operation,
                           "exc_type": type(exc).__name__, "context": context or {}})


@pytest.fixture
def tel(monkeypatch):
    from prometheus.telemetry import tracker

    recorder = _RecordingTelemetry()
    monkeypatch.setattr(tracker, "get_telemetry_handle", lambda: recorder)
    return recorder


async def run(provider: OllamaProvider, request: ApiMessageRequest):
    deltas, done = [], None
    async for ev in provider.stream_message(request):
        if isinstance(ev, ApiTextDeltaEvent):
            deltas.append(ev.text)
        elif isinstance(ev, ApiMessageCompleteEvent):
            done = ev
    assert done is not None, "stream ended without a completion event"
    return deltas, done


def thinking_blocks(done) -> list[str]:
    return [b.thinking for b in done.message.content if isinstance(b, ThinkingBlock)]


# ---------------------------------------------------------------------------
# 1. Decided per model: only a model that lists "thinking" gets a control
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(MAIN_BYTES))
@pytest.mark.parametrize("answer", [
    NO_THINKING,                                   # a model without the capability
    404,                                           # not found (e.g. a wrong tag)
    500,                                           # the server failed
    b"not json",                                   # unreadable
    {"details": {"family": "qwen35"}},             # no capabilities list at all
], ids=["no-thinking", "404", "500", "bad-json", "no-capabilities"])
async def test_every_model_that_does_not_list_thinking_gets_todays_bytes(fake, case, answer):
    fake.show = {QWEN25: answer}
    await run(_provider(case), _request(case))
    assert fake.chat_calls == [MAIN_BYTES[case]]
    # The capability question is exactly the model name: the body the backend
    # registry sends, which the parity recordings already answer.
    assert [json.loads(b) for b in fake.show_calls] == [{"model": QWEN25}]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["gemma4:cloud", "gpt-oss:120b-cloud", "gpt-oss:20b:CLOUD"])
async def test_a_cloud_model_is_never_asked_and_gets_todays_bytes(fake, name):
    fake.show = {name: THINKING}
    await run(_provider(), _request(model=name))
    assert fake.show_calls == []                   # /api/show would reach ollama.com
    assert fake.chat_calls == [MAIN_BYTES["plain"].replace(QWEN25.encode(), name.encode())]


@pytest.mark.asyncio
@pytest.mark.parametrize("suppress", [None, True])
async def test_a_thinking_model_gets_thinking_off_by_default(fake, suppress):
    fake.show = {QWEN25: THINKING}
    await run(_provider(), _request(suppress_thinking=suppress))
    # Exactly today's body plus the one field /v1 honours ("think" is dropped).
    assert fake.chat_calls == [MAIN_BYTES["plain"][:-1] + b',"reasoning_effort":"none"}']


@pytest.mark.asyncio
async def test_a_caller_that_opts_in_gets_ollamas_own_default_thinking_on(fake):
    fake.show = {QWEN25: THINKING}
    await run(_provider(), _request(suppress_thinking=False))
    assert fake.chat_calls == [MAIN_BYTES["plain"]]


@pytest.mark.asyncio
async def test_the_capability_is_asked_once_per_model_and_a_failure_is_not_remembered(fake):
    p = _provider()
    fake.show = {QWEN25: 503}
    fake.completions = [sse(content="a")] * 5
    await run(p, _request())                       # unknown: today's bytes
    fake.show = {QWEN25: THINKING, QWEN35: NO_THINKING}
    await run(p, _request())                       # asked again — and now it thinks
    await run(p, _request())                       # cached
    await run(p, _request(model=QWEN35))           # another model: its own question
    assert [json.loads(b)["model"] for b in fake.show_calls] == [QWEN25, QWEN25, QWEN35]
    assert ["reasoning_effort" in b for b in fake.chat_bodies] == [False, True, True, False]


# ---------------------------------------------------------------------------
# 3. The reasoning is kept, out of the reply
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("new_shape", [False, True], ids=["0.23-chunks", "0.34-chunks"])
async def test_streamed_reasoning_is_a_thinking_block_never_reply_text(fake, new_shape):
    fake.show = {QWEN35: THINKING}
    fake.completions = [sse(reasoning="The user wants two.", content="two",
                            usage=(40, 17), new_shape=new_shape)]
    deltas, done = await run(_provider(), _request(model=QWEN35, suppress_thinking=False))
    assert "".join(deltas) == "two"                 # nothing of the thought streamed
    assert done.message.text == "two"
    assert thinking_blocks(done) == ["The user wants two."]
    # completion_tokens already counts the thought: reported as the server did.
    assert (done.usage.input_tokens, done.usage.output_tokens) == (40, 17)


@pytest.mark.asyncio
async def test_a_thought_on_the_message_is_never_sent_back(fake):
    fake.show = {QWEN35: THINKING}
    fake.completions = [sse(reasoning="Hidden plan.", content="one"), sse(content="two")]
    p = _provider()
    _, first = await run(p, _request(model=QWEN35, suppress_thinking=False))
    assert thinking_blocks(first) == ["Hidden plan."]           # kept on the message...
    history = [ConversationMessage.from_user_text("one?"), first.message,
               ConversationMessage.from_user_text("two?")]
    await run(p, _request(model=QWEN35, messages=history))
    # ...and never sent back to the server.
    assert b"Hidden plan" not in fake.chat_calls[1] and b"reasoning\"" not in fake.chat_calls[1]


@pytest.mark.asyncio
async def test_a_model_that_is_not_known_to_think_still_keeps_its_thought(fake, tel):
    # /api/show failed, so nothing was sent — but the server's default for a
    # thinking model is ON, and what it thought must not leak into the reply.
    fake.show = {QWEN35: 500}
    fake.completions = [sse(reasoning="Pondering.", content="done")]
    _, done = await run(_provider(), _request(model=QWEN35))
    assert (done.message.text, thinking_blocks(done)) == ("done", ["Pondering."])
    assert tel.calls == []


# ---------------------------------------------------------------------------
# 4. Thought, but answered nothing
# ---------------------------------------------------------------------------

BUDGET_SPENT = sse(reasoning="Let me think about this at length", finish="length", usage=(50, 64))


@pytest.mark.asyncio
async def test_thinking_on_and_the_budget_spent_is_recorded_and_retried_once_thinking_off(fake, tel):
    fake.show = {QWEN35: THINKING}
    fake.completions = [BUDGET_SPENT, sse(content="ok", usage=(52, 3))]
    deltas, done = await run(_provider(), _request(model=QWEN35, suppress_thinking=False))

    first, second = fake.chat_bodies
    assert "reasoning_effort" not in first                     # thinking was on
    assert second == {**first, "reasoning_effort": "none"}     # the one retry, thinking off
    assert (deltas, done.message.text, done.stop_reason) == (["ok"], "ok", "stop")
    assert thinking_blocks(done) == ["Let me think about this at length"]
    assert (done.usage.input_tokens, done.usage.output_tokens) == (102, 67)   # both spent

    (rec,) = tel.calls
    assert (rec["subsystem"], rec["operation"], rec["exc_type"]) == (
        "ollama_provider", "stream_message", "EmptyCompletionError")
    assert rec["context"] == {
        "model": QWEN35, "finish_reason": "length", "output_tokens": 64,
        "reasoning_chars": len("Let me think about this at length"),
        "used_reasoning_fallback": False, "budget_exhausted_on_thinking": False,
        "reasoning_only": True, "retried_with_thinking_off": True,
    }


@pytest.mark.asyncio
async def test_a_retry_that_still_only_thinks_is_recorded_and_not_retried_again(fake, tel):
    fake.show = {QWEN35: THINKING}
    fake.completions = [BUDGET_SPENT, BUDGET_SPENT]
    _, done = await run(_provider(), _request(model=QWEN35, suppress_thinking=False))
    assert len(fake.chat_calls) == 2
    assert [c["context"]["retried_with_thinking_off"] for c in tel.calls] == [True, False]
    # No answer — and the thought is still not the answer.
    assert done.message.text == "" and thinking_blocks(done)


@pytest.mark.asyncio
@pytest.mark.parametrize("setup", ["thinking-already-off", "gpt-oss", "capability-unknown"])
async def test_no_retry_when_thinking_off_is_not_a_fix(fake, tel, setup):
    model, suppress = QWEN35, False
    if setup == "thinking-already-off":
        fake.show, suppress = {QWEN35: THINKING}, None      # "none" was already sent
    elif setup == "gpt-oss":
        model = GPT_OSS                                      # "none" cannot stop harmony
        fake.show = {GPT_OSS: show("completion", "tools", "thinking", family="gptoss")}
    else:
        fake.show = {QWEN35: 500}                            # not known to think: no control
    fake.completions = [BUDGET_SPENT]
    deltas, done = await run(_provider(), _request(model=model, suppress_thinking=suppress))
    assert len(fake.chat_calls) == 1
    (rec,) = tel.calls
    assert rec["context"]["retried_with_thinking_off"] is False
    assert (deltas, done.message.text, done.stop_reason) == ([], "", "length")
    assert thinking_blocks(done) == ["Let me think about this at length"]


@pytest.mark.asyncio
async def test_a_tool_call_with_no_text_is_an_answer_not_a_failure(fake, tel):
    fake.show = {QWEN35: THINKING}
    call = {"index": 0, "id": "call_1", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}}
    fake.completions = [sse(reasoning="Read it.", tool_call=call, finish="tool_calls")]
    _, done = await run(_provider("tools"), _request("tools", model=QWEN35, suppress_thinking=False))
    assert len(fake.chat_calls) == 1 and tel.calls == []
    assert [b.name for b in done.message.tool_uses] == ["read_file"]
    assert (done.message.text, thinking_blocks(done)) == ("", ["Read it."])


@pytest.mark.asyncio
async def test_a_plain_answer_is_untouched(fake, tel):
    fake.show = {QWEN25: NO_THINKING}
    deltas, done = await run(_provider(), _request())
    assert (deltas, done.message.text, done.stop_reason) == (["two"], "two", "stop")
    assert [type(b) for b in done.message.content] == [TextBlock]
    assert (done.usage.input_tokens, done.usage.output_tokens) == (30, 5)
    assert tel.calls == []
