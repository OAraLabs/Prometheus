"""Ollama thinking control (WP-X.35), on the native ``/api/chat`` route.

Ollama turns thinking ON for every model whose /api/show capabilities list
``thinking`` unless the request says otherwise, and streams the thought as
``message.thinking``. Driven through the REAL ``stream_message`` against a fake
Ollama that answers /api/show, /api/ps and /api/chat the way 0.23.0 does.

The byte pins are the native request bodies (audit 2026-10-16, fix 5b): a
model that does not list ``thinking`` — or whose capabilities cannot be read —
must get exactly those bytes. The /v1 bodies they replace were pinned at
5cb7af0; the change is the endpoint's own shape (``options.num_predict`` for
``max_tokens``, ``think: false`` for ``reasoning_effort: "none"``, no
``stream_options``: the final line always carries the counts).
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

# The native request bodies (see the module docstring).
MAIN_BYTES = {
    'plain': (
        b'{"model":"qwen2.5:7b-instruct","messages":[{"role":"system","content":"You are terse."},{"role":"user","content":"two"}],"stream":true,"options":{"num_predict":64}}'
    ),
    'tools': (
        b'{"model":"qwen2.5:7b-instruct","messages":[{"role":"user","content":"read a.txt"}],"stream":true,"options":{"num_predict":128},"tools":[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]}'
    ),
    'json_grammar': (
        b'{"model":"qwen2.5:7b-instruct","messages":[{"role":"user","content":"x"}],"stream":true,"options":{"num_predict":8},"format":"json","grammar":"root ::= \\"x\\""}'
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

def ndjson(*, reasoning: str = "", content: str = "", finish: str = "stop",
           usage: tuple[int, int] = (30, 5), tool_call: dict | None = None) -> bytes:
    """One streamed completion as Ollama's /api/chat sends it: thought lines
    first (``message.thinking``), then content, then a call whole, then the
    final ``done`` line with the counts."""
    lines = []
    for i, word in enumerate(reasoning.split(" ") if reasoning else []):
        piece = word if i == 0 else " " + word
        lines.append({"model": "m", "message": {"role": "assistant", "content": "", "thinking": piece},
                      "done": False})
    for i, word in enumerate(content.split(" ") if content else []):
        piece = word if i == 0 else " " + word
        lines.append({"model": "m", "message": {"role": "assistant", "content": piece}, "done": False})
    if tool_call:
        lines.append({"model": "m", "message": {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                      "done": False})
    lines.append({"model": "m", "message": {"role": "assistant", "content": ""}, "done": True,
                  "done_reason": finish, "prompt_eval_count": usage[0], "eval_count": usage[1]})
    return "".join(json.dumps(line) + "\n" for line in lines).encode()


class FakeOllama:
    """Routes by path. ``show`` maps a model to its /api/show answer: a dict
    (the JSON body), bytes (a raw body), or an int (that HTTP status).
    ``completions`` is the queue /api/chat answers with, in order: NDJSON
    bytes, an int (that status), or a (status, json body) pair. /api/ps lists
    nothing loaded, so no num_ctx is sent unless the request budgeted one."""

    def __init__(self, show: dict | None = None, completions: list[bytes] | None = None):
        self.show = show or {}
        self.completions = list(completions or [ndjson(content="two")])
        self.show_calls: list[bytes] = []
        self.chat_calls: list[bytes] = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/show":
            assert req.method == "POST", req.method     # the only method /api/show takes
            self.show_calls.append(req.content)
            answer = self.show.get(json.loads(req.content).get("model"), 404)
            if isinstance(answer, int):
                return httpx.Response(answer, json={"error": "not found"})
            if isinstance(answer, bytes):
                return httpx.Response(200, content=answer)
            return httpx.Response(200, json=answer)
        if req.url.path == "/api/ps":
            assert req.method == "GET", req.method
            return httpx.Response(200, json={"models": []})
        assert req.url.path == "/api/chat", req.url.path
        assert req.method == "POST", req.method
        self.chat_calls.append(req.content)
        body = self.completions.pop(0)
        if isinstance(body, int):
            return httpx.Response(body, json={"error": "unavailable"})
        if isinstance(body, tuple):
            return httpx.Response(body[0], json=body[1])
        return httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"})

    @property
    def chat_bodies(self) -> list[dict]:
        return [json.loads(b) for b in self.chat_calls]


def show(*caps: str, family: str = "qwen35") -> dict:
    return {"capabilities": list(caps), "details": {"family": family}}


NO_THINKING = show("completion", "tools", family="qwen2")
THINKING = show("completion", "tools", "thinking")


@pytest.fixture
def fake(monkeypatch):
    import prometheus.providers.retry as retry

    # The transport retry's backoff is read per attempt; no real waiting here.
    monkeypatch.setattr(retry, "BASE_DELAY", 0.0)
    monkeypatch.setattr(retry, "MAX_DELAY", 0.0)
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
@pytest.mark.parametrize("case", sorted(MAIN_BYTES))
@pytest.mark.parametrize("suppress", [None, True])
async def test_a_thinking_model_gets_thinking_off_by_default(fake, case, suppress):
    fake.show = {QWEN25: THINKING}
    await run(_provider(case), _request(case, suppress_thinking=suppress))
    # Exactly the plain body plus the one field that turns thinking off.
    assert fake.chat_calls == [MAIN_BYTES[case][:-1] + b',"think":false}']


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(MAIN_BYTES))
async def test_a_caller_that_opts_in_gets_ollamas_own_default_thinking_on(fake, case):
    fake.show = {QWEN25: THINKING}
    await run(_provider(case), _request(case, suppress_thinking=False))
    assert fake.chat_calls == [MAIN_BYTES[case]]


@pytest.mark.asyncio
async def test_a_harmony_model_also_gets_thinking_off_by_default(fake):
    # gpt-oss is excluded from the RETRY (think: false cannot stop its analysis
    # channel), not from the default: the field is valid and drops its Reasoning line.
    fake.show = {GPT_OSS: show("completion", "tools", "thinking", family="gptoss")}
    await run(_provider(), _request(model=GPT_OSS))
    assert fake.chat_bodies[0]["think"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["cloudy:7b", "qwen3:8b-wordcloud", "pointcloud-qwen3:8b"])
async def test_a_local_name_that_merely_mentions_cloud_is_still_asked(fake, name):
    fake.show = {name: THINKING}
    await run(_provider(), _request(model=name))
    assert [json.loads(b) for b in fake.show_calls] == [{"model": name}]
    assert fake.chat_bodies[0]["think"] is False


@pytest.mark.asyncio
async def test_the_capability_is_asked_once_per_model_and_a_failure_is_not_remembered(fake):
    p = _provider()
    fake.show = {QWEN25: 503}
    fake.completions = [ndjson(content="a")] * 5
    await run(p, _request())                       # unknown: today's bytes
    fake.show = {QWEN25: THINKING, QWEN35: NO_THINKING}
    await run(p, _request())                       # asked again — and now it thinks
    await run(p, _request())                       # cached
    await run(p, _request(model=QWEN35))           # another model: its own question
    assert [json.loads(b)["model"] for b in fake.show_calls] == [QWEN25, QWEN25, QWEN35]
    assert ["think" in b for b in fake.chat_bodies] == [False, True, True, False]


@pytest.mark.asyncio
@pytest.mark.parametrize("unreadable", [b"not json", {"details": {"family": "qwen35"}}],
                         ids=["bad-json", "no-capabilities"])
async def test_an_unreadable_answer_is_not_remembered_either(fake, unreadable):
    p = _provider()
    fake.show = {QWEN35: unreadable}
    fake.completions = [ndjson(content="a")] * 2
    await run(p, _request(model=QWEN35))
    fake.show = {QWEN35: THINKING}
    await run(p, _request(model=QWEN35))
    assert len(fake.show_calls) == 2
    assert ["think" in b for b in fake.chat_bodies] == [False, True]


# ---------------------------------------------------------------------------
# A server that refuses ``think`` (a 400 naming it)
# ---------------------------------------------------------------------------

OLD_SERVER_400 = (400, {"error": "invalid think value: false (must be \"high\", \"medium\", "
                                 "\"low\", true, or false)"})


@pytest.mark.asyncio
async def test_a_server_that_refuses_the_field_gets_todays_request_from_then_on(fake, tel):
    p = _provider()
    fake.show = {QWEN35: THINKING}
    fake.completions = [OLD_SERVER_400, ndjson(content="four"), ndjson(content="five")]
    _, done = await run(p, _request(model=QWEN35))
    assert done.message.text == "four"                       # the turn is answered, not failed
    await run(p, _request(model=QWEN35))
    first, fallback, later = fake.chat_bodies
    assert first["think"] is False
    assert "think" not in fallback and "think" not in later
    assert fake.chat_calls[1] == fake.chat_calls[2] == MAIN_BYTES["plain"].replace(
        QWEN25.encode(), QWEN35.encode())
    assert tel.calls == []


@pytest.mark.asyncio
async def test_any_other_400_is_still_an_error(fake):
    fake.show = {QWEN35: THINKING}
    fake.completions = [(400, {"error": "messages: invalid role"})]
    with pytest.raises(httpx.HTTPStatusError):
        await run(_provider(), _request(model=QWEN35))
    assert len(fake.chat_calls) == 1


@pytest.mark.asyncio
async def test_no_retry_when_the_server_refuses_thinking_off(fake, tel):
    # Opt-in turn spent its budget; the thinking-off retry is refused: the
    # first attempt stands, and the model is not asked with the field again.
    p = _provider()
    fake.show = {QWEN35: THINKING}
    fake.completions = [BUDGET_SPENT, OLD_SERVER_400, BUDGET_SPENT]
    _, done = await run(p, _request(model=QWEN35, suppress_thinking=False))
    assert (done.message.text, done.stop_reason) == ("", "length")
    assert len(fake.chat_calls) == 2
    await run(p, _request(model=QWEN35, suppress_thinking=False))   # spends it again...
    assert len(fake.chat_calls) == 3                                 # ...and is not retried
    assert [c["context"]["retried_with_thinking_off"] for c in tel.calls] == [True, False]


# ---------------------------------------------------------------------------
# 3. The reasoning is kept, out of the reply
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streamed_reasoning_is_a_thinking_block_never_reply_text(fake):
    fake.show = {QWEN35: THINKING}
    fake.completions = [ndjson(reasoning="The user wants two.", content="two", usage=(40, 17))]
    deltas, done = await run(_provider(), _request(model=QWEN35, suppress_thinking=False))
    assert "".join(deltas) == "two"                 # nothing of the thought streamed
    assert done.message.text == "two"
    assert thinking_blocks(done) == ["The user wants two."]
    # eval_count already counts the thought: reported as the server did.
    assert (done.usage.input_tokens, done.usage.output_tokens) == (40, 17)


@pytest.mark.asyncio
async def test_a_thought_on_the_message_is_never_sent_back(fake):
    fake.show = {QWEN35: THINKING}
    fake.completions = [ndjson(reasoning="Hidden plan.", content="one"), ndjson(content="two")]
    p = _provider()
    _, first = await run(p, _request(model=QWEN35, suppress_thinking=False))
    assert thinking_blocks(first) == ["Hidden plan."]           # kept on the message...
    history = [ConversationMessage.from_user_text("one?"), first.message,
               ConversationMessage.from_user_text("two?")]
    await run(p, _request(model=QWEN35, messages=history))
    # ...and never sent back to the server.
    assert b"Hidden plan" not in fake.chat_calls[1] and b"thinking\"" not in fake.chat_calls[1]


@pytest.mark.asyncio
async def test_a_model_that_is_not_known_to_think_still_keeps_its_thought(fake, tel):
    # /api/show failed, so nothing was sent — but the server's default for a
    # thinking model is ON, and what it thought must not leak into the reply.
    fake.show = {QWEN35: 500}
    fake.completions = [ndjson(reasoning="Pondering.", content="done")]
    _, done = await run(_provider(), _request(model=QWEN35))
    assert (done.message.text, thinking_blocks(done)) == ("done", ["Pondering."])
    assert tel.calls == []


# ---------------------------------------------------------------------------
# 4. Thought, but answered nothing
# ---------------------------------------------------------------------------

BUDGET_SPENT = ndjson(reasoning="Let me think about this at length", finish="length", usage=(50, 64))


@pytest.mark.asyncio
async def test_thinking_on_and_the_budget_spent_is_recorded_and_retried_once_thinking_off(fake, tel):
    fake.show = {QWEN35: THINKING}
    fake.completions = [BUDGET_SPENT, ndjson(content="ok", usage=(52, 3))]
    deltas, done = await run(_provider(), _request(model=QWEN35, suppress_thinking=False))

    first, second = fake.chat_bodies
    assert "think" not in first                                # thinking was on
    assert second == {**first, "think": False}                 # the one retry, thinking off
    assert (deltas, done.message.text, done.stop_reason) == (["ok"], "ok", "stop")
    assert thinking_blocks(done) == ["Let me think about this at length"]
    # One prompt (sent twice): input is its size, the meter's figure; both
    # attempts' output was really generated.
    assert (done.usage.input_tokens, done.usage.output_tokens) == (52, 67)

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
    fake.completions = [BUDGET_SPENT,
                        ndjson(reasoning="Still pondering", finish="stop", usage=(51, 9))]
    _, done = await run(_provider(), _request(model=QWEN35, suppress_thinking=False))
    assert len(fake.chat_calls) == 2
    assert [(c["context"]["retried_with_thinking_off"], c["context"]["finish_reason"],
             c["context"]["output_tokens"], c["context"]["reasoning_chars"]) for c in tel.calls] == [
        (True, "length", 64, len("Let me think about this at length")),
        (False, "stop", 9, len("Still pondering")),
    ]
    # No answer — and the thought is still not the answer.
    assert (done.message.text, done.stop_reason) == ("", "stop")
    assert thinking_blocks(done) == ["Let me think about this at length\n\nStill pondering"]
    assert (done.usage.input_tokens, done.usage.output_tokens) == (51, 73)


@pytest.mark.asyncio
async def test_a_model_that_stops_mid_thought_is_recorded_and_retried(fake, tel):
    # Not only a spent budget: EOS inside the thought ("stop") answers nothing too.
    fake.show = {QWEN35: THINKING}
    fake.completions = [ndjson(reasoning="Hmm", finish="stop", usage=(40, 12)),
                        ndjson(content="yes", usage=(41, 2))]
    _, done = await run(_provider(), _request(model=QWEN35, suppress_thinking=False))
    assert done.message.text == "yes" and len(fake.chat_calls) == 2
    assert tel.calls[0]["context"]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_whitespace_is_not_an_answer_and_silence_is_not_a_thought(fake, tel):
    fake.show = {QWEN35: THINKING}
    fake.completions = [ndjson(reasoning="Thinking", content="\n\n", finish="length"),
                        ndjson(content="ok"),
                        ndjson(content="", finish="stop")]            # nothing at all
    p = _provider()
    _, done = await run(p, _request(model=QWEN35, suppress_thinking=False))
    assert done.message.text == "ok" and len(tel.calls) == 1        # "\n\n" was not an answer
    _, empty = await run(p, _request(model=QWEN35, suppress_thinking=False))
    assert empty.message.text == "" and len(fake.chat_calls) == 3   # no thought: no record, no retry
    assert len(tel.calls) == 1


@pytest.mark.asyncio
async def test_the_retry_keeps_the_tools(fake, tel):
    # Coding sessions opt in AND carry tools: the retry is the same request,
    # thinking off — never a request that lost its tools.
    fake.show = {QWEN35: THINKING}
    fake.completions = [BUDGET_SPENT, ndjson(content="done")]
    await run(_provider("tools"), _request("tools", model=QWEN35, suppress_thinking=False))
    first, second = fake.chat_bodies
    assert "tools" in first and second == {**first, "think": False}


@pytest.mark.asyncio
async def test_a_transient_failure_of_the_retry_never_reruns_the_spent_attempt(fake, tel):
    fake.show = {QWEN35: THINKING}
    fake.completions = [BUDGET_SPENT, 503, ndjson(content="ok")]
    _, done = await run(_provider(), _request(model=QWEN35, suppress_thinking=False))
    assert done.message.text == "ok"
    assert ["think" in b for b in fake.chat_bodies] == [False, True, True]
    assert len(tel.calls) == 1


@pytest.mark.asyncio
async def test_a_retry_that_keeps_failing_leaves_the_first_attempt(fake, tel):
    fake.show = {QWEN35: THINKING}
    fake.completions = [BUDGET_SPENT] + [503] * 10
    _, done = await run(_provider(), _request(model=QWEN35, suppress_thinking=False))
    assert (done.message.text, done.stop_reason) == ("", "length")
    assert thinking_blocks(done) == ["Let me think about this at length"]
    assert ["think" in b for b in fake.chat_bodies][0] is False
    assert all("think" in b for b in fake.chat_bodies[1:])   # attempt 1 never re-run
    assert len(tel.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("setup", ["thinking-already-off", "gpt-oss", "capability-unknown"])
async def test_no_retry_when_thinking_off_is_not_a_fix(fake, tel, setup):
    model, suppress = QWEN35, False
    if setup == "thinking-already-off":
        fake.show, suppress = {QWEN35: THINKING}, None      # think: false was already sent
    elif setup == "gpt-oss":
        model = "my-reasoner:latest"                         # harmony by FAMILY, not by name:
        fake.show = {model: show("completion", "tools", "thinking", family="gptoss")}
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
    call = {"id": "call_1", "function": {"index": 0, "name": "read_file", "arguments": {"path": "a.txt"}}}
    fake.completions = [ndjson(reasoning="Read it.", tool_call=call, finish="stop")]
    _, done = await run(_provider("tools"), _request("tools", model=QWEN35, suppress_thinking=False))
    assert len(fake.chat_calls) == 1 and tel.calls == []
    assert [(b.name, b.input) for b in done.message.tool_uses] == [("read_file", {"path": "a.txt"})]
    assert (done.message.text, thinking_blocks(done)) == ("", ["Read it."])
    assert done.stop_reason == "tool_calls"          # what the /v1 route reported for a call


@pytest.mark.asyncio
async def test_a_plain_answer_is_untouched(fake, tel):
    fake.show = {QWEN25: NO_THINKING}
    deltas, done = await run(_provider(), _request())
    assert (deltas, done.message.text, done.stop_reason) == (["two"], "two", "stop")
    assert [type(b) for b in done.message.content] == [TextBlock]
    assert (done.usage.input_tokens, done.usage.output_tokens) == (30, 5)
    assert tel.calls == []
