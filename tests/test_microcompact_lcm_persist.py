"""Does lcm.db keep a tool result whole when microcompaction trims it mid-run?

Microcompaction (``agent_loop._microcompact_old_results``) cuts old tool results
down to an excerpt so a small local window survives a long run. It runs at the
top of each round and replaces the result block IN PLACE on the message objects
the loop holds. Both persist paths share those objects with the session and
write a run's new messages to LCM only when the run ends:

* the gateways (Telegram/Slack/Discord) call ``AgentLoop.run_async``, which
  shallow-copies the session list, then ``ChatSession.add_result_messages``;
* the WS bridge runs ``run_loop`` over ``session.get_messages()`` itself, then
  ``ChatSession.persist_loop_result``.

So a run long enough for microcompaction to reach its own earliest results
would write the trimmed text to lcm.db, while ``lcm_grep``/``lcm_expand`` and
the memory extractor all read lcm.db as the record of what happened.

Every test runs a real loop on a local adapter tier (microcompaction on, the
shipped ``microcompact_after_turns=3``) against a real LCMEngine on a temp
database, and reads the rows back. The cloud-tier control runs the same session
with microcompaction skipped, so a pass there shows the reading side can see a
whole result.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from prometheus.adapter import ModelAdapter
from prometheus.engine.agent_loop import AgentLoop, LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.session import ChatSession
from prometheus.engine.usage import UsageSnapshot
from prometheus.memory.lcm_engine import LCMEngine
from prometheus.memory.lcm_types import CompactionConfig
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

TRIMMED = "[microcompacted]"


def _body(n: int) -> str:
    """A long result whose LAST line only survives if nothing cut it."""
    lines = [f"read {n} begins"]
    lines += [f"line {i} of read {n}: the quick brown fox jumps" for i in range(60)]
    lines.append(f"read {n} ends")
    return "\n".join(lines)


class _NInput(BaseModel):
    n: int


class _Read(BaseTool):
    name = "read_chunk"
    description = "returns a long chunk"
    input_model = _NInput

    def __init__(self, body=_body) -> None:  # noqa: ANN001
        super().__init__()
        self._make = body

    def is_read_only(self, arguments) -> bool:  # noqa: ANN001
        return True

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output=self._make(arguments.n), is_error=False)


class _Reads(ModelProvider):
    """Calls read_chunk ``rounds`` times (n = first, first+1, ...), then answers."""

    def __init__(self, rounds: int, first: int = 0) -> None:
        self.rounds = rounds
        self.first = first
        self.calls = 0
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        k = self.calls
        self.calls += 1
        if k < self.rounds:
            n = self.first + k
            content = [ToolUseBlock(id=f"t{n}", name="read_chunk", input={"n": n})]
        else:
            content = [TextBlock(text="done")]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


def _registry(body=_body) -> ToolRegistry:  # noqa: ANN001
    registry = ToolRegistry()
    registry.register(_Read(body))
    return registry


def _engine(tmp_path) -> LCMEngine:
    # Explicit config: never read a prometheus.yaml off the search path.
    return LCMEngine(_Reads(0), config=CompactionConfig(), db_path=tmp_path / "lcm.db")


def _in_memory(session: ChatSession) -> str:
    return "\n".join(
        getattr(block, "content", "") or ""
        for msg in session.get_messages()
        if isinstance(msg.content, list)
        for block in msg.content
        if isinstance(getattr(block, "content", None), str)
    )


def _stored(engine: LCMEngine, session_id: str) -> str:
    rows = engine.conversation_store.get_all_messages(session_id)
    return "\n".join(f"{r.content}\n{r.content_json or ''}" for r in rows)


def _ws_turn(engine, session, *, rounds, first=0, tier="light", body=_body) -> _Reads:  # noqa: ANN001
    """The WS bridge's shape: run_loop over the session's own list, then persist."""
    provider = _Reads(rounds, first)
    ctx = LoopContext(
        provider=provider,
        model="stub-model",
        system_prompt="",
        max_tokens=128,
        tool_registry=_registry(body),
        adapter=ModelAdapter(tier=tier),
        lcm_engine=engine,
        session_id="web",
    )

    async def go() -> None:
        session.add_user_message("go")
        messages = session.get_messages()
        original_len = len(messages)
        async for _ in run_loop(ctx, messages, session_id=session.session_id):
            pass
        session.persist_loop_result(original_len)

    asyncio.run(go())
    return provider


def _gateway_turn(engine, session, *, rounds, first=0, tier="light", body=_body) -> _Reads:  # noqa: ANN001
    """The Telegram/Slack/Discord shape: run_async, then add_result_messages."""
    provider = _Reads(rounds, first)
    loop = AgentLoop(
        provider,
        model="stub-model",
        max_tokens=128,
        tool_registry=_registry(body),
        adapter=ModelAdapter(tier=tier),
        lcm_engine=engine,
    )

    async def go() -> None:
        session.add_user_message("go")
        pre_len = len(session.get_messages())
        result = await loop.run_async(
            system_prompt="",
            messages=session.get_messages(),
            session_id=session.session_id,
            session_state=session,
        )
        session.add_result_messages(result.messages, pre_len)

    asyncio.run(go())
    return provider


ROUNDS = 6  # with after_turns=3, the earliest results are trimmed before the run ends


@pytest.mark.parametrize("turn", [_ws_turn, _gateway_turn], ids=["ws_bridge", "gateway"])
def test_a_long_local_run_stores_every_tool_result_whole(tmp_path, turn):
    engine = _engine(tmp_path)
    session = ChatSession("web:conv-1", lcm_engine=engine)
    turn(engine, session, rounds=ROUNDS)

    assert TRIMMED in _in_memory(session), (
        "harness: microcompaction never fired, so this test measures nothing"
    )
    stored = _stored(engine, session.session_id)
    assert all(f"read {n} begins" in stored for n in range(ROUNDS)), (
        "harness: not every tool result reached lcm.db at all"
    )
    cut = [n for n in range(ROUNDS) if f"read {n} ends" not in stored]
    assert not cut and TRIMMED not in stored, (
        f"lcm.db holds the microcompacted text of tool results {cut}, not the "
        f"originals: the in-run trim reached the durable record"
    )


@pytest.mark.parametrize("turn", [_ws_turn, _gateway_turn], ids=["ws_bridge", "gateway"])
def test_control_a_cloud_run_skips_microcompaction_and_stores_whole(tmp_path, turn):
    engine = _engine(tmp_path)
    session = ChatSession("web:conv-1", lcm_engine=engine)
    turn(engine, session, rounds=ROUNDS, tier="off")

    assert TRIMMED not in _in_memory(session)
    stored = _stored(engine, session.session_id)
    assert all(f"read {n} ends" in stored for n in range(ROUNDS))
    assert TRIMMED not in stored


@pytest.mark.parametrize("turn", [_ws_turn, _gateway_turn], ids=["ws_bridge", "gateway"])
def test_a_later_turn_trimming_older_results_leaves_their_rows_alone(tmp_path, turn):
    # Turn 1 is too short to trim anything, so its results persist whole. Turn 2
    # then trims them in the session's memory; their durable rows must not move.
    engine = _engine(tmp_path)
    session = ChatSession("web:conv-1", lcm_engine=engine)
    turn(engine, session, rounds=2)
    assert TRIMMED not in _in_memory(session)

    turn(engine, session, rounds=ROUNDS, first=100)
    assert "read 0 ends" not in _in_memory(session), (
        "harness: turn 2 never trimmed turn 1's results in memory"
    )
    stored = _stored(engine, session.session_id)
    assert "read 0 ends" in stored and "read 1 ends" in stored


# ── the full text rides the lcm.db write and nothing else ──────────────────


def _trimmed_ids(text: str) -> list[int]:
    return [n for n in range(200) if f"{TRIMMED} read {n} begins" in text]


def _request_wire(request) -> str:  # noqa: ANN001
    """Every serialization a provider builds from a request, as one string."""
    import json

    from prometheus.engine.messages import render_messages_for_model
    from prometheus.providers.anthropic import _build_anthropic_messages
    from prometheus.providers.llama_cpp import LlamaCppProvider

    rendered = render_messages_for_model(request.messages)
    parts = [
        json.dumps(LlamaCppProvider()._build_request_payload(request), default=str),
        json.dumps(_build_anthropic_messages(rendered), default=str),
        json.dumps([m.to_openai_param() for m in rendered], default=str),
        "\n".join(m.model_dump_json() for m in request.messages),
        "\n".join(m.content_json for m in request.messages),
        "\n".join(repr(m) + str(m.content) for m in request.messages),
    ]
    return "\n".join(parts)


def test_a_trimmed_results_full_text_reaches_no_request_and_no_frame(tmp_path):
    """Condition 2 of the fix: the full text exists only for the lcm.db write.

    Every request the provider received is serialized the ways the providers
    serialize it (llama.cpp payload, Anthropic messages, OpenAI params, the
    pydantic dumps, repr). Once a result has been cut down, none of them may
    carry its full text. Then the WS ``switch_session`` frames — the one place
    a client is sent in-memory history — are checked the same way.
    """
    import json

    from prometheus.engine.session import SessionManager
    from prometheus.web.ws_server import WebSocketBridge

    engine = _engine(tmp_path)
    mgr = SessionManager()
    mgr.lcm_engine = engine
    session = mgr.get_or_create("web:conv-1")
    provider = _ws_turn(engine, session, rounds=ROUNDS)

    trimmed_any = False
    for request in provider.requests:
        wire = _request_wire(request)
        for n in _trimmed_ids(wire):
            trimmed_any = True
            assert f"read {n} ends" not in wire, (
                f"the full text of trimmed result {n} reached a provider request"
            )
    assert trimmed_any, "harness: no request ever carried a trimmed result"

    class _Recorder:
        def __init__(self) -> None:
            self.frames: list[dict] = []

        async def send(self, raw: str) -> None:
            self.frames.append(json.loads(raw))

    ws = _Recorder()
    bridge = WebSocketBridge(session_mgr=mgr)
    asyncio.run(bridge._handle_client_message(
        ws, json.dumps({"type": "switch_session", "payload": {"session_id": "web:conv-1"}})
    ))
    frames = json.dumps(ws.frames)
    cut = _trimmed_ids(frames)
    assert cut, "harness: the history frames carried no trimmed result"
    assert all(f"read {n} ends" not in frames for n in cut), (
        "the full text of a trimmed result reached a switch_session frame"
    )


# ── redaction still applies to the full text ──────────────────────────────


def _token() -> str:
    # Built at runtime: a GitHub classic token shape the store redacts. Never a
    # literal, so the pre-commit secret scanner has nothing to find here.
    return "gh" + "p_" + ("aB3xY9" * 6)[:36]


def _body_with_secret(n: int) -> str:
    lines = [f"read {n} begins"]
    lines += [f"line {i} of read {n}: the quick brown fox jumps" for i in range(30)]
    lines.append(f"config dump: GITHUB_TOKEN={_token()}")  # far past the 500-char excerpt
    lines += [f"line {i} of read {n}: the quick brown fox jumps" for i in range(30, 60)]
    lines.append(f"read {n} ends")
    return "\n".join(lines)


@pytest.mark.parametrize("turn", [_ws_turn, _gateway_turn], ids=["ws_bridge", "gateway"])
def test_a_secret_in_a_trimmed_result_is_stored_redacted(tmp_path, turn):
    from prometheus.security.log_redaction import REDACTED

    engine = _engine(tmp_path)
    session = ChatSession("web:conv-1", lcm_engine=engine)
    turn(engine, session, rounds=ROUNDS, body=_body_with_secret)

    assert TRIMMED in _in_memory(session), "harness: microcompaction never fired"
    stored = _stored(engine, session.session_id)
    assert all(f"read {n} ends" in stored for n in range(ROUNDS)), (
        "the full text of a trimmed result did not reach lcm.db"
    )
    assert _token() not in stored, "a secret in a trimmed result reached lcm.db unredacted"
    assert stored.count(f"GITHUB_TOKEN={REDACTED}") >= ROUNDS


# ── what a restart restores ───────────────────────────────────────────────


def test_a_rehydrated_session_gets_the_full_result_back(tmp_path):
    """Deliberate, and the one request-side effect of storing the full text.

    With ``sessions.rehydrate`` on (off in the shipped config), a cold session
    is rebuilt from lcm.db after a restart. A result trimmed in its last run
    used to come back as the excerpt, because that was all lcm.db held. It now
    comes back whole (or shortened at restart, inside the restore budget).
    """
    from prometheus.engine.session import SessionManager

    engine = _engine(tmp_path)
    before = SessionManager()
    before.lcm_engine = engine
    _ws_turn(engine, before.get_or_create("web:conv-1"), rounds=ROUNDS)

    after = SessionManager()  # the restart
    after.lcm_engine = engine
    after.rehydrate_enabled = True
    assert after.rehydrate_if_cold("web:conv-1") > 0
    restored = _in_memory(after.get_or_create("web:conv-1"))
    assert TRIMMED not in restored
    assert "read 0 ends" in restored or "shortened at restart" in restored
