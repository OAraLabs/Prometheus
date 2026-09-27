"""A microcompact row names the turn's conversation (WP-X.21 T5, docs/audits/TELEMETRY-GAPS.md).

Microcompaction rewrites old tool results in the history, which also throws away
the provider's cached prompt prefix, so it records a ``subsystem_runs`` row each
time it fires. That row took its session from ``context.session_id``. On the web
path the daemon shares ONE context across every conversation and pins the
routing namespace ``"web"`` on it, so every web conversation's rows were filed
under ``"web"`` — 468 in the 30 days before the audit — and none could be joined
back to the conversation whose history was rewritten. #258 and #458 fixed the
same substitution in the loop's other writers; this one was missed.

Pinned here, through a real ``run_loop`` turn and the rows actually written:

* the row carries the TURN's session (the per-call argument), not the shared
  context's namespace;
* an ephemeral turn's row carries no session, like every other row the turn
  writes;
* a context whose own session IS the conversation (Telegram, the CLI) records
  it as before.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

CONVERSATION = "web:conv-1"


class _NInput(BaseModel):
    n: int


class _BigRead(BaseTool):
    """Returns a large result that differs per call, so the repeat detector
    never sees the same bytes twice."""

    name = "big_read"
    description = "returns a large result"
    input_model = _NInput

    def is_read_only(self, arguments) -> bool:  # noqa: ANN001
        return True

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output=f"result {arguments.n}\n" + "x" * 2000, is_error=False)


class _ThreeReadsProvider(ModelProvider):
    """Rounds 0-2 each call big_read; round 3 answers. Microcompaction runs at
    the top of round 3 and trims round 0's result."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream_message(self, request):  # noqa: ANN001
        n = self.calls
        self.calls += 1
        if n < 3:
            content = [ToolUseBlock(id=f"t{n}", name="big_read", input={"n": n})]
        else:
            content = [TextBlock(text="done")]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


def _microcompact_rows(tmp_path, *, context_session: str | None, turn_session: str | None) -> list:
    from prometheus.adapter import ModelAdapter

    registry = ToolRegistry()
    registry.register(_BigRead())
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    ctx = LoopContext(
        provider=_ThreeReadsProvider(),
        model="stub-model",
        system_prompt="",
        max_tokens=128,
        tool_registry=registry,
        adapter=ModelAdapter(tier="light"),  # a local tier: microcompaction runs
        telemetry=tel,
        session_id=context_session,
        microcompact_after_turns=2,
    )

    async def drain() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")], session_id=turn_session):
            pass

    asyncio.run(drain())
    tel.close()
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT session_id, round_index FROM subsystem_runs"
        " WHERE subsystem = 'agent_loop' AND operation = 'microcompact' ORDER BY rowid"
    ).fetchall()
    con.close()
    assert rows, "the harness must make microcompaction fire, or these tests measure nothing"
    return rows


def test_a_web_turn_files_its_row_under_its_conversation_not_the_namespace(tmp_path):
    # The web path: one context shared by every conversation, pinned to "web";
    # the turn's own conversation arrives as run_loop's session_id argument.
    rows = _microcompact_rows(tmp_path, context_session="web", turn_session=CONVERSATION)
    assert {s for s, _ in rows} == {CONVERSATION}


def test_an_ephemeral_turn_files_its_row_under_no_session(tmp_path):
    from prometheus.config.ephemeral import set_session_ephemeral

    set_session_ephemeral(CONVERSATION, True)
    rows = _microcompact_rows(tmp_path, context_session="web", turn_session=CONVERSATION)
    assert {s for s, _ in rows} == {None}, "an ephemeral turn's rows name no session"


def test_a_context_whose_session_is_the_conversation_keeps_it(tmp_path):
    # Telegram and the CLI build a context per conversation and pass no
    # per-call id: the context's own session is the turn's.
    rows = _microcompact_rows(tmp_path, context_session="telegram:42", turn_session=None)
    assert {s for s, _ in rows} == {"telegram:42"}


@pytest.fixture(autouse=True)
def _no_global_handle(monkeypatch):
    import prometheus.telemetry.tracker as tracker

    monkeypatch.setattr(tracker, "_telemetry_singleton", None)
