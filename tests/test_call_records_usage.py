"""A ``call()`` row records the completion's tokens, and the compactor's its conversation
(WP-X.21 T10, docs/audits/TELEMETRY-GAPS.md).

``LLMCallEnvelope.call()`` streamed the completion and kept only its text, so every
row it wrote had no tokens: the compactor's ``summarize_span`` calls on the 4090
(123 / 175 in the audit's windows), and every other ``call()`` user — the memory
extractor, knowledge synth, skill creator and refiner, curator. The compactor's
conversation went into the summary, not the ``session_id`` column.

Pinned here through the real compactor and the real envelope, against the rows
actually written.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from prometheus.context.compactor import ContextCompactor
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.learning.llm_envelope import LLMCallEnvelope
from prometheus.providers.base import ApiMessageCompleteEvent, ApiTextDeltaEvent, ModelProvider
from prometheus.telemetry.tracker import ToolCallTelemetry

USAGE = UsageSnapshot(input_tokens=812, output_tokens=40, cached_input_tokens=600, cache_write_tokens=0)


class _Summarizer(ModelProvider):
    async def stream_message(self, request):  # noqa: ANN001
        yield ApiTextDeltaEvent(text="SUMMARY")
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text="SUMMARY")]),
            usage=USAGE, stop_reason="stop")


def _history(n_turns: int, filler: int = 400) -> list[ConversationMessage]:
    msgs: list[ConversationMessage] = []
    for i in range(n_turns):
        msgs.append(ConversationMessage.from_user_text(f"question {i}: " + "x" * filler))
        msgs.append(ConversationMessage(role="assistant",
                                        content=[TextBlock(text=f"answer {i}: " + "y" * filler)]))
    return msgs


def _summary_rows(tmp_path, session_id: str) -> list[sqlite3.Row]:
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    compactor = ContextCompactor(provider=_Summarizer(), model="test-model", effective_limit=3000,
                                 reserve_tokens=500, threshold_pct=0.4, protect_recent_turns=3,
                                 telemetry=tel)
    asyncio.run(compactor.apply(_history(12), session_id=session_id))
    tel.close()
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT input_tokens, output_tokens, cached_input_tokens, cache_write_tokens,"
                       " session_id FROM subsystem_runs WHERE operation='summarize_span'").fetchall()
    con.close()
    assert rows, "the history must be long enough to summarise, or this measures nothing"
    return rows


def test_a_summary_call_records_its_tokens_and_its_conversation(tmp_path):
    [row] = _summary_rows(tmp_path, "telegram:42")
    assert (row["input_tokens"], row["output_tokens"]) == (812, 40), "call() kept only the text"
    assert (row["cached_input_tokens"], row["cache_write_tokens"]) == (600, 0)
    assert row["session_id"] == "telegram:42", "the conversation went into the summary, not the column"


def test_an_ephemeral_conversations_summary_names_no_session(tmp_path):
    from prometheus.config.ephemeral import set_session_ephemeral

    set_session_ephemeral("telegram:43", True)
    [row] = _summary_rows(tmp_path, "telegram:43")
    assert row["session_id"] is None, "the loop's rule: no session on an ephemeral turn"
    assert row["input_tokens"] == 812


def test_any_call_user_records_the_completions_tokens(tmp_path):
    """Every call() subsystem gains its tokens; a call that serves no conversation
    passes no session and records none."""
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    envelope = LLMCallEnvelope("memory_extractor", telemetry=tel, on_failure="return_none")
    text = asyncio.run(envelope.call(provider=_Summarizer(), model="m", prompt="p", operation="extract"))
    tel.close()
    assert text == "SUMMARY"
    con = sqlite3.connect(db)
    [row] = con.execute("SELECT input_tokens, output_tokens, session_id FROM subsystem_runs"
                        " WHERE subsystem='memory_extractor'").fetchall()
    con.close()
    assert row == (812, 40, None)


@pytest.fixture(autouse=True)
def _no_global_handle(monkeypatch):
    import prometheus.telemetry.tracker as tracker

    monkeypatch.setattr(tracker, "_telemetry_singleton", None)
