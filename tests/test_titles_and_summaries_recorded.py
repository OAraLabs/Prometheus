"""Session titles leave a usage row, and titles and LCM summaries name their conversation
(WP-X.21 T11; docs/audits/TELEMETRY-GAPS.md).

The title call ran once per new conversation and wrote nothing; LCM summaries now
leave a row (the envelope, earlier in this bundle) but it named no conversation,
though the compactor that asks for one works on exactly one. Both rows now carry the
conversation under the loop's rule: none at all on an ephemeral turn. The requests
are unchanged.

Pinned here through ``maybe_title_session`` and the real LCM compactor, against the
rows actually written.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ApiTextDeltaEvent, ModelProvider
from prometheus.telemetry.tracker import ToolCallTelemetry


class _Answers(ModelProvider):
    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        yield ApiTextDeltaEvent(text=self.text)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text=self.text)]),
            usage=UsageSnapshot(input_tokens=120, output_tokens=6), stop_reason="stop")


@pytest.fixture
def tel(tmp_path, monkeypatch):
    import prometheus.telemetry.tracker as tracker

    handle = ToolCallTelemetry(tmp_path / "telemetry.db")
    monkeypatch.setattr(tracker, "_telemetry_singleton", handle)
    yield handle
    handle.close()


def _rows(tel, subsystem: str) -> list[tuple]:
    return tel._conn.execute(
        "SELECT operation, outcome, session_id, input_tokens, output_tokens FROM subsystem_runs"
        " WHERE subsystem = ? ORDER BY rowid", (subsystem,)).fetchall()


class _TitleStore:
    def __init__(self) -> None:
        self.titles: dict[str, str] = {}

    def get_session_title(self, session_id):  # noqa: ANN001
        return self.titles.get(session_id)

    def set_session_title(self, session_id, title):  # noqa: ANN001
        self.titles[session_id] = title


def _title(session_id: str):
    from prometheus.engine.session_titles import maybe_title_session

    store, provider = _TitleStore(), _Answers("Gear inventory")
    exchange = [ConversationMessage.from_user_text("how many gears are there?"),
                ConversationMessage(role="assistant", content=[TextBlock(text="There are 3.")])]
    asyncio.run(maybe_title_session(store, provider, "Qwen3.8-27B", session_id, exchange))
    return store, provider


def test_a_title_call_leaves_a_row_that_names_its_conversation(tel):
    store, provider = _title("desktop:t1")
    assert store.titles == {"desktop:t1": "Gear inventory"}, "the title itself is unchanged"
    assert _rows(tel, "session_titles") == [("generate", "success", "desktop:t1", 120, 6)]
    [request] = provider.requests
    assert request.system_prompt == "You name conversations. Respond with only the title."


def test_an_ephemeral_conversations_title_row_names_no_session(tel):
    from prometheus.config.ephemeral import set_session_ephemeral

    set_session_ephemeral("desktop:t2", True)
    _title("desktop:t2")
    assert _rows(tel, "session_titles") == [("generate", "success", None, 120, 6)]


def test_an_lcm_compactions_summary_rows_name_the_conversation(tel, tmp_path):
    from prometheus.memory.lcm_compaction import LCMCompactor
    from prometheus.memory.lcm_conversation_store import LCMConversationStore
    from prometheus.memory.lcm_summarize import LCMSummarizer
    from prometheus.memory.lcm_summary_store import LCMSummaryStore
    from prometheus.memory.lcm_types import CompactionConfig, MessagePart

    db = tmp_path / "lcm.db"
    conv, sums = LCMConversationStore(db_path=db), LCMSummaryStore(db_path=db)
    compactor = LCMCompactor(conv, sums, LCMSummarizer(_Answers("a summary")),
                             CompactionConfig(fresh_tail_count=2, compaction_batch_size=3))
    for i in range(10):
        conv.insert_message(MessagePart(role="user", content=f"message {i}", session_id="telegram:5",
                                        turn_index=i, token_count=5))
    result = asyncio.run(compactor.compact("telegram:5"))
    conv.close()
    sums.close()
    rows = _rows(tel, "lcm_summarizer")
    assert result.summaries_created >= 1 and len(rows) == result.summaries_created
    assert {r[2] for r in rows} == {"telegram:5"}, "the compactor works on one conversation"
