"""Regression: KnowledgeSynthesizer builds a valid message + surfaces failures.

Before fix/knowledge-synth-envelope, ``_generate_insight`` constructed
``ConversationMessage(role="user", content=prompt)`` — a str where a
``list[ContentBlock]`` is required — raising pydantic ``list_type`` on EVERY
AutoDream dream cycle (the phase had never produced an insight). Routing the
call through ``LLMCallEnvelope`` makes the correct ``content=[TextBlock(...)]``
shape structural and pushes failures into telemetry.

Side effects asserted on the OUTGOING request payload the provider receives
and on the telemetry DB — never by counting calls.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.providers.base import ApiTextDeltaEvent
from prometheus.sentinel.knowledge_synth import KnowledgeSynthesizer
from prometheus.telemetry.tracker import ToolCallTelemetry


class _FakeStore:
    """Returns >=3 facts so synthesis proceeds."""

    def search_memories(self, entity: str, limit: int):
        return [
            {"fact": f"{entity} fact {i}", "confidence": 0.9} for i in range(2)
        ]


class _CapturingProvider:
    """Captures the ApiMessageRequest it is handed; streams a canned insight."""

    def __init__(self) -> None:
        self.request = None

    async def stream_message(self, request):  # noqa: ANN001 — provider duck type
        self.request = request
        yield ApiTextDeltaEvent(text="Entities A and B both relate to X.")


class _RaisingProvider:
    async def stream_message(self, request):  # noqa: ANN001
        raise RuntimeError("model down")
        yield  # pragma: no cover — makes this an async generator


async def test_request_message_content_is_contentblock_list() -> None:
    provider = _CapturingProvider()
    ks = KnowledgeSynthesizer(store=_FakeStore(), provider=provider, model="m")

    insight = await ks._generate_insight(["EntA", "EntB"])

    # The call succeeded → an insight came back (it never could, pre-fix).
    assert insight is not None
    assert "relate to X" in insight.insight
    assert insight.tokens_used >= 1

    # Load-bearing: the OUTGOING request's message content is a LIST of
    # ContentBlock (the bug passed a bare str here → pydantic list_type).
    req = provider.request
    assert req is not None
    msg = req.messages[0]
    assert isinstance(msg, ConversationMessage)
    assert isinstance(msg.content, list)
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], TextBlock)
    assert "EntA" in msg.content[0].text  # the prompt rode through intact


async def test_old_str_content_shape_would_have_raised() -> None:
    """Pin the exact failure the fix prevents: a str content is rejected."""
    with pytest.raises(Exception) as exc:  # pydantic ValidationError
        ConversationMessage(role="user", content="a bare string")
    assert "list" in str(exc.value).lower()


async def test_call_failure_surfaces_to_telemetry(tmp_path: Path) -> None:
    telemetry = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    ks = KnowledgeSynthesizer(
        store=_FakeStore(), provider=_RaisingProvider(), model="m",
        telemetry=telemetry,
    )

    # No insight on failure (the legacy contract is preserved)...
    assert await ks._generate_insight(["EntA", "EntB"]) is None

    # ...AND the failure is now visible in telemetry.silent_failures (side
    # effect: a row exists), where pre-fix it was only an ERROR log.
    conn = sqlite3.connect(tmp_path / "telemetry.db")
    try:
        rows = conn.execute(
            "SELECT subsystem, operation FROM silent_failures").fetchall()
    finally:
        conn.close()
    assert ("knowledge_synth", "generate_insight") in rows
