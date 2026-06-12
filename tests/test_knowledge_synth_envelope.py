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

import re
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


def test_insight_page_slug_confined_to_queries_dir(tmp_path: Path) -> None:
    """Entity names are model-extracted free text — the live 2026-06-12
    cycle produced entities like ``/home/will/promet`` and ``$backup_root``,
    whose separators made the filename point into a nonexistent directory
    (FileNotFoundError) and could in principle traverse out of the wiki.
    The slug must be a single safe path component inside wiki/queries."""
    from prometheus.sentinel.knowledge_synth import SynthInsight

    ks = KnowledgeSynthesizer(
        store=_FakeStore(), provider=_CapturingProvider(), model="m",
        wiki_root=tmp_path / "wiki",
    )
    hostile = SynthInsight(
        entities=["$backup_root", ".srt-file-/home/will/promet", "../../evil"],
        insight="text",
        tokens_used=1,
    )

    ks._write_insight_page(hostile)  # crashed pre-fix

    queries = tmp_path / "wiki" / "queries"
    written = list(queries.glob("insight-*.md"))
    assert len(written) == 1
    page = written[0]
    # Confined: direct child of queries/, safe charset, nothing escaped.
    assert page.parent == queries
    assert re.fullmatch(r"insight-\d{8}-[a-z0-9-]+\.md", page.name)
    assert not list((tmp_path / "wiki").rglob("evil*"))
    assert "$backup_root" in page.read_text(encoding="utf-8")  # content keeps raw entities


def test_insight_page_slug_empty_entities_fallback(tmp_path: Path) -> None:
    """All-junk entities must still yield a writable name, not ``insight--.md``."""
    from prometheus.sentinel.knowledge_synth import SynthInsight

    ks = KnowledgeSynthesizer(
        store=_FakeStore(), provider=_CapturingProvider(), model="m",
        wiki_root=tmp_path / "wiki",
    )
    ks._write_insight_page(SynthInsight(entities=["///", "$$$"], insight="t", tokens_used=1))

    written = list((tmp_path / "wiki" / "queries").glob("insight-*-cluster.md"))
    assert len(written) == 1
