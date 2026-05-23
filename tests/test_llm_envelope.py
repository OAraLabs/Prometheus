"""Sprint S4 A1 — LLMCallEnvelope tests.

Covers Sprint 4 Work Stream 1 acceptance:

  - test_envelope_records_success
  - test_envelope_records_failure
  - test_envelope_re_raises_by_default
  - test_envelope_swallows_when_explicit
  - test_envelope_validates_message_shape (cites ed8f1a6)

Plus migration sanity checks that the four pre-existing _call_model
helpers (SkillCreator / SkillRefiner / MemoryExtractor / Curator) now
construct an envelope at __init__.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.learning.llm_envelope import (
    LLMCallEnvelope,
    LLMCallResult,
    LLMCallShapeError,
)
from prometheus.providers.base import ApiTextDeltaEvent
from prometheus.telemetry.tracker import ToolCallTelemetry

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fake providers
# ---------------------------------------------------------------------------


class _ScriptedProvider:
    """Yields a fixed sequence of text deltas, then completes."""

    def __init__(self, parts: list[str]) -> None:
        self._parts = parts

    async def stream_message(self, request: Any):
        for p in self._parts:
            yield ApiTextDeltaEvent(text=p)


class _FailingProvider:
    """Raises a ValueError mid-stream — same shape an adapter error could take."""

    async def stream_message(self, request: Any):
        raise ValueError("synthetic provider failure")
        yield  # unreachable; makes this a generator


# ---------------------------------------------------------------------------
# Sprint S4 acceptance tests
# ---------------------------------------------------------------------------


class TestLLMCallEnvelope:
    """Each test corresponds to a bullet in Sprint 4 Work Stream 1 spec."""

    def test_envelope_records_success(self, tmp_path: Path) -> None:
        """A successful call writes a subsystem_runs row with outcome=success."""
        tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
        env = LLMCallEnvelope("test_sub", telemetry=tel)

        text = asyncio.run(
            env.call(
                provider=_ScriptedProvider(["hello, ", "world"]),
                model="m",
                prompt="say hi",
                operation="greet",
            )
        )
        assert text == "hello, world"

        runs = tel.runs_since(0, subsystem="test_sub")
        assert len(runs) == 1
        assert runs[0]["outcome"] == "success"
        assert runs[0]["operation"] == "greet"
        assert runs[0]["duration_ms"] >= 0

    def test_envelope_records_failure(self, tmp_path: Path) -> None:
        """An exception during the call writes a silent_failures row."""
        tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
        env = LLMCallEnvelope("test_sub", telemetry=tel, on_failure="return_none")

        result = asyncio.run(
            env.call(
                provider=_FailingProvider(),
                model="m",
                prompt="boom",
                operation="explode",
                context={"reason": "regression coverage"},
            )
        )
        assert result is None

        failures = tel.silent_failures_since(0, subsystem="test_sub")
        assert len(failures) == 1
        row = failures[0]
        assert row["operation"] == "explode"
        assert row["exception_type"] == "ValueError"
        assert "synthetic provider failure" in row["exception_msg"]
        assert row["traceback"]  # non-empty
        assert "regression coverage" in row["context"]

        # And the run is recorded as failed so /health can spot it.
        runs = tel.runs_since(0, subsystem="test_sub")
        assert any(r["outcome"] == "failed" for r in runs)

    def test_envelope_re_raises_by_default(self, tmp_path: Path) -> None:
        """on_failure='raise' (default) propagates the exception."""
        tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
        env = LLMCallEnvelope("test_sub", telemetry=tel)
        # default mode

        async def _go():
            return await env.call(
                provider=_FailingProvider(),
                model="m",
                prompt="boom",
                operation="explode",
            )

        with pytest.raises(ValueError, match="synthetic provider failure"):
            asyncio.run(_go())

        # Telemetry row still landed before the re-raise.
        failures = tel.silent_failures_since(0, subsystem="test_sub")
        assert len(failures) == 1

    def test_envelope_swallows_when_explicit(self, tmp_path: Path) -> None:
        """on_failure='log_only' returns LLMCallResult; on_failure='return_none' returns None."""
        tel = ToolCallTelemetry(db_path=tmp_path / "t.db")

        # log_only mode — returns an LLMCallResult always
        env_log = LLMCallEnvelope("test_sub_log", telemetry=tel, on_failure="log_only")
        result = asyncio.run(
            env_log.call(
                provider=_FailingProvider(),
                model="m",
                prompt="boom",
                operation="explode",
            )
        )
        assert isinstance(result, LLMCallResult)
        assert result.text == ""
        assert result.error is not None
        assert "synthetic provider failure" in str(result.error)

        # Same envelope on success path also returns LLMCallResult
        ok = asyncio.run(
            env_log.call(
                provider=_ScriptedProvider(["all good"]),
                model="m",
                prompt="hi",
                operation="greet",
            )
        )
        assert isinstance(ok, LLMCallResult)
        assert ok.text == "all good"
        assert ok.error is None

        # return_none mode — plain str | None
        env_none = LLMCallEnvelope("test_sub_none", telemetry=tel, on_failure="return_none")
        out = asyncio.run(
            env_none.call(
                provider=_FailingProvider(),
                model="m",
                prompt="boom",
                operation="explode",
            )
        )
        assert out is None

        # And both modes wrote silent_failures rows.
        for sub in ("test_sub_log", "test_sub_none"):
            assert tel.silent_failures_since(0, subsystem=sub)

    def test_envelope_validates_message_shape(self) -> None:
        """validate_messages catches the ed8f1a6 shape explicitly.

        Cites PR #1 / ed8f1a6: ConversationMessage.content must be
        list[ContentBlock]. The envelope's primary API (which takes a
        plain prompt string) makes the bug structurally impossible at the
        call site, but `validate_messages` exists so future callers that
        pre-build messages can be pre-flighted.
        """
        # Happy path — well-formed message
        good = ConversationMessage(role="user", content=[TextBlock(text="hi")])
        LLMCallEnvelope.validate_messages([good])

        # ed8f1a6 shape — content is a str. Bypass pydantic so we can
        # construct the bad shape that production code accidentally built
        # before ed8f1a6 fixed it.
        bad = ConversationMessage(role="user", content=[TextBlock(text="hi")])
        object.__setattr__(bad, "content", "raw prompt as a string")
        with pytest.raises(LLMCallShapeError, match="ed8f1a6 shape"):
            LLMCallEnvelope.validate_messages([bad])

        # dict-shaped block — also rejected
        bad2 = ConversationMessage(role="user", content=[TextBlock(text="hi")])
        object.__setattr__(bad2, "content", [{"fake": "block"}])
        with pytest.raises(LLMCallShapeError, match="expected a ContentBlock"):
            LLMCallEnvelope.validate_messages([bad2])

        # Non-ConversationMessage entry
        with pytest.raises(LLMCallShapeError, match="expected ConversationMessage"):
            LLMCallEnvelope.validate_messages(["not a message"])  # type: ignore[list-item]


class TestLLMCallEnvelopeConstructionGuards:
    """Defensive checks on the envelope's own constructor."""

    def test_invalid_on_failure_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="on_failure must be one of"):
            LLMCallEnvelope("x", on_failure="silent")  # type: ignore[arg-type]

    def test_telemetry_optional(self) -> None:
        """An envelope with telemetry=None still routes the call; failures still re-raise / return None."""
        env = LLMCallEnvelope("x", telemetry=None, on_failure="return_none")
        result = asyncio.run(
            env.call(
                provider=_FailingProvider(),
                model="m",
                prompt="boom",
                operation="explode",
            )
        )
        assert result is None  # behavior preserved without a telemetry sink

    def test_telemetry_write_failure_does_not_break_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If telemetry plumbing is broken, the call itself still succeeds.

        This is the "telemetry must never break the action" invariant from
        the audit. We monkeypatch the telemetry write to raise.
        """
        tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
        monkeypatch.setattr(
            tel, "record_run",
            MagicMock(side_effect=RuntimeError("telemetry broken")),
        )
        env = LLMCallEnvelope("x", telemetry=tel)
        text = asyncio.run(
            env.call(
                provider=_ScriptedProvider(["ok"]),
                model="m",
                prompt="hi",
                operation="probe",
            )
        )
        assert text == "ok"  # action succeeded despite broken telemetry


class TestSubsystemEnvelopeMigration:
    """Sanity: each of the four pre-existing _call_model helpers now uses LLMCallEnvelope."""

    def test_skill_creator_constructs_envelope(self) -> None:
        from prometheus.learning.skill_creator import SkillCreator

        creator = SkillCreator(MagicMock(), model="m", telemetry=None)
        assert isinstance(creator._envelope, LLMCallEnvelope)
        assert creator._envelope._subsystem == "skill_creator"
        assert creator._envelope._on_failure == "return_none"

    def test_skill_refiner_constructs_envelope(self) -> None:
        from prometheus.learning.skill_refiner import SkillRefiner

        refiner = SkillRefiner(MagicMock(), model="m", telemetry=None)
        assert isinstance(refiner._envelope, LLMCallEnvelope)
        assert refiner._envelope._subsystem == "skill_refiner"
        assert refiner._envelope._on_failure == "return_none"

    def test_memory_extractor_constructs_envelope(self) -> None:
        from prometheus.memory.extractor import MemoryExtractor
        from prometheus.memory.store import MemoryStore

        store = MagicMock(spec=MemoryStore)
        extractor = MemoryExtractor(store, MagicMock(), model="m", telemetry=None)
        assert isinstance(extractor._envelope, LLMCallEnvelope)
        assert extractor._envelope._subsystem == "memory_extractor"
        assert extractor._envelope._on_failure == "return_none"

    def test_curator_constructs_envelope(self) -> None:
        from prometheus.learning.curator import Curator

        curator = Curator(MagicMock(), model="m", telemetry=None)
        assert isinstance(curator._envelope, LLMCallEnvelope)
        assert curator._envelope._subsystem == "curator"
        # Curator uses 'raise' so run_once's outer try/except keeps working.
        assert curator._envelope._on_failure == "raise"
