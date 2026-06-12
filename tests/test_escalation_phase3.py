"""Phase 3 integration — post-turn hook, session/LCM trust tagging, /escalations.

End-to-end with a mocked agent loop returning a recorded failed turn and a
recorded teacher fixture streamed through the real envelope: the corrective
reply is delivered with a visible note, the teacher message lands in the
session AND lcm.db tagged ("teacher_escalation", untrusted), the SKILL.md
exists on disk, and the golden-trace row exists. No live network.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from prometheus.engine.agent_loop import RunResult
from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.engine.session import SessionManager
from prometheus.escalation.teacher import TeacherEscalation, build_trace_from_messages
from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.telegram import TelegramAdapter
from prometheus.learning.skill_creator import SkillCreator
from prometheus.memory.lcm_engine import LCMEngine
from prometheus.memory.lcm_types import CompactionConfig
from prometheus.providers.base import ApiTextDeltaEvent
from prometheus.telemetry.tracker import ToolCallTelemetry

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "escalation"
GOOD_TEACHER = (FIXTURE_DIR / "teacher_reply_good.md").read_text(encoding="utf-8")
STALLED_TEACHER = (FIXTURE_DIR / "teacher_reply_stalled.md").read_text(encoding="utf-8")

REQUEST = "Deploy the new build to the staging box."
FAILED_REPLY = "The deploy script could not reach the server."


class _FakeTeacherProvider:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        yield ApiTextDeltaEvent(text=self._text)


def _failed_turn_messages() -> list[ConversationMessage]:
    """A recorded failed agent turn: tool error, no recovery in the reply."""
    return [
        ConversationMessage.from_user_text(REQUEST),  # index 0 — pre_len slice
        ConversationMessage(
            role="assistant",
            content=[ToolUseBlock(id="tu_1", name="bash",
                                  input={"command": "./deploy.sh"})],
        ),
        ConversationMessage(
            role="user",
            content=[ToolResultBlock(
                tool_use_id="tu_1",
                content="Error: connection refused (deploy target unreachable)",
                is_error=True,
            )],
        ),
        ConversationMessage(
            role="assistant", content=[TextBlock(text=FAILED_REPLY)]),
    ]


def _build_adapter(tmp_path: Path, teacher_text: str | None):
    """Real adapter + session + LCM + telemetry; mocked agent loop."""
    telemetry = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    lcm = LCMEngine(MagicMock(), config=CompactionConfig(),
                    db_path=tmp_path / "lcm.db")
    sm = SessionManager()
    sm.lcm_engine = lcm
    session = sm.get_or_create("telegram:99")

    agent_loop = MagicMock()
    agent_loop._model_router = None
    agent_loop.run_async = AsyncMock(return_value=RunResult(
        text=FAILED_REPLY, messages=_failed_turn_messages(), turns=2))

    tool_registry = MagicMock()
    tool_registry.list_schemas.return_value = [{"name": "bash"}]

    adapter = TelegramAdapter(
        config=PlatformConfig(platform=Platform.TELEGRAM, token=""),
        agent_loop=agent_loop,
        tool_registry=tool_registry,
        system_prompt="test",
        model_name="gemma-test",
        model_provider="llama_cpp",
        session_manager=sm,
    )

    engine = None
    auto_dir = tmp_path / "auto"
    auto_dir.mkdir(exist_ok=True)
    if teacher_text is not None:
        provider = _FakeTeacherProvider(teacher_text)
        creator = SkillCreator(provider, model="teacher-test",
                               auto_dir=auto_dir, telemetry=telemetry)
        engine = TeacherEscalation(
            teacher_model="teacher-test",
            telemetry=telemetry,
            provider=provider,
            skill_creator=creator,
        )
    adapter.escalation_engine = engine
    return adapter, session, auto_dir


def _last_lcm_row(tmp_path: Path):
    conn = sqlite3.connect(tmp_path / "lcm.db")
    try:
        return conn.execute(
            "SELECT role, provenance, is_trusted FROM lcm_messages"
            " WHERE session_id='telegram:99'"
            " ORDER BY turn_index DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()


# ---------------------------------------------------------------------------

async def test_e2e_failed_turn_escalates_tags_session_and_lcm(tmp_path):
    adapter, session, auto_dir = _build_adapter(tmp_path, GOOD_TEACHER)

    text = await adapter._run_agent_turn(
        session, REQUEST, session_id="telegram:99")

    # Corrective reply delivered, with the visible note — never silent.
    assert text.startswith("The deploy failed because")
    assert "(System note:" in text and "teacher" in text

    # Session tail: the teacher message, tagged.
    last = session.messages[-1]
    assert last.role == "assistant"
    assert last.provenance == "teacher_escalation"
    assert last.is_trusted is False

    # LCM row: provenance survives the round-trip (acceptance #2, last leg).
    assert _last_lcm_row(tmp_path) == ("assistant", "teacher_escalation", 0)

    # Skill on disk + golden trace row.
    assert (auto_dir / "diagnose-unreachable-deploy-target.md").exists()
    conn = sqlite3.connect(tmp_path / "telemetry.db")
    rows = conn.execute(
        "SELECT payload FROM signal_events WHERE signal_type='teacher_escalation'"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    assert json.loads(rows[0][0])["status"] == "escalated"


async def test_e2e_teacher_failure_keeps_local_reply_with_note(tmp_path):
    adapter, session, auto_dir = _build_adapter(tmp_path, STALLED_TEACHER)

    text = await adapter._run_agent_turn(
        session, REQUEST, session_id="telegram:99")

    assert text.startswith(FAILED_REPLY)
    assert "did not succeed" in text  # loud failure note
    assert list(auto_dir.iterdir()) == []  # no skill
    # No teacher message in session or LCM.
    assert all(m.provenance != "teacher_escalation" for m in session.messages)
    row = _last_lcm_row(tmp_path)
    assert row is not None and row[1] != "teacher_escalation"


async def test_no_engine_and_inert_engine_change_nothing(tmp_path):
    # No engine at all.
    adapter, session, _ = _build_adapter(tmp_path, None)
    text = await adapter._run_agent_turn(
        session, REQUEST, session_id="telegram:99")
    assert text == FAILED_REPLY

    # Inert engine (teacher_model unset — the shipped default).
    adapter.escalation_engine = TeacherEscalation.from_config({})
    text = await adapter._run_agent_turn(
        session, REQUEST, session_id="telegram:99")
    assert text == FAILED_REPLY
    conn = sqlite3.connect(tmp_path / "telemetry.db")
    n = conn.execute("SELECT COUNT(*) FROM signal_events").fetchone()[0]
    conn.close()
    assert n == 0


async def test_hook_failure_never_breaks_the_reply(tmp_path):
    adapter, session, _ = _build_adapter(tmp_path, None)
    broken = MagicMock()
    broken.maybe_escalate = AsyncMock(side_effect=RuntimeError("boom"))
    adapter.escalation_engine = broken
    text = await adapter._run_agent_turn(
        session, REQUEST, session_id="telegram:99")
    assert text == FAILED_REPLY  # local reply unaffected


async def test_cmd_escalations_reports_stats(tmp_path):
    adapter, session, _ = _build_adapter(tmp_path, GOOD_TEACHER)
    await adapter._run_agent_turn(session, REQUEST, session_id="telegram:99")

    update = MagicMock()
    update.effective_chat.id = 7
    adapter.send = AsyncMock()
    await adapter._cmd_escalations(update, MagicMock())
    sent = adapter.send.call_args[0][1]
    assert "Teacher escalation" in sent
    assert "Fired: 1" in sent
    assert "Skills written: 1" in sent
    assert "telegram:99=1" in sent

    # Engine absent → explicit unavailable message, no crash.
    adapter.escalation_engine = None
    adapter.send.reset_mock()
    await adapter._cmd_escalations(update, MagicMock())
    assert "not available" in adapter.send.call_args[0][1]


def test_build_trace_pairs_uses_with_results():
    msgs = _failed_turn_messages()
    trace = build_trace_from_messages(msgs[1:])
    assert trace == [{
        "tool_name": "bash",
        "arguments": {"command": "./deploy.sh"},
        "result": "Error: connection refused (deploy target unreachable)",
        "is_error": True,
    }]
