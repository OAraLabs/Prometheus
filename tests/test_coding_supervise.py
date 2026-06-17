"""Adversarial hard-gate tests for the coding mid-run control channel (Loop Manager Sprint 2).

Side-effect assertions, not mock call-counts. The channel must be FAIL-SAFE (a broken control
file NEVER wedges a run), abandon a forgotten pause ON THE WALL from INSIDE the pause loop,
inject a trust-tagged correction into the next episode that does NOT poison the fact store, and
pause/resume deterministically. ScriptedModel drives the loop; real sandbox, real git, real
telemetry, real MemoryExtractor.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import MagicMock

import pytest

from prometheus.coding.control import (
    EMPTY,
    Injection,
    control_path,
    with_injection,
    with_paused,
    write_state,
)
from prometheus.coding.sandbox import ProcessSandbox
from prometheus.coding.session import CodingSession, CodingTask
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiStreamEvent,
    ModelProvider,
)
from prometheus.telemetry.tracker import ToolCallTelemetry

ACCEPT = "python3 -m pytest tests/ -q"


# --------------------------------------------------------------------------- #
# Harness — a clean repo + a scripted model that records what it was shown
# --------------------------------------------------------------------------- #

def _make_repo(tmp_path: Path, *, buggy: bool) -> Path:
    root = tmp_path / "target-repo"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    body = "    return a - b\n" if buggy else "    return a + b\n"
    (root / "src" / "calc.py").write_text("def add(a, b):\n" + body)
    (root / "tests" / "test_add.py").write_text(
        "import sys, pathlib\n"
        "sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))\n"
        "from calc import add\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"],
        cwd=root, check=True,
    )
    return root


def _text_turn(text: str) -> ConversationMessage:
    return ConversationMessage(role="assistant", content=[TextBlock(text=text)])


def _tool_turn(*blocks: ToolUseBlock) -> ConversationMessage:
    return ConversationMessage(role="assistant", content=list(blocks))


def _completing_turns() -> list[ConversationMessage]:
    """A 2-turn script that converts a CLEAN repo: run acceptance (green), declare done."""
    return [
        _tool_turn(ToolUseBlock(id="t1", name="code_run", input={"command": ACCEPT})),
        _text_turn("acceptance passes. done."),
    ]


class ScriptedModel(ModelProvider):
    """Plays a fixed sequence; records every request so a test can inspect what the model saw."""

    def __init__(self, turns: list[ConversationMessage]) -> None:
        self._turns = turns
        self.calls = 0
        self.seen_requests: list[ApiMessageRequest] = []

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        self.seen_requests.append(request)
        message = self._turns[self.calls] if self.calls < len(self._turns) else _text_turn("(exhausted)")
        self.calls += 1
        yield ApiMessageCompleteEvent(
            message=message,
            usage=UsageSnapshot(input_tokens=10, output_tokens=5),
            stop_reason="stop",
        )


def _session(repo, model, *, control_dir=None, telemetry=None, max_wall_seconds=1200.0, max_rounds=30):
    return CodingSession(
        provider=model,
        model="scripted",
        sandbox=ProcessSandbox(root=repo),
        task=CodingTask(task_id="t-sup", description="Make add() add.", acceptance_command=ACCEPT),
        telemetry=telemetry,
        control_dir=str(control_dir) if control_dir else None,
        max_wall_seconds=max_wall_seconds,
        max_rounds=max_rounds,
    )


def _has_control_row(tel: ToolCallTelemetry, operation: str) -> bool:
    return tel._conn.execute(
        "SELECT 1 FROM subsystem_runs WHERE subsystem='coding_control' AND operation=? LIMIT 1",
        (operation,),
    ).fetchone() is not None


async def _wait_for_control_row(tel: ToolCallTelemetry, operation: str, timeout: float = 8.0) -> bool:
    for _ in range(int(timeout / 0.02)):
        await asyncio.sleep(0.02)
        if _has_control_row(tel, operation):
            return True
    return False


def _content(msg: ConversationMessage) -> str:
    return str(msg.content)


# --------------------------------------------------------------------------- #
# 1. FAIL-SAFE — the gate. A broken control file NEVER derails a run.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw", [
    pytest.param(b"", id="empty-file"),
    pytest.param(b"\x00\xff\x01 not even text", id="garbage-bytes"),
    pytest.param(b'{"paused": tr', id="truncated-json"),
    pytest.param(b'{"paused": true, "injections": [{"id": "x", "te', id="truncated-mid-injection"),
])
def test_failsafe_corrupt_control_completes_normally(tmp_path, raw):
    repo = _make_repo(tmp_path, buggy=False)
    cdir = tmp_path / "control"
    cdir.mkdir()
    control_path(cdir).write_bytes(raw)  # a malformed control file is present from the start
    report = asyncio.run(_session(repo, ScriptedModel(_completing_turns()), control_dir=cdir).run())
    # Despite the corrupt control file, the run reads it as no-control and completes normally.
    assert report.status == "success", f"corrupt control {raw!r} must not wedge or derail the run"


def test_failsafe_control_file_vanishes_during_pause(tmp_path, monkeypatch):
    """File appears (paused), then VANISHES between seam re-reads — the run must continue."""
    monkeypatch.setattr("prometheus.coding.session._CONTROL_POLL_SECONDS", 0.02)
    repo = _make_repo(tmp_path, buggy=False)
    cdir = tmp_path / "control"
    cdir.mkdir()
    tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
    write_state(control_path(cdir), with_paused(EMPTY, True))  # start paused
    model = ScriptedModel(_completing_turns())

    async def drive():
        task = asyncio.create_task(_session(repo, model, control_dir=cdir, telemetry=tel).run())
        assert await _wait_for_control_row(tel, "pause"), "run should have paused"
        assert model.calls == 0  # paused before any episode
        control_path(cdir).unlink()  # the control file VANISHES mid-pause
        return await task

    report = asyncio.run(drive())
    assert report.status == "success"  # vanished file → read EMPTY → resume → complete


# --------------------------------------------------------------------------- #
# 2. PAUSED-PAST-CAP — abandonment fires from INSIDE the pause loop, on the wall.
# --------------------------------------------------------------------------- #

def test_paused_past_cap_abandons_from_inside_pause(tmp_path, monkeypatch):
    monkeypatch.setattr("prometheus.coding.session._CONTROL_POLL_SECONDS", 0.02)
    repo = _make_repo(tmp_path, buggy=True)
    cdir = tmp_path / "control"
    cdir.mkdir()
    tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
    write_state(control_path(cdir), with_paused(EMPTY, True))  # paused and NEVER resumed
    model = ScriptedModel(_completing_turns())

    report = asyncio.run(
        _session(repo, model, control_dir=cdir, telemetry=tel, max_wall_seconds=0.4).run()
    )
    # The wall is monotonic; pause never stops it. The run abandons on the wall, from the pause.
    assert report.status == "failed_abandoned"
    assert "wall" in report.reason.lower() and "paused" in report.reason.lower()
    # Proof it abandoned from INSIDE the pause loop, not after an episode: the model never ran.
    assert model.calls == 0
    assert _has_control_row(tel, "pause")
    # And the test reaching here at all proves it did not hang.


# --------------------------------------------------------------------------- #
# 3. INJECT — (a) reaches the next episode trust-tagged; (b) not minable as a fact.
# --------------------------------------------------------------------------- #

def test_inject_correction_reaches_next_episode_trust_tagged(tmp_path):
    repo = _make_repo(tmp_path, buggy=False)
    cdir = tmp_path / "control"
    cdir.mkdir()
    tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
    write_state(control_path(cdir), with_injection(EMPTY, Injection(id="i1", text="FOCUS_ON_THE_PARSER_FIRST")))
    model = ScriptedModel(_completing_turns())

    report = asyncio.run(_session(repo, model, control_dir=cdir, telemetry=tel).run())
    assert report.status == "success"

    # (a) the correction is in the FIRST episode's model context, provenance=supervisor, trusted,
    #     and NOT untrusted-bannered (it is actionable guidance, not inert DATA).
    ep1 = model.seen_requests[0].messages
    sup = [m for m in ep1 if getattr(m, "provenance", None) == "supervisor"]
    assert sup, "the supervisor correction must appear in the next episode's context"
    assert sup[0].is_trusted is True
    assert "FOCUS_ON_THE_PARSER_FIRST" in _content(sup[0])
    assert "UNTRUSTED INPUT" not in _content(sup[0])  # actionable, not DATA-bannered

    # telemetry recorded the inject as a distinct coding_control row
    assert _has_control_row(tel, "inject")


async def test_inject_supervisor_steer_is_not_mined_into_memory(tmp_path):
    """(b) provenance='supervisor' is excluded by the REAL MemoryExtractor — the steer does not
    poison the fact store (mirrors test_extractor_provenance with the supervisor tag)."""
    from prometheus.memory.extractor import MemoryExtractor
    from prometheus.memory.lcm_conversation_store import LCMConversationStore
    from prometheus.memory.lcm_types import MessagePart
    from prometheus.memory.store import MemoryStore

    conv = LCMConversationStore(db_path=tmp_path / "lcm.db")
    conv.add_message("coding:t-sup", MessagePart(
        role="user", content="My name is Will and I love PIZZA_USERFACT.",
        session_id="coding:t-sup", turn_index=0, provenance="user", is_trusted=True,
    ))
    # The human supervisor steer, exactly as the seam injects it: provenance='supervisor'.
    conv.add_message("coding:t-sup", MessagePart(
        role="user", content="SUPERVISOR_STEER refactor the parser first",
        session_id="coding:t-sup", turn_index=1, provenance="supervisor", is_trusted=True,
    ))

    store = MemoryStore(db_path=tmp_path / "memory.db")
    extractor = MemoryExtractor(store, MagicMock(), lcm_conversation_store=conv)
    captured: dict[str, str] = {}

    async def fake_call_model(prompt: str) -> str:
        captured["prompt"] = prompt
        facts: list[str] = []
        if "PIZZA_USERFACT" in prompt:
            facts.append('{"entity_type":"person","entity_name":"Will","fact":"loves PIZZA_USERFACT","confidence":0.9}')
        if "SUPERVISOR_STEER" in prompt:
            facts.append('{"entity_type":"concept","entity_name":"steer","fact":"SUPERVISOR_STEER","confidence":0.9}')
        return "\n".join(facts)

    extractor._call_model = fake_call_model
    count, facts = await extractor.run_once()

    assert "PIZZA_USERFACT" in captured["prompt"]        # the genuine user fact was mined
    assert "SUPERVISOR_STEER" not in captured["prompt"]  # the steer NEVER reached the extractor
    blob = " ".join(str(m) for m in store.get_all_memories())
    assert "SUPERVISOR_STEER" not in blob                # the steer does not poison the fact store


# --------------------------------------------------------------------------- #
# 4. PAUSE/RESUME — pause holds before the next episode; resume completes.
# --------------------------------------------------------------------------- #

def test_pause_holds_then_resume_completes(tmp_path, monkeypatch):
    monkeypatch.setattr("prometheus.coding.session._CONTROL_POLL_SECONDS", 0.02)
    repo = _make_repo(tmp_path, buggy=False)
    cdir = tmp_path / "control"
    cdir.mkdir()
    tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
    write_state(control_path(cdir), with_paused(EMPTY, True))
    model = ScriptedModel(_completing_turns())

    async def drive():
        task = asyncio.create_task(_session(repo, model, control_dir=cdir, telemetry=tel).run())
        assert await _wait_for_control_row(tel, "pause"), "run should have paused"
        assert model.calls == 0, "pause must HOLD before the next episode runs"
        await asyncio.sleep(0.1)
        assert model.calls == 0, "still held while paused"
        write_state(control_path(cdir), with_paused(EMPTY, False))  # RESUME
        return await task

    report = asyncio.run(drive())
    assert report.status == "success"   # resumed and completed correctly
    assert model.calls > 0              # the model ran only AFTER resume
    assert _has_control_row(tel, "resume")
