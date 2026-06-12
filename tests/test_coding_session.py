"""CodingSession end-to-end with a scripted model — real sandbox, real git,
real tool dispatch through run_loop; only the model's turns are scripted.

The load-bearing scenario is bakeoff F3 made structural: the model runs a
PASSING test and claims done while the task's acceptance command still
fails — the session's own ground-truth acceptance run catches it, injects
the real failure, and the (scripted) fix then lands. silent_wrong_answer
cannot survive this session design, by construction.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import AsyncIterator

import pytest

from prometheus.coding.sandbox import ProcessSandbox
from prometheus.coding.session import CodingRunReport, CodingSession, CodingTask
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiStreamEvent,
    ModelProvider,
)

ACCEPT = "python3 -m pytest tests/ -q"


def _make_repo(tmp_path: Path, *, buggy: bool) -> Path:
    root = tmp_path / "target-repo"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    body = "    return a - b\n" if buggy else "    return a + b\n"
    (root / "src" / "calc.py").write_text("def add(a, b):\n" + body)
    (root / "tests" / "test_ok.py").write_text(
        "def test_always_green():\n    assert True\n"
    )
    (root / "tests" / "test_add.py").write_text(
        "import sys, pathlib\n"
        "sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))\n"
        "from calc import add\n"
        "\n"
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


class ScriptedModel(ModelProvider):
    """Plays a fixed sequence of assistant turns, one per model call."""

    def __init__(self, turns: list[ConversationMessage]) -> None:
        self._turns = turns
        self.calls = 0

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        if self.calls >= len(self._turns):
            # Script exhausted — end the turn with empty text so the
            # session's policy (not the provider) decides what happens.
            message = _text_turn("(script exhausted)")
        else:
            message = self._turns[self.calls]
        self.calls += 1
        yield ApiMessageCompleteEvent(
            message=message,
            usage=UsageSnapshot(input_tokens=10, output_tokens=5),
            stop_reason="stop",
        )


def _run(session: CodingSession) -> CodingRunReport:
    return asyncio.run(session.run())


def _session(repo: Path, turns: list[ConversationMessage], **kw) -> tuple[CodingSession, ProcessSandbox]:
    sandbox = ProcessSandbox(root=repo)
    task = CodingTask(
        task_id="t-test",
        description="Make the test suite pass: add() must add.",
        acceptance_command=ACCEPT,
    )
    session = CodingSession(
        provider=ScriptedModel(turns),
        model="scripted",
        sandbox=sandbox,
        task=task,
        **kw,
    )
    return session, sandbox


# --------------------------------------------------------------------------- #
# The F3 scenario — confident-but-wrong caught by ground truth, then fixed
# --------------------------------------------------------------------------- #


class TestGroundTruthCatchesFalseConfidence:

    def test_full_arc(self, tmp_path: Path):
        repo = _make_repo(tmp_path, buggy=True)
        turns = [
            # Episode 1: run only the green test, then claim done.
            _tool_turn(ToolUseBlock(
                id="t1", name="code_run",
                input={"command": "python3 -m pytest tests/test_ok.py -q"},
            )),
            _text_turn("test_ok passes — the task is complete."),
            # Episode 2 (after ground-truth rejection): fix the bug, run
            # the real acceptance command, then finish.
            _tool_turn(ToolUseBlock(
                id="t2", name="code_str_replace",
                input={"path": "src/calc.py",
                       "old_str": "    return a - b",
                       "new_str": "    return a + b"},
            )),
            _tool_turn(ToolUseBlock(
                id="t3", name="code_run", input={"command": ACCEPT},
            )),
            _text_turn("Acceptance passes now. Done."),
        ]
        session, sandbox = _session(repo, turns)
        report = _run(session)

        assert report.status == "success"
        assert report.acceptance_exit == 0
        assert report.episodes == 2
        # The fix actually landed on disk…
        assert "a + b" in (sandbox.root / "src" / "calc.py").read_text()
        # …on the artifact branch, committed.
        head = subprocess.run(
            ["git", "log", "--oneline", "-1"], cwd=sandbox.root,
            capture_output=True, text=True,
        ).stdout
        assert "coding task t-test: success" in head
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=sandbox.root,
            capture_output=True, text=True,
        ).stdout.strip()
        assert branch == "coding/t-test"

    def test_ground_truth_rejection_was_injected(self, tmp_path: Path):
        repo = _make_repo(tmp_path, buggy=True)
        turns = [
            _tool_turn(ToolUseBlock(
                id="t1", name="code_run",
                input={"command": "python3 -m pytest tests/test_ok.py -q"},
            )),
            _text_turn("done."),
            # After the injection the script just gives up (text only) —
            # we only care that the injected message reached the model.
        ]
        session, _ = _session(repo, turns, max_rounds=4)
        provider: ScriptedModel = session._provider  # type: ignore[assignment]
        report = _run(session)

        assert report.status == "failed_abandoned"
        # The model was called again AFTER its false claim — i.e. the
        # ground-truth rejection re-engaged it rather than accepting.
        assert provider.calls >= 3


# --------------------------------------------------------------------------- #
# Done-is-a-verdict layer 1 — no evidence, no exit
# --------------------------------------------------------------------------- #


class TestNoEvidenceRejection:

    def test_claim_without_any_test_run_is_rejected_then_recovers(self, tmp_path: Path):
        repo = _make_repo(tmp_path, buggy=False)  # acceptance passes from base
        turns = [
            _text_turn("Looked at the code; everything is fine. Done."),
            # After the no-evidence injection:
            _tool_turn(ToolUseBlock(
                id="t1", name="code_run", input={"command": ACCEPT},
            )),
            _text_turn("Verified: acceptance exits 0."),
        ]
        session, _ = _session(repo, turns)
        report = _run(session)

        assert report.status == "success"
        assert report.episodes == 2
        assert report.rounds_used == 3


# --------------------------------------------------------------------------- #
# Caps — honest abandonment with the artifact committed
# --------------------------------------------------------------------------- #


class TestTurnLimitExhaustion:

    def test_model_that_never_stops_is_abandoned_not_crashed(self, tmp_path: Path):
        # A model that tool-calls forever exhausts run_loop's per-episode
        # turn allowance, which RAISES. The session must convert that to
        # honest abandonment, not propagate the RuntimeError.
        repo = _make_repo(tmp_path, buggy=True)
        # Every turn views a file (a tool call) → the model never "stops",
        # so run_loop hits max_turns and raises.
        forever = [
            _tool_turn(ToolUseBlock(id=f"v{i}", name="code_view",
                                    input={"path": "src/calc.py"}))
            for i in range(20)
        ]
        session, _ = _session(repo, forever, max_rounds=3)
        report = _run(session)
        assert report.status == "failed_abandoned"
        assert report.rounds_used <= 3 + 1  # bounded by the cap, no crash


class TestCaps:

    def test_round_cap_abandons_honestly(self, tmp_path: Path):
        repo = _make_repo(tmp_path, buggy=True)
        turns = [_text_turn("hmm.")] * 5  # never runs anything
        session, sandbox = _session(repo, turns, max_rounds=2)
        report = _run(session)

        assert report.status == "failed_abandoned"
        assert "round cap" in report.reason
        assert report.acceptance_exit not in (0, None)
        assert report.rounds_used == 2
        head = subprocess.run(
            ["git", "log", "--oneline", "-1"], cwd=sandbox.root,
            capture_output=True, text=True,
        ).stdout
        assert "failed_abandoned" in head

    def test_green_at_cap_reports_success(self, tmp_path: Path):
        # Work is actually done but the model never produced evidence —
        # the cap fires, ground truth says green, the report says success
        # (the evidence is the run itself, honestly labeled).
        repo = _make_repo(tmp_path, buggy=False)
        turns = [_text_turn("thinking...")] * 5
        session, _ = _session(repo, turns, max_rounds=2)
        report = _run(session)

        assert report.status == "success"
        assert "green at cap" in report.reason


# --------------------------------------------------------------------------- #
# Telemetry side effect — the run writes its terminal row
# --------------------------------------------------------------------------- #


class TestTerminalTelemetry:

    def test_terminal_row_written(self, tmp_path: Path):
        from prometheus.telemetry.tracker import ToolCallTelemetry

        repo = _make_repo(tmp_path, buggy=False)
        tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
        turns = [
            _tool_turn(ToolUseBlock(id="t1", name="code_run", input={"command": ACCEPT})),
            _text_turn("done"),
        ]
        session, _ = _session(repo, turns, telemetry=tel)
        report = _run(session)
        assert report.status == "success"

        row = tel._conn.execute(
            "SELECT outcome, session_id, model FROM subsystem_runs"
            " WHERE subsystem='coding_mode' AND operation='run'"
        ).fetchone()
        assert row == ("success", "coding:t-test", "scripted")
        # And the per-round envelope rows carry the coding session id too.
        loop_rows = tel._conn.execute(
            "SELECT COUNT(*) FROM subsystem_runs"
            " WHERE subsystem='agent_loop' AND session_id='coding:t-test'"
        ).fetchone()[0]
        assert loop_rows == 2
