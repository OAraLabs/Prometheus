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
import json as _json
import subprocess
from pathlib import Path
from typing import AsyncIterator

import pytest

from prometheus.coding.sandbox import ProcessSandbox
from prometheus.coding.session import (
    _REPORT_SCAN_BYTES,
    CodingRunReport,
    CodingSession,
    CodingTask,
    parse_coding_report,
)
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
            # Envelope rounds only — run_loop's per-run 'tool_advertisement'
            # row (deferred-loading accounting) also carries this session id.
            " AND operation != 'tool_advertisement'"
        ).fetchone()[0]
        assert loop_rows == 2


# --------------------------------------------------------------------------- #
# parse_coding_report — recovering the report from a run's captured output
# (Beacon#130 / audit P9.10)
# --------------------------------------------------------------------------- #
_REAL_REPORT = {
    "task_id": "a12345678",
    "status": "success",
    "reason": "acceptance command exited 0",
    "rounds_used": 8,
    "episodes": 2,
    "wall_seconds": 91.3,
    "acceptance_exit": 0,
    "acceptance_output_tail": "1 passed in 0.4s",
    "branch": "coding/t11",
    "diff_stat": " src/a.py | 4 +-",
    "sandbox_root": "/tmp/sandbox/a12345678",
}


class TestParseCodingReport:
    def test_parses_a_clean_report(self):
        assert parse_coding_report(_json.dumps(_REAL_REPORT)) == _REAL_REPORT

    def test_parses_after_log_noise(self):
        """The real shape: rounds of output, then the report last."""
        out = "cloning…\nround 1: thinking\n" * 50 + _json.dumps(_REAL_REPORT)
        assert parse_coding_report(out)["status"] == "success"

    def test_report_larger_than_the_old_4000_tail(self):
        """THE BUG: a report the client could never recover from a 4000-byte tail.

        diff_stat is uncapped in the producer (one line per changed file), so a
        large refactor overflows. Measured against 124 real task logs the largest
        report was 2.1 KB — under the cap, which is why nobody had hit it.
        """
        big = dict(_REAL_REPORT, diff_stat="\n".join(f" src/m_{i}.py | {i} +-" for i in range(400)),
                   acceptance_output_tail="z" * 3000)
        blob = _json.dumps(big)
        assert len(blob) > 4000
        parsed = parse_coding_report("noise\n" * 400 + blob)
        assert parsed["diff_stat"] == big["diff_stat"]
        assert parsed["acceptance_output_tail"] == "z" * 3000

    def test_braces_and_quotes_inside_string_fields_do_not_fool_it(self):
        """raw_decode, not brace-matching: acceptance output and diff_stat both
        contain braces, quotes and newlines in real runs."""
        hairy = dict(_REAL_REPORT,
                     acceptance_output_tail='{"assert": "failed"}\n  File "x.py", line 3\n  "quoted"',
                     diff_stat=" src/a.py | 2 +-\n+def f(): return {'k': 'v'}")
        assert parse_coding_report(_json.dumps(hairy))["status"] == "success"

    # ── the refusal direction: things that must NOT be returned as the report ──

    def test_json_with_status_but_no_report_markers_is_not_the_report(self):
        """A tool result or a pytest --json blob has a status too. Returning one
        as "the report" would render as a finished run with invented fields —
        worse than null."""
        assert parse_coding_report('{"status": "ok", "unrelated": true}') is None

    def test_array_and_non_dict_json_are_skipped(self):
        assert parse_coding_report('[{"status": "passed"}]') is None

    def test_no_report_printed_yet_returns_none(self):
        """A run still in progress: null, not a fabricated object. This is the
        state the client previously could not distinguish from a truncated
        report — both rendered "No report"."""
        assert parse_coding_report("cloning…\nround 1: thinking\n") is None

    def test_empty_and_none_are_none(self):
        assert parse_coding_report("") is None
        assert parse_coding_report(None) is None

    def test_malformed_json_does_not_raise(self):
        """A truncated/partial object is the whole point of the bug — the parser
        must return None, never propagate."""
        assert parse_coding_report('{"status": "succ') is None
        assert parse_coding_report("{not json at all") is None

    def test_a_report_shaped_object_before_the_final_one_wins_the_last(self):
        """Scanning BACKWARDS returns the last report-shaped object, which is the
        one that matters (the run's verdict). An earlier decoy must not win."""
        decoy = _json.dumps(dict(_REAL_REPORT, status="failed_error", task_id="decoy"))
        final = _json.dumps(dict(_REAL_REPORT, status="success", task_id="final"))
        assert parse_coding_report(decoy + "\nlater output\n" + final)["task_id"] == "final"

    def test_finds_a_report_at_the_far_edge_of_the_scan_window(self):
        """Boundary of the bounded scan. A report just INSIDE the window is found
        even with megabytes of noise before it — the window is measured from the
        END, and the report is the last object, so depth of preceding noise is
        irrelevant. Measured: unbounded scanning of a 34 MB real output file took
        865 ms inside an async route; bounded, 6 ms.
        """
        noise = "x" * (_REPORT_SCAN_BYTES * 8)  # 2 MB, 8x the window
        assert parse_coding_report(noise + _json.dumps(_REAL_REPORT))["status"] == "success"

    def test_perf_on_a_very_large_output_is_bounded(self):
        """34 MB is not hypothetical — one exists on disk. The scan must stay
        sub-millisecond-scale regardless, because it runs in an async route."""
        import time

        huge = "log line\n" * 4_000_000  # ~36 MB
        t0 = time.monotonic()
        assert parse_coding_report(huge) is None
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0, f"scan should be bounded by the window, took {elapsed:.2f}s"
