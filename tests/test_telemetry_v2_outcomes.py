"""Telemetry v2 turn outcomes (WP-X.54 T-4).

T-3 writes a ``turns`` row per turn and leaves ``outcome`` NULL. These tests
pin who fills it, with what, and when, by the rows actually written to a real
``telemetry.db`` through the real queued writer:

* the daemon, at turn end: ``forced_stop`` / ``error_terminal`` from the
  loop's own stop reason (Will's split, 2026-10-03);
* the user, at the next message's ingress: ``accepted_user`` /
  ``user_corrected`` on clear signals only, NULL otherwise;
* the heartbeat sweep: ``abandoned`` when no message followed within the
  window, in batches, never over a set outcome;
* coding-mode acceptance: ``accepted_verified`` / ``rejected_verified`` on the
  one episode the acceptance command judged.

The rules are Will's rulings on the T-4 proposal (2026-10-03). Nothing on the
turn path waits on any of it: every write goes through the queue.
"""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

from prometheus.engine.agent_loop import LoopContext, _StopReason, run_loop
from prometheus.engine.messages import ConversationMessage
from prometheus.learning import pair_capture
from prometheus.telemetry import outcomes
from prometheus.telemetry.outcomes import (
    WINDOW_SECONDS,
    classify_next_message,
    coding_acceptance,
    note_user_message,
    sweep_abandoned,
    turn_end_outcome,
)
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.telemetry.writer import TelemetryV2Writer

# Reused from the T-3 tests: the scripted provider and the tool doubles.
from tests.test_telemetry_v2_loop_writes import _call, _prose, _registry, _Script

SESSION = "desktop:outcomes"
T0 = 1_790_000_000.0  # a fixed clock origin; every time below is relative to it


@pytest.fixture(autouse=True)
def _clean_state():
    pair_capture.configure({"capture_enabled": False})
    outcomes.reset_memory()
    yield
    outcomes.reset_memory()
    pair_capture.configure({"capture_enabled": False})


@pytest.fixture
def tel(tmp_path):
    t = ToolCallTelemetry(tmp_path / "telemetry.db")
    yield t
    t.close()


def _rows(db: Path, sql: str, *args) -> list[sqlite3.Row]:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


def _outcome(tel: ToolCallTelemetry, turn_id: str) -> tuple:
    assert tel.v2_writer().flush()
    [row] = _rows(tel.db_path, "SELECT outcome, outcome_source, outcome_at FROM turns "
                               "WHERE turn_id = ?", turn_id)
    return tuple(row)


def _turn(tel: ToolCallTelemetry, turn_id: str, *, started: float, ended: float | None,
          session: str = SESSION, surface: str = "beacon", **extra) -> None:
    """Seed a turns row the way T-3 writes it: a start upsert, then an end upsert."""
    w = tel.v2_writer()
    w.upsert("turns", {"turn_id": turn_id, "session_id": session, "surface": surface,
                       "mode": "agent", "started_at": started}, key="turn_id")
    if ended is not None:
        w.upsert("turns", {"turn_id": turn_id, "session_id": session, "ended_at": ended,
                           "terminal_kind": "prose", **extra}, key="turn_id")
    assert w.flush()


# --------------------------------------------------------------------------- #
# The classifier: clear signals only (rulings 1, 1a-1c)
# --------------------------------------------------------------------------- #

PREV = "rename the config loader function to load_settings across the package"


class TestClassifier:

    @pytest.mark.parametrize("text", [
        "no", "No.", "no, the other file", "nope", "Nope, wrong one",
        "that's wrong", "That’s wrong, the total is 12", "that is wrong",
        "wrong", "incorrect", "not what I asked", "that's not what I asked for",
        "try again", "please try again", "redo", "redo it with tabs",
        "still broken", "I meant the staging box", "you forgot the tests",
        "you didn't save it", "it still doesn't work", "didn't work",
        "same error as before", "you misunderstood me",
    ])
    def test_a_clear_correction_is_user_corrected(self, text):
        assert classify_next_message(text, PREV) == "user_corrected"

    @pytest.mark.parametrize("text", ["actually", "wait", "still", "again",
                                      "actually, use the blue one", "wait what",
                                      "still there?", "again please"])
    def test_an_ambiguous_opener_alone_is_null(self, text):
        """Ruling 1a: actually / wait / still / again on their own say nothing clear."""
        assert classify_next_message(text, PREV) is None

    @pytest.mark.parametrize("text", ["actually that's wrong", "wait, it didn't work",
                                      "still not working", "again the same error"])
    def test_an_ambiguous_opener_with_a_correction_is_user_corrected(self, text):
        assert classify_next_message(text, PREV) == "user_corrected"

    @pytest.mark.parametrize("text", ["no worries", "No problem!", "no thanks",
                                      "nope, that's it", "Nope that's it"])
    def test_the_exclusions_are_not_corrections(self, text):
        """Ruling 1b: these open with "no" and correct nothing."""
        assert classify_next_message(text, PREV) != "user_corrected"

    def test_a_near_repeat_of_the_request_is_user_corrected(self):
        again = "rename the config loader function to load_settings across the whole package"
        assert classify_next_message(again, PREV) == "user_corrected"

    def test_a_one_word_swap_is_not_a_repeat(self):
        """A new request that differs by one word (the parity model_switch
        scenario) overlaps 0.71, but it drops a word of the previous request, so
        it is not "essentially the same request": NULL, never user_corrected
        (Will, 2026-10-03)."""
        prev = "Reply with exactly the word: one"
        assert classify_next_message("Reply with exactly the word: two", prev) is None

    def test_a_repeat_under_four_qualifying_words_is_null(self):
        """Ruling 1c: below 4 words of 3+ letters the overlap measures nothing."""
        assert classify_next_message("fix the bug", "fix the bug") is None

    @pytest.mark.parametrize("text", ["thanks", "Thank you so much!", "perfect", "great, thanks",
                                      "ok", "looks good", "\U0001f44d", "that works, cheers"])
    def test_an_ack_is_accepted_user(self, text):
        assert classify_next_message(text, PREV) == "accepted_user"

    def test_an_ack_needs_no_previous_request(self):
        assert classify_next_message("thanks!", None) == "accepted_user"

    def test_a_different_new_request_is_accepted_user(self):
        new = "what's the weather forecast in Lisbon this weekend"
        assert classify_next_message(new, PREV) == "accepted_user"

    def test_a_new_request_with_no_previous_to_compare_is_null(self):
        """Without the previous request (a restart), different-or-repeat is a guess."""
        assert classify_next_message("what's the weather forecast in Lisbon", None) is None

    def test_a_middling_overlap_is_null(self):
        middling = "rename the config writer function too and update every caller"
        assert classify_next_message(middling, PREV) is None

    def test_a_short_unclear_message_is_null(self):
        assert classify_next_message("hmm", PREV) is None
        assert classify_next_message("", PREV) is None


# --------------------------------------------------------------------------- #
# Turn end: forced_stop / error_terminal from the stop reason (ruling 2)
# --------------------------------------------------------------------------- #


class TestTurnEnd:

    def test_the_split_covers_every_stop_reason(self):
        """A closed term: a new stop reason must be placed on one side by hand."""
        reasons = {v for k, v in vars(_StopReason).items() if not k.startswith("_")}
        assert reasons == outcomes.FORCED_STOP_REASONS | outcomes.ERROR_TERMINAL_REASONS
        assert not outcomes.FORCED_STOP_REASONS & outcomes.ERROR_TERMINAL_REASONS

    @pytest.mark.parametrize("reason,kind,expected", [
        ("circuit_breaker_trip", "tool_call", "forced_stop"),
        ("cancelled", "tool_call", "forced_stop"),
        ("max_turns_exhausted", "tool_call", "forced_stop"),
        ("empty_response", "empty", "error_terminal"),
        ("context_preflight_refusal", None, "error_terminal"),
        ("provider_error", "error", "error_terminal"),
        (None, "error", "error_terminal"),
        (None, "prose", None),
        (None, None, None),
    ])
    def test_the_mapping(self, reason, kind, expected):
        assert turn_end_outcome(reason, kind) == expected

    def _run(self, tmp_path, turns, raises=None, **ctx_kw):
        db = tmp_path / "telemetry.db"
        t = ToolCallTelemetry(db)
        ctx = LoopContext(provider=_Script(turns), model="stub-model", system_prompt="",
                          max_tokens=128, tool_registry=_registry(), telemetry=t, **ctx_kw)

        async def drain():
            async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")],
                                    session_id=SESSION, surface="beacon"):
                pass

        if raises:
            with pytest.raises(raises):
                asyncio.run(drain())
        else:
            asyncio.run(drain())
        t.close()
        [row] = _rows(db, "SELECT * FROM turns")
        return row

    def test_a_circuit_breaker_run_is_forced_stop_at_its_end(self, tmp_path):
        row = self._run(tmp_path, [_call("broken_tool", f"b{i}", count=i) for i in range(1, 12)])
        assert (row["outcome"], row["outcome_source"]) == ("forced_stop", "daemon")
        assert row["outcome_at"] == row["ended_at"]

    def test_a_provider_error_is_error_terminal(self, tmp_path):
        row = self._run(tmp_path, [ConnectionError("gone")], raises=ConnectionError)
        assert (row["outcome"], row["outcome_source"]) == ("error_terminal", "daemon")
        assert row["outcome_at"] == row["ended_at"]

    def test_an_empty_give_up_is_error_terminal(self, tmp_path):
        row = self._run(tmp_path, [_prose(""), _prose("")])
        assert row["outcome"] == "error_terminal"

    def test_a_prose_answer_leaves_the_outcome_to_the_user(self, tmp_path):
        row = self._run(tmp_path, [_prose("four")])
        assert (row["outcome"], row["outcome_source"], row["outcome_at"]) == (None, None, None)

    def test_a_failing_outcome_write_never_breaks_the_turn(self, tmp_path, monkeypatch, caplog):
        """Wrapped like record(): a WARNING, and the turn ends normally."""
        def boom(*a, **k):
            raise RuntimeError("outcome writer exploded")

        monkeypatch.setattr(outcomes, "stamp_turn_end", boom)
        row = self._run(tmp_path, [_call("broken_tool", f"b{i}", count=i) for i in range(1, 12)])
        assert row["forced_stop_reason"] == "circuit_breaker_trip"
        assert row["outcome"] is None
        assert "outcome" in caplog.text


# --------------------------------------------------------------------------- #
# Ingress: the next user message labels the turn before it (rulings 1, d)
# --------------------------------------------------------------------------- #


class TestIngress:

    def test_an_ack_within_the_window_is_accepted_user(self, tel):
        _turn(tel, "t1", started=T0, ended=T0 + 5)
        note_user_message(SESSION, "thanks!", telemetry=tel, at=T0 + 60)
        assert _outcome(tel, "t1") == ("accepted_user", "user_signal", T0 + 60)

    def test_a_correction_is_user_corrected(self, tel):
        _turn(tel, "t1", started=T0, ended=T0 + 5)
        note_user_message(SESSION, "no, that's wrong", telemetry=tel, at=T0 + 60)
        assert _outcome(tel, "t1") == ("user_corrected", "user_signal", T0 + 60)

    def test_the_previous_request_comes_from_the_previous_message(self, tel):
        note_user_message(SESSION, PREV, telemetry=tel, at=T0 - 1)
        _turn(tel, "t1", started=T0, ended=T0 + 5)
        note_user_message(SESSION, PREV + " now", telemetry=tel, at=T0 + 60)
        assert _outcome(tel, "t1")[0] == "user_corrected"

    def test_an_unclear_message_writes_nothing(self, tel):
        _turn(tel, "t1", started=T0, ended=T0 + 5)
        note_user_message(SESSION, "actually", telemetry=tel, at=T0 + 60)
        assert _outcome(tel, "t1") == (None, None, None)

    def test_a_message_after_the_window_leaves_it_to_the_sweep(self, tel):
        _turn(tel, "t1", started=T0, ended=T0 + 5)
        note_user_message(SESSION, "thanks", telemetry=tel, at=T0 + 5 + WINDOW_SECONDS + 1)
        assert _outcome(tel, "t1") == (None, None, None)

    def test_only_the_first_message_after_a_turn_counts(self, tel):
        """An unclear first reply is not overruled by a second one."""
        _turn(tel, "t1", started=T0, ended=T0 + 5)
        note_user_message(SESSION, "wait", telemetry=tel, at=T0 + 30)
        note_user_message(SESSION, "thanks", telemetry=tel, at=T0 + 40)
        assert _outcome(tel, "t1") == (None, None, None)

    def test_a_message_sent_mid_turn_never_labels_that_turn(self, tel):
        """Ruling: it counts only for the turn before it. Turn A is running when
        M2 arrives (Beacon joins it to A, then runs B on it). M3, after B, labels
        B. A keeps NULL, and the sweep leaves it NULL too: a message followed it."""
        _turn(tel, "z", started=T0, ended=T0 + 5)
        note_user_message(SESSION, "what's the weather forecast in Lisbon this weekend",
                          telemetry=tel, at=T0 + 10)            # M1: starts A
        _turn(tel, "a", started=T0 + 11, ended=None)            # A running
        note_user_message(SESSION, "thanks", telemetry=tel, at=T0 + 20)   # M2, mid-turn
        _turn(tel, "a", started=T0 + 11, ended=T0 + 30)         # A ends
        _turn(tel, "b", started=T0 + 31, ended=T0 + 40)         # B runs on M2
        note_user_message(SESSION, "perfect", telemetry=tel, at=T0 + 50)  # M3
        assert _outcome(tel, "a") == (None, None, None)
        assert _outcome(tel, "b")[0] == "accepted_user"
        sweep_abandoned(tel, now=T0 + 10 * WINDOW_SECONDS)
        assert _outcome(tel, "a") == (None, None, None)

    def test_it_never_overwrites_a_set_outcome(self, tel):
        _turn(tel, "t1", started=T0, ended=T0 + 5, forced_stop_reason="circuit_breaker_trip")
        tel.v2_writer().upsert("turns", {"turn_id": "t1", "session_id": SESSION,
                                         "outcome": "forced_stop", "outcome_source": "daemon",
                                         "outcome_at": T0 + 5}, key="turn_id")
        note_user_message(SESSION, "try again", telemetry=tel, at=T0 + 60)
        assert _outcome(tel, "t1") == ("forced_stop", "daemon", T0 + 5)

    @pytest.mark.parametrize("session", ["system", ""])
    def test_system_and_ephemeral_sessions_are_never_labelled(self, tel, session):
        _turn(tel, "t1", started=T0, ended=T0 + 5, session=session)
        note_user_message(session, "thanks", telemetry=tel, at=T0 + 60)
        assert _outcome(tel, "t1") == (None, None, None)

    def test_with_telemetry_off_it_does_nothing_and_never_raises(self, monkeypatch):
        from prometheus.telemetry import tracker

        monkeypatch.setattr(tracker, "_telemetry_singleton", None)
        note_user_message(SESSION, "thanks")  # no handle: a no-op

    def test_it_uses_the_daemon_handle_by_default(self, tel, monkeypatch):
        from prometheus.telemetry import tracker

        monkeypatch.setattr(tracker, "_telemetry_singleton", tel)
        _turn(tel, "t1", started=T0, ended=T0 + 5)
        note_user_message(SESSION, "thanks", at=T0 + 60)
        assert _outcome(tel, "t1")[0] == "accepted_user"

    def test_ingress_never_waits_on_the_database(self, tel):
        """The lock is held by another connection; the hook still returns at once."""
        _turn(tel, "t1", started=T0, ended=T0 + 5)
        con = sqlite3.connect(tel.db_path, timeout=0)
        con.execute("BEGIN IMMEDIATE")
        try:
            t = time.monotonic()
            note_user_message(SESSION, "thanks", telemetry=tel, at=T0 + 60)
            assert time.monotonic() - t < 0.5
        finally:
            con.rollback()
            con.close()
        assert _outcome(tel, "t1")[0] == "accepted_user"


# --------------------------------------------------------------------------- #
# The heartbeat sweep: abandoned (rulings 1d, 1e)
# --------------------------------------------------------------------------- #


class TestSweep:

    def test_no_message_within_the_window_is_abandoned(self, tel):
        _turn(tel, "t1", started=T0, ended=T0 + 5)
        sweep_abandoned(tel, now=T0 + 5 + WINDOW_SECONDS + 1)
        assert _outcome(tel, "t1") == ("abandoned", "daemon", T0 + 5 + WINDOW_SECONDS)

    def test_not_before_the_window_closes(self, tel):
        _turn(tel, "t1", started=T0, ended=T0 + 5)
        sweep_abandoned(tel, now=T0 + 5 + WINDOW_SECONDS - 1)
        assert _outcome(tel, "t1") == (None, None, None)

    def test_a_later_turn_within_the_window_means_a_message_came(self, tel):
        _turn(tel, "t1", started=T0, ended=T0 + 5)
        _turn(tel, "t2", started=T0 + 100, ended=T0 + 110)
        sweep_abandoned(tel, now=T0 + 10 * WINDOW_SECONDS)
        assert _outcome(tel, "t1") == (None, None, None)
        assert _outcome(tel, "t2")[0] == "abandoned"

    def test_a_later_turn_after_the_window_does_not_save_it(self, tel):
        _turn(tel, "t1", started=T0, ended=T0 + 5)
        _turn(tel, "t2", started=T0 + 5 + WINDOW_SECONDS + 60, ended=T0 + 5 + WINDOW_SECONDS + 70)
        sweep_abandoned(tel, now=T0 + 10 * WINDOW_SECONDS)
        assert _outcome(tel, "t1")[0] == "abandoned"

    @pytest.mark.parametrize("extra", [
        {"forced_stop_reason": "circuit_breaker_trip"},
        {"terminal_kind": "error"},
    ])
    def test_a_daemon_stopped_turn_is_never_abandoned(self, tel, extra):
        """Even one whose turn-end outcome never landed (a turn from before T-4)."""
        _turn(tel, "t1", started=T0, ended=T0 + 5, **extra)
        sweep_abandoned(tel, now=T0 + 10 * WINDOW_SECONDS)
        assert _outcome(tel, "t1") == (None, None, None)

    def test_it_never_overwrites_a_set_outcome(self, tel):
        _turn(tel, "t1", started=T0, ended=T0 + 5, outcome="error_terminal",
              outcome_source="daemon", outcome_at=T0 + 5)
        sweep_abandoned(tel, now=T0 + 10 * WINDOW_SECONDS)
        assert _outcome(tel, "t1") == ("error_terminal", "daemon", T0 + 5)

    @pytest.mark.parametrize("kw", [
        {"session": "system"}, {"session": ""}, {"surface": "cli"},
        {"surface": "coding_mode", "coding_run_id": "run-1"},
    ])
    def test_coding_system_ephemeral_and_unhooked_turns_are_excluded(self, tel, kw):
        _turn(tel, "t1", started=T0, ended=T0 + 5, **kw)
        sweep_abandoned(tel, now=T0 + 10 * WINDOW_SECONDS)
        assert _outcome(tel, "t1") == (None, None, None)

    def test_it_runs_in_batches(self, tel):
        for i in range(5):
            _turn(tel, f"t{i}", started=T0 + i, ended=T0 + i + 1, session=f"desktop:s{i}")
        sweep_abandoned(tel, now=T0 + 10 * WINDOW_SECONDS, batch=2)
        assert tel.v2_writer().flush()
        n = _rows(tel.db_path, "SELECT COUNT(*) FROM turns WHERE outcome = 'abandoned'")[0][0]
        assert n == 2
        sweep_abandoned(tel, now=T0 + 10 * WINDOW_SECONDS, batch=2)
        sweep_abandoned(tel, now=T0 + 10 * WINDOW_SECONDS, batch=2)
        assert tel.v2_writer().flush()
        n = _rows(tel.db_path, "SELECT COUNT(*) FROM turns WHERE outcome = 'abandoned'")[0][0]
        assert n == 5

    def test_a_refused_sweep_is_a_silent_failure_never_an_exception(self, tel):
        _turn(tel, "t1", started=T0, ended=T0 + 5)
        con = sqlite3.connect(tel.db_path)
        con.execute("DROP TABLE turns")
        con.commit()
        con.close()
        sweep_abandoned(tel, now=T0 + 10 * WINDOW_SECONDS)  # must not raise
        assert tel.v2_writer().flush()
        rows = _rows(tel.db_path, "SELECT subsystem, operation FROM silent_failures "
                                  "WHERE subsystem = 'telemetry_writer'")
        assert ("telemetry_writer", "outcome_sweep") in [tuple(r) for r in rows]

    def test_with_no_telemetry_it_does_nothing(self):
        sweep_abandoned(None)


class TestHeartbeatSweep:

    def test_the_heartbeat_runs_the_sweep(self, tel):
        from prometheus.gateway.heartbeat import Heartbeat

        _turn(tel, "t1", started=time.time() - 3 * WINDOW_SECONDS,
              ended=time.time() - 2 * WINDOW_SECONDS)
        hb = Heartbeat(telemetry=tel)
        asyncio.run(hb._sweep_outcomes())
        assert _outcome(tel, "t1")[0] == "abandoned"

    def test_the_sweep_is_throttled(self, tel, monkeypatch):
        from prometheus.gateway import heartbeat as hb_mod

        calls = []
        monkeypatch.setattr(hb_mod, "sweep_abandoned", lambda t, **k: calls.append(t))
        hb = hb_mod.Heartbeat(telemetry=tel)
        asyncio.run(hb._sweep_outcomes())
        asyncio.run(hb._sweep_outcomes())
        assert calls == [tel], "one sweep per OUTCOME_SWEEP_INTERVAL, not one per tick"

    def test_run_forever_calls_it(self, tel, monkeypatch):
        from prometheus.gateway import heartbeat as hb_mod

        hb = hb_mod.Heartbeat(telemetry=tel, interval=0)
        seen = []

        async def sweep():
            seen.append(True)
            hb.stop()

        monkeypatch.setattr(hb, "_sweep_outcomes", sweep)
        asyncio.run(asyncio.wait_for(hb.run_forever(), 5))
        assert seen

    def test_the_daemon_hands_the_heartbeat_its_tracker(self):
        src = (Path(__file__).resolve().parents[1] / "src/prometheus/daemon.py").read_text()
        start = src.index("heartbeat = Heartbeat(")
        assert "telemetry=telemetry" in src[start:src.index(")\n", start)]


# --------------------------------------------------------------------------- #
# Coding acceptance: the episode it judged (rulings 3, 4)
# --------------------------------------------------------------------------- #


class TestCodingAcceptance:

    def _episodes(self, tel, n, run="run-1"):
        for i in range(n):
            _turn(tel, f"e{i}", started=T0 + 10 * i, ended=T0 + 10 * i + 5, session="coding:t",
                  surface="coding_mode", coding_run_id=run)

    def test_a_pass_stamps_only_the_latest_episode(self, tel):
        self._episodes(tel, 3)
        coding_acceptance(tel, "run-1", 0, at=T0 + 100)
        assert _outcome(tel, "e2") == ("accepted_verified", "acceptance_command", T0 + 100)
        assert _outcome(tel, "e1") == (None, None, None)
        assert _outcome(tel, "e0") == (None, None, None)

    @pytest.mark.parametrize("exit_code", [1, None])
    def test_a_failure_or_timeout_is_rejected_verified(self, tel, exit_code):
        self._episodes(tel, 1)
        coding_acceptance(tel, "run-1", exit_code, at=T0 + 100)
        assert _outcome(tel, "e0") == ("rejected_verified", "acceptance_command", T0 + 100)

    def test_it_never_overwrites_a_forced_stop(self, tel):
        self._episodes(tel, 1)
        tel.v2_writer().upsert("turns", {"turn_id": "e0", "session_id": "coding:t",
                                         "outcome": "forced_stop", "outcome_source": "daemon",
                                         "outcome_at": T0 + 5}, key="turn_id")
        coding_acceptance(tel, "run-1", 0, at=T0 + 100)
        assert _outcome(tel, "e0")[0] == "forced_stop"

    def test_another_run_is_untouched(self, tel):
        self._episodes(tel, 1, run="run-other")
        coding_acceptance(tel, "run-1", 0, at=T0 + 100)
        assert _outcome(tel, "e0") == (None, None, None)

    def _session(self, tmp_path, test_body):
        from prometheus.coding.sandbox import ProcessSandbox
        from prometheus.coding.session import CodingSession, CodingTask

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "test_x.py").write_text(f"def test_x():\n    {test_body}\n")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "b"],
                       cwd=repo, check=True)
        db = tmp_path / "telemetry.db"
        t = ToolCallTelemetry(db)
        session = CodingSession(
            provider=_Script([_prose("done")] * 3), model="stub-model",
            sandbox=ProcessSandbox(root=repo),
            task=CodingTask(task_id="t-x", description="d", acceptance_command="python3 -m pytest -q"),
            telemetry=t, max_rounds=2, coding_run_id="t-x-0badcafe",
        )
        report = asyncio.run(session.run())
        t.close()
        return report, _rows(db, "SELECT * FROM turns ORDER BY started_at")

    def test_a_real_run_that_goes_green_stamps_its_last_episode(self, tmp_path):
        report, turns = self._session(tmp_path, "assert True")
        assert report.status == "success"
        assert turns[-1]["outcome"] == "accepted_verified"
        assert turns[-1]["outcome_source"] == "acceptance_command"
        assert all(t["outcome"] is None for t in turns[:-1]), \
            "episodes rejected for no evidence were never judged by the acceptance command"

    def test_a_real_run_that_stays_red_stamps_its_last_episode_rejected(self, tmp_path):
        report, turns = self._session(tmp_path, "assert False")
        assert report.status == "failed_abandoned"
        assert turns[-1]["outcome"] == "rejected_verified"


# --------------------------------------------------------------------------- #
# The writer verb the outcomes ride on
# --------------------------------------------------------------------------- #


class TestWriterCall:

    def test_a_call_runs_in_order_after_the_rows_queued_before_it(self, tmp_path):
        ToolCallTelemetry(tmp_path / "t.db").close()
        w = TelemetryV2Writer(tmp_path / "t.db")
        seen = []
        w.upsert("turns", {"turn_id": "x", "session_id": "s"}, key="turn_id")
        w.call("probe", lambda conn: seen.append(
            conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]))
        w.close()
        assert seen == [1]

    def test_a_failing_call_is_a_silent_failure_and_the_next_row_lands(self, tmp_path):
        ToolCallTelemetry(tmp_path / "t.db").close()
        w = TelemetryV2Writer(tmp_path / "t.db")
        w.call("probe", lambda conn: conn.execute("SELECT nope FROM nowhere"))
        w.upsert("turns", {"turn_id": "x", "session_id": "s"}, key="turn_id")
        w.close()
        assert _rows(tmp_path / "t.db", "SELECT COUNT(*) FROM turns")[0][0] == 1
        rows = _rows(tmp_path / "t.db", "SELECT operation FROM silent_failures")
        assert [r[0] for r in rows] == ["probe"]

    def test_a_label_that_is_not_an_identifier_is_refused(self, tmp_path):
        ToolCallTelemetry(tmp_path / "t.db").close()
        w = TelemetryV2Writer(tmp_path / "t.db")
        try:
            with pytest.raises(ValueError):
                w.call("drop table; --", lambda conn: None)
        finally:
            w.close()


# --------------------------------------------------------------------------- #
# Ingress wiring: every hooked surface calls the hook, before the turn
# --------------------------------------------------------------------------- #


class TestIngressWiring:

    def test_beacon_labels_the_previous_turn_when_a_message_arrives(self, tel, monkeypatch):
        """Through the real WS handler: the label lands, and before the turn lock."""
        from prometheus.telemetry import tracker
        from prometheus.web.ws_server import WebSocketBridge

        monkeypatch.setattr(tracker, "_telemetry_singleton", tel)
        now = time.time()
        _turn(tel, "t1", started=now - 20, ended=now - 10)

        class _Session:
            messages: list = []

            def add_user_message(self, content, **kw):
                self.messages.append(content)
                return len(self.messages)

            def last_persisted_row_id(self):
                return 1

        class _Mgr:
            def get(self, sid):
                return None

            def get_or_create(self, sid):
                return _Session()

        bridge = WebSocketBridge(loop_context=None, session_mgr=_Mgr())

        async def _noop(*a, **k):
            return None

        monkeypatch.setattr(bridge, "broadcast", _noop)
        asyncio.run(bridge._handle_send_message(SESSION, "thanks!"))
        assert _outcome(tel, "t1")[0] == "accepted_user"

    def test_a_slash_command_is_not_a_signal(self, tel, monkeypatch):
        from prometheus.telemetry import tracker
        from prometheus.web.ws_server import WebSocketBridge

        monkeypatch.setattr(tracker, "_telemetry_singleton", tel)
        calls = []
        monkeypatch.setattr(outcomes, "note_user_message", lambda *a, **k: calls.append(a))
        bridge = WebSocketBridge(loop_context=None, session_mgr=None)

        async def _noop(*a, **k):
            return None

        monkeypatch.setattr(bridge, "broadcast", _noop)
        asyncio.run(bridge._handle_send_message(SESSION, "/help"))
        assert calls == []

    def test_rest_chat_labels_under_the_recorded_session_id(self, tmp_path, monkeypatch):
        """/api/chat records its turns as web:<id> (T-3), so the hook must too."""
        calls = []
        monkeypatch.setattr(outcomes, "note_user_message",
                            lambda sid, text, **k: calls.append((sid, text)))
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from prometheus.engine.session import SessionManager
        from prometheus.skills.registry import SkillRegistry
        from prometheus.web.server import create_app
        from tests.test_api_chat_reaches_the_model import _RecordingLoop

        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
        app = create_app({"gateway": {"system_prompt": "sys"}}, session_mgr=SessionManager(),
                         skill_registry=SkillRegistry(), agent_loop=_RecordingLoop())
        r = TestClient(app).post("/api/chat", json={"session_id": "abc", "content": "hello there"})
        assert r.status_code == 200, r.text
        assert calls == [("web:abc", "hello there")]

    @pytest.mark.parametrize("path,func,session_expr,text_expr", [
        ("src/prometheus/gateway/telegram.py", "_dispatch_to_agent", "event.session_key()",
         "event.text"),
        ("src/prometheus/gateway/slack.py", "_dispatch_to_agent", "session_id", "text"),
        ("src/prometheus/gateway/discord.py", "_dispatch_to_agent", "session_id", "event.text"),
    ])
    def test_each_gateway_calls_the_hook_before_its_turn(self, path, func, session_expr,
                                                          text_expr):
        src = (Path(__file__).resolve().parents[1] / path).read_text()
        start = src.index(f"def {func}(")
        end = src.find("\n    async def ", start + 1)
        body = src[start:end if end != -1 else len(src)]
        hook = f"note_user_message({session_expr}, {text_expr}"
        assert hook in body, f"{path}:{func} does not call {hook}…)"
        turn_markers = [m for m in ("add_user_message(", "_run_agent_turn(") if m in body]
        assert turn_markers and all(body.index(hook) < body.index(m) for m in turn_markers), \
            "the hook must run before the message joins the session"


# --------------------------------------------------------------------------- #
# Parity: outcome_at is a wall-clock time (Will's open item from T-1)
# --------------------------------------------------------------------------- #


def test_the_parity_harness_normalizes_outcome_at():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from parity import normalize as norm

    assert norm._norm_value("outcome_at", T0 + 0.25, norm._Ordinals()) == "<time>"
    assert norm._norm_value("outcome_at", None, norm._Ordinals()) is None, \
        "whether it is NULL is still compared"
