"""LONGHAUL-1 acceptance — the progress-aware repeat detector.

Enters at the REAL REST surface (POST /api/chat/send → WebSocketBridge →
run_loop → real tool registry → real `bash` tool) with the boundary recorder
as the only double, and asserts on OUTCOMES: did the turn actually stop, does
the operator-facing message name the tool/args/count, and did a telemetry row
land in `subsystem_runs`. No test here asserts that a function was called.

Both directions are covered on purpose. A detector that halts loops but also
halts legitimate long work is not an improvement, so the productive-repeat
test is as load-bearing as the killer test.
"""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("fastapi")

from prometheus.telemetry.tracker import ToolCallTelemetry  # noqa: E402
from tests.support.real_app import (  # noqa: E402
    BOUNDARY_DOUBLE,
    RecordingProvider,
    build_real_app,
)

SESSION = "web"


def _build(tmp_path, script):
    """Real app + real telemetry on a scratch DB. Telemetry is the production
    class writing a real SQLite file — not a stub, so the row assertion is a
    real read-back."""
    rec = RecordingProvider(label="primary:local", script=script)
    # Point the REAL tools at a workspace that exists on every machine. The
    # shipped default is absent on CI, and bash then returns a workspace-lock
    # violation instead of running -- which trips the detector via the
    # identical-ERROR path and makes a success-path test pass for the wrong
    # reason. That is exactly what the first cut of this file did.
    h = build_real_app(
        primary=rec,
        tool_config={"workspace_root": str(tmp_path)},
    )
    tel = ToolCallTelemetry(str(tmp_path / "telemetry.db"))
    h.loop_context.telemetry = tel
    return h, rec, tel


def _repeat_rows(tel) -> list[tuple]:
    """Read subsystem_runs back out of the real DB."""
    con = sqlite3.connect(tel.db_path if hasattr(tel, "db_path") else str(tel._conn))
    try:
        return con.execute(
            "SELECT subsystem, operation, outcome, summary_json FROM subsystem_runs "
            "WHERE subsystem = 'repeat_detector'"
        ).fetchall()
    finally:
        con.close()


def _tool_result_texts(rec) -> list[str]:
    """Every tool result that rode back to the model, read off the OUTBOUND
    boundary record. This is how a test proves the real tool actually RAN and
    what it returned — rather than trusting that it did."""
    out = []
    # The LAST request carries the whole history; iterating every request would
    # count each result once per subsequent round.
    for req in rec.requests[-1:]:
        for msg in req.messages:
            for block in (msg.content if isinstance(msg.content, list) else []):
                if getattr(block, "type", None) == "tool_result":
                    out.append(block.content)
    return out


def _final_text(h) -> str:
    """Last assistant text in the REAL session history."""
    sess = h.session_mgr.get_or_create(SESSION)
    for msg in reversed(sess.messages):
        if msg.role == "assistant" and (msg.text or "").strip():
            return msg.text
    return ""


# --------------------------------------------------------------------------- #
# 1 — THE KILLER: identical call, identical result, three times → halt
# --------------------------------------------------------------------------- #


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_unproductive_repeat_halts_the_turn(tmp_path):
    """Same `bash` call returning byte-identical output must stop the turn at
    the 3rd occurrence — well before max_tool_iterations (25) would notice.

    The 5 scripted tool rounds are deliberate headroom: if the detector does
    not fire, the loop consumes all of them and the assertions on request
    count and halt text both go red.
    """
    # Explicit cwd: the default (~/.prometheus/workspace) does not exist on a
    # clean machine, and bash then errors instead of running -- which is a
    # different scenario than this test means to exercise (see
    # test_failing_call_still_halts for that one).
    cmd = {"command": "echo LONGHAUL_CONSTANT", "cwd": str(tmp_path)}
    script = [("tool", "bash", cmd)] * 5 + [("text", "should never be reached")]
    h, rec, tel = _build(tmp_path, script)

    with h.client:
        h.send_turn(SESSION, "please repeat that command")

        # OUTCOME 1 — the loop stopped at the trip, not at the cap and not at
        # the end of the script. 3 outbound requests = round 1 + 2 + 3.
        assert len(rec.requests) == 3, (
            f"expected the turn to halt after the 3rd identical call "
            f"(3 outbound rounds), saw {len(rec.requests)} — detector did not trip"
        )

        # OUTCOME 2 — the halt is LOUD and names tool, args and count.
        text = _final_text(h)
        assert "Halted: no progress" in text, f"no halt message in history: {text!r}"
        assert "bash" in text, f"halt message does not name the tool: {text!r}"
        assert "3 times" in text, f"halt message does not carry the count: {text!r}"
        assert "LONGHAUL_CONSTANT" in text, (
            f"halt message does not carry the arguments: {text!r}"
        )
        assert "should never be reached" not in text, "loop ran past the halt"

        # The tool really RAN and really returned the identical payload. Without
        # this the test passes just as happily when bash refuses with a
        # workspace-lock violation every round — an identical ERROR also trips
        # the detector, and an earlier cut of this test was green for exactly
        # that wrong reason.
        results = _tool_result_texts(rec)
        assert results, "no tool results reached the boundary"
        assert all("LONGHAUL_CONSTANT" in r for r in results), (
            f"bash did not actually execute — results were {results!r}"
        )
        assert not any("Workspace lock violation" in r for r in results), results

        # OUTCOME 3 — a real telemetry row, read back out of real SQLite.
        rows = _repeat_rows(tel)
        assert len(rows) == 1, f"expected 1 repeat_detector row, got {rows!r}"
        subsystem, operation, outcome, summary = rows[0]
        assert (subsystem, operation, outcome) == ("repeat_detector", "trip", "failed")
        assert '"tool_name": "bash"' in summary, summary
        assert '"repeat_count": 3' in summary, summary


# --------------------------------------------------------------------------- #
# 2 — THE OTHER DIRECTION: identical call, DIFFERENT results → no halt
# --------------------------------------------------------------------------- #


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_productive_repeat_does_not_halt(tmp_path):
    """The same command issued 4× while returning NEW data every time is
    progress, not a loop, and must run to completion.

    This is the polling/tailing shape, and it is the whole reason the detector
    keys on results rather than on call count. The counter file makes the
    differing output deterministic — no clock, no randomness.
    """
    counter = tmp_path / "counter.txt"
    cmd = {"command": f"echo x >> {counter}; wc -l < {counter}", "cwd": str(tmp_path)}
    script = [("tool", "bash", cmd)] * 4 + [("text", "finished counting")]
    h, rec, tel = _build(tmp_path, script)

    with h.client:
        h.send_turn(SESSION, "keep polling until it changes")

        # Ran every scripted round + the closing text: 5 outbound requests.
        assert len(rec.requests) == 5, (
            f"expected all 4 tool rounds + final text (5 outbound), saw "
            f"{len(rec.requests)} — a productive repeat was halted"
        )

        text = _final_text(h)
        assert "Halted: no progress" not in text, (
            f"productive repeat was halted — FALSE POSITIVE: {text!r}"
        )
        assert "finished counting" in text, f"turn did not complete: {text!r}"

        # Each round really returned NEW data — that is what makes this a
        # productive repeat rather than a loop.
        results = _tool_result_texts(rec)
        assert len(set(results)) == len(results) == 4, (
            f"expected 4 distinct real results, got {results!r}"
        )

        assert _repeat_rows(tel) == [], (
            "a repeat_detector telemetry row was written for productive work"
        )


# --------------------------------------------------------------------------- #
# 3 — the `printf ''` shape from the live telemetry replay
# --------------------------------------------------------------------------- #


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_no_op_command_halts(tmp_path):
    """The `printf ''` shape found in the live replay must halt.

    Note what the real tool layer does here: BashTool substitutes the constant
    "(no output)" for empty output (tools/builtin/bash.py), so a no-op command
    reaches the detector as an IDENTICAL result rather than an empty one. It
    trips either way — but that is why the empty branch is exercised at unit
    level below instead of being asserted here: asserting "returned nothing"
    through bash would be asserting something the tool layer never produces.
    """
    script = [
        ("tool", "bash", {"command": "printf ''", "cwd": str(tmp_path)})
    ] * 5 + [("text", "unreached")]
    h, rec, tel = _build(tmp_path, script)

    with h.client:
        h.send_turn(SESSION, "run the no-op")

        assert len(rec.requests) == 3, (
            f"expected halt on the 3rd no-op call, saw {len(rec.requests)} rounds"
        )
        text = _final_text(h)
        assert "Halted: no progress" in text, text
        assert "unreached" not in text, "loop ran past the halt"
        rows = _repeat_rows(tel)
        assert len(rows) == 1, rows


# --------------------------------------------------------------------------- #
# 4 — unit-level: the empty branch, and the reset-on-progress rule
# --------------------------------------------------------------------------- #


def test_empty_result_is_unproductive_from_the_first_occurrence():
    """A tool that genuinely returns "" has no prior to differ from, so
    emptiness must be judged on its own. Unit-level because the real tool
    layer substitutes placeholder text (see test_no_op_command_halts)."""
    from prometheus.engine.agent_loop import _ProgressRepeatDetector

    class _Call:
        name = "grep"
        input = {"pattern": "nothing-matches-this"}

    det = _ProgressRepeatDetector()
    assert det.record(_Call(), "") is None       # 1st
    assert det.record(_Call(), "") is None       # 2nd
    trip = det.record(_Call(), "")               # 3rd → trip
    assert trip is not None and trip.count == 3 and trip.empty is True


def test_progress_resets_the_signature_history():
    """read(A), read(A), read(B) must leave NO loaded gun behind: the B result
    is progress, so the next repeat starts counting from zero.

    This is the rule that keeps the smoke test alive — its reads of hello.txt
    bracket an edit_file and legitimately return different bytes.
    """
    from prometheus.engine.agent_loop import _ProgressRepeatDetector

    class _Call:
        name = "read_file"
        input = {"path": "/tmp/hello.txt"}

    det = _ProgressRepeatDetector()
    assert det.record(_Call(), "A") is None
    assert det.record(_Call(), "A") is None      # unproductive, count 2
    assert det.record(_Call(), "B") is None      # progress → reset
    # Had the reset merely "not incremented", this next identical pair would
    # trip. It must not.
    assert det.record(_Call(), "B") is None      # count 2 again
    assert det.record(_Call(), "C") is None      # progress → reset again
    assert det.record(_Call(), "C") is None
    assert det.record(_Call(), "C") is not None  # now a genuine 3-run trips


def test_distinct_arguments_never_collide():
    """Keyed on (name, input): the same tool with different arguments is
    ordinary work and must never accumulate toward a trip."""
    from prometheus.engine.agent_loop import _ProgressRepeatDetector

    def _call(path):
        return type("C", (), {"name": "read_file", "input": {"path": path}})()

    det = _ProgressRepeatDetector()
    for i in range(10):
        assert det.record(_call(f"/tmp/f{i}.txt"), "") is None, (
            "distinct arguments accumulated toward a trip"
        )


# --------------------------------------------------------------------------- #
# 5 — regression: a call that FAILS every round must halt too
# --------------------------------------------------------------------------- #


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_all_error_rounds_are_left_to_the_circuit_breaker(tmp_path):
    """A call that errors EVERY round must terminate the turn — but via the
    circuit breaker, not this detector.

    The division of labour matters and is not cosmetic. The breaker gets one
    diagnose-and-recover attempt that can SALVAGE the turn (model fallback on a
    formatting error, a tier bump); the repeat detector can only halt. An
    earlier cut of this change tripped first and killed turns the breaker would
    have recovered — caught by
    tests/test_wiring.py::test_trip_handler_calls_diagnose_and_recover.

    Deliberately not mocked: a cwd that does not exist makes the real bash tool
    fail deterministically on any machine.
    """
    missing = tmp_path / "does-not-exist"
    script = [
        ("tool", "bash", {"command": "echo nope", "cwd": str(missing)})
    ] * 8 + [("text", "unreached")]
    h, rec, tel = _build(tmp_path, script)

    with h.client:
        h.send_turn(SESSION, "run the doomed command")

        text = _final_text(h)
        assert "unreached" not in text, "the doomed turn ran to the end of the script"
        # The breaker owns this outcome, so no repeat_detector row is written.
        assert _repeat_rows(tel) == [], (
            "the repeat detector preempted the circuit breaker on an all-error "
            "round — that trades a recoverable turn for a dead one"
        )
        assert "Halted: no progress" not in text, text


def test_blocked_results_are_unproductive_by_construction():
    """A result the repeat guard SYNTHESISED without executing the tool carries
    no new information, whatever its text says.

    Its text embeds the running failure count, so it CHANGES every round.
    Judging it by fingerprint read as fresh data and reset the counter that
    should have been climbing — which is how a doomed call rode the flat cap to
    exhaustion. Unit-level because reaching a blocked result in a round that is
    not all-errors needs a mixed parallel batch the scripted recorder cannot
    express.
    """
    from prometheus.engine.agent_loop import _ProgressRepeatDetector

    class _Call:
        name = "bash"
        input = {"command": "doomed"}

    det = _ProgressRepeatDetector()
    # Text differs every time, exactly like the real BLOCKED message.
    assert det.record(_Call(), "BLOCKED: already failed 2 times", blocked=True) is None
    assert det.record(_Call(), "BLOCKED: already failed 3 times", blocked=True) is None
    trip = det.record(_Call(), "BLOCKED: already failed 4 times", blocked=True)
    assert trip is not None and trip.count == 3, (
        "changing BLOCKED text reset the counter — the detector is judging "
        "synthesised results by their bytes again"
    )


def test_window_evicts_old_occurrences():
    """The window is ROLLING: occurrences pushed out by other work no longer
    count toward a trip.

    Two unproductive hits on one signature, then a full window of other calls,
    then a third hit — the first two have aged out, so this must NOT trip.
    Without the window bound the detector would accumulate across an entire
    turn and halt work that had long since moved on.
    """
    from prometheus.engine.agent_loop import _ProgressRepeatDetector

    def _call(name, arg):
        return type("C", (), {"name": name, "input": {"a": arg}})()

    det = _ProgressRepeatDetector()
    assert det.record(_call("grep", "target"), "") is None
    assert det.record(_call("grep", "target"), "") is None  # 2 unproductive

    for i in range(det.window):
        assert det.record(_call("read_file", f"other-{i}"), f"body-{i}") is None

    # The two earlier hits have aged out of the window.
    assert det.record(_call("grep", "target"), "") is None, (
        "an occurrence that aged out of the window still counted toward a trip"
    )
