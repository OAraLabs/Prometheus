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
    h = build_real_app(primary=rec)
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
    cmd = {"command": "echo LONGHAUL_CONSTANT"}
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
    cmd = {"command": f"echo x >> {counter}; wc -l < {counter}"}
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
    script = [("tool", "bash", {"command": "printf ''"})] * 5 + [("text", "unreached")]
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
