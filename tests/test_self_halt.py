"""LONGHAUL-2 acceptance — the self-halt.

The 2026-08-17 finding: a stuck turn was reproducible on demand (30 steps in
~25s), the divergence detector warned on every iteration and took no action,
and only a manual interrupt stopped it. LONGHAUL-1 (#263) halted the
identical-arguments shape and recorded the varied-arguments shape as its
blind spot — while raising the named backstop (max_tool_iterations) to 500.
This file pins the two mechanisms that close that blind spot:

1. VARIED-ARGS TRACK (always on): the same READ-ONLY tool, 6 adjacent calls,
   no new bytes → halt. Read-only is the load-bearing gate — quiet-but-
   effective bash runs and constant-ack message broadcasts are legitimate
   varied-args-same-result workflows, and both come from tools that declare
   themselves NOT read-only.
2. DIVERGENCE PERSISTENCE HALT (divergence.halt_on_repetition, default on):
   the repetition floor held for 3 CONSECUTIVE evaluations ends the turn
   forward. Consecutive is calibrated from live traffic: healthy turns blip
   the floor for exactly one evaluation.

Same conventions as tests/test_acceptance_repeat_detector.py: the REAL REST
surface, real tools, the boundary recorder as the only double, assertions on
outcomes. Both directions covered — a halt that also stops legitimate work
is not an improvement, so the no-halt tests are as load-bearing as the
killers.
"""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("fastapi")

from prometheus.coordinator.divergence import DivergenceDetector  # noqa: E402
from prometheus.telemetry.tracker import ToolCallTelemetry  # noqa: E402
from tests.support.real_app import (  # noqa: E402
    BOUNDARY_DOUBLE,
    RecordingProvider,
    build_real_app,
)

SESSION = "web"


def _build(tmp_path, script, divergence: dict | None = None):
    rec = RecordingProvider(label="primary:local", script=script)
    h = build_real_app(
        primary=rec,
        tool_config={"workspace_root": str(tmp_path)},
    )
    tel = ToolCallTelemetry(str(tmp_path / "telemetry.db"))
    h.loop_context.telemetry = tel
    if divergence is not None:
        h.loop_context.divergence_detector = DivergenceDetector(
            {"divergence": divergence}
        )
    return h, rec, tel


def _rows(tel, subsystem: str) -> list[tuple]:
    con = sqlite3.connect(str(tel._db_path))
    try:
        return con.execute(
            "SELECT operation, outcome, summary_json FROM subsystem_runs "
            "WHERE subsystem = ?", (subsystem,),
        ).fetchall()
    finally:
        con.close()


def _final_text(h) -> str:
    sess = h.session_mgr.get_or_create(SESSION)
    for msg in reversed(sess.messages):
        if msg.role == "assistant" and (msg.text or "").strip():
            return msg.text
    return ""


def _grep_call(i: int, root: str) -> tuple:
    # Six DIFFERENT patterns against the same empty directory: the real grep
    # tool returns byte-identical "(no matches)" for every one — new
    # arguments, no new information. The motivating trace.
    return ("tool", "grep", {"pattern": f"needle-{i}", "root": root})


# --------------------------------------------------------------------------- #
# 1 — THE KILLER: varied arguments, read-only tool, nothing new, six times
# --------------------------------------------------------------------------- #


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_varied_argument_flail_halts_at_six(tmp_path):
    search_root = tmp_path / "empty"
    search_root.mkdir()
    script = [_grep_call(i, str(search_root)) for i in range(9)] + [
        ("text", "should never be reached")
    ]
    h, rec, tel = _build(tmp_path, script)

    with h.client:
        h.send_turn(SESSION, "find where the needle constant is defined")

        # The loop stopped at the 6th adjacent no-information call — not at
        # the end of the script, and nowhere near max_tool_iterations.
        assert len(rec.requests) == 6, (
            f"expected the varied-args track to halt after 6 adjacent "
            f"no-match greps, saw {len(rec.requests)} outbound rounds"
        )

        text = _final_text(h)
        assert "varying arguments" in text
        assert "grep" in text

        rows = _rows(tel, "repeat_detector")
        assert rows, "no repeat_detector telemetry row landed"
        assert any('"varied_args": true' in r[2] for r in rows)


# --------------------------------------------------------------------------- #
# 2 — the other direction: varied reads that RETURN things keep running
# --------------------------------------------------------------------------- #


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_productive_varied_searches_do_not_halt(tmp_path):
    search_root = tmp_path / "corpus"
    search_root.mkdir()
    for i in range(9):
        (search_root / f"f{i}.txt").write_text(f"needle-{i} lives here\n")
    script = [_grep_call(i, str(search_root)) for i in range(9)] + [
        ("text", "done searching")
    ]
    h, rec, tel = _build(tmp_path, script)

    with h.client:
        h.send_turn(SESSION, "map every needle constant")
        # Every search matched something different — 9 tool rounds + the
        # closing text round all execute.
        assert len(rec.requests) == 10
        assert _rows(tel, "repeat_detector") == []


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_quiet_mutating_bash_run_does_not_halt(tmp_path):
    # Seven DIFFERENT mkdir commands, each legitimately returning nothing.
    # bash is not read-only, so the varied-args track must not apply — this
    # is the false-positive shape the read-only gate exists for.
    script = [
        ("tool", "bash", {"command": f"mkdir -p d{i}", "cwd": str(tmp_path)})
        for i in range(7)
    ] + [("text", "made the directories")]
    h, rec, tel = _build(tmp_path, script)

    with h.client:
        h.send_turn(SESSION, "create the directory scaffold")
        assert len(rec.requests) == 8
        assert _rows(tel, "repeat_detector") == []


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_interleaved_tool_breaks_the_adjacent_run(tmp_path):
    search_root = tmp_path / "empty"
    search_root.mkdir()
    readable = tmp_path / "notes.txt"
    readable.write_text("real content\n")
    script = (
        [_grep_call(i, str(search_root)) for i in range(5)]
        + [("tool", "read_file", {"path": str(readable)})]
        + [_grep_call(i + 10, str(search_root)) for i in range(5)]
        + [("text", "done")]
    )
    h, rec, tel = _build(tmp_path, script)

    with h.client:
        h.send_turn(SESSION, "search around")
        # Adjacency is the reset: 5 + 1 + 5 never has 6 same-tool adjacent
        # no-information calls, so all 11 tool rounds + text run.
        assert len(rec.requests) == 12
        assert _rows(tel, "repeat_detector") == []


# --------------------------------------------------------------------------- #
# 3 — divergence persistence halt: the non-read-only shape (bash flail)
# --------------------------------------------------------------------------- #

_DIV_ON = {"enabled": True, "threshold": 0.7, "halt_on_repetition": True}


def _empty_bash(i: int, cwd: str) -> tuple:
    # Varied commands, every one silent: `:` is a no-op; the comment varies
    # the arguments. This is the 2026-08-17 shape — bash, different args,
    # no information — which the read-only gate deliberately exempts from
    # track 1 and this mechanism exists to stop.
    return ("tool", "bash", {"command": f": probe-{i}", "cwd": cwd})


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_sustained_divergence_halts_the_turn(tmp_path):
    script = [_empty_bash(i, str(tmp_path)) for i in range(14)] + [
        ("text", "should never be reached")
    ]
    h, rec, tel = _build(tmp_path, script, divergence=_DIV_ON)

    with h.client:
        h.send_turn(SESSION, "figure out why the build is broken")

        assert len(rec.requests) < 14, (
            "the sustained-divergence halt never fired — the turn consumed "
            "the whole flail script"
        )
        text = _final_text(h)
        assert "Halted" in text
        assert "repetition" in text

        rows = _rows(tel, "divergence")
        assert any(r[0] == "halt" for r in rows), (
            "no divergence/halt telemetry row landed"
        )


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_halt_off_restores_warn_only(tmp_path):
    div = dict(_DIV_ON, halt_on_repetition=False)
    script = [_empty_bash(i, str(tmp_path)) for i in range(14)] + [
        ("text", "flail complete")
    ]
    h, rec, tel = _build(tmp_path, script, divergence=div)

    with h.client:
        h.send_turn(SESSION, "figure out why the build is broken")
        # Warn-only: the whole script runs, exactly the pre-LONGHAUL-2 world.
        assert len(rec.requests) == 15
        assert _rows(tel, "divergence") == []


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_single_blip_does_not_halt(tmp_path):
    # Live calibration: healthy turns blip the repetition floor for ONE
    # evaluation. Five silent probes followed by commands that produce
    # distinct output must run to completion — halting here would kill
    # real turns weekly.
    script = (
        [_empty_bash(i, str(tmp_path)) for i in range(5)]
        + [
            ("tool", "bash", {"command": f"echo found-{i}", "cwd": str(tmp_path)})
            for i in range(5)
        ]
        + [("text", "recovered and finished")]
    )
    h, rec, tel = _build(tmp_path, script, divergence=_DIV_ON)

    with h.client:
        h.send_turn(SESSION, "figure out why the build is broken")
        assert len(rec.requests) == 11
        assert _rows(tel, "divergence") == []
