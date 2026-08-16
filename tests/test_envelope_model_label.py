"""Subsystem telemetry rows must name the model that produced them.

THE ASYMMETRY THAT SCOPED THIS. ``agent_loop`` had 7,973 ``subsystem_runs``
rows with ZERO empty model labels while six subsystems on the SAME
``LLMCallEnvelope`` had 100% empty (~12,900 rows). It was never the callers —
all of them already pass ``model`` to ``call()``. The envelope has two
telemetry paths: ``stream()`` writes the full row with
``model=request.model``, and ``call()`` routed to ``_record_success`` /
``_record_failure``, which wrote an abbreviated row and dropped it.

So the fix is in the envelope, at the layer that dropped the value, not at
six call sites that were already supplying it.

Everything here reads rows back from ``subsystem_runs``. Asserting on the
call arguments would pass against the bug: the arguments were always right.

``tool_registration`` is the control and must stay EMPTY — it never builds an
envelope and has no model to name. It is asserted so a later "fill every null
label" change has something to break against.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from prometheus.learning.llm_envelope import LLMCallEnvelope
from prometheus.providers.base import ApiTextDeltaEvent
from prometheus.telemetry.tracker import ToolCallTelemetry

pytestmark = pytest.mark.integration

#: The six that invoke the envelope via call() and recorded no model.
SIX = [
    "memory_extractor", "knowledge_synth", "skill_refiner",
    "skill_creator", "context_compactor", "curator",
]

MODEL = "Qwen3.8-27B-UD-Q4_K_XL.gguf"


class _Provider:
    """Minimal stand-in: streams one chunk, records nothing itself."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def stream_message(self, request):
        if self.fail:
            raise RuntimeError("provider exploded")
        # ApiTextDeltaEvent — the concrete class. ApiStreamEvent is a UNION alias The wrong name here raised inside
        # the generator, so every "success" case silently took the FAILURE
        # path and passed because the failure writer happened to be fixed —
        # green for the wrong reason, and the success mutations survived.
        yield ApiTextDeltaEvent(text="ok")


def _tracker(tmp_path) -> ToolCallTelemetry:
    return ToolCallTelemetry(db_path=tmp_path / "telemetry.db")


def _rows(tracker, subsystem):
    """Read back from the DB — not from what the caller passed in."""
    with sqlite3.connect(tracker.db_path) as conn:
        return conn.execute(
            "select model from subsystem_runs where subsystem=?", (subsystem,)
        ).fetchall()


def _outcomes(tracker, subsystem):
    with sqlite3.connect(tracker.db_path) as conn:
        return [r[0] for r in conn.execute(
            "select outcome from subsystem_runs where subsystem=?", (subsystem,))]


def _run_through_envelope(tracker, subsystem, *, fail=False):
    env = LLMCallEnvelope(subsystem=subsystem, telemetry=tracker,
                          on_failure="return_none")
    asyncio.run(env.call(provider=_Provider(fail=fail), model=MODEL,
                         prompt="p", operation="_call_model"))


class TestTheSixNowRecordTheirModel:
    @pytest.mark.parametrize("subsystem", SIX)
    def test_a_successful_run_records_the_model(self, tmp_path, subsystem):
        t = _tracker(tmp_path)
        _run_through_envelope(t, subsystem)
        rows = _rows(t, subsystem)
        assert rows, f"{subsystem} wrote no subsystem_runs row at all"
        assert _outcomes(t, subsystem) == ["success"], (
            "this case must exercise the SUCCESS writer; a broken double that "
            "falls into the failure path makes this test green for the wrong "
            "reason (it did, once)"
        )
        assert all(r[0] == MODEL for r in rows), (
            f"{subsystem} recorded {rows!r}; the model was in scope at the "
            "call and was dropped by the envelope's abbreviated writer"
        )

    @pytest.mark.parametrize("subsystem", SIX)
    def test_a_failed_run_also_records_the_model(self, tmp_path, subsystem):
        """The failure writer dropped it too, and a failed row is the one an
        operator most needs to attribute to a model."""
        t = _tracker(tmp_path)
        _run_through_envelope(t, subsystem, fail=True)
        rows = _rows(t, subsystem)
        assert rows and all(r[0] == MODEL for r in rows), rows

    def test_the_label_is_read_from_the_db_not_the_arguments(self, tmp_path):
        """Guard against the test that would have passed against the bug."""
        t = _tracker(tmp_path)
        _run_through_envelope(t, "curator")
        with sqlite3.connect(t.db_path) as conn:
            got = conn.execute(
                "select model from subsystem_runs where subsystem='curator'"
            ).fetchone()
        assert got and got[0] == MODEL


class TestToolRegistrationStaysEmpty:
    """The control. It never constructs an envelope, so it has no model.

    A synthetic label here would make the column look uniform and mean
    nothing — worse than the null, which is honest.
    """

    def test_a_row_with_no_model_stays_null(self, tmp_path):
        t = _tracker(tmp_path)
        t.record_run(subsystem="tool_registration", operation="register",
                     outcome="success", duration_ms=1.0)
        rows = _rows(t, "tool_registration")
        assert rows, "no row written"
        assert all(r[0] in (None, "") for r in rows), (
            f"tool_registration acquired a model label: {rows!r}. It builds no "
            "envelope and calls no model; a label here would be invented."
        )

    def test_it_is_not_swept_up_by_a_fill_every_null_change(self, tmp_path):
        t = _tracker(tmp_path)
        t.record_run(subsystem="tool_registration", operation="register",
                     outcome="success", duration_ms=1.0)
        _run_through_envelope(t, "curator")
        with sqlite3.connect(t.db_path) as conn:
            got = dict(conn.execute(
                "select subsystem, coalesce(model,'') from subsystem_runs"
            ).fetchall())
        assert got["curator"] == MODEL
        assert got["tool_registration"] == "", (
            "both subsystems must NOT converge on one label — they are "
            "different things and the column is right to disagree"
        )


class TestNoLabelIsEverInvented:
    """The envelope must record what it was given, including nothing.

    Caught by a surviving mutation (M-SYNTHETIC: `model = model or "unknown"`).
    Every other test here supplies a model, so nothing observed the case where
    one is absent — and a placeholder would make the column look uniform while
    asserting something false about which model ran.
    """

    @pytest.mark.parametrize("supplied", ["", None])
    def test_an_absent_model_stays_absent(self, tmp_path, supplied):
        t = _tracker(tmp_path)
        env = LLMCallEnvelope(subsystem="curator", telemetry=t,
                              on_failure="return_none")
        asyncio.run(env.call(provider=_Provider(), model=supplied,
                             prompt="p", operation="_call_model"))
        rows = _rows(t, "curator")
        assert rows and all(r[0] in (None, "") for r in rows), (
            f"the envelope invented a label: {rows!r}"
        )
