"""/api/lcm reports what the model was SENT — the context meter's numerator.

The route used to report the LCM assembler's view of the conversation: the store's text, and
nothing else. The loop's request also carries the system prompt, the tool schemas, and every tool
call and result, so on a live 730-message agent session the meter read 4,018 tokens while the loop
was sending 110k-243k per round (2026-09-23) — fifty times low, on the one surface that exists to
say "you are nearly full". Beacon desktop and iOS both draw this route, so both were wrong.

Rows are written through ``ToolCallTelemetry.record_run`` — the loop's real write path — not
inserted by hand, so a change to how the loop records a round is a change these tests see.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.memory.lcm_types import AssemblyResult, MessagePart, SummaryNode  # noqa: E402
from prometheus.telemetry.tracker import ToolCallTelemetry, set_telemetry_handle  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

LOCAL_MODEL = "Qwen3.8-27B-UD-Q4_K_XL.gguf"
CONFIG = {
    "model": {"model": LOCAL_MODEL, "provider": "llama_cpp"},
    "context": {"effective_limit": 72000},
}
SID = "beacon:measured"


class _Engine:
    """The assembly as the live session produced it: 4,018 = 3,747 fresh + 271 summary."""

    def assemble(self, session_id, token_budget):
        return AssemblyResult(
            summaries=[SummaryNode()],
            fresh_messages=[MessagePart(role="user", content="c") for _ in range(3)],
            total_tokens=4018, fresh_tokens=3747, summary_tokens=271, compression_ratio=4.96,
        )


def _round(tel, sid, tokens, *, age_s, subsystem="agent_loop", operation="loop_round",
           model="qwen3.8-max"):
    tel.record_run(subsystem, operation, "success", input_tokens=tokens, output_tokens=100,
                   session_id=sid, model=model)
    tel._conn.execute(
        "UPDATE subsystem_runs SET timestamp = ? WHERE rowid = (SELECT max(rowid) FROM subsystem_runs)",
        (time.time() - age_s,),
    )
    tel._conn.commit()


@pytest.fixture
def tel(tmp_path):
    t = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    set_telemetry_handle(t)
    yield t
    set_telemetry_handle(None)


def _get(sid=SID):
    client = TestClient(create_app(
        CONFIG, lcm_engine=_Engine(), detected_context_size=32768, local_model=LOCAL_MODEL,
    ))
    return client.get(f"/api/lcm/{sid}").json()


def test_the_meter_reports_the_last_request_the_loop_sent(tel):
    _round(tel, SID, 110_520, age_s=3600)
    _round(tel, SID, 242_676, age_s=60)
    body = _get()
    assert body["total_tokens"] == 242_676, "the meter must show what the model was sent"
    assert body["basis"] == "last_request"
    assert body["measured_model"] == "qwen3.8-max"
    assert isinstance(body["measured_at"], float)
    assert body["assembled"] is True, "a measured reading is a reading — clients test assembled === false"


def test_the_newest_request_wins_not_the_largest(tel):
    """Compaction shrinks the context. A meter showing the historical peak would never come down."""
    _round(tel, SID, 498_739, age_s=7200)
    _round(tel, SID, 228_592, age_s=30)
    assert _get()["total_tokens"] == 228_592


def test_other_sessions_other_subsystems_and_unreported_rounds_are_ignored(tel):
    _round(tel, SID, 50_000, age_s=120)
    _round(tel, "beacon:another-session", 400_000, age_s=10)
    _round(tel, SID, 900, age_s=5, subsystem="titler", operation="title")
    _round(tel, SID, 0, age_s=1)  # a round whose provider reported no usage
    assert _get()["total_tokens"] == 50_000


def test_the_split_is_dropped_because_it_no_longer_sums_but_the_store_facts_stay(tel):
    _round(tel, SID, 242_676, age_s=60)
    body = _get()
    assert body["fresh_tokens"] is None and body["summary_tokens"] is None, (
        "3,747 + 271 is the assembly's total, not this one — drawn inside the bar it would claim "
        "a composition of the 242k the model saw"
    )
    assert body["fresh_count"] == 3 and body["summary_count"] == 1
    assert body["compression_ratio"] == pytest.approx(4.96)


def test_with_nothing_measured_the_assembly_is_reported_and_says_so(tel):
    body = _get("beacon:never-sent")
    assert body["basis"] == "lcm_assembly"
    assert body["total_tokens"] == 4018
    assert body["fresh_tokens"] == 3747 and body["summary_tokens"] == 271
    assert "measured_at" not in body


def test_with_no_telemetry_at_all_the_route_still_answers():
    set_telemetry_handle(None)
    body = _get()
    assert body["basis"] == "lcm_assembly" and body["total_tokens"] == 4018


def test_a_timestamp_tie_resolves_to_the_later_insert(tel):
    """record_run's ids are random uuid4 hex, so they cannot order anything; rowid is insert order."""
    for n in (111, 222, 333):
        tel.record_run("agent_loop", "loop_round", "success", input_tokens=n, session_id=SID, model="m")
    tel._conn.execute("UPDATE subsystem_runs SET timestamp = 1000.0")
    tel._conn.commit()
    assert tel.last_request_tokens(SID)["input_tokens"] == 333
    assert tel.last_request_tokens("beacon:unknown") is None
