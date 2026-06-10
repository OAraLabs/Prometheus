"""Tests for Sprint 3: ToolCallTelemetry."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from prometheus.telemetry.tracker import ToolCallTelemetry


@pytest.fixture
def tel(tmp_path: Path) -> ToolCallTelemetry:
    return ToolCallTelemetry(db_path=tmp_path / "telemetry.db")


class TestToolCallTelemetry:

    def test_record_and_report_single(self, tel):
        tel.record(
            model="qwen2.5",
            tool_name="bash",
            success=True,
            retries=0,
            latency_ms=100.0,
        )
        report = tel.report()
        assert report["total_calls"] == 1
        assert report["overall_success_rate"] == 1.0
        assert "qwen2.5" in report["models"]
        assert "bash" in report["models"]["qwen2.5"]
        assert report["models"]["qwen2.5"]["bash"]["calls"] == 1
        assert report["models"]["qwen2.5"]["bash"]["successes"] == 1
        assert report["models"]["qwen2.5"]["bash"]["success_rate"] == 1.0

    def test_report_empty_db(self, tel):
        report = tel.report()
        assert report["total_calls"] == 0
        assert report["overall_success_rate"] == 0.0
        assert report["models"] == {}
        assert report["tools"] == {}

    def test_success_rate_calculation(self, tel):
        tel.record("model", "bash", success=True, retries=0, latency_ms=50)
        tel.record("model", "bash", success=True, retries=0, latency_ms=60)
        tel.record("model", "bash", success=False, retries=2, latency_ms=200)
        report = tel.report()
        assert report["models"]["model"]["bash"]["calls"] == 3
        assert report["models"]["model"]["bash"]["successes"] == 2
        assert abs(report["models"]["model"]["bash"]["success_rate"] - 2 / 3) < 0.001
        assert report["models"]["model"]["bash"]["failures"] == 1

    def test_avg_retries(self, tel):
        tel.record("m", "t", success=False, retries=2, latency_ms=100)
        tel.record("m", "t", success=False, retries=4, latency_ms=100)
        report = tel.report()
        assert report["models"]["m"]["t"]["avg_retries"] == 3.0

    def test_avg_latency(self, tel):
        tel.record("m", "t", success=True, retries=0, latency_ms=100)
        tel.record("m", "t", success=True, retries=0, latency_ms=200)
        report = tel.report()
        assert report["models"]["m"]["t"]["avg_latency_ms"] == 150.0

    def test_per_tool_report(self, tel):
        tel.record("modelA", "bash", success=True, retries=0, latency_ms=100)
        tel.record("modelB", "bash", success=False, retries=1, latency_ms=200,
                   error_type="validation_failed")
        report = tel.report()
        assert "bash" in report["tools"]
        bash = report["tools"]["bash"]
        assert bash["calls"] == 2
        assert bash["success_rate"] == 0.5
        assert "validation_failed" in bash["error_types"]

    def test_multiple_models(self, tel):
        tel.record("qwen", "bash", success=True, retries=0, latency_ms=100)
        tel.record("llama", "bash", success=False, retries=1, latency_ms=300)
        report = tel.report()
        assert "qwen" in report["models"]
        assert "llama" in report["models"]
        assert report["total_calls"] == 2

    def test_error_detail_stored(self, tel):
        tel.record(
            "m", "t",
            success=False,
            retries=0,
            latency_ms=0,
            error_type="validation_failed",
            error_detail="missing required param: command",
        )
        report = tel.report()
        assert report["tools"]["t"]["error_types"]["validation_failed"] == 1

    def test_db_path_expanduser(self, tmp_path):
        # Verify no crash with tilde paths (we just use tmp_path for isolation)
        tel = ToolCallTelemetry(db_path=tmp_path / "sub" / "tel.db")
        tel.record("m", "t", success=True, retries=0, latency_ms=1)
        assert tel.report()["total_calls"] == 1
        tel.close()

    def test_persistence_across_instances(self, tmp_path):
        db = tmp_path / "tel.db"
        t1 = ToolCallTelemetry(db_path=db)
        t1.record("m", "bash", success=True, retries=0, latency_ms=50)
        t1.close()

        t2 = ToolCallTelemetry(db_path=db)
        report = t2.report()
        assert report["total_calls"] == 1
        t2.close()


class TestPolicyDenialDenominators:
    """D3 denominator honesty: SecurityGate/hook denials are policy outcomes,
    not model tool-calling failures — rates must judge only non-denied calls."""

    def test_health_summary_separates_denials_from_failures(self, tel):
        tel.record(model="m", tool_name="bash", success=True)
        tel.record(model="m", tool_name="bash", success=True)
        tel.record(model="m", tool_name="bash", success=False,
                   error_type="tool_error", error_detail="exit 1")
        tel.record(model="m", tool_name="bash", success=False,
                   error_type="permission_denied",
                   error_detail="Command matches deny list entry")
        tel.record(model="m", tool_name="bash", success=False,
                   error_type="hook_blocked")

        tc = tel.health_summary(since=0.0)["tool_calls"]
        assert tc["total"] == 5         # denials still count as calls made
        assert tc["denials"] == 2
        assert tc["failures"] == 1      # only the real tool_error
        assert tc["success_rate"] == pytest.approx(2 / 3)  # judged calls only

    def test_report_surfaces_denials_outside_rates(self, tel):
        tel.record(model="m", tool_name="bash", success=True, latency_ms=100.0)
        tel.record(model="m", tool_name="bash", success=False,
                   error_type="permission_denied")

        rep = tel.report()
        bash = rep["tools"]["bash"]
        assert bash["calls"] == 1               # judged calls only
        assert bash["denials"] == 1
        assert bash["success_rate"] == 1.0      # not 0.5
        # the denial must not appear in the failure error-type histogram
        assert "permission_denied" not in bash["error_types"]
        assert rep["total_calls"] == 1
        assert rep["total_denials"] == 1
        assert rep["overall_success_rate"] == 1.0
        # per-model view carries the same split
        m_bash = rep["models"]["m"]["bash"]
        assert m_bash["calls"] == 1
        assert m_bash["denials"] == 1
        assert m_bash["failures"] == 0

    def test_report_all_denied_tool_keeps_zero_rate_not_failure(self, tel):
        tel.record(model="m", tool_name="bash", success=False,
                   error_type="permission_denied")
        rep = tel.report()
        bash = rep["tools"]["bash"]
        assert bash["calls"] == 0
        assert bash["denials"] == 1
        assert bash["success_rate"] == 0.0
        assert rep["total_calls"] == 0
        assert rep["overall_success_rate"] == 0.0
