"""/api/usage says "not recorded" for cache counts it never had, not 0
(WP-X.21, reader half of T6; docs/audits/TELEMETRY-GAPS.md).

``subsystem_runs.cached_input_tokens`` is NULL when the provider reported
nothing about caching, deliberately distinct from 0, "the cache was cold"
(#119). The reader threw that away: ``usage_rollup`` summed the column as
``COALESCE(SUM(...), 0)``, so every model reported 0 cached tokens. On the mini
no row has ever held a cache count, so the usage view said "0 cached" for 14,764
rounds whose cache use was simply unrecorded — the same collapse ``cost_usd:
null`` exists to prevent.

Pinned here: a group with no recorded count reports ``cached_input_tokens:
null``; ``cache_reported_runs`` says how many of its runs did record one; a
recorded 0 stays 0; the totals follow the same rule.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import prometheus.telemetry.tracker as tracker  # noqa: E402
from prometheus.telemetry.tracker import ToolCallTelemetry  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402


def _round(tel, model, cached=None):
    tel.record_run("agent_loop", "loop_round", "success", input_tokens=1000, output_tokens=10,
                   model=model, session_id="s1", cached_input_tokens=cached)


def _models(tel) -> dict:
    return {m["model"]: m for m in tel.usage_rollup()["models"]}


def test_a_model_with_no_recorded_cache_count_reports_null_not_zero(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    _round(tel, "qwen3.8-max")
    _round(tel, "qwen3.8-max")
    m = _models(tel)["qwen3.8-max"]
    assert m["cached_input_tokens"] is None, "nothing was recorded: that is not '0 cached'"
    assert m["cache_reported_runs"] == 0
    assert m["runs"] == 2


def test_a_recorded_zero_stays_zero(tmp_path):
    """A cold cache is a finding, and must not turn into 'not recorded'."""
    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    _round(tel, "claude-haiku-4-5", cached=0)
    m = _models(tel)["claude-haiku-4-5"]
    assert (m["cached_input_tokens"], m["cache_reported_runs"]) == (0, 1)


def test_a_mixed_model_sums_what_was_recorded_and_says_how_much(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    _round(tel, "Qwen3.8-27B.gguf", cached=900)
    _round(tel, "Qwen3.8-27B.gguf")  # a round written before cache counts were kept
    m = _models(tel)["Qwen3.8-27B.gguf"]
    assert (m["cached_input_tokens"], m["cache_reported_runs"], m["runs"]) == (900, 1, 2)


def _get_usage(tmp_path, monkeypatch, seed) -> dict:
    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    seed(tel)
    monkeypatch.setattr(tracker, "_telemetry_singleton", tel)
    body = TestClient(create_app({})).get("/api/usage").json()
    return body


def test_the_endpoint_reports_null_per_model_and_in_the_totals(tmp_path, monkeypatch):
    def seed(tel):
        _round(tel, "qwen3.8-max")
        _round(tel, "grok-4.5")

    body = _get_usage(tmp_path, monkeypatch, seed)
    assert {m["model"]: m["cached_input_tokens"] for m in body["models"]} == {
        "qwen3.8-max": None, "grok-4.5": None}
    assert body["totals"]["cached_input_tokens"] is None
    assert body["totals"]["cache_reported_runs"] == 0
    assert body["totals"]["runs"] == 2


def test_the_totals_sum_only_what_was_recorded(tmp_path, monkeypatch):
    def seed(tel):
        _round(tel, "qwen3.8-max")                       # unrecorded
        _round(tel, "Qwen3.8-27B.gguf", cached=900)
        _round(tel, "claude-haiku-4-5", cached=0)        # recorded cold

    body = _get_usage(tmp_path, monkeypatch, seed)
    assert body["totals"]["cached_input_tokens"] == 900
    assert body["totals"]["cache_reported_runs"] == 2
    assert body["totals"]["runs"] == 3
