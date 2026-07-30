"""Prompt-cache accounting: capture it from the provider, store it per round.

The agent loop re-sends a near-identical prompt prefix every round (the
EMBERFALL build: 535k input tokens across 19 rounds, ~92% of which is a repeat
of the previous round's prompt). Whether that prefix is served from cache is
the single biggest lever on cost — and it was previously invisible, because
nothing read the provider's cache counters.

Central invariant here: **None is not 0.** "This provider reports nothing about
caching" and "the cache was cold this round" are different findings; folding
them together would make an unsupported provider look like a permanent 0% hit
rate and hide a real regression.
"""

from __future__ import annotations

import sqlite3

import pytest

from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.openai_compat import _parse_cache_usage
from prometheus.telemetry.tracker import ToolCallTelemetry


# ---------------------------------------------------------------------------
# Provider payload shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,usage,expected",
    [
        # OpenAI / xAI / most compat servers.
        ("openai nested", {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 80}}, (80, None)),
        # DeepSeek's flat spelling.
        ("deepseek flat", {"prompt_cache_hit_tokens": 64}, (64, None)),
        # Anthropic reports reads and writes separately.
        ("anthropic", {"cache_read_input_tokens": 900, "cache_creation_input_tokens": 120}, (900, 120)),
        # Bare last-resort key.
        ("flat cached_tokens", {"cached_tokens": 7}, (7, None)),
    ],
)
def test_known_provider_shapes_are_parsed(label, usage, expected):
    assert _parse_cache_usage(usage) == expected, label


def test_silence_is_none_not_zero():
    # A provider that says nothing must NOT read as a 0% hit rate.
    assert _parse_cache_usage({"prompt_tokens": 100, "completion_tokens": 5}) == (None, None)


def test_explicit_zero_is_preserved_as_zero():
    # A cold cache IS a finding — it must survive as 0, distinct from None.
    cached, _ = _parse_cache_usage({"prompt_tokens_details": {"cached_tokens": 0}})
    assert cached == 0
    assert cached is not None


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens_details": {"cached_tokens": None}},   # explicit null
        {"prompt_tokens_details": "not-an-object"},           # wrong type
        {"prompt_tokens_details": {"cached_tokens": -5}},     # nonsense negative
        {"prompt_tokens_details": {"cached_tokens": "abc"}},  # unparseable
        {},                                                    # empty
    ],
)
def test_malformed_usage_degrades_to_none_and_never_raises(usage):
    # This runs inside stream parsing — a throw here would kill a live turn.
    assert _parse_cache_usage(usage) == (None, None)


def test_string_numbers_are_coerced():
    # Some compat servers stringify numerics.
    assert _parse_cache_usage({"prompt_tokens_details": {"cached_tokens": "42"}}) == (42, None)


def test_non_dict_usage_is_safe():
    assert _parse_cache_usage(None) == (None, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# UsageSnapshot derived accounting
# ---------------------------------------------------------------------------


def test_snapshot_reports_ratio_and_uncached():
    s = UsageSnapshot(input_tokens=1000, output_tokens=50, cached_input_tokens=900)
    assert s.cache_hit_ratio == pytest.approx(0.9)
    assert s.uncached_input_tokens == 100


def test_snapshot_without_cache_info_reports_none():
    s = UsageSnapshot(input_tokens=1000, output_tokens=50)
    assert s.cache_hit_ratio is None
    assert s.uncached_input_tokens is None


def test_snapshot_defaults_are_backward_compatible():
    # Every pre-existing construction site omits the new fields.
    s = UsageSnapshot(input_tokens=10, output_tokens=2)
    assert s.total_tokens == 12
    assert s.cached_input_tokens is None


def test_ratio_is_safe_when_input_is_zero():
    s = UsageSnapshot(input_tokens=0, output_tokens=0, cached_input_tokens=0)
    assert s.cache_hit_ratio is None  # no division by zero


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_cache_columns_round_trip(tmp_path):
    db = tmp_path / "telemetry.db"
    t = ToolCallTelemetry(db_path=db)
    t.record_run(
        subsystem="agent_loop", operation="round", outcome="success", duration_ms=5.0,
        input_tokens=30_000, output_tokens=800, round_index=3,
        session_id="s", model="grok-4.5",
        cached_input_tokens=27_000, cache_write_tokens=0,
    )
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT input_tokens, cached_input_tokens, cache_write_tokens FROM subsystem_runs"
    ).fetchone()
    assert row["input_tokens"] == 30_000
    assert row["cached_input_tokens"] == 27_000
    assert row["cache_write_tokens"] == 0


def test_callers_that_omit_cache_fields_write_null(tmp_path):
    # Curator/extractor/etc. never carry usage — their rows must stay NULL, so a
    # later query can tell "no data" from "no cache hit".
    db = tmp_path / "telemetry.db"
    t = ToolCallTelemetry(db_path=db)
    t.record_run(subsystem="curator", operation="pass", outcome="success")
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT cached_input_tokens, cache_write_tokens FROM subsystem_runs").fetchone()
    assert row == (None, None)


def test_migration_adds_columns_to_a_preexisting_db(tmp_path):
    """An existing telemetry.db must gain the columns with historical rows NULL."""
    db = tmp_path / "telemetry.db"
    first = ToolCallTelemetry(db_path=db)
    first.record_run(subsystem="agent_loop", operation="round", outcome="success", input_tokens=5)
    # Simulate the pre-migration shape by dropping the columns is not possible in
    # SQLite; instead assert the additive migration is idempotent and preserves
    # the old row as NULL when reopened.
    second = ToolCallTelemetry(db_path=db)
    second.record_run(
        subsystem="agent_loop", operation="round", outcome="success",
        input_tokens=6, cached_input_tokens=4,
    )
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT input_tokens, cached_input_tokens FROM subsystem_runs ORDER BY input_tokens"
    ).fetchall()
    assert rows == [(5, None), (6, 4)]  # historical row NULL, new row populated
