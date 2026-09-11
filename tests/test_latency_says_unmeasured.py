"""Latency must be able to say "not measured", and the digest must show it.

THE DEFECT
----------
`tool_calls.latency_ms` was `REAL NOT NULL DEFAULT 0.0`, so "nobody measured
this" and "this took zero milliseconds" were the same stored value. Only the
real execution path times anything; permission denials, validation failures
and repeat-guard blocks record calls that never ran and took the default.
`AVG(latency_ms)` then mixed never-ran rows in as zero-duration executions.

WHAT IS TESTED HERE, AND WHY IT IS THE OUTPUT AND NOT THE TAG
--------------------------------------------------------------
The guard that catches the failure mode is not "the code sets a source". It is
that **the string a human reads differs depending on which rows fed it**. A tag
that lives in a dict and never reaches the page is a tag nobody can act on, and
the whole defect class here is a surface reporting a state it never measured.

`test_the_three_renderings_are_pairwise_distinct` is that test.

NO BACKFILL — AND THAT IS WHY `unknown` EXISTS
-----------------------------------------------
Rows written before schema v2 are left exactly as they are. Rewriting a stored
`0.0` to NULL would assert that an exact zero always meant "unmeasured" — true
of the code as it stood, and precisely the inference that rots. So aggregates
containing a pre-v2 row report `unknown` rather than a number that looks
measured.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.sentinel.telemetry_digest import TelemetryDigest  # noqa: E402
from prometheus.telemetry.latency import (  # noqa: E402
    LATENCY_MEASURED,
    LATENCY_UNKNOWN,
    LATENCY_UNMEASURED,
    LatencyAggregate,
    render_latency,
)
from prometheus.telemetry.tracker import (  # noqa: E402
    TELEMETRY_SCHEMA_VERSION,
    ToolCallTelemetry,
)


# ═══════════════════════════════════════════════════════════════════════════
# THE GUARD: three provenances, three distinct things a human reads
# ═══════════════════════════════════════════════════════════════════════════

def _aggregate(*, measured=(), unmeasured=0, pre_v2=()):
    agg = LatencyAggregate()
    for value in measured:
        agg.add(value, pre_v2=False)
    for _ in range(unmeasured):
        agg.add(None, pre_v2=False)
    for value in pre_v2:
        agg.add(value, pre_v2=True)
    return agg


def test_the_three_renderings_are_pairwise_distinct():
    """THE test. Not "the tag is set" — that the OUTPUT differs."""
    measured = render_latency(*_aggregate(measured=(10.0, 20.0)).resolve())
    unmeasured = render_latency(*_aggregate(unmeasured=3).resolve())
    unknown = render_latency(*_aggregate(measured=(20.0,), pre_v2=(10.0,)).resolve())

    rendered = {"measured": measured, "unmeasured": unmeasured, "unknown": unknown}
    assert len(set(rendered.values())) == 3, (
        "two provenances render identically, so a reader cannot tell them "
        f"apart: {rendered}"
    )
    # And each says something a reader can act on.
    assert measured == "15.0ms"
    assert "not measured" in unmeasured
    assert "unverified" in unknown and "schema v2" in unknown


def test_unmeasured_does_not_render_as_a_number():
    """The original symptom: a tool with only denials read as '0.0ms'."""
    assert render_latency(*_aggregate(unmeasured=5).resolve()) != "0.0ms"
    assert "0" not in render_latency(*_aggregate(unmeasured=5).resolve())


def test_one_pre_v2_row_makes_the_whole_aggregate_unknown():
    """`unknown` is sticky, deliberately.

    By the time anyone reads the average it has already mixed provenances.
    Reporting a clean number for a window that is part guesswork is the
    failure this exists to prevent.
    """
    value, source = _aggregate(measured=(10.0,) * 99, pre_v2=(0.0,)).resolve()
    assert source == LATENCY_UNKNOWN, (
        "99 measured rows and 1 pre-v2 row reported as fully measured"
    )


def test_measured_and_unmeasured_are_not_the_same_value():
    """Zero is a measurement; absence is not."""
    zero_measured, src_zero = _aggregate(measured=(0.0,)).resolve()
    absent, src_absent = _aggregate(unmeasured=1).resolve()

    assert (zero_measured, src_zero) == (0.0, LATENCY_MEASURED)
    assert (absent, src_absent) == (None, LATENCY_UNMEASURED)
    assert render_latency(zero_measured, src_zero) != render_latency(absent, src_absent)


# ═══════════════════════════════════════════════════════════════════════════
# THE DIGEST renders it — end to end, through the real formatter
# ═══════════════════════════════════════════════════════════════════════════

def _report(**tools):
    return {"total_calls": 10, "overall_success_rate": 1.0, "tools": tools}


def test_the_digest_renders_the_three_cases_distinctly():
    """Through `_format_summary`, the string an operator actually sees."""
    summary = TelemetryDigest._format_summary(
        _report(
            timed={"avg_latency_ms": 15.0, "avg_latency_source": LATENCY_MEASURED},
            denied={"avg_latency_ms": None, "avg_latency_source": LATENCY_UNMEASURED},
            historic={"avg_latency_ms": 15.0, "avg_latency_source": LATENCY_UNKNOWN},
        ),
        [],
    )
    lines = {
        line.split(":")[0].strip(): line.split(":", 1)[1].strip()
        for line in summary.splitlines()
        if line.startswith("    ") and ":" in line
    }
    assert set(lines) == {"timed", "denied", "historic"}, summary
    assert len(set(lines.values())) == 3, (
        f"the digest renders two provenances identically:\n{summary}"
    )
    assert lines["denied"] != lines["timed"]
    assert lines["historic"] != lines["timed"], (
        "a pre-v2 average is presented as though it were measured"
    )


def test_a_latency_spike_is_not_raised_across_a_provenance_change():
    """An alert manufactured out of a provenance difference is a false alarm.

    A baseline window whose average includes never-ran rows sits artificially
    low. Comparing a measured current window against it produces a "spike" that
    describes the schema change, not the system.
    """
    current = {"tool": {"calls": 10, "success_rate": 1.0, "avg_retries": 0.0,
                        "avg_latency_ms": 100.0,
                        "avg_latency_source": LATENCY_MEASURED}}
    baseline = {"tool": {"calls": 10, "success_rate": 1.0, "avg_retries": 0.0,
                         "avg_latency_ms": 10.0,
                         "avg_latency_source": LATENCY_UNKNOWN}}

    # _compare is an instance method that does not touch self; calling it
    # unbound keeps the test on the real comparison logic without standing up
    # a whole digest and its DB.
    anomalies = TelemetryDigest._compare(
        TelemetryDigest.__new__(TelemetryDigest),
        {"tools": current}, {"tools": baseline},
    )
    assert not [a for a in anomalies if a.metric == "latency_spike"], (
        "a 10x 'spike' was raised comparing measured rows against a window of "
        "unknown provenance"
    )


def test_a_real_spike_between_two_measured_windows_still_fires():
    """The guard must not silence genuine regressions."""
    current = {"tool": {"calls": 10, "success_rate": 1.0, "avg_retries": 0.0,
                        "avg_latency_ms": 100.0,
                        "avg_latency_source": LATENCY_MEASURED}}
    baseline = {"tool": {"calls": 10, "success_rate": 1.0, "avg_retries": 0.0,
                         "avg_latency_ms": 10.0,
                         "avg_latency_source": LATENCY_MEASURED}}

    # _compare is an instance method that does not touch self; calling it
    # unbound keeps the test on the real comparison logic without standing up
    # a whole digest and its DB.
    anomalies = TelemetryDigest._compare(
        TelemetryDigest.__new__(TelemetryDigest),
        {"tools": current}, {"tools": baseline},
    )
    assert [a for a in anomalies if a.metric == "latency_spike"], (
        "a genuine 10x latency regression was suppressed"
    )


# ═══════════════════════════════════════════════════════════════════════════
# THE MIGRATION
# ═══════════════════════════════════════════════════════════════════════════

def _v1_db(path: Path) -> None:
    """A pre-v2 telemetry DB, with the NOT NULL column and rows in it."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE tool_calls (
            id TEXT PRIMARY KEY, timestamp REAL NOT NULL, model TEXT NOT NULL,
            tool_name TEXT NOT NULL, success INTEGER NOT NULL,
            retries INTEGER NOT NULL DEFAULT 0,
            latency_ms REAL NOT NULL DEFAULT 0.0,
            error_type TEXT, error_detail TEXT
        );
        INSERT INTO schema_meta VALUES ('schema_version','1');
        """
    )
    conn.execute(
        "INSERT INTO tool_calls (id,timestamp,model,tool_name,success,latency_ms,error_type)"
        " VALUES ('a',1.0,'m','bash',1,0.0,'permission_denied')"
    )
    conn.execute(
        "INSERT INTO tool_calls (id,timestamp,model,tool_name,success,latency_ms)"
        " VALUES ('b',2.0,'m','bash',1,7.5)"
    )
    conn.commit()
    conn.close()


def test_the_migration_relaxes_the_column_and_keeps_every_row(tmp_path):
    db = tmp_path / "telemetry.db"
    _v1_db(db)

    telemetry = ToolCallTelemetry(db_path=db)

    info = telemetry._conn.execute("PRAGMA table_info(tool_calls)").fetchall()
    latency = next(r for r in info if r[1] == "latency_ms")
    assert latency[3] == 0, "latency_ms is still NOT NULL after the migration"
    assert TELEMETRY_SCHEMA_VERSION == 2

    rows = dict(
        telemetry._conn.execute("SELECT id, latency_ms FROM tool_calls").fetchall()
    )
    assert rows == {"a": 0.0, "b": 7.5}, (
        f"the migration changed existing rows: {rows}. Pre-v2 rows must be "
        f"copied verbatim — backfilling 0.0 to NULL encodes a rotting inference."
    )


def test_the_migration_is_idempotent(tmp_path):
    db = tmp_path / "telemetry.db"
    _v1_db(db)
    first = ToolCallTelemetry(db_path=db)
    boundary = first.latency_boundary()
    first._conn.close()

    second = ToolCallTelemetry(db_path=db)
    assert second.latency_boundary() == boundary, (
        "the boundary moved on a second open — pre-v2 rows would drift into "
        "looking post-v2"
    )
    rows = dict(
        second._conn.execute("SELECT id, latency_ms FROM tool_calls").fetchall()
    )
    assert rows == {"a": 0.0, "b": 7.5}


def test_an_unmeasured_call_is_stored_as_null(tmp_path):
    telemetry = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    telemetry.record(model="m", tool_name="bash", success=True, latency_ms=12.5)
    telemetry.record(model="m", tool_name="bash", success=False,
                     error_type="permission_denied")

    stored = [
        r[0] for r in telemetry._conn.execute(
            "SELECT latency_ms FROM tool_calls ORDER BY timestamp"
        ).fetchall()
    ]
    assert stored == [12.5, None], (
        f"an unmeasured call was not stored as NULL: {stored}"
    )


def test_rows_written_after_the_boundary_are_not_tagged_unknown(tmp_path):
    """The boundary must actually separate the two eras.

    If everything came back `unknown`, the tag would be useless noise and the
    distinct-rendering test above would be measuring nothing real.
    """
    telemetry = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    telemetry.record(model="m", tool_name="bash", success=True, latency_ms=5.0)
    telemetry.record(model="m", tool_name="bash", success=True, latency_ms=15.0)

    report = telemetry.report()
    td = report["tools"]["bash"]
    assert td["avg_latency_source"] == LATENCY_MEASURED, td
    assert td["avg_latency_ms"] == pytest.approx(10.0)


def test_a_fresh_database_has_no_pre_v2_era(tmp_path):
    """A DB created at v2 must not classify its own backdated rows as unknown.

    ⚠ THIS IS A DEFECT THE FIX ITSELF INTRODUCED, and an existing test caught
    it. The first version stamped `latency_nullable_since = now` for every
    database, including one that had never had the NOT NULL column. Any row
    written afterwards with an older timestamp — a backfill, an import, or any
    test inserting backdated rows — then compared as "before the boundary" and
    the whole aggregate reported `unknown`.

    `tests/test_sentinel.py::test_detects_latency_spike` inserts rows dated
    three days back and went red: the spike was suppressed because its own
    freshly-created database had declared all of its data unverifiable.

    A database that has never had the NOT NULL column has NO pre-v2 era, so
    the boundary is 0.0 rather than `now`. Only an actual rebuild stamps the
    current instant.
    """
    telemetry = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    assert telemetry.latency_boundary() == 0.0, (
        "a fresh v2 database stamped a non-zero boundary — rows older than it "
        "will be misreported as pre-v2"
    )

    long_ago = time.time() - 3 * 86400
    for value in (100.0, 200.0):
        telemetry._conn.execute(
            "INSERT INTO tool_calls (id, timestamp, model, tool_name, success,"
            " retries, latency_ms) VALUES (hex(randomblob(16)), ?, 'm', 'bash',"
            " 1, 0, ?)",
            (long_ago, value),
        )
    telemetry._conn.commit()

    td = telemetry.report()["tools"]["bash"]
    assert td["avg_latency_source"] == LATENCY_MEASURED, (
        f"backdated rows in a fresh v2 DB were tagged {td['avg_latency_source']!r}"
    )
    assert td["avg_latency_ms"] == pytest.approx(150.0)


def test_a_migrated_database_does_stamp_the_migration_instant(tmp_path):
    """The other half: a REAL rebuild must mark its pre-existing rows.

    Without this, the fix above ("fresh DBs get 0.0") could be implemented as
    "always 0.0", which would silently claim every pre-v2 row was measured —
    the original defect, reinstated.
    """
    db = tmp_path / "telemetry.db"
    _v1_db(db)
    telemetry = ToolCallTelemetry(db_path=db)

    boundary = telemetry.latency_boundary()
    assert boundary is not None and boundary > 0.0, (
        "a migrated database did not record when NULL became expressible, so "
        "its pre-v2 rows are indistinguishable from post-v2 ones"
    )

    td = telemetry.report()["tools"]["bash"]
    assert td["avg_latency_source"] == LATENCY_UNKNOWN, (
        "rows that predate the rebuild were reported as measured"
    )
