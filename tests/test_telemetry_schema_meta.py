"""Telemetry schema versioning + node attribution — FOUNDATION 1.3.

The first real schema version telemetry.db has ever had. The policy under
test: a DB written by a NEWER schema refuses before being touched (the
rollback case the additive column migrator silently tolerated); a legacy DB
adopts silently (machine-owned, single writer — bookkeeping, not policy);
and every trace row names the node that produced it.

Also pinned: the default DB path resolves through get_config_dir(), so the
writer and `prometheus reset-data` (the eraser) agree under a custom
PROMETHEUS_CONFIG_DIR. It did not — the writer hardcoded the default.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from prometheus.config.node_identity import ensure_node_identity
from prometheus.config.paths import get_config_dir
from prometheus.telemetry.tracker import (
    TELEMETRY_SCHEMA_VERSION,
    TelemetrySchemaError,
    ToolCallTelemetry,
)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "telemetry.db"


class TestSchemaMeta:
    def test_fresh_db_is_stamped_current(self, db_path: Path) -> None:
        ToolCallTelemetry(db_path)
        conn = sqlite3.connect(db_path)
        meta = dict(conn.execute("SELECT key, value FROM schema_meta"))
        conn.close()
        assert meta["schema_version"] == str(TELEMETRY_SCHEMA_VERSION)
        assert "created_at" in meta
        assert "created_by" in meta

    def test_newer_schema_refuses_naming_both_versions(self, db_path: Path) -> None:
        ToolCallTelemetry(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
            (str(TELEMETRY_SCHEMA_VERSION + 41),),
        )
        conn.commit()
        conn.close()
        with pytest.raises(TelemetrySchemaError) as exc:
            ToolCallTelemetry(db_path)
        assert str(TELEMETRY_SCHEMA_VERSION + 41) in str(exc.value)
        assert str(TELEMETRY_SCHEMA_VERSION) in str(exc.value)

    def test_refusal_leaves_the_newer_db_untouched(self, db_path: Path) -> None:
        # Refuse-before-touch: no table creation, no ALTERs, no stamp
        # rewrite against a DB from the future.
        ToolCallTelemetry(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE schema_meta SET value = '99' WHERE key = 'schema_version'"
        )
        conn.commit()
        conn.close()
        before = db_path.read_bytes()
        with pytest.raises(TelemetrySchemaError):
            ToolCallTelemetry(db_path)
        assert db_path.read_bytes() == before

    def test_legacy_db_adopts_and_gains_node_id_columns(self, db_path: Path) -> None:
        # A DB from before versioning: has the tables, has no schema_meta.
        ToolCallTelemetry(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE schema_meta")
        conn.commit()
        conn.close()

        ToolCallTelemetry(db_path)  # re-open: adopt, don't refuse
        conn = sqlite3.connect(db_path)
        version = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        tool_cols = {r[1] for r in conn.execute("PRAGMA table_info(tool_calls)")}
        run_cols = {r[1] for r in conn.execute("PRAGMA table_info(subsystem_runs)")}
        conn.close()
        assert version == str(TELEMETRY_SCHEMA_VERSION)
        assert "node_id" in tool_cols
        assert "node_id" in run_cols


class TestNodeAttribution:
    def _row(self, db_path: Path, table: str) -> dict:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(f"SELECT * FROM {table}").fetchone()
        conn.close()
        return dict(row)

    def test_tool_call_rows_carry_the_node_id(self, db_path: Path) -> None:
        identity = ensure_node_identity()
        tel = ToolCallTelemetry(db_path)
        tel.record(model="m", tool_name="bash", success=True)
        assert self._row(db_path, "tool_calls")["node_id"] == identity.pubkey

    def test_subsystem_run_rows_carry_the_node_id(self, db_path: Path) -> None:
        identity = ensure_node_identity()
        tel = ToolCallTelemetry(db_path)
        tel.record_run(subsystem="curator", operation="pass", outcome="success")
        assert self._row(db_path, "subsystem_runs")["node_id"] == identity.pubkey

    def test_rows_before_identity_are_null_then_pick_it_up(
        self, db_path: Path
    ) -> None:
        # Lazy resolution: telemetry never mints identity, and construction
        # order is not guaranteed — the CLI builds telemetry before the
        # entry point's first-run mint. NULL honestly until the identity
        # exists, stamped from the first write after it does.
        tel = ToolCallTelemetry(db_path)
        tel.record(model="m", tool_name="bash", success=True)
        assert self._row(db_path, "tool_calls")["node_id"] is None

        identity = ensure_node_identity()
        tel.record(model="m", tool_name="grep", success=True)
        conn = sqlite3.connect(db_path)
        stamped = conn.execute(
            "SELECT node_id FROM tool_calls WHERE tool_name = 'grep'"
        ).fetchone()[0]
        conn.close()
        assert stamped == identity.pubkey


class TestDbPathResolution:
    def test_bare_construction_lands_in_config_dir(self) -> None:
        # conftest points PROMETHEUS_CONFIG_DIR at tmp_path; a bare
        # ToolCallTelemetry() must land there — the same place reset-data
        # erases — not at a hardcoded ~/.prometheus.
        tel = ToolCallTelemetry()
        expected = (get_config_dir() / "telemetry.db").resolve()
        assert tel._db_path == expected
        assert expected.exists()
