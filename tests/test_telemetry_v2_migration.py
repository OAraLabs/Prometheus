"""Telemetry v2 (WP-X.54 T-1): the migration round-trips on a real database.

The audit's rule for this schema change is "additive only": new tables, new
nullable columns, one new ``schema_meta`` key, and nothing else moves. A
rolled-back binary must still open the file, and an operator must be able to
remove the additions by hand and get the original database back. So the test
does exactly that, on a database that predates the change:

1. snapshot every table (columns, row count, max rowid, a digest of every row);
2. open it with the current code (``ToolCallTelemetry`` / ``PairStore``);
3. check that what changed is exactly the expected additions, that every
   pre-existing row is byte-identical, that the rowids (the golden exporter's
   cursor) did not move, and that ``schema_version`` did not move;
4. open it again: nothing changes the second time (the boundary is set once);
5. drop the additions, and check the result equals the original snapshot.

Two sources. ``tests/fixtures/telemetry_v2/pre_v2_*.sql`` are dumps of small
databases written by the code BEFORE this change (generated from origin/main at
6e85d5a, ids replaced by fixed ones), so the hermetic run needs no live data.
``PROMETHEUS_TELEMETRY_V2_COPIES=<dir>`` points the same test at copies of the
live ``telemetry.db`` and ``training.db`` (taken with the sqlite backup API).
The copies are copied again into tmp_path first, so the directory is never
written.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

import pytest

from prometheus.learning.pair_capture import PairStore
from prometheus.telemetry.tracker import TELEMETRY_SCHEMA_VERSION, ToolCallTelemetry

FIXTURES = Path(__file__).parent / "fixtures" / "telemetry_v2"
COPIES_ENV = "PROMETHEUS_TELEMETRY_V2_COPIES"

# What T-1 adds. Spelled out here, not derived from the code, so an extra
# column or table nobody asked for is a failure rather than a silent addition.
EXPECTED_NEW_TABLES = {
    "telemetry.db": {"turns", "responses", "tool_sets"},
    "training.db": set(),
}
EXPECTED_NEW_COLUMNS = {
    "telemetry.db": {
        "tool_calls": [
            "turn_id", "round_index", "tool_use_id", "repair_kind",
            "raw_before_repair", "retry_index", "result_summary",
        ],
    },
    "training.db": {
        "training_pairs": ["turn_id", "round_index", "repair_kind", "outcome"],
    },
}
EXPECTED_NEW_META_KEYS = {"telemetry.db": {"telemetry_v2_since"}, "training.db": set()}


def _tables(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def _snapshot(path: Path) -> dict:
    """Everything a reader could notice about the file, per table."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        out: dict = {"tables": {}, "indexes": {}, "meta": {}}
        for table in _tables(conn):
            info = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            cols = [r[1] for r in info]
            # schema_meta is a key/value table read by key. Every open
            # INSERT OR REPLACEs schema_version (FOUNDATION 1.3, before this
            # change), which gives that row a new rowid each time, so its
            # rowids are not part of what a reader can notice.
            keyed = table == "schema_meta"
            order = "key" if keyed else "rowid"
            digest = hashlib.sha256()
            count = 0
            for row in conn.execute(
                    f'SELECT {", ".join(chr(34) + c + chr(34) for c in cols)} '
                    f'FROM "{table}" ORDER BY {order}'):
                digest.update(repr(row).encode())
                count += 1
            max_rowid = None if keyed else conn.execute(
                f'SELECT MAX(rowid) FROM "{table}"').fetchone()[0]
            out["tables"][table] = {
                # (name, declared type, notnull, default, pk) — everything but cid
                "columns": [tuple(r[1:]) for r in info],
                "count": count,
                "max_rowid": max_rowid,
                "rows": digest.hexdigest(),
            }
        out["indexes"] = dict(conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"))
        if "schema_meta" in out["tables"]:
            out["meta"] = dict(conn.execute("SELECT key, value FROM schema_meta"))
        return out
    finally:
        conn.close()


def _project(path: Path, table: str, cols: list[str]) -> tuple[int, str]:
    """Row count and digest of ``cols`` only — the pre-existing columns."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        digest = hashlib.sha256()
        n = 0
        for row in conn.execute(
                f'SELECT {", ".join(chr(34) + c + chr(34) for c in cols)} '
                f'FROM "{table}" ORDER BY rowid'):
            digest.update(repr(row).encode())
            n += 1
        return n, digest.hexdigest()
    finally:
        conn.close()


def _open_with_current_code(name: str, path: Path) -> None:
    if name == "telemetry.db":
        ToolCallTelemetry(path).close()
    else:
        PairStore(path).close()


def _roll_back(path: Path, before: dict, after: dict) -> None:
    """Remove exactly what the migration added — the documented rollback."""
    conn = sqlite3.connect(str(path))
    try:
        new_tables = set(after["tables"]) - set(before["tables"])
        for index in set(after["indexes"]) - set(before["indexes"]):
            conn.execute(f'DROP INDEX "{index}"')
        for table in new_tables:
            conn.execute(f'DROP TABLE "{table}"')
        for table, info in before["tables"].items():
            old = {c[0] for c in info["columns"]}
            for col in after["tables"][table]["columns"]:
                if col[0] not in old:
                    conn.execute(f'ALTER TABLE "{table}" DROP COLUMN "{col[0]}"')
        for key in set(after["meta"]) - set(before["meta"]):
            conn.execute("DELETE FROM schema_meta WHERE key = ?", (key,))
        conn.commit()
    finally:
        conn.close()


def _check_round_trip(name: str, path: Path) -> None:
    before = _snapshot(path)

    _open_with_current_code(name, path)
    after = _snapshot(path)

    # Exactly the expected new tables, all empty — T-1 writes nothing to them.
    new_tables = set(after["tables"]) - set(before["tables"])
    assert new_tables == EXPECTED_NEW_TABLES[name]
    for table in new_tables:
        assert after["tables"][table]["count"] == 0, table
    assert set(before["tables"]) <= set(after["tables"])  # nothing dropped

    for table, old in before["tables"].items():
        new = after["tables"][table]
        old_names = [c[0] for c in old["columns"]]
        # Old columns unchanged and in place; additions only at the end.
        assert new["columns"][: len(old["columns"])] == old["columns"], table
        added = [c for c in new["columns"][len(old["columns"]):]]
        assert [c[0] for c in added] == EXPECTED_NEW_COLUMNS[name].get(table, []), table
        for col in added:
            assert col[2] == 0, f"{table}.{col[0]} must be nullable"
            assert col[3] is None, f"{table}.{col[0]} must default to NULL"
        # Every pre-existing value byte-identical, and the rowids (the golden
        # exporter's cursor) where they were. schema_meta gains exactly the
        # boundary row; it is compared key by key below.
        if table != "schema_meta":
            assert _project(path, table, old_names) == (old["count"], old["rows"]), table
            assert new["max_rowid"] == old["max_rowid"], table
        # The added columns hold NULL on every old row: no backfill.
        if added:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                for col in added:
                    n = conn.execute(
                        f'SELECT COUNT(*) FROM "{table}" WHERE "{col[0]}" IS NOT NULL'
                    ).fetchone()[0]
                    assert n == 0, f"{table}.{col[0]} was backfilled"
            finally:
                conn.close()

    # Indexes that existed are untouched.
    for index, sql in before["indexes"].items():
        assert after["indexes"].get(index) == sql, index

    # schema_meta: no version bump (a rolled-back v3 binary must still open
    # the file); every old key unchanged; one new boundary key.
    if before["meta"]:
        assert after["meta"]["schema_version"] == before["meta"]["schema_version"] == str(
            TELEMETRY_SCHEMA_VERSION)
        for key, value in before["meta"].items():
            assert after["meta"][key] == value, key
    assert set(after["meta"]) - set(before["meta"]) == EXPECTED_NEW_META_KEYS[name]
    if name == "telemetry.db":
        float(after["meta"]["telemetry_v2_since"])  # a number, not a label

    # Idempotent: a second open changes nothing, the boundary included.
    _open_with_current_code(name, path)
    assert _snapshot(path) == after

    # Rollback: dropping the additions gives back the original database.
    _roll_back(path, before, after)
    assert _snapshot(path) == before


def _load_fixture(name: str, dest: Path) -> Path:
    path = dest / name
    conn = sqlite3.connect(str(path))
    conn.executescript((FIXTURES / f"pre_v2_{name.split('.')[0]}.sql").read_text())
    conn.close()
    return path


@pytest.mark.parametrize("name", ["telemetry.db", "training.db"])
def test_the_migration_round_trips_on_a_database_from_before_it(name, tmp_path):
    _check_round_trip(name, _load_fixture(name, tmp_path))


@pytest.mark.parametrize("name", ["telemetry.db", "training.db"])
def test_the_migration_round_trips_on_a_copy_of_the_live_database(name, tmp_path):
    src_dir = os.environ.get(COPIES_ENV)
    if not src_dir:
        pytest.skip(f"set {COPIES_ENV} to a directory holding copies of the live databases")
    src = Path(src_dir) / name
    if not src.exists():
        pytest.skip(f"{src} not found")
    dest = tmp_path / name
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(str(dest))
    with d:
        s.backup(d)
    s.close()
    d.close()
    _check_round_trip(name, dest)


def test_a_fresh_database_gets_the_whole_v2_schema(tmp_path):
    ToolCallTelemetry(tmp_path / "telemetry.db").close()
    PairStore(tmp_path / "training.db").close()
    tel = _snapshot(tmp_path / "telemetry.db")
    assert EXPECTED_NEW_TABLES["telemetry.db"] <= set(tel["tables"])
    cols = [c[0] for c in tel["tables"]["tool_calls"]["columns"]]
    assert set(EXPECTED_NEW_COLUMNS["telemetry.db"]["tool_calls"]) <= set(cols)
    assert "telemetry_v2_since" in tel["meta"]
    pairs = [c[0] for c in _snapshot(tmp_path / "training.db")["tables"]["training_pairs"]["columns"]]
    assert set(EXPECTED_NEW_COLUMNS["training.db"]["training_pairs"]) <= set(pairs)


def test_the_boundary_is_set_once_and_never_moves(tmp_path):
    path = _load_fixture("telemetry.db", tmp_path)
    ToolCallTelemetry(path).close()
    first = _snapshot(path)["meta"]["telemetry_v2_since"]
    ToolCallTelemetry(path).close()
    assert _snapshot(path)["meta"]["telemetry_v2_since"] == first


def test_the_new_pair_fields_are_stored_but_stay_out_of_the_dedupe_hash(tmp_path):
    # The miner re-runs with no cursor; dedupe is sha256(context + rejected).
    # A field outside that hash cannot make a re-run double-write.
    store = PairStore(tmp_path / "training.db")
    kw = dict(pair_source="retry_success", model_id="m", tool_name="bash",
              context={"messages": []}, rejected={"name": "bash", "input": {"cmd": "ls"}},
              chosen={"name": "bash", "input": {"command": "ls"}})
    assert store.add_pair(**kw, turn_id="desktop:s:t1", round_index=2,
                          repair_kind="fuzzy_name", outcome="accepted_user") is True
    assert store.add_pair(**kw, turn_id="desktop:s:t2", round_index=5,
                          repair_kind="type_coerce", outcome=None) is False
    row = store.rows_since()[0]
    assert (row["turn_id"], row["round_index"], row["repair_kind"], row["outcome"]) == (
        "desktop:s:t1", 2, "fuzzy_name", "accepted_user")
    plain = PairStore(tmp_path / "training2.db")
    plain.add_pair(**kw)
    assert plain.rows_since()[0]["context_hash"] == row["context_hash"]
    store.close()
    plain.close()
