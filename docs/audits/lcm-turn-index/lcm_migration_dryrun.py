"""Dry run of the REAL turn-index migration against a copy of a nightly snapshot.

snapshot (read-only, immutable) --SQLite backup API--> :memory:, then
``run_turn_index_migration`` on that copy with its own backup kept in memory too.
Nothing is written to disk, and only aggregates are printed.

Run on the production host from a checkout of the branch under test:
    PYTHONPATH=<checkout>/src python3 lcm_migration_dryrun.py <snapshot>/data/lcm.db
"""

import json
import sqlite3
import sys
import time

from prometheus.memory.lcm_conversation_store import (
    TURN_INDEX_GUARD_TRIGGER,
    TURN_INDEX_INDEX,
)
from prometheus.memory.lcm_turn_index_migration import run_turn_index_migration


def duplicate_stats(conn: sqlite3.Connection) -> dict[str, int]:
    groups = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(n), 0), COALESCE(SUM(n - 1), 0), COUNT(DISTINCT sid)"
        " FROM (SELECT session_id AS sid, COUNT(*) AS n FROM lcm_messages"
        "       GROUP BY session_id, turn_index HAVING COUNT(*) > 1)"
    ).fetchone()
    return {"duplicated_pairs": groups[0], "rows_in_pairs": groups[1],
            "colliding_rows": groups[2], "affected_sessions": groups[3]}


def tables(conn: sqlite3.Connection) -> dict[str, list[str]]:
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
    return {n: [c[1] for c in conn.execute(f'PRAGMA table_info("{n}")')] for n in names}


out: dict[str, object] = {"sqlite": sqlite3.sqlite_version}
src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro&immutable=1", uri=True)
mem = sqlite3.connect(":memory:", isolation_level=None)
t0 = time.perf_counter()
src.backup(mem)
out["copy_to_memory_s"] = round(time.perf_counter() - t0, 2)
src.close()

pages = mem.execute("PRAGMA page_count").fetchone()[0]
page_size = mem.execute("PRAGMA page_size").fetchone()[0]
out["db_bytes"] = pages * page_size
out["rows"] = mem.execute("SELECT COUNT(*) FROM lcm_messages").fetchone()[0]
out["sessions"] = mem.execute("SELECT COUNT(DISTINCT session_id) FROM lcm_messages").fetchone()[0]
out["before"] = duplicate_stats(mem)
out["user_version_before"] = mem.execute("PRAGMA user_version").fetchone()[0]
schema_before = tables(mem)

result = run_turn_index_migration(mem, backup_path=":memory:")
out["result"] = {k: v for k, v in vars(result).items() if k != "backup_path"}

out["after"] = duplicate_stats(mem)
out["rows_after"] = mem.execute("SELECT COUNT(*) FROM lcm_messages").fetchone()[0]
out["user_version_after"] = mem.execute("PRAGMA user_version").fetchone()[0]
out["turn_index_indexes_after(name: unique)"] = {
    r[1]: bool(r[2]) for r in mem.execute("PRAGMA index_list(lcm_messages)")
    if r[1] == TURN_INDEX_INDEX or "turn" in r[1]
}
out["guard_trigger_after"] = mem.execute(
    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' AND name = ?",
    (TURN_INDEX_GUARD_TRIGGER,),
).fetchone()[0] == 1
out["tables_and_columns_unchanged"] = tables(mem) == schema_before
out["temp_tables_left"] = mem.execute("SELECT COUNT(*) FROM sqlite_temp_master").fetchone()[0]
try:
    mem.execute("INSERT INTO lcm_messages_fts(lcm_messages_fts, rank) VALUES('integrity-check', 1)")
    out["fts_integrity"] = "ok"
except sqlite3.Error as exc:
    out["fts_integrity"] = f"error: {exc}"
t0 = time.perf_counter()
out["integrity_check"] = mem.execute("PRAGMA integrity_check").fetchone()[0]
out["integrity_check_s"] = round(time.perf_counter() - t0, 2)

# An old build's insert at a key a real row holds: refused, and nothing deleted.
sid, ti = mem.execute("SELECT session_id, turn_index FROM lcm_messages LIMIT 1").fetchone()
before = mem.execute("SELECT COUNT(*) FROM lcm_messages").fetchone()[0]
try:
    mem.execute(
        "INSERT OR REPLACE INTO lcm_messages (id, session_id, turn_index, role, content,"
        " content_json, token_count, timestamp, compacted, provenance, is_trusted)"
        " VALUES ('dry-run-old-build', ?, ?, 'user', '', NULL, 0, 0, 0, 'user', 1)",
        (sid, ti),
    )
    out["old_build_insert_or_replace"] = "ACCEPTED (unexpected)"
except sqlite3.IntegrityError as exc:
    out["old_build_insert_or_replace"] = f"refused: {type(exc).__name__}"
out["old_build_rows_unchanged"] = (
    mem.execute("SELECT COUNT(*) FROM lcm_messages").fetchone()[0] == before
)

rerun = run_turn_index_migration(mem, backup_path=":memory:")
out["second_run_status"] = rerun.status
mem.close()
print(json.dumps(out, indent=1))
