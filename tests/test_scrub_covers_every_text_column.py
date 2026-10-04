"""Every text column in a capture store is scrubbed, or says why it is not.

``oara scrub`` (security/scrub_capture_stores.py) rewrites an EXPLICIT list of
(table, column) pairs, and silently skips a listed column that does not exist.
So a new text column is unscrubbed until someone remembers to list it, and a
renamed one drops out without a word. Nothing checked either (audit 2026-09-30,
Q11). This test builds each store with its real constructor, reads the schema
back with ``PRAGMA table_info``, and fails on:

* a text column that is neither in the scrub's targets nor in ``NOT_SCRUBBED``
  below, which says why each one cannot carry a secret;
* a stale entry: a target or an exemption that names a column that no longer
  exists, or an exemption for a column the scrub already covers;
* a full-text index the scrub would not rebuild after rewriting its table.

"Text column" means any column SQLite could store text in: a declared type
with TEXT, CHAR or CLOB in it, or no declared type at all.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from prometheus.coordinator.divergence import CheckpointStore
from prometheus.learning.pair_capture import PairStore
from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_summary_store import LCMSummaryStore
from prometheus.memory.store import MemoryStore
from prometheus.security import scrub_capture_stores as scrub
from prometheus.telemetry.tracker import ToolCallTelemetry


def _build_telemetry(path: Path) -> None:
    ToolCallTelemetry(path).close()


def _build_training(path: Path) -> None:
    PairStore(path).close()


def _build_lcm(path: Path) -> None:
    # Three stores share lcm.db; the daemon builds all three on it.
    for cls in (LCMConversationStore, LCMSummaryStore, CheckpointStore):
        store = cls(path)
        conn = getattr(store, "_conn", None)
        if conn is not None:
            conn.close()


def _build_memory(path: Path) -> None:
    store = MemoryStore(path)
    store._conn.close()


STORES = {
    "telemetry.db": (_build_telemetry, scrub.TELEMETRY_TARGETS, {}),
    "training.db": (_build_training, scrub.TRAINING_TARGETS, {}),
    "lcm.db": (_build_lcm, scrub.LCM_TARGETS, scrub.LCM_FTS),
    "memory.db": (_build_memory, scrub.MEMORY_TARGETS, scrub.MEMORY_FTS),
}

# (store, table, column) -> why this text column cannot carry a secret.
ROW_ID = "row id the store mints (uuid4 hex or a hash), never user input"
SESSION = "session id: a surface prefix plus an id the daemon or client mints"
TURN = "turn id: <session id>:<uuid4 hex>, minted by the loop"
ENUM = "a value from a fixed set the code defines, never free text"
NAME = "a tool, model or subsystem NAME from code or config, not conversation"
ID_LIST = "JSON list of row ids the store minted"
TIME = "a timestamp the code writes"
_n = {
    # telemetry.db
    ("telemetry.db", "tool_calls", "id"): ROW_ID,
    ("telemetry.db", "tool_calls", "model"): NAME,
    ("telemetry.db", "tool_calls", "served_model"): NAME + " (echoed by the model server)",
    ("telemetry.db", "tool_calls", "tool_name"): NAME,
    ("telemetry.db", "tool_calls", "error_type"): ENUM,
    ("telemetry.db", "tool_calls", "session_id"): SESSION,
    ("telemetry.db", "tool_calls", "node_id"): "the node's PUBLIC key",
    ("telemetry.db", "tool_calls", "tool_schema"): (
        "the JSON schema of a tool as offered to the model: from code, config or an MCP "
        "server's tool list, not from the conversation. Capture-time redaction does not "
        "touch it either; a schema that embedded a credential would be a config defect"),
    ("telemetry.db", "tool_calls", "turn_id"): TURN,
    ("telemetry.db", "tool_calls", "tool_use_id"): "tool-call id the model server or daemon mints",
    ("telemetry.db", "tool_calls", "repair_kind"): ENUM,
    ("telemetry.db", "silent_failures", "id"): ROW_ID,
    ("telemetry.db", "silent_failures", "subsystem"): NAME,
    ("telemetry.db", "silent_failures", "operation"): NAME,
    ("telemetry.db", "silent_failures", "exception_type"): "an exception class name",
    ("telemetry.db", "subsystem_runs", "id"): ROW_ID,
    ("telemetry.db", "subsystem_runs", "subsystem"): NAME,
    ("telemetry.db", "subsystem_runs", "operation"): NAME,
    ("telemetry.db", "subsystem_runs", "outcome"): ENUM,
    ("telemetry.db", "subsystem_runs", "session_id"): SESSION,
    ("telemetry.db", "subsystem_runs", "model"): NAME,
    ("telemetry.db", "subsystem_runs", "node_id"): "the node's PUBLIC key",
    ("telemetry.db", "subsystem_runs", "billing_mode"): ENUM,
    ("telemetry.db", "subsystem_runs", "billing_marker"): "a host-marker constant from telemetry/cost.py",
    ("telemetry.db", "signal_events", "timestamp"): TIME,
    ("telemetry.db", "signal_events", "signal_type"): ENUM,
    ("telemetry.db", "signal_events", "source_subsystem"): NAME,
    ("telemetry.db", "signal_events", "read_at"): TIME,
    ("telemetry.db", "circuit_breaker_diagnostics", "id"): ROW_ID,
    ("telemetry.db", "circuit_breaker_diagnostics", "model_id"): NAME,
    ("telemetry.db", "circuit_breaker_diagnostics", "adapter_tier"): ENUM,
    ("telemetry.db", "circuit_breaker_diagnostics", "tool_name"): NAME,
    ("telemetry.db", "circuit_breaker_diagnostics", "failure_category"): ENUM,
    ("telemetry.db", "circuit_breaker_diagnostics", "recovery_method"): ENUM,
    ("telemetry.db", "schema_meta", "key"): "a schema bookkeeping key the code defines",
    ("telemetry.db", "schema_meta", "value"): "a schema version, label or timestamp the code writes",
    # telemetry.db — telemetry v2 (WP-X.54). The free text (responses.prose,
    # tool_calls.raw_before_repair, tool_calls.result_summary) is scrubbed.
    ("telemetry.db", "turns", "turn_id"): TURN,
    ("telemetry.db", "turns", "session_id"): SESSION,
    ("telemetry.db", "turns", "coding_run_id"): "coding-run id the daemon mints",
    ("telemetry.db", "turns", "surface"): ENUM,
    ("telemetry.db", "turns", "mode"): ENUM,
    ("telemetry.db", "turns", "models_used"): "JSON list of model NAMES",
    ("telemetry.db", "turns", "tools_used"): "JSON list of tool NAMES",
    ("telemetry.db", "turns", "terminal_kind"): ENUM,
    ("telemetry.db", "turns", "forced_stop_reason"): ENUM,
    ("telemetry.db", "turns", "task_class"): ENUM,
    ("telemetry.db", "turns", "outcome"): ENUM,
    ("telemetry.db", "turns", "outcome_source"): ENUM,
    ("telemetry.db", "responses", "session_id"): SESSION,
    ("telemetry.db", "responses", "turn_id"): TURN,
    ("telemetry.db", "responses", "loop_round_id"): "subsystem_runs.id, " + ROW_ID,
    ("telemetry.db", "responses", "provider"): NAME,
    ("telemetry.db", "responses", "adapter_tier"): ENUM,
    ("telemetry.db", "responses", "tool_set_hash"): "16 hex of sha256 over a list of tool names",
    ("telemetry.db", "responses", "response_kind"): ENUM,
    ("telemetry.db", "responses", "mode"): ENUM,
    ("telemetry.db", "responses", "surface"): ENUM,
    ("telemetry.db", "responses", "forced_tool_choice"): "the NAME of the tool a round was forced to call",
    ("telemetry.db", "tool_sets", "tool_set_hash"): "16 hex of sha256 over a list of tool names",
    ("telemetry.db", "tool_sets", "tool_names"): "JSON list of tool NAMES (spec 7: names only)",
    # training.db
    ("training.db", "training_pairs", "id"): ROW_ID,
    ("training.db", "training_pairs", "context_hash"): "sha256 of the already-redacted pair",
    ("training.db", "training_pairs", "pair_source"): ENUM,
    ("training.db", "training_pairs", "model_id"): NAME,
    ("training.db", "training_pairs", "tool_name"): NAME,
    ("training.db", "training_pairs", "turn_id"): TURN,
    ("training.db", "training_pairs", "repair_kind"): ENUM,
    ("training.db", "training_pairs", "outcome"): ENUM,
    # lcm.db
    ("lcm.db", "lcm_messages", "id"): ROW_ID,
    ("lcm.db", "lcm_messages", "session_id"): SESSION,
    ("lcm.db", "lcm_messages", "role"): ENUM,
    ("lcm.db", "lcm_messages", "provenance"): ENUM,
    ("lcm.db", "lcm_summaries", "id"): ROW_ID,
    ("lcm.db", "lcm_summaries", "session_id"): SESSION,
    ("lcm.db", "lcm_summaries", "parent_ids"): ID_LIST,
    ("lcm.db", "lcm_summaries", "source_message_ids"): ID_LIST,
    ("lcm.db", "checkpoints", "task_id"): TURN,
    ("lcm.db", "checkpoints", "goal_hash"): "a hash of the goal text",
    ("lcm.db", "message_client_ids", "client_msg_id"): "message id a client mints",
    ("lcm.db", "message_client_ids", "session_id"): SESSION,
    ("lcm.db", "session_backends", "key"): "a picker key from the model catalog",
    ("lcm.db", "session_backends", "session_id"): SESSION,
    ("lcm.db", "session_backends", "set_by"): "the surface that set it (router, rest, ...)",
    ("lcm.db", "session_forks", "origin_session"): SESSION,
    ("lcm.db", "session_forks", "session_id"): SESSION,
    ("lcm.db", "session_pins", "session_id"): SESSION,
    ("lcm.db", "session_profiles", "profile"): "a profile NAME from config",
    ("lcm.db", "session_profiles", "session_id"): SESSION,
    ("lcm.db", "session_titles", "session_id"): SESSION,
    ("lcm.db", "session_tombstones", "session_id"): SESSION,
    ("lcm.db", "session_workspaces", "path"): "a workspace directory path the operator linked",
    ("lcm.db", "session_workspaces", "session_id"): SESSION,
    ("lcm.db", "session_workspaces", "set_by"): "the surface that set it (router, rest, ...)",
    # memory.db
    ("memory.db", "memories", "id"): ROW_ID,
    ("memory.db", "memories", "entity_type"): (
        "a short category label (person, project, ...) the extractor assigns; the fact "
        "itself and the entity name are scrubbed"),
    ("memory.db", "memories", "source_event_ids"): ID_LIST,
    ("memory.db", "messages", "id"): ROW_ID,
    ("memory.db", "messages", "role"): ENUM,
    ("memory.db", "messages", "session_id"): SESSION,
    ("memory.db", "summaries", "id"): ROW_ID,
    ("memory.db", "summaries", "source_message_ids"): ID_LIST,
    ("memory.db", "extractor_cursors", "scope"): "a cursor scope NAME the extractor defines",
    ("memory.db", "store_marks", "name"): (
        "a mark NAME the code defines (decay_charged_through); the value is a timestamp"),
}
NOT_SCRUBBED: dict[tuple[str, str, str], str] = _n


def _is_text(declared: str) -> bool:
    t = (declared or "").upper()
    return not t or any(s in t for s in ("TEXT", "CHAR", "CLOB"))


def _schema(path: Path) -> tuple[dict[str, dict[str, str]], set[str], set[str]]:
    """{table: {column: declared type}}, the FTS5 tables, and their shadow tables."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'").fetchall()
        fts = {name for name, sql in rows if sql and "USING FTS5" in sql.upper()}
        shadow = {name for name, _ in rows
                  for f in fts if name.startswith(f + "_")}
        tables = {}
        for name, _ in rows:
            if name in fts or name in shadow:
                continue
            tables[name] = {r[1]: r[2] for r in conn.execute(f'PRAGMA table_info("{name}")')}
        return tables, fts, shadow
    finally:
        conn.close()


@pytest.fixture(scope="module")
def schemas(tmp_path_factory):
    root = tmp_path_factory.mktemp("stores")
    out = {}
    for name, (build, _, _) in STORES.items():
        build(root / name)
        out[name] = _schema(root / name)
    return out


def _targets(name: str) -> set[tuple[str, str]]:
    return {(table, col) for table, _key, cols in STORES[name][1] for col in cols}


@pytest.mark.parametrize("name", sorted(STORES))
def test_every_text_column_is_scrubbed_or_exempt_with_a_reason(name, schemas):
    tables, _, _ = schemas[name]
    targets = _targets(name)
    missing = sorted(
        f"{table}.{col} ({declared or 'untyped'})"
        for table, cols in tables.items()
        for col, declared in cols.items()
        if _is_text(declared)
        and (table, col) not in targets
        and (name, table, col) not in NOT_SCRUBBED
    )
    assert not missing, (
        f"{name}: text column(s) the scrub does not rewrite and nobody has exempted. "
        "Add each to the store's *_TARGETS in security/scrub_capture_stores.py, or to "
        f"NOT_SCRUBBED here with the reason it cannot carry a secret: {missing}")


@pytest.mark.parametrize("name", sorted(STORES))
def test_no_scrub_target_names_a_column_that_does_not_exist(name, schemas):
    tables, _, _ = schemas[name]
    stale = sorted(f"{t}.{c}" for t, c in _targets(name) if c not in tables.get(t, {}))
    assert not stale, f"{name}: the scrub lists column(s) the schema no longer has: {stale}"
    keys = sorted(f"{t}.{k}" for t, k, _ in STORES[name][1] if k not in tables.get(t, {}))
    assert not keys, f"{name}: the scrub keys rows by column(s) that do not exist: {keys}"


def test_every_exemption_names_a_real_unscrubbed_text_column(schemas):
    problems = []
    for (name, table, col), why in NOT_SCRUBBED.items():
        tables, _, _ = schemas[name]
        if col not in tables.get(table, {}):
            problems.append(f"{name} {table}.{col}: no such column")
        elif not _is_text(tables[table][col]):
            problems.append(f"{name} {table}.{col}: not a text column")
        elif (table, col) in _targets(name):
            problems.append(f"{name} {table}.{col}: already scrubbed")
        if not why.strip():
            problems.append(f"{name} {table}.{col}: no reason given")
    assert not problems, problems


@pytest.mark.parametrize("name", sorted(STORES))
def test_every_full_text_index_is_rebuilt_after_its_table_is_rewritten(name, schemas):
    _, fts, _ = schemas[name]
    rebuilt = set(STORES[name][2].values())
    assert fts == rebuilt, (
        f"{name}: full-text index(es) {sorted(fts - rebuilt)} would keep the old terms "
        f"after a scrub; stale entries {sorted(rebuilt - fts)}")


def test_the_writers_redacted_columns_are_the_scrubs_targets_too():
    # Capture-time redaction and the after-the-fact scrub must cover the same
    # free text, or a row written before a redactor fix is never cleaned.
    from prometheus.telemetry.writer import REDACTED_COLUMNS
    targets = _targets("telemetry.db")
    for table, cols in REDACTED_COLUMNS.items():
        for col in cols:
            assert (table, col) in targets, f"{table}.{col}"


def test_scrub_apply_leaves_no_canary_in_any_v2_text_column(tmp_path):
    # Spec 11: "scrub --apply on a copy leaves no canary in any new column".
    # The canary goes in by raw SQL, past capture-time redaction, as a row
    # written before a redactor existed would have.
    token = "ghp_" + "Abcdefghij" * 4
    db = tmp_path / "telemetry.db"
    ToolCallTelemetry(db).close()
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO tool_calls (id, timestamp, model, tool_name, success, "
        "raw_before_repair, result_summary) VALUES ('c1', 0, 'm', 'bash', 1, ?, ?)",
        ('{"name": "bash", "input": {"command": "echo ' + token + '"}}', "out: " + token))
    conn.execute(
        "INSERT INTO responses (ts, session_id, round_index, response_kind, prose) "
        "VALUES (0, 'desktop:s', 0, 'prose', ?)", ("done. " + token,))
    conn.commit()
    conn.close()

    counts = scrub.scrub_sqlite(db, scrub.TELEMETRY_TARGETS, apply=True, stamp="t")

    assert counts["tool_calls.raw_before_repair"] == 1
    assert counts["tool_calls.result_summary"] == 1
    assert counts["responses.prose"] == 1
    conn = sqlite3.connect(str(db))
    try:
        for table, col in (("tool_calls", "raw_before_repair"),
                           ("tool_calls", "result_summary"), ("responses", "prose")):
            (value,) = conn.execute(f"SELECT {col} FROM {table}").fetchone()
            assert token not in value, f"{table}.{col}"
    finally:
        conn.close()
