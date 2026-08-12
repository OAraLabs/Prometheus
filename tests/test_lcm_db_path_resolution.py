"""LCM database path resolution — real files, no mocks.

WHY THIS FILE EXISTS
--------------------
Four sites named ``lcm.db`` and they did not agree. Three private
``_default_db_path()`` copies returned ``get_config_dir() / "lcm.db"``;
``LCMEngine`` passed ``get_data_dir() / "lcm.db"`` to the two stores it builds.
So "the shared lcm.db" — a phrase in all three class docstrings — was two
files: conversations and summaries in ``data/``, checkpoints in the config root.

The defect had no side from which it looked like one:

* ``CheckpointStore`` wrote to the config-root file and found its own table
  there, so checkpointing "worked".
* ``LCMConversationStore``'s schema *also* creates a ``checkpoints`` table, in
  the data-dir file. A reader who queried the data-dir file found the table
  present and empty — which is indistinguishable from "no checkpoints yet".

Mocked tests structurally cannot catch that class. Asserting
``store._db_path == expected`` passes happily while the process writes
somewhere else, and patching the resolver proves only that the patch worked.
So every test here **constructs the real stores, writes real rows, and reads
them back through a separate connection**.

Two locations are made observable:

  data_db   — ``<config>/data/lcm.db``. Everything must land here.
  legacy_db — ``<config>/lcm.db``, where the old default resolved. Must stay
              absent. A store that re-derives its own path creates it, and the
              negative assertion fires.

The negative assertion is the load-bearing half: a positive-only suite passes
for a store that opens *both* files.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from prometheus.config.paths import (
    get_config_dir,
    get_lcm_db_path,
    get_legacy_lcm_db_path,
)


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """Yield (data_db, legacy_db) with a per-test config dir.

    ``PROMETHEUS_DATA_DIR`` is deliberately cleared rather than set: the
    default ``<config>/data`` resolution is the thing under test, and pinning
    it with an env var would test the env var instead.
    """
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("PROMETHEUS_DATA_DIR", raising=False)

    data_db = config_dir / "data" / "lcm.db"     # the one true file
    legacy_db = config_dir / "lcm.db"            # must never be created
    yield data_db, legacy_db


def _assert_legacy_untouched(legacy_db: Path, consumer: str) -> None:
    """The negative half. A store that re-derives its path trips this."""
    strays = [legacy_db] + [
        legacy_db.parent / (legacy_db.name + suffix) for suffix in ("-wal", "-shm")
    ]
    found = [str(p) for p in strays if p.exists()]
    assert not found, (
        f"{consumer} opened the LEGACY config-root lcm.db instead of the "
        f"data-dir one — it is not resolving through get_lcm_db_path(). "
        f"Stray files: {found}"
    )


# ---------------------------------------------------------------------------
# 1. Every bare-constructed store lands on the same real file
# ---------------------------------------------------------------------------


def test_conversation_store_defaults_to_the_data_dir_file(roots):
    """A bare LCMConversationStore() writes a message into <data>/lcm.db."""
    data_db, legacy_db = roots
    from prometheus.memory.lcm_conversation_store import LCMConversationStore

    from prometheus.memory.lcm_types import MessagePart

    store = LCMConversationStore()
    store.add_message(
        "sess-conv", MessagePart(role="user", content="conversation store marker")
    )
    store.close()

    assert data_db.exists(), "LCMConversationStore did not create <data>/lcm.db"
    with sqlite3.connect(data_db) as conn:
        rows = conn.execute(
            "SELECT content FROM lcm_messages WHERE session_id = ?", ("sess-conv",)
        ).fetchall()
    assert [r[0] for r in rows] == ["conversation store marker"], (
        "the message did not land in the data-dir file"
    )
    _assert_legacy_untouched(legacy_db, "LCMConversationStore")


def test_summary_store_defaults_to_the_data_dir_file(roots):
    """A bare LCMSummaryStore() writes a summary node into <data>/lcm.db."""
    data_db, legacy_db = roots
    from prometheus.memory.lcm_summary_store import LCMSummaryStore
    from prometheus.memory.lcm_types import SummaryNode

    store = LCMSummaryStore()
    store.add_summary(
        "sess-sum", SummaryNode(id="sum-1", summary_text="summary store marker")
    )
    store.close()

    assert data_db.exists(), "LCMSummaryStore did not create <data>/lcm.db"
    with sqlite3.connect(data_db) as conn:
        rows = conn.execute(
            "SELECT summary_text FROM lcm_summaries WHERE session_id = ?",
            ("sess-sum",),
        ).fetchall()
    assert [r[0] for r in rows] == ["summary store marker"], (
        "the summary did not land in the data-dir file"
    )
    _assert_legacy_untouched(legacy_db, "LCMSummaryStore")


def test_checkpoint_store_defaults_to_the_data_dir_file(roots):
    """A bare CheckpointStore() writes a checkpoint into <data>/lcm.db.

    This is the site that actually diverged in production: ``__main__.py``
    constructs ``CheckpointStore()`` with no arguments on every boot.
    """
    data_db, legacy_db = roots
    from prometheus.coordinator.divergence import Checkpoint, CheckpointStore

    store = CheckpointStore()
    store.save(
        Checkpoint(
            task_id="task-cp",
            step_number=1,
            goal_description="checkpoint store marker",
            goal_hash="hash",
            messages_snapshot=[{"role": "user", "content": "hi"}],
            tool_calls=[],
        )
    )
    store.close()

    assert data_db.exists(), "CheckpointStore did not create <data>/lcm.db"
    with sqlite3.connect(data_db) as conn:
        rows = conn.execute(
            "SELECT goal_description FROM checkpoints WHERE task_id = ?", ("task-cp",)
        ).fetchall()
    assert [r[0] for r in rows] == ["checkpoint store marker"], (
        "the checkpoint did not land in the data-dir file"
    )
    _assert_legacy_untouched(legacy_db, "CheckpointStore")


def test_engine_default_matches_the_bare_store_default(roots):
    """LCMEngine's default and a bare store's default must be one file.

    The original defect in one assertion: the engine passed an explicit path
    that disagreed with the default its own stores would have chosen.
    """
    data_db, legacy_db = roots
    from unittest.mock import MagicMock

    from prometheus.memory.lcm_conversation_store import LCMConversationStore
    from prometheus.memory.lcm_engine import LCMEngine

    engine = LCMEngine(provider=MagicMock())
    try:
        engine_path = Path(engine._db_path).resolve()
    finally:
        engine.close()

    store = LCMConversationStore()
    try:
        store_path = Path(store._db_path).resolve()
    finally:
        store.close()

    assert engine_path == store_path == data_db.resolve(), (
        f"LCMEngine and a bare LCMConversationStore resolve to different "
        f"files: engine={engine_path}, store={store_path}. This is FL-3 "
        f"exactly — 'the shared lcm.db' being two files."
    )
    _assert_legacy_untouched(legacy_db, "LCMEngine")


# ---------------------------------------------------------------------------
# 2. The admission direction — the shared-file claim must actually hold
#
# Breach tests prove the stores stay out of the legacy file. This proves the
# thing the docstrings promise: that a checkpoint and a conversation written by
# two different classes are visible in ONE database (Standing-Principles §2c).
# ---------------------------------------------------------------------------


def test_all_three_stores_share_one_database(roots):
    """One connection sees rows written by all three store classes."""
    data_db, legacy_db = roots
    from prometheus.coordinator.divergence import Checkpoint, CheckpointStore
    from prometheus.memory.lcm_conversation_store import LCMConversationStore
    from prometheus.memory.lcm_summary_store import LCMSummaryStore
    from prometheus.memory.lcm_types import SummaryNode

    from prometheus.memory.lcm_types import MessagePart

    conv = LCMConversationStore()
    conv.add_message("shared", MessagePart(role="user", content="conv row"))
    conv.close()

    sums = LCMSummaryStore()
    sums.add_summary("shared", SummaryNode(id="s-shared", summary_text="summary row"))
    sums.close()

    cps = CheckpointStore()
    cps.save(
        Checkpoint(
            task_id="shared",
            step_number=1,
            goal_description="checkpoint row",
            goal_hash="h",
            messages_snapshot=[],
            tool_calls=[],
        )
    )
    cps.close()

    with sqlite3.connect(data_db) as conn:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("lcm_messages", "lcm_summaries", "checkpoints")
        }

    assert counts == {"lcm_messages": 1, "lcm_summaries": 1, "checkpoints": 1}, (
        f"the three stores are not sharing one database: {counts}. A zero here "
        f"is the FL-3 signature — the table exists (both schemas create it) so "
        f"an empty count reads as 'nothing written yet' rather than 'wrong file'."
    )
    _assert_legacy_untouched(legacy_db, "the three stores together")


# ---------------------------------------------------------------------------
# 3. Resolver behaviour
# ---------------------------------------------------------------------------


def test_resolver_returns_the_data_dir_file(roots):
    """Out of the box: <config>/data/lcm.db, and the legacy name differs."""
    data_db, legacy_db = roots

    assert get_lcm_db_path() == data_db
    assert get_legacy_lcm_db_path() == legacy_db
    assert get_lcm_db_path() != get_legacy_lcm_db_path(), (
        "the two helpers must name different files — if they ever converge, "
        "reset-data's legacy cleanup silently becomes a no-op"
    )


def test_data_dir_env_var_moves_the_db(tmp_path, monkeypatch):
    """PROMETHEUS_DATA_DIR relocates the file; the config root is unaffected."""
    config_dir = tmp_path / "cfg"
    data_dir = tmp_path / "elsewhere"
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(data_dir))

    assert get_lcm_db_path() == data_dir / "lcm.db"
    assert get_legacy_lcm_db_path() == config_dir / "lcm.db"


def test_explicit_db_path_still_wins(tmp_path, monkeypatch):
    """The default is a default. Tests, the gym and sandboxes pass their own."""
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("PROMETHEUS_DATA_DIR", raising=False)
    from prometheus.coordinator.divergence import CheckpointStore
    from prometheus.memory.lcm_conversation_store import LCMConversationStore

    explicit = tmp_path / "explicit.db"
    conv = LCMConversationStore(db_path=explicit)
    conv.close()
    CheckpointStore(db_path=explicit).close()

    assert explicit.exists()
    assert not get_lcm_db_path().exists(), (
        "an explicit db_path must not also create the default file"
    )


# ---------------------------------------------------------------------------
# 4. The migration decision, pinned as deliberate (Standing-Principles §10)
#
# Checkpoint rows in a pre-fix config-root lcm.db are NOT copied forward. The
# trade is in the PR body; this test exists so the decision cannot be quietly
# reversed, and so the reasoning is discoverable from the code.
#
# Copying is not merely unnecessary, it is unsafe: CheckpointStore.delete_after
# removes checkpoints during a rollback, so a copy that ran on every
# construction would resurrect rows a rollback had deliberately deleted — the
# cure re-introducing the disease, which is §10's own second failure.
# ---------------------------------------------------------------------------


def test_legacy_checkpoints_are_not_migrated_and_do_not_leak(roots):
    """A populated legacy file is ignored: not read, not copied, not deleted."""
    data_db, legacy_db = roots
    from prometheus.coordinator.divergence import CheckpointStore

    legacy_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(legacy_db) as conn:
        conn.executescript("""
            CREATE TABLE checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                goal_hash TEXT NOT NULL,
                goal_description TEXT,
                messages_json TEXT NOT NULL,
                tool_calls_json TEXT NOT NULL,
                divergence_score REAL DEFAULT 0.0,
                created_at REAL NOT NULL,
                UNIQUE(task_id, step_number)
            );
        """)
        conn.execute(
            "INSERT INTO checkpoints (task_id, step_number, goal_hash, "
            "goal_description, messages_json, tool_calls_json, created_at) "
            "VALUES ('old', 1, 'h', 'stale', '[]', '[]', 0.0)"
        )

    store = CheckpointStore()
    assert store.get_latest("old") is None, (
        "CheckpointStore read the legacy file — the fix must not silently "
        "reach back into the old location"
    )

    with sqlite3.connect(data_db) as conn:
        migrated = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
    assert migrated == 0, (
        "legacy checkpoints were copied into the new file. This is a "
        "DELIBERATE non-migration: an unconditional copy would resurrect rows "
        "that CheckpointStore.delete_after removed during a rollback. If you "
        "are adding a migration, it must be one-shot and marked — read "
        "Standing-Principles CROSS-CUTTING §10 first."
    )

    with sqlite3.connect(legacy_db) as conn:
        still_there = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
    assert still_there == 1, (
        "the legacy file was mutated. Deleting it is reset-data's job, on "
        "explicit user confirmation — never a side effect of construction."
    )


def test_reset_data_lists_the_legacy_file(roots):
    """reset-data promises 'all user data'; the legacy file must be a target."""
    data_db, legacy_db = roots
    import inspect

    from prometheus import __main__ as prom_main

    src = inspect.getsource(prom_main._reset_data)
    assert "get_legacy_lcm_db_path()" in src, (
        "_reset_data() no longer targets the legacy config-root lcm.db. Its "
        "checkpoints.messages_json holds full conversation messages, so "
        "'delete all user data' leaving it behind is a privacy miss, not "
        "clutter."
    )
    assert "get_lcm_db_path()" in src, (
        "_reset_data() re-derives the current lcm.db path instead of calling "
        "the resolver"
    )


# ---------------------------------------------------------------------------
# 5. Source guard — no site may resolve the LCM db path independently
# ---------------------------------------------------------------------------

_SRC = Path(__file__).resolve().parent.parent / "src" / "prometheus"
_RESOLVER = _SRC / "config" / "paths.py"

_DB_FILENAME = "lcm.db"


def _independent_resolutions(source: str) -> list[str]:
    """Return every site in ``source`` that names the LCM db file itself.

    Two shapes, both drawn from the pre-fix tree:

      * ``<anything> / "lcm.db"``      — LCMEngine's inline ``get_data_dir()``
                                         expression, and reset-data's
                                         ``data_dir / "lcm.db"``
      * ``_DB_NAME = "lcm.db"``        — the module constant the three private
                                         ``_default_db_path()`` copies divided
                                         by, one indirection out

    This walks the AST rather than grepping lines, for two reasons that both
    bit the first version of this guard:

      * a text ban fires inside prose that *explains* the old form — the
        CheckpointStore docstring documenting the split went red against a
        guard hunting the thing it describes (Standing-Principles §3c)
      * matching the NAME ``_DB_NAME`` flagged ``memory/store.py``, where the
        same constant holds ``"memory.db"``. The guard answered cleanly, about
        a different database (§2b)

    Matching the literal VALUE, in expression position only, is immune to both.
    """
    offenders: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.right, ast.Constant)
            and node.right.value == _DB_FILENAME
        ):
            offenders.append(f"line {node.lineno}: <expr> / {_DB_FILENAME!r}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, ast.Constant) and value.value == _DB_FILENAME:
                offenders.append(
                    f"line {node.lineno}: a module constant holding "
                    f"{_DB_FILENAME!r}"
                )
    return offenders


def test_no_site_resolves_the_lcm_db_path_independently():
    """Fail if a fifth site appears. This is what stops the drift returning."""
    offenders: list[str] = []
    for py in sorted(_SRC.rglob("*.py")):
        if py.resolve() == _RESOLVER:
            continue  # the one legitimate place
        rel = py.relative_to(_SRC.parent.parent)
        for hit in _independent_resolutions(py.read_text(encoding="utf-8")):
            offenders.append(f"{rel}:{hit}")

    assert not offenders, (
        "These sites resolve the LCM database path independently instead of "
        "calling get_lcm_db_path(). Four such sites disagreed before "
        "2026-08-12 and split 'the shared lcm.db' into two files; every one "
        "must go through prometheus.config.paths.\n  " + "\n  ".join(offenders)
    )


def test_source_guard_actually_matches_the_old_forms():
    """The guard is only worth having if it recognises what it hunts for.

    Every positive here is a real line from the pre-fix tree. The negatives are
    the two ways the first version of this guard was wrong.
    """
    # The historical forms — each must be caught.
    assert _independent_resolutions(
        'self._db_path = db_path or (get_data_dir() / "lcm.db")'
    )
    assert _independent_resolutions('("data/lcm.db", data_dir / "lcm.db"),')
    assert _independent_resolutions('_DB_NAME = "lcm.db"')
    assert _independent_resolutions('cfg = get_config_dir() / "lcm.db"')

    # The fixed forms — a guard that fires on these is unsatisfiable.
    assert not _independent_resolutions(
        "self._db_path = db_path or get_lcm_db_path()"
    )
    assert not _independent_resolutions(
        '("lcm.db (legacy)", get_legacy_lcm_db_path()),'
    )

    # Prose describing the old form must NOT trip it (§3c): the most careful
    # sentence in the codebase is the one that explains the bug.
    assert not _independent_resolutions(
        'def f():\n    """It used to read get_config_dir() / "lcm.db" here."""\n'
    )
    assert not _independent_resolutions(
        '# the old default was get_data_dir() / "lcm.db"\nx = 1'
    )

    # A different database is a different subject (§2b): memory/store.py holds
    # `_DB_NAME = "memory.db"` and divides by it exactly like the old LCM code.
    assert not _independent_resolutions(
        '_DB_NAME = "memory.db"\ndef p():\n    return get_config_dir() / _DB_NAME\n'
    )


def test_the_private_default_helpers_are_gone():
    """The three _default_db_path() copies must not come back.

    A re-added copy that happens to return the right path today is the exact
    shape that diverged before: correct at birth, drifting silently after.
    """
    survivors: list[str] = []
    for name in (
        "memory/lcm_conversation_store.py",
        "memory/lcm_summary_store.py",
        "coordinator/divergence.py",
    ):
        text = (_SRC / name).read_text(encoding="utf-8")
        if "_default_db_path" in text:
            survivors.append(name)
    assert not survivors, (
        f"{survivors} still define or call a private _default_db_path(). "
        f"There is one resolver: prometheus.config.paths.get_lcm_db_path()."
    )


def test_resolver_lives_under_the_config_dir_by_default(tmp_path, monkeypatch):
    """A sanity pin on the relationship the fix depends on.

    get_data_dir() defaults to <config>/data, so the LCM file stays inside the
    config tree. If that ever changes, reset-data's dir_targets and the backup
    tooling need revisiting — this test is where that shows up.
    """
    config_dir = tmp_path / "cfg"
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("PROMETHEUS_DATA_DIR", raising=False)

    assert get_config_dir() in get_lcm_db_path().parents
