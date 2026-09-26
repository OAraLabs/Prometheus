"""memory.db facts are redacted before they are kept, and the scrub covers old ones.

X.37 (#596) redacted token shapes before LCM and telemetry rows exist. memory.db
was left out, and a snapshot showed why it matters: one extracted fact held a
GitHub-shaped token, and facts do not sit still. Recall puts them into the
system prompt of later turns, and the wiki compiles them into pages.

Every fact goes through ``MemoryStore.persist_memory`` (the extractor, the
memory tool, /note) or ``update_memory`` (consolidation), so those two are the
write-time chokepoints. ``scripts/scrub_capture_stores.py --memory`` rewrites
facts written before them.

The fake token is assembled at runtime: .githooks/pre-commit scans whole files.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from prometheus.memory.store import MemoryStore
from prometheus.security import REDACTED

TOKEN = "gh" + "p_" + ("aB3xY9" * 6)
REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "scrub_capture_stores.py"


def _matches(conn: sqlite3.Connection, term: str) -> int:
    return conn.execute("SELECT count(*) FROM memories_fts WHERE memories_fts MATCH ?",
                        ('"' + term + '"',)).fetchone()[0]


def test_a_fact_is_kept_redacted_and_its_index_cannot_find_the_token(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.persist_memory("tool", "GitHub", f"the deploy token is {TOKEN}", 0.9,
                         source_event_ids=["e1"], relationship=f"uses {TOKEN}", tags=[f"tok {TOKEN}"])
    conn = sqlite3.connect(tmp_path / "memory.db")
    fact, rel, tags = conn.execute("SELECT fact, relationship, tags FROM memories").fetchone()
    assert fact == f"the deploy token is {REDACTED}" and rel == f"uses {REDACTED}"
    assert json.loads(tags) == [f"tok {REDACTED}"]
    assert _matches(conn, TOKEN) == 0 and _matches(conn, "redacted") == 1


def test_the_same_fact_again_lands_on_the_one_redacted_row(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    first = store.persist_memory("tool", "GitHub", f"token {TOKEN}", 0.5, source_event_ids=["e1"])
    second = store.persist_memory("tool", "GitHub", f"token {TOKEN}", 0.7, source_event_ids=["e2"])
    assert first == second
    count, mentions = sqlite3.connect(tmp_path / "memory.db").execute(
        "SELECT count(*), max(mention_count) FROM memories").fetchone()
    assert (count, mentions) == (1, 2)


def test_an_update_is_redacted_too(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    mid = store.persist_memory("tool", "GitHub", "clean fact", 0.5, source_event_ids=["e1"])
    store.update_memory(mid, fact=f"now {TOKEN}", tags=[TOKEN])
    fact, tags = sqlite3.connect(tmp_path / "memory.db").execute(
        "SELECT fact, tags FROM memories").fetchone()
    assert TOKEN not in fact and TOKEN not in tags and json.loads(tags) == [REDACTED]


def test_a_clean_fact_is_kept_as_given(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.persist_memory("person", "Ada", "prefers tabs, 4 wide", 0.5, source_event_ids=["e1"],
                         tags=["style"])
    assert sqlite3.connect(tmp_path / "memory.db").execute(
        "SELECT fact, tags FROM memories").fetchone() == ("prefers tabs, 4 wide", '["style"]')


# ---------------------------------------------------------------------------
# The scrub
# ---------------------------------------------------------------------------

def _seed_old(tmp_path: Path) -> Path:
    """A memory.db holding a fact written before the chokepoints: raw SQL, as an
    older store left it. The FTS trigger indexes the token with it."""
    db = tmp_path / "memory.db"
    MemoryStore(db)
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO memories (id, entity_type, entity_name, relationship, fact, confidence,"
                 " source_event_ids, last_mentioned, mention_count, tags, timestamp, manual)"
                 " VALUES ('m1', 'tool', 'GitHub', 'fact', ?, 0.9, '[\"e1\"]', 1.0, 1, '[]', 1.0, 0)",
                 (f"the deploy token is {TOKEN}",))
    conn.execute("INSERT INTO memories (id, entity_type, entity_name, relationship, fact, confidence,"
                 " source_event_ids, last_mentioned, mention_count, tags, timestamp, manual)"
                 " VALUES ('m2', 'person', 'Ada', 'fact', 'prefers tabs', 0.5, '[\"e2\"]', 1.0, 1, '[]',"
                 " 1.0, 0)")
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    return db


def _scrub(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    none = tmp_path / "none"
    env = {**os.environ, "PROMETHEUS_HOME": str(tmp_path / "home")}
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--telemetry", str(none / "t.db"), "--training", str(none / "tr.db"),
         "--trajectories", str(none), "--lcm", str(none / "lcm.db"), "--memory", str(tmp_path / "memory.db"),
         *args], capture_output=True, text=True, timeout=120, env=env)


def test_the_dry_run_counts_facts_and_touches_nothing(tmp_path):
    db = _seed_old(tmp_path)
    before = db.read_bytes()
    r = _scrub(tmp_path)
    assert r.returncode == 0, r.stderr
    assert "memories.fact                                   1 would change" in r.stdout
    assert TOKEN not in r.stdout
    assert db.read_bytes() == before and not list(tmp_path.glob("*.pre-scrub-*"))


def test_apply_rewrites_the_fact_and_leaves_no_copy_in_the_file(tmp_path):
    db = _seed_old(tmp_path)
    assert TOKEN.encode() in db.read_bytes()
    r = _scrub(tmp_path, "--apply")
    assert r.returncode == 0, r.stderr
    assert "1 row(s)/line(s) rewritten" in r.stdout
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT fact FROM memories WHERE id='m1'").fetchone()[0] == (
        f"the deploy token is {REDACTED}")
    assert _matches(conn, TOKEN) == 0 and _matches(conn, "redacted") == 1
    assert _matches(conn, "tabs") == 1  # the clean fact is still indexed
    conn.close()
    # Not only unreachable: gone from the file (secure_delete, the FTS rebuild, the WAL truncate).
    assert TOKEN.encode() not in db.read_bytes()
    [backup] = tmp_path.glob("memory.db.pre-scrub-*")
    assert TOKEN.encode() in backup.read_bytes()
    r2 = _scrub(tmp_path, "--apply")
    assert r2.returncode == 0 and "0 row(s)/line(s) rewritten" in r2.stdout
    # A second run, perhaps in the same second, adds a backup and never overwrites the first.
    backups = list(tmp_path.glob("memory.db.pre-scrub-*"))
    assert len(backups) == 2 and TOKEN.encode() in backup.read_bytes()
