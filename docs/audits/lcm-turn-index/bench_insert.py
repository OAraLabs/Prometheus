"""Synthetic insert-path benchmark: the pre-fix insert vs the fixed one.

No user data: a DB shaped like production (112 sessions, 18,879 rows, one session
holding 8,345) filled with filler text.

* pre-fix: ``INSERT OR REPLACE`` under the legacy non-unique index (the old
  ``insert_message`` statement, reproduced inline);
* fixed: ``LCMConversationStore.insert_message`` on a migrated DB (UNIQUE
  index, ``INSERT ... ON CONFLICT(id) DO NOTHING``, read-back of the stored index);
* anchor: ``LCMConversationStore.next_turn_index``, paid once per numbering
  (re)start, not per row.

Run from a checkout: PYTHONPATH=src python docs/audits/lcm-turn-index/bench_insert.py
"""

import statistics
import tempfile
import time
from pathlib import Path

from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_turn_index_migration import migrate_turn_index
from prometheus.memory.lcm_types import MessagePart

BIG = "telegram:big"
REPS = 400
OLD_INSERT = (
    "INSERT OR REPLACE INTO lcm_messages"
    " (id, session_id, turn_index, role, content, content_json, token_count, timestamp,"
    " compacted, provenance, is_trusted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)"
)


def build(directory: str, *, migrated: bool) -> LCMConversationStore:
    db = Path(directory) / "lcm.db"
    store = LCMConversationStore(db)
    conn = store._conn
    conn.execute("BEGIN")
    n = 0
    for sid, count in [(BIG, 8345)] + [(f"s{i}", 95) for i in range(111)]:
        for ti in range(count):
            conn.execute(
                "INSERT INTO lcm_messages (id, session_id, turn_index, role, content, timestamp)"
                " VALUES (?, ?, ?, 'user', ?, ?)",
                (f"{sid}-{ti}", sid, ti, "x" * 400, float(n)),
            )
            n += 1
    conn.execute("COMMIT")
    if migrated:
        assert migrate_turn_index(db).status == "indexed"
    return store


def old_insert(store: LCMConversationStore, turn_index: int) -> None:
    """The pre-fix insert_message, statement for statement."""
    mid = f"new-{turn_index}"
    store._conn.execute(OLD_INSERT, (mid, BIG, turn_index, "user", "y" * 400, None, 0,
                                     time.time(), "user", 1))
    rowid = store._conn.execute(
        "SELECT rowid FROM lcm_messages WHERE id = ?", (mid,)).fetchone()[0]
    store._conn.execute(
        "INSERT OR REPLACE INTO lcm_messages_fts (rowid, content) VALUES (?, ?)",
        (rowid, "y" * 400))
    store._conn.commit()


def quantiles(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return (round(statistics.median(ordered) * 1e6, 1),
            round(ordered[int(len(ordered) * 0.95)] * 1e6, 1))


def run(*, fixed: bool) -> tuple[tuple[float, float], tuple[float, float] | None]:
    with tempfile.TemporaryDirectory() as directory:
        store = build(directory, migrated=fixed)
        inserts: list[float] = []
        anchors: list[float] = []
        nxt = 8345
        for _ in range(REPS):
            if fixed:
                t = time.perf_counter()
                nxt = store.next_turn_index(BIG)
                anchors.append(time.perf_counter() - t)
                t = time.perf_counter()
                store.insert_message(MessagePart(role="user", content="y" * 400,
                                                 session_id=BIG, turn_index=nxt))
            else:
                t = time.perf_counter()
                old_insert(store, nxt)
            inserts.append(time.perf_counter() - t)
            nxt += 1
        store.close()
        return quantiles(inserts), (quantiles(anchors) if anchors else None)


for round_no in range(3):
    for fixed in (False, True):
        insert, anchor = run(fixed=fixed)
        label = "fixed  " if fixed else "pre-fix"
        print(f"round {round_no} {label} insert p50/p95 us {insert}  anchor p50/p95 us {anchor}")
