"""Retention for tombstoned sessions — the self-healing half of store hygiene.

A tombstone hides a session; it never removes it. That is correct for "take this out of my list",
and it is why 92% of the durable store was hidden probe traffic before the 2026-09-01 audit: the
only thing that ever cleaned up was a person deciding to.

WHAT THIS WILL AND WILL NOT TOUCH
---------------------------------
Only sessions that are BOTH tombstoned AND have no activity newer than their tombstone — the same
predicate ``list_sessions`` uses to hide them. A session that spoke again after being forgotten has
un-hidden itself and is out of scope, permanently.

Two windows, because the two populations are not alike:

* machine traffic (``smoke:``, ``bakeoff:``, ``eval:``, ``gym:``, ``coding:``) is disposable by
  definition and gets a short one.
* everything else is a CONVERSATION the user chose to forget. Forgetting is not deleting, and the
  audit found real chats among the tombstoned — including one asking the agent whether it
  remembered a stated preference. Its window is long, and deliberately so.

DRY RUN IS THE DEFAULT. A retention tool whose default invocation destroys data is a footgun, and
this one is reachable from cron.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from prometheus.memory.session_kind import is_machine_session

if TYPE_CHECKING:  # pragma: no cover
    from prometheus.memory.lcm_conversation_store import LCMConversationStore

log = logging.getLogger(__name__)

DAY_SECONDS = 86_400

#: Disposable by definition — a smoke run is worthless the day after it passed.
DEFAULT_MACHINE_DAYS = 7
#: A conversation the user forgot. Long, because forgetting is not deleting.
DEFAULT_CONVERSATION_DAYS = 90


@dataclass
class RetentionCandidate:
    session_id: str
    messages: int
    tombstoned_at: float
    age_days: float
    machine: bool


@dataclass
class RetentionPlan:
    candidates: list[RetentionCandidate] = field(default_factory=list)
    skipped_too_recent: int = 0
    skipped_revived: int = 0

    @property
    def machine(self) -> list[RetentionCandidate]:
        return [c for c in self.candidates if c.machine]

    @property
    def conversations(self) -> list[RetentionCandidate]:
        return [c for c in self.candidates if not c.machine]


def plan_retention(
    store: "LCMConversationStore",
    *,
    machine_days: int = DEFAULT_MACHINE_DAYS,
    conversation_days: int = DEFAULT_CONVERSATION_DAYS,
    now: float | None = None,
) -> RetentionPlan:
    """Decide what retention WOULD purge. Reads only — nothing is destroyed here.

    Separated from the doing so the decision can be printed, diffed and argued with before any
    row is lost, and so the rule itself is testable without a purge.
    """
    now = time.time() if now is None else now
    plan = RetentionPlan()
    rows = store._conn.execute(
        """
        SELECT s.session_id AS sid, s.n AS n, s.last_ts AS last_ts, t.deleted_at AS tomb
        FROM (SELECT session_id, count(*) AS n, MAX(timestamp) AS last_ts
              FROM lcm_messages GROUP BY session_id) s
        JOIN session_tombstones t ON t.session_id = s.session_id
        """
    ).fetchall()
    for r in rows:
        sid, n, last_ts, tomb = r["sid"], r["n"], r["last_ts"], r["tomb"]
        # Spoke after being forgotten: it is visible again, and out of scope for good.
        if last_ts > tomb:
            plan.skipped_revived += 1
            continue
        machine = is_machine_session(sid)
        age_days = (now - tomb) / DAY_SECONDS
        window = machine_days if machine else conversation_days
        if age_days < window:
            plan.skipped_too_recent += 1
            continue
        plan.candidates.append(
            RetentionCandidate(sid, n, tomb, round(age_days, 1), machine)
        )
    return plan


def apply_retention(store: "LCMConversationStore", plan: RetentionPlan) -> dict[str, int]:
    """Purge exactly what *plan* names. Irreversible.

    Re-checks each session against the plan's own rule immediately before purging: a session that
    spoke between planning and applying must survive, and a long-running cron makes that gap real.
    """
    purged = skipped = 0
    for c in plan.candidates:
        row = store._conn.execute(
            """
            SELECT MAX(m.timestamp) AS last_ts, t.deleted_at AS tomb
            FROM lcm_messages m JOIN session_tombstones t ON t.session_id = m.session_id
            WHERE m.session_id = ?
            """,
            (c.session_id,),
        ).fetchone()
        if row is None or row["last_ts"] is None or row["last_ts"] > row["tomb"]:
            skipped += 1
            log.info("retention: %s spoke since planning — left alone", c.session_id)
            continue
        counts = store.purge_session(c.session_id)
        purged += 1
        log.info(
            "retention: purged %s (%s, %.1f days) — %s",
            c.session_id, "machine" if c.machine else "conversation", c.age_days, counts,
        )
    return {"purged": purged, "skipped": skipped}
