"""Retention: what it takes, and far more importantly what it refuses to take.

Written against the 2026-09-01 audit, where 92% of the store was hidden probe traffic AND the
hidden set contained real conversations. Both facts have to survive automation: the probes should
go without a human, and the conversations must not.
"""

from __future__ import annotations

import time

import pytest

from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_types import MessagePart
from prometheus.memory.retention import (
    DAY_SECONDS,
    apply_retention,
    plan_retention,
)

NOW = 1_800_000_000.0


@pytest.fixture()
def store(tmp_path):
    return LCMConversationStore(tmp_path / "lcm.db")


def add(store, sid, *, when, n=1):
    for i in range(n):
        m = MessagePart(role="user", content=f"{sid} #{i}", timestamp=when)
        store.add_message(sid, m)
    # add_message stamps its own timestamp; force the one the test means
    store._conn.execute("UPDATE lcm_messages SET timestamp = ? WHERE session_id = ?", (when, sid))
    store._conn.commit()


def tombstone(store, sid, *, when):
    store._conn.execute(
        "INSERT OR REPLACE INTO session_tombstones (session_id, deleted_at) VALUES (?, ?)",
        (sid, when),
    )
    store._conn.commit()


def test_old_machine_traffic_is_collected(store):
    add(store, "smoke:s4-1", when=NOW - 40 * DAY_SECONDS)
    tombstone(store, "smoke:s4-1", when=NOW - 30 * DAY_SECONDS)
    plan = plan_retention(store, now=NOW)
    assert [c.session_id for c in plan.machine] == ["smoke:s4-1"]


def test_a_forgotten_CONVERSATION_survives_the_machine_window(store):
    """The heart of it. 30 days is ancient for a smoke run and nothing for a chat."""
    add(store, "beacon:real", when=NOW - 40 * DAY_SECONDS)
    tombstone(store, "beacon:real", when=NOW - 30 * DAY_SECONDS)
    plan = plan_retention(store, now=NOW)
    assert plan.candidates == []
    assert plan.skipped_too_recent == 1


def test_a_forgotten_conversation_is_collected_only_after_its_own_window(store):
    add(store, "beacon:real", when=NOW - 200 * DAY_SECONDS)
    tombstone(store, "beacon:real", when=NOW - 100 * DAY_SECONDS)
    plan = plan_retention(store, now=NOW)
    assert [c.session_id for c in plan.conversations] == ["beacon:real"]


def test_a_session_that_spoke_after_being_forgotten_is_never_a_candidate(store):
    """Revived = visible again. Out of scope permanently, however old the tombstone."""
    tombstone(store, "smoke:revived", when=NOW - 400 * DAY_SECONDS)
    add(store, "smoke:revived", when=NOW - 399 * DAY_SECONDS)
    plan = plan_retention(store, now=NOW)
    assert plan.candidates == []
    assert plan.skipped_revived == 1


def test_a_session_with_no_tombstone_is_never_a_candidate(store):
    add(store, "smoke:live", when=NOW - 900 * DAY_SECONDS)
    plan = plan_retention(store, now=NOW)
    assert plan.candidates == []


def test_planning_destroys_nothing(store):
    add(store, "smoke:s4-1", when=NOW - 40 * DAY_SECONDS, n=3)
    tombstone(store, "smoke:s4-1", when=NOW - 30 * DAY_SECONDS)
    plan_retention(store, now=NOW)
    assert len(store.get_messages("smoke:s4-1")) == 3


def test_apply_purges_only_what_the_plan_named(store):
    add(store, "smoke:doomed", when=NOW - 40 * DAY_SECONDS, n=2)
    tombstone(store, "smoke:doomed", when=NOW - 30 * DAY_SECONDS)
    add(store, "beacon:keeper", when=NOW - 400 * DAY_SECONDS, n=2)
    tombstone(store, "beacon:keeper", when=NOW - 10 * DAY_SECONDS)  # inside the 90-day window
    plan = plan_retention(store, now=NOW)
    result = apply_retention(store, plan)
    assert result == {"purged": 1, "skipped": 0}
    assert store.get_messages("smoke:doomed") == []
    assert len(store.get_messages("beacon:keeper")) == 2


def test_apply_re_checks_and_spares_a_session_that_spoke_since_planning(store):
    """A long cron makes the gap between deciding and doing real."""
    add(store, "smoke:racing", when=NOW - 40 * DAY_SECONDS, n=1)
    tombstone(store, "smoke:racing", when=NOW - 30 * DAY_SECONDS)
    plan = plan_retention(store, now=NOW)
    assert len(plan.candidates) == 1
    add(store, "smoke:racing", when=NOW, n=1)  # it speaks again, after the plan was made
    result = apply_retention(store, plan)
    assert result == {"purged": 0, "skipped": 1}
    assert len(store.get_messages("smoke:racing")) == 2


def test_windows_are_configurable(store):
    add(store, "beacon:real", when=NOW - 40 * DAY_SECONDS)
    tombstone(store, "beacon:real", when=NOW - 30 * DAY_SECONDS)
    assert plan_retention(store, now=NOW).candidates == []
    tight = plan_retention(store, conversation_days=10, now=NOW)
    assert [c.session_id for c in tight.candidates] == ["beacon:real"]
