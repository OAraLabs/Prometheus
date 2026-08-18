"""A golden trace keyed to a SURFACE must not be paired with a conversation.

`lcm_context_resolver` rejoins the two halves of a trainable example: telemetry
records the tool call, the LCM store holds the conversation that prompted it,
and they meet on session_id. When that id names a surface rather than a
conversation — the shared web LoopContext pins the literal "web" — the rows
behind it are an accumulation of unrelated turns, and the "example" produced is
a tool call paired with SOMEONE ELSE'S context.

This guard is for the writer that has not been written yet. The 377 rows that
carried "web" before #258 fixed the writer are permanently inert — is_golden is
computed at write time and all of them are 0 — and stay as historical record.
They were spared because is_golden requires a CLOUD provider and three
unrelated configuration facts happened to align. That is not a check.
"""

from __future__ import annotations

import pytest

from prometheus.sentinel.golden_trace_exporter import (
    SHARED_SURFACE_IDS,
    lcm_context_resolver,
)


class _Store:
    """An LCM store that WOULD hand back context, so a skip is the guard
    working rather than an empty store masquerading as one."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def get_messages(self, session_id, limit=500):
        self.asked.append(session_id)
        return [
            _Msg("user", "someone else's question", 100.0),
            _Msg("assistant", "someone else's answer", 101.0),
        ]


class _Msg:
    def __init__(self, role, content, timestamp):
        self.role, self.content, self.timestamp = role, content, timestamp


@pytest.mark.parametrize("surface", sorted(SHARED_SURFACE_IDS))
def test_a_surface_id_resolves_to_no_context(surface):
    store = _Store()
    resolve = lcm_context_resolver(store)
    assert resolve({"session_id": surface, "timestamp": 200.0}) == [], (
        f"a trace keyed to the surface {surface!r} was paired with context"
    )


def test_the_store_is_not_even_consulted_for_a_surface_id():
    """Skip before the lookup: a shared-namespace read is never useful and the
    corpus must not depend on the store happening to be empty."""
    store = _Store()
    resolve = lcm_context_resolver(store)
    resolve({"session_id": "web", "timestamp": 200.0})
    assert store.asked == [], f"store was queried for a surface id: {store.asked}"


def test_a_real_conversation_still_resolves():
    """The guard must not cost the corpus its legitimate examples — including
    the BARE-LITERAL ids that make up most of the real store (UUIDs, probe
    names), which a shape rule would have refused."""
    store = _Store()
    resolve = lcm_context_resolver(store)
    for real in ("telegram:8139235390", "web:s-42", "desktop:abc",
                 "3872cf88-4168-4f2b-9a9d-d5134b5199f0", "lcm-accept-1"):
        got = resolve({"session_id": real, "timestamp": 200.0})
        assert got, f"legitimate session {real!r} resolved to nothing"
        assert got[0]["role"] == "user"


def test_the_surface_set_is_pinned_to_the_permission_literals():
    """One fact, not two: the values that classify as ORIGIN_USER *because a
    human is at that surface* are exactly the values unusable as conversation
    keys *because the surface is shared*. If checker.py's set changes, this
    fails rather than drifting."""
    from prometheus.permissions.checker import _USER_SESSION_LITERALS

    assert set(SHARED_SURFACE_IDS) == set(_USER_SESSION_LITERALS) | {"system"}, (
        "the exporter's surface set and permissions.checker's surface literals "
        "have diverged.\n"
        f"  exporter : {sorted(SHARED_SURFACE_IDS)}\n"
        f"  checker  : {sorted(_USER_SESSION_LITERALS)} (+ 'system')\n"
        "A new surface literal must be added here too, or golden traces from "
        "it will be paired with unrelated context."
    )


def test_no_store_still_returns_no_resolver():
    """Unchanged honest failure: without a store, export nothing rather than
    emitting context-free rows."""
    assert lcm_context_resolver(None) is None
