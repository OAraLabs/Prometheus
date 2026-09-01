"""`/api/status` reports how much of the durable store is hidden.

The store reached 92% tombstoned before anyone noticed, because no surface reported it. A number
nobody can see is a number nobody maintains, so this one sits with the other health figures.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.engine.session import SessionManager  # noqa: E402
from prometheus.memory.lcm_conversation_store import LCMConversationStore  # noqa: E402
from prometheus.memory.lcm_types import MessagePart  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402


def _engine(store):
    class _E:
        conversation_store = store
    return _E()


def _client(store):
    return TestClient(create_app({}, session_mgr=SessionManager(), lcm_engine=_engine(store)))


@pytest.fixture()
def store(tmp_path):
    return LCMConversationStore(tmp_path / "lcm.db")


def test_a_clean_store_reports_zero_hidden(store):
    store.add_message("beacon:a", MessagePart(role="user", content="hello"))
    body = _client(store).get("/api/status").json()["store"]
    assert body["sessions"] == 1 and body["hidden"] == 0 and body["hidden_pct"] == 0.0


def test_a_tombstoned_session_is_counted_as_hidden(store):
    store.add_message("beacon:a", MessagePart(role="user", content="hello"))
    store.add_message("beacon:b", MessagePart(role="user", content="bye"))
    store.tombstone_session("beacon:b")
    body = _client(store).get("/api/status").json()["store"]
    assert body["hidden"] == 1
    assert body["hidden_pct"] == 50.0


def test_a_revived_session_is_not_hidden(store):
    """Tombstoned then spoke again = visible. The figure must match what the listing shows."""
    store.add_message("beacon:a", MessagePart(role="user", content="one"))
    store.tombstone_session("beacon:a")
    store.add_message("beacon:a", MessagePart(role="user", content="two"))
    body = _client(store).get("/api/status").json()["store"]
    assert body["hidden"] == 0


def test_the_machine_split_is_the_actionable_half(store):
    """A high hidden_pct with a LOW hidden_machine means probes are accumulating under
    conversation ids — the exact shape retention cannot act on quickly."""
    for sid in ("beacon:p1", "beacon:p2", "smoke:p3"):
        store.add_message(sid, MessagePart(role="user", content="x"))
        store.tombstone_session(sid)
    body = _client(store).get("/api/status").json()["store"]
    assert body["hidden"] == 3
    assert body["hidden_machine"] == 1  # only smoke: named itself


def test_no_store_reports_null_not_zeros():
    """"No store to ask" and "a perfectly clean store" are different answers, and zeros would
    make the second indistinguishable from the first."""
    client = TestClient(create_app({}, session_mgr=SessionManager(), lcm_engine=None))
    assert client.get("/api/status").json()["store"] is None
