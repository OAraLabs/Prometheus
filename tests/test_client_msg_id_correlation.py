"""Persisting client_msg_id so a missed WS echo can still retire an optimistic row.

Audit P9.6 / Beacon#144.

THE DEFECT. A client writes an optimistic row keyed by its own client_msg_id when it sends, then
re-keys it to the durable rowid when the WS user-echo arrives. MISS that echo — a socket drop
between POST /api/chat/send returning 200 and the echo, or a client restart in that window — and
nothing ever retires the optimistic row: reconcile inserts the confirmed row under str(row_id), a
DIFFERENT primary key, so both persist and the duplicate renders at the bottom of the thread
forever (row_id IS NULL sorts last). Only the slash-command case had a fix
(dropTransientCommandRow); the ordinary user-message case did not.

Content-matching is not the alternative and was retired on purpose: two identical sends are
legitimately two messages, so matching on text would merge them. The correlation id is the only
deterministic key, and the daemon did not persist it.

WHY A SIDE TABLE, NOT A COLUMN ON lcm_messages. client_msg_id is a transport correlation token,
not conversation content. lcm_messages rows are serialized into content_json, returned by every
history read, and rendered into model context — a wire token does not belong in a prompt. Side
table, same reasoning as session_titles (a rename never rewrites history).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.memory.lcm_conversation_store import LCMConversationStore  # noqa: E402
from prometheus.memory.lcm_types import MessagePart  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402


# --------------------------------------------------------------------------- #
# The store contract
# --------------------------------------------------------------------------- #


def _store(tmp_path: Path) -> LCMConversationStore:
    return LCMConversationStore(db_path=tmp_path / "lcm.db")


class TestStoreRoundTrip:
    def test_set_then_get(self, tmp_path):
        s = _store(tmp_path)
        s.set_message_client_id("beacon:a", 42, "cmid-1")
        assert s.get_message_client_ids("beacon:a") == {42: "cmid-1"}

    def test_scoped_per_session(self, tmp_path):
        """One session's ids must not leak into another's history read."""
        s = _store(tmp_path)
        s.set_message_client_id("beacon:a", 42, "cmid-a")
        s.set_message_client_id("beacon:b", 42, "cmid-b")  # SAME rowid, different session
        assert s.get_message_client_ids("beacon:a") == {42: "cmid-a"}
        assert s.get_message_client_ids("beacon:b") == {42: "cmid-b"}

    def test_unknown_session_is_empty_not_error(self, tmp_path):
        s = _store(tmp_path)
        assert s.get_message_client_ids("beacon:never-wrote") == {}

    def test_blank_id_and_missing_rowid_are_no_ops(self, tmp_path):
        """Best-effort by contract: nothing half-written, and no raise into the send path.

        A blank id is what a client that did not generate one sends; a None rowid is what a persist
        that surfaced no row returns. Either way there is nothing to correlate, and storing a blank
        would make 'no id recorded' indistinguishable from 'the id was empty'.
        """
        s = _store(tmp_path)
        s.set_message_client_id("beacon:a", 42, "")
        s.set_message_client_id("beacon:a", 42, "   ")
        s.set_message_client_id("beacon:a", None, "cmid")
        assert s.get_message_client_ids("beacon:a") == {}

    def test_same_rowid_replaces_rather_than_ignoring(self, tmp_path):
        """A rowid is durable and unique, so a second write is a CORRECTION.

        INSERT OR IGNORE would keep the stale id — and then the row that actually correlates to the
        client's optimistic row would stay stranded, which is the whole defect.
        """
        s = _store(tmp_path)
        s.set_message_client_id("beacon:a", 42, "first")
        s.set_message_client_id("beacon:a", 42, "second")
        assert s.get_message_client_ids("beacon:a") == {42: "second"}

    def test_survives_a_store_reopen(self, tmp_path):
        """The point of persisting it at all: the missed echo may arrive after a restart."""
        db = tmp_path / "lcm.db"
        s1 = LCMConversationStore(db_path=db)
        s1.set_message_client_id("beacon:a", 42, "cmid-1")
        s1.close()
        s2 = LCMConversationStore(db_path=db)
        assert s2.get_message_client_ids("beacon:a") == {42: "cmid-1"}

    def test_purge_session_removes_the_rows(self, tmp_path):
        """A forgotten session takes its correlation ids with it.

        purge_session enumerates its side tables deliberately so a NEW one is a visible omission
        rather than a silent survivor. This is the test that makes that omission loud — without the
        table in that list, the ids outlive the session they belong to.
        """
        s = _store(tmp_path)
        s.set_message_client_id("beacon:doomed", 1, "c1")
        s.set_message_client_id("beacon:doomed", 2, "c2")
        s.set_message_client_id("beacon:keeper", 1, "c3")

        counts = s.purge_session("beacon:doomed")

        assert counts.get("message_client_ids") == 2, counts
        assert s.get_message_client_ids("beacon:doomed") == {}
        assert s.get_message_client_ids("beacon:keeper") == {1: "c3"}


# --------------------------------------------------------------------------- #
# The wire: GET /api/sessions/{id}/messages
# --------------------------------------------------------------------------- #


def _client(store):
    class _LCM:
        conversation_store = store

    return TestClient(create_app({}, lcm_engine=_LCM()))


class TestHistoryExposesIt:
    def test_recorded_id_is_returned_on_its_row(self, tmp_path):
        s = _store(tmp_path)
        m = MessagePart(role="user", content="hello", session_id="s", turn_index=0, timestamp=100.0)
        s.insert_message(m)
        s.set_message_client_id("s", m.row_id, "cmid-abc")

        body = _client(s).get("/api/sessions/s/messages").json()
        assert len(body["messages"]) == 1
        assert body["messages"][0]["client_msg_id"] == "cmid-abc"

    def test_a_row_without_an_id_omits_the_key_entirely(self, tmp_path):
        """Omit, do not null — the same contract row_id uses on chat_done.

        An older client's decoder is unchanged, and a newer client can tell 'no id recorded' (key
        absent) from 'the id was the empty string' (never stored, see the no-op test above).
        """
        s = _store(tmp_path)
        m = MessagePart(role="user", content="hello", session_id="s", turn_index=0, timestamp=100.0)
        s.insert_message(m)

        body = _client(s).get("/api/sessions/s/messages").json()
        assert "client_msg_id" not in body["messages"][0]

    def test_ids_land_on_their_own_rows_not_the_first(self, tmp_path):
        """The lookup is per-rowid, so a partial set must not smear across the page.

        Assistant turns are never given a client_msg_id (only a user send carries one), which is
        exactly the mixed case a real history read produces.
        """
        s = _store(tmp_path)
        u1 = MessagePart(role="user", content="first", session_id="s", turn_index=0, timestamp=100.0)
        a1 = MessagePart(role="assistant", content="reply", session_id="s", turn_index=1, timestamp=101.0)
        u2 = MessagePart(role="user", content="second", session_id="s", turn_index=2, timestamp=102.0)
        for m in (u1, a1, u2):
            s.insert_message(m)
        s.set_message_client_id("s", u1.row_id, "c-1")
        s.set_message_client_id("s", u2.row_id, "c-2")

        body = _client(s).get("/api/sessions/s/messages").json()
        by_id = {m["message_id"]: m for m in body["messages"]}
        assert by_id[u1.row_id].get("client_msg_id") == "c-1"
        assert "client_msg_id" not in by_id[a1.row_id], "an assistant turn carries no client id"
        assert by_id[u2.row_id].get("client_msg_id") == "c-2"

    def test_a_store_without_the_method_still_serves_history(self, tmp_path):
        """Beacon must work against any daemon vintage — and so must the daemon's own route.

        A store that predates these methods (or a duck-typed fake in another test) must degrade to
        'no ids', not 500 the endpoint that every client polls. Delegates everything except the new
        method, so the message rows are real; only the lookup is absent.
        """
        s = _store(tmp_path)
        m = MessagePart(role="user", content="hello", session_id="s", turn_index=0, timestamp=100.0)
        s.insert_message(m)

        class _OldStore:
            def __getattr__(self, name):
                if name == "get_message_client_ids":
                    raise AttributeError(name)  # an older store genuinely lacks it
                return getattr(s, name)

        class _LCM:
            conversation_store = _OldStore()

        r = TestClient(create_app({}, lcm_engine=_LCM())).get("/api/sessions/s/messages")
        assert r.status_code == 200, "a store without the lookup must not 500 history"
        assert "client_msg_id" not in r.json()["messages"][0]


# --------------------------------------------------------------------------- #
# The write site: ws_server must record it where BOTH halves are in hand
# --------------------------------------------------------------------------- #


class TestWriteWiring:
    """The block that records the id belongs in _handle_send_message ONLY.

    run_turn_awaited also persists a user turn and also has a row_id, but it takes no
    client_msg_id — so there is nothing to correlate. Writing it there is not merely useless, it
    is a NameError. This pins the placement structurally, because the mistake is invisible to every
    other test here: they call the store and the route directly, and would stay green with the
    write in the wrong method or absent entirely.
    """

    def test_the_write_is_in_handle_send_message(self):
        import inspect

        from prometheus.web import ws_server

        src = inspect.getsource(ws_server.WebSocketBridge._handle_send_message)
        assert "set_message_client_id" in src, (
            "_handle_send_message must record the correlation id — it is the only place holding "
            "both the incoming client_msg_id and the rowid from the persist"
        )

    def test_the_write_is_not_in_run_turn_awaited(self):
        """It must not RECORD there. Note it does legitimately mention the id: its echo frame
        carries `"client_msg_id": None`, which is correct — that method has no client id to report.
        So the assertion is on the write, not on the name; asserting the name absent would be wrong
        (it failed once, correctly, on exactly that).
        """
        import inspect

        from prometheus.web import ws_server

        src = inspect.getsource(ws_server.WebSocketBridge.run_turn_awaited)
        assert "set_message_client_id" not in src, (
            "run_turn_awaited has no client_msg_id parameter; recording there is a NameError"
        )
        # And the method genuinely takes no id, which is why recording there cannot work.
        params = inspect.signature(ws_server.WebSocketBridge.run_turn_awaited).parameters
        assert "client_msg_id" not in params

    def test_recorded_once_per_send(self):
        """Exactly one call site — two would double-write or write on the wrong path."""
        import inspect

        from prometheus.web import ws_server

        total = inspect.getsource(ws_server).count("set_message_client_id(session_id")
        assert total == 1, f"expected one write site, found {total}"
