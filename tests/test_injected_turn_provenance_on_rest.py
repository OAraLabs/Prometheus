"""Injected turns must not come back off the wire looking like the user.

The file-mutation verifier's summary is appended to the transcript with
``role: "user"`` — that IS its wire role to the model, and changing it would
break role alternation. What was wrong is that it also carried
``provenance="user"``, so nothing downstream could tell it apart from
something a human typed:

* ``ChatSession.persist_loop_result`` wrote it to LCM as a user turn, and
  ``GET /api/sessions/{id}/messages`` replayed it as ``role: "user"`` — a
  Beacon chat bubble nobody wrote. (Telegram never showed it because that
  gateway renders only the assistant reply, which is why it went unnoticed
  until the verifier was wired for web/Beacon.)
* the MemoryExtractor mines user-provenance rows for facts, so
  "[FILE MUTATION VERIFIER] Files touched this turn: ..." was eligible to be
  banked as something the user said.

The fix is ``from_injected`` plus surfacing ``provenance`` on the REST route,
which is the field a client filters or badges on. These tests pin the whole
path: constructed turn → LCM row → REST payload.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.engine.messages import ConversationMessage  # noqa: E402
from prometheus.memory.lcm_conversation_store import LCMConversationStore  # noqa: E402
from prometheus.memory.lcm_types import MessagePart  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

SUMMARY = "[FILE MUTATION VERIFIER]\nFiles touched this turn:\n   ✓ /tmp/x.py — write: created"


def _client(store):
    class _LCM:
        conversation_store = store

    return TestClient(create_app({}, lcm_engine=_LCM()))


def _persist(store, msg: ConversationMessage, *, session_id="s", turn_index=0):
    """Mirror ChatSession._persist_to_lcm's field mapping."""
    part = MessagePart(
        role=msg.role,
        content=msg.text,
        content_json=msg.content_json,
        session_id=session_id,
        turn_index=turn_index,
        provenance=msg.provenance,
        is_trusted=msg.is_trusted,
    )
    store.insert_message(part)
    return part


def test_verifier_summary_is_distinguishable_from_a_typed_turn(tmp_path):
    """The load-bearing case: both rows are role:"user" on the wire, and
    ``provenance`` is the ONLY thing that separates them."""
    store = LCMConversationStore(tmp_path / "lcm.db")
    _persist(store, ConversationMessage.from_user_text("fix the parser"), turn_index=0)
    _persist(
        store,
        ConversationMessage.from_injected(
            SUMMARY, provenance="file_mutation_verifier", is_trusted=True,
        ),
        turn_index=1,
    )

    body = _client(store).get("/api/sessions/s/messages").json()
    typed, injected = body["messages"]

    # Same role — a client keying on role alone renders BOTH as user bubbles.
    assert typed["role"] == injected["role"] == "user"

    assert typed["provenance"] == "user"
    assert injected["provenance"] == "file_mutation_verifier", (
        "the injected summary is indistinguishable from a typed turn — this is "
        "the state that made it a chat bubble the user never wrote"
    )
    # Machinery-authored, so trusted: no untrusted-input banner to the model.
    assert typed["is_trusted"] is True
    assert injected["is_trusted"] is True


def test_untrusted_injection_reports_its_trust_state(tmp_path):
    """``provenance`` says who; ``is_trusted`` says whether the content may be
    acted on. Third-party data (a task result) is both non-user AND untrusted."""
    store = LCMConversationStore(tmp_path / "lcm.db")
    _persist(
        store,
        ConversationMessage.from_injected(
            "job 42 output: ...", provenance="task_supervisor", is_trusted=False,
        ),
    )

    (row,) = _client(store).get("/api/sessions/s/messages").json()["messages"]
    assert row["provenance"] == "task_supervisor"
    assert row["is_trusted"] is False


def test_legacy_rows_read_back_as_trusted_user_turns(tmp_path):
    """Rows written before provenance was persisted have no tag. The store
    defaults them to the SAFE history values, and the route must pass those
    through rather than inventing a null a client has to special-case."""
    store = LCMConversationStore(tmp_path / "lcm.db")
    store.insert_message(MessagePart(role="user", content="old row", session_id="s"))

    (row,) = _client(store).get("/api/sessions/s/messages").json()["messages"]
    assert row["provenance"] == "user"
    assert row["is_trusted"] is True
