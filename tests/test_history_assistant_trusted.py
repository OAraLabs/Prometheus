"""The history API reports an ordinary assistant reply as trusted.

``ConversationMessage.is_trusted`` defaults to False, the safe posture for
injected material, and nothing sets it on the model's own replies. So every
assistant row since 2026-06-10 is stored with ``is_trusted = 0`` (14,246 rows on
the live store, 2026-10-15), and Beacon, which badges any row whose
``is_trusted`` is false, showed "untrusted" on every reply. The model never saw
the difference: the untrusted banner applies only to user-role turns of
non-user provenance.

Pinned: an assistant row of user provenance reads true whatever is stored; a
tool result, an injected turn and an assistant row of machinery provenance keep
their stored value; a human's turn stays true.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_types import MessagePart
from prometheus.web.server import _wire_is_trusted, create_app

SID = "beacon:trust-test"


def _rows(tmp_path) -> dict[str, dict]:  # noqa: ANN001
    store = LCMConversationStore(tmp_path / "lcm.db")
    for turn, part in enumerate([
        MessagePart(role="user", content="hi", is_trusted=True),
        # As the daemon stores a reply today: the dataclass default.
        MessagePart(role="assistant", content="reply", is_trusted=False),
        # A tool result: user role, user provenance, untrusted data.
        MessagePart(role="user", content="tool output", is_trusted=False),
        MessagePart(role="user", content="worker report", provenance="task_supervisor",
                    is_trusted=False),
        MessagePart(role="assistant", content="summary", provenance="compactor", is_trusted=False),
    ]):
        part.session_id, part.turn_index = SID, turn
        store.insert_message(part)
    client = TestClient(create_app({}, lcm_engine=SimpleNamespace(conversation_store=store)))
    resp = client.get(f"/api/sessions/{SID}/messages")
    assert resp.status_code == 200, resp.text
    return {m["content"]: m for m in resp.json()["messages"]}


def test_an_ordinary_assistant_reply_is_reported_trusted(tmp_path):
    rows = _rows(tmp_path)
    assert rows["reply"]["is_trusted"] is True


def test_every_other_row_is_reported_as_stored(tmp_path):
    rows = _rows(tmp_path)
    assert rows["hi"]["is_trusted"] is True
    assert rows["tool output"]["is_trusted"] is False
    assert rows["worker report"]["is_trusted"] is False
    assert rows["summary"]["is_trusted"] is False, "a machinery-authored assistant row keeps its flag"
    assert rows["worker report"]["provenance"] == "task_supervisor"


def test_wire_is_trusted():
    assert _wire_is_trusted(SimpleNamespace(role="assistant", provenance="user", is_trusted=False))
    assert _wire_is_trusted(SimpleNamespace(role="assistant", provenance=None, is_trusted=False))
    assert not _wire_is_trusted(SimpleNamespace(role="user", provenance="user", is_trusted=False))
    assert not _wire_is_trusted(SimpleNamespace(role="assistant", provenance="cron", is_trusted=False))
