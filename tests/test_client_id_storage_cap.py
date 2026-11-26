"""The client-id STORAGE cap is wider than the typed-chat WIRE cap.

An ACP idempotency claim is ``acp-key:`` + a schema-legal key of up to 255
chars (263 total). The column is plain TEXT with no length constraint, so the
widening is the writer's guard: storage accepts up to 263, still refuses
longer by name (never truncates), and the wire surfaces keep their own
128-char refusal unchanged. A 255-char key must round-trip through a real
store close/reopen (the restart the claim exists for).
"""

from __future__ import annotations

import pytest

from prometheus.memory.lcm_conversation_store import (
    CLIENT_ID_STORAGE_MAX,
    CLIENT_MSG_ID_MAX,
    LCMConversationStore,
)


def test_the_storage_cap_fits_a_prefixed_255_char_key():
    assert CLIENT_ID_STORAGE_MAX == 263
    assert len("acp-key:") + 255 == CLIENT_ID_STORAGE_MAX
    assert CLIENT_MSG_ID_MAX == 128  # the wire cap is unchanged


@pytest.mark.parametrize("n", [128, 129, 255, 263])
def test_values_up_to_the_storage_cap_round_trip(tmp_path, n):
    store = LCMConversationStore(tmp_path / "lcm.db")
    cid = "k" * n
    store.set_message_client_id("acp:s1", 7, cid)
    assert store.get_message_client_ids("acp:s1") == {7: cid}
    assert store.has_message_client_id("acp:s1", cid)
    store.close()
    # the restart the claim exists for
    reopened = LCMConversationStore(tmp_path / "lcm.db")
    assert reopened.get_message_client_ids("acp:s1") == {7: cid}
    assert reopened.has_message_client_id("acp:s1", cid)


def test_the_prefixed_schema_max_key_round_trips(tmp_path):
    store = LCMConversationStore(tmp_path / "lcm.db")
    claim = "acp-key:" + "k" * 255
    assert len(claim) == 263
    store.set_message_client_id("acp:s1", 1, claim)
    store.close()
    reopened = LCMConversationStore(tmp_path / "lcm.db")
    assert reopened.has_message_client_id("acp:s1", claim)


def test_over_the_storage_cap_is_refused_never_truncated(tmp_path):
    store = LCMConversationStore(tmp_path / "lcm.db")
    cid = "z" * (CLIENT_ID_STORAGE_MAX + 1)
    store.set_message_client_id("s1", 7, cid)
    assert store.get_message_client_ids("s1") == {}
    assert not store.has_message_client_id("s1", cid)
    # and no truncated prefix was stored either
    assert not store.has_message_client_id("s1", cid[:CLIENT_ID_STORAGE_MAX])
