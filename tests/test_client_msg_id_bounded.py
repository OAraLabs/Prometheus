"""`client_msg_id` on `POST /api/chat/send` is bounded where the daemon REMEMBERS it (review of #912, low).

The send dedupe keeps each accepted (session, client_msg_id) in an in-memory table (4096 entries) so a retry that
arrives while the first send is in flight is caught. The id is the client's own string, so one device could pin a lot of
memory with long ids. An id over `CLIENT_MSG_ID_MAX` (200 characters, after trimming) is therefore not remembered:
it is dispatched as before and never deduplicated (a real client's id is a UUID, 36 characters). At the limit it is
still deduplicated.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from tests.test_chat_send_dedupes_client_msg_id import _client, _send  # noqa: E402

LIMIT = 200


def test_an_id_at_the_limit_is_still_deduplicated():
    client, bridge = _client()
    cid = "c" * LIMIT
    _send(client, cid=cid)
    again = _send(client, cid=cid)
    assert again.json().get("duplicate") is True and len(bridge.calls) == 1


def test_an_id_over_the_limit_is_dispatched_every_time_and_never_remembered():
    client, bridge = _client()
    cid = "c" * (LIMIT + 1)
    first, again = _send(client, cid=cid), _send(client, cid=cid)
    assert first.status_code == 200 and again.status_code == 200
    assert "duplicate" not in first.json() and "duplicate" not in again.json()
    assert len(bridge.calls) == 2
    assert [call[2] for call in bridge.calls] == [cid, cid]          # it still reaches the bridge as sent


def test_the_limit_is_on_the_trimmed_id():
    client, bridge = _client()
    padded = "  " + "c" * LIMIT + "  "
    _send(client, cid=padded)
    assert _send(client, cid=padded).json().get("duplicate") is True and len(bridge.calls) == 1


def test_the_limit_is_a_named_constant_the_guide_states():
    from pathlib import Path

    import prometheus.web.server as server

    assert server.CLIENT_MSG_ID_MAX == LIMIT
    guide = (Path(__file__).resolve().parent.parent / "docs" / "guide" / "api.md").read_text(encoding="utf-8")
    assert "200 characters" in guide and "client_msg_id" in guide
