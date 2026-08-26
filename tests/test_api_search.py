"""GET/POST /api/search — global conversation search (branch feat/api-search).

Frozen-spec assertions (beacon-search-spec.md v2): two grouped scopes, never
cross-rank-merged; message hits carry the DURABLE ROWID as message_id (the wire
cursor contract); summary hits carry anchor UUIDs; FTS5-operator input must not
500; min-length/limit clamped at the endpoint, not just the client.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.memory.lcm_conversation_store import LCMConversationStore  # noqa: E402
from prometheus.memory.lcm_summary_store import LCMSummaryStore  # noqa: E402
from prometheus.memory.lcm_types import MessagePart, SummaryNode  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402


class _Engine:
    """Minimal lcm_engine double exposing the two stores the endpoint touches."""

    def __init__(self, conv: LCMConversationStore, summ: LCMSummaryStore):
        self.conversation_store = conv
        self.summary_store = summ


def _client(tmp_path, token: str = ""):
    conv = LCMConversationStore(tmp_path / "lcm.db")
    summ = LCMSummaryStore(tmp_path / "lcm.db")

    # Seed two sessions with distinct vocabulary.
    for sid, phrase in (("telegram:123", "kling key rotation"), ("cli:42", "beacon search spec")):
        m = MessagePart(role="user", content=f"talking about the {phrase} today", session_id=sid)
        conv.insert_message(m)
        node = SummaryNode(
            source_message_ids=[m.message_id],
            summary_text=f"summary covering {phrase}",
            depth=0,
        )
        summ.insert_summary(node, session_id=sid)

    cfg = {"web": {"api_token": token}} if token else {}
    client = TestClient(create_app(cfg, lcm_engine=_Engine(conv, summ)))
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client, conv, headers


def test_both_scopes_grouped(tmp_path):
    client, _, _ = _client(tmp_path)
    r = client.post("/api/search", json={"q": "kling"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"query", "returned", "messages", "summaries"}
    assert body["returned"] == 2
    assert len(body["messages"]) == 1
    assert len(body["summaries"]) == 1
    # No cross-scope merging: kind is uniform within each bucket.
    assert all(h["kind"] == "message" for h in body["messages"])
    assert all(h["kind"] == "summary" for h in body["summaries"])
    msg = body["messages"][0]
    summ = body["summaries"][0]
    assert msg["session_id"] == "telegram:123"
    assert "⟦" in msg["snippet"] and "⟧" in msg["snippet"]
    assert summ["anchor_message_ids"], "summary hit must carry anchor UUIDs"
    assert summ["summary_text"] == "summary covering kling key rotation"
    # BM25 passthrough: negative, ascending=better.
    assert msg["score"] < 0


def test_message_id_is_the_durable_rowid(tmp_path):
    """The wire cursor contract: returned message_id must exist as a rowid in
    the SAME session — guards the silent-misnavigation bug (uuid vs rowid)."""
    client, conv, _ = _client(tmp_path)
    body = client.post("/api/search", json={"q": "beacon"}).json()
    hit = body["messages"][0]
    rowid = int(hit["message_id"])
    rows = conv._conn.execute(
        "SELECT session_id FROM lcm_messages WHERE rowid = ?", (rowid,)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["session_id"] == hit["session_id"]
    # And it is NOT the uuid — the two are different id spaces.
    assert hit["message_id"] != hit["uuid"]


def test_session_scope(tmp_path):
    client, _, _ = _client(tmp_path)
    body = client.post(
        "/api/search", json={"q": "kling", "session_id": "cli:42"}
    ).json()
    assert body["returned"] == 0


def test_scope_filter(tmp_path):
    client, _, _ = _client(tmp_path)
    for scope, n_msg, n_sum in (
        ("messages", 1, 0),
        ("summaries", 0, 1),
        ("both", 1, 1),
    ):
        body = client.post("/api/search", json={"q": "kling", "scope": scope}).json()
        assert len(body["messages"]) == n_msg, scope
        assert len(body["summaries"]) == n_sum, scope


def test_limit_clamped_and_respected(tmp_path):
    client, conv, _ = _client(tmp_path)
    for i in range(5):
        conv.insert_message(
            MessagePart(role="assistant", content=f"zephyr log line {i}", session_id="cli:42")
        )
    body = client.post("/api/search", json={"q": "zephyr", "scope": "messages", "limit": 3}).json()
    assert len(body["messages"]) == 3
    # Over-max clamps to 50, not error and not >50.
    body = client.post(
        "/api/search", json={"q": "zephyr", "scope": "messages", "limit": 500}
    ).json()
    assert len(body["messages"]) == 5  # only 5 exist; 500 clamped, no error


def test_min_length_rejected(tmp_path):
    client, _, _ = _client(tmp_path)
    assert client.post("/api/search", json={"q": "kl"}).status_code == 400
    assert client.post("/api/search", json={"q": ""}).status_code == 400


def test_empty_and_punctuation_query_zero_results(tmp_path):
    client, _, _ = _client(tmp_path)
    # 3+ chars of pure punctuation passes min-length but sanitizes to no-match.
    body = client.post("/api/search", json={"q": "!!!"}).json()
    assert body["returned"] == 0
    assert body["messages"] == [] and body["summaries"] == []


def test_fts5_operators_never_500(tmp_path):
    """Users type quotes, stars, AND, dashes — that is the contract. The
    endpoint must not 500 and must not leak raw operator semantics."""
    client, _, _ = _client(tmp_path)
    for q in ('"kling', "kling*", "kling AND rotation", "-kling", "kling OR beacon", 'a"b'):
        r = client.post("/api/search", json={"q": q})
        assert r.status_code == 200, f"query {q!r} → {r.status_code}"


def test_invalid_scope_rejected(tmp_path):
    client, _, _ = _client(tmp_path)
    assert client.post("/api/search", json={"q": "kling", "scope": "bogus"}).status_code == 400


def test_auth_rejection(tmp_path):
    client, _, _ = _client(tmp_path, token="sekrit")
    assert client.post("/api/search", json={"q": "kling"}).status_code == 401
    assert (
        client.post(
            "/api/search", json={"q": "kling"}, headers={"Authorization": "Bearer sekrit"}
        ).status_code
        == 200
    )


def test_post_is_the_only_verb(tmp_path):
    """GET is deliberately gone — a querystring lands in access logs, and
    search terms over private conversations are sensitive. POST-only is the
    decision, not an oversight."""
    client, _, _ = _client(tmp_path)
    assert client.get("/api/search", params={"q": "kling"}).status_code in (404, 405)
    assert client.post("/api/search", json={"q": "kling"}).status_code == 200


def test_malformed_json_is_400_not_500(tmp_path):
    """The client path is POST; a malformed body must 400, not raise to a 500."""
    client, _, _ = _client(tmp_path)
    assert client.post("/api/search", content=b"{not json", headers={"Content-Type": "application/json"}).status_code == 400
    assert client.post("/api/search", json=["a", "list"]).status_code == 400  # not an object


def test_post_validates_body(tmp_path):
    client, _, _ = _client(tmp_path)
    assert client.post("/api/search", json={"q": "kl"}).status_code == 400
    assert client.post("/api/search", json={"q": "kling", "limit": "nope"}).status_code == 400


def test_no_lcm_engine_503(tmp_path):
    client = TestClient(create_app({}))
    assert client.post("/api/search", json={"q": "kling"}).status_code == 503
