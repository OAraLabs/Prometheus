"""Session display titles (GRAFT-MOBILE-BRIDGE 7).

A session id is <gateway>:<uuid> and carries no name anywhere in the daemon —
usable on a wide desktop sidebar, unusable as a phone list. These pin the
title surface end to end:

  * store: set/get/clear round-trip, and list_sessions carries the title;
  * REST: GET /api/sessions exposes ``title`` (null until one exists),
    PUT /api/sessions/{id}/title renames, clears, validates, and clips;
  * generation: clip_title obeys the ≤48-char word-boundary contract,
    first_exchange finds the first real user→assistant pair, and
    maybe_title_session fills ABSENCE ONLY — a manual rename is never
    overwritten, and a failing provider degrades to "no title".
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.engine.session_titles import (
    MAX_TITLE_CHARS,
    clip_title,
    first_exchange,
    maybe_title_session,
)
from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_types import MessagePart


def _store(tmp_path) -> LCMConversationStore:
    return LCMConversationStore(tmp_path / "lcm.db")


def _seed(store: LCMConversationStore, sid: str) -> None:
    store.insert_message(MessagePart(role="user", content="hello", session_id=sid))


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


def test_title_round_trip_and_clear(tmp_path):
    store = _store(tmp_path)
    assert store.get_session_title("s1") is None
    store.set_session_title("s1", "Fix the build")
    assert store.get_session_title("s1") == "Fix the build"
    store.set_session_title("s1", "Fix the build, again")
    assert store.get_session_title("s1") == "Fix the build, again"
    # Blank clears the row — present/absent stays a clean signal.
    store.set_session_title("s1", "   ")
    assert store.get_session_title("s1") is None


def test_list_sessions_carries_the_title(tmp_path):
    store = _store(tmp_path)
    _seed(store, "telegram:1")
    _seed(store, "desktop:2")
    store.set_session_title("telegram:1", "Grocery planning")

    rows = {r["session_id"]: r for r in store.list_sessions()}
    assert rows["telegram:1"]["title"] == "Grocery planning"
    assert rows["desktop:2"]["title"] is None


# --------------------------------------------------------------------------- #
# REST
# --------------------------------------------------------------------------- #


@pytest.fixture()
def client_and_store(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from prometheus.web.server import create_app

    store = _store(tmp_path)

    class _Lcm:
        conversation_store = store

    app = create_app({})
    app.state.lcm_engine = _Lcm()
    return TestClient(app), store


def test_get_sessions_exposes_title(client_and_store):
    client, store = client_and_store
    _seed(store, "desktop:t")
    store.set_session_title("desktop:t", "Reactor maintenance")

    rows = client.get("/api/sessions").json()
    row = next(r for r in rows if r["session_id"] == "desktop:t")
    assert row["title"] == "Reactor maintenance"


def test_get_sessions_title_is_null_until_one_exists(client_and_store):
    client, store = client_and_store
    _seed(store, "desktop:untitled")
    rows = client.get("/api/sessions").json()
    row = next(r for r in rows if r["session_id"] == "desktop:untitled")
    assert row["title"] is None


def test_put_title_renames_and_clears(client_and_store):
    client, store = client_and_store
    _seed(store, "desktop:r")

    resp = client.put("/api/sessions/desktop:r/title", json={"title": "  Ship it  "})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "session_id": "desktop:r", "title": "Ship it"}
    assert store.get_session_title("desktop:r") == "Ship it"

    # Empty title = reset to automatic (clears the row).
    resp = client.put("/api/sessions/desktop:r/title", json={"title": ""})
    assert resp.status_code == 200
    assert resp.json()["title"] is None
    assert store.get_session_title("desktop:r") is None


def test_put_title_clips_to_the_contract(client_and_store):
    client, store = client_and_store
    long = "a word " * 30
    resp = client.put("/api/sessions/desktop:c/title", json={"title": long})
    stored = store.get_session_title("desktop:c")
    assert resp.status_code == 200
    assert stored is not None and len(stored) <= MAX_TITLE_CHARS + 1  # +1 for the ellipsis


def test_put_title_rejects_non_string(client_and_store):
    client, _ = client_and_store
    assert client.put("/api/sessions/x/title", json={"title": 7}).status_code == 400


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def test_clip_title_word_boundary_and_quotes():
    assert clip_title('  "Fix   the build"  ') == "Fix the build"
    long = "planning the greenhouse irrigation schedule for the spring beds"
    clipped = clip_title(long)
    assert len(clipped) <= MAX_TITLE_CHARS + 1
    assert clipped.endswith("…")
    assert not clipped[:-1].endswith(" ")


class _Msg:
    def __init__(self, role: str, text: str) -> None:
        self.role = role
        self.text = text


def test_first_exchange_skips_empty_and_tool_noise():
    msgs = [
        _Msg("user", ""),               # empty send — not the opener
        _Msg("user", "plan my week"),
        _Msg("assistant", ""),          # tool_use-only turn (text is '')
        _Msg("assistant", "Here is a plan."),
    ]
    assert first_exchange(msgs) == ("plan my week", "Here is a plan.")
    assert first_exchange([_Msg("user", "hi")]) is None  # no reply yet


class _FakeProvider:
    """Streams one complete-message event carrying the canned title."""

    def __init__(self, title: str) -> None:
        self._title = title

    def stream_message(self, request):
        from prometheus.providers.base import ApiTextDeltaEvent

        async def _gen():
            yield ApiTextDeltaEvent(text=self._title)

        return _gen()


class _BoomProvider:
    def stream_message(self, request):
        async def _gen():
            raise RuntimeError("provider down")
            yield  # pragma: no cover

        return _gen()


def _run(coro):
    return asyncio.run(coro)


def test_generation_fills_absence(tmp_path):
    store = _store(tmp_path)
    msgs = [_Msg("user", "plan my week"), _Msg("assistant", "Here is a plan.")]
    _run(maybe_title_session(store, _FakeProvider("Weekly planning"), "m", "s1", msgs))
    assert store.get_session_title("s1") == "Weekly planning"


def test_generation_never_overwrites(tmp_path):
    store = _store(tmp_path)
    store.set_session_title("s1", "My name, chosen by me")
    msgs = [_Msg("user", "x"), _Msg("assistant", "y")]
    _run(maybe_title_session(store, _FakeProvider("Machine name"), "m", "s1", msgs))
    assert store.get_session_title("s1") == "My name, chosen by me"


def test_generation_failure_degrades_to_no_title(tmp_path):
    store = _store(tmp_path)
    msgs = [_Msg("user", "x"), _Msg("assistant", "y")]
    _run(maybe_title_session(store, _BoomProvider(), "m", "s1", msgs))
    assert store.get_session_title("s1") is None


def test_generation_skips_sessions_with_no_exchange(tmp_path):
    store = _store(tmp_path)
    _run(maybe_title_session(store, _FakeProvider("nope"), "m", "s1",
                             [_Msg("user", "only me")]))
    assert store.get_session_title("s1") is None
