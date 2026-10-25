"""The purge routes and what the assistant learned from a chat.

``GET /api/sessions/{id}/purge-preview`` shows the owner what a purge would remove (the chat and
what was learned from it) without removing anything. ``POST .../purge`` then retracts what was
learned BEFORE it deletes the chat, because the chat's message ids are how a fact names its
sources and the mapping disappears with the rows. The engine is tested in
``test_purge_retraction.py``; this file is the contract at the HTTP edge.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.engine.session import SessionManager  # noqa: E402
from prometheus.memory.lcm_conversation_store import LCMConversationStore  # noqa: E402
from prometheus.memory.lcm_types import MessagePart  # noqa: E402
from prometheus.memory.store import MANUAL_SOURCE, MemoryStore  # noqa: E402
from prometheus.memory.wiki_compiler import WikiCompiler  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

DOOMED = "desktop:doomed"
KEEPER = "telegram:keeper"
SECRET = "zorblattxylo"


def _engine(store: LCMConversationStore):
    class _Engine:
        conversation_store = store

        def ingest_sync(self, session_id, role, content, turn_index=0, content_json=None,
                        provenance="user", is_trusted=True):
            m = MessagePart(role=role, content=content, session_id=session_id,
                            turn_index=turn_index, provenance=provenance, is_trusted=is_trusted)
            store.add_message(session_id, m)
            return m.message_id

    return _Engine()


def _say(lcm, session, text, mid):
    part = MessagePart(role="user", content=text)
    part.message_id = mid
    return lcm.add_message(session, part)


@pytest.fixture()
def world(tmp_path: Path) -> SimpleNamespace:
    lcm = LCMConversationStore(tmp_path / "lcm.db")
    mem = MemoryStore(db_path=tmp_path / "memory.db")
    wiki = tmp_path / "wiki"
    d1 = _say(lcm, DOOMED, f"Zephyr Corp card is {SECRET}", "a0000001")
    d2 = _say(lcm, DOOMED, "Zephyr Corp again", "a0000002")
    d3 = _say(lcm, DOOMED, "The Bluebird club meets on Fridays", "a0000003")
    k1 = _say(lcm, KEEPER, "Marlowe moved to Lisbon", "b0000001")
    only = mem.persist_memory("organization", "Zephyr Corp", f"Zephyr Corp card is {SECRET}", 0.9,
                              source_event_ids=[d1, d2])
    mem.update_memory(only, mention_count=2)
    shared = mem.persist_memory("person", "Marlowe", "Marlowe lives in Lisbon", 0.9,
                                source_event_ids=[d1, k1])
    mem.update_memory(shared, mention_count=2)
    # A /note records the source "manual"; it is tied to this chat only because it was merged into a
    # fact the extractor learned here.
    mem.persist_memory("topic", "Bluebird club", "Bluebird club meets on Fridays", 0.9,
                       source_event_ids=[d3])
    note = mem.persist_memory("note", "Bluebird club", "Bluebird club meets on Fridays", 1.0,
                              source_event_ids=[MANUAL_SOURCE], manual=True)
    compiler = WikiCompiler(store=mem, wiki_root=wiki)
    compiler.regenerate_all()

    engine = _engine(lcm)
    mgr = SessionManager()
    mgr.lcm_engine = engine
    for sid in (DOOMED, KEEPER):
        mgr.get_or_create(sid).add_user_message("hello")
    client = TestClient(create_app({}, session_mgr=mgr, lcm_engine=engine))
    client.app.state.memory_store = mem
    client.app.state.wiki_compiler = compiler
    return SimpleNamespace(client=client, lcm=lcm, mem=mem, wiki=wiki, compiler=compiler,
                           only=only, shared=shared, note=note, tmp=tmp_path)


def _wiki_bytes(wiki: Path) -> dict[str, bytes]:
    return {str(p.relative_to(wiki)): p.read_bytes() for p in sorted(wiki.rglob("*")) if p.is_file()}


def test_the_preview_shows_what_would_go_and_changes_nothing(world):
    wiki_before = _wiki_bytes(world.wiki)
    facts_before = world.mem.get_all_memories(limit=100)
    messages_before = len(world.lcm.message_ids_for_session(DOOMED))
    assert messages_before >= 2

    r = world.client.get(f"/api/sessions/{DOOMED}/purge-preview")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reversible"] is False
    assert body["messages"] == messages_before
    derived = body["derived"]
    assert derived["available"] is True
    assert [f["id"] for f in derived["facts_removed"]] == [world.only]
    assert derived["facts_removed"][0]["reason"] == "its only source is this chat"
    assert [f["id"] for f in derived["facts_kept"]] == [world.shared]
    # Zephyr Corp loses its only facts. Marlowe's fact STAYS (the other chat names him) but it had two
    # mentions, one per chat; with one left it is under the wiki's two-mention page threshold, so the
    # page goes too. The purged chat's mentions no longer prop a page up.
    assert derived["wiki_pages_removed"] == ["clients/Zephyr Corp.md", "people/Marlowe.md"]
    assert derived["totals"]["facts_removed"] == 1 and derived["truncated"] is False
    assert body["not_covered"], "the preview must say what a purge does not reach"

    # nothing moved
    assert len(world.lcm.message_ids_for_session(DOOMED)) == messages_before
    assert world.mem.get_all_memories(limit=100) == facts_before
    assert _wiki_bytes(world.wiki) == wiki_before


def test_the_preview_says_so_when_memory_is_not_running(world):
    world.client.app.state.memory_store = None
    body = world.client.get(f"/api/sessions/{DOOMED}/purge-preview").json()
    assert body["derived"]["available"] is False
    assert "memory store" in body["derived"]["reason"]


def test_purge_retracts_what_was_learned_and_then_deletes_the_chat(world):
    r = world.client.post(f"/api/sessions/{DOOMED}/purge", json={"confirm": DOOMED})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["derived"]["available"] is True
    assert body["derived"]["facts_removed"] == 1 and body["derived"]["facts_kept"] == 1
    assert body["derived"]["late_sweep"] == {"ok": True, "facts_removed": 0}
    assert body["not_covered"]

    assert world.lcm.message_ids_for_session(DOOMED) == []
    assert world.mem.get_memory(world.only) is None
    assert world.mem.get_memory(world.shared)["source_event_ids"] == ["b0000001"]
    assert not (world.wiki / "clients" / "Zephyr Corp.md").exists()
    assert SECRET not in "".join(
        p.read_text(encoding="utf-8", errors="ignore") for p in world.wiki.rglob("*.md"))


def test_purge_still_needs_the_session_named_twice(world):
    r = world.client.post(f"/api/sessions/{DOOMED}/purge", json={"confirm": "nope"})
    assert r.status_code == 400
    assert world.mem.get_memory(world.only) is not None


def test_purge_without_a_memory_store_still_purges_and_says_what_it_could_not_do(world):
    world.client.app.state.memory_store = None

    r = world.client.post(f"/api/sessions/{DOOMED}/purge", json={"confirm": DOOMED})

    assert r.status_code == 200, r.text
    derived = r.json()["derived"]
    assert derived["available"] is False and "memory store" in derived["reason"]
    assert world.lcm.message_ids_for_session(DOOMED) == []
    assert world.mem.get_memory(world.only) is not None, "nothing could retract it, and the response says so"


def test_a_failed_retraction_stops_the_purge_before_the_chat_is_touched(world, monkeypatch):
    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr("prometheus.memory.purge_derived.retract_before_purge", boom)
    messages_before = len(world.lcm.message_ids_for_session(DOOMED))

    r = world.client.post(f"/api/sessions/{DOOMED}/purge", json={"confirm": DOOMED})

    assert r.status_code == 500
    assert "nothing was purged" in r.json()["error"]
    assert len(world.lcm.message_ids_for_session(DOOMED)) == messages_before, "the chat must still be there"
    assert world.mem.get_memory(world.only) is not None


def test_a_failed_final_sweep_is_reported_not_hidden(world, monkeypatch):
    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr("prometheus.memory.purge_derived.sweep_after_purge", boom)

    r = world.client.post(f"/api/sessions/{DOOMED}/purge", json={"confirm": DOOMED})

    assert r.status_code == 200
    sweep = r.json()["derived"]["late_sweep"]
    assert sweep["ok"] is False and "sweep" in sweep["error"]
    assert world.lcm.message_ids_for_session(DOOMED) == []


def test_a_purge_that_taught_nothing_leaves_the_wiki_alone(world):
    _say(world.lcm, "slack:quiet", "just chatting", "c0000001")
    before = _wiki_bytes(world.wiki)

    r = world.client.post("/api/sessions/slack:quiet/purge", json={"confirm": "slack:quiet"})

    assert r.status_code == 200
    assert r.json()["derived"]["facts_removed"] == 0
    assert _wiki_bytes(world.wiki) == before, "no fact changed, so no page should be rewritten"


# ── notes (/note) ──────────────────────────────────────────────────────

def test_the_preview_lists_the_notes_that_would_stay(world):
    body = world.client.get(f"/api/sessions/{DOOMED}/purge-preview").json()

    derived = body["derived"]
    assert derived["forget_notes"] is False
    assert [n["id"] for n in derived["notes_kept"]] == [world.note]
    assert derived["notes_kept"][0]["fact"] == "Bluebird club meets on Fridays"
    assert derived["totals"]["notes_kept"] == 1
    assert world.note not in [f["id"] for f in derived["facts_removed"]]


def test_the_preview_can_show_the_purge_that_also_forgets_notes_and_changes_nothing(world):
    wiki_before = _wiki_bytes(world.wiki)
    facts_before = world.mem.get_all_memories(limit=100)

    body = world.client.get(f"/api/sessions/{DOOMED}/purge-preview?forget_notes=true").json()

    derived = body["derived"]
    assert derived["forget_notes"] is True
    assert derived["notes_kept"] == []
    removed = {f["id"]: f for f in derived["facts_removed"]}
    assert world.note in removed and removed[world.note]["manual"] is True
    assert world.mem.get_all_memories(limit=100) == facts_before
    assert _wiki_bytes(world.wiki) == wiki_before


def test_purge_keeps_notes_unless_asked_to_forget_them(world):
    r = world.client.post(f"/api/sessions/{DOOMED}/purge", json={"confirm": DOOMED})

    assert r.status_code == 200, r.text
    derived = r.json()["derived"]
    assert derived["notes_kept"] == 1 and derived["forget_notes"] is False
    assert world.mem.get_memory(world.note)["source_event_ids"] == [MANUAL_SOURCE]
    assert world.lcm.message_ids_for_session(DOOMED) == []


def test_purge_forgets_notes_when_the_request_says_so(world):
    r = world.client.post(f"/api/sessions/{DOOMED}/purge",
                          json={"confirm": DOOMED, "forget_notes": True})

    assert r.status_code == 200, r.text
    derived = r.json()["derived"]
    assert derived["forget_notes"] is True and derived["notes_kept"] == 0
    assert derived["facts_removed"] == 2, "the Zephyr Corp fact and the note"
    assert world.mem.get_memory(world.note) is None
    assert world.lcm.message_ids_for_session(DOOMED) == []


@pytest.mark.parametrize("value", ["yes", "true", 1, None, [True]])
def test_a_forget_notes_that_is_not_a_boolean_is_refused_before_anything_is_touched(world, value):
    messages_before = len(world.lcm.message_ids_for_session(DOOMED))

    r = world.client.post(f"/api/sessions/{DOOMED}/purge",
                          json={"confirm": DOOMED, "forget_notes": value})

    assert r.status_code == 400
    assert "forget_notes" in r.json()["error"] and "nothing was purged" in r.json()["error"]
    assert len(world.lcm.message_ids_for_session(DOOMED)) == messages_before
    assert world.mem.get_memory(world.only) is not None
    assert world.mem.get_memory(world.note) is not None
