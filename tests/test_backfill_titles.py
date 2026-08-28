"""GRAFT-MOBILE-BRIDGE 9 — one-off title backfill for pre-Piece-7 sessions.

Acceptance, from the spec:

  1. Dry run lists untitled, eligible sessions and the title each would
     receive — and stores nothing.
  2. A real run titles them; a second run is a no-op.
  3. Interrupting mid-run and resuming completes without duplicating work
     (modelled with --limit: a capped run then a rerun).
  4. Tombstoned and single-message sessions are skipped.
  5. (Daemon responsiveness is a pacing property — sequential + --delay — not
     assertable in a unit test; the sequential shape is what these pin.)

The real-run write path is maybe_title_session itself, so absence-only
semantics (a manual rename is never overwritten) hold by construction.
"""

from __future__ import annotations

import asyncio

from prometheus.engine.session_titles import backfill_titles
from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_types import MessagePart


class _FakeProvider:
    """Streams one canned title per call and counts the calls."""

    def __init__(self, title: str) -> None:
        self._title = title
        self.calls = 0

    def stream_message(self, request):
        from prometheus.providers.base import ApiTextDeltaEvent

        self.calls += 1

        async def _gen():
            yield ApiTextDeltaEvent(text=self._title)

        return _gen()


class _BoomProvider:
    def stream_message(self, request):
        async def _gen():
            raise RuntimeError("provider down")
            yield  # pragma: no cover

        return _gen()


def _store(tmp_path) -> LCMConversationStore:
    return LCMConversationStore(tmp_path / "lcm.db")


def _exchange(store, sid: str) -> None:
    store.insert_message(MessagePart(role="user", content="plan my week", session_id=sid))
    store.insert_message(MessagePart(role="assistant", content="Here is a plan.", session_id=sid))


def _run(coro):
    return asyncio.run(coro)


def test_dry_run_reports_and_stores_nothing(tmp_path):
    store = _store(tmp_path)
    _exchange(store, "telegram:a")
    _exchange(store, "desktop:b")
    lines: list[str] = []

    counts = _run(backfill_titles(store, _FakeProvider("Weekly planning"), "m",
                                  dry_run=True, log=lines.append))

    assert counts["would_title"] == 2 and counts["titled"] == 0
    assert len(lines) == 2 and all("would title" in l and "Weekly planning" in l for l in lines)
    assert store.get_session_title("telegram:a") is None
    assert store.get_session_title("desktop:b") is None


def test_real_run_titles_and_second_run_is_a_noop(tmp_path):
    store = _store(tmp_path)
    _exchange(store, "telegram:a")
    _exchange(store, "desktop:b")
    provider = _FakeProvider("Weekly planning")

    first = _run(backfill_titles(store, provider, "m", log=lambda *_: None))
    assert first["titled"] == 2
    assert store.get_session_title("telegram:a") == "Weekly planning"

    second = _run(backfill_titles(store, provider, "m", log=lambda *_: None))
    assert second["titled"] == 0 and second["already_titled"] == 2
    assert provider.calls == 2, "a rerun makes no model calls"


def test_interrupted_run_resumes_without_duplicating_work(tmp_path):
    store = _store(tmp_path)
    _exchange(store, "telegram:a")
    _exchange(store, "desktop:b")
    provider = _FakeProvider("A name")

    capped = _run(backfill_titles(store, provider, "m", limit=1, log=lambda *_: None))
    assert capped["titled"] == 1 and capped["eligible"] == 2

    resumed = _run(backfill_titles(store, provider, "m", log=lambda *_: None))
    assert resumed["titled"] == 1 and resumed["already_titled"] == 1
    assert provider.calls == 2, "the resumed run only pays for the remainder"


def test_tombstoned_and_single_message_sessions_are_skipped(tmp_path):
    store = _store(tmp_path)
    _exchange(store, "telegram:gone")
    store.tombstone_session("telegram:gone")
    store.insert_message(MessagePart(role="user", content="just me", session_id="ios:solo"))
    provider = _FakeProvider("Never used")

    counts = _run(backfill_titles(store, provider, "m", log=lambda *_: None))

    assert provider.calls == 0
    assert counts["titled"] == 0 and counts["too_short"] == 1
    assert store.get_session_title("telegram:gone") is None
    assert store.get_session_title("ios:solo") is None


def test_manual_rename_is_never_overwritten(tmp_path):
    store = _store(tmp_path)
    _exchange(store, "telegram:named")
    store.set_session_title("telegram:named", "My name, chosen by me")

    counts = _run(backfill_titles(store, _FakeProvider("Machine name"), "m",
                                  log=lambda *_: None))

    assert counts["already_titled"] == 1
    assert store.get_session_title("telegram:named") == "My name, chosen by me"


def test_provider_failure_degrades_to_no_title(tmp_path):
    store = _store(tmp_path)
    _exchange(store, "telegram:a")

    counts = _run(backfill_titles(store, _BoomProvider(), "m", log=lambda *_: None))

    assert counts["failed"] == 1 and counts["titled"] == 0
    assert store.get_session_title("telegram:a") is None
