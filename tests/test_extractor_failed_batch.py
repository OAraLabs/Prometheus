"""A failed extraction batch holds the cursor; it never marks unmined rows as consumed.

``_call_model`` runs inside an LLMCallEnvelope with ``on_failure="return_none"``,
so a provider outage, a rate limit or a timeout comes back as ``None``.
``_process_batch`` turned that into ``(0, [])``, the same value as "the model found
nothing", and ``run_once`` then advanced the cursor over every row it had read.
Those messages were never mined and never would be: a short outage was a
permanent hole in memory.

The rule pinned here: a batch whose model call failed is not consumed. The
cursor advances only over rows before it, the pass stops there (later batches
could not be advanced past it anyway, so mining them now would mine them twice),
and the next pass retries from the failed batch. There is no retry limit: a
provider that is down for hours must not turn into skipped rows when it comes
back.

Found by sepehrtaji-dev in #832.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.memory.extractor import MemoryExtractor
from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_types import MessagePart
from prometheus.memory.store import EXTRACTOR_GLOBAL_SCOPE, MemoryStore

SESSION_A = "telegram:111"
SESSION_B = "telegram:222"


class _NullProvider:
    """Never called: ``_call_model`` is replaced below."""


@pytest.fixture
def conv(tmp_path) -> LCMConversationStore:
    return LCMConversationStore(tmp_path / "lcm.db")


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    return MemoryStore(db_path=tmp_path / "memory.db")


def _ingest(conv: LCMConversationStore, session_id: str, text: str) -> int:
    msg = MessagePart(role="user", content=text, session_id=session_id)
    conv.insert_message(msg)
    return msg.row_id


class _Model:
    """The model call, replaced at ``_call_model`` so the batch, the cursor and
    the persistence stay real. ``down`` makes it fail the way the envelope
    reports a failure: by returning ``None``."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.down = False
        self.calls = 0
        self.mined: list[str] = []  # messages seen by SUCCESSFUL calls

    async def __call__(self, prompt: str) -> str | None:
        self.calls += 1
        if self.down:
            return None
        self.mined.extend(t for t in self.texts if f"[user]: {t}" in prompt)
        return ""


def _extractor(store, conv, texts, *, batch_size=15) -> tuple[MemoryExtractor, _Model]:
    store.set_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE, 0)
    ext = MemoryExtractor(
        store=store, provider=_NullProvider(), lcm_conversation_store=conv,
        batch_size=batch_size,
    )
    model = _Model(texts)
    ext._call_model = model  # type: ignore[method-assign]
    return ext, model


def _run(ext: MemoryExtractor, session_id: str | None = None) -> None:
    asyncio.run(ext.run_once(session_id=session_id))


def test_a_failed_model_call_does_not_consume_its_batch(store, conv):
    texts = ["Ada moved to Lisbon", "Ada's sister is Grace"]
    for t in texts:
        _ingest(conv, SESSION_A, t)
    ext, model = _extractor(store, conv, texts)

    model.down = True
    _run(ext)
    assert store.get_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE) == 0
    assert store.get_extractor_cursor(SESSION_A) == 0

    model.down = False
    _run(ext)
    assert model.mined == texts, "the failed batch's messages were never retried"


def test_the_cursor_advances_up_to_the_failed_batch_and_no_further(store, conv):
    texts = [f"message {n}" for n in range(1, 6)]
    rows = [_ingest(conv, SESSION_A, t) for t in texts]
    ext, model = _extractor(store, conv, texts, batch_size=2)

    # Batch 1 (messages 1-2) succeeds, batch 2 (3-4) fails.
    original = model.__call__

    async def _fail_second(prompt):
        if model.calls == 1:
            model.down = True
        return await original(prompt)

    ext._call_model = _fail_second  # type: ignore[method-assign]
    _run(ext)

    assert model.calls == 2, "the pass kept calling the model after a batch failed"
    assert store.get_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE) == rows[1]
    assert store.get_extractor_cursor(SESSION_A) == rows[1]

    model.down = False
    ext._call_model = model  # type: ignore[method-assign]
    _run(ext)
    assert model.mined == texts, "a row was lost or mined twice across the failure"
    assert store.get_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE) == rows[-1]


def test_a_long_outage_never_turns_into_skipped_rows(store, conv):
    texts = ["Ada works at the observatory"]
    _ingest(conv, SESSION_A, texts[0])
    ext, model = _extractor(store, conv, texts)

    model.down = True
    for _ in range(24):  # twelve hours at the 30-minute cadence
        _run(ext)
    assert store.get_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE) == 0

    model.down = False
    _run(ext)
    assert model.mined == texts


def test_a_failed_flush_holds_only_its_own_session(store, conv):
    _ingest(conv, SESSION_B, "B: waiting for the sweep")
    _ingest(conv, SESSION_A, "A: flushed during an outage")
    ext, model = _extractor(store, conv, ["A: flushed during an outage"])

    model.down = True
    _run(ext, session_id=SESSION_A)
    assert store.get_extractor_cursor(SESSION_A) == 0
    assert store.get_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE) == 0


def test_an_empty_reply_is_a_success_and_consumes_the_batch(store, conv):
    # "The model found nothing" must not be confused with "the call failed",
    # or a batch with nothing to remember would be retried forever.
    texts = ["hello"]
    row = _ingest(conv, SESSION_A, texts[0])
    ext, model = _extractor(store, conv, texts)

    _run(ext)
    assert store.get_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE) == row
    _run(ext)
    assert model.calls == 1, "a consumed batch was sent to the model again"
