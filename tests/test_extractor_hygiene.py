"""Extraction hygiene — machine sessions are not mined; paraphrases fold.

2026-07 wiki-quality audit: bakeoff/eval fixture chatter filed the eval
library "marshmallow" as a client organization (47 sources on the live vault
page), and people/will.md carried ~80 near-duplicate path-trivia facts.
Two extractor-layer fixes, both covered here:

  1. Machine-harness sessions (bakeoff:/coding:/eval:/gym:/smoke:/"system")
     are excluded from extraction entirely.
  2. A new fact that merely paraphrases an existing fact of the same entity
     folds into that row (mention_count + source union via persist_memory's
     exact-normalized dedup) instead of minting a new row.

Same side-effect idiom as test_extractor_provenance: the fake model mines a
fact for ANY marker it sees, so leaked chatter WOULD poison the store.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prometheus.memory.extractor import (
    MemoryExtractor,
    _fact_token_set,
    _is_machine_session,
    _near_dup_similarity,
)
from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_types import MessagePart
from prometheus.memory.store import MemoryStore

_USER_MARKER = "PIZZATOPPING_USERFACT"
_FIXTURE_MARKER = "MARSHMALLOW_FIXTUREFACT"


def _extractor(tmp_path, conv):
    store = MemoryStore(db_path=tmp_path / "memory.db")
    return store, MemoryExtractor(store, MagicMock(), lcm_conversation_store=conv)


# --------------------------------------------------------------------------- #
# 1. Machine-session exclusion
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "session_id,machine",
    [
        ("bakeoff:t02_off_by_one_length:r1:1781192853", True),
        ("coding:abc123", True),
        ("eval:golden-t3", True),
        ("gym:harvest-1", True),
        ("smoke:inject", True),
        ("system", True),
        ("telegram:8139235390", False),
        ("desktop:s4-1766869.1781103365257", False),
        ("web:1", False),
        ("6820c27f-5146-4f9a-963f-586c33f42981", False),
        ("", False),
        (None, False),
    ],
)
def test_is_machine_session(session_id, machine):
    assert _is_machine_session(session_id) is machine


async def test_machine_session_chatter_is_not_mined(tmp_path):
    conv = LCMConversationStore(db_path=tmp_path / "lcm.db")
    conv.add_message("telegram:1", MessagePart(
        role="user", content=f"My name is Will and I love {_USER_MARKER}.",
        session_id="telegram:1", turn_index=0, timestamp=1.0,
    ))
    # Harness chatter — exactly the live-vault pollution shape (bakeoff eval
    # fixtures talking about the marshmallow library).
    for i, sid in enumerate([
        "bakeoff:t02_off_by_one_length:r1:1781192853",
        "coding:task-9",
        "gym:harvest-1",
        "system",
    ]):
        conv.add_message(sid, MessagePart(
            role="user",
            content=f"fix src/marshmallow/validate.py {_FIXTURE_MARKER}",
            session_id=sid, turn_index=0, timestamp=2.0 + i,
        ))

    store, extractor = _extractor(tmp_path, conv)
    captured: dict[str, str] = {}

    async def fake_call_model(prompt: str) -> str:
        captured["prompt"] = prompt
        facts: list[str] = []
        if _USER_MARKER in prompt:
            facts.append(
                '{"entity_type": "person", "entity_name": "Will",'
                f' "fact": "loves {_USER_MARKER}", "confidence": 0.9}}'
            )
        if _FIXTURE_MARKER in prompt:
            facts.append(
                '{"entity_type": "organization", "entity_name": "marshmallow",'
                f' "fact": "{_FIXTURE_MARKER}", "confidence": 0.9}}'
            )
        return "\n".join(facts)

    extractor._call_model = fake_call_model
    count, facts = await extractor.run_once()

    # The genuine user turn was mined...
    assert count == 1, f"expected exactly the user fact, got {facts}"
    assert _USER_MARKER in captured["prompt"]
    # ...and no harness turn ever reached the extractor.
    assert _FIXTURE_MARKER not in captured["prompt"]
    blob = " ".join(str(m) for m in store.get_all_memories())
    assert _FIXTURE_MARKER not in blob
    assert "marshmallow" not in blob

    # The watermark advanced past the skipped machine rows too — they are not
    # re-scanned on the next cadence.
    count2, _ = await extractor.run_once()
    assert count2 == 0


async def test_supervisor_provenance_still_excluded_in_chat_session(tmp_path):
    """The provenance='supervisor' exclusion is independent of the machine-
    session gate — pinned here on a chat session because the coding-session
    variant (test_coding_supervise) is now short-circuited by that gate."""
    conv = LCMConversationStore(db_path=tmp_path / "lcm.db")
    conv.add_message("telegram:1", MessagePart(
        role="user", content=f"I love {_USER_MARKER}.",
        session_id="telegram:1", turn_index=0, provenance="user", is_trusted=True,
    ))
    conv.add_message("telegram:1", MessagePart(
        role="user", content="SUPERVISOR_STEER refactor the parser first",
        session_id="telegram:1", turn_index=1, provenance="supervisor", is_trusted=True,
    ))

    store, extractor = _extractor(tmp_path, conv)
    captured: dict[str, str] = {}

    async def fake_call_model(prompt: str) -> str:
        captured["prompt"] = prompt
        return (
            '{"entity_type": "person", "entity_name": "Will",'
            f' "fact": "loves {_USER_MARKER}", "confidence": 0.9}}'
        )

    extractor._call_model = fake_call_model
    await extractor.run_once()

    assert _USER_MARKER in captured["prompt"]
    assert "SUPERVISOR_STEER" not in captured["prompt"]


# --------------------------------------------------------------------------- #
# 2. Near-duplicate fact folding
# --------------------------------------------------------------------------- #

async def test_paraphrase_folds_into_existing_row(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory.db")
    seed_fact = "The user's local path prefix is /home/will/"
    store.persist_memory(
        entity_type="person", entity_name="Will", fact=seed_fact,
        confidence=0.8, source_event_ids=["seed-1"],
    )

    conv = LCMConversationStore(db_path=tmp_path / "lcm.db")
    conv.add_message("telegram:1", MessagePart(
        role="user", content="ls /home/will/ please",
        session_id="telegram:1", turn_index=0,
    ))
    extractor = MemoryExtractor(store, MagicMock(), lcm_conversation_store=conv)

    async def fake_call_model(prompt: str) -> str:
        # A lexical paraphrase of the seeded fact — the exact-normalized dedup
        # in persist_memory would NOT match this, so pre-fix it minted row #2.
        return (
            '{"entity_type": "person", "entity_name": "Will",'
            ' "fact": "The user\'s local path includes the directory /home/will/",'
            ' "confidence": 0.9}'
        )

    extractor._call_model = fake_call_model
    count, _ = await extractor.run_once()
    assert count == 1

    mems = [m for m in store.get_all_memories() if m["entity_name"] == "Will"]
    assert len(mems) == 1, f"paraphrase must fold, got rows: {mems}"
    mem = mems[0]
    assert mem["fact"] == seed_fact          # the stored text stays canonical
    assert mem["mention_count"] == 2         # the re-mention was counted...
    assert "seed-1" in mem["source_event_ids"]
    assert len(mem["source_event_ids"]) == 2  # ...and its provenance unioned in


async def test_genuinely_different_fact_persists_as_new_row(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.persist_memory(
        entity_type="person", entity_name="Will",
        fact="Will works at Capstone Recruiting",
        confidence=0.9, source_event_ids=["seed-1"],
    )

    conv = LCMConversationStore(db_path=tmp_path / "lcm.db")
    conv.add_message("telegram:1", MessagePart(
        role="user", content="I live in Salt Lake City",
        session_id="telegram:1", turn_index=0,
    ))
    extractor = MemoryExtractor(store, MagicMock(), lcm_conversation_store=conv)

    async def fake_call_model(prompt: str) -> str:
        return (
            '{"entity_type": "person", "entity_name": "Will",'
            ' "fact": "Will lives in Salt Lake City", "confidence": 0.9}'
        )

    extractor._call_model = fake_call_model
    await extractor.run_once()

    mems = [m for m in store.get_all_memories() if m["entity_name"] == "Will"]
    assert len(mems) == 2, "a genuinely new fact must NOT be folded away"


# Threshold pins on real pairs from the live people/will.md pollution — if a
# tunable change flips one of these, this is the test that says so.
@pytest.mark.parametrize(
    "a,b",
    [
        (
            "The file path contains the username 'will'.",
            "The file path contains the username 'will', suggesting the "
            "repository is located in his home directory.",
        ),
        (
            "The path includes the name 'will'",
            "The file path indicates the user's name or directory name is will",
        ),
        (
            "The user's local path prefix is /home/will/",
            "The user's local path includes the directory /home/will/",
        ),
    ],
)
def test_near_dup_pairs_fold(a, b):
    sim = _near_dup_similarity(_fact_token_set(a), _fact_token_set(b))
    assert sim >= 0.75, f"{a!r} vs {b!r}: {sim:.2f}"


@pytest.mark.parametrize(
    "a,b",
    [
        (
            "The user's home directory contains the Prometheus repository.",
            "The user's home directory contains a project named bakeoff-harness.",
        ),
        (
            "Will works at Capstone Recruiting",
            "Will lives in Salt Lake City",
        ),
        (
            "Will uses uv to run Python in Prometheus",
            "Will prefers pytest for running the test suite",
        ),
    ],
)
def test_distinct_pairs_do_not_fold(a, b):
    sim = _near_dup_similarity(_fact_token_set(a), _fact_token_set(b))
    assert sim < 0.75, f"{a!r} vs {b!r}: {sim:.2f}"


def test_empty_token_set_never_folds():
    # A fact made entirely of scaffolding must not fold into anything.
    assert _near_dup_similarity(_fact_token_set("is the of a"), _fact_token_set("x")) == 0.0
