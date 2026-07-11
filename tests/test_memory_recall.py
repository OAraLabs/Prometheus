"""Passive memory recall (MEMORY-3 follow-up).

Closes the "memory/wiki is write-only to the model" gap
(MEMORY-AUDIT-20260617 §3): stored facts relevant to the latest user
message ride the per-run system prompt as a request-only
"# Recalled memory" section.

Contract pinned here:
- keyword/FTS matching finds relevant facts from whole-sentence queries
  (the match_any=OR path — implicit AND over a full sentence never matches);
- manual (/note) facts outrank ambient ones; near-duplicate facts collapse;
  one entity cannot flood the block; fact and char budgets hold;
- recall is REQUEST-ONLY: the block reaches the provider's system prompt but
  never the durable message list (else the extractor re-ingests its own
  output — feedback loop);
- fail-open: a raising store/recall never blocks the turn.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.engine.agent_loop import AgentLoop, LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.memory.recall import MemoryRecall, RecallConfig
from prometheus.memory.store import MemoryStore
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiTextDeltaEvent,
    ModelProvider,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _store(tmp_path) -> MemoryStore:
    return MemoryStore(db_path=tmp_path / "memory.db")


def _persist(store, entity, fact, *, etype="concept", confidence=0.9, manual=False):
    return store.persist_memory(
        etype,
        entity,
        fact,
        confidence,
        source_event_ids=["test-evt"],
        manual=manual,
    )


class _RecordingProvider(ModelProvider):
    """One text reply; records every request's system prompt."""

    def __init__(self) -> None:
        self.system_prompts: list[str] = []

    async def stream_message(self, request):  # noqa: ANN001
        self.system_prompts.append(request.system_prompt or "")
        msg = ConversationMessage(role="assistant", content=[TextBlock(text="ok")])
        yield ApiTextDeltaEvent(text="ok")
        yield ApiMessageCompleteEvent(
            message=msg,
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


def _drive(ctx: LoopContext, messages: list[ConversationMessage]) -> None:
    async def _run() -> None:
        async for _ in run_loop(ctx, messages):
            pass

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# RecallConfig
# ---------------------------------------------------------------------------

def test_config_defaults_and_overrides():
    assert RecallConfig.from_config(None) == RecallConfig()
    assert RecallConfig.from_config({}) == RecallConfig()

    cfg = RecallConfig.from_config(
        {"memory": {"recall": {"enabled": False, "max_facts": 3, "max_chars": 400}}}
    )
    assert cfg.enabled is False
    assert cfg.max_facts == 3
    assert cfg.max_chars == 400
    # untouched knobs keep defaults
    assert cfg.min_confidence == RecallConfig().min_confidence


def test_config_tolerates_null_sections():
    # YAML "memory:" with nothing under it parses to None — must not raise.
    assert RecallConfig.from_config({"memory": None}) == RecallConfig()
    assert RecallConfig.from_config({"memory": {"recall": None}}) == RecallConfig()


# ---------------------------------------------------------------------------
# MemoryRecall behavior against a real store
# ---------------------------------------------------------------------------

def test_recall_matches_whole_sentence_query(tmp_path):
    """A full user sentence (stopwords, punctuation) matches a stored fact.

    This is the match_any=OR contract: implicit-AND FTS over the sentence
    would require every token to appear in one fact and return nothing.
    """
    store = _store(tmp_path)
    _persist(store, "beacon", "Beacon is the desktop dashboard for Prometheus.")
    _persist(store, "kraken", "Kraken is an unrelated codename.")

    recall = MemoryRecall(store=store)
    block = recall.recall("hey, can you restart the beacon app for me please?")

    assert block.startswith("# Recalled memory")
    assert "desktop dashboard" in block
    assert "Kraken" not in block


def test_recall_empty_cases(tmp_path):
    store = _store(tmp_path)
    recall = MemoryRecall(store=store)
    assert recall.recall("") == ""                    # blank message
    assert recall.recall("can you do the thing") == ""  # empty store
    _persist(store, "beacon", "Beacon is the dashboard.")
    assert recall.recall("of the and to") == ""       # stopword-only
    assert recall.recall("zzzunmatchable") == ""      # no FTS hit


def test_recall_disabled_returns_empty(tmp_path):
    store = _store(tmp_path)
    _persist(store, "beacon", "Beacon is the dashboard.")
    recall = MemoryRecall(store=store, config=RecallConfig(enabled=False))
    assert recall.recall("tell me about beacon") == ""


def test_manual_facts_rank_first(tmp_path):
    store = _store(tmp_path)
    _persist(store, "updates", "The user posts updates in the standup channel.")
    _persist(
        store,
        "updates",
        "I prefer concise technical updates over long explanations",
        manual=True,
    )

    block = MemoryRecall(store=store).recall("write my updates summary")
    lines = [l for l in block.splitlines() if l.startswith("- ")]
    assert lines and lines[0].startswith("- [user note] ")
    assert "concise technical updates" in lines[0]


def test_near_duplicate_facts_collapse(tmp_path):
    store = _store(tmp_path)
    # Different entities so store-level dedup (per-entity) doesn't collapse
    # them first — recall-level dedup must.
    _persist(store, "will", "The user's username is will.")
    _persist(store, "will-account", "the user's username is WILL")

    block = MemoryRecall(store=store).recall("what is the username again")
    assert block.lower().count("username is will") == 1


def test_per_entity_cap_diversifies(tmp_path):
    store = _store(tmp_path)
    for i in range(5):
        _persist(store, "marshmallow", f"Marshmallow fact number {i} about validate.")
    _persist(store, "prometheus", "Prometheus is the daemon under marshmallow tests.")

    cfg = RecallConfig(per_entity_cap=2, max_facts=6)
    block = MemoryRecall(store=store, config=cfg).recall(
        "tell me about marshmallow validate"
    )
    entity_lines = [l for l in block.splitlines() if l.startswith("- marshmallow:")]
    assert len(entity_lines) == 2


def test_budgets_hold(tmp_path):
    store = _store(tmp_path)
    for i in range(20):
        _persist(store, f"entity{i}", f"Prometheus deployment fact {i} " + "x" * 40)

    cfg = RecallConfig(max_facts=4, max_chars=600)
    block = MemoryRecall(store=store, config=cfg).recall("prometheus deployment")
    assert len(block) <= 600
    assert sum(1 for l in block.splitlines() if l.startswith("- ")) <= 4


def test_min_confidence_filters(tmp_path):
    store = _store(tmp_path)
    _persist(store, "rumor", "A low-confidence rumor about the deploy.", confidence=0.2)
    block = MemoryRecall(store=store).recall("what about the deploy rumor")
    assert block == ""


def test_recall_fails_open_on_store_error():
    class _BrokenStore:
        def search_memories(self, **kwargs):  # noqa: ANN003
            raise RuntimeError("db is toast")

    assert MemoryRecall(store=_BrokenStore()).recall("anything at all") == ""


# ---------------------------------------------------------------------------
# store: match_any
# ---------------------------------------------------------------------------

def test_search_memories_match_any_or_semantics(tmp_path):
    store = _store(tmp_path)
    _persist(store, "beacon", "Beacon is the desktop dashboard.")

    # Implicit AND over sentence tokens: no single fact holds them all.
    assert store.search_memories(query="beacon unrelatedword") == []
    # OR mode: any token may match.
    hits = store.search_memories(query="beacon unrelatedword", match_any=True)
    assert len(hits) == 1
    # Punctuation-only stays safe in both modes.
    assert store.search_memories(query="?!.", match_any=True) == []


# ---------------------------------------------------------------------------
# store: FTS5 external-content integrity (the live-db corruption class)
# ---------------------------------------------------------------------------

def _make_legacy_corrupt_db(path) -> None:
    """Replicate a pre-trigger production DB with a corrupted FTS index.

    The old store hand-rolled FTS maintenance: INSERT without rowid (mints
    index rows whose rowids drift from the content table) and plain DELETE
    (never removes index entries). Both leave the index referencing content
    rowids that don't exist → "fts5: missing row N from content table" on
    MATCH. Discovered live on ~/.prometheus/memory.db (2026-07-10).
    """
    import sqlite3

    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY, entity_type TEXT NOT NULL,
            entity_name TEXT NOT NULL, relationship TEXT NOT NULL,
            fact TEXT NOT NULL, confidence REAL NOT NULL DEFAULT 0.5,
            source_event_ids TEXT NOT NULL DEFAULT '[]',
            last_mentioned REAL NOT NULL, mention_count INTEGER NOT NULL DEFAULT 1,
            tags TEXT NOT NULL DEFAULT '[]', timestamp REAL NOT NULL,
            manual INTEGER NOT NULL DEFAULT 0
        );
        CREATE VIRTUAL TABLE memories_fts USING fts5(
            id UNINDEXED, entity_name, fact,
            content='memories', content_rowid='rowid'
        );
        CREATE TABLE messages (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT NOT NULL, timestamp REAL NOT NULL,
            compressed INTEGER NOT NULL DEFAULT 0
        );
        CREATE VIRTUAL TABLE messages_fts USING fts5(
            id UNINDEXED, content, content='messages', content_rowid='rowid'
        );
        CREATE TABLE summaries (
            id TEXT PRIMARY KEY, source_message_ids TEXT NOT NULL DEFAULT '[]',
            summary_text TEXT NOT NULL, level INTEGER NOT NULL DEFAULT 1,
            timestamp REAL NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO memories (id, entity_type, entity_name, relationship, fact,"
        " confidence, last_mentioned, timestamp)"
        " VALUES ('aaa', 'concept', 'beacon', 'fact',"
        " 'Beacon is the desktop dashboard.', 0.9, 1.0, 1.0)"
    )
    # Old buggy pattern: no rowid → FTS assigns its own; then a second write
    # for a row that was later deleted from the content table.
    conn.execute(
        "INSERT INTO memories_fts (id, entity_name, fact)"
        " VALUES ('aaa', 'beacon', 'Beacon is the desktop dashboard.')"
    )
    conn.execute(
        "INSERT INTO memories_fts (id, entity_name, fact)"
        " VALUES ('bbb', 'ghost', 'An orphaned index entry with no content row.')"
    )
    conn.commit()
    conn.close()


def test_legacy_corrupt_fts_raises_without_migration(tmp_path):
    """Prove the fixture reproduces the live failure mode."""
    import sqlite3

    db = tmp_path / "legacy.db"
    _make_legacy_corrupt_db(db)
    conn = sqlite3.connect(db)
    with pytest.raises(sqlite3.DatabaseError, match="missing row|malformed"):
        conn.execute(
            "SELECT m.id FROM memories m JOIN memories_fts fts ON m.id = fts.id"
            " WHERE memories_fts MATCH '\"ghost\"'"
        ).fetchall()
    conn.close()


def test_store_open_heals_legacy_corruption(tmp_path):
    """Opening the store migrates: rebuild + triggers → FTS search works."""
    db = tmp_path / "legacy.db"
    _make_legacy_corrupt_db(db)

    store = MemoryStore(db_path=db)
    hits = store.search_memories(query="beacon dashboard", match_any=True)
    assert [h["id"] for h in hits] == ["aaa"]
    # Orphaned index entry is gone after the rebuild.
    assert store.search_memories(query="ghost", match_any=True) == []
    # Migration is one-shot and snapshotted the non-empty legacy DB.
    import sqlite3

    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    conn.close()
    assert list(tmp_path.glob("legacy.db.backup-*")), "expected pre-migration snapshot"


def test_fresh_store_skips_snapshot(tmp_path):
    MemoryStore(db_path=tmp_path / "fresh.db")
    assert not list(tmp_path.glob("fresh.db.backup-*"))


def test_triggers_keep_fts_consistent_through_lifecycle(tmp_path):
    """insert → update → delete → dedupe never desyncs the index."""
    store = _store(tmp_path)
    mid = _persist(store, "beacon", "Beacon is the desktop dashboard.")

    store.update_memory(mid, fact="Beacon is the Electron control panel.")
    assert store.search_memories(query="electron", match_any=True)
    assert store.search_memories(query="dashboard", match_any=True) == []

    store.delete_memory(mid)
    assert store.search_memories(query="electron", match_any=True) == []

    # dedupe_existing deletes losers via raw grouping — triggers must follow.
    _persist(store, "will", "The username is will.")
    _persist(store, "will", "the username is WILL")  # normalized-equal later
    store.dedupe_existing()
    hits = store.search_memories(query="username", match_any=True)
    assert len(hits) == 1


# ---------------------------------------------------------------------------
# run_loop integration: request-only injection
# ---------------------------------------------------------------------------

def test_run_loop_injects_recall_request_only(tmp_path):
    store = _store(tmp_path)
    _persist(store, "beacon", "Beacon is the desktop dashboard for Prometheus.")

    provider = _RecordingProvider()
    ctx = LoopContext(
        provider=provider,
        model="test",
        system_prompt="BASE PROMPT",
        max_tokens=64,
        memory_recall=MemoryRecall(store=store),
    )
    messages = [ConversationMessage.from_user_text("what is beacon?")]
    _drive(ctx, messages)

    assert provider.system_prompts, "provider was never called"
    assert "# Recalled memory" in provider.system_prompts[0]
    assert provider.system_prompts[0].startswith("BASE PROMPT")
    # REQUEST-ONLY: durable history and the shared context prompt stay clean.
    assert all("# Recalled memory" not in (m.text or "") for m in messages)
    assert "# Recalled memory" not in ctx.system_prompt


def test_run_loop_without_recall_is_byte_identical(tmp_path):
    provider = _RecordingProvider()
    ctx = LoopContext(
        provider=provider,
        model="test",
        system_prompt="BASE PROMPT",
        max_tokens=64,
    )
    _drive(ctx, [ConversationMessage.from_user_text("what is beacon?")])
    assert provider.system_prompts == ["BASE PROMPT"]


def test_run_loop_recall_error_fails_open(tmp_path):
    class _Raising:
        def recall(self, query):  # noqa: ANN001
            raise RuntimeError("boom")

    provider = _RecordingProvider()
    ctx = LoopContext(
        provider=provider,
        model="test",
        system_prompt="BASE PROMPT",
        max_tokens=64,
        memory_recall=_Raising(),
    )
    _drive(ctx, [ConversationMessage.from_user_text("hello there daemon")])
    assert provider.system_prompts == ["BASE PROMPT"]


def test_agent_loop_late_assignment_reaches_context(tmp_path):
    """The daemon wires recall AFTER AgentLoop construction — run_async must
    read the attribute at call time, not capture it at __init__."""
    store = _store(tmp_path)
    _persist(store, "beacon", "Beacon is the desktop dashboard for Prometheus.")

    provider = _RecordingProvider()
    loop = AgentLoop(provider=provider, model="test")
    loop.memory_recall = MemoryRecall(store=store)  # late, like the daemon

    result = asyncio.run(
        loop.run_async(system_prompt="BASE PROMPT", user_message="what is beacon?")
    )
    assert result.text == "ok"
    assert "# Recalled memory" in provider.system_prompts[0]
