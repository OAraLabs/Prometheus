"""Tests for the memory module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from prometheus.memory.pointer import MemoryPointer
from prometheus.memory.store import MemoryStore


# ---------------------------------------------------------------------------
# MemoryStore — acceptance test (from sprint spec)
# ---------------------------------------------------------------------------


def test_acceptance_test():
    """Verbatim acceptance test from SPRINT-5 spec."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "memory.db"
        memory = MemoryStore(db_path=db_path)

        memory.persist_memory("person", "Dr. Pham", "nephrologist candidate", 0.95)

        results = memory.search_memories(entity="Dr. Pham")
        assert len(results) > 0, "Should find Dr. Pham"
        assert results[0]["entity_name"] == "Dr. Pham"
        assert results[0]["fact"] == "nephrologist candidate"

        pointer_path = Path(tmp) / "MEMORY.md"
        pointers = MemoryPointer(pointer_path=pointer_path)
        pointers.add_pointer("Dr. Pham | nephrologist candidate | high confidence")
        all_entries = pointers.get_all()
        assert len(all_entries) == 1
        assert "Dr. Pham" in all_entries[0]

        memory.close()


# ---------------------------------------------------------------------------
# MemoryStore — messages
# ---------------------------------------------------------------------------


def _make_store(tmp: str) -> MemoryStore:
    return MemoryStore(db_path=Path(tmp) / "memory.db")


def test_add_and_get_messages():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        mid = store.add_message("sess1", "user", "Hello world")
        msgs = store.get_messages("sess1")
        assert len(msgs) == 1
        assert msgs[0]["id"] == mid
        assert msgs[0]["content"] == "Hello world"
        store.close()


def test_get_messages_filtered_by_session():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        store.add_message("sess1", "user", "A")
        store.add_message("sess2", "user", "B")
        msgs = store.get_messages("sess1")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "A"
        store.close()


# ---------------------------------------------------------------------------
# MemoryStore — memories
# ---------------------------------------------------------------------------


def test_persist_and_search_memory():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        store.persist_memory("person", "Alice", "software engineer", 0.9)
        results = store.search_memories(entity="Alice")
        assert len(results) == 1
        assert results[0]["entity_name"] == "Alice"
        store.close()


def test_persist_deduplicates():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        id1 = store.persist_memory("person", "Bob", "loves coffee", 0.8)
        id2 = store.persist_memory("person", "Bob", "loves coffee", 0.9)
        # Same entity_name + fact → should update, not insert
        assert id1 == id2
        results = store.search_memories(entity="Bob")
        assert len(results) == 1
        assert results[0]["mention_count"] == 2
        store.close()


def test_persist_different_facts():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        store.persist_memory("person", "Carol", "works at Acme", 0.9)
        store.persist_memory("person", "Carol", "uses Python", 0.85)
        results = store.search_memories(entity="Carol")
        assert len(results) == 2
        store.close()


def test_search_by_entity_type():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        store.persist_memory("tool", "pytest", "test runner", 0.95)
        store.persist_memory("person", "Dave", "engineer", 0.8)
        tools = store.search_memories(entity_type="tool")
        assert all(r["entity_type"] == "tool" for r in tools)
        assert any(r["entity_name"] == "pytest" for r in tools)
        store.close()


def test_search_by_fts_query():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        store.persist_memory("person", "Eve", "works as a cardiologist", 0.9)
        store.persist_memory("person", "Frank", "works as a neurologist", 0.9)
        results = store.search_memories(query="cardiologist")
        assert len(results) >= 1
        assert any(r["entity_name"] == "Eve" for r in results)
        store.close()


def test_get_memory_by_id():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        mid = store.persist_memory("concept", "FTS5", "SQLite full-text search", 0.95)
        result = store.get_memory(mid)
        assert result is not None
        assert result["fact"] == "SQLite full-text search"
        store.close()


def test_get_missing_memory_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        assert store.get_memory("nonexistent-id") is None
        store.close()


def test_min_confidence_filter():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        store.persist_memory("person", "George", "uncertain fact", 0.3)
        store.persist_memory("person", "Henry", "sure fact", 0.9)
        high = store.search_memories(min_confidence=0.8)
        names = {r["entity_name"] for r in high}
        assert "Henry" in names
        assert "George" not in names
        store.close()


# ---------------------------------------------------------------------------
# MemoryStore — summaries
# ---------------------------------------------------------------------------


def test_add_and_get_summary():
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        sid = store.add_summary("Session was about Python.", ["m1", "m2"])
        summaries = store.get_summaries()
        assert len(summaries) == 1
        assert summaries[0]["id"] == sid
        assert summaries[0]["source_message_ids"] == ["m1", "m2"]
        store.close()


# ---------------------------------------------------------------------------
# MemoryPointer
# ---------------------------------------------------------------------------


def test_pointer_add_and_get():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "MEMORY.md"
        p = MemoryPointer(pointer_path=path)
        p.add_pointer("Alice | engineer | high confidence")
        entries = p.get_all()
        assert len(entries) == 1
        assert "Alice" in entries[0]


def test_pointer_remove():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "MEMORY.md"
        p = MemoryPointer(pointer_path=path)
        p.add_pointer("Alice | engineer")
        p.add_pointer("Bob | manager")
        removed = p.remove_pointer("Alice")
        assert removed
        entries = p.get_all()
        assert all("Alice" not in e for e in entries)
        assert any("Bob" in e for e in entries)


def test_pointer_replace():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "MEMORY.md"
        p = MemoryPointer(pointer_path=path)
        p.add_pointer("Alice | junior engineer")
        replaced = p.replace_pointer("junior engineer", "Alice | senior engineer")
        assert replaced
        entries = p.get_all()
        assert any("senior" in e for e in entries)
        assert all("junior" not in e for e in entries)


def test_pointer_char_limit_prunes_oldest():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "MEMORY.md"
        p = MemoryPointer(pointer_path=path, max_chars=50)
        p.add_pointer("First entry that is reasonably long")
        p.add_pointer("Second entry that is also long enough to trigger prune")
        entries = p.get_all()
        # Should have pruned the first entry
        assert not any("First entry" in e for e in entries)


def test_pointer_format_for_prompt():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "MEMORY.md"
        p = MemoryPointer(pointer_path=path)
        p.add_pointer("Alice | engineer")
        prompt = p.format_for_prompt()
        assert "Memory Pointers" in prompt
        assert "Alice" in prompt


def test_pointer_format_for_prompt_empty():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "MEMORY.md"
        p = MemoryPointer(pointer_path=path)
        assert p.format_for_prompt() == ""


# ---------------------------------------------------------------------------
# MemoryStore context manager
# ---------------------------------------------------------------------------


def test_store_context_manager():
    with tempfile.TemporaryDirectory() as tmp:
        with MemoryStore(db_path=Path(tmp) / "memory.db") as store:
            store.persist_memory("person", "Ingrid", "data scientist", 0.85)
            results = store.search_memories(entity="Ingrid")
            assert len(results) == 1
