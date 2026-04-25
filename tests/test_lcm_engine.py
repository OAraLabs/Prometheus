"""Tests for LCMEngine's public contract — the store accessors used by LCM tools.

The tools in ``src/prometheus/tools/builtin/lcm_*.py`` reach into the engine
for FTS5 search and summary-node lookup via ``engine.summary_store`` and
``engine.conversation_store``. Those accessors existed in the tool code long
before they existed on the engine: an AttributeError on a bare ``LCMEngine``
instance is how the Phase 4 Haiku /claude pilot surfaced the mismatch.

These tests lock the public contract in place so a future internal rename of
``_sum_store`` / ``_conv_store`` can't silently break the tools again.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_engine import LCMEngine
from prometheus.memory.lcm_summary_store import LCMSummaryStore


@pytest.fixture()
def engine(tmp_path):
    """Build a throwaway LCMEngine against an isolated SQLite db."""
    eng = LCMEngine(provider=MagicMock(), db_path=tmp_path / "lcm.db")
    try:
        yield eng
    finally:
        eng.close()


def test_lcm_engine_exposes_summary_store(engine):
    """``engine.summary_store`` must be the same instance as ``_sum_store``
    and be a real LCMSummaryStore. LCM tools (lcm_grep, lcm_expand,
    lcm_expand_query, lcm_describe) all consume this property."""
    assert engine.summary_store is engine._sum_store
    assert isinstance(engine.summary_store, LCMSummaryStore)


def test_lcm_engine_exposes_conversation_store(engine):
    """``engine.conversation_store`` must be the same instance as
    ``_conv_store`` and be a real LCMConversationStore."""
    assert engine.conversation_store is engine._conv_store
    assert isinstance(engine.conversation_store, LCMConversationStore)
