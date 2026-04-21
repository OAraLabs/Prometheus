"""Tests for SENTINEL proactive daemon subsystem (Sprint 9).

Covers: SignalBus, ActivityObserver, AutoDreamEngine, WikiLinter,
MemoryConsolidator, TelemetryDigest, SentinelStatusTool, WikiLintTool.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prometheus.sentinel.signals import ActivitySignal, SignalBus


def _make_memory_store(db_path):
    """Import MemoryStore without triggering memory/__init__.py circular imports."""
    import sys
    import importlib.util

    if "prometheus.memory.store" in sys.modules:
        return sys.modules["prometheus.memory.store"].MemoryStore(db_path=db_path)

    # Load store module by file path to avoid triggering memory/__init__.py
    store_path = (
        Path(__file__).resolve().parent.parent
        / "src" / "prometheus" / "memory" / "store.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prometheus.memory.store", str(store_path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {store_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prometheus.memory.store"] = mod
    spec.loader.exec_module(mod)
    return mod.MemoryStore(db_path=db_path)


def _make_telemetry(db_path):
    """Import ToolCallTelemetry directly."""
    import importlib
    mod = importlib.import_module("prometheus.telemetry.tracker")
    return mod.ToolCallTelemetry(db_path=db_path)


# ======================================================================
# SignalBus
# ======================================================================


class TestSignalBus:
    """Test async pub/sub signal bus."""

    @pytest.mark.asyncio
    async def test_emit_calls_subscriber(self):
        bus = SignalBus()
        received = []
        async def handler(sig: ActivitySignal):
            received.append(sig)
        bus.subscribe("test", handler)
        sig = ActivitySignal(kind="test", payload={"a": 1})
        await bus.emit(sig)
        assert len(received) == 1
        assert received[0].kind == "test"
        assert received[0].payload == {"a": 1}

    @pytest.mark.asyncio
    async def test_wildcard_receives_all(self):
        bus = SignalBus()
        received = []
        async def handler(sig: ActivitySignal):
            received.append(sig.kind)
        bus.subscribe("*", handler)
        await bus.emit(ActivitySignal(kind="alpha"))
        await bus.emit(ActivitySignal(kind="beta"))
        assert received == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_subscriber_exception_caught(self):
        bus = SignalBus()
        async def bad_handler(sig: ActivitySignal):
            raise RuntimeError("boom")
        good_calls = []
        async def good_handler(sig: ActivitySignal):
            good_calls.append(1)
        bus.subscribe("test", bad_handler)
        bus.subscribe("test", good_handler)
        await bus.emit(ActivitySignal(kind="test"))
        # Good handler still called despite bad handler raising
        assert len(good_calls) == 1

    @pytest.mark.asyncio
    async def test_history_bounded(self):
        bus = SignalBus(history_limit=5)
        for i in range(10):
            await bus.emit(ActivitySignal(kind=f"s{i}"))
        assert bus.signal_count == 5
        recent = bus.recent()
        assert len(recent) == 5
        assert recent[0].kind == "s5"

    @pytest.mark.asyncio
    async def test_recent_filters_by_kind(self):
        bus = SignalBus()
        await bus.emit(ActivitySignal(kind="a"))
        await bus.emit(ActivitySignal(kind="b"))
        await bus.emit(ActivitySignal(kind="a"))
        assert len(bus.recent("a")) == 2
        assert len(bus.recent("b")) == 1

    @pytest.mark.asyncio
    async def test_subscriber_count(self):
        bus = SignalBus()
        async def noop(sig): pass
        bus.subscribe("a", noop)
        bus.subscribe("b", noop)
        bus.subscribe("*", noop)
        assert bus.subscriber_count == 3


# ======================================================================
# ActivityObserver
# ======================================================================


class TestActivityObserver:
    """Test pattern detection and nudge delivery."""

    @pytest.mark.asyncio
    async def test_sends_nudge_on_extraction_spike(self):
        from prometheus.sentinel.observer import ActivityObserver

        bus = SignalBus()
        gateway = MagicMock()
        gateway.send = AsyncMock()
        observer = ActivityObserver(
            bus, gateway=gateway,
            config={"extraction_spike_threshold": 5, "nudge_chat_id": 123},
        )
        await observer.start()

        await bus.emit(ActivitySignal(
            kind="extraction_complete",
            payload={"count": 10, "facts": 10},
            source="extractor",
        ))
        gateway.send.assert_called_once()
        assert "10 new facts" in gateway.send.call_args[0][1]

    @pytest.mark.asyncio
    async def test_respects_nudge_cooldown(self):
        from prometheus.sentinel.observer import ActivityObserver

        bus = SignalBus()
        gateway = MagicMock()
        gateway.send = AsyncMock()
        observer = ActivityObserver(
            bus, gateway=gateway,
            config={
                "extraction_spike_threshold": 5,
                "nudge_cooldown_minutes": 60,
                "nudge_chat_id": 123,
            },
        )
        await observer.start()

        # First nudge goes through
        await bus.emit(ActivitySignal(
            kind="extraction_complete",
            payload={"count": 10, "facts": 10},
        ))
        assert gateway.send.call_count == 1

        # Second nudge within cooldown is queued
        await bus.emit(ActivitySignal(
            kind="extraction_complete",
            payload={"count": 15, "facts": 15},
        ))
        assert gateway.send.call_count == 1  # Still 1
        assert len(observer.pending_nudges) == 1

    @pytest.mark.asyncio
    async def test_no_crash_without_gateway(self):
        from prometheus.sentinel.observer import ActivityObserver

        bus = SignalBus()
        observer = ActivityObserver(bus, gateway=None, config={})
        await observer.start()

        # Should not crash
        await bus.emit(ActivitySignal(
            kind="extraction_complete",
            payload={"count": 100, "facts": 100},
        ))
        assert observer.started

    @pytest.mark.asyncio
    async def test_error_streak_detection(self):
        from prometheus.sentinel.observer import ActivityObserver

        bus = SignalBus()
        gateway = MagicMock()
        gateway.send = AsyncMock()
        observer = ActivityObserver(
            bus, gateway=gateway,
            config={"error_streak_threshold": 3, "nudge_chat_id": 123},
        )
        await observer.start()

        for _ in range(3):
            await bus.emit(ActivitySignal(
                kind="tool_executed",
                payload={"success": False, "tool_name": "bash"},
            ))
        gateway.send.assert_called_once()
        assert "failed" in gateway.send.call_args[0][1].lower()


# ======================================================================
# AutoDreamEngine
# ======================================================================


class TestAutoDreamEngine:
    """Test idle-time dream cycles."""

    @pytest.mark.asyncio
    async def test_run_cycle_calls_all_phases(self):
        from prometheus.sentinel.autodream import AutoDreamEngine
        from prometheus.sentinel.wiki_lint import WikiLinter, LintResult
        from prometheus.sentinel.memory_consolidator import (
            MemoryConsolidator,
            ConsolidationResult,
        )
        from prometheus.sentinel.telemetry_digest import TelemetryDigest, DigestResult

        bus = SignalBus()
        linter = MagicMock(spec=WikiLinter)
        linter.lint.return_value = LintResult()
        consolidator = MagicMock(spec=MemoryConsolidator)
        consolidator.consolidate.return_value = ConsolidationResult()
        digest = MagicMock(spec=TelemetryDigest)
        digest.generate.return_value = DigestResult()

        engine = AutoDreamEngine(
            bus,
            wiki_linter=linter,
            memory_consolidator=consolidator,
            telemetry_digest=digest,
            knowledge_synth=None,
            config={"synthesis_enabled": False},
        )
        await engine.start()

        results = await engine.run_cycle()
        assert len(results) == 3
        linter.lint.assert_called_once()
        consolidator.consolidate.assert_called_once()
        digest.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_emits_dream_complete_signal(self):
        from prometheus.sentinel.autodream import AutoDreamEngine

        bus = SignalBus()
        received = []
        async def capture(sig):
            received.append(sig)
        bus.subscribe("dream_complete", capture)

        engine = AutoDreamEngine(bus, config={"synthesis_enabled": False})
        await engine.start()
        await engine.run_cycle()

        assert len(received) == 1
        assert received[0].kind == "dream_complete"

    @pytest.mark.asyncio
    async def test_phase_failure_does_not_block_others(self):
        from prometheus.sentinel.autodream import AutoDreamEngine
        from prometheus.sentinel.wiki_lint import WikiLinter
        from prometheus.sentinel.memory_consolidator import (
            MemoryConsolidator,
            ConsolidationResult,
        )

        bus = SignalBus()
        linter = MagicMock(spec=WikiLinter)
        linter.lint.side_effect = RuntimeError("lint exploded")
        consolidator = MagicMock(spec=MemoryConsolidator)
        consolidator.consolidate.return_value = ConsolidationResult()

        engine = AutoDreamEngine(
            bus,
            wiki_linter=linter,
            memory_consolidator=consolidator,
            config={"synthesis_enabled": False},
        )
        await engine.start()
        results = await engine.run_cycle()

        assert len(results) == 2
        assert results[0].error is not None  # lint failed
        assert results[1].error is None  # consolidation succeeded
        consolidator.consolidate.assert_called_once()

    @pytest.mark.asyncio
    async def test_starts_dreaming_on_idle_signal(self):
        from prometheus.sentinel.autodream import AutoDreamEngine

        bus = SignalBus()
        engine = AutoDreamEngine(bus, config={"synthesis_enabled": False})
        await engine.start()

        assert not engine.dreaming
        # Emit idle_start but cancel the dream loop quickly
        await bus.emit(ActivitySignal(kind="idle_start", source="heartbeat"))
        # Give the task a moment to start
        await asyncio.sleep(0.05)
        assert engine.dreaming

        # Stop dreaming
        await bus.emit(ActivitySignal(kind="idle_end", source="heartbeat"))
        await asyncio.sleep(0.05)
        assert not engine.dreaming


# ======================================================================
# WikiLinter
# ======================================================================


class TestWikiLinter:
    """Test wiki health checks."""

    def _setup_wiki(self, tmp_path: Path) -> Path:
        """Create a minimal wiki structure."""
        wiki = tmp_path / "wiki"
        for subdir in ("people", "topics", "queries"):
            (wiki / subdir).mkdir(parents=True)
        # Index
        (wiki / "index.md").write_text(
            "# Wiki Index\n- [[Alice]]\n- [[Quantum]]\n", encoding="utf-8"
        )
        return wiki

    def test_finds_orphan_pages(self, tmp_path: Path):
        from prometheus.sentinel.wiki_lint import WikiLinter

        wiki = self._setup_wiki(tmp_path)
        # Alice is in index, Bob is not
        (wiki / "people" / "Alice.md").write_text(
            "---\ntype: person\n---\n# Alice\n", encoding="utf-8"
        )
        (wiki / "people" / "Bob.md").write_text(
            "---\ntype: person\n---\n# Bob\n", encoding="utf-8"
        )

        linter = WikiLinter(wiki_root=wiki)
        result = linter.lint()
        orphans = [i for i in result.issues if i.category == "orphan"]
        assert len(orphans) == 1
        assert "Bob" in orphans[0].detail

    def test_finds_broken_links(self, tmp_path: Path):
        from prometheus.sentinel.wiki_lint import WikiLinter

        wiki = self._setup_wiki(tmp_path)
        (wiki / "people" / "Alice.md").write_text(
            "---\ntype: person\n---\n# Alice\nWorks with [[NonExistent]]\n",
            encoding="utf-8",
        )

        linter = WikiLinter(wiki_root=wiki)
        result = linter.lint()
        broken = [i for i in result.issues if i.category == "broken_link"]
        assert len(broken) == 1
        assert "NonExistent" in broken[0].detail

    def test_finds_missing_crossrefs(self, tmp_path: Path):
        from prometheus.sentinel.wiki_lint import WikiLinter

        wiki = self._setup_wiki(tmp_path)
        (wiki / "people" / "Alice.md").write_text(
            "---\ntype: person\n---\n# Alice\nStudies Quantum computing\n",
            encoding="utf-8",
        )
        (wiki / "topics" / "Quantum.md").write_text(
            "---\ntype: concept\n---\n# Quantum\nA physics topic\n",
            encoding="utf-8",
        )

        linter = WikiLinter(wiki_root=wiki)
        result = linter.lint()
        missing = [i for i in result.issues if i.category == "missing_crossref"]
        assert len(missing) >= 1

    def test_healthy_wiki_returns_no_issues(self, tmp_path: Path):
        from prometheus.sentinel.wiki_lint import WikiLinter

        wiki = self._setup_wiki(tmp_path)
        # Create a single well-linked page
        (wiki / "people" / "Alice.md").write_text(
            "---\ntype: person\n---\n# Alice\n", encoding="utf-8"
        )

        linter = WikiLinter(wiki_root=wiki)
        result = linter.lint()
        # At most only info-level issues (no errors or warnings expected
        # for a single linked page)
        errors = [i for i in result.issues if i.severity == "error"]
        assert len(errors) == 0

    def test_empty_wiki_returns_no_issues(self, tmp_path: Path):
        from prometheus.sentinel.wiki_lint import WikiLinter

        wiki = tmp_path / "wiki"
        wiki.mkdir()
        linter = WikiLinter(wiki_root=wiki)
        result = linter.lint()
        assert not result.has_issues


# ======================================================================
# MemoryConsolidator
# ======================================================================


class TestMemoryConsolidator:
    """Test memory dedup, decay, and tombstone."""

    def _make_store(self, tmp_path: Path):
        return _make_memory_store(tmp_path / "test_memory.db")

    def test_merges_duplicate_facts(self, tmp_path: Path):
        from prometheus.sentinel.memory_consolidator import MemoryConsolidator

        store = self._make_store(tmp_path)
        store.persist_memory("person", "Alice", "Alice works at Acme Corp", 0.8)
        store.persist_memory("person", "Alice", "Alice works at Acme Corporation", 0.6)

        consolidator = MemoryConsolidator(store, similarity_threshold=0.75)
        result = consolidator.consolidate()
        assert result.duplicates_merged >= 1

        # Only one should remain
        remaining = store.search_memories(entity="Alice")
        assert len(remaining) == 1
        assert remaining[0]["confidence"] == 0.8  # Kept the higher one

    def test_decays_stale_confidence(self, tmp_path: Path):
        from prometheus.sentinel.memory_consolidator import MemoryConsolidator

        store = self._make_store(tmp_path)
        mid = store.persist_memory("person", "Bob", "Bob likes pizza", 0.7)

        # Manually backdate last_mentioned
        old_time = time.time() - (100 * 86400)  # 100 days ago
        store.update_memory(mid, last_mentioned=old_time)

        consolidator = MemoryConsolidator(store, stale_days=90, decay_rate=0.05)
        result = consolidator.consolidate()
        assert result.confidence_decayed >= 1

        mem = store.get_memory(mid)
        assert mem is not None
        assert mem["confidence"] < 0.7

    def test_tombstones_low_confidence(self, tmp_path: Path):
        from prometheus.sentinel.memory_consolidator import MemoryConsolidator

        store = self._make_store(tmp_path)
        mid = store.persist_memory("concept", "X", "X is something", 0.05)

        consolidator = MemoryConsolidator(store, min_confidence=0.1)
        result = consolidator.consolidate()
        assert result.tombstoned >= 1

        mem = store.get_memory(mid)
        assert mem is None  # Deleted


# ======================================================================
# TelemetryDigest
# ======================================================================


class TestTelemetryDigest:
    """Test telemetry anomaly detection."""

    def _make_telemetry(self, tmp_path: Path):
        return _make_telemetry(tmp_path / "test_telemetry.db")

    def test_empty_returns_no_anomalies(self, tmp_path: Path):
        from prometheus.sentinel.telemetry_digest import TelemetryDigest

        tel = self._make_telemetry(tmp_path)
        digest = TelemetryDigest(tel)
        result = digest.generate()
        assert not result.has_anomalies
        assert result.total_calls == 0

    def test_detects_success_rate_drop(self, tmp_path: Path):
        from prometheus.sentinel.telemetry_digest import TelemetryDigest

        tel = self._make_telemetry(tmp_path)

        # Baseline: 10 successful calls (older, within 7 day window)
        baseline_time = time.time() - (3 * 86400)  # 3 days ago
        for _ in range(10):
            tel._conn.execute(
                "INSERT INTO tool_calls (id, timestamp, model, tool_name, success, retries, latency_ms)"
                " VALUES (hex(randomblob(16)), ?, 'model', 'bash', 1, 0, 100.0)",
                (baseline_time,),
            )

        # Current: 10 calls with 50% failure rate (recent, within 24h)
        recent_time = time.time() - 3600  # 1 hour ago
        for i in range(10):
            tel._conn.execute(
                "INSERT INTO tool_calls (id, timestamp, model, tool_name, success, retries, latency_ms)"
                " VALUES (hex(randomblob(16)), ?, 'model', 'bash', ?, 0, 100.0)",
                (recent_time, 1 if i < 5 else 0),
            )
        tel._conn.commit()

        digest = TelemetryDigest(tel, period_hours=24, baseline_hours=168)
        result = digest.generate()
        assert result.has_anomalies
        rate_drops = [a for a in result.anomalies if a.metric == "success_rate_drop"]
        assert len(rate_drops) >= 1

    def test_detects_latency_spike(self, tmp_path: Path):
        from prometheus.sentinel.telemetry_digest import TelemetryDigest

        tel = self._make_telemetry(tmp_path)

        # Baseline: many normal latency calls (older, outside current 24h)
        # This creates a baseline avg of ~100ms
        baseline_time = time.time() - (3 * 86400)
        for _ in range(50):
            tel._conn.execute(
                "INSERT INTO tool_calls (id, timestamp, model, tool_name, success, retries, latency_ms)"
                " VALUES (hex(randomblob(16)), ?, 'model', 'bash', 1, 0, 100.0)",
                (baseline_time,),
            )

        # Current: 3x latency (within 24h window)
        # Baseline includes these too, but 50 old records at 100ms
        # dominate the baseline average, keeping it near 100ms
        recent_time = time.time() - 3600
        for _ in range(10):
            tel._conn.execute(
                "INSERT INTO tool_calls (id, timestamp, model, tool_name, success, retries, latency_ms)"
                " VALUES (hex(randomblob(16)), ?, 'model', 'bash', 1, 0, 300.0)",
                (recent_time,),
            )
        tel._conn.commit()

        digest = TelemetryDigest(tel, period_hours=24, baseline_hours=168)
        result = digest.generate()
        spikes = [a for a in result.anomalies if a.metric == "latency_spike"]
        assert len(spikes) >= 1


# ======================================================================
# SentinelStatusTool
# ======================================================================


class TestSentinelStatusTool:
    """Test sentinel_status tool."""

    @pytest.mark.asyncio
    async def test_returns_status_when_wired(self):
        from prometheus.tools.builtin.sentinel_status import (
            SentinelStatusTool,
            SentinelStatusInput,
            set_sentinel_components,
        )
        from prometheus.sentinel.autodream import AutoDreamEngine
        from prometheus.sentinel.observer import ActivityObserver
        from prometheus.tools.base import ToolExecutionContext

        bus = SignalBus()
        observer = ActivityObserver(bus)
        await observer.start()
        autodream = AutoDreamEngine(bus)
        await autodream.start()
        set_sentinel_components(bus, observer, autodream)

        tool = SentinelStatusTool()
        ctx = ToolExecutionContext(cwd=Path.cwd())
        result = await tool.execute(SentinelStatusInput(), ctx)
        assert not result.is_error
        assert "SENTINEL Status" in result.output

    @pytest.mark.asyncio
    async def test_returns_error_when_not_initialized(self):
        from prometheus.tools.builtin import sentinel_status
        from prometheus.tools.builtin.sentinel_status import (
            SentinelStatusTool,
            SentinelStatusInput,
        )
        from prometheus.tools.base import ToolExecutionContext

        # Reset singletons
        sentinel_status._signal_bus = None
        sentinel_status._observer = None
        sentinel_status._autodream = None

        tool = SentinelStatusTool()
        ctx = ToolExecutionContext(cwd=Path.cwd())
        result = await tool.execute(SentinelStatusInput(), ctx)
        assert result.is_error


# ======================================================================
# WikiLintTool
# ======================================================================


class TestWikiLintTool:
    """Test wiki_lint tool."""

    @pytest.mark.asyncio
    async def test_returns_results_when_wired(self, tmp_path: Path):
        from prometheus.tools.builtin.wiki_lint_tool import (
            WikiLintTool,
            WikiLintInput,
            set_wiki_linter,
        )
        from prometheus.sentinel.wiki_lint import WikiLinter
        from prometheus.tools.base import ToolExecutionContext

        wiki = tmp_path / "wiki"
        wiki.mkdir()
        linter = WikiLinter(wiki_root=wiki)
        set_wiki_linter(linter)

        tool = WikiLintTool()
        ctx = ToolExecutionContext(cwd=Path.cwd())
        result = await tool.execute(WikiLintInput(), ctx)
        assert not result.is_error
        assert "healthy" in result.output.lower()

    @pytest.mark.asyncio
    async def test_returns_error_when_not_initialized(self):
        from prometheus.tools.builtin import wiki_lint_tool
        from prometheus.tools.builtin.wiki_lint_tool import (
            WikiLintTool,
            WikiLintInput,
        )
        from prometheus.tools.base import ToolExecutionContext

        wiki_lint_tool._wiki_linter = None

        tool = WikiLintTool()
        ctx = ToolExecutionContext(cwd=Path.cwd())
        result = await tool.execute(WikiLintInput(), ctx)
        assert result.is_error


# ======================================================================
# MemoryStore extensions
# ======================================================================


class TestMemoryStoreExtensions:
    """Test new update_memory, delete_memory, get_all_memories."""

    def _make_store(self, tmp_path: Path):
        return _make_memory_store(tmp_path / "test_ext.db")

    def test_update_memory(self, tmp_path: Path):
        store = self._make_store(tmp_path)
        mid = store.persist_memory("person", "Alice", "Alice is tall", 0.5)
        store.update_memory(mid, confidence=0.9, mention_count=5)
        mem = store.get_memory(mid)
        assert mem["confidence"] == 0.9
        assert mem["mention_count"] == 5

    def test_delete_memory(self, tmp_path: Path):
        store = self._make_store(tmp_path)
        mid = store.persist_memory("person", "Bob", "Bob is short", 0.5)
        store.delete_memory(mid)
        assert store.get_memory(mid) is None

    def test_get_all_memories(self, tmp_path: Path):
        store = self._make_store(tmp_path)
        store.persist_memory("person", "A", "Fact A", 0.8)
        store.persist_memory("person", "B", "Fact B", 0.3)
        store.persist_memory("person", "C", "Fact C", 0.05)

        all_mems = store.get_all_memories(min_confidence=0.0)
        assert len(all_mems) == 3

        filtered = store.get_all_memories(min_confidence=0.5)
        assert len(filtered) == 1


# ======================================================================
# Telemetry report(since=) extension
# ======================================================================


class TestTelemetryReportSince:
    """Test since parameter on ToolCallTelemetry.report()."""

    def test_since_filters_old_records(self, tmp_path: Path):
        tel = _make_telemetry(tmp_path / "test_tel.db")

        # Old record
        tel._conn.execute(
            "INSERT INTO tool_calls (id, timestamp, model, tool_name, success, retries, latency_ms)"
            " VALUES ('old', ?, 'model', 'bash', 1, 0, 100.0)",
            (time.time() - 86400 * 30,),  # 30 days ago
        )
        # Recent record
        tel._conn.execute(
            "INSERT INTO tool_calls (id, timestamp, model, tool_name, success, retries, latency_ms)"
            " VALUES ('new', ?, 'model', 'bash', 1, 0, 200.0)",
            (time.time() - 3600,),  # 1 hour ago
        )
        tel._conn.commit()

        # Without since: both records
        report_all = tel.report()
        assert report_all["total_calls"] == 2

        # With since: only recent
        report_recent = tel.report(since=time.time() - 86400)
        assert report_recent["total_calls"] == 1
