"""SignalBus Persistence sprint tests.

Closes the drive-by from PR #4 ("SignalBus history is in-memory only;
Sprint 1 emissions evaporate on restart"). These tests pin three contracts:

  1. **Persistence precedes broadcast.** Every ``emit`` writes a
     ``signal_events`` row before notifying subscribers.
  2. **Existing subscribers are unaffected.** The in-memory deque and
     callback wiring continue to work as Sprint 1 + Sprint 9 designed.
  3. **History survives recreation.** The whole point of the sprint:
     destroy the bus, build a new one against the same telemetry handle,
     and prior events are still queryable.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from prometheus.sentinel.signals import ActivitySignal, SignalBus
from prometheus.telemetry.tracker import ToolCallTelemetry


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def telemetry(tmp_path: Path) -> ToolCallTelemetry:
    """Fresh on-disk telemetry per test — exercises the real schema."""
    return ToolCallTelemetry(db_path=tmp_path / "telemetry.db")


@pytest.fixture
def bus(telemetry: ToolCallTelemetry) -> SignalBus:
    return SignalBus(telemetry=telemetry, history_limit=500)


def _make_signal(kind: str = "skill_created", **payload: Any) -> ActivitySignal:
    return ActivitySignal(
        kind=kind,
        payload=dict(payload),
        source=payload.pop("_source", "SkillCreator"),
    )


# ---------------------------------------------------------------------------
# WIRING — persist before broadcast, deque still populated
# ---------------------------------------------------------------------------


class TestEmissionWiring:
    @pytest.mark.asyncio
    async def test_emission_writes_signal_events_row_before_broadcast(
        self, bus: SignalBus, telemetry: ToolCallTelemetry
    ) -> None:
        """The subscriber callback observes a DB row already in place by
        the time it runs — proves write-before-broadcast ordering."""
        observed_rows: list[int] = []

        async def subscriber(sig: ActivitySignal) -> None:
            # At the moment this callback runs, the row must already exist.
            rows = telemetry.signal_events_since(limit=10)
            observed_rows.append(len(rows))

        bus.subscribe("skill_created", subscriber)
        await bus.emit(_make_signal("skill_created", name="demo"))

        assert observed_rows == [1], (
            "Expected exactly 1 row visible when the subscriber ran. "
            f"Got {observed_rows}. If 0, the broadcast ran before the "
            f"DB commit — the sprint contract is violated."
        )

    @pytest.mark.asyncio
    async def test_emission_populates_in_memory_deque_alongside_db(
        self, bus: SignalBus, telemetry: ToolCallTelemetry
    ) -> None:
        await bus.emit(_make_signal("memory_updated", target="MEMORY.md"))

        # Hot cache: still served from deque.
        assert bus.signal_count == 1
        in_mem = bus.recent(limit=5)
        assert len(in_mem) == 1
        assert in_mem[0].kind == "memory_updated"

        # Cold tail: DB row also written.
        rows = telemetry.signal_events_since(limit=5)
        assert len(rows) == 1
        assert rows[0]["signal_type"] == "memory_updated"
        assert rows[0]["payload"] == {"target": "MEMORY.md"}

    @pytest.mark.asyncio
    async def test_falls_back_to_telemetry_singleton_when_no_explicit_handle(
        self, telemetry: ToolCallTelemetry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Constructor with no telemetry arg resolves via get_telemetry_handle."""
        from prometheus.telemetry import tracker as _tracker

        monkeypatch.setattr(_tracker, "_telemetry_singleton", telemetry)
        bus = SignalBus()  # no explicit telemetry
        await bus.emit(_make_signal("curator_report", outcome="success"))

        rows = telemetry.signal_events_since(limit=5)
        assert len(rows) == 1
        assert rows[0]["signal_type"] == "curator_report"


# ---------------------------------------------------------------------------
# QUERY API — history() and recent()
# ---------------------------------------------------------------------------


class TestQueryAPI:
    @pytest.mark.asyncio
    async def test_history_query_filters_by_signal_type(
        self, bus: SignalBus
    ) -> None:
        await bus.emit(_make_signal("skill_created", name="alpha"))
        await bus.emit(_make_signal("memory_updated", target="MEMORY.md"))
        await bus.emit(_make_signal("skill_created", name="beta"))

        skills = bus.history(signal_type="skill_created", limit=10)
        assert {s.payload["name"] for s in skills} == {"alpha", "beta"}

        mem = bus.history(signal_type="memory_updated", limit=10)
        assert len(mem) == 1
        assert mem[0].payload["target"] == "MEMORY.md"

    @pytest.mark.asyncio
    async def test_history_query_respects_since_and_limit(
        self, bus: SignalBus, telemetry: ToolCallTelemetry
    ) -> None:
        # Plant 5 events, then query with limit=2 and a since-cutoff between
        # event 2 and event 3.
        for i in range(5):
            await bus.emit(_make_signal("skill_created", name=f"e{i}"))

        all_rows = telemetry.signal_events_since(limit=10)
        # Newest-first; rows[2] is the middle event (e2). Use rows[2]'s
        # timestamp as the lower bound — should return events e2, e3, e4
        # (3 newer-or-equal), capped to limit=2 → e4, e3.
        cutoff = all_rows[2]["timestamp"]
        filtered = bus.history(since=cutoff, limit=2)
        assert len(filtered) == 2
        names = [s.payload["name"] for s in filtered]
        # Newest-first ordering preserved.
        assert names == ["e4", "e3"]

    @pytest.mark.asyncio
    async def test_recent_falls_back_to_db_when_deque_short(
        self, telemetry: ToolCallTelemetry
    ) -> None:
        """A fresh bus pointed at a DB with prior events backfills from DB."""
        # Plant 7 events via Bus #1.
        bus1 = SignalBus(telemetry=telemetry, history_limit=500)
        for i in range(7):
            await bus1.emit(_make_signal("skill_created", name=f"e{i}"))

        # Bus #2: same DB, EMPTY deque.
        bus2 = SignalBus(telemetry=telemetry, history_limit=500)
        assert bus2.signal_count == 0  # in-memory deque is empty
        # Ask for 5 — none in memory, must backfill from DB.
        results = bus2.recent(limit=5)
        assert len(results) == 5
        # All 5 must be the most recent (e6, e5, e4, e3, e2), chronological:
        names = [s.payload["name"] for s in results]
        assert set(names) == {f"e{i}" for i in range(2, 7)}


# ---------------------------------------------------------------------------
# FAILURE-MODE — persistence error doesn't block broadcast
# ---------------------------------------------------------------------------


class TestPersistenceFailureContract:
    @pytest.mark.asyncio
    async def test_persistence_failure_does_not_block_broadcast(
        self, telemetry: ToolCallTelemetry
    ) -> None:
        """A DB-side failure must still reach all subscribers."""
        bus = SignalBus(telemetry=telemetry)
        received: list[ActivitySignal] = []

        async def subscriber(sig: ActivitySignal) -> None:
            received.append(sig)

        bus.subscribe("skill_created", subscriber)

        # Break the DB: close the underlying connection so any write raises.
        telemetry._conn.close()

        # Emission must NOT raise.
        await bus.emit(_make_signal("skill_created", name="ghost"))

        # Subscriber DID receive the event despite persistence failure.
        assert len(received) == 1
        assert received[0].payload["name"] == "ghost"

    @pytest.mark.asyncio
    async def test_persistence_failure_writes_silent_failure_row(
        self, telemetry: ToolCallTelemetry, tmp_path: Path
    ) -> None:
        """The DB-write failure path lands a silent_failure row so the
        operator can see it via /health, even though emit kept going.

        Setup: a second telemetry handle reading the same DB observes the
        failure row. We use a deliberately corrupted db_path on the *bus's*
        telemetry — the writer fails, but we can still read silent_failures
        through a sibling handle in the same file.
        """
        # Bus telemetry: same file, but we'll close its conn mid-test so
        # signal_event writes fail. record_silent_failure also fails (same
        # closed conn) — but the contract is "best-effort, never blocks the
        # broadcast", which we already verified above. Here we exercise the
        # path where the silent_failure write itself succeeds. To do that,
        # we point the bus at telemetry whose signal_events write fails but
        # whose silent_failures table is still writable. Easiest: drop the
        # signal_events table while leaving the rest intact.
        telemetry._conn.execute("DROP TABLE signal_events")
        telemetry._conn.commit()

        bus = SignalBus(telemetry=telemetry)
        await bus.emit(_make_signal("skill_created", name="goes-to-silent"))

        # silent_failure row is now present, subsystem="signal_bus".
        failures = telemetry.silent_failures_since(0.0, subsystem="signal_bus")
        assert len(failures) == 1, (
            f"Expected one silent_failure row for signal_bus. "
            f"Got {failures}."
        )
        assert failures[0]["operation"] == "persist_event"
        assert "signal_type" in (failures[0]["context"] or "")


# ---------------------------------------------------------------------------
# REGRESSION — existing subscribers preserved
# ---------------------------------------------------------------------------


class TestExistingSubscribersStillWork:
    @pytest.mark.asyncio
    async def test_telegram_style_subscriber_still_receives_events(
        self, bus: SignalBus
    ) -> None:
        """Sprint 1's Telegram subscriber pattern — single-kind subscribe."""
        seen: list[str] = []

        async def telegram_on_skill_created(sig: ActivitySignal) -> None:
            seen.append(sig.payload.get("name", "?"))

        bus.subscribe("skill_created", telegram_on_skill_created)
        await bus.emit(_make_signal("skill_created", name="demo-skill"))
        await bus.emit(_make_signal("memory_updated", target="USER.md"))

        # Only the matching-kind event reached the subscriber, as before.
        assert seen == ["demo-skill"]

    @pytest.mark.asyncio
    async def test_beacon_ws_style_wildcard_subscriber_still_receives_events(
        self, bus: SignalBus
    ) -> None:
        """Beacon WebSocketBridge subscribes to "*" — every emission must
        broadcast, just like before persistence was added."""
        seen: list[str] = []

        async def ws_bridge(sig: ActivitySignal) -> None:
            seen.append(sig.kind)

        bus.subscribe("*", ws_bridge)
        await bus.emit(_make_signal("skill_created", name="x"))
        await bus.emit(_make_signal("memory_updated", target="MEMORY.md"))
        await bus.emit(_make_signal("curator_report", outcome="success"))

        assert seen == ["skill_created", "memory_updated", "curator_report"]


# ---------------------------------------------------------------------------
# RESTART DURABILITY — the whole point of the sprint
# ---------------------------------------------------------------------------


class TestRestartDurability:
    @pytest.mark.asyncio
    async def test_events_survive_signal_bus_recreation(
        self, tmp_path: Path
    ) -> None:
        """Plant events through Bus #1. Destroy it. Build Bus #2 against
        the same DB. The first bus's events are still queryable.

        This is the sprint's load-bearing test — the whole reason this
        sprint exists. If this passes, "what did you do last week" works
        across daemon restarts."""
        db_path = tmp_path / "telemetry.db"

        # ── Phase 1: emit through Bus #1 ──────────────────────────────
        tel_1 = ToolCallTelemetry(db_path=db_path)
        bus_1 = SignalBus(telemetry=tel_1, history_limit=500)
        await bus_1.emit(_make_signal(
            "skill_created", name="restart-test-skill", _source="SkillCreator",
        ))
        await bus_1.emit(_make_signal(
            "memory_updated", target="MEMORY.md", _source="MemoryTool",
        ))
        await bus_1.emit(_make_signal(
            "curator_report", outcome="success", _source="Curator",
        ))
        tel_1.close()

        # ── Phase 2: fresh bus, fresh telemetry, same DB ─────────────
        tel_2 = ToolCallTelemetry(db_path=db_path)
        bus_2 = SignalBus(telemetry=tel_2, history_limit=500)

        # In-memory deque is empty on the fresh bus.
        assert bus_2.signal_count == 0

        # history() — the durable query path — returns all 3.
        all_events = bus_2.history(limit=10)
        assert len(all_events) == 3
        kinds = {e.kind for e in all_events}
        assert kinds == {"skill_created", "memory_updated", "curator_report"}

        # Filtered query still works post-restart.
        skill_events = bus_2.history(signal_type="skill_created")
        assert len(skill_events) == 1
        assert skill_events[0].payload["name"] == "restart-test-skill"
        assert skill_events[0].source == "SkillCreator"

        # recent() backfills from DB since the deque is empty.
        recent = bus_2.recent(limit=10)
        assert len(recent) == 3


# ---------------------------------------------------------------------------
# /events COMMAND SURFACE
# ---------------------------------------------------------------------------


class TestEventsCommand:
    """Functional tests for cmd_events — the Telegram /events surface."""

    @pytest.fixture
    def wired_telemetry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> ToolCallTelemetry:
        tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
        from prometheus.telemetry import tracker as _tracker
        monkeypatch.setattr(_tracker, "_telemetry_singleton", tel)
        return tel

    @pytest.mark.asyncio
    async def test_events_recent_returns_correct_shape(
        self, wired_telemetry: ToolCallTelemetry
    ) -> None:
        from prometheus.gateway.commands import cmd_events

        bus = SignalBus(telemetry=wired_telemetry)
        await bus.emit(_make_signal("skill_created", name="alpha"))
        await bus.emit(_make_signal("memory_updated", target="MEMORY.md"))

        out = cmd_events()
        assert "📡 /events recent" in out
        assert "skill_created" in out
        assert "memory_updated" in out
        # Event id appears bracketed in the list
        assert "[" in out and "]" in out

    @pytest.mark.asyncio
    async def test_events_skills_filters_to_skill_types(
        self, wired_telemetry: ToolCallTelemetry
    ) -> None:
        from prometheus.gateway.commands import cmd_events

        bus = SignalBus(telemetry=wired_telemetry)
        await bus.emit(_make_signal("skill_created", name="a"))
        await bus.emit(_make_signal("memory_updated", target="x"))
        await bus.emit(_make_signal("skill_refined", name="b"))

        out = cmd_events(arg="skills")
        assert "skill_created" in out
        assert "skill_refined" in out
        # The filter must EXCLUDE non-skill events.
        assert "memory_updated" not in out

    @pytest.mark.asyncio
    async def test_events_show_returns_full_payload(
        self, wired_telemetry: ToolCallTelemetry
    ) -> None:
        from prometheus.gateway.commands import cmd_events

        bus = SignalBus(telemetry=wired_telemetry)
        await bus.emit(_make_signal(
            "skill_created", name="show-me", path="/tmp/x.md", _source="SkillCreator",
        ))

        rows = wired_telemetry.signal_events_since(limit=1)
        assert len(rows) == 1
        event_id = rows[0]["id"]

        out = cmd_events(arg=f"show {event_id}")
        assert f"Event #{event_id}" in out
        assert "skill_created" in out
        assert "SkillCreator" in out
        # Full payload rendered as pretty JSON.
        assert '"name": "show-me"' in out
        assert '"path": "/tmp/x.md"' in out

    @pytest.mark.asyncio
    async def test_events_show_handles_missing_id(
        self, wired_telemetry: ToolCallTelemetry
    ) -> None:
        from prometheus.gateway.commands import cmd_events

        out = cmd_events(arg="show 99999")
        assert "no event with id=99999" in out

    def test_events_unknown_subcommand_helpful(
        self, wired_telemetry: ToolCallTelemetry
    ) -> None:
        from prometheus.gateway.commands import cmd_events

        out = cmd_events(arg="garbage")
        assert "unknown subcommand" in out.lower()
        # Suggests valid subcommands
        assert "recent" in out
        assert "skills" in out


# ---------------------------------------------------------------------------
# BEACON HTTP ENDPOINT
# ---------------------------------------------------------------------------


_fastapi_unavailable = False
try:  # noqa: SIM105
    import fastapi  # noqa: F401
    import starlette  # noqa: F401
except Exception:
    _fastapi_unavailable = True


@pytest.mark.skipif(
    _fastapi_unavailable,
    reason="fastapi/starlette not installed in this env (web is an optional dep)",
)
class TestApiEventsRecent:
    @pytest.mark.asyncio
    async def test_api_events_recent_returns_hydration_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET /api/events/recent returns the same shape Beacon expects."""
        tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
        from prometheus.telemetry import tracker as _tracker
        monkeypatch.setattr(_tracker, "_telemetry_singleton", tel)

        bus = SignalBus(telemetry=tel)
        await bus.emit(_make_signal(
            "skill_created", name="api-alpha", _source="SkillCreator",
        ))
        await bus.emit(_make_signal(
            "memory_updated", target="MEMORY.md", _source="MemoryTool",
        ))

        # Drive the endpoint via FastAPI's TestClient — exercises route
        # registration, query-string parsing, JSON shape.
        from prometheus.web.server import create_app
        from starlette.testclient import TestClient

        app = create_app(config={}, signal_bus=bus, session_mgr=None,
                         telemetry=tel)
        with TestClient(app) as client:
            resp = client.get("/api/events/recent?limit=10")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)
            assert len(data) == 2
            # Each row carries the keys Beacon needs.
            for row in data:
                assert set(row.keys()) >= {
                    "id", "timestamp", "signal_type",
                    "payload", "source_subsystem",
                }
            # Filter by type works.
            resp_typed = client.get("/api/events/recent?type=skill_created")
            data_typed = resp_typed.json()
            assert len(data_typed) == 1
            assert data_typed[0]["signal_type"] == "skill_created"
            assert data_typed[0]["payload"]["name"] == "api-alpha"


# Behavioural-only mirror of the FastAPI integration test — exercises the
# query path Beacon's endpoint depends on, independent of fastapi being
# installed. Catches regressions in the telemetry reader the endpoint
# wraps even when CI skips the HTTP test above.
class TestApiEventsRecentBehavioural:
    @pytest.mark.asyncio
    async def test_telemetry_reader_serves_beacon_hydration_shape(
        self, tmp_path: Path
    ) -> None:
        tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
        bus = SignalBus(telemetry=tel)
        await bus.emit(_make_signal(
            "skill_created", name="api-alpha", _source="SkillCreator",
        ))
        await bus.emit(_make_signal(
            "memory_updated", target="MEMORY.md", _source="MemoryTool",
        ))

        # Reader behind the endpoint
        rows = tel.signal_events_since(limit=10)
        assert len(rows) == 2
        for row in rows:
            assert set(row.keys()) >= {
                "id", "timestamp", "signal_type", "payload", "source_subsystem",
            }

        # Type filter mirrors `?type=skill_created`
        typed = tel.signal_events_since(signal_type="skill_created")
        assert len(typed) == 1
        assert typed[0]["payload"]["name"] == "api-alpha"
