"""AutoDream decay is charged by elapsed time, not once per cycle.

Before: every consolidation pass subtracted ``decay_rate x (periods overdue)``
from the CURRENT confidence, with no memory of the last pass. So how often
AutoDream ran set how fast facts decayed. At 48 cycles a day, a fact 90+ days
unmentioned lost 0.15 per cycle and reached the 0.1 floor within hours. Live,
on 2026-10-03: all 2,121 stale facts of 2,173 sat at exactly 0.1, and the
oldest of them were last mentioned only 120-150 days before.

Now a fact loses ``decay_rate`` for every 30 days it spends stale, however
many passes that time is split across. A watermark in the store records how
far decay has been charged, so a restart never charges a window twice. The
delete step (``_tombstone``) and its threshold are deliberately untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

DAY = 86400.0
T0 = 1_800_000_000.0


def _store(db_path):
    # Same import route as tests/test_sentinel.py: avoid memory/__init__ cycles.
    import importlib.util

    if "prometheus.memory.store" in sys.modules:
        return sys.modules["prometheus.memory.store"].MemoryStore(db_path=db_path)
    path = Path(__file__).resolve().parents[1] / "src" / "prometheus" / "memory" / "store.py"
    spec = importlib.util.spec_from_file_location("prometheus.memory.store", str(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prometheus.memory.store"] = mod
    spec.loader.exec_module(mod)
    return mod.MemoryStore(db_path=db_path)


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    from prometheus.sentinel import memory_consolidator as mc

    c = _Clock(T0)
    monkeypatch.setattr(mc, "time", SimpleNamespace(time=c.time))
    return c


def _fact(store, *, conf: float, last_mentioned: float, name: str = "Bob") -> str:
    mid = store.persist_memory("person", name, f"{name} likes pizza", conf,
                               source_event_ids=["s"])
    store.update_memory(mid, last_mentioned=last_mentioned)
    return mid


def _conf(store, mid) -> float:
    mem = store.get_memory(mid)
    assert mem is not None
    return mem["confidence"]


def _consolidator(store):
    from prometheus.sentinel.memory_consolidator import MemoryConsolidator

    return MemoryConsolidator(store, stale_days=90, decay_rate=0.05, min_confidence=0.1)


class TestElapsedDecay:
    def test_running_every_half_hour_decays_the_same_as_running_once(self, tmp_path, clock):
        """The bug: cadence set the decay speed. 48 passes = 1 pass over the same day."""
        often = _store(tmp_path / "often.db")
        once = _store(tmp_path / "once.db")
        last = T0 - 100 * DAY  # 10 days past the 90-day cutoff
        a = _fact(often, conf=0.7, last_mentioned=last)
        b = _fact(once, conf=0.7, last_mentioned=last)
        c_often, c_once = _consolidator(often), _consolidator(once)

        c_often.consolidate()
        c_once.consolidate()
        for _ in range(48):  # one day of 30-minute cycles
            clock.now += 1800
            c_often.consolidate()
        c_once.consolidate()

        assert _conf(often, a) == pytest.approx(_conf(once, b), abs=1e-9)
        assert _conf(often, a) == pytest.approx(0.7 - 0.05 * 11 / 30, abs=1e-9), \
            "11 days stale at 0.05 per 30 days"

    def test_a_fact_loses_the_rate_per_thirty_stale_days(self, tmp_path, clock):
        store = _store(tmp_path / "m.db")
        mid = _fact(store, conf=0.8, last_mentioned=T0 - 90 * DAY)  # just went stale
        con = _consolidator(store)
        con.consolidate()
        assert _conf(store, mid) == pytest.approx(0.8)
        for _ in range(30):
            clock.now += DAY
            con.consolidate()
        assert _conf(store, mid) == pytest.approx(0.75, abs=1e-9)

    def test_time_before_a_fact_went_stale_is_never_charged(self, tmp_path, clock):
        store = _store(tmp_path / "m.db")
        mid = _fact(store, conf=0.6, last_mentioned=T0 - 80 * DAY)  # stale in 10 days
        con = _consolidator(store)
        con.consolidate()
        clock.now += 15 * DAY  # 5 of these 15 days are stale
        con.consolidate()
        assert _conf(store, mid) == pytest.approx(0.6 - 0.05 * 5 / 30, abs=1e-9)

    def test_a_restart_never_charges_the_same_window_twice(self, tmp_path, clock):
        db = tmp_path / "m.db"
        store = _store(db)
        mid = _fact(store, conf=0.7, last_mentioned=T0 - 120 * DAY)
        _consolidator(store).consolidate()
        after_first = _conf(store, mid)
        assert after_first == pytest.approx(0.7 - 0.05 * 30 / 30, abs=1e-9)

        clock.now += 3600  # the daemon restarts: new store handle, new consolidator
        reopened = _store(db)
        _consolidator(reopened).consolidate()
        assert _conf(reopened, mid) == pytest.approx(after_first - 0.05 * (3600 / DAY) / 30,
                                                     abs=1e-9)

    def test_a_fresh_fact_is_untouched(self, tmp_path, clock):
        store = _store(tmp_path / "m.db")
        mid = _fact(store, conf=0.7, last_mentioned=T0 - 10 * DAY)
        con = _consolidator(store)
        for _ in range(100):
            clock.now += 1800
            con.consolidate()
        assert _conf(store, mid) == pytest.approx(0.7)

    def test_decay_still_stops_at_the_floor(self, tmp_path, clock):
        store = _store(tmp_path / "m.db")
        mid = _fact(store, conf=0.3, last_mentioned=T0 - 2_000 * DAY)
        result = _consolidator(store).consolidate()
        assert _conf(store, mid) == pytest.approx(0.1)
        assert result.confidence_decayed == 1
        assert result.tombstoned == 0, "the floor equals the delete threshold: unchanged"


class TestStoreMarks:
    def test_a_mark_round_trips_and_survives_reopening(self, tmp_path):
        db = tmp_path / "m.db"
        store = _store(db)
        assert store.get_mark("decay_charged_through") is None
        store.set_mark("decay_charged_through", T0)
        assert store.get_mark("decay_charged_through") == T0
        assert _store(db).get_mark("decay_charged_through") == T0
