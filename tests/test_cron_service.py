"""Timezone tests for prometheus.gateway.cron_service.next_run_time.

Added on branch fix/cron-briefing-tz. Proves cron expressions are evaluated in
local time (default America/New_York) with DST handled, not in UTC.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from prometheus.gateway.cron_service import next_run_time

NY = ZoneInfo("America/New_York")


def test_january_base_fires_0730_local_est() -> None:
    # Fixed JANUARY instant; naive base is interpreted as local NY time.
    nxt = next_run_time("30 7 * * *", base=datetime(2026, 1, 15, 12, 0), tz=NY)
    assert nxt.tzinfo is not None                       # tz-aware
    assert (nxt.hour, nxt.minute) == (7, 30)            # 07:30 LOCAL (not 03:30 UTC)
    assert nxt.utcoffset() == timedelta(hours=-5)       # EST
    assert nxt.date() == datetime(2026, 1, 16).date()


def test_july_base_fires_0730_local_edt_proves_dst() -> None:
    # Fixed JULY instant; same wall-clock 07:30 local, but EDT offset -> DST works.
    nxt = next_run_time("30 7 * * *", base=datetime(2026, 7, 15, 12, 0), tz=NY)
    assert (nxt.hour, nxt.minute) == (7, 30)
    assert nxt.utcoffset() == timedelta(hours=-4)       # EDT
    assert nxt.tzinfo is not None


def test_aware_utc_base_is_normalized_to_local() -> None:
    # 2026-01-15T17:00Z == 12:00 EST; next 07:30 is the following morning, EST.
    base = datetime(2026, 1, 15, 17, 0, tzinfo=timezone.utc)
    nxt = next_run_time("30 7 * * *", base=base, tz=NY)
    assert (nxt.hour, nxt.minute) == (7, 30)
    assert nxt.utcoffset() == timedelta(hours=-5)
    assert nxt.date() == datetime(2026, 1, 16).date()


def test_result_is_tz_aware() -> None:
    nxt = next_run_time("0 0 * * *", base=datetime(2026, 3, 1, 0, 0), tz=NY)
    assert nxt.tzinfo is not None


def test_no_longer_hardcodes_utc_base() -> None:
    # Regression: the old implementation did `base or datetime.now(timezone.utc)`,
    # so a naive base produced a naive/UTC result. Now the result is tz-aware and
    # offset from UTC by the local zone (never +00:00 for America/New_York).
    nxt = next_run_time("30 7 * * *", base=datetime(2026, 1, 15, 12, 0), tz=NY)
    assert nxt.tzinfo is not None
    assert nxt.utcoffset() != timedelta(0)


def test_default_timezone_is_not_utc() -> None:
    # No tz argument -> resolver uses gateway.cron_timezone or America/New_York.
    # NY is never at UTC+0, so a non-zero offset proves the base is not UTC.
    nxt = next_run_time("0 0 * * *")
    assert nxt.tzinfo is not None
    assert nxt.utcoffset() != timedelta(0)
