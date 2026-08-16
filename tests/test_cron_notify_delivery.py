"""Cron failure notifications must not lie about delivery.

On 2026-08-15 Telegram went dark for ~45 minutes. The `daily_news_briefing_pm`
job failed at 19:30:37 *because* its own send timed out, and the scheduler
then logged "Cron failure notification sent" — over the same dead channel.
The alert about the outage was destroyed by the outage.

The cause: `TelegramAdapter.send()` does not raise on a failed send. It
catches internally and returns `SendResult(success=False, error=...)`. The
notifier awaited it inside a try/except and treated "it returned" as "it
delivered", so the except branch was dead code for the failure that mattered.

Worse, the failed send still stamped the throttle — so the outage cost the
notification *and* suppressed an hour of real failures behind it.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

import prometheus.gateway.cron_scheduler as cs


class _Result:
    """Stands in for gateway.platform_base.SendResult."""

    def __init__(self, success: bool, error: str | None = None) -> None:
        self.success = success
        self.error = error


class _Gateway:
    """A gateway whose send outcome the test controls, like the real one."""

    def __init__(self, *, outcomes=None, raises: bool = False) -> None:
        self.sent: list[tuple[int, str]] = []
        self._outcomes = list(outcomes or [])
        self._raises = raises

    async def send(self, chat_id, text, **kw):  # noqa: ANN001
        if self._raises:
            raise RuntimeError("telegram down")
        self.sent.append((chat_id, text))
        if self._outcomes:
            return self._outcomes.pop(0)
        return _Result(True)


def _entry(name="daily_news_briefing_pm", *, status="failed", rc=1):
    return {
        "name": name,
        "command": "python3 -m prometheus.jobs.daily_briefing",
        "status": status,
        "returncode": rc,
        "stdout": "",
        "stderr": "ERROR provider returned an empty briefing",
    }


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Fresh spool + clean notifier state per test."""
    monkeypatch.setattr(cs, "get_data_dir", lambda: tmp_path)
    cs._NOTIFIER_GATEWAY = None
    cs._NOTIFIER_CHAT_ID = None
    cs._LAST_NOTIFY.clear()
    yield
    cs._NOTIFIER_GATEWAY = None
    cs._NOTIFIER_CHAT_ID = None
    cs._LAST_NOTIFY.clear()


# ---------------------------------------------------------------------------
# The lie
# ---------------------------------------------------------------------------


def test_failed_send_is_not_reported_as_sent(caplog):
    gw = _Gateway(outcomes=[_Result(False, "Timed out")])
    cs.set_cron_notifier(gw, 123)

    with caplog.at_level(logging.INFO):
        asyncio.run(cs._maybe_notify_failure(_entry()))

    assert "notification sent" not in caplog.text.lower()
    assert "NOT delivered" in caplog.text
    assert "Timed out" in caplog.text


def test_failed_send_is_spooled():
    gw = _Gateway(outcomes=[_Result(False, "Timed out")])
    cs.set_cron_notifier(gw, 123)

    asyncio.run(cs._maybe_notify_failure(_entry()))

    spooled = cs.load_undelivered()
    assert len(spooled) == 1
    assert spooled[0]["name"] == "daily_news_briefing_pm"
    assert spooled[0]["error"] == "Timed out"
    assert "rc=1" in spooled[0]["text"]


def test_raising_gateway_is_also_spooled():
    cs.set_cron_notifier(_Gateway(raises=True), 123)

    asyncio.run(cs._maybe_notify_failure(_entry()))

    spooled = cs.load_undelivered()
    assert len(spooled) == 1
    assert "RuntimeError" in spooled[0]["error"]


# ---------------------------------------------------------------------------
# The throttle must not be poisoned by a failed send
# ---------------------------------------------------------------------------


def test_undelivered_notification_does_not_start_the_cooldown():
    """One outage must not also silence the next hour of real failures."""
    gw = _Gateway(outcomes=[_Result(False, "Timed out"), _Result(True)])
    cs.set_cron_notifier(gw, 123)

    asyncio.run(cs._maybe_notify_failure(_entry()))   # fails, no throttle
    asyncio.run(cs._maybe_notify_failure(_entry()))   # must be attempted

    assert len(gw.sent) == 2
    assert "daily_news_briefing_pm" in cs._LAST_NOTIFY


def test_delivered_notification_does_start_the_cooldown():
    gw = _Gateway()
    cs.set_cron_notifier(gw, 123)

    asyncio.run(cs._maybe_notify_failure(_entry()))
    asyncio.run(cs._maybe_notify_failure(_entry()))

    assert len(gw.sent) == 1, "second send should be throttled"


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def test_flush_replays_spooled_notifications():
    cs.set_cron_notifier(_Gateway(outcomes=[_Result(False, "Timed out")]), 123)
    asyncio.run(cs._maybe_notify_failure(_entry("job_a")))
    assert len(cs.load_undelivered()) == 1

    recovered = _Gateway()
    cs.set_cron_notifier(recovered, 123)
    assert asyncio.run(cs.flush_undelivered()) == 1

    assert len(recovered.sent) == 1
    assert "job_a" in recovered.sent[0][1]
    assert cs.load_undelivered() == []


def test_flush_stops_at_the_first_failure_and_keeps_the_rest():
    cs.set_cron_notifier(_Gateway(outcomes=[_Result(False, "down")] * 3), 123)
    for name in ("job_a", "job_b", "job_c"):
        asyncio.run(cs._maybe_notify_failure(_entry(name)))
    assert len(cs.load_undelivered()) == 3

    # first replays, second fails — third must not be dropped
    partial = _Gateway(outcomes=[_Result(True), _Result(False, "down again")])
    cs.set_cron_notifier(partial, 123)
    assert asyncio.run(cs.flush_undelivered()) == 1

    remaining = cs.load_undelivered()
    assert [e["name"] for e in remaining] == ["job_b", "job_c"]


def test_flush_is_a_noop_without_a_spool_or_notifier():
    assert asyncio.run(cs.flush_undelivered()) == 0

    cs.set_cron_notifier(_Gateway(), 123)
    assert asyncio.run(cs.flush_undelivered()) == 0


def test_spool_survives_a_restart(tmp_path, monkeypatch):
    """The spool is on disk, so a daemon restart doesn't lose the alert."""
    cs.set_cron_notifier(_Gateway(outcomes=[_Result(False, "down")]), 123)
    asyncio.run(cs._maybe_notify_failure(_entry()))

    # simulate a restart: module state cleared, disk untouched
    cs._NOTIFIER_GATEWAY = None
    cs._LAST_NOTIFY.clear()

    assert len(cs.load_undelivered()) == 1


# ---------------------------------------------------------------------------
# Spool hygiene
# ---------------------------------------------------------------------------


def test_spool_is_capped():
    for i in range(cs.MAX_UNDELIVERED + 10):
        cs.spool_undelivered(f"job_{i}", f"text {i}", "down")

    spooled = cs.load_undelivered()
    assert len(spooled) == cs.MAX_UNDELIVERED
    assert spooled[-1]["name"] == f"job_{cs.MAX_UNDELIVERED + 9}"
    assert spooled[0]["name"] == "job_10", "oldest dropped, newest kept"


def test_corrupt_spool_lines_are_skipped():
    cs.spool_undelivered("job_a", "text", None)
    path = cs.get_undelivered_path()
    path.write_text(path.read_text() + "{not json\n", encoding="utf-8")

    spooled = cs.load_undelivered()
    assert len(spooled) == 1
    assert spooled[0]["name"] == "job_a"


def test_emptying_the_spool_removes_the_file():
    cs.spool_undelivered("job_a", "text", None)
    assert cs.get_undelivered_path().exists()

    cs.set_cron_notifier(_Gateway(), 123)
    asyncio.run(cs.flush_undelivered())

    assert not cs.get_undelivered_path().exists()


# ---------------------------------------------------------------------------
# Back-compat: absence of a result is not evidence of failure
# ---------------------------------------------------------------------------


def test_gateway_returning_none_still_counts_as_delivered(caplog):
    """Stub gateways predate SendResult — don't spool their sends."""

    class _LegacyGateway:
        def __init__(self) -> None:
            self.sent = []

        async def send(self, chat_id, text, **kw):  # noqa: ANN001
            self.sent.append((chat_id, text))

    gw = _LegacyGateway()
    cs.set_cron_notifier(gw, 123)

    with caplog.at_level(logging.INFO):
        asyncio.run(cs._maybe_notify_failure(_entry()))

    assert len(gw.sent) == 1
    assert "notification sent" in caplog.text.lower()
    assert cs.load_undelivered() == []


def test_successful_sendresult_is_not_spooled():
    cs.set_cron_notifier(_Gateway(outcomes=[_Result(True)]), 123)
    asyncio.run(cs._maybe_notify_failure(_entry()))
    assert cs.load_undelivered() == []


def test_spool_file_is_jsonl():
    cs.spool_undelivered("job_a", "line one\nline two", "down")
    lines = cs.get_undelivered_path().read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, "embedded newlines must not split the record"
    assert json.loads(lines[0])["text"] == "line one\nline two"
