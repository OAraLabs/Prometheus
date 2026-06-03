"""Tests for cron failure notifications (cron_scheduler.set_cron_notifier).

Covers the gap left by the heartbeat task-notifier: cron subprocesses run via
asyncio.create_subprocess_exec and never appear in BackgroundTaskManager, so
the heartbeat can't see them. This hook is what surfaces a failing cron job
(e.g. the daily briefing) to Telegram.
"""

from __future__ import annotations

import asyncio
import time

import prometheus.gateway.cron_scheduler as cs


class _FakeGateway:
    def __init__(self, *, raise_on_send: bool = False) -> None:
        self.sent: list[tuple[int, str]] = []
        self._raise = raise_on_send

    async def send(self, chat_id, text, **kw):  # noqa: ANN001
        if self._raise:
            raise RuntimeError("telegram down")
        self.sent.append((chat_id, text))


def _entry(name="daily_news_briefing", *, status="failed", rc=1,
           stderr="ERROR daily briefing failed: provider returned an empty briefing",
           command="python3 -m prometheus.jobs.daily_briefing"):
    return {
        "name": name,
        "command": command,
        "status": status,
        "returncode": rc,
        "stdout": "",
        "stderr": stderr,
    }


def _reset_state():
    cs._NOTIFIER_GATEWAY = None
    cs._NOTIFIER_CHAT_ID = None
    cs._LAST_NOTIFY.clear()


def test_failure_sends_one_message_with_diagnostic():
    _reset_state()
    gw = _FakeGateway()
    cs.set_cron_notifier(gw, 8139235390)
    asyncio.run(cs._maybe_notify_failure(_entry()))
    assert len(gw.sent) == 1
    chat, text = gw.sent[0]
    assert chat == 8139235390
    assert "daily_news_briefing" in text
    assert "rc=1" in text
    assert "empty briefing" in text          # stderr tail included
    assert "python3 -m prometheus" in text   # command preview included


def test_success_never_notifies():
    _reset_state()
    gw = _FakeGateway()
    cs.set_cron_notifier(gw, 1)
    asyncio.run(cs._maybe_notify_failure(_entry(status="success", rc=0)))
    assert gw.sent == []


def test_throttle_blocks_second_send_within_cooldown():
    _reset_state()
    gw = _FakeGateway()
    cs.set_cron_notifier(gw, 1)
    asyncio.run(cs._maybe_notify_failure(_entry()))
    asyncio.run(cs._maybe_notify_failure(_entry()))  # immediately again
    assert len(gw.sent) == 1


def test_throttle_releases_after_cooldown():
    _reset_state()
    gw = _FakeGateway()
    cs.set_cron_notifier(gw, 1)
    asyncio.run(cs._maybe_notify_failure(_entry()))
    # Pretend the last notification was >cooldown ago.
    cs._LAST_NOTIFY["daily_news_briefing"] = time.time() - cs.NOTIFY_COOLDOWN_SECONDS - 1
    asyncio.run(cs._maybe_notify_failure(_entry()))
    assert len(gw.sent) == 2


def test_throttle_is_per_job_name():
    _reset_state()
    gw = _FakeGateway()
    cs.set_cron_notifier(gw, 1)
    asyncio.run(cs._maybe_notify_failure(_entry(name="job_a")))
    asyncio.run(cs._maybe_notify_failure(_entry(name="job_b")))
    # Two different jobs both notify even within the cooldown window.
    assert len(gw.sent) == 2
    assert {gw.sent[0][1].split('\n', 1)[0], gw.sent[1][1].split('\n', 1)[0]} == {
        "⚠️ Cron job failed: job_a", "⚠️ Cron job failed: job_b",
    }


def test_no_notifier_is_a_silent_noop():
    _reset_state()  # gateway/chat_id stay None
    # Must not raise even though no notifier is wired.
    asyncio.run(cs._maybe_notify_failure(_entry()))


def test_setter_can_disable_by_passing_none():
    _reset_state()
    gw = _FakeGateway()
    cs.set_cron_notifier(gw, 1)
    cs.set_cron_notifier(None, None)
    asyncio.run(cs._maybe_notify_failure(_entry()))
    assert gw.sent == []


def test_send_failure_does_not_raise_and_does_not_consume_throttle():
    _reset_state()
    gw = _FakeGateway(raise_on_send=True)
    cs.set_cron_notifier(gw, 1)
    # Should not raise even though gateway.send blows up.
    asyncio.run(cs._maybe_notify_failure(_entry()))
    # And the throttle slot stayed open, so a recovered gateway can still notify.
    gw._raise = False
    asyncio.run(cs._maybe_notify_failure(_entry()))
    assert len(gw.sent) == 1
