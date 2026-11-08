"""Prometheus says so when nothing can answer an approval prompt.

``security.approval_queue.enabled`` ships false, and the queue is built only with the Telegram gateway. Without a
queue ``request_approval`` returns False at once, so every call that needs an approval is refused in seconds as
"Permission denied" and none is ever listed to approve. ``cautious`` reads as "asks the owner" and behaves as
"refuses everything"; a headless install ran that way for days. The daemon now warns at boot and ``oara doctor`` has
a row, both from ``permissions/approval_channel.py``, so they cannot disagree. Nothing is switched on for the operator.
"""

from __future__ import annotations

import inspect

import pytest

from prometheus.cli.doctor import check_approval_channel
from prometheus.permissions.approval_channel import approval_channel_gap


def _config(mode="default", *, enabled=False, telegram=False):
    return {
        "security": {"permission_mode": mode, "approval_queue": {"enabled": enabled}},
        "gateway": {"telegram_enabled": telegram},
    }


# ── the shipped default ──────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["default", "strict", "cautious"])
def test_a_mode_that_asks_with_no_queue_is_a_gap_and_names_the_key(mode):
    gap = approval_channel_gap(_config(mode))
    assert gap is not None
    assert "security.approval_queue.enabled is false" in gap.message
    assert "Permission denied" in gap.message
    assert "approval_queue.enabled: true" in gap.fix and "autonomous" in gap.fix


def test_an_empty_config_is_the_shipped_default_and_a_gap():
    assert approval_channel_gap({}) is not None
    assert approval_channel_gap(None) is not None


def test_autonomous_asks_nothing_so_there_is_nothing_to_answer():
    assert approval_channel_gap(_config("autonomous")) is None


# ── what the daemon would build ──────────────────────────────────────────────

def test_a_queue_with_the_telegram_gateway_is_no_gap():
    assert approval_channel_gap(_config("cautious", enabled=True, telegram=True)) is None


def test_the_queue_enabled_without_telegram_is_a_gap_that_says_why():
    gap = approval_channel_gap(_config("cautious", enabled=True, telegram=False))
    assert gap is not None
    assert "only built together with the Telegram gateway" in gap.message
    assert "telegram_enabled" in gap.message


# ── what the daemon knows at boot ────────────────────────────────────────────

def test_a_queue_that_was_built_is_no_gap_whatever_the_config_says():
    assert approval_channel_gap(_config("cautious"), queue_built=True) is None


def test_telegram_configured_but_the_queue_not_built_says_the_gateway_did_not_start():
    gap = approval_channel_gap(_config("cautious", enabled=True, telegram=True), queue_built=False)
    assert gap is not None and "did not start" in gap.message


def test_the_daemon_reports_the_gap_it_found():
    # The wiring: the startup warning reads the same function the doctor row does.
    import prometheus.daemon as daemon

    source = inspect.getsource(daemon)
    assert "approval_channel_gap(" in source
    assert "queue_built=" in source


# ── oara doctor ──────────────────────────────────────────────────────────────

def test_doctor_warns_with_the_fix_on_the_shipped_default():
    row = check_approval_channel(_config("cautious"))
    assert row.status == "warning"
    assert "Permission denied" in row.message
    assert row.fix and "approval_queue.enabled: true" in row.fix


def test_doctor_is_ok_when_something_can_answer():
    assert check_approval_channel(_config("cautious", enabled=True, telegram=True)).status == "ok"
    assert check_approval_channel(_config("autonomous")).status == "ok"


def test_doctor_lists_the_row():
    import prometheus.cli.doctor as doctor

    assert "check_approval_channel(config)" in inspect.getsource(doctor)
