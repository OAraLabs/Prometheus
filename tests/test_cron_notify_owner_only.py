"""Cron failure and task notifications go only to the owner's private chat — never a group fallback.

The notifier target used to be `briefing_chat_id or allowed_chat_ids[0]`. When `briefing_chat_id` was unset and
the first allowed chat was a group, failure notices (and the heartbeat's task notices, which share the target)
went to that whole group. A Telegram private chat's id is positive and equals the user's id; a group or channel
id is negative and shared. The target is now the configured private `briefing_chat_id` only, and nothing when
it is unset or is a group.
"""

from __future__ import annotations

import logging

import pytest

from prometheus.config.shipped_defaults import resolve_notify_chat_id

OWNER_DM = 456
GROUP = -1_001_234_567_890


def test_a_private_briefing_chat_is_the_target():
    assert resolve_notify_chat_id({"briefing_chat_id": OWNER_DM, "allowed_chat_ids": [OWNER_DM, GROUP]}) == OWNER_DM


def test_no_briefing_chat_means_no_notifications_even_with_allowed_chats():
    # The old group fallback: allowed_chat_ids[0] is a group, and failure notices went there.
    assert resolve_notify_chat_id({"allowed_chat_ids": [GROUP, OWNER_DM]}) is None
    assert resolve_notify_chat_id({"allowed_chat_ids": [OWNER_DM]}) is None
    assert resolve_notify_chat_id({}) is None
    assert resolve_notify_chat_id(None) is None


def test_a_group_briefing_chat_is_refused_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_notify_chat_id({"briefing_chat_id": GROUP}) is None
    assert any("briefing_chat_id" in r.getMessage() and "private" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("value", ["not-a-number", "", 0, None, "  "])
def test_a_missing_or_unparseable_briefing_chat_is_no_target(value):
    assert resolve_notify_chat_id({"briefing_chat_id": value, "allowed_chat_ids": [GROUP]}) is None


def test_a_private_chat_given_as_a_numeric_string_is_accepted():
    assert resolve_notify_chat_id({"briefing_chat_id": str(OWNER_DM)}) == OWNER_DM
