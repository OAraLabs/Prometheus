"""What is the owner's alone goes only to the owner's PRIVATE Telegram chat — never a group fallback.

Three things the daemon sends have no chat of their own: tool-approval prompts for a request raised elsewhere (for
example a worker's, from a non-Telegram session), cron-failure notices with the resend queue, and the heartbeat's
background-task notices. Their target used to be the first allowed chat — ``allowed_chat_ids[0]`` for approvals and
``briefing_chat_id or allowed_chat_ids[0]`` for notices — so on a bot whose first allowed chat was a group, every
member of it saw them, approval prompts with their (redacted) arguments included.

A Telegram private chat's id is positive and equals the user's own id; a group, supergroup or channel id is
negative and shared. One resolver now picks the target for all of them: a private ``briefing_chat_id``, else the
owner's private chat, else nothing. A group is never chosen, not even when configured as ``briefing_chat_id``.
"""

from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from prometheus.config.shipped_defaults import resolve_owner_chat_id

OWNER_DM = 456
OTHER_DM = 789
GROUP = -1_001_234_567_890
DAEMON = Path(__file__).resolve().parent.parent / "src" / "prometheus" / "daemon.py"


# ── the resolver ─────────────────────────────────────────────────────────────

def test_a_private_briefing_chat_is_the_target():
    assert resolve_owner_chat_id({"briefing_chat_id": OWNER_DM, "allowed_chat_ids": [GROUP, OTHER_DM]}) == OWNER_DM


def test_without_a_briefing_chat_the_owners_private_chat_is_the_target_never_the_group():
    # The old fallback took allowed_chat_ids[0]: here, the group.
    assert resolve_owner_chat_id({"allowed_chat_ids": [GROUP, OWNER_DM]}) == OWNER_DM
    assert resolve_owner_chat_id({"allowed_chat_ids": [OWNER_DM]}) == OWNER_DM


def test_an_explicit_owner_is_the_target_when_only_groups_are_allowed():
    cfg = {"allowed_chat_ids": [GROUP], "owner_user_ids": [str(OWNER_DM)]}
    assert resolve_owner_chat_id(cfg) == OWNER_DM


def test_only_groups_and_no_owner_means_nothing_is_sent_and_says_so(caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_owner_chat_id({"allowed_chat_ids": [GROUP]}) is None
    assert any("private" in r.getMessage() and "briefing_chat_id" in r.getMessage() for r in caplog.records)


def test_no_telegram_chats_at_all_is_nothing_and_quiet(caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_owner_chat_id({}) is None
        assert resolve_owner_chat_id(None) is None
    assert not caplog.records, "a gateway with no chats configured has nothing to warn about here"


def test_a_group_briefing_chat_is_refused_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_owner_chat_id({"briefing_chat_id": GROUP}) is None
    assert any("briefing_chat_id" in r.getMessage() and "private" in r.getMessage() for r in caplog.records)


def test_a_refused_group_briefing_chat_falls_back_to_the_owners_private_chat():
    assert resolve_owner_chat_id({"briefing_chat_id": GROUP, "allowed_chat_ids": [GROUP, OWNER_DM]}) == OWNER_DM


@pytest.mark.parametrize("value", ["not-a-number", "", 0, None, "  ", True])
def test_an_unusable_briefing_chat_falls_through(value):
    assert resolve_owner_chat_id({"briefing_chat_id": value, "allowed_chat_ids": [GROUP]}) is None
    assert resolve_owner_chat_id({"briefing_chat_id": value, "allowed_chat_ids": [OWNER_DM]}) == OWNER_DM


def test_a_private_chat_given_as_a_numeric_string_is_accepted():
    assert resolve_owner_chat_id({"briefing_chat_id": str(OWNER_DM)}) == OWNER_DM


# ── the approval queue, fed the resolver's answer ────────────────────────────

def test_an_approval_prompt_with_no_chat_of_its_own_goes_to_the_owner_not_the_group():
    from prometheus.permissions.approval_queue import ApprovalQueue
    from prometheus.permissions.checker import SecurityGate

    tg = AsyncMock()
    tg.send = AsyncMock()
    default_chat = resolve_owner_chat_id({"allowed_chat_ids": [GROUP, OWNER_DM]})
    queue = ApprovalQueue(security_gate=SecurityGate(), telegram_adapter=tg, timeout_seconds=0.1,
                          default_chat_id=default_chat)
    asyncio.run(queue.request_approval("bash", "git push"))
    targets = {call[0][0] for call in tg.send.call_args_list}
    assert targets == {OWNER_DM}, targets


def test_with_no_private_chat_an_approval_prompt_is_not_posted_to_the_group():
    from prometheus.permissions.approval_queue import ApprovalQueue
    from prometheus.permissions.checker import SecurityGate

    tg = AsyncMock()
    tg.send = AsyncMock()
    queue = ApprovalQueue(security_gate=SecurityGate(), telegram_adapter=tg, timeout_seconds=0.1,
                          default_chat_id=resolve_owner_chat_id({"allowed_chat_ids": [GROUP]}))
    asyncio.run(queue.request_approval("bash", "git push"))
    assert not tg.send.call_args_list, "nothing goes to Telegram; the request still waits in Beacon and the API"


# ── the daemon's wiring (static, like test_config_defaults_equality) ─────────

def _daemon() -> ast.Module:
    return ast.parse(DAEMON.read_text())


def test_the_daemon_never_takes_the_first_allowed_chat_as_a_target():
    bad = [n.lineno for n in ast.walk(_daemon())
           if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value == 0
           and "allowed_chat_ids" in ast.unparse(n.value)]
    assert not bad, f"daemon.py picks allowed_chat_ids[0] as a chat at line(s) {bad}"


def _assigned_from_resolver(tree: ast.Module, name: str) -> bool:
    return any(isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets)
               and "resolve_owner_chat_id" in ast.unparse(n.value) for n in ast.walk(tree)) or any(
        isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets)
        and isinstance(n.value, ast.Name) and _assigned_from_resolver(tree, n.value.id) for n in ast.walk(tree))


def test_the_approval_queue_default_chat_comes_from_the_owner_resolver():
    tree = _daemon()
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "ApprovalQueue"]
    assert calls, "the daemon builds an ApprovalQueue"
    for call in calls:
        value = {k.arg: k.value for k in call.keywords}["default_chat_id"]
        assert isinstance(value, ast.Name) and _assigned_from_resolver(tree, value.id), ast.unparse(value)


@pytest.mark.parametrize("callee, keyword", [("Heartbeat", "notify_chat_id"), ("set_cron_notifier", None)])
def test_the_notices_target_comes_from_the_owner_resolver(callee, keyword):
    tree = _daemon()
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == callee]
    assert calls, callee
    for call in calls:
        value = {k.arg: k.value for k in call.keywords}[keyword] if keyword else call.args[1]
        assert isinstance(value, ast.Name) and _assigned_from_resolver(tree, value.id), ast.unparse(value)
