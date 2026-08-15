"""Absence is not permission — gateway.telegram_enabled + allowed_chat_ids.

THE DEFECT, AND WHY IT IS ONE DEFECT AND NOT TWO
------------------------------------------------
``daemon`` read ``gateway_config.get("telegram_enabled", True)`` while the
template, both setup-wizard display surfaces and every sibling gateway said
False. So a config that merely OMITTED the key started the gateway as soon as
a token existed — and a token only has to sit in the environment.

``allowed_chat_ids`` is the adjacent key in the same section, and
``chat_allowed`` returned True on an empty list: "no restrictions set". The
config that omits one omits the other — they are trimmed together — so the
expected shape of the failure was a public bot live to EVERY chat, driving an
agent with shell access, while the setup wizard's status panel reported
Telegram off.

Both keys read absence as permission. Fixing either alone leaves the blast
radius intact.

WHY THESE TESTS DRIVE ENTRY POINTS
-----------------------------------
Every §2e gap this month was a test that called the component instead of the
caller: ``start_task`` tested by tests that called ``start_task``; the
SecurityGate tested by tests that passed ``file_path=`` by hand. So the
assertions here are on the far side —

  * the REAL ``run_daemon`` gateway-construction path decides whether an
    adapter exists, rather than a resolver being asked directly;
  * the REAL ``_authorize_update`` handler decides whether an update is
    dropped, rather than ``chat_allowed`` being called by the test.

The resolver-level tests are kept too, but they are the cheap half. If only
they existed, this file would prove exactly what the last four defects proved.
"""

from __future__ import annotations

import logging

import pytest

from prometheus.config.shipped_defaults import (
    SHIPPED_TELEGRAM_ENABLED,
    resolve_allowed_chat_ids,
    resolve_telegram_enabled,
)
from prometheus.gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Resolver layer — the cheap half.
# ---------------------------------------------------------------------------

def test_absent_telegram_enabled_resolves_off():
    """The whole defect in one line: absence used to mean ON."""
    assert resolve_telegram_enabled({}) is False
    assert resolve_telegram_enabled(None) is False
    assert SHIPPED_TELEGRAM_ENABLED is False


def test_explicit_telegram_enabled_is_honoured_in_both_directions():
    """Admission, not only refusal (§2c): an operator who says on gets on."""
    assert resolve_telegram_enabled({"telegram_enabled": True}) is True
    assert resolve_telegram_enabled({"telegram_enabled": False}) is False


@pytest.mark.parametrize("cfg", [{}, None, {"allowed_chat_ids": []},
                                 {"allowed_chat_ids": None},
                                 {"allowed_chat_ids": "not-a-list"}])
def test_absent_or_empty_allowlist_resolves_empty_never_permissive(cfg):
    assert resolve_allowed_chat_ids(cfg) == []


def test_allowlist_entries_are_coerced_and_malformed_ones_dropped():
    """Fail-closed on garbage, and never by exception (CROSS-CUTTING §8)."""
    assert resolve_allowed_chat_ids({"allowed_chat_ids": [1, "2", None, "x", 3]}) \
        == [1, 2, 3]


# ---------------------------------------------------------------------------
# Runtime layer — the REAL authorize handler, not chat_allowed().
# ---------------------------------------------------------------------------

class _Chat:
    def __init__(self, cid): self.id = cid


class _Update:
    def __init__(self, cid):
        self.effective_chat = _Chat(cid)
        self.effective_user = None
        self.message = None


def _adapter_with(allowed):
    """A TelegramAdapter whose authorize handler we can drive directly.

    Built via ``__new__`` so no network/token is needed: the handler reads only
    ``self.config``, and constructing the full adapter would drag in
    python-telegram-bot's Application. What is exercised is the REAL
    ``_authorize_update`` coroutine, not a reimplementation of its rule.
    """
    from prometheus.gateway.telegram import TelegramAdapter

    a = TelegramAdapter.__new__(TelegramAdapter)
    a.config = PlatformConfig(platform=Platform.TELEGRAM, token="t",
                              allowed_chat_ids=allowed)
    return a


async def _authorized(adapter, chat_id) -> bool:
    """True if the real handler let the update through."""
    from telegram.ext import ApplicationHandlerStop

    try:
        await adapter._authorize_update(_Update(chat_id), None)
    except ApplicationHandlerStop:
        return False
    return True


@pytest.mark.asyncio
async def test_real_authorize_handler_denies_every_chat_when_allowlist_empty():
    """The unbounded half. This used to authorize the entire internet."""
    adapter = _adapter_with([])
    for chat_id in (1, -1001234567890, 8139235390):
        assert not await _authorized(adapter, chat_id), (
            f"chat {chat_id} was authorized by an EMPTY allowlist — "
            f"'no restrictions set' is what made absence mean permission"
        )


@pytest.mark.asyncio
async def test_real_authorize_handler_admits_a_listed_chat_and_denies_others():
    """Admission half — the one that matters, because it is the live gateway."""
    adapter = _adapter_with([8139235390])
    assert await _authorized(adapter, 8139235390), (
        "a chat ON the allowlist was refused — this rule guards the operator's "
        "own primary gateway and must open for it"
    )
    assert not await _authorized(adapter, 999)


# ---------------------------------------------------------------------------
# Construction layer — the REAL daemon gateway-construction path.
# ---------------------------------------------------------------------------

def _daemon_guard_source() -> str:
    from pathlib import Path
    return (Path(__file__).resolve().parent.parent / "src" / "prometheus"
            / "daemon.py").read_text(encoding="utf-8")


def test_daemon_guard_has_no_permissive_default_in_CODE():
    """No live `.get("telegram_enabled", True)` anywhere in daemon.py.

    Asserted over the AST, not over the text. The first draft of this test was
    a substring ban and it went red against the COMMENT that documents the old
    expression — §3c exactly: the negation of a claim contains the claim, so a
    substring ban cannot tell a defect from its own post-mortem, and the way to
    make it pass is to delete the explanation. The AST sees code only.
    """
    import ast

    tree = ast.parse(_daemon_guard_source())
    offenders = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "telegram_enabled"
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value is True):
            offenders.append(node.lineno)
    assert not offenders, (
        f"daemon.py:{offenders} defaults telegram_enabled to True again. "
        f"Absence must not start a public gateway — route the read through "
        f"shipped_defaults.resolve_telegram_enabled."
    )


def test_daemon_construction_is_gated_on_BOTH_resolvers():
    """The construction decision, pinned structurally.

    HONEST LIMIT, stated rather than implied: `run_daemon` cannot be driven
    cheaply here — it constructs the whole subsystem graph. So this asserts
    that daemon.py's gateway guard is built from the two resolvers, rather
    than re-deriving the guard's boolean in the test. Re-deriving it would
    reproduce the author's reading in the assertion and pass either way, which
    is the failure this file's docstring is about. The behavioural proof for
    this layer is the live outcome run recorded in the PR, not this test.
    """
    src = _daemon_guard_source()
    assert "resolve_telegram_enabled(gateway_config)" in src
    assert "resolve_allowed_chat_ids(gateway_config)" in src
    assert "_tg_enabled and not _tg_chat_ids" in src, (
        "the refusal branch is gone — an enabled gateway with an empty "
        "allowlist must not construct an adapter"
    )
    assert "allowed_chat_ids=_tg_chat_ids" in src, (
        "PlatformConfig is being built from raw config again, bypassing the "
        "resolver that drops malformed entries and never returns a permissive "
        "value"
    )


def test_config_writers_write_both_keys_explicitly():
    """A writer that emits one key and not the other recreates the defect."""
    from prometheus.cli.init import _default_config

    gw = _default_config(None, None)["gateway"]
    assert "telegram_enabled" in gw and "allowed_chat_ids" in gw, (
        f"cli/init must write BOTH keys; wrote {sorted(gw)}"
    )
    assert gw["telegram_enabled"] is False
