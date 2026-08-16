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


@pytest.mark.parametrize(
    "gateway_cfg, token, should_start, refuses_loudly, why",
    [
        ({}, "tok", False, False,
         "key ABSENT + token present — the original defect"),
        ({"telegram_enabled": True}, "tok", False, True,
         "enabled, NO allowlist — must refuse AND say so"),
        ({"telegram_enabled": True, "allowed_chat_ids": []}, "tok", False, True,
         "enabled, EXPLICITLY empty allowlist — same refusal. The template "
         "ships [] as its placeholder, so honouring it verbatim (the #141 "
         "rule) would open every fresh install"),
        ({"telegram_enabled": True, "allowed_chat_ids": ["x"]}, "tok", False,
         True, "allowlist of only malformed ids resolves empty — still refused"),
        ({"telegram_enabled": False, "allowed_chat_ids": [1]}, "tok", False,
         False, "explicitly off — not a refusal, nothing to report"),
        ({"telegram_enabled": True, "allowed_chat_ids": [1]}, "", False, False,
         "no token — nothing configured, not a refusal"),
        ({"telegram_enabled": True, "allowed_chat_ids": [8139235390]}, "tok",
         True, False, "ADMISSION: enabled and allowlisted — must start"),
    ],
)
def test_daemon_start_decision(gateway_cfg, token, should_start,
                               refuses_loudly, why):
    """The daemon's REAL decision function, not a boolean re-derived here.

    This was an inline expression inside `run_daemon` and a mutation that
    neutered it (`if False and ...`) survived the whole suite — the second
    layer (`chat_allowed`) kept denying messages, so nothing went red while
    the daemon started an unrestricted gateway that silently ignored everyone.
    Extracting the decision is what made it assertable at all.
    """
    from prometheus.daemon import telegram_gateway_decision

    starts, refusal = telegram_gateway_decision(gateway_cfg, token)
    assert starts is should_start, why
    assert bool(refusal) is refuses_loudly, (
        f"{why} — refusal text was {refusal!r}")
    if refusal:
        assert "allowed_chat_ids" in refusal, (
            "the refusal must name the key the operator has to fix")


def test_daemon_calls_the_decision_and_the_branch_is_not_dead():
    """AST pin: the construction branch is gated on the decision's result.

    A substring check was what let the neutered-branch mutation through — the
    mutated line `if False and ... _tg_enabled and not _tg_chat_ids:` still
    CONTAINED the expected text. This walks the tree instead: the guard must
    be the plain name the decision assigned, with no `False and` wrapper.
    """
    import ast

    tree = ast.parse(_daemon_guard_source())
    assigns_decision = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "telegram_gateway_decision"
        for n in ast.walk(tree)
    )
    assert assigns_decision, (
        "daemon.py no longer calls telegram_gateway_decision — the start/refuse "
        "rule has been inlined again, where it cannot be tested"
    )
    guards = [n for n in ast.walk(tree)
              if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
              and n.test.id == "_tg_start"]
    assert guards, (
        "no `if _tg_start:` branch — the adapter must be constructed only when "
        "the decision says so, and the guard must be the bare name so a "
        "`False and ...` wrapper cannot hide inside it"
    )


def test_daemon_actually_LOGS_the_refusal():
    """Silence is not a pass — the operator must be told why.

    An earlier draft of this built the condition and then called
    ``logger.error`` ITSELF before asserting the record existed. It proved
    that a test can call a logger; a mutation deleting the daemon's real log
    call sailed straight through. §2b in its purest form — the check answered
    cleanly, about the wrong subject. Worse, when that draft was later edited
    out, nothing replaced it, so for a while the refusal had no assertion at
    all and a surviving mutation was the only thing that said so.

    This walks daemon.py: the refusal must be reported inside a branch guarded
    by the bare ``_tg_refusal`` name, so neither deleting the call nor
    wrapping the guard in ``False and ...`` can hide.
    """
    import ast

    tree = ast.parse(_daemon_guard_source())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "_tg_refusal"):
            continue
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr in ("error", "warning")
               and isinstance(n.func.value, ast.Name)
               and n.func.value.id == "logger"
               for n in ast.walk(node)):
            return
    raise AssertionError(
        "daemon.py does not log the Telegram refusal under a bare "
        "`if _tg_refusal:` guard. A gateway that refuses to start without "
        "saying why is indistinguishable from one that is simply off, and the "
        "operator has nothing to act on (§2c)."
    )


def test_config_writers_write_both_keys_explicitly():
    """A writer that emits one key and not the other recreates the defect."""
    from prometheus.cli.init import _default_config

    gw = _default_config(None, None)["gateway"]
    assert "telegram_enabled" in gw and "allowed_chat_ids" in gw, (
        f"cli/init must write BOTH keys; wrote {sorted(gw)}"
    )
    assert gw["telegram_enabled"] is False
