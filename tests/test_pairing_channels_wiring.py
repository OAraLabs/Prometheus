"""The seams that connect the operator's channels to a running daemon.

The behaviour of each channel is tested where it lives (``test_pairing_ws_frames``, ``test_pairing_telegram``,
``test_pair_cli``). What is pinned HERE is that the daemon actually connects them, because a channel that is
correct and never attached is exactly the "config-dark" failure this repo keeps a rule against:

* the launcher attaches the pairing runtime to the WebSocket bridge, and attaches Telegram only when the
  owner opted in;
* ``oara pair`` is a registered subcommand and is dispatched;
* ``pairing.telegram_prompts`` is in the shipped template, off;
* ``oara doctor`` has a row that says so when Telegram prompts are on but nothing could receive one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

import prometheus.__main__ as entry
import prometheus.web.launcher as launcher
from prometheus.cli.doctor import check_pairing_telegram

REPO = Path(__file__).resolve().parents[1]


def _called_names(module) -> set[str]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }


# ── wiring ───────────────────────────────────────────────────────────────────

def test_the_launcher_attaches_the_pairing_runtime_to_the_bridge():
    assert "attach_pairing" in _called_names(launcher)


def test_the_launcher_attaches_telegram_prompts_through_the_adapter():
    assert "TelegramPairingPrompts" in _called_names(launcher)


def test_oara_pair_is_registered_and_dispatched():
    source = Path(entry.__file__).read_text(encoding="utf-8")
    assert "add_pair_subparser" in _called_names(entry)
    assert 'args.command == "pair"' in source and "run_pair_command" in source


# ── the shipped template ─────────────────────────────────────────────────────

def test_the_template_ships_telegram_prompts_off():
    template = yaml.safe_load((REPO / "config" / "prometheus.yaml.default").read_text(encoding="utf-8"))
    assert template["pairing"]["telegram_prompts"] is False


# ── doctor ───────────────────────────────────────────────────────────────────

def test_no_row_when_the_prompts_are_off():
    assert check_pairing_telegram({}) is None
    assert check_pairing_telegram({"pairing": {"telegram_prompts": False}}) is None


def _config(**gateway) -> dict:
    return {"pairing": {"telegram_prompts": True}, "gateway": gateway}


def test_on_with_a_private_chat_is_ok_and_says_how_many():
    row = check_pairing_telegram(_config(telegram_enabled=True, allowed_chat_ids=[7_000_001, -100123]))
    assert row.status == "ok" and "1 private chat" in row.message


def test_on_but_telegram_is_not_enabled_is_a_warning():
    row = check_pairing_telegram(_config(telegram_enabled=False, allowed_chat_ids=[7_000_001]))
    assert row.status == "warning" and "telegram_enabled" in (row.fix or row.message)


def test_on_with_only_groups_in_the_allowlist_is_a_warning():
    row = check_pairing_telegram(_config(telegram_enabled=True, allowed_chat_ids=[-1001234567890]))
    assert row.status == "warning" and "private" in row.message


def test_on_with_no_allowed_chats_is_a_warning():
    row = check_pairing_telegram(_config(telegram_enabled=True, allowed_chat_ids=[]))
    assert row.status == "warning"
