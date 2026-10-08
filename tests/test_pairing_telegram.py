"""Telegram pairing prompts: opt-in, plain text, private chats only, and the same decision as everywhere.

Telegram is a third-party relay for a message that carries a source address and a match code, so it is
OFF by default (``pairing.telegram_prompts``). When the owner turns it on:

* the prompt is sent as PLAIN text, because the device name is typed on the new device and anyone on the
  network can make it ``*bold*`` or a link. ``TelegramAdapter.send`` falls back to ``config.parse_mode``
  when given ``None``, so the prompt goes to the bot API directly with an explicit ``parse_mode=None``;
* it goes only to PRIVATE chats in ``allowed_chat_ids`` (positive ids). A group in that list gets nothing:
  membership is not an identity, and any member could otherwise approve;
* a tap runs the SAME decision the REST route runs (``PairingRuntime.approve`` / ``deny``), as
  ``telegram:<chat>``, behind the adapter's existing group -1 authorisation;
* every message is edited to its final state and loses its buttons, whoever decided and wherever.

The contract listed one claim as unverified: that ``_authorize_update`` covers a callback query because the
update carries a chat. It is tested here with real ``telegram`` objects.
"""

from __future__ import annotations

import datetime as dt
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import CallbackQuery, Chat, Message, Update, User
from telegram.ext import ApplicationHandlerStop

from prometheus.config.pair_requests import PairingSettings
from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.telegram import TelegramAdapter
from prometheus.gateway.telegram_pairing import CALLBACK_PATTERN, TelegramPairingPrompts
from prometheus.tools.base import ToolRegistry
from tests.support.pairing_world import T0, World

OWNER = 7_000_001            # a private chat (positive id)
SECOND_OWNER = 7_000_002
GROUP = -1_001_234_567_890   # a supergroup, in the allowlist by mistake
STRANGER = 999_000_111


def _adapter(allowed: list[int]) -> TelegramAdapter:
    cfg = PlatformConfig(platform=Platform.TELEGRAM, token="test-token", allowed_chat_ids=allowed)
    agent_loop = AsyncMock()
    agent_loop._model_router = None
    adapter = TelegramAdapter(config=cfg, agent_loop=agent_loop, tool_registry=ToolRegistry(),
                              model_name="test-model-v1", model_provider="llama_cpp")
    sent: list[dict] = []
    counter = iter(range(100, 10_000))

    async def send_message(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(message_id=next(counter))

    adapter._app = MagicMock()
    adapter._app.bot.send_message = AsyncMock(side_effect=send_message)
    adapter._app.bot.edit_message_text = AsyncMock()
    adapter.sent = sent                      # type: ignore[attr-defined]
    return adapter


def _world(tmp_path, **pairing) -> World:
    return World(tmp_path, recording=False, **pairing)


def _prompts(world: World, adapter: TelegramAdapter) -> TelegramPairingPrompts:
    return TelegramPairingPrompts(adapter, world.runtime)


def _on(world: World, adapter: TelegramAdapter) -> TelegramPairingPrompts:
    world.runtime.settings = PairingSettings(telegram_prompts=True)
    prompts = _prompts(world, adapter)
    assert prompts.attach() is True
    return prompts


def _callback(chat_id: int, data: str, *, chat_type: str = "private", message_id: int = 77):
    query = MagicMock()
    query.data = data
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    query.message = SimpleNamespace(message_id=message_id, chat=SimpleNamespace(id=chat_id, type=chat_type))
    update = MagicMock()
    update.callback_query = query
    update.effective_chat = SimpleNamespace(id=chat_id, type=chat_type)
    update.effective_user = SimpleNamespace(id=4242)
    return update, query


# ── off by default ───────────────────────────────────────────────────────────

def test_the_setting_defaults_to_off_and_reads_a_boolean():
    assert PairingSettings().telegram_prompts is False
    assert PairingSettings.from_config(None).telegram_prompts is False
    assert PairingSettings.from_config({"pairing": {"telegram_prompts": True}}).telegram_prompts is True
    assert PairingSettings.from_config({"pairing": {"telegram_prompts": False}}).telegram_prompts is False


def test_a_setting_that_is_not_a_boolean_is_off_and_says_so(caplog):
    with caplog.at_level("WARNING"):
        s = PairingSettings.from_config({"pairing": {"telegram_prompts": "yes please"}})
    assert s.telegram_prompts is False
    assert any("telegram_prompts" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_with_it_off_nothing_is_registered_and_nothing_is_sent(tmp_path):
    world = _world(tmp_path)
    adapter = _adapter([OWNER])
    prompts = _prompts(world, adapter)
    assert prompts.attach() is False
    adapter._app.add_handler.assert_not_called()
    response, _, _ = world.request()
    assert response.status_code == 201 and response.json()["notified"] is False
    assert adapter.sent == []


# ── the prompt ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_request_is_sent_to_each_private_chat_as_plain_text_with_two_buttons(tmp_path):
    world = _world(tmp_path)
    adapter = _adapter([OWNER, SECOND_OWNER])
    _on(world, adapter)
    created, _, _ = world.created(source="192.0.2.42")
    assert created["notified"] is True
    assert sorted(m["chat_id"] for m in adapter.sent) == [OWNER, SECOND_OWNER]
    message = adapter.sent[0]
    assert message["parse_mode"] is None, "explicitly plain text, never the config's MarkdownV2"
    assert message["text"] == (
        "Jennifer's MacBook wants to connect.\n"
        f"Code {created['match_code']}\n\n"
        "macOS · 192.0.2.42 · expires in 5 min\n"
        "The name is typed on the new device and is not checked. "
        "Approve only if the code matches the one on its screen.")
    buttons = [b for row in message["reply_markup"].inline_keyboard for b in row]
    assert [b.text for b in buttons] == ["Approve", "Deny"]
    assert [b.callback_data for b in buttons] == [f"pair:a:{created['request_id']}", f"pair:d:{created['request_id']}"]
    assert all(len(b.callback_data.encode()) <= 64 for b in buttons), "Telegram's callback_data limit"


@pytest.mark.asyncio
async def test_a_group_in_the_allowlist_gets_nothing(tmp_path):
    world = _world(tmp_path)
    adapter = _adapter([GROUP])
    prompts = _prompts(world, adapter)
    world.runtime.settings = PairingSettings(telegram_prompts=True)
    prompts.attach()
    created, _, _ = world.created()
    assert adapter.sent == [] and created["notified"] is False


@pytest.mark.asyncio
async def test_only_the_private_chat_among_a_mixed_allowlist_is_messaged(tmp_path):
    world = _world(tmp_path)
    adapter = _adapter([GROUP, OWNER])
    _on(world, adapter)
    world.created()
    assert [m["chat_id"] for m in adapter.sent] == [OWNER]


@pytest.mark.asyncio
async def test_a_name_that_looks_like_markdown_is_sent_verbatim_and_unparsed(tmp_path):
    world = _world(tmp_path)
    adapter = _adapter([OWNER])
    _on(world, adapter)
    world.created(device_name="*Will's phone* [click](http://evil.example)")
    assert "*Will's phone* [click](http://evil.example) wants to connect." in adapter.sent[0]["text"]
    assert adapter.sent[0]["parse_mode"] is None


@pytest.mark.asyncio
async def test_a_failed_send_does_not_break_the_request_or_claim_it_was_told(tmp_path):
    world = _world(tmp_path)
    adapter = _adapter([OWNER])
    adapter._app.bot.send_message = AsyncMock(side_effect=ConnectionError("telegram is down"))
    _on(world, adapter)
    response, _, _ = world.request()
    assert response.status_code == 201 and response.json()["notified"] is False


@pytest.mark.asyncio
async def test_a_bot_that_is_not_running_is_not_a_channel(tmp_path):
    world = _world(tmp_path)
    adapter = _adapter([OWNER])
    adapter._app = None
    world.runtime.settings = PairingSettings(telegram_prompts=True)
    assert _prompts(world, adapter).attach() is False
    assert world.request()[0].json()["notified"] is False


def test_the_handler_is_registered_only_with_the_setting_on(tmp_path):
    world = _world(tmp_path)
    adapter = _adapter([OWNER])
    _on(world, adapter)
    (call,) = adapter._app.add_handler.call_args_list
    handler = call.args[0]
    assert handler.pattern.pattern == CALLBACK_PATTERN.pattern


def test_the_callback_pattern_matches_only_our_own_data():
    rid = "a" * 32
    assert CALLBACK_PATTERN.fullmatch(f"pair:a:{rid}") and CALLBACK_PATTERN.fullmatch(f"pair:d:{rid}")
    for bad in (f"pair:x:{rid}", f"pair:a:{'A' * 32}", f"pair:a:{'a' * 31}", f"pair:a:{'a' * 33}",
                f"pair:a:{rid}\n", f"xpair:a:{rid}", f"pair:a:{rid}x", "", "approve:abc"):
        assert not CALLBACK_PATTERN.fullmatch(bad), bad


# ── a tap ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_approve_from_an_allowed_private_chat_decides_as_that_chat(tmp_path):
    world = _world(tmp_path)
    adapter = _adapter([OWNER])
    prompts = _on(world, adapter)
    created, requester, _ = world.created()
    update, query = _callback(OWNER, f"pair:a:{created['request_id']}")
    await prompts.on_callback(update, MagicMock())
    polled = world.poll(created).json()
    assert polled["status"] == "approved"
    assert requester.unseal(created["request_id"], polled["sealed"])["name"] == "Jennifer's MacBook"
    query.answer.assert_awaited_once()
    kwargs = query.edit_message_text.await_args.kwargs
    assert re.fullmatch(r"Jennifer's MacBook — approved from Telegram at \d\d:\d\d", kwargs["text"])
    assert kwargs["parse_mode"] is None and kwargs["reply_markup"] is None, "plain text, and the buttons are gone"
    row = next(d for d in world.devices.list_devices() if d.id == polled["device_id"])
    assert world.devices.is_owner(row.id) is False, "an approval from Telegram is a scoped device like any other"


@pytest.mark.asyncio
async def test_the_audit_line_names_the_telegram_chat(tmp_path, caplog):
    world = _world(tmp_path)
    adapter = _adapter([OWNER])
    prompts = _on(world, adapter)
    created, _, _ = world.created()
    update, _ = _callback(OWNER, f"pair:a:{created['request_id']}")
    with caplog.at_level("INFO"):
        await prompts.on_callback(update, MagicMock())
    assert any(r.getMessage().startswith("pairing: approved") and f"telegram:{OWNER}" in r.getMessage()
               for r in caplog.records)


@pytest.mark.asyncio
async def test_deny_decides_and_mints_nothing(tmp_path):
    world = _world(tmp_path)
    adapter = _adapter([OWNER])
    prompts = _on(world, adapter)
    created, _, _ = world.created()
    before = len(world.devices.list_devices())
    update, query = _callback(OWNER, f"pair:d:{created['request_id']}")
    await prompts.on_callback(update, MagicMock())
    assert world.poll(created).json()["status"] == "denied"
    assert len(world.devices.list_devices()) == before
    assert "denied" in query.edit_message_text.await_args.kwargs["text"].lower()


@pytest.mark.asyncio
async def test_a_second_tap_is_told_it_was_already_decided_and_loses_its_buttons(tmp_path):
    world = _world(tmp_path)
    adapter = _adapter([OWNER, SECOND_OWNER])
    prompts = _on(world, adapter)
    created, _, _ = world.created()
    first, _ = _callback(OWNER, f"pair:a:{created['request_id']}")
    await prompts.on_callback(first, MagicMock())
    second, query = _callback(SECOND_OWNER, f"pair:d:{created['request_id']}", message_id=88)
    await prompts.on_callback(second, MagicMock())
    assert "already approved" in query.answer.await_args.args[0].lower() + str(query.answer.await_args.kwargs).lower()
    assert world.poll(created).json()["status"] == "approved", "the first decision stands"
    assert query.edit_message_text.await_args.kwargs.get("reply_markup") is None


@pytest.mark.asyncio
async def test_a_tap_after_the_ttl_says_expired(tmp_path):
    world = _world(tmp_path)
    adapter = _adapter([OWNER])
    prompts = _on(world, adapter)
    created, _, _ = world.created()
    world.clock.now += 301
    update, query = _callback(OWNER, f"pair:a:{created['request_id']}")
    await prompts.on_callback(update, MagicMock())
    assert "expired" in (str(query.answer.await_args.args) + str(query.answer.await_args.kwargs)).lower()
    assert len(world.devices.list_devices()) == 2, "only the owner and scoped devices the world started with"


@pytest.mark.asyncio
async def test_a_tap_from_a_group_decides_nothing_even_if_the_group_is_allowed(tmp_path):
    world = _world(tmp_path)
    adapter = _adapter([OWNER, GROUP])
    prompts = _on(world, adapter)
    created, _, _ = world.created()
    update, query = _callback(GROUP, f"pair:a:{created['request_id']}", chat_type="supergroup")
    await prompts.on_callback(update, MagicMock())
    assert world.poll(created).json()["status"] == "pending"
    query.answer.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_tap_from_a_chat_that_is_not_allowed_decides_nothing(tmp_path):
    """Belt and braces: the adapter's group -1 check stops this first, but the handler does not rely on it."""
    world = _world(tmp_path)
    adapter = _adapter([OWNER])
    prompts = _on(world, adapter)
    created, _, _ = world.created()
    update, _ = _callback(STRANGER, f"pair:a:{created['request_id']}")
    await prompts.on_callback(update, MagicMock())
    assert world.poll(created).json()["status"] == "pending"


@pytest.mark.asyncio
async def test_an_unknown_request_is_answered_not_raised(tmp_path):
    world = _world(tmp_path)
    adapter = _adapter([OWNER])
    prompts = _on(world, adapter)
    update, query = _callback(OWNER, f"pair:a:{'f' * 32}")
    await prompts.on_callback(update, MagicMock())
    query.answer.assert_awaited_once()


# ── the adapter's own authorisation covers a callback ────────────────────────

def _real_callback_update(chat_id: int, chat_type: str = "private") -> Update:
    chat = Chat(id=chat_id, type=chat_type)
    message = Message(message_id=1, date=dt.datetime.now(dt.UTC), chat=chat)
    query = CallbackQuery(id="1", from_user=User(id=5, first_name="x", is_bot=False), chat_instance="c",
                          data=f"pair:a:{'a' * 32}", message=message)
    return Update(update_id=1, callback_query=query)


@pytest.mark.asyncio
async def test_the_adapters_authorisation_stops_a_callback_from_an_unlisted_chat():
    adapter = _adapter([OWNER])
    update = _real_callback_update(STRANGER)
    assert update.effective_chat.id == STRANGER, "PTB derives effective_chat from the callback's message"
    with pytest.raises(ApplicationHandlerStop):
        await adapter._authorize_update(update, MagicMock())


@pytest.mark.asyncio
async def test_the_adapters_authorisation_lets_a_callback_from_an_allowed_chat_through():
    adapter = _adapter([OWNER])
    await adapter._authorize_update(_real_callback_update(OWNER), MagicMock())


# ── the other channels' decisions reach Telegram ─────────────────────────────

@pytest.mark.asyncio
async def test_a_decision_made_elsewhere_edits_every_message_and_removes_the_buttons(tmp_path):
    world = _world(tmp_path)
    adapter = _adapter([OWNER, SECOND_OWNER])
    _on(world, adapter)
    created, _, _ = world.created()
    sent = {m["chat_id"]: i + 100 for i, m in enumerate(adapter.sent)}
    world.approve(created)                                              # on Beacon / the terminal
    edits = adapter._app.bot.edit_message_text.await_args_list
    assert sorted(c.kwargs["chat_id"] for c in edits) == [OWNER, SECOND_OWNER]
    for call in edits:
        assert call.kwargs["parse_mode"] is None and call.kwargs.get("reply_markup") is None
        assert call.kwargs["text"].startswith("Jennifer's MacBook — approved on Beacon at ")
        assert call.kwargs["message_id"] in sent.values()


@pytest.mark.asyncio
async def test_each_final_state_is_worded(tmp_path):
    world = _world(tmp_path)
    adapter = _adapter([OWNER])
    _on(world, adapter)
    one, _, _ = world.created(source="192.0.2.1")
    two, _, two_client = world.created(source="192.0.2.2")
    three, _, _ = world.created(source="192.0.2.3")
    world.as_("global", "POST", f"/api/pair/requests/{one['request_id']}/deny", headers={"X-Pairing-Via": "cli"})
    two_client.delete(f"/api/pair/requests/{two['request_id']}", headers={"X-Pairing-Secret": two["poll_secret"]})
    world.clock.now += 301
    world.as_("global", "GET", "/api/pair/requests")
    texts = [c.kwargs["text"] for c in adapter._app.bot.edit_message_text.await_args_list]
    assert any("denied in the terminal" in t for t in texts)
    assert any("cancelled by the new device" in t for t in texts)
    assert any(t.startswith("Jennifer's MacBook — expired") for t in texts)
    assert three


@pytest.mark.asyncio
async def test_the_time_in_a_final_state_is_local_hh_mm(tmp_path):
    world = _world(tmp_path)
    adapter = _adapter([OWNER])
    _on(world, adapter)
    created, _, _ = world.created()
    world.approve(created)
    text = adapter._app.bot.edit_message_text.await_args.kwargs["text"]
    expected = dt.datetime.fromtimestamp(T0).strftime("%H:%M")
    assert text.endswith(f" at {expected}")


# ── boot ─────────────────────────────────────────────────────────────────────

def test_turning_it_on_with_no_private_chat_warns_at_boot_instead_of_doing_nothing_quietly(tmp_path, caplog):
    world = _world(tmp_path)
    world.runtime.settings = PairingSettings(telegram_prompts=True)
    with caplog.at_level("WARNING"):
        attached = _prompts(world, _adapter([GROUP])).attach()
    assert attached is False
    assert any("telegram_prompts" in r.getMessage() and "private" in r.getMessage() for r in caplog.records)
