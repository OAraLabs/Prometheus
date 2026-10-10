"""Telegram prompts for pairing requests: opt-in, plain text, private chats, the same decision as everywhere.

Contract: docs/PAIRING-APPROVAL-API.md, 5.2. When a new device asks to join and the owner turned
``pairing.telegram_prompts`` on, each of their private chats gets

    Jennifer's MacBook wants to connect.
    Code 4821

    macOS · 192.168.1.42 · expires in 5 min
    The name is typed on the new device and is not checked. Approve only if the code matches the one on its screen.

    [ Approve ]  [ Deny ]

and a tap decides it. What keeps that safe:

* **Off by default, and then nothing exists**: no handler is registered, no message is sent. Telegram is a
  third-party relay for a message that carries a source address and a match code.
* **Plain text only.** The device name is typed on the new device, so anyone on the network can make it
  ``*bold*`` or a link. The adapter's ``send_prompt`` / ``edit_prompt`` pass ``parse_mode=None`` to the bot API.
* **Private chats only**: positive chat ids in ``allowed_chat_ids``. A group in that list gets nothing, and a
  tap from one decides nothing, because group membership is not an identity and any member could approve.
* **The same decision.** A tap calls ``PairingRuntime.approve`` / ``deny``, the code REST and the terminal
  run, as ``telegram:<chat id>``. It is already behind the adapter's group -1 ``_authorize_update`` (PTB
  derives ``effective_chat`` for a callback query from its message; ``tests/test_pairing_telegram.py`` pins it),
  and this handler checks the chat again rather than rely on that.
* **Every message ends in its final state with no buttons**, whoever decided and wherever: edited by the
  ``resolved`` event, and by the tap itself, so a restart (which loses the message ids held in memory) still
  leaves the tapped message right. A tap on an old message answers "already approved" or "expired".

``callback_data`` is ``pair:a:<request id>`` or ``pair:d:<request id>`` (39 bytes, under Telegram's 64).

Source: novel code for Prometheus, 2026-10-08.
"""

from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import CallbackQueryHandler, ContextTypes

from prometheus.config.pair_requests import (
    CodeMismatch,
    NotPending,
    RequestExpired,
    UnknownRequest,
)

logger = logging.getLogger("prometheus.pairing")

#: What this module accepts as callback data, and nothing else. ``\Z`` rather than ``$``: no trailing newline.
CALLBACK_PATTERN = re.compile(r"^pair:[ad]:[0-9a-f]{32}\Z")

_PLATFORM_NAMES = {"macos": "macOS", "ios": "iOS", "windows": "Windows", "linux": "Linux", "android": "Android"}
_WHERE = {"telegram": "from Telegram", "beacon": "on Beacon", "cli": "in the terminal"}
_MAX_REMEMBERED = 64


def _hhmm(when: float | None = None) -> str:
    return time.strftime("%H:%M", time.localtime(when if when is not None else time.time()))


def prompt_text(payload: dict[str, Any]) -> str:
    platform = _PLATFORM_NAMES.get(payload["platform"], "unknown platform")
    ttl = int(payload["ttl_seconds"])
    window = f"{ttl // 60} min" if ttl % 60 == 0 else f"{ttl} s"
    return (
        f"{payload['device_name']} wants to connect.\n"
        f"Code {payload['match_code']}\n\n"
        f"{platform} · {payload['source_ip']} · expires in {window}\n"
        "The name is typed on the new device and is not checked. "
        "Approve only if the code matches the one on its screen."
    )


def final_text(name: str, resolution: str, by: str, when: float | None = None) -> str:
    """The message once it is decided. The name is plain text, like everything here."""
    if resolution in ("approved", "denied"):
        return f"{name} — {resolution} {_WHERE.get(by, 'elsewhere')} at {_hhmm(when)}"
    if resolution == "canceled":
        return f"{name} — cancelled by the new device"
    return f"{name} — expired"


def _keyboard(request_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("Approve", callback_data=f"pair:a:{request_id}"),
        InlineKeyboardButton("Deny", callback_data=f"pair:d:{request_id}"),
    ]])


class TelegramPairingPrompts:
    """Sends the prompts, handles the taps, and edits the messages when the request is decided."""

    def __init__(self, adapter: Any, runtime: Any) -> None:
        self.adapter = adapter
        self.runtime = runtime
        # request id -> (device name, [(chat id, message id)]); in memory, bounded, lost on restart.
        self._messages: OrderedDict[str, tuple[str, list[tuple[int, int]]]] = OrderedDict()

    def private_chat_ids(self) -> list[int]:
        """The allowed chats that are PRIVATE (a positive id). Groups, supergroups and channels are negative."""
        return [chat for chat in self.adapter.config.allowed_chat_ids if chat > 0]

    def attach(self) -> bool:
        """Register the tap handler and subscribe to pairing events, if the owner opted in and it can work.

        Returns False, with a WARNING when the owner asked for it, in every case where it cannot.
        """
        if not self.runtime.settings.telegram_prompts:
            return False
        if not self.private_chat_ids():
            logger.warning(
                "pairing.telegram_prompts is on but gateway.allowed_chat_ids has no private chat (a positive id), "
                "so no pairing prompt will be sent to Telegram; groups are never used for this")
            return False
        if not self.adapter.add_handler(CallbackQueryHandler(self.on_callback, pattern=CALLBACK_PATTERN)):
            logger.warning("pairing.telegram_prompts is on but the Telegram bot is not running, "
                           "so no pairing prompt will be sent to Telegram")
            return False
        self.runtime.notifier.subscribe(self._on_event)
        logger.info("pairing: Telegram prompts are on for %d private chat(s)", len(self.private_chat_ids()))
        return True

    # -- events from the pairing runtime --------------------------------------

    async def _on_event(self, kind: str, payload: dict[str, Any]) -> bool:
        if kind == "pending":
            return await self._announce(payload)
        if kind == "resolved":
            await self._finish(payload)
        return False

    async def _announce(self, payload: dict[str, Any]) -> bool:
        text, keyboard = prompt_text(payload), _keyboard(payload["request_id"])
        sent: list[tuple[int, int]] = []
        for chat_id in self.private_chat_ids():
            message_id = await self.adapter.send_prompt(chat_id, text, keyboard)
            if message_id is not None:
                sent.append((chat_id, message_id))
        if sent:
            self._messages[payload["request_id"]] = (payload["device_name"], sent)
            while len(self._messages) > _MAX_REMEMBERED:
                self._messages.popitem(last=False)
        return bool(sent)

    async def _finish(self, payload: dict[str, Any]) -> None:
        remembered = self._messages.pop(payload["request_id"], None)
        if remembered is None:
            return
        name, messages = remembered
        text = final_text(name, payload["resolution"], payload["by"], payload.get("resolved_at"))
        for chat_id, message_id in messages:
            await self.adapter.edit_prompt(chat_id, message_id, text)

    # -- a tap ------------------------------------------------------------------

    async def on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        data = (query.data or "") if query is not None else ""
        if query is None or CALLBACK_PATTERN.fullmatch(data) is None:
            if query is not None:
                await query.answer()
            return
        chat = update.effective_chat
        if chat is None or getattr(chat, "type", None) != "private" or not self.adapter.config.chat_allowed(chat.id):
            await query.answer("Not allowed.")
            return
        action, request_id = data[5], data[7:]
        decided_by = f"telegram:{chat.id}"
        name = self._name_of(request_id)
        try:
            if action == "a":
                await self.runtime.approve(request_id, via="telegram", decided_by=decided_by)
                verdict, resolution = "Approved.", "approved"
            else:
                await self.runtime.deny(request_id, via="telegram", decided_by=decided_by)
                verdict, resolution = "Denied.", "denied"
        except UnknownRequest:
            await query.answer("That request is gone.")
            return
        except RequestExpired:
            await query.answer("Expired: too late to decide.")
            await self._edit_tapped(query, final_text(name, "expired", "system"))
            return
        except NotPending as exc:
            await query.answer(f"Already {exc.status}.")
            await self._edit_tapped(query, f"{name} — already {exc.status}")
            return
        except CodeMismatch:                                   # unreachable: a tap never retypes a code
            await query.answer("The code does not match.")
            return
        await query.answer(verdict)
        await self._edit_tapped(query, final_text(name, resolution, "telegram"))

    def _name_of(self, request_id: str) -> str:
        remembered = self._messages.get(request_id)
        if remembered is not None:
            return remembered[0]
        req = self.runtime.store().get(request_id)
        return req.device_name if req is not None else "This request"

    @staticmethod
    async def _edit_tapped(query: Any, text: str) -> None:
        try:
            await query.edit_message_text(text=text, parse_mode=None, reply_markup=None)
        except Exception as exc:                               # "message is not modified" when the event got there first
            logger.info("pairing: could not edit the tapped Telegram message: %s", exc)
