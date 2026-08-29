"""Control-plane separation — /approve must not queue behind a turn.

THE DEFECT (2026-08-17, handoff item 2): PTB's default is
max_concurrent_updates=1 — the update fetcher awaits each update to
completion — and the agent turn ran inline in the text handler. A
66-minute turn therefore blocked every subsequent update bot-wide: a
correctly-typed /approve answered 36 minutes after its request had
expired, contradicting the (correct) expiry notice. Sharpest form: an
approval raised BY the current turn waits on an event only a /approve
update can set, and that update cannot be dequeued until the turn ends —
timeout was the only possible outcome, by construction.

THE FIX: block=False on the DATA-plane handlers (text + media). The
counterintuitive half is pinned here: marking the control commands
non-blocking would do nothing — the handler holding the fetcher is the
one that must let go.

THE COVERAGE LESSON: the old "answers even while a turn is in flight"
claim shipped false because every test called the command layer directly,
bypassing PTB dispatch. The dispatch-level tests below drive a real
telegram.ext.Application.process_update — a hanging data-plane handler
must not stop a control command from completing.
"""

from __future__ import annotations

import asyncio
import datetime
import inspect

import pytest
from telegram import Chat, Message, MessageEntity, Update, User
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from prometheus.gateway.telegram import TelegramAdapter


class TestRegistrationProperties:
    """Assert the wiring, not just the function (Standing-Principles §2e),
    in the same style as test_telegram_command_auth's group -1 guard."""

    def test_data_plane_handlers_are_non_blocking(self) -> None:
        src = inspect.getsource(TelegramAdapter)
        assert "self._handle_text, block=False" in src, (
            "the text handler must register block=False — it runs the agent "
            "turn inline, and with PTB's single-slot fetcher a blocking "
            "registration holds every /approve for the turn's duration"
        )
        for media in ("_handle_photo", "_handle_voice",
                      "_handle_document", "_handle_sticker"):
            assert f"self.{media}, block=False" in src, (
                f"{media} dispatches agent work and must not hold the fetcher"
            )

    def test_control_commands_stay_blocking(self) -> None:
        # Millisecond state operations keep default block=True: their
        # mutual ordering stays strict, and none of them needs detaching —
        # the fetcher is only ever held by the data plane now.
        src = inspect.getsource(TelegramAdapter)
        assert 'CommandHandler("approve", self._cmd_approve, block=False)' not in src
        assert 'CommandHandler("deny", self._cmd_deny, block=False)' not in src


def _app_with(handlers) -> Application:
    app = Application.builder().token("123:test-token").build()
    for h in handlers:
        app.add_handler(h)
    # process_update refuses an uninitialized app, and CommandHandler reads
    # bot.username to strip @mentions — but full initialize() would hit the
    # network. These tests exercise DISPATCH semantics, not startup, so the
    # two initialization side effects they need are supplied directly.
    app._initialized = True
    app.bot._bot_user = User(
        id=12345, first_name="testbot", is_bot=True, username="testbot"
    )
    return app


def _message(app: Application, text: str, *, command: bool) -> Update:
    entities = (
        [MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0,
                       length=len(text.split()[0]))]
        if command else None
    )
    msg = Message(
        message_id=1,
        date=datetime.datetime.now(datetime.timezone.utc),
        chat=Chat(id=1000, type=Chat.PRIVATE),
        from_user=User(id=42, first_name="op", is_bot=False),
        text=text,
        entities=entities,
    )
    msg.set_bot(app.bot)
    upd = Update(update_id=next(_ids), message=msg)
    upd.set_bot(app.bot)
    return upd


def _id_gen():
    i = 0
    while True:
        i += 1
        yield i


_ids = _id_gen()


class TestDispatchLevel:
    @pytest.mark.asyncio
    async def test_control_command_answers_while_data_plane_hangs(self) -> None:
        """The killer, at the transport: a text handler that never returns
        (a stand-in for the 66-minute turn) must not stop /approve from
        completing."""
        turn_running = asyncio.Event()
        release = asyncio.Event()
        approvals: list[str] = []

        async def slow_turn(update, context):  # noqa: ANN001
            turn_running.set()
            await release.wait()

        async def approve(update, context):  # noqa: ANN001
            approvals.append(update.message.text)

        app = _app_with([
            MessageHandler(filters.TEXT & ~filters.COMMAND, slow_turn,
                           block=False),
            CommandHandler("approve", approve),
        ])
        try:
            # The data-plane update must RETURN from process_update while
            # its handler still runs — that is what block=False buys.
            await asyncio.wait_for(
                app.process_update(_message(app, "long task please",
                                            command=False)),
                timeout=2,
            )
            await asyncio.wait_for(turn_running.wait(), timeout=2)

            # And the control-plane update completes while the turn hangs.
            await asyncio.wait_for(
                app.process_update(_message(app, "/approve 9321b362",
                                            command=True)),
                timeout=2,
            )
            assert approvals == ["/approve 9321b362"]
        finally:
            release.set()
            await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_blocking_registration_reproduces_the_defect(self) -> None:
        """The other direction: with default block=True, process_update on
        the text update does NOT return while the handler hangs — the exact
        head-of-line block the finding documented. If this test ever fails,
        PTB's dispatch semantics changed and the block=False fix needs
        re-verifying."""
        release = asyncio.Event()

        async def slow_turn(update, context):  # noqa: ANN001
            await release.wait()

        app = _app_with([
            MessageHandler(filters.TEXT & ~filters.COMMAND, slow_turn),
        ])
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    app.process_update(_message(app, "long task please",
                                                command=False)),
                    timeout=0.3,
                )
        finally:
            release.set()
            await asyncio.sleep(0)
