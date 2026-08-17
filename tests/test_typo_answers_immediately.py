"""A mistyped command answers at once, and names the near miss.

THE NIGHT THIS COMES FROM
-------------------------
2026-08-17, two typos in a row, both of which did nothing:

* ``/approve awlways`` — the mistyped-scope guard required ``len(tokens) > 1``,
  so it only fired when an id FOLLOWED the typo. The bare form is the common
  one. ``awlways`` fell through and was looked up as a request id, and the
  operator got ``No pending request: awlways`` — 66 minutes later, because the
  reply queued behind an in-flight turn.
* ``/appove always`` — matched no ``CommandHandler``, and the text handler is
  ``TEXT & ~COMMAND``, which excludes anything starting with "/". **No handler
  ran at all.** That one would never have answered, at any latency.

Two different mechanisms, one operator experience: silence. A typo that does
nothing is indistinguishable from a daemon ignoring you.

WHAT IS AND IS NOT FIXED HERE
-----------------------------
These are the PARSE halves. The 66-minute delay is a separate defect —
gateway commands queueing behind an in-flight turn — and is not addressed
here. The unknown-command path does no queueing and no agent work, so it
answers while a turn is in flight; the ``/approve`` path still rides the
gateway's normal dispatch.
"""

from __future__ import annotations

import pytest

from prometheus.gateway import commands as cmds


class _Queue:
    """Real enough: an empty pending dict, which is the live shape."""

    pending: dict = {}

    async def approve(self, request_id, scope="once", grant=None) -> bool:
        # Nothing pending, so a well-formed id legitimately finds no request.
        return False


# ── half 1: the mistyped scope ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bare_mistyped_scope_is_caught_not_looked_up_as_an_id():
    """THE ONE FROM THE FIELD. ``/approve awlways``, no id after it."""
    out = await cmds.cmd_approve(_Queue(), "awlways")
    assert "No pending request" not in out, (
        "a mistyped scope was looked up as a request id — the exact reply "
        "Will got at 01:14 for a command sent at 00:08"
    )
    assert "awlways" in out and "always" in out, (
        f"the answer should name the typo and the near miss: {out!r}"
    )


@pytest.mark.asyncio
async def test_the_guard_still_fires_with_an_id_after_the_typo():
    """The case the old guard DID cover must keep working."""
    out = await cmds.cmd_approve(_Queue(), "forever a1b2c3d4")
    assert "No pending request" not in out


@pytest.mark.asyncio
@pytest.mark.parametrize("typo,expected", [
    ("awlways", "always"),
    ("alwyas", "always"),
    ("untl-restart", "until-restart"),
])
async def test_near_misses_are_suggested(typo, expected):
    out = await cmds.cmd_approve(_Queue(), typo)
    assert expected in out, f"{typo!r} did not suggest {expected!r}: {out!r}"


@pytest.mark.asyncio
async def test_a_real_request_id_is_still_treated_as_an_id():
    """8-hex ids must not be mistaken for typo'd scopes."""
    out = await cmds.cmd_approve(_Queue(), "a1b2c3d4")
    assert "Did you mean" not in out


@pytest.mark.asyncio
async def test_gibberish_falls_back_to_usage_rather_than_guessing():
    out = await cmds.cmd_approve(_Queue(), "zzzqqqxxx")
    assert "Usage" in out or "usage" in out


# ── half 2: the unknown command ───────────────────────────────────────────

def test_near_miss_names_the_command_the_operator_meant():
    """``/appove always`` — the one that matched no handler at all."""
    hint = cmds._near_miss("appove", ["approve", "deny", "pending"], "/", "command")
    assert hint is not None, "no suggestion for a one-letter command typo"
    assert "/approve" in hint, hint


def test_near_miss_declines_to_guess_when_nothing_is_close():
    assert cmds._near_miss("qqqqzzzz", ["approve", "deny"], "/", "command") is None


def test_unknown_command_handler_is_registered_after_every_command_handler():
    """Order is load-bearing: PTB runs the FIRST match per group.

    Registered before the CommandHandlers, the catch-all would swallow every
    command in the bot.
    """
    import inspect

    from prometheus.gateway import telegram as tg

    src = inspect.getsource(tg.TelegramAdapter)
    catch_all = src.index("MessageHandler(filters.COMMAND, self._cmd_unknown)")
    last_cmd = src.rindex('CommandHandler("', 0, catch_all)
    assert last_cmd < catch_all, (
        "the unknown-command catch-all is registered before a CommandHandler; "
        "PTB runs the first match per group, so it would swallow real commands"
    )
    assert 'CommandHandler("' not in src[catch_all:], (
        "a CommandHandler is registered AFTER the catch-all and is therefore "
        "unreachable"
    )
