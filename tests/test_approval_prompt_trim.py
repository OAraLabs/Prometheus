"""The approval prompt is short, and the extents stay whole behind /remember.

THE COST BEING REMOVED
----------------------
The prompt carried all four verb+extent lines on EVERY request — eleven
non-blank lines, four of them repeating the same two paths twice each. Grants
are rare and approve-once is common, so the common case paid the rare case's
cost every single time.

THE PROPERTY THAT MUST SURVIVE THE TRIM
---------------------------------------
**A verb and its extent are read together or not at all.** SPRINT-CONSENT
exists because consent was obtained under a false description: the prompt
stated duration and never extent, while ``derive_grant`` widened one file to
its whole parent directory. A menu that named the four verbs and deferred
their extents to a further round trip would rebuild that defect in a new
shape — the operator would pick a verb before seeing its reach.

So the trim moves WHERE the lines are read, never whether they are read
whole. ``test_remember_keeps_verb_and_extent_on_one_line`` is the assertion
that would fail if someone later "tidied" /remember into a bare verb list.

``prospective_extents`` is untouched by this change — both surfaces still
render from it, and nothing is re-derived.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from prometheus.gateway import commands as cmds
from prometheus.permissions.approval_queue import (
    ApprovalQueue, PendingAction, prospective_extents)

PATH = "/home/will/projects/notes/todo.md"


class _FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, chat_id, text, parse_mode=None) -> None:
        self.sent.append(text)


async def _prompt(queue: ApprovalQueue, **kw) -> str:
    task = asyncio.create_task(
        queue.request_approval("write_file", f"write {PATH}",
                               grant_file_path=PATH, **kw)
    )
    for _ in range(400):
        await asyncio.sleep(0.005)
        if queue._telegram.sent:
            break
    task.cancel()
    with contextlib.suppress(BaseException):
        await task
    assert queue._telegram.sent, "no prompt was sent"
    return queue._telegram.sent[0]


def _queue(extra_pending: int = 0) -> ApprovalQueue:
    q = ApprovalQueue(telegram_adapter=_FakeTelegram(), timeout_seconds=1800)
    q._default_chat_id = 1
    for i in range(extra_pending):
        q.pending[f"filler{i}"] = PendingAction(
            request_id=f"filler{i}", tool_name="bash",
            description=f"filler {i}", grant_file_path=None,
            grant_command="ls",
        )
    return q


@pytest.mark.asyncio
async def test_prompt_no_longer_carries_the_four_extent_lines():
    q = _queue()
    msg = await _prompt(q)
    body = [ln for ln in msg.splitlines() if ln.strip()]

    assert len(body) <= 7, f"prompt grew back to {len(body)} lines:\n{msg}"
    assert "/approve — approve this ONCE (or /deny)" in msg
    assert "/remember — options that create a lasting grant" in msg
    # The extents themselves are gone from the prompt.
    assert "grants write_file" not in msg
    # And the path is named ONCE, not five times.
    assert msg.count(PATH) == 1, f"path repeated {msg.count(PATH)}x"


@pytest.mark.asyncio
async def test_remember_reproduces_every_extent_the_prompt_used_to_carry():
    """Nothing is lost by the move — only relocated."""
    q = _queue()
    await _prompt(q)
    rid = next(a.request_id for a in q.pending.values()
               if a.tool_name == "write_file")
    action = q.pending[rid]
    out = await cmds.cmd_remember(q, rid)

    extents = prospective_extents(action)
    assert extents, "fixture should have derivable extents"
    for verb, what in extents.items():
        assert f"/approve {verb} — grants {what}" in out, (
            f"/remember dropped the {verb!r} option that the prompt used to "
            f"show. The trim may relocate the options, never lose them."
        )


@pytest.mark.asyncio
async def test_remember_keeps_verb_and_extent_on_one_line():
    """THE GOVERNING PROPERTY — the one this sprint is about.

    A verb may never appear without the extent it would grant on the SAME
    line. Splitting them across a round trip is consent under a false
    description wearing a new shape.
    """
    q = _queue()
    await _prompt(q)
    rid = next(a.request_id for a in q.pending.values()
               if a.tool_name == "write_file")
    out = await cmds.cmd_remember(q, rid)

    for line in out.splitlines():
        if "/approve " in line and "—" in line:
            assert "grants " in line, (
                f"a verb is offered with no extent on its line: {line!r}"
            )


@pytest.mark.asyncio
async def test_approve_all_is_absent_when_it_would_mean_approve():
    """One pending request: /approve all IS /approve, so it is not offered.

    Showing a multi-request verb for a single request is noise carrying
    breadth, which is the wrong direction for a trim whose point is to stop
    describing reach the operator does not need yet.
    """
    q = _queue()
    msg = await _prompt(q)
    assert "/approve all" not in msg


@pytest.mark.asyncio
async def test_approve_all_states_its_extent_when_it_is_offered():
    """It was the ONLY verb whose extent went undescribed.

    Trimming the four described options while leaving the one undescribed
    option in place would have inverted the sprint. Its breadth spans
    REQUESTS, so its extent is the count — and it says it creates no grant,
    which is also why it is not filed under /remember.
    """
    q = _queue(extra_pending=2)
    msg = await _prompt(q)
    line = next(ln for ln in msg.splitlines() if ln.startswith("/approve all"))
    assert "3 pending" in line, f"no count in: {line!r}"
    assert "creates no lasting grant" in line, f"no grant-extent in: {line!r}"


@pytest.mark.asyncio
async def test_no_derivable_grant_does_not_advertise_remember():
    """An option that cannot be honoured is not offered — but is explained."""
    q = ApprovalQueue(telegram_adapter=_FakeTelegram(), timeout_seconds=1800)
    q._default_chat_id = 1
    task = asyncio.create_task(
        q.request_approval("some_tool", "a thing with no structured target")
    )
    for _ in range(400):
        await asyncio.sleep(0.005)
        if q._telegram.sent:
            break
    msg = q._telegram.sent[0]
    rid = next(iter(q.pending))
    out = await cmds.cmd_remember(q, rid)
    task.cancel()
    with contextlib.suppress(BaseException):
        await task

    assert "/remember" not in msg, "offered a grant menu that cannot be filled"
    assert "no specific target" in out, (
        "an operator who types /remember anyway is owed the reason, not an "
        "empty menu"
    )


@pytest.mark.asyncio
async def test_remember_rejects_a_mistyped_argument_immediately():
    q = _queue()
    await _prompt(q)
    assert (await cmds.cmd_remember(q, "awlways")).startswith("Usage:")
