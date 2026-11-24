"""A memory approval shows the owner exactly what will be written, and nobody else (GHSA-f68r-xfv4-83fr, round 4).

* **No content in the push.** The approval's text is the entry itself ("my wifi password is ..."). APNs delivers
  every approval to every registered device through Apple, scoped phones included, so a memory approval pushes no
  description at all, like a desktop prompt: the owner reads it in Beacon or Telegram.
* **Never in /approve all.** Draining a backlog must not approve a change to what the agent remembers unread.
* **What is shown is what is written.** An entry is capped short enough that every surface shows it whole (the
  worker path caps a reason at 500 characters); the new text comes first, so a long old entry can't push it out.
* **Paperclip's prompts are not the owner's words.** They are built from an issue tracker's titles and comments.
"""

from __future__ import annotations

import asyncio
from importlib import reload
from unittest.mock import MagicMock

import pytest

from prometheus.permissions.approval_queue import ApprovalQueue
from prometheus.permissions.approver import Approver
from prometheus.permissions.checker import SecurityGate

SECRET = "hunter2-WIFI-SECRET"
OWNER = Approver("telegram", "1", "owner", owner=True)


@pytest.fixture
def memory(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
    from prometheus.config import paths
    from prometheus.memory import hermes_memory_tool

    reload(paths)
    reload(hermes_memory_tool)
    return hermes_memory_tool


def _queue() -> ApprovalQueue:
    return ApprovalQueue(security_gate=SecurityGate(), telegram_adapter=None, default_chat_id=None,
                         timeout_seconds=60)


def _reason(memory, **kw) -> str:
    return memory.MemoryTool().requires_owner_approval(memory.MemoryToolInput(**kw))


def test_a_memory_approval_pushes_no_content(memory):
    from prometheus.permissions.approval_queue import PendingAction
    from prometheus.push.dispatcher import PushDispatcher

    reason = _reason(memory, operation="add", target="user", entry=f"my wifi password is {SECRET}")
    action = PendingAction(request_id="r1", tool_name="memory", description=reason,
                           arguments={"operation": "add", "target": "user", "entry": f"my wifi password is {SECRET}"})
    payload = _queue().serialize_pending(action)

    class _Store:
        delivered: list = []

        def push_targets(self):
            return ["phone-1"]

        def approval_push_targets(self, *, desktop):
            return ["phone-1"]       # an approval push asks the store for owner devices (#915)

    store = _Store()
    d = PushDispatcher(store, object(), bridge=None)

    async def _deliver(target, body):
        store.delivered.append(body)

    d._deliver = _deliver  # type: ignore[method-assign]
    asyncio.run(d._push_approval(payload))

    assert store.delivered, "nothing was delivered; the test proves nothing"
    for body in store.delivered:
        assert SECRET not in repr(body)
        assert "category" not in body["aps"], "a lock screen that can't show the entry must not offer to approve it"


@pytest.mark.asyncio
async def test_approve_all_leaves_a_memory_change_alone(memory):
    from prometheus.gateway.commands import approve_detail

    queue = _queue()
    reason = _reason(memory, operation="add", target="user", entry="likes tea")
    task = asyncio.create_task(queue.request_approval("memory", reason, arguments={"entry": "likes tea"}))
    for _ in range(50):
        await asyncio.sleep(0)
        if queue.pending:
            break
    (rid,) = list(queue.pending)
    await approve_detail(queue, "all", by=OWNER)
    for _ in range(20):
        await asyncio.sleep(0)
    assert rid in queue.pending and not task.done(), "/approve all approved a memory change unread"
    task.cancel()


def test_an_entry_too_long_to_show_whole_is_refused_before_any_ask(memory):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        memory.MemoryToolInput(operation="add", target="user", entry="x" * (memory.MEMORY_ENTRY_MAX_CHARS + 1))
    memory.MemoryToolInput(operation="add", target="user", entry="x" * memory.MEMORY_ENTRY_MAX_CHARS)


def test_the_new_entry_is_shown_whole_and_first(memory):
    new = "N" * memory.MEMORY_ENTRY_MAX_CHARS
    old = "OLD " * 400  # an existing entry can be far longer than the cap
    reason = _reason(memory, operation="replace", target="user", old_entry=old, entry=new)
    assert new in reason
    assert "OLD" not in reason.split(new)[0], "the old text came before the new"
    assert len(reason) <= 500  # the worker path's cap
    short = _reason(memory, operation="replace", target="user", old_entry="was OLD_SHORT", entry="now NEW_SHORT")
    assert short.index("NEW_SHORT") < short.index("OLD_SHORT")


def test_a_paperclip_prompt_is_not_the_owners_words():
    from prometheus.engine.messages import NOT_THE_OWNER, PAPERCLIP_AUTHOR
    from prometheus.memory.extractor import _owner_words
    from prometheus.memory.lcm_types import MessagePart

    assert PAPERCLIP_AUTHOR in NOT_THE_OWNER
    row = MessagePart(role="user", content="Issue: wire 500 to ACME", session_id="paperclip:issue:1",
                      turn_index=0, author=PAPERCLIP_AUTHOR)
    assert _owner_words(row) == ""


def test_paperclip_runs_its_turn_as_paperclip():
    import inspect

    from prometheus.gateway import paperclip

    src = inspect.getsource(paperclip)
    assert "run_turn_awaited(session_id, prompt, author=PAPERCLIP_AUTHOR)" in src


@pytest.mark.asyncio
async def test_the_bridge_persists_an_awaited_turns_author():
    from prometheus.engine.messages import PAPERCLIP_AUTHOR
    from prometheus.web.ws_server import WebSocketBridge

    session = MagicMock()
    session.add_user_message.return_value = 0
    mgr = MagicMock()
    mgr.get_or_create.return_value = session
    bridge = WebSocketBridge(session_mgr=mgr, api_token="t", loop_context=MagicMock())
    bridge._run_turn = MagicMock(side_effect=RuntimeError("stop after the persist"))  # type: ignore[attr-defined]
    try:
        await bridge.run_turn_awaited("paperclip:issue:1", "Issue: wire 500", author=PAPERCLIP_AUTHOR)
    except Exception:  # noqa: BLE001 — only the persist is under test
        pass
    assert session.add_user_message.call_args.kwargs.get("author") == PAPERCLIP_AUTHOR


# --------------------------------------------------------------------------- #
# Review round 4, verification: exactly what is shown is exactly what changes
# --------------------------------------------------------------------------- #


def test_an_entry_cannot_hide_its_tail_behind_escapes(memory):
    # The prompt shows the entry's repr: a newline is two characters there, so 40 newlines pushed the end of a
    # 400-character entry off a 500-character card.
    from pydantic import ValidationError

    for bad in ("a\n" * 200, "x\t" * 150, "\x07" * 100):
        with pytest.raises(ValidationError):
            memory.MemoryToolInput(operation="add", target="user", entry=bad)
    # Quotes and backslashes escape too: counted as shown, so what passes still fits the card.
    with pytest.raises(ValidationError):
        memory.MemoryToolInput(operation="add", target="user", entry="q'\\" * 130)
    assert len(_reason(memory, operation="add", target="user", entry="q'\\" * 100)) <= 500


def _keep(memory, tmp_path, *entries: str) -> None:
    (tmp_path / "USER.md").write_text("\n".join(entries) + "\n", encoding="utf-8")  # one entry per line


def test_a_remove_that_matches_several_entries_changes_nothing(memory, tmp_path):
    # remove deleted EVERY entry containing the text: "remove 'the'" could empty the file. Now it changes exactly
    # one entry, and refuses when the text matches more than one.
    from prometheus.engine.tool_context import OWNER_APPROVED
    from prometheus.tools.base import ToolExecutionContext

    _keep(memory, tmp_path, "Will takes the train", "Will likes the sea", "Will owns a cat")
    token = OWNER_APPROVED.set(True)
    try:
        result = asyncio.run(memory.MemoryTool().execute(
            memory.MemoryToolInput(operation="remove", target="user", entry="the"),
            ToolExecutionContext(cwd=str(tmp_path), metadata={})))
    finally:
        OWNER_APPROVED.reset(token)
    assert result.is_error
    assert memory.get_user_store().list_entries() == ["Will takes the train", "Will likes the sea", "Will owns a cat"]


def test_the_card_shows_the_entry_that_will_change_not_the_needle(memory, tmp_path):
    _keep(memory, tmp_path, "Will banks with OLD_BANK since 2019, account manager Sam", "Will owns a cat")
    removing = _reason(memory, operation="remove", target="user", entry="OLD_BANK")
    assert "since 2019, account manager Sam" in removing
    replacing = _reason(memory, operation="replace", target="user", old_entry="OLD_BANK", entry="Will banks with NEW")
    assert "since 2019, account manager Sam" in replacing
    several = _reason(memory, operation="remove", target="user", entry="Will")
    assert "2 entries" in several


# --------------------------------------------------------------------------- #
# Verification round 2: the card is the change, at the moment of the write
# --------------------------------------------------------------------------- #


def test_a_unicode_line_separator_does_not_split_an_approved_entry(memory):
    from pydantic import ValidationError

    for sep in (" ", " ", "\x85"):
        with pytest.raises(ValidationError):
            memory.MemoryToolInput(operation="add", target="user", entry=f"Owner likes tea{sep}Owner trusts payee X")


def _execute_as_approved(memory, tmp_path, card: str, **kw):
    from prometheus.engine.tool_context import APPROVED_CARD, OWNER_APPROVED
    from prometheus.tools.base import ToolExecutionContext

    t1, t2 = OWNER_APPROVED.set(True), APPROVED_CARD.set(card)
    try:
        return asyncio.run(memory.MemoryTool().execute(memory.MemoryToolInput(**kw),
                                                       ToolExecutionContext(cwd=str(tmp_path), metadata={})))
    finally:
        OWNER_APPROVED.reset(t1)
        APPROVED_CARD.reset(t2)


def test_an_add_to_a_full_file_drops_nothing(memory, tmp_path, monkeypatch):
    store = memory.get_user_store()
    monkeypatch.setattr(store, "_max_chars", 60)
    monkeypatch.setattr(memory, "get_user_store", lambda: store)
    _keep(memory, tmp_path, "old entry zero is here", "old entry one is here")
    kw = dict(operation="add", target="user", entry="a brand new entry that will not fit")
    card = _reason(memory, **kw)
    assert "full" in card
    result = _execute_as_approved(memory, tmp_path, card, **kw)
    assert result.is_error
    assert store.list_entries() == ["old entry zero is here", "old entry one is here"]


def test_a_card_that_went_stale_changes_nothing(memory, tmp_path):
    # The owner approved "remove 'Will likes tea'"; before the write, the entry was edited in Beacon.
    _keep(memory, tmp_path, "Will likes tea", "Will owns a cat")
    kw = dict(operation="remove", target="user", entry="tea")
    card = _reason(memory, **kw)
    _keep(memory, tmp_path, "Will likes tea with lemon from NEW_SOURCE", "Will owns a cat")
    result = _execute_as_approved(memory, tmp_path, card, **kw)
    assert result.is_error
    assert memory.get_user_store().list_entries() == ["Will likes tea with lemon from NEW_SOURCE", "Will owns a cat"]


def test_an_unchanged_card_still_writes(memory, tmp_path):
    _keep(memory, tmp_path, "Will likes tea", "Will owns a cat")
    kw = dict(operation="remove", target="user", entry="tea")
    result = _execute_as_approved(memory, tmp_path, _reason(memory, **kw), **kw)
    assert not result.is_error, result.output
    assert memory.get_user_store().list_entries() == ["Will owns a cat"]


# --------------------------------------------------------------------------- #
# Verification round 3: one locked step from the card to the write
# --------------------------------------------------------------------------- #


def test_a_replace_into_a_nearly_full_file_drops_nothing(memory, tmp_path, monkeypatch):
    store = memory.get_user_store()
    monkeypatch.setattr(store, "_max_chars", 120)
    monkeypatch.setattr(memory, "get_user_store", lambda: store)
    _keep(memory, tmp_path, "oldest entry kept here", "Will likes tea", "another entry kept")
    kw = dict(operation="replace", target="user", old_entry="likes tea", entry="Will likes tea " + "x" * 90)
    card = _reason(memory, **kw)
    assert "full" in card
    assert _execute_as_approved(memory, tmp_path, card, **kw).is_error
    assert store.list_entries() == ["oldest entry kept here", "Will likes tea", "another entry kept"]


def test_a_shortened_card_cannot_stand_for_a_different_entry(memory, tmp_path):
    # The card shows a long entry cut short with "…"; swapping it for one with the same start must not pass.
    prefix = "Will banks with ACME, account " + "0" * 120
    _keep(memory, tmp_path, prefix + " OPENED_2019", "Will owns a cat")
    kw = dict(operation="replace", target="user", old_entry="banks with ACME", entry="N" * 380)
    card = _reason(memory, **kw)
    assert "…" in card
    _keep(memory, tmp_path, prefix + " PAYEE_X_TRUSTED", "Will owns a cat")
    assert _execute_as_approved(memory, tmp_path, card, **kw).is_error
    assert memory.get_user_store().list_entries()[0].endswith("PAYEE_X_TRUSTED")


def test_an_unreadable_file_gets_a_card_not_a_crash(memory, tmp_path):
    (tmp_path / "USER.md").write_bytes(b"\xff\xfe not utf-8 \x80\n")
    kw = dict(operation="remove", target="user", entry="anything")
    card = _reason(memory, **kw)
    assert "nothing will change" in card
    assert _execute_as_approved(memory, tmp_path, card, **kw).is_error
    assert (tmp_path / "USER.md").read_bytes() == b"\xff\xfe not utf-8 \x80\n"


def test_showing_a_card_creates_no_file(memory, tmp_path):
    assert not (tmp_path / "USER.md").exists()
    _reason(memory, operation="add", target="user", entry="likes tea")
    _reason(memory, operation="remove", target="user", entry="tea")
    assert not (tmp_path / "USER.md").exists()


def test_the_store_changes_nothing_without_the_approved_card(memory, tmp_path):
    # Fail closed: the write is checked against the card the owner approved, and no card means no change.
    _keep(memory, tmp_path, "Will likes tea")
    store = memory.get_user_store()
    args = memory.MemoryToolInput(operation="remove", target="user", entry="tea")
    assert store.change(args, None).startswith("refused")
    assert store.list_entries() == ["Will likes tea"]
