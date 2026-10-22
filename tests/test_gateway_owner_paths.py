"""Gateway trust, the paths around the first cut: every way a group member could still act as the owner.

tests/test_gateway_owner_trust.py pins the headline: approve, deny, the gate, /note and the learning paths. These
pin the rest of the merged plan:

- the approval queue itself refuses a chat answer that is not the owner's, so a path that reaches it without going
  through the /approve core is refused too;
- every command that changes what the agent may do, or what it learns, is the owner's, refused at each gateway's one
  chokepoint (Telegram's group -1 handler, Slack's global middleware, Discord's command callback);
- a prompt a member queued is still theirs when the owner's turn drains it;
- in a group session no assistant reply is ever mined, because it can repeat what a member said;
- a member's turn never runs a tool on its own, even a read-only one, a remembered grant or the gate turned off:
  every call waits for the owner;
- teacher escalation and GEPA's reply learn nothing from a member's turn;
- Discord without owner_user_ids falls back to the bot application's owner; desktop tasks and pairing use the same
  owner as everything else;
- a group session still compacts past a member's turn, and the summary of a span that holds one is not trusted.

Stand-ins only: the fakes the suite already drives each gateway with, and a scripted model. No network, no GPU box.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from prometheus.engine.messages import ConversationMessage, TextBlock, ToolResultBlock
from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.platform_base import MessageEvent
from prometheus.permissions.approval_queue import ApprovalResult
from prometheus.permissions.approver import Approver
from tests.test_approver_recorded import _answered_by, _pending, _queue
from tests.test_gateway_owner_trust import (
    GROUP,
    MEMBER,
    OWNER,
    RID,
    _ctx,
    _last_user_message,
    _tg_adapter,
    _tg_dispatch_adapter,
    _tg_update,
)

REPO = Path(__file__).resolve().parents[1]
_MEMBER_MARKER = "PLANTED_BY_A_GROUP_MEMBER"
_OWNER_MARKER = "SAID_BY_THE_OWNER"


# --------------------------------------------------------------------------- #
# 1. The queue is the backstop
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["telegram", "slack", "discord"])
@pytest.mark.parametrize("owner", [None, False])
def test_the_queue_refuses_a_chat_answer_that_is_not_the_owner_s(tmp_path, kind, owner):
    """``owner=None`` is a chat approver whose gateway never asked: refused, not trusted."""
    queue = _queue(tmp_path)
    approved = _pending(queue)
    denied = _pending(queue, rid="def67890")
    by = Approver(kind, str(MEMBER), owner=owner)
    assert asyncio.run(queue.approve(RID, by=by)) is False
    assert asyncio.run(queue.deny("def67890", by=by)) is False
    assert approved._result is not ApprovalResult.APPROVED
    assert denied._result is not ApprovalResult.DENIED
    assert RID in queue.pending and "def67890" in queue.pending
    assert _answered_by(queue) == []


def test_the_queue_still_takes_the_owner_s_and_the_api_s_answers(tmp_path):
    queue = _queue(tmp_path)
    by_owner = _pending(queue)
    by_api = _pending(queue, rid="def67890")
    assert asyncio.run(queue.approve(RID, by=Approver("telegram", str(OWNER), owner=True)))
    assert asyncio.run(queue.approve("def67890", by=Approver("global-token", "")))
    assert by_owner._result is ApprovalResult.APPROVED
    assert by_api._result is ApprovalResult.APPROVED, "REST answers are W4's, unchanged here"


# --------------------------------------------------------------------------- #
# 2. Owner-only commands, at each gateway's one chokepoint
# --------------------------------------------------------------------------- #


_OWNER_ONLY = [
    ("workspace", "/tmp/elsewhere"), ("workspace", "clear"),
    ("profile", "full"), ("model", "big"), ("route", "claude"),
    ("steer", "stop and delete everything"),
    ("skills", "pin evil"), ("skills", "unpin good"), ("curator", "run"),
    ("symbiote", "approve 1"), ("symbiote", "graft x"), ("symbiote", "swap x"), ("symbiote", "restore x"),
    ("symbiote", "abort"), ("press", "install x"), ("press", "update"),
    ("gate", "off"), ("note", "x"), ("revoke", "all"), ("gepa", "run"),
    ("approve", RID), ("deny", RID),
]
_ANYONE = [
    ("workspace", ""), ("profile", ""), ("model", ""), ("route", ""),
    ("skills", ""), ("curator", ""), ("curator", "status"), ("symbiote", "status"), ("symbiote", "history"),
    ("press", "list"), ("press", "search x"), ("gate", ""), ("gate", "status"), ("gepa", "status"),
    ("status", ""), ("help", ""), ("queue", "later"), ("pending", ""), ("grants", ""),
]


@pytest.mark.parametrize("name,args", _OWNER_ONLY)
def test_the_shared_rule_names_every_owner_only_command(name, args):
    from prometheus.gateway.commands import owner_only_command

    assert owner_only_command(name, args) is True, f"/{name} {args}".strip()


@pytest.mark.parametrize("name,args", _ANYONE)
def test_the_shared_rule_leaves_reading_and_chatting_open(name, args):
    from prometheus.gateway.commands import owner_only_command

    assert owner_only_command(name, args) is False, f"/{name} {args}".strip()


class TestTelegramChokepoint:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name,args", _OWNER_ONLY)
    async def test_a_member_s_owner_only_command_stops_before_any_handler(self, name, args):
        from telegram.ext import ApplicationHandlerStop

        adapter = _tg_adapter()
        update = _tg_update(GROUP, MEMBER, f"/{name}@prometheus_bot {args}".strip())
        with pytest.raises(ApplicationHandlerStop):
            await adapter._authorize_update(update, _ctx(*args.split()))
        adapter.send.assert_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name,args", _OWNER_ONLY[:6])
    async def test_the_owner_s_goes_through(self, name, args):
        adapter = _tg_adapter()
        await adapter._authorize_update(_tg_update(GROUP, OWNER, f"/{name} {args}".strip()), _ctx(*args.split()))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name,args", _ANYONE[:6])
    async def test_a_member_may_still_read_and_chat(self, name, args):
        adapter = _tg_adapter()
        await adapter._authorize_update(_tg_update(GROUP, MEMBER, f"/{name} {args}".strip()), _ctx(*args.split()))
        await adapter._authorize_update(_tg_update(GROUP, MEMBER, "what's the weather"), _ctx())


@pytest.fixture()
def fake_bolt_response(monkeypatch):
    created: list = []

    class BoltResponse:
        def __init__(self, status=200, body=""):
            self.status = status
            self.body = body
            created.append(self)

    mod = types.ModuleType("slack_bolt.response")
    mod.BoltResponse = BoltResponse
    monkeypatch.setitem(sys.modules, "slack_bolt.response", mod)
    monkeypatch.setitem(sys.modules, "slack_bolt", types.ModuleType("slack_bolt"))
    return created


class TestSlackChokepoint:
    def _adapter(self):
        from tests.test_gateway_g1 import _make_slack_adapter

        adapter = _make_slack_adapter()
        adapter.config.allowed_channels = ["C1"]
        return adapter

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name,args", _OWNER_ONLY)
    async def test_a_member_s_owner_only_command_never_reaches_its_listener(self, fake_bolt_response, name, args):
        adapter = self._adapter()
        nxt = AsyncMock()
        body = {"channel_id": "C1", "user_id": "UMEMBER", "command": f"/prometheus-{name}", "text": args}
        await adapter._authorize_request(body, nxt)
        nxt.assert_not_called()
        assert fake_bolt_response, "refused without a reply"

    @pytest.mark.asyncio
    async def test_the_owner_s_and_a_member_s_chat_go_through(self, fake_bolt_response):
        adapter = self._adapter()
        nxt = AsyncMock()
        await adapter._authorize_request({"channel_id": "C1", "user_id": "UOWNER",
                                          "command": "/prometheus-workspace", "text": "/tmp"}, nxt)
        await adapter._authorize_request({"channel_id": "C1", "user_id": "UMEMBER", "type": "event_callback",
                                          "event": {"type": "message", "text": "hi"}}, nxt)
        assert nxt.await_count == 2


class TestDiscordChokepoint:
    def _adapter(self):
        from tests.test_discord import _make_adapter

        adapter = _make_adapter()
        adapter.config.owner_user_ids = ["4242"]
        return adapter

    @pytest.mark.parametrize("name,args", _OWNER_ONLY)
    def test_a_member_s_owner_only_command_is_refused(self, name, args):
        adapter = self._adapter()
        interaction = SimpleNamespace(user=SimpleNamespace(id=1337, name="m"))
        assert adapter._owner_refusal_for_command(interaction, name, args)

    def test_the_owner_s_is_not(self):
        adapter = self._adapter()
        interaction = SimpleNamespace(user=SimpleNamespace(id=4242, name="o"))
        assert adapter._owner_refusal_for_command(interaction, "workspace", "/tmp") is None

    def test_every_app_command_passes_the_owner_check(self):
        """The check sits in ``_register``'s callback, the one place all families go through (AST, so CI without
        discord.py still proves the wiring)."""
        source = inspect.getsource(sys.modules["prometheus.gateway.discord"].DiscordAdapter._register)
        tree = ast.parse(inspect.cleandoc("\n" + source) if source.startswith(" ") else source)
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert "_owner_refusal_for_command" in calls


# --------------------------------------------------------------------------- #
# 3. A queued prompt keeps its author
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_member_s_queued_prompt_is_still_theirs_when_the_owner_s_turn_drains_it():
    adapter, agent_loop = _tg_dispatch_adapter()
    queue_update = _tg_update(GROUP, MEMBER, f"/queue {_MEMBER_MARKER}")
    await adapter._cmd_queue(queue_update, _ctx(_MEMBER_MARKER))
    await adapter.on_message(MessageEvent(chat_id=GROUP, user_id=OWNER, text="hello", message_id=1,
                                          platform=Platform.TELEGRAM))
    assert agent_loop.run_async.await_count == 2, "control: the queued prompt ran"
    drained = _last_user_message(agent_loop)
    assert _MEMBER_MARKER in drained.text
    assert drained.provenance == "member" and drained.is_trusted is False, "a member's queued prompt ran as the owner"
    assert drained.author == f"telegram:{MEMBER}"


# --------------------------------------------------------------------------- #
# 4. Slack and Discord store who wrote it, as Telegram does
# --------------------------------------------------------------------------- #


def _session_rows(adapter) -> list[ConversationMessage]:
    return [m for s in adapter.session_manager._sessions.values() for m in s.messages if m.role == "user"]


@pytest.mark.asyncio
async def test_slack_stores_a_channel_member_s_message_as_theirs():
    from tests.test_gateway_g1 import _make_slack_adapter

    adapter = _make_slack_adapter()
    result = MagicMock(text="ok", messages=[])
    adapter.agent_loop.run_async = AsyncMock(return_value=result)
    adapter._add_reaction = AsyncMock()
    adapter._remove_reaction = AsyncMock()
    say = AsyncMock()
    await adapter._dispatch_to_agent("C1", "UMEMBER", "hi from a member", "1.0", None, say)
    await adapter._dispatch_to_agent("C1", "UOWNER", "hi from the owner", "2.0", None, say)
    by_text = {m.text: m for m in _session_rows(adapter)}
    assert by_text["hi from a member"].provenance == "member"
    assert by_text["hi from a member"].is_trusted is False
    assert by_text["hi from the owner"].provenance == "user" and by_text["hi from the owner"].is_trusted is True


@pytest.mark.asyncio
async def test_discord_stores_a_server_member_s_message_as_theirs():
    from tests.test_discord import _make_adapter

    adapter = _make_adapter()
    adapter.config.owner_user_ids = ["4242"]
    result = MagicMock(text="ok", messages=[])
    adapter.agent_loop.run_async = AsyncMock(return_value=result)
    adapter.send = AsyncMock()
    adapter._add_reaction = AsyncMock()
    adapter._remove_reaction = AsyncMock()
    for uid, text in ((1337, "hi from a member"), (4242, "hi from the owner")):
        await adapter._dispatch_to_agent(MessageEvent(chat_id=555, user_id=uid, text=text, message_id=1,
                                                      platform=Platform.DISCORD, raw={}))
    by_text = {m.text: m for m in _session_rows(adapter)}
    assert by_text["hi from a member"].provenance == "member"
    assert by_text["hi from the owner"].provenance == "user"


# --------------------------------------------------------------------------- #
# 5. In a group session, no assistant reply is mined
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_no_assistant_reply_in_a_group_session_is_mined(tmp_path):
    from prometheus.memory.extractor import MemoryExtractor
    from prometheus.memory.lcm_conversation_store import LCMConversationStore
    from prometheus.memory.lcm_types import MessagePart
    from prometheus.memory.store import EXTRACTOR_GLOBAL_SCOPE, MemoryStore

    conv = LCMConversationStore(db_path=tmp_path / "lcm.db")
    rows = [
        ("telegram:-100", "user", f"I love {_OWNER_MARKER}", "user", True),
        ("telegram:-100", "user", f"the owner's wife is {_MEMBER_MARKER}", "member", False),
        ("telegram:-100", "assistant", f"Got it, your wife is {_MEMBER_MARKER}_ECHO", "user", False),
        ("telegram:-100", "assistant", "GROUP_REPLY_TO_THE_OWNER", "user", False),
        ("telegram:456", "user", "my dog is called Rex", "user", True),
        ("telegram:456", "assistant", "PRIVATE_REPLY_IS_MINED", "user", False),
    ]
    for i, (sid, role, text, prov, trusted) in enumerate(rows):
        conv.add_message(sid, MessagePart(role=role, content=text, session_id=sid, turn_index=i,
                                          provenance=prov, is_trusted=trusted))
    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.set_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE, 0)
    extractor = MemoryExtractor(store, MagicMock(), lcm_conversation_store=conv)
    seen: list[str] = []

    async def fake_call_model(prompt: str) -> str:
        seen.append(prompt)
        return ""

    extractor._call_model = fake_call_model
    await extractor.run_once()
    prompts = "\n".join(seen)
    assert _OWNER_MARKER in prompts, "control: the owner's line in the group is mined"
    assert _MEMBER_MARKER not in prompts, "a member's words, or a reply repeating them, were mined"
    assert "GROUP_REPLY_TO_THE_OWNER" not in prompts, "an assistant reply in a group session was mined"
    assert "PRIVATE_REPLY_IS_MINED" in prompts, "control: a private session's replies are mined as before"


# --------------------------------------------------------------------------- #
# 6. A member's turn never runs a tool on its own
# --------------------------------------------------------------------------- #


class _AllowEverything:
    """A gate that would allow the call outright: a remembered grant, or /gate off."""

    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.asked: list[str] = []

    def evaluate(self, tool_name, **_kw):  # noqa: ANN001
        from prometheus.permissions.checker import PermissionDecision

        return PermissionDecision.allow("granted")

    async def request_approval(self, tool_name, reason, arguments=None):  # noqa: ANN001
        self.asked.append(tool_name)
        return self.answer


def _run_turn(tmp_path, message: ConversationMessage, gate: _AllowEverything) -> list:
    import time

    from prometheus.engine.agent_loop import LoopContext
    from prometheus.telemetry.tracker import ToolCallTelemetry
    from tests.test_ephemeral_mode import _drive, _registry, _ScriptedProvider, _text_round, _tool_round

    context = LoopContext(provider=_ScriptedProvider([_tool_round("t1"), _text_round()]), model="test",
                          system_prompt="S", max_tokens=256, tool_registry=_registry(),
                          telemetry=ToolCallTelemetry(db_path=tmp_path / f"tel-{time.monotonic_ns()}.db"),
                          permission_checker=gate)
    messages = [message]
    _drive(context, messages, "telegram:-100")
    return messages


def _ran(messages: list) -> bool:
    for m in messages:
        for block in m.content:
            if isinstance(block, ToolResultBlock) and "echoed:" in str(block.content):
                return True
    return False


def test_a_member_s_turn_asks_the_owner_before_any_tool_even_a_granted_read_only_one(tmp_path):
    gate = _AllowEverything(answer=False)
    member = ConversationMessage.from_member_text("echo something", author=f"telegram:{MEMBER}")
    messages = _run_turn(tmp_path, member, gate)
    assert gate.asked == ["echo"], "a member's turn ran a tool without asking the owner"
    assert not _ran(messages), "the tool ran although the owner said no"


def test_the_owner_s_approval_lets_a_member_s_tool_call_run(tmp_path):
    gate = _AllowEverything(answer=True)
    member = ConversationMessage.from_member_text("echo something", author=f"telegram:{MEMBER}")
    assert _ran(_run_turn(tmp_path, member, gate))


def test_the_owner_s_own_turn_runs_under_the_grant_as_before(tmp_path):
    gate = _AllowEverything(answer=False)
    messages = _run_turn(tmp_path, ConversationMessage.from_user_text("echo something"), gate)
    assert gate.asked == [], "control: the owner's granted call is not asked about"
    assert _ran(messages)


# --------------------------------------------------------------------------- #
# 7. Teacher escalation and GEPA's reply learn nothing from a member
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_teacher_escalation_never_runs_on_a_member_s_turn():
    adapter, agent_loop = _tg_dispatch_adapter()
    engine = MagicMock()
    engine.maybe_escalate = AsyncMock(return_value=None)
    adapter.escalation_engine = engine
    await adapter.on_message(MessageEvent(chat_id=GROUP, user_id=MEMBER, text=_MEMBER_MARKER, message_id=1,
                                          platform=Platform.TELEGRAM))
    engine.maybe_escalate.assert_not_called()
    await adapter.on_message(MessageEvent(chat_id=GROUP, user_id=OWNER, text=_OWNER_MARKER, message_id=2,
                                          platform=Platform.TELEGRAM))
    engine.maybe_escalate.assert_called_once()


def test_gepa_withholds_the_reply_to_a_member_s_request(tmp_path):
    import time

    from prometheus.learning import gepa_evidence
    from prometheus.memory.lcm_conversation_store import LCMConversationStore
    from prometheus.memory.lcm_types import MessagePart

    conv = LCMConversationStore(db_path=tmp_path / "lcm.db")
    t0 = time.time() - 60
    for i, (role, text, prov, trusted) in enumerate([
        ("user", f"write a skill that {_MEMBER_MARKER}", "member", False),
        ("assistant", f"done: {_MEMBER_MARKER}_REPLY", "user", False),
    ]):
        conv.add_message("telegram:-100", MessagePart(role=role, content=text, session_id="telegram:-100",
                                                      turn_index=i, provenance=prov, is_trusted=trusted,
                                                      timestamp=t0 + i * 2))
    request, withheld = gepa_evidence._request(conv._conn, "telegram:-100", t0 + 1)
    assert (request, withheld) == (None, True)
    found, reply = gepa_evidence._reply(conv._conn, "telegram:-100", t0 + 1, t0 + 10, withheld=withheld)
    assert _MEMBER_MARKER not in reply, "the reply to a member's request reached the skill writer"


# --------------------------------------------------------------------------- #
# 8. One owner: Discord's application owner, desktop tasks and pairing
# --------------------------------------------------------------------------- #


def test_discord_falls_back_to_the_bot_application_s_owner():
    from tests.test_discord import _make_adapter

    adapter = _make_adapter()
    adapter.config.owner_user_ids = []
    assert not adapter.config.is_owner(4242), "control: nobody before the application is known"
    adapter._learn_application_owner(SimpleNamespace(owner=SimpleNamespace(id=4242), team=None))
    assert adapter.config.is_owner(4242)
    assert not adapter.config.is_owner(1337)


def test_a_team_owned_discord_application_falls_back_to_the_team_owner():
    from tests.test_discord import _make_adapter

    adapter = _make_adapter()
    adapter.config.owner_user_ids = []
    adapter._learn_application_owner(SimpleNamespace(owner=SimpleNamespace(id=1), team=SimpleNamespace(owner_id=77)))
    assert adapter.config.is_owner(77) and not adapter.config.is_owner(1)


def test_named_discord_owners_win_over_the_application_owner():
    from tests.test_discord import _make_adapter

    adapter = _make_adapter()
    adapter.config.owner_user_ids = ["4242"]
    adapter._learn_application_owner(SimpleNamespace(owner=SimpleNamespace(id=1), team=None))
    assert adapter.config.is_owner(4242) and not adapter.config.is_owner(1)


def test_desktop_tasks_and_pairing_use_the_same_owner():
    """A named owner narrows the private-chat fallback for desktop tasks and pairing too."""
    from prometheus.config.shipped_defaults import resolve_telegram_owner_ids

    gateway = {"allowed_chat_ids": [OWNER, 777, GROUP], "owner_user_ids": [OWNER]}
    assert resolve_telegram_owner_ids(gateway) == [OWNER]
    assert resolve_telegram_owner_ids({"allowed_chat_ids": [OWNER, 777, GROUP]}) == [OWNER, 777]

    from prometheus.gateway.telegram_pairing import TelegramPairing

    adapter = SimpleNamespace(config=PlatformConfig(platform=Platform.TELEGRAM, token="t",
                                                    allowed_chat_ids=[OWNER, 777], owner_user_ids=[str(OWNER)]))
    pairing = TelegramPairing.__new__(TelegramPairing)
    pairing.adapter = adapter
    assert pairing.private_chat_ids() == [OWNER], "a pairing prompt would go to a private chat that is not the owner"


# --------------------------------------------------------------------------- #
# 9. A group session still compacts; its summary is not trusted
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_group_session_compacts_past_a_member_s_turn_and_the_summary_is_not_trusted():
    from tests.test_context_compactor import _compactor

    msgs: list[ConversationMessage] = []
    for i in range(10):
        text = f"turn {i} " + "x" * 400
        if i == 1:
            msgs.append(ConversationMessage.from_member_text(text, author=f"telegram:{MEMBER}"))
        else:
            msgs.append(ConversationMessage.from_user_text(text))
        msgs.append(ConversationMessage(role="assistant", content=[TextBlock(text=f"answer {i} " + "y" * 400)]))
    compactor = _compactor(protect_recent_turns=2)
    assert compactor._select_span_end(msgs) > 4, "the span stopped at the member's turn: this session never compacts"
    out = await compactor.apply(msgs, session_id="telegram:-100")
    summary = out[0]
    assert summary.provenance == "compactor", "control: the session compacted"
    assert summary.is_trusted is False, "a summary that folds in a member's words was stored as trusted"


@pytest.mark.asyncio
async def test_an_owner_only_session_s_summary_stays_trusted():
    from tests.test_context_compactor import _compactor

    msgs: list[ConversationMessage] = []
    for i in range(10):
        msgs.append(ConversationMessage.from_user_text(f"turn {i} " + "x" * 400))
        msgs.append(ConversationMessage(role="assistant", content=[TextBlock(text=f"answer {i} " + "y" * 400)]))
    out = await _compactor(protect_recent_turns=2).apply(msgs, session_id="telegram:456")
    assert out[0].provenance == "compactor" and out[0].is_trusted is True
