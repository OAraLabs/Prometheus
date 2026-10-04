"""The door — a PERSON starts, consents to, follows and stops a desktop task.

computer-use v1.1 PR 5 (design §5.1, ⚑ ruling: who may start). Built
against a ``FixtureDriver`` (no display, no SDK), the REAL ``SecurityGate``,
the REAL loop and the REAL computer approval channel — the two seams that
are substituted are the ones that prove something by being substituted: the
DRIVER (what reached it is ``dispatched``) and the CHOOSER (a script, so the
test names exactly which row a model would have picked).

What these pin, in the words of Will's rulings (2026-10-04):

* W2 — the app pick covers clicks and Tab/Escape in that app only. Return,
  typing, menus and "Send/Delete/Pay"-type buttons always ask. Door prompts
  never offer a lasting grant.
* W3 — only a person may start a task: an allowed chat user, or a Beacon
  device marked for computer use. Never the API token; never from inside a
  tool call.
* W5 — Telegram carries the consent sentence and the approval prompt (with
  the text to be typed); progress is counts only.
* Stop halts before the next action, through the ``before_act`` fence.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from prometheus.computer.chooser import ScriptedChooser
from prometheus.computer.discovery import AppRecord, WindowRecord
from prometheus.computer.driver import FixtureDriver
from prometheus.computer.types import Element, Observation
from prometheus.permissions.approver import Approver
from prometheus.permissions.audit import AuditLogger
from prometheus.permissions.checker import (
    Grant,
    PermissionMode,
    SecurityGate,
)

TARGET = "box"
APP = "gedit"
PID = 4242
WID = 7
SESSION = "telegram:456"
TYPED = "hello from the test"

#: The person: a Telegram user whose own id is on the allowlist.
PERSON = Approver("telegram", "456", "will")
#: What a prompt-injected model can hold: the daemon's API token.
MODEL = Approver("global-token", "global", "API token")


def _elements():
    return (
        Element(0, "tok-save", "push button", "Save"),
        Element(1, "tok-send", "push button", "Send"),
        Element(2, "tok-name", "text", "Name"),
        Element(3, "tok-file", "menu item", "File"),
    )


def _obs(i: int) -> Observation:
    els = _elements()
    return Observation(
        target=TARGET, app=APP, pid=PID, window_id=WID,
        snapshot_id=f"s{i}", elements=els, degraded=False, truncated=False,
        elements_complete=False, total_element_count=len(els),
        returned_element_count=len(els), web_content_seen=False,
    )


class _Driver(FixtureDriver):
    """A FixtureDriver with hooks: ``on_observe(n)`` / ``on_act(n)`` run in
    the worker thread the loop calls them from, exactly where a real stop
    would race them."""

    def __init__(self, n: int = 60) -> None:
        super().__init__(
            [_obs(i) for i in range(n)],
            apps=[AppRecord(pid=PID, name=APP)],
            windows=[WindowRecord(window_id=WID, pid=PID, app_name=APP,
                                  z_index=1)],
        )
        self.observes = 0
        self.on_observe = None
        self.on_act = None

    def observe(self, target, app, pid, window_id):
        self.observes += 1
        if self.on_observe is not None:
            self.on_observe(self.observes)
        return super().observe(target, app, pid, window_id)

    def act(self, verb, arguments):
        out = super().act(verb, arguments)
        if self.on_act is not None:
            self.on_act(len(self.dispatched))
        return out


class _Integration:
    """Stands in for ComputerIntegration: healthy, one local target bound."""

    enabled = True

    def __init__(self, driver) -> None:
        self._driver = driver
        self.probes = 0

    async def probe(self, *, force: bool = False):
        self.probes += 1
        return {"state": "ready"}

    def driver(self):
        return self._driver

    def local_target(self):
        return TARGET

    @property
    def state(self):
        return "ready"


class _Telegram:
    """Records what the channel would have sent to a chat."""

    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []

    async def send(self, chat_id, text, parse_mode=None, **_):
        self.sent.append((chat_id, text))
        return SimpleNamespace(success=True, message_id=1)


def _rig(tmp_path, ids, *, limits=None, driver=None, telegram=None,
         device_store=None, people_ids=(456,)):
    from prometheus.computer.approvals import ComputerApprovalChannel
    from prometheus.computer.door import PersonCheck
    from prometheus.computer.task import ComputerTaskRunner

    gate = SecurityGate(mode=PermissionMode.DEFAULT,
                        audit_logger=AuditLogger(tmp_path / "audit"))
    people = PersonCheck(device_store=device_store,
                         telegram_user_ids=set(people_ids))
    channel = ComputerApprovalChannel(security_gate=gate, people=people,
                                      telegram_adapter=telegram)
    driver = driver or _Driver()
    runner = ComputerTaskRunner(
        integration=_Integration(driver), gate=gate, channel=channel,
        people=people, limits=limits,
        chooser_factory=lambda: ScriptedChooser(list(ids)),
        skip_preconditions=True, heartbeat_s=3600,
    )
    return SimpleNamespace(runner=runner, channel=channel, driver=driver,
                           gate=gate, people=people)


async def _bind(rig, scope="task"):
    return await rig.runner.bind(SESSION, APP, scope=scope, by=PERSON,
                                 surface="telegram")


async def _start(rig, *, text=None, by=PERSON, notify=None, goal="save it"):
    from prometheus.computer.task import ComputerTaskInput

    return await rig.runner.start(
        ComputerTaskInput(goal=goal, app=APP, text=text),
        session_id=SESSION, surface="telegram", by=by, notify=notify)


async def _next_pending(channel, seen: set[str], timeout: float = 5.0):
    """The next prompt the channel is holding, by request id."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        for rid in list(channel.pending):
            if rid not in seen:
                seen.add(rid)
                return rid
        await asyncio.sleep(0.01)
    raise AssertionError("no approval prompt arrived")


async def _answer(rig, verdicts, by=PERSON):
    """Answer the next len(verdicts) prompts; returns the actions answered."""
    seen: set[str] = set()
    answered = []
    for verdict in verdicts:
        rid = await _next_pending(rig.channel, seen)
        answered.append(rig.channel.pending[rid])
        if verdict == "approve":
            assert await rig.channel.approve(rid, by=by)
        else:
            assert await rig.channel.deny(rid, by=by)
    return answered


def _verbs(driver):
    return [(verb, args.get("key") or args.get("element_token"))
            for verb, args in driver.dispatched]


# ── W2: WHAT THE APP PICK COVERS ────────────────────────────────────────────

async def test_the_binding_lets_a_covered_click_through_without_asking(tmp_path):
    rig = _rig(tmp_path, ["click-0"])
    await _bind(rig)
    task = await _start(rig)
    done = await rig.runner.wait(task.task_id, timeout=10)
    assert _verbs(rig.driver) == [("click", "tok-save")]
    assert done.approvals == 0, "a covered click must not prompt"
    assert not rig.channel.pending


async def test_tab_and_escape_are_covered_and_return_always_asks(tmp_path):
    rig = _rig(tmp_path, ["key-tab", "key-escape", "key-return"])
    await _bind(rig)
    task = await _start(rig)
    answered = await _answer(rig, ["deny"])
    done = await rig.runner.wait(task.task_id, timeout=10)
    assert _verbs(rig.driver) == [("press_key", "tab"),
                                  ("press_key", "escape")]
    assert answered[0].tool_name == "computer_press_key"
    assert answered[0].arguments.get("key") == "return"
    assert done.approvals == 1
    assert done.outcome == "refused"


async def test_typing_always_asks_and_the_prompt_shows_the_text(tmp_path):
    rig = _rig(tmp_path, ["set-2"])
    await _bind(rig)
    task = await _start(rig, text=TYPED)
    answered = await _answer(rig, ["approve"])
    done = await rig.runner.wait(task.task_id, timeout=10)
    assert TYPED in str(answered[0].arguments), (
        "approving a keystroke without seeing it is not consent")
    assert rig.driver.dispatched[0][0] == "set_value"
    assert rig.driver.dispatched[0][1]["text"] == TYPED
    assert done.approvals == 1


async def test_a_menu_always_asks(tmp_path):
    rig = _rig(tmp_path, ["click-3"])
    await _bind(rig)
    task = await _start(rig)
    answered = await _answer(rig, ["deny"])
    await rig.runner.wait(task.task_id, timeout=10)
    assert answered[0].tool_name == "computer_click"
    assert rig.driver.dispatched == []


async def test_a_high_consequence_label_asks_even_when_covered(tmp_path):
    rig = _rig(tmp_path, ["click-1"])
    await _bind(rig)
    task = await _start(rig)
    answered = await _answer(rig, ["deny"])
    done = await rig.runner.wait(task.task_id, timeout=10)
    assert "Send" in answered[0].description
    assert rig.driver.dispatched == []
    assert done.outcome == "refused"


async def test_a_high_consequence_label_under_a_stored_grant_is_refused(tmp_path):
    """A remembered grant never calls the approver, so it cannot be asked —
    the before_act fence refuses instead of letting "Send" through silently."""
    rig = _rig(tmp_path, ["click-1"])
    rig.gate.add_grant(Grant(kind="computer_action",
                             value=f"{TARGET}:{APP}:-:click:background",
                             tool_name="computer_click"))
    await _bind(rig)
    task = await _start(rig)
    done = await rig.runner.wait(task.task_id, timeout=10)
    assert rig.driver.dispatched == []
    assert done.outcome == "refused"
    assert "always asks" in done.reason


async def test_door_prompts_offer_no_lasting_scope(tmp_path):
    from prometheus.computer.door import OnceOnly

    rig = _rig(tmp_path, ["set-2"])
    await _bind(rig)
    task = await _start(rig, text=TYPED)
    rid = await _next_pending(rig.channel, set())
    action = rig.channel.pending[rid]
    assert rig.channel.serialize_pending(action)["extents"] == {}, (
        "a door prompt must not offer until-restart or always")
    with pytest.raises(OnceOnly):
        await rig.channel.approve(rid, by=PERSON, scope="always")
    assert rig.gate.list_grants() == []
    assert await rig.channel.approve(rid, by=PERSON)
    await rig.runner.wait(task.task_id, timeout=10)
    assert rig.gate.list_grants() == [], "a task never mints a lasting grant"


async def test_the_resolution_row_says_binding(tmp_path):
    rig = _rig(tmp_path, ["click-0"])
    binding = await _bind(rig)
    task = await _start(rig)
    await rig.runner.wait(task.task_id, timeout=10)
    rows = rig.gate._audit.query_recent(limit=50)
    assert any(f"binding {binding.binding_id}" in (r.reason or "")
               for r in rows), [r.reason for r in rows]


# ── W3: ONLY A PERSON ───────────────────────────────────────────────────────

async def test_the_api_token_cannot_start_a_task(tmp_path):
    from prometheus.computer.door import NotAPerson

    rig = _rig(tmp_path, ["click-0"])
    await _bind(rig)
    with pytest.raises(NotAPerson):
        await _start(rig, by=MODEL)
    assert rig.driver.dispatched == []


async def test_the_api_token_cannot_bind_an_app(tmp_path):
    from prometheus.computer.door import NotAPerson

    rig = _rig(tmp_path, ["click-0"])
    with pytest.raises(NotAPerson):
        await rig.runner.bind(SESSION, APP, scope="session", by=MODEL,
                              surface="rest")


async def test_a_chat_user_not_on_the_allowlist_cannot_start(tmp_path):
    from prometheus.computer.door import NotAPerson

    rig = _rig(tmp_path, ["click-0"])
    await _bind(rig)
    with pytest.raises(NotAPerson):
        await _start(rig, by=Approver("telegram", "999", "stranger"))


async def test_a_model_credential_cannot_answer_a_desktop_prompt(tmp_path):
    from prometheus.computer.door import NotAPerson

    rig = _rig(tmp_path, ["set-2"])
    await _bind(rig)
    task = await _start(rig, text=TYPED)
    rid = await _next_pending(rig.channel, set())
    with pytest.raises(NotAPerson):
        await rig.channel.approve(rid, by=MODEL)
    assert rid in rig.channel.pending, "the prompt is still waiting"
    # Refusing is the safe direction: anyone authenticated may deny.
    assert await rig.channel.deny(rid, by=MODEL)
    done = await rig.runner.wait(task.task_id, timeout=10)
    assert rig.driver.dispatched == []
    assert done.outcome == "refused"


async def test_a_start_from_inside_a_tool_call_is_refused(tmp_path):
    from prometheus.computer.door import InToolContext
    from prometheus.engine.tool_context import tool_execution

    rig = _rig(tmp_path, ["click-0"])
    await _bind(rig)
    with tool_execution("bash"):
        with pytest.raises(InToolContext):
            await _start(rig)
    assert rig.driver.dispatched == []


async def test_the_agent_loop_marks_tool_execution(tmp_path):
    """The flag the door reads is set by the agent loop around EVERY tool
    call — so no in-process path from a model reaches the runner."""
    from pydantic import BaseModel

    from prometheus.engine.agent_loop import LoopContext, _execute_tool_call
    from prometheus.engine.model_adapter import ModelAdapter
    from prometheus.engine.tool_context import in_tool_execution
    from prometheus.telemetry.tracker import ToolCallTelemetry
    from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

    seen = []

    class _In(BaseModel):
        x: str = ""

    class _Probe(BaseTool):
        name = "door_probe"
        description = "records whether it runs inside tool execution"
        input_model = _In

        def is_read_only(self, arguments):  # noqa: ANN001
            return True

        async def execute(self, arguments, context):  # noqa: ANN001
            seen.append(in_tool_execution())
            return ToolResult(output="ok")

    reg = ToolRegistry()
    reg.register(_Probe())
    ctx = LoopContext(
        provider=None, model="m", system_prompt="", max_tokens=64,
        tool_registry=reg, adapter=ModelAdapter(tier=ModelAdapter.TIER_LIGHT),
        telemetry=ToolCallTelemetry(db_path=tmp_path / "tel.db"),
        session_id="telegram:42",
    )
    assert not in_tool_execution()
    await _execute_tool_call(ctx, "door_probe", "t1", {"x": "y"})
    assert seen == [True]
    assert not in_tool_execution(), "the flag must not leak past the call"


# ── STOP: HALTS BEFORE THE NEXT ACTION ──────────────────────────────────────

async def test_a_stop_mid_task_halts_at_the_before_act_fence(tmp_path):
    """A stored grant means no approver is ever asked — the before_act fence
    is the ONLY check between a stop and the next dispatch. The stop lands
    while step 2 is observing (in its worker thread, as a real one would)."""
    rig = _rig(tmp_path, ["click-0", "click-0", "click-0"])
    rig.gate.add_grant(Grant(kind="computer_action",
                             value=f"{TARGET}:{APP}:-:click:background",
                             tool_name="computer_click"))
    await _bind(rig)
    task = await _start(rig)
    loop = asyncio.get_running_loop()
    stopped = threading.Event()

    def on_observe(n):
        # observe 1 = step 1, observe 2 = its verify, observe 3 = step 2.
        if n == 3:
            stopped.set()
            asyncio.run_coroutine_threadsafe(
                _stop(rig.runner, task.task_id), loop).result(5)

    rig.driver.on_observe = on_observe
    done = await rig.runner.wait(task.task_id, timeout=10)
    assert stopped.is_set()
    assert _verbs(rig.driver) == [("click", "tok-save")], (
        "exactly the action before the stop, and nothing after it")
    assert done.outcome == "stopped"


async def _stop(runner, task_id):
    return runner.stop(task_id)


async def test_a_stop_while_a_prompt_waits_denies_it_and_nothing_lands(tmp_path):
    rig = _rig(tmp_path, ["set-2"])
    await _bind(rig)
    task = await _start(rig, text=TYPED)
    rid = await _next_pending(rig.channel, set())
    assert rig.runner.stop(task.task_id)
    done = await rig.runner.wait(task.task_id, timeout=10)
    assert rid not in rig.channel.pending, "stop resolves the task's prompts"
    assert rig.driver.dispatched == []
    assert done.outcome == "stopped"


async def test_stop_session_stops_every_task_in_the_session(tmp_path):
    rig = _rig(tmp_path, ["set-2"])
    await _bind(rig)
    task = await _start(rig, text=TYPED)
    await _next_pending(rig.channel, set())
    assert rig.runner.stop_session(SESSION) is True
    done = await rig.runner.wait(task.task_id, timeout=10)
    assert done.outcome == "stopped"
    assert rig.runner.stop_session(SESSION) is False, "idempotent"


# ── CEILINGS AND THE WINDOW ─────────────────────────────────────────────────

async def test_the_step_ceiling_ends_the_task(tmp_path):
    from prometheus.computer.task import TaskLimits

    rig = _rig(tmp_path, ["click-0"] * 5, limits=TaskLimits(max_steps=2))
    await _bind(rig)
    task = await _start(rig)
    done = await rig.runner.wait(task.task_id, timeout=10)
    assert len(rig.driver.dispatched) == 2
    assert done.outcome == "limit"


async def test_the_approval_ceiling_refuses_past_it(tmp_path):
    from prometheus.computer.task import TaskLimits

    rig = _rig(tmp_path, ["set-2", "set-2"], limits=TaskLimits(max_approvals=1))
    await _bind(rig)
    task = await _start(rig, text=TYPED)
    await _answer(rig, ["approve"])
    done = await rig.runner.wait(task.task_id, timeout=10)
    assert len(rig.driver.dispatched) == 1
    assert done.outcome == "limit"
    assert not rig.channel.pending


async def test_a_vanished_window_ends_the_task(tmp_path):
    rig = _rig(tmp_path, ["click-0", "click-0"])
    await _bind(rig)

    def on_act(n):
        rig.driver.windows.clear()

    rig.driver.on_act = on_act
    task = await _start(rig)
    done = await rig.runner.wait(task.task_id, timeout=10)
    assert len(rig.driver.dispatched) == 1
    assert done.outcome == "failed"
    assert "window" in done.reason


async def test_one_task_per_target(tmp_path):
    from prometheus.computer.door import TaskBusy

    rig = _rig(tmp_path, ["set-2"])
    await _bind(rig)
    task = await _start(rig, text=TYPED)
    await _next_pending(rig.channel, set())
    with pytest.raises(TaskBusy):
        await _start(rig, text=TYPED)
    rig.runner.stop(task.task_id)
    await rig.runner.wait(task.task_id, timeout=10)


async def test_no_binding_means_the_door_asks_which_app(tmp_path):
    from prometheus.computer.door import NeedsConsent

    rig = _rig(tmp_path, ["click-0"])
    with pytest.raises(NeedsConsent) as info:
        await _start(rig)
    assert info.value.options == [APP]
    assert "Return, typing, menus" in info.value.sentence
    assert rig.driver.dispatched == []


async def test_a_chat_binding_lasts_for_the_one_task(tmp_path):
    rig = _rig(tmp_path, ["click-0"])
    await _bind(rig, scope="task")
    task = await _start(rig)
    await rig.runner.wait(task.task_id, timeout=10)
    assert rig.runner.binding_for(SESSION) is None, (
        "a chat pick covers the task that asked, and nothing after it")


async def test_the_binding_sentence_says_what_it_covers(tmp_path):
    rig = _rig(tmp_path, ["click-0"])
    session = await rig.runner.bind(SESSION, APP, scope="session", by=PERSON,
                                    surface="rest")
    text = session.sentence()
    assert "click and press Tab/Escape in gedit" in text
    assert "Return, typing, menus and anything on a web page ask every time" in text
    assert "until you turn this off" in text
    assert session.expires_at - session.created_at <= 8 * 3600


# ── THE APPROVAL ROUTER: /approve all SKIPS DESKTOP ENTRIES ─────────────────

async def test_approve_all_skips_desktop_entries(tmp_path):
    from prometheus.gateway.commands import approve_detail
    from prometheus.permissions.approval_queue import (
        ApprovalQueue,
        ApprovalQueues,
        ApprovalResult,
        PendingAction,
    )

    rig = _rig(tmp_path, [])
    primary = ApprovalQueue(security_gate=rig.gate)
    plain = PendingAction(request_id="aaaa1111", tool_name="bash",
                          description="ls")
    primary.pending[plain.request_id] = plain
    desk = PendingAction(request_id="bbbb2222", tool_name="computer_click",
                         description="Click the push button 'Save'",
                         task_id="t1", session_id=SESSION, once_only=True)
    rig.channel.pending[desk.request_id] = desk
    router = ApprovalQueues(primary, rig.channel)
    outcome = await approve_detail(router, "all", by=PERSON)
    assert plain._result is ApprovalResult.APPROVED
    assert desk._result is not ApprovalResult.APPROVED
    assert "skipped 1 desktop" in outcome.text


# ── TELEGRAM: THE /computer FAMILY ──────────────────────────────────────────

async def test_telegram_first_use_asks_and_yes_starts(tmp_path):
    from prometheus.gateway.commands import cmd_computer

    tg = _Telegram()
    rig = _rig(tmp_path, ["click-0"], telegram=tg)
    sent: list[str] = []

    async def notify(text):
        sent.append(text)

    kw = dict(by=PERSON, surface="telegram", session_id=SESSION,
              chat_id=456, notify=notify)
    ask = await cmd_computer(rig.runner, "save the document", **kw)
    assert "Prometheus may click and press Tab/Escape in gedit" in ask
    assert "/computer yes" in ask
    assert rig.driver.dispatched == [], "nothing runs before the answer"
    started = await cmd_computer(rig.runner, "yes", **kw)
    assert "Started" in started
    task = rig.runner.tasks(SESSION)[0]
    await rig.runner.wait(task.task_id, timeout=10)
    assert _verbs(rig.driver) == [("click", "tok-save")]


async def test_telegram_progress_is_counts_only_and_prompts_show_the_text(tmp_path):
    """W5: the approval prompt carries the text to be typed; the milestone
    messages carry counts and never an element's label or the text."""
    from prometheus.gateway.commands import cmd_computer

    tg = _Telegram()
    rig = _rig(tmp_path, ["click-0", "set-2"], telegram=tg)
    sent: list[str] = []

    async def notify(text):
        sent.append(text)

    kw = dict(by=PERSON, surface="telegram", session_id=SESSION,
              chat_id=456, notify=notify)
    await cmd_computer(rig.runner, f'fill it in text:"{TYPED}"', **kw)
    await cmd_computer(rig.runner, "yes", **kw)
    await _answer(rig, ["approve"])
    task = rig.runner.tasks(SESSION)[0]
    await rig.runner.wait(task.task_id, timeout=10)
    prompts = [text for _, text in tg.sent]
    assert any(TYPED in p for p in prompts), prompts
    assert all(chat == 456 for chat, _ in tg.sent), "to the chat that started it"
    progress = "\n".join(sent)
    assert "2 steps" in progress and "1 approval" in progress, progress
    for content in (TYPED, "Save", "Name", "tok-"):
        assert content not in progress, (content, progress)


async def test_telegram_stop_stops_a_running_task(tmp_path):
    from prometheus.gateway.commands import cmd_computer

    rig = _rig(tmp_path, ["set-2"])
    await _bind(rig)
    task = await _start(rig, text=TYPED)
    await _next_pending(rig.channel, set())
    reply = await cmd_computer(rig.runner, "stop", by=PERSON,
                               surface="telegram", session_id=SESSION,
                               chat_id=456, notify=None)
    assert "Stopped" in reply
    done = await rig.runner.wait(task.task_id, timeout=10)
    assert done.outcome == "stopped"
    assert rig.driver.dispatched == []


async def test_telegram_stranger_cannot_start(tmp_path):
    from prometheus.gateway.commands import cmd_computer

    rig = _rig(tmp_path, ["click-0"])
    reply = await cmd_computer(rig.runner, "save it",
                               by=Approver("telegram", "999"),
                               surface="telegram", session_id="telegram:999",
                               chat_id=999, notify=None)
    assert "only a person" in reply.lower()
    assert rig.driver.dispatched == []


async def test_computer_off_says_so(tmp_path):
    from prometheus.gateway.commands import cmd_computer

    reply = await cmd_computer(None, "save it", by=PERSON, surface="telegram",
                               session_id=SESSION, chat_id=456, notify=None)
    assert "computer use is off" in reply.lower()


async def test_slack_and_discord_refuse_in_v11():
    from prometheus.gateway.commands import cmd_computer

    for surface in ("slack", "discord"):
        reply = await cmd_computer(object(), "save it",
                                   by=Approver(surface, "u1"),
                                   surface=surface, session_id=f"{surface}:1",
                                   chat_id=None, notify=None)
        assert "telegram" in reply.lower() and "beacon" in reply.lower()


async def test_the_web_chat_slash_cannot_start_a_task(tmp_path):
    """The typed /computer path carries no person credential yet, so it
    refuses to start; status and stop still answer."""
    from prometheus.gateway.commands import CommandContext, run_session_command

    rig = _rig(tmp_path, ["click-0"])
    ctx = CommandContext(session_id="web:abc", computer_runner=rig.runner)
    reply = await run_session_command("computer", "save it", ctx)
    assert "telegram" in reply.lower()
    assert rig.driver.dispatched == []
    status = await run_session_command("computer", "status", ctx)
    assert "computer use" in status.lower()


# ── REST ────────────────────────────────────────────────────────────────────

TOKEN = "global-test-token"


@pytest.fixture
def rest(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.config.device_store import DeviceStore
    from prometheus.web.server import create_app

    monkeypatch.setenv("PROMETHEUS_API_TOKEN", TOKEN)
    store = DeviceStore(tmp_path / "devices.db")
    rig = _rig(tmp_path, ["click-0"], device_store=store)
    app = create_app({}, device_store=store, computer_runner=rig.runner)
    client = TestClient(app)
    phone = store.mint("Will's iPhone", "ios")
    store.set_computer(phone["id"], True, by="telegram:456")
    other = store.mint("tablet", "other")
    return SimpleNamespace(client=client, store=store, rig=rig, phone=phone,
                           other=other, app=app)


def _h(token):
    return {"Authorization": f"Bearer {token}"}


def test_the_global_token_gets_401_on_every_door_route(rest):
    c = rest.client
    sid = "web:abc"
    calls = [
        ("put", f"/api/sessions/{sid}/computer", {"app": APP}),
        ("post", "/api/computer/tasks", {"session_id": sid, "goal": "x"}),
        ("get", "/api/computer/apps", None),
        ("put", f"/api/devices/{rest.other['id']}/computer", {"computer": True}),
    ]
    for method, path, body in calls:
        kw = {"headers": _h(TOKEN)}
        if body is not None:
            kw["json"] = body
        res = getattr(c, method)(path, **kw)
        assert res.status_code == 401, (method, path, res.status_code, res.text)


def test_a_device_not_marked_for_computer_use_gets_401(rest):
    res = rest.client.post("/api/computer/tasks",
                           json={"session_id": "web:abc", "goal": "x"},
                           headers=_h(rest.other["token"]))
    assert res.status_code == 401, res.text


def test_rest_first_use_asks_then_the_pick_is_the_consent(rest):
    c, sid = rest.client, "web:abc"
    first = c.post("/api/computer/tasks", json={"session_id": sid, "goal": "save"},
                   headers=_h(rest.phone["token"]))
    assert first.status_code == 409, first.text
    assert first.json()["error"] == "needs_app"
    assert first.json()["options"] == [APP]
    bound = c.put(f"/api/sessions/{sid}/computer", json={"app": APP},
                  headers=_h(rest.phone["token"]))
    assert bound.status_code == 200, bound.text
    body = bound.json()
    assert body["state"] == "on" and body["scope"] == "session"
    assert "ask every time" in body["describes"]
    started = c.post("/api/computer/tasks", json={"session_id": sid, "goal": "save"},
                     headers=_h(rest.phone["token"]))
    assert started.status_code == 202, started.text
    task_id = started.json()["task_id"]
    state = c.get(f"/api/computer/tasks/{task_id}", headers=_h(TOKEN))
    assert state.status_code == 200


def test_rest_stop_is_never_refused(rest):
    """Stopping can only end a task, so any authenticated caller may."""
    res = rest.client.post("/api/computer/tasks/nope/stop", headers=_h(TOKEN))
    assert res.status_code in (200, 404), res.text


def test_only_a_marked_device_may_mark_another(rest):
    c = rest.client
    res = c.put(f"/api/devices/{rest.other['id']}/computer",
                json={"computer": True}, headers=_h(rest.other["token"]))
    assert res.status_code == 401
    res = c.put(f"/api/devices/{rest.other['id']}/computer",
                json={"computer": True}, headers=_h(rest.phone["token"]))
    assert res.status_code == 200, res.text
    assert rest.store.computer_allowed(rest.other["id"])


def _desk_prompt(rest, *, arguments):
    from prometheus.permissions.approval_queue import PendingAction

    action = PendingAction(
        request_id="cafe1234", tool_name="computer_set_value",
        description="Set the text 'Name' to the prepared text",
        arguments=arguments, task_id="t1", session_id="web:abc",
        once_only=True)
    rest.rig.channel.pending[action.request_id] = action
    return action


def test_the_global_token_cannot_approve_a_desktop_prompt(rest):
    from prometheus.permissions.approval_queue import ApprovalResult

    action = _desk_prompt(rest, arguments={"text": TYPED})
    res = rest.client.post(f"/api/approvals/{action.request_id}/approve",
                           json={"scope": "once"}, headers=_h(TOKEN))
    assert res.status_code == 401, res.text
    assert action._result is not ApprovalResult.APPROVED


def test_an_old_ios_build_cannot_approve_typed_text(rest):
    from prometheus.permissions.approval_queue import ApprovalResult

    action = _desk_prompt(rest, arguments={"text": TYPED})
    res = rest.client.post(f"/api/approvals/{action.request_id}/approve",
                           json={"scope": "once"},
                           headers=_h(rest.phone["token"]))
    assert res.status_code == 409, res.text
    assert "desktop or Telegram" in res.json()["error"]
    assert action._result is not ApprovalResult.APPROVED
    res = rest.client.post(f"/api/approvals/{action.request_id}/approve",
                           json={"scope": "once"},
                           headers={**_h(rest.phone["token"]),
                                    "X-Beacon-Caps": "approval-arguments"})
    assert res.status_code == 200, res.text
    assert action._result is ApprovalResult.APPROVED


def test_rest_approve_all_skips_desktop_prompts(rest):
    from prometheus.permissions.approval_queue import ApprovalResult

    action = _desk_prompt(rest, arguments=None)
    res = rest.client.post("/api/approvals/all/approve", json={"scope": "once"},
                           headers=_h(rest.phone["token"]))
    assert res.status_code == 200, res.text
    assert res.json().get("skipped") == 1
    assert action._result is not ApprovalResult.APPROVED


def test_the_device_listing_says_which_are_marked(rest):
    rows = rest.client.get("/api/devices", headers=_h(TOKEN)).json()
    marked = {r["id"]: r["computer"] for r in rows}
    assert marked[rest.phone["id"]] is True
    assert marked[rest.other["id"]] is False


# ── THE CHAT STOP REACHES THE TASK ──────────────────────────────────────────

async def test_the_chat_stop_stops_a_task_even_with_a_chat_turn_running(tmp_path):
    """Beacon's Stop (WS ``interrupt`` / POST /api/chat/interrupt) calls
    ``interrupt_turn``. A chat message during a task neither hides the task
    from Stop nor gets cancelled in its place."""
    from prometheus.web.ws_server import WebSocketBridge

    rig = _rig(tmp_path, ["set-2"])
    await rig.runner.bind(SESSION, APP, scope="session", by=PERSON,
                          surface="telegram")
    task = await _start(rig, text=TYPED)
    await _next_pending(rig.channel, set())

    bridge = WebSocketBridge(loop_context=object())
    bridge.computer_runner = rig.runner
    chat_turn = asyncio.create_task(asyncio.sleep(30))
    bridge._turn_tasks[SESSION] = chat_turn
    assert bridge.interrupt_turn(SESSION) is True
    done = await rig.runner.wait(task.task_id, timeout=10)
    assert done.outcome == "stopped"
    await asyncio.sleep(0)
    assert chat_turn.cancelled() or chat_turn.cancelling()
    assert rig.driver.dispatched == []
