"""The door: one person-started desktop task, start to stop (design §5.1).

    a person ─► bind(app) ─► start(goal) ─► [stop? → window? → loop.step] × N ─► end
                    │                               │
                    └── Binding ◄── SessionConsent ◄┘  (the loop's approver)

WHAT THIS MODULE IS, AND IS NOT
--------------------------------
* It is how a PERSON starts computer use: from Telegram (``/computer``), from
  Beacon (``PUT /api/sessions/{id}/computer`` then ``POST
  /api/computer/tasks``). Every surface funnels into
  :meth:`ComputerTaskRunner.start`.
* It is NOT a tool. Nothing here registers anything; ``computer.registered``
  stays 0 with the switch on or off (``tests/test_computer_registration_pin.py``).
  Registering ``computer_task`` for a model is L1, a separate ruling.

WHO (W3) — :mod:`prometheus.computer.door`. Only a person's credential binds
an app or starts a task, and never from inside a tool call.

WHAT THE APP PICK COVERS (W2)
-----------------------------
The answer to "which app may I use?" is a :class:`Binding` — and it is the
consent. It covers ``click`` and ``press_key`` with Tab or Escape, in that
app, on that target, in the background, with positively NO web content
(site ``-``). "That app" is the app under any name it was resolved by when
the person picked it ("Text Editor" is also ``gnome-text-editor``), never
under a launcher many apps share (``python3``, ``flatpak``). Everything else
asks every time, approve-once:

* Return (it activates whatever has focus, a default "Send" included);
* setting or typing text, and menus — a menu verb, or a click on a menu item;
* a label on the high-consequence list (Send, Delete, Pay, …);
* anything whose site is not ``-``;
* and once the per-task approval ceiling is reached, nothing more is asked:
  the task ends.

The gate is unchanged. ``SessionConsent`` is the loop's ``approve`` callback:
it re-derives the extent with the gate's own ``computer_extent_for`` and
either answers from the binding (writing an audit row naming it) or asks the
person through the computer approval channel.

THE FENCE (``before_act``, from #653)
-------------------------------------
A remembered grant lets the gate allow an action WITHOUT asking anyone, so
the per-action checks that must hold anyway run in the loop's synchronous
``before_act`` seam — with no await between it and the dispatch: the stop,
the key set, the high-consequence and menu rule, and the ceiling. A label
that always asks, under a grant that cannot ask, is REFUSED.

STOP (§5.1.7)
-------------
Cooperative and honest. A stop is checked before each step, by
``SessionConsent`` on entry and on return, and in ``before_act``, the last
synchronous point before dispatch; pending prompts are denied
(``deny_task``). No new action starts after a stop is acknowledged. One
already-dispatched driver call cannot be recalled and may still land — the
task says so ("in flight at stop") rather than claiming an instant stop.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from pydantic import BaseModel

from prometheus.computer.actions import schema_for
from prometheus.computer.chooser import RuleChooser
from prometheus.computer.discovery import (
    RESOLVED,
    AppIdentity,
    AppRecord,
    WindowRecord,
    is_shared_launcher,
    resolve_app,
    resolve_window,
)
from prometheus.computer.door import (
    ComputerUseOff,
    InToolContext,
    IntegrationDown,
    NeedsConsent,
    PersonCheck,
    TaskBusy,
)
from prometheus.computer.loop import VERB_FOR_TOOL_NAME, ComputerUseLoop
from prometheus.computer.types import Observation
from prometheus.engine.tool_context import in_tool_execution
from prometheus.permissions.approval_queue import ApprovalResult
from prometheus.permissions.approver import Approver, in_process
from prometheus.permissions.computer_extent import (
    computer_extent_for,
    normalise_term,
)
from prometheus.permissions.computer_schema import (
    DELIVERY_BACKGROUND,
    SITE_NONE,
)

logger = logging.getLogger(__name__)

# ── floors (deliberately not config keys, design §5.3.4) ────────────────────

#: What a binding covers besides clicks. The press_key extent does not name
#: the key, and the key set includes backspace and delete, so the KEY is
#: checked here.
BINDING_KEYS: frozenset[str] = frozenset({"tab", "escape"})

#: A Beacon binding lasts for the session, at most this long (the same
#: figure Cua uses for its browser-profile grants).
MAX_BINDING_SECONDS = 8 * 3600

#: An answer that took longer than this was given about a window that may
#: have changed: the step is dropped and the next one looks again and asks.
APPROVAL_WAIT_FLOOR_S = 30.0

#: App text is capped where it crosses a boundary (the prompt, the log).
APP_TEXT_CAP = 120

#: Labels that always ask, even when the binding covers the verb and even
#: under a remembered grant. Cua RFC 4268's risk tags, as data. A denylist on
#: app text: a mitigation, not a control — Return and payload verbs ask
#: regardless, because no label list can see what Return would activate.
HIGH_CONSEQUENCE: tuple[str, ...] = (
    "send", "delete", "remove", "pay", "transfer", "purchase", "submit",
    "confirm", "sign",
)

_MENU_ROLES: frozenset[str] = frozenset({"menu", "menu item", "menu bar",
                                        "menubar", "menuitem"})

#: Verbs a binding never covers (payload or menu).
_ALWAYS_ASK_VERBS: frozenset[str] = frozenset({"type_text", "set_value",
                                              "invoke_menu"})

SCOPE_SESSION = "session"
SCOPE_TASK = "task"

OUTCOMES = ("done", "abstained", "stopped", "refused", "failed", "limit")


class ComputerTaskInput(BaseModel):
    """What a person asked for. Also the schema a future tool would take —
    registering it is L1, not v1.1."""

    goal: str
    #: "my editor", "gedit"; None → the binding's app, or ask.
    app: str | None = None
    #: The ONLY text that may be typed. Never extracted from ``goal``.
    text: str | None = None
    #: A declared target name; None → the single declared local target.
    target: str | None = None


@dataclass
class TaskLimits:
    max_steps: int = 20
    max_seconds: int = 600
    max_approvals: int = 10
    max_reobserve: int = 3

    @classmethod
    def from_config(cls, block: Mapping[str, Any] | None,
                    errors: list[str] | None = None) -> TaskLimits:
        out = cls()
        raw = block if isinstance(block, Mapping) else {}
        for key in ("max_steps", "max_seconds", "max_approvals",
                    "max_reobserve"):
            if key not in raw:
                continue
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                if errors is not None:
                    errors.append(f"computer_use.task.{key} must be a positive "
                                  f"integer; using {getattr(out, key)}")
                continue
            setattr(out, key, value)
        return out


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", (text or "").lower()) if w]


def high_consequence(label: str) -> str | None:
    """The high-consequence word a label carries, or None."""
    for word in _words(label):
        for risky in HIGH_CONSEQUENCE:
            if word.startswith(risky):
                return risky
    return None


def cap_app_text(text: str, cap: int = APP_TEXT_CAP) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= cap else text[: cap - 1] + "…"


def binding_sentence(app: str, target: str, scope: str) -> str:
    """The sentence the person reads — exactly what the binding grants."""
    until = ("until you turn this off" if scope == SCOPE_SESSION
             else "for this task")
    return (
        f"Prometheus may click and press Tab/Escape in {app} on {target}, in "
        f"the background, and read everything shown in its windows (not just "
        f"the front one), {until}. Return, typing, menus and anything on a "
        f"web page ask every time, and so do buttons like Send, Delete or "
        f"Pay."
    )


@dataclass
class Binding:
    """A person's answer to "which app may I use?" — the consent (W2)."""

    binding_id: str
    session_id: str
    target: str
    app: str
    scope: str
    set_by: dict[str, str]
    created_at: float
    expires_at: float
    task_id: str | None = None
    #: The app as it was resolved when the person picked it: every name it
    #: goes by. Not shown anywhere — the person reads ``app``.
    identity: AppIdentity | None = None

    def expired(self, now: float) -> bool:
        return now >= self.expires_at

    def _app_names(self) -> set[str]:
        """The app terms this pick covers: the name the person picked, and
        the app's other names — except a launcher many apps share.

        ⚠ THE SAME APP, NOT A WIDER ONE. The driver may name a window's app
        by its executable (``gnome-text-editor``) where the person picked its
        display name ("Text Editor"); without this, every click and Tab in the
        picked app asked. A shared launcher (``python3``, ``flatpak``) is not
        accepted: there is no pid here, so it would cover every app started
        the same way."""
        names = {normalise_term(self.app)}
        if self.identity is not None:
            names |= {normalise_term(n) for n in self.identity.names
                      if not is_shared_launcher(n)}
        return names

    def covers(self, extent: Any, arguments: Mapping[str, Any]) -> bool:
        """Does THIS answer cover THIS extent? Anything not positively
        covered asks — see the module docstring for the list."""
        if extent is None:
            return False
        if extent.target != self.target:
            return False
        if normalise_term(extent.app) not in self._app_names():
            return False
        if extent.site != SITE_NONE or extent.delivery != DELIVERY_BACKGROUND:
            return False
        if extent.payload_params:
            return False
        if extent.verb == "click":
            return True
        if extent.verb == "press_key":
            return str(arguments.get("key", "")).lower() in BINDING_KEYS
        return False

    def sentence(self) -> str:
        return binding_sentence(self.app, self.target, self.scope)

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": "on", "session_id": self.session_id,
            "target": self.target, "app": self.app,
            "describes": self.sentence(), "scope": self.scope,
            "covers": ["click", "press_key:tab", "press_key:escape"],
            "set_by": {"surface": self.set_by.get("surface", "")},
            "expires_at": self.expires_at,
        }


@dataclass
class ComputerTask:
    task_id: str
    session_id: str
    goal: str
    app: str
    target: str
    surface: str
    started_by: Approver
    text: str | None = None
    chat_id: int | None = None
    status: str = "running"
    outcome: str | None = None
    reason: str = ""
    steps: int = 0
    approvals: int = 0
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    stop_requested: bool = False
    in_flight_at_stop: bool = False
    _handle: asyncio.Task | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        """For the REST status route. No element, no token, no typed text."""
        return {
            "task_id": self.task_id, "session_id": self.session_id,
            "app": self.app, "target": self.target, "goal": self.goal,
            "surface": self.surface, "status": self.status,
            "outcome": self.outcome, "reason": self.reason,
            "steps": self.steps, "approvals": self.approvals,
            "started_at": self.started_at, "ended_at": self.ended_at,
            "in_flight_at_stop": self.in_flight_at_stop,
            "text_chars": len(self.text) if self.text else 0,
        }


class _SeeingDriver:
    """The task's driver, unchanged — except that it remembers the last few
    observations, so consent can see WHICH element an approval is about
    (its role and label) from the snapshot the candidate was built from."""

    def __init__(self, driver: Any) -> None:
        self._driver = driver
        self._seen: dict[str, Observation] = {}
        self._order: list[str] = []

    def observe(self, target: str, app: str, pid: int, window_id: int):
        obs = self._driver.observe(target, app, pid, window_id)
        sid = getattr(obs, "snapshot_id", None)
        if sid:
            self._seen[sid] = obs
            self._order.append(sid)
            while len(self._order) > 4:
                self._seen.pop(self._order.pop(0), None)
        return obs

    def act(self, verb: str, arguments: dict[str, Any]):
        return self._driver.act(verb, arguments)

    def element_for(self, arguments: Mapping[str, Any]):
        obs = self._seen.get(str(arguments.get("snapshot_id") or ""))
        token = arguments.get("element_token")
        if obs is None or not token:
            return None
        for el in obs.elements:
            if el.element_token == token:
                return el
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._driver, name)


def _what(verb: str, element: Any, arguments: Mapping[str, Any]) -> str:
    """What will be acted on, in words (D16). App text capped and marked."""
    if verb == "press_key":
        return f"Press {arguments.get('key', '?')}"
    thing = cap_app_text(element.describe()) if element is not None else "an element"
    if verb in ("set_value", "type_text"):
        return f"Set the {thing} to the text below"
    if verb == "invoke_menu":
        return f"Open the menu {thing}"
    return f"Click the {thing}"


class SessionConsent:
    """The loop's ``approve`` and ``before_act`` for ONE task."""

    def __init__(self, runner: ComputerTaskRunner, task: ComputerTask,
                 binding: Binding, driver: _SeeingDriver) -> None:
        self._runner = runner
        self._task = task
        self._binding = binding
        self._driver = driver
        self._prompted: tuple[str, str] | None = None
        self.stale_approval = False
        self.limit_hit = False
        self.fence_reason = ""
        self.dispatching = False
        #: How this step's action was consented to: "binding" | "prompt" |
        #: None (then a stored grant allowed it, or nothing did).
        self.consent: str | None = None

    def begin_step(self) -> None:
        self._prompted = None
        self.stale_approval = False
        self.fence_reason = ""
        self.consent = None

    @staticmethod
    def _key(arguments: Mapping[str, Any]) -> tuple[str, str]:
        return (str(arguments.get("snapshot_id") or ""),
                str(arguments.get("element_token") or arguments.get("key") or ""))

    def _stopped(self) -> bool:
        return self._task.stop_requested

    # ── approve: the loop's callback when the gate says APPROVE ─────────

    async def approve(self, tool_name: str, reason: str, *,
                      arguments: Mapping[str, Any]) -> bool:
        if self._stopped():
            return False
        verb = VERB_FOR_TOOL_NAME.get(tool_name)
        element = self._driver.element_for(arguments)
        extent = None
        try:
            if verb is not None:
                extent, unknown = computer_extent_for(
                    tool_name, dict(arguments), schema=schema_for(verb))
                if unknown:
                    extent = None
        except Exception:  # noqa: BLE001 - anything unclear asks
            extent = None

        if (extent is not None and verb not in _ALWAYS_ASK_VERBS
                and self._binding.covers(extent, arguments)
                and not self._label_asks(element)):
            self._runner._audit_binding(tool_name, self._binding, extent)
            self.consent = "binding"
            return True

        # It has to ask. The ceiling first — past it, nothing more is asked.
        if self._task.approvals >= self._runner.limits.max_approvals:
            self.limit_hit = True
            self.fence_reason = (f"the task reached its approval ceiling "
                                 f"({self._runner.limits.max_approvals})")
            return False
        self._task.approvals += 1
        what = _what(verb or tool_name, element, arguments)
        description = f"{what} — {reason}" if reason else what
        shown = {k: arguments[k] for k in ("text", "key") if k in arguments}
        started = self._runner._clock()
        live = self._runner.live

        async def waiting(action: Any) -> None:
            if live is not None:
                await live.step(
                    self._task, status="awaiting_approval", verb=verb,
                    description=what,
                    extent=extent.value if extent is not None else "",
                    approval_request_id=action.request_id)

        result = await self._runner.channel.request_for_task(
            tool_name=tool_name, description=description, extent=extent,
            arguments=shown or None, task_id=self._task.task_id,
            session_id=self._task.session_id, chat_id=self._task.chat_id,
            on_pending=waiting)
        waited = self._runner._clock() - started
        if self._stopped():
            return False
        if result is not ApprovalResult.APPROVED:
            return False
        if waited > APPROVAL_WAIT_FLOOR_S:
            # The window may have changed while the person decided. Look
            # again and ask again rather than act on an old picture.
            self.stale_approval = True
            return False
        self._prompted = self._key(arguments)
        self.consent = "prompt"
        return True

    def _label_asks(self, element: Any) -> bool:
        if element is None:
            return False
        if str(element.role or "").lower() in _MENU_ROLES:
            return True
        return high_consequence(element.label or "") is not None

    # ── before_act: the synchronous fence, no await before the dispatch ──

    def before_act(self, candidate: Any, extent: Any, decision: Any) -> bool:
        if self._stopped():
            self.fence_reason = "stopped"
            return False
        arguments = candidate.arguments
        prompted = self._prompted == self._key(arguments)
        verb = VERB_FOR_TOOL_NAME.get(candidate.tool_name, "")
        if not prompted:
            label = str(candidate.target_description or "")
            if verb in _ALWAYS_ASK_VERBS:
                self.fence_reason = (f"{verb} always asks, and nothing asked "
                                     f"— refused")
                return False
            if verb == "press_key" and str(arguments.get("key", "")).lower() \
                    not in BINDING_KEYS:
                self.fence_reason = (f"pressing {arguments.get('key')} always "
                                     f"asks, and a remembered grant cannot ask "
                                     f"— refused rather than let through")
                return False
            element = self._driver.element_for(arguments)
            if self._label_asks(element) or high_consequence(label):
                self.fence_reason = (
                    f"{cap_app_text(label)} always asks, and a remembered "
                    f"grant cannot ask — refused rather than let through")
                return False
        self.dispatching = True
        return True


class ComputerTaskRunner:
    """Every surface's door into a desktop task. See the module docstring."""

    def __init__(
        self,
        *,
        integration: Any,
        gate: Any,
        channel: Any,
        people: PersonCheck,
        limits: TaskLimits | None = None,
        aliases: Mapping[str, list[str]] | None = None,
        chooser_factory: Callable[[], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        heartbeat_s: float = 60.0,
        skip_preconditions: bool = False,
    ) -> None:
        self.integration = integration
        self.gate = gate
        self.channel = channel
        self.people = people
        self.limits = limits or TaskLimits()
        self.aliases = dict(aliases or {})
        self._chooser_factory = chooser_factory or RuleChooser
        self._clock = clock
        self._wall = wall
        self._heartbeat_s = heartbeat_s
        self._skip_preconditions = skip_preconditions
        self._bindings: dict[str, Binding] = {}
        self._tasks: dict[str, ComputerTask] = {}
        #: Telegram's pending "which app?" questions (see gateway.commands).
        self.proposals: dict[tuple[str, str], Any] = {}
        #: The cockpit's action log (computer.livestream), or None.
        self.live: Any = None
        #: Fire-and-forget work (a stop's prompt denials). Held here because
        #: the event loop keeps only a WEAK reference to a task: one nobody
        #: holds can be collected before it runs.
        self._background: set[asyncio.Task] = set()

    # ── health and discovery ────────────────────────────────────────────

    async def _driver(self) -> Any:
        """A forced probe, then the bound driver — or a refusal naming why."""
        if not getattr(self.integration, "enabled", False):
            raise ComputerUseOff("Computer use is off on this daemon.")
        await self.integration.probe(force=True)
        driver = self.integration.driver()
        if driver is None:
            raise IntegrationDown(
                f"The desktop driver is not ready "
                f"({getattr(self.integration, 'state', 'unknown')}) — "
                f"see /computer status.")
        return driver

    def _target(self) -> str:
        name = self.integration.local_target()
        if not name:
            raise IntegrationDown("No local target is declared.")
        return str(name)

    async def _discover(self, driver: Any) -> tuple[list[AppRecord], list[WindowRecord]]:
        apps = await asyncio.to_thread(driver.list_apps)
        windows = await asyncio.to_thread(driver.list_windows, None, True)
        return list(apps), list(windows)

    async def app_names(self) -> list[str]:
        """Running apps with an on-screen window — names only."""
        driver = await self._driver()
        apps, windows = await self._discover(driver)
        res = resolve_app("", apps, windows, self.aliases)
        return list(res.options)

    async def _resolve(self, phrase: str | None):
        driver = await self._driver()
        apps, windows = await self._discover(driver)
        if phrase:
            return resolve_app(phrase, apps, windows, self.aliases)
        # No app named: the one running app with a window is the proposal;
        # several (or none) is a question.
        res = resolve_app("", apps, windows, self.aliases)
        if len(res.options) == 1:
            return resolve_app(res.options[0], apps, windows, self.aliases)
        return res

    async def propose(self, phrase: str | None) -> tuple[str | None, list[str]]:
        """(the app a phrase resolves to, or None; the names to choose from)."""
        res = await self._resolve(phrase)
        if res.status == RESOLVED and res.app is not None:
            return res.app.name, [res.app.name]
        return None, list(res.options)

    # ── the binding (the consent) ───────────────────────────────────────

    def _guard(self, by: Approver) -> None:
        if in_tool_execution():
            raise InToolContext(
                "A desktop task cannot be started or consented to from "
                "inside a tool call — only a person may.")
        self.people.require(by)

    async def bind(self, session_id: str, app: str, *, scope: str,
                   by: Approver, surface: str) -> Binding:
        """A person picked ``app`` for ``session_id``. That IS the consent."""
        self._guard(by)
        if scope not in (SCOPE_SESSION, SCOPE_TASK):
            raise ValueError(f"scope must be {SCOPE_SESSION!r} or {SCOPE_TASK!r}")
        res = await self._resolve(app)
        if res.status != RESOLVED or res.app is None:
            raise NeedsConsent(res.question or "Pick an app that is running.",
                               options=res.options)
        target = self._target()
        now = self._wall()
        binding = Binding(
            binding_id=uuid.uuid4().hex[:8], session_id=session_id,
            target=target, app=res.app.name, scope=scope,
            set_by={"surface": surface, "by": by.label},
            created_at=now, expires_at=now + MAX_BINDING_SECONDS,
            identity=AppIdentity.of(res.app))
        self._bindings[session_id] = binding
        self._audit_note(by, f"computer binding {binding.binding_id} on: "
                             f"{binding.app} ({scope}) for {session_id}")
        if self.live is not None:
            await self.live.binding(binding, "on")
        return binding

    def binding_for(self, session_id: str) -> Binding | None:
        binding = self._bindings.get(session_id)
        if binding is not None and binding.expired(self._wall()):
            self._bindings.pop(session_id, None)
            return None
        return binding

    def unbind(self, session_id: str) -> bool:
        binding = self._bindings.pop(session_id, None)
        if binding is None:
            return False
        if self.live is not None:
            self._spawn(self.live.binding(binding, "off"))
        return True

    # ── tasks ───────────────────────────────────────────────────────────

    async def start(
        self,
        inp: ComputerTaskInput,
        *,
        session_id: str,
        surface: str,
        by: Approver,
        notify: Callable[[str], Awaitable[Any]] | None = None,
        chat_id: int | None = None,
    ) -> ComputerTask:
        """Start a task. Returns at once; the task runs in the background."""
        self._guard(by)
        driver = await self._driver()
        target = inp.target or self._target()
        binding = self.binding_for(session_id)
        if binding is None or (inp.app and not _same(inp.app, binding.app,
                                                     self.aliases)):
            name, options = await self.propose(inp.app)
            raise NeedsConsent(
                "Pick the app this task may use — the pick is the consent.",
                options=options,
                sentence=binding_sentence(name or "<the app you pick>", target,
                                          SCOPE_TASK if surface == "telegram"
                                          else SCOPE_SESSION),
                app=name)
        if binding.task_id is not None and binding.task_id in self._tasks \
                and self._tasks[binding.task_id].status == "running":
            raise TaskBusy("A desktop task is already running in this session.")
        for other in self._tasks.values():
            if other.status == "running" and other.target == binding.target:
                raise TaskBusy(
                    f"Desktop task {other.task_id} is already running on "
                    f"{other.target} — one at a time. /computer stop first.")
        apps, windows = await self._discover(driver)
        res = resolve_app(binding.app, apps, windows, self.aliases)
        if res.status != RESOLVED or res.app is None:
            raise NeedsConsent(res.question or f"{binding.app} is not on screen.",
                               options=res.options)
        task = ComputerTask(
            task_id=uuid.uuid4().hex[:8], session_id=session_id,
            goal=inp.goal, app=binding.app, target=binding.target,
            surface=surface, started_by=by, text=inp.text, chat_id=chat_id)
        if binding.scope == SCOPE_TASK:
            binding.task_id = task.task_id
        self._tasks[task.task_id] = task
        self._audit_note(by, f"computer task {task.task_id} started in "
                             f"{task.app} for {session_id} ({surface})")
        chooser = self._chooser_factory()
        if self.live is not None:
            await self.live.task_started(
                task, limits=self.limits,
                chooser=str(getattr(chooser, "name", "rule")))
        task._handle = asyncio.create_task(
            self._run(task, driver, binding, res.app.pid, notify, chooser,
                      identity=AppIdentity.of(res.app)),
            name=f"computer-task-{task.task_id}")
        return task

    def get(self, task_id: str) -> ComputerTask | None:
        return self._tasks.get(task_id)

    def tasks(self, session_id: str | None = None) -> list[ComputerTask]:
        return [t for t in self._tasks.values()
                if session_id is None or t.session_id == session_id]

    def running(self) -> list[ComputerTask]:
        return [t for t in self._tasks.values() if t.status == "running"]

    async def wait(self, task_id: str, timeout: float | None = None) -> ComputerTask:
        task = self._tasks[task_id]
        if task._handle is not None:
            await asyncio.wait_for(asyncio.shield(task._handle), timeout)
        return task

    # ── stop: never refused ─────────────────────────────────────────────

    def stop(self, task_id: str) -> bool:
        """Ask a task to stop. Never refused to any caller: it can only end
        something. True if a running task was told."""
        task = self._tasks.get(task_id)
        if task is None or task.status != "running":
            return False
        task.stop_requested = True
        self._spawn(self.channel.deny_task(task_id,
                                           by=in_process("computer-stop")))
        return True

    def _spawn(self, coro: Any) -> None:
        loop = _running_loop()
        if loop is None:
            coro.close()
            return
        job = loop.create_task(coro)
        self._background.add(job)
        job.add_done_callback(self._background.discard)

    def stop_session(self, session_id: str) -> bool:
        """Stop every running task in a chat session (the chat Stop)."""
        hit = False
        for task in list(self._tasks.values()):
            if task.session_id == session_id and task.status == "running":
                hit = self.stop(task.task_id) or hit
        return hit

    # ── the run ─────────────────────────────────────────────────────────

    async def _run(self, task: ComputerTask, driver: Any, binding: Binding,
                   pid: int, notify, chooser: Any = None, *,
                   identity: AppIdentity | None = None) -> None:
        seeing = _SeeingDriver(driver)
        consent = SessionConsent(self, task, binding, seeing)
        picker = _StopAwareChooser(chooser or self._chooser_factory(), task)
        loop = ComputerUseLoop(
            seeing, picker,
            self.gate, approve=consent.approve, origin="user",
            skip_preconditions=self._skip_preconditions,
            before_act=consent.before_act)
        beat = (asyncio.create_task(self._heartbeat(task, notify))
                if notify is not None else None)
        started = self._clock()
        history: list[str] = []
        reobserve = 0
        outcome, reason = "failed", ""
        try:
            while True:
                if task.stop_requested:
                    outcome, reason = "stopped", "stopped by a person"
                    break
                if task.steps >= self.limits.max_steps:
                    outcome, reason = "limit", (f"reached the step ceiling "
                                                f"({self.limits.max_steps})")
                    break
                if self._clock() - started > self.limits.max_seconds:
                    outcome, reason = "limit", (f"reached the time ceiling "
                                                f"({self.limits.max_seconds}s)")
                    break
                window = await asyncio.to_thread(resolve_window, driver, pid)
                if window is None:
                    outcome, reason = "failed", ("the app's window is gone — "
                                                 "nothing else is used instead")
                    break
                consent.begin_step()
                consent.dispatching = False
                step_started = self._clock()
                result = await loop.step(
                    task.goal, task.target, task.app, pid, window.window_id,
                    text_to_type=task.text, history=history,
                    identity=identity)
                after_stop = (result.status == "executed"
                              and task.stop_requested)
                if self.live is not None:
                    await self._log_step(task, result, consent, picker,
                                         after_stop=after_stop,
                                         started=step_started)
                if result.status == "executed":
                    task.steps += 1
                    history = list(result.history)
                    reobserve = 0
                    if after_stop:
                        task.in_flight_at_stop = True
                    continue
                if result.status == "reobserve":
                    reobserve += 1
                    if reobserve >= self.limits.max_reobserve:
                        outcome, reason = "limit", "looked again too many times"
                        break
                    continue
                if result.status == "abstained":
                    if task.stop_requested:
                        outcome, reason = "stopped", "stopped by a person"
                        break
                    outcome = "done" if task.steps else "abstained"
                    reason = ("nothing more serves the goal" if task.steps
                              else "nothing in the window serves the goal")
                    break
                if result.status == "refused":
                    if task.stop_requested:
                        outcome, reason = "stopped", "stopped by a person"
                        break
                    if consent.limit_hit:
                        outcome, reason = "limit", consent.fence_reason
                        break
                    if consent.stale_approval:
                        continue
                    outcome = "refused"
                    reason = consent.fence_reason or result.reason
                    break
                outcome, reason = "failed", result.reason or result.status
                break
        except asyncio.CancelledError:
            outcome, reason = "stopped", "the daemon cancelled the task"
            raise
        except Exception as exc:  # noqa: BLE001 - a task ends, the daemon does not
            logger.warning("computer task %s failed", task.task_id, exc_info=True)
            outcome, reason = "failed", f"{exc.__class__.__name__}: {exc}"
        finally:
            if beat is not None:
                beat.cancel()
            task.status = "ended"
            task.outcome = outcome
            task.reason = reason
            task.ended_at = self._wall()
            if binding.scope == SCOPE_TASK and \
                    self._bindings.get(task.session_id) is binding:
                self._bindings.pop(task.session_id, None)
            try:
                await self.channel.deny_task(task.task_id,
                                             by=in_process("computer-task-end"))
            except Exception:  # noqa: BLE001
                logger.debug("deny_task at task end failed", exc_info=True)
            self._audit_note(task.started_by,
                             f"computer task {task.task_id} ended: {outcome} "
                             f"({task.steps} steps, {task.approvals} approvals)")
            if self.live is not None:
                try:
                    await self.live.task_ended(task, summary=end_message(task))
                    if binding.scope == SCOPE_TASK:
                        await self.live.binding(binding, "off")
                except Exception:  # noqa: BLE001 - the log never ends a task
                    logger.debug("action log at task end failed", exc_info=True)
            if notify is not None:
                await _quiet(notify, end_message(task))

    async def _log_step(self, task: ComputerTask, result: Any,
                        consent: SessionConsent, picker: _StopAwareChooser,
                        *, after_stop: bool, started: float) -> None:
        candidate = result.candidate
        verb = (VERB_FOR_TOOL_NAME.get(candidate.tool_name)
                if candidate is not None else None)
        how = consent.consent
        if how is None and result.status == "executed":
            how = "grant"  # the gate allowed it from a remembered grant
        status = "in_flight_at_stop" if after_stop else result.status
        try:
            await self.live.step(
                task, status=status, verb=verb,
                description=candidate.description if candidate else None,
                extent=result.extent, consent=how,
                chooser=picker.last_view(),
                verified=result.verified, after_stop=after_stop,
                candidates_offered=result.candidates_offered,
                duration_ms=int((self._clock() - started) * 1000),
                reason=consent.fence_reason or result.reason)
        except Exception:  # noqa: BLE001 - the log never ends a task
            logger.debug("action log step failed", exc_info=True)

    async def _heartbeat(self, task: ComputerTask, notify) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_s)
            if task.status != "running":
                return
            await _quiet(notify, (
                f"Still running desktop task {task.task_id}: "
                f"{_n(task.steps, 'step')}, {_n(task.approvals, 'approval')} "
                f"so far. /computer stop to stop it."))

    # ── audit ───────────────────────────────────────────────────────────

    def _audit_binding(self, tool_name: str, binding: Binding, extent: Any) -> None:
        from prometheus.permissions.audit import AuditDecision

        audit = getattr(self.gate, "_audit", None)
        if audit is None:
            return
        try:
            audit.log(
                tool_name=tool_name, decision=AuditDecision.CONFIRM_APPROVED,
                trust_level=getattr(self.gate, "_mode_trust_level", lambda: 0)(),
                reason=(f"confirm_approved: binding {binding.binding_id} "
                        f"({extent.value})"),
                tool_input=None, user_id=binding.set_by.get("by"))
        except Exception:  # noqa: BLE001
            logger.debug("binding audit write failed", exc_info=True)

    def _audit_note(self, by: Approver, text: str) -> None:
        from prometheus.permissions.audit import AuditDecision

        audit = getattr(self.gate, "_audit", None)
        if audit is None:
            return
        try:
            audit.log(tool_name="computer_task", decision=AuditDecision.ALLOW,
                      trust_level=0, reason=text, tool_input=None,
                      user_id=by.label)
        except Exception:  # noqa: BLE001
            logger.debug("computer task audit write failed", exc_info=True)

    # ── status ──────────────────────────────────────────────────────────

    def status_text(self) -> str:
        snap = {}
        try:
            snap = self.integration.snapshot()
        except Exception:  # noqa: BLE001
            pass
        lines = [f"Computer use: on — driver {snap.get('state', 'unknown')}"
                 + (f" (cua-driver {snap['version']})" if snap.get("version") else "")]
        for check in snap.get("checks") or []:
            if check.get("state") != "ok":
                lines.append(f"  {check.get('name')}: {check.get('state')} — "
                             f"{check.get('detail')}")
        running = self.running()
        if running:
            for t in running:
                lines.append(f"Running: {t.task_id} in {t.app} — "
                             f"{_n(t.steps, 'step')}, {_n(t.approvals, 'approval')}")
        else:
            lines.append("No desktop task is running.")
        return "\n".join(lines)


class _StopAwareChooser:
    """Wraps the chooser: once a stop is requested, it answers ``abstain``
    (design §5.1.7 — the chooser wrapper is one of the stop checks)."""

    def __init__(self, chooser: Any, task: ComputerTask) -> None:
        self._chooser = chooser
        self._task = task
        self.name = getattr(chooser, "name", "rule")
        self._last: Any = None

    def choose(self, request):
        from prometheus.computer.types import CANDIDATE_ABSTAIN, Choice

        if self._task.stop_requested:
            return Choice(CANDIDATE_ABSTAIN, source=self.name)
        choice = self._chooser.choose(request)
        self._last = choice
        if self._task.stop_requested:
            return Choice(CANDIDATE_ABSTAIN, source=self.name)
        return choice

    def last_view(self) -> dict[str, Any]:
        """Who picked, how sure, and why — never the table it picked from."""
        choice = self._last
        return {
            "name": self.name,
            "confidence": getattr(choice, "confidence", None),
            "reason": getattr(choice, "reason", None),
        }


def end_message(task: ComputerTask) -> str:
    """The last milestone — counts only (W5), and only runner-authored words:
    never an element's label, never the typed text."""
    words = {
        "done": "finished",
        "abstained": "found nothing to do",
        "stopped": "stopped",
        "refused": "ended: a step was refused",
        "failed": "ended: it could not continue",
        "limit": "ended: it reached a limit",
    }.get(task.outcome or "", f"ended ({task.outcome})")
    tail = (" One action was in flight at the stop and may have landed."
            if task.in_flight_at_stop else "")
    return (f"Desktop task {task.task_id} {words} — "
            f"{_n(task.steps, 'step')}, {_n(task.approvals, 'approval')}.{tail}")


def _n(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _same(a: str, b: str, aliases: Mapping[str, Any]) -> bool:
    if normalise_term(a) == normalise_term(b):
        return True
    for name in aliases.get(a.strip().lower(), []) or []:
        if normalise_term(name) == normalise_term(b):
            return True
    return False


async def _quiet(notify, text: str) -> None:
    try:
        await notify(text)
    except Exception:  # noqa: BLE001 - a lost milestone never ends a task
        logger.warning("computer task milestone not delivered", exc_info=True)


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None
