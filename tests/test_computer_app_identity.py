"""Check 1b knows an app by every name the door resolved it by.

Task 36ed5742 was refused at check 1b: "the driver says this window belongs to
'gnome-text-editor', not 'Text Editor'". It was the same app. The door
resolves a phrase through ANY of an app's identifiers (its display name, its
bundle/desktop id, its launch-path basename) but kept only the display name,
and ``same_app`` compared the driver's report against that one name. Cua
names a window's app by its process (``gnome-text-editor``), and the
``.desktop`` entry calls it "Text Editor".

What this file pins:

* The door records every identifier of the app it resolved, plus the pid
  where known (``AppIdentity``).
* Check 1b passes when the driver's report equals ANY recorded identifier
  (case-insensitively, with the extent's own spelling rule), and requires the
  same pid when both pids are known. Anything else refuses with the existing
  message.
* The app pick covers the app under any of its names (Will, 2026-10-04,
  for the re-test's step 2). A binding carries the identity resolved when the
  person picked the app, so a "Text Editor" binding covers a click or a Tab
  the extent calls ``gnome-text-editor``. Nothing else in ``covers`` moves:
  target, site, delivery, payload, verbs and keys are as W2 set them.
* EXCEPT A SHARED LAUNCHER. ``covers`` has no pid, so a launcher basename
  many apps share (python3, flatpak, env, bash, java, electron…) is never a
  name a binding accepts. Check 1b keeps its pid match and still accepts it.
* The person still reads the display name: the sentence, ``as_dict`` and the
  Telegram text say "Text Editor", and no remembered grant is written.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.computer.chooser import RuleChooser, ScriptedChooser
from prometheus.computer.discovery import (
    AppIdentity, AppRecord, WindowRecord, is_shared_launcher,
)
from prometheus.computer.driver import FixtureDriver
from prometheus.computer.loop import ComputerUseLoop
from prometheus.computer.types import Element, Observation
from prometheus.permissions.checker import PermissionMode, SecurityGate
from prometheus.permissions.computer_extent import ComputerExtent
from prometheus.permissions.computer_schema import (
    DELIVERY_BACKGROUND, SITE_NONE, SITE_UNKNOWN,
)

PID = 4242
WID = 7

#: GNOME Text Editor as Cua's ``list_apps`` reports it on the box: the
#: ``.desktop`` display name, its desktop id, and the executable.
TEXT_EDITOR = AppRecord(
    pid=PID, name="Text Editor", running=True, active=True,
    bundle_id="org.gnome.TextEditor",
    launch_path="/usr/bin/gnome-text-editor")


def _obs(app: str, pid: int = PID, snapshot: str = "s1") -> Observation:
    return Observation(
        target="box", app=app, pid=pid, window_id=WID, snapshot_id=snapshot,
        elements=(Element(0, "tok-save", "push button", "Save"),))


def _step(obs, *, identity, asked_for="Text Editor"):
    prompted: list = []

    async def approver(tool_name, reason, arguments=None):
        prompted.append(reason)
        return True

    driver = FixtureDriver([obs, obs])
    loop = ComputerUseLoop(
        driver=driver, chooser=RuleChooser(prefer=("save",)),
        gate=SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None),
        approve=approver, skip_preconditions=True)
    result = asyncio.run(loop.step("save", "box", asked_for, obs.pid, WID,
                                   identity=identity))
    return result, driver, prompted


# ── WHAT THE DOOR RECORDS ───────────────────────────────────────────────────

def test_the_identity_records_every_identifier_and_the_pid():
    ident = AppIdentity.of(TEXT_EDITOR)
    assert ident.names == ("Text Editor", "org.gnome.TextEditor",
                           "gnome-text-editor")
    assert ident.pid == PID


def test_an_absent_identifier_or_pid_is_not_recorded():
    ident = AppIdentity.of(AppRecord(pid=0, name="gedit", bundle_id=None,
                                     launch_path="/usr/bin/gedit"))
    assert ident.names == ("gedit",), "one name, recorded once"
    assert ident.pid is None, "pid 0 is no pid"


# ── CHECK 1B ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("reported", [
    "gnome-text-editor",       # task 36ed5742: the executable's name
    "Text Editor",             # the display name
    "org.gnome.TextEditor",    # the desktop id
    "GNOME-Text-Editor",       # case-insensitive
])
def test_the_driver_naming_the_app_by_any_recorded_identifier_passes(reported):
    result, driver, _ = _step(_obs(reported),
                              identity=AppIdentity.of(TEXT_EDITOR))
    assert result.status != "blocked", result.reason
    assert result.ok and driver.dispatched


def test_the_same_name_in_another_process_is_refused():
    result, driver, prompted = _step(_obs("gnome-text-editor", pid=PID + 1),
                                     identity=AppIdentity.of(TEXT_EDITOR))
    assert result.status == "blocked"
    assert "belongs to 'gnome-text-editor'" in result.reason
    assert "refusing to act in an app nobody asked for" in result.reason
    assert str(PID + 1) in result.reason and str(PID) in result.reason
    assert driver.dispatched == [] and not prompted


def test_a_different_app_is_refused_with_the_existing_message():
    result, driver, prompted = _step(_obs("Firefox"),
                                     identity=AppIdentity.of(TEXT_EDITOR))
    assert result.status == "blocked"
    assert result.reason == (
        "the driver says this window belongs to 'Firefox', not 'Text Editor' "
        "— refusing to act in an app nobody asked for")
    assert driver.dispatched == [] and not prompted


def test_a_phrase_the_app_was_found_by_is_not_one_of_its_names():
    """An alias is the operator's word for the app, not the app's: a window
    the driver reports under the alias is not thereby the app."""
    result, driver, _ = _step(_obs("my editor"),
                              identity=AppIdentity.of(TEXT_EDITOR))
    assert result.status == "blocked"
    assert driver.dispatched == []


@pytest.mark.parametrize("recorded_pid, observed_pid", [
    (None, 0),       # no pid on either side
    (None, PID + 1),  # only the observation has one
    (PID, 0),        # only the door has one
], ids=["neither", "observed-only", "recorded-only"])
def test_without_both_pids_the_identifier_match_decides(recorded_pid,
                                                        observed_pid):
    ident = AppIdentity(names=AppIdentity.of(TEXT_EDITOR).names,
                        pid=recorded_pid)
    ok, driver, _ = _step(_obs("gnome-text-editor", pid=observed_pid),
                          identity=ident)
    assert ok.ok and driver.dispatched, ok.reason
    other, driver, _ = _step(_obs("Firefox", pid=observed_pid),
                             identity=ident)
    assert other.status == "blocked" and driver.dispatched == []


def test_a_step_given_no_identity_compares_the_app_it_was_asked_for():
    """A caller that never resolved through the door (the wrapped tools, the
    probes) keeps today's check: the one name it passed, no pid."""
    prompted: list = []

    async def approver(tool_name, reason, arguments=None):
        prompted.append(reason)
        return True

    for reported, ok in (("Text Editor", True), ("gnome-text-editor", False)):
        obs = _obs(reported)
        driver = FixtureDriver([obs, obs])
        loop = ComputerUseLoop(
            driver=driver, chooser=RuleChooser(prefer=("save",)),
            gate=SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None),
            approve=approver, skip_preconditions=True)
        result = asyncio.run(loop.step("save", "box", "Text Editor", PID, WID))
        assert result.ok is ok, (reported, result.reason)


# ── THE BINDING: THE PICK COVERS THE APP UNDER ANY OF ITS NAMES ─────────────

#: Launchers many apps share. Each is an identifier of the app that ran it,
#: so check 1b accepts it — but a binding never does.
SHARED = [
    "/usr/bin/python3", "/usr/bin/python3.12", "/usr/bin/python",
    "/usr/bin/flatpak", "/usr/bin/snap", "/usr/bin/env", "/bin/sh",
    "/bin/bash", "/usr/bin/java", "/usr/lib/electron/electron",
    "/usr/bin/electron28", "/usr/bin/node",
]


def _binding(app: AppRecord = TEXT_EDITOR, *, identity=True):
    from prometheus.computer.task import Binding

    return Binding(
        binding_id="b1", session_id="telegram:456", target="box",
        app=app.name, scope="task", set_by={"surface": "telegram"},
        created_at=0.0, expires_at=1e12,
        identity=AppIdentity.of(app) if identity else None)


def _extent(app: str, verb: str = "click", *, site=SITE_NONE,
            delivery=DELIVERY_BACKGROUND, payload=(), target="box"):
    return ComputerExtent(target=target, app=app, verb=verb,
                          delivery=delivery, payload_params=payload, site=site)


@pytest.mark.parametrize("reported", [
    "gnome-text-editor", "GNOME-Text-Editor", "org.gnome.TextEditor",
    "Text Editor",
])
def test_the_pick_covers_a_click_and_a_tab_under_any_of_the_apps_names(
        reported):
    b = _binding()
    assert b.covers(_extent(reported), {})
    assert b.covers(_extent(reported, "press_key"), {"key": "Tab"})
    assert b.covers(_extent(reported, "press_key"), {"key": "escape"})


@pytest.mark.parametrize("extent, arguments", [
    (_extent("Firefox"), {}),
    (_extent("firefox", "press_key"), {"key": "tab"}),
    (_extent("gnome-text-editor", "press_key"), {"key": "Return"}),
    (_extent("gnome-text-editor", "set_value", payload=("text",)),
     {"text": "hi"}),
    (_extent("gnome-text-editor", "type_text", payload=("text",)),
     {"text": "hi"}),
    (_extent("gnome-text-editor", "invoke_menu"), {}),
    (_extent("gnome-text-editor", site=SITE_UNKNOWN), {}),
    (_extent("gnome-text-editor", delivery="foreground"), {}),
    (_extent("gnome-text-editor", target="laptop"), {}),
], ids=["another-app", "another-app-tab", "return", "set-value", "type-text",
        "menu", "site-unknown", "foreground", "another-target"])
def test_everything_else_the_pick_never_covered_it_still_does_not(
        extent, arguments):
    assert not _binding().covers(extent, arguments)


@pytest.mark.parametrize("launcher", SHARED)
def test_a_shared_launcher_name_is_never_covered(launcher):
    """``covers`` has no pid, so two apps launched the same way would be one
    app to it. The launcher is still an identifier for check 1b, whose pid
    match tells them apart."""
    app = AppRecord(pid=PID, name="Mail Merge", running=True,
                    launch_path=launcher)
    name = launcher.rsplit("/", 1)[-1]
    assert is_shared_launcher(name)
    assert not _binding(app).covers(_extent(name), {})
    assert not _binding(app).covers(_extent(name, "press_key"), {"key": "tab"})
    assert _binding(app).covers(_extent("Mail Merge"), {}), (
        "the name the person picked is covered as it always was")
    assert AppIdentity.of(app).matches(name, PID), "check 1b moved"
    assert not AppIdentity.of(app).matches(name, PID + 1)


@pytest.mark.parametrize("name", [
    "gnome-text-editor", "gedit", "firefox", "org.gnome.TextEditor",
    "Text Editor", "pythonista", "bashtop", "snapshot", "javelin",
])
def test_an_apps_own_name_is_not_a_shared_launcher(name):
    assert not is_shared_launcher(name)


def test_a_binding_without_an_identity_covers_its_one_name():
    b = _binding(identity=False)
    assert b.covers(_extent("Text Editor"), {})
    assert not b.covers(_extent("gnome-text-editor"), {})


def test_the_person_still_reads_the_display_name():
    b = _binding()
    assert "in Text Editor on box" in b.sentence()
    assert "gnome-text-editor" not in b.sentence()
    d = b.as_dict()
    assert d["app"] == "Text Editor"
    assert set(d) == {"state", "session_id", "target", "app", "describes",
                      "scope", "covers", "set_by", "expires_at"}, (
        "as_dict grew a key — the identity is not for display")
    assert "gnome-text-editor" not in str(d)


# ── THROUGH THE DOOR: TASK 36ED5742'S SHAPE ─────────────────────────────────

@pytest.fixture(autouse=True)
def _linux_site_rule(monkeypatch):
    """As in test_computer_door.py: only a platform that can flag web content
    may answer site ``-``, so the Linux rule is applied on every runner."""
    import sys

    from prometheus.computer import candidates

    monkeypatch.setattr(candidates, "_PLATFORMS_THAT_FLAG_WEB",
                        ("linux", sys.platform))


class _Integration:
    enabled = True

    def __init__(self, driver) -> None:
        self._driver = driver

    async def probe(self, *, force: bool = False):
        return {"state": "ready"}

    def driver(self):
        return self._driver

    def local_target(self):
        return "box"

    @property
    def state(self):
        return "ready"


def _door(tmp_path, observed_app: str, *, ids=("click-0",),
          observed_pid: int = PID, app: AppRecord = TEXT_EDITOR):
    from prometheus.computer.approvals import ComputerApprovalChannel
    from prometheus.computer.door import PersonCheck
    from prometheus.computer.task import ComputerTaskRunner
    from prometheus.permissions.audit import AuditLogger

    observations = [
        Observation(
            target="box", app=observed_app, pid=observed_pid, window_id=WID,
            snapshot_id=f"s{i}",
            elements=(Element(0, "tok-save", "push button", "Save"),
                      Element(1, "tok-name", "text", "Name")),
            degraded=False, truncated=False, elements_complete=False,
            total_element_count=2, returned_element_count=2,
            web_content_seen=False)
        for i in range(10)
    ]
    driver = FixtureDriver(
        observations, apps=[app],
        windows=[WindowRecord(window_id=WID, pid=PID,
                              app_name=observed_app, z_index=1)])
    gate = SecurityGate(mode=PermissionMode.DEFAULT,
                        audit_logger=AuditLogger(tmp_path / "audit"))
    people = PersonCheck(device_store=None, telegram_user_ids={456})
    channel = ComputerApprovalChannel(security_gate=gate, people=people,
                                      telegram_adapter=None)
    runner = ComputerTaskRunner(
        integration=_Integration(driver), gate=gate, channel=channel,
        people=people, chooser_factory=lambda: ScriptedChooser(list(ids)),
        skip_preconditions=True, heartbeat_s=3600)
    return runner, channel, driver, gate


async def _run_task(runner, channel, *, answer: str | None,
                    app: str = "Text Editor", text: str | None = None):
    """Bind ``app``, start a task, and answer the first prompt if told to.
    Returns the finished task and the binding the pick made."""
    from prometheus.computer.task import ComputerTaskInput
    from prometheus.permissions.approver import Approver

    person = Approver("telegram", "456", "will")
    binding = await runner.bind("telegram:456", app, scope="task", by=person,
                                surface="telegram")
    task = await runner.start(
        ComputerTaskInput(goal="save it", app=app, text=text),
        session_id="telegram:456", surface="telegram", by=person)
    if answer is not None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        while not channel.pending and loop.time() < deadline:
            await asyncio.sleep(0.01)
        [rid] = list(channel.pending)
        if answer == "approve":
            assert await channel.approve(rid, by=person)
        else:
            assert await channel.deny(rid, by=person)
    return await runner.wait(task.task_id, timeout=10), binding


async def test_the_refused_task_reaches_its_window_and_the_pick_covers_the_click(
        tmp_path):
    runner, channel, driver, gate = _door(tmp_path, "gnome-text-editor")
    done, binding = await _run_task(runner, channel, answer=None)
    assert "belongs to" not in done.reason, done.reason
    assert [v for v, _ in driver.dispatched] == ["click"]
    assert done.approvals == 0, "the app pick covers a click in that app"
    assert binding.app == "Text Editor"
    assert gate.list_grants() == [], "a covered click wrote a remembered grant"


async def test_the_pick_covers_a_tab_the_driver_reports_by_another_name(
        tmp_path):
    """The re-test's step 2."""
    runner, channel, driver, _ = _door(tmp_path, "gnome-text-editor",
                                       ids=("key-tab",))
    done, _ = await _run_task(runner, channel, answer=None)
    assert [(v, a.get("key")) for v, a in driver.dispatched] == [
        ("press_key", "tab")]
    assert done.approvals == 0


async def test_return_still_asks(tmp_path):
    runner, channel, driver, _ = _door(tmp_path, "gnome-text-editor",
                                       ids=("key-return",))
    done, _ = await _run_task(runner, channel, answer="deny")
    assert done.approvals == 1
    assert driver.dispatched == []


async def test_typing_still_asks(tmp_path):
    runner, channel, driver, _ = _door(tmp_path, "gnome-text-editor",
                                       ids=("set-1",))
    done, _ = await _run_task(runner, channel, answer="deny", text="hello")
    assert done.approvals == 1
    assert driver.dispatched == []


async def test_an_app_reported_only_by_a_shared_launcher_asks(tmp_path):
    """Check 1b lets it through (same pid); the pick does not cover it."""
    app = AppRecord(pid=PID, name="Mail Merge", running=True, active=True,
                    launch_path="/usr/bin/python3")
    runner, channel, driver, _ = _door(tmp_path, "python3", app=app)
    done, binding = await _run_task(runner, channel, answer="approve",
                                    app="Mail Merge")
    assert "belongs to" not in done.reason, done.reason
    assert done.approvals == 1, "a shared launcher name was covered"
    assert [v for v, _ in driver.dispatched] == ["click"]


async def test_the_door_refuses_the_same_name_in_another_process(tmp_path):
    runner, channel, driver, _ = _door(tmp_path, "gnome-text-editor",
                                       observed_pid=PID + 1)
    done, _ = await _run_task(runner, channel, answer=None)
    assert "refusing to act in an app nobody asked for" in done.reason
    assert driver.dispatched == []


async def test_the_door_refuses_another_app(tmp_path):
    runner, channel, driver, _ = _door(tmp_path, "Firefox")
    done, _ = await _run_task(runner, channel, answer=None)
    assert "belongs to 'Firefox', not 'Text Editor'" in done.reason
    assert driver.dispatched == []
