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
* CONSENT IS NOT WIDENED. The extent's app term is still the driver's name,
  and a binding still covers only the app it names — so a "Text Editor"
  binding does not cover a click the extent calls ``gnome-text-editor``; that
  click asks. W2 (Will, 2026-10-04) is unchanged.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.computer.chooser import RuleChooser, ScriptedChooser
from prometheus.computer.discovery import AppIdentity, AppRecord, WindowRecord
from prometheus.computer.driver import FixtureDriver
from prometheus.computer.loop import ComputerUseLoop
from prometheus.computer.types import Element, Observation
from prometheus.permissions.checker import PermissionMode, SecurityGate

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


def _door(tmp_path, observed_app: str, observed_pid: int = PID):
    from prometheus.computer.approvals import ComputerApprovalChannel
    from prometheus.computer.door import PersonCheck
    from prometheus.computer.task import ComputerTaskRunner
    from prometheus.permissions.audit import AuditLogger

    observations = [
        Observation(
            target="box", app=observed_app, pid=observed_pid, window_id=WID,
            snapshot_id=f"s{i}",
            elements=(Element(0, "tok-save", "push button", "Save"),),
            degraded=False, truncated=False, elements_complete=False,
            total_element_count=1, returned_element_count=1,
            web_content_seen=False)
        for i in range(10)
    ]
    driver = FixtureDriver(
        observations, apps=[TEXT_EDITOR],
        windows=[WindowRecord(window_id=WID, pid=PID,
                              app_name="gnome-text-editor", z_index=1)])
    gate = SecurityGate(mode=PermissionMode.DEFAULT,
                        audit_logger=AuditLogger(tmp_path / "audit"))
    people = PersonCheck(device_store=None, telegram_user_ids={456})
    channel = ComputerApprovalChannel(security_gate=gate, people=people,
                                      telegram_adapter=None)
    runner = ComputerTaskRunner(
        integration=_Integration(driver), gate=gate, channel=channel,
        people=people, chooser_factory=lambda: ScriptedChooser(["click-0"]),
        skip_preconditions=True, heartbeat_s=3600)
    return runner, channel, driver


async def _run_task(runner, channel, *, answer: str | None):
    from prometheus.computer.task import ComputerTaskInput
    from prometheus.permissions.approver import Approver

    person = Approver("telegram", "456", "will")
    await runner.bind("telegram:456", "Text Editor", scope="task", by=person,
                      surface="telegram")
    task = await runner.start(
        ComputerTaskInput(goal="save it", app="Text Editor"),
        session_id="telegram:456", surface="telegram", by=person)
    prompts = 0
    if answer is not None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        while not channel.pending and loop.time() < deadline:
            await asyncio.sleep(0.01)
        for rid in list(channel.pending):
            prompts += 1
            if answer == "approve":
                assert await channel.approve(rid, by=person)
            else:
                assert await channel.deny(rid, by=person)
    return await runner.wait(task.task_id, timeout=10), prompts


async def test_the_task_that_was_refused_now_reaches_its_window(tmp_path):
    runner, channel, driver = _door(tmp_path, "gnome-text-editor")
    done, prompts = await _run_task(runner, channel, answer="approve")
    assert "belongs to" not in done.reason, done.reason
    assert driver.dispatched and driver.dispatched[0][0] == "click"
    # ⚠ CONSENT IS NOT WIDENED. The binding names "Text Editor"; the extent
    # names the driver's "gnome-text-editor". The click is NOT covered by the
    # pick — it asks. Letting a binding cover the driver's name for its app
    # would widen what the person's pick consents to, and that is Will's call.
    assert prompts == 1 and done.approvals == 1


async def test_the_door_refuses_the_same_name_in_another_process(tmp_path):
    runner, channel, driver = _door(tmp_path, "gnome-text-editor",
                                    observed_pid=PID + 1)
    done, _ = await _run_task(runner, channel, answer=None)
    assert "refusing to act in an app nobody asked for" in done.reason
    assert driver.dispatched == []


async def test_the_door_refuses_another_app(tmp_path):
    runner, channel, driver = _door(tmp_path, "Firefox")
    done, _ = await _run_task(runner, channel, answer=None)
    assert "belongs to 'Firefox', not 'Text Editor'" in done.reason
    assert driver.dispatched == []
