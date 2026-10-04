"""Discovery: "my editor" becomes ONE window or a question (computer-use v1.1, PR 4).

D8: ``list_windows`` and ``list_apps`` exist in the SDK and nothing called
them, so a caller had to already know a pid and a window id — and the only
way to get them was out of band. The door has to turn a word into a window.

D19: ``Observation.app`` was ``out.app_name or app`` — when the driver
reported no app name, the CONSENT TERM was whatever the caller said. A
binding re-derived from the same claim would cover an unidentified window.
Now the app term comes from the driver or not at all, and the loop refuses a
window whose driver-reported app is missing or is not the app it was asked
to act in.

Rules (design §5.1.3): an operator alias first, then a case-insensitive match
on the app's name, bundle id, or launch-path basename. ONE running match
with an on-screen window is proposed; zero or several is a QUESTION listing
names only — never a pid, never a window id. Nothing is ever launched. The
window is the app's frontmost on-screen window, re-resolved before every
step; a vanished window ends the task.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from prometheus.computer import cua
from prometheus.computer.chooser import RuleChooser
from prometheus.computer.discovery import (
    AppRecord, WindowRecord, frontmost_window, resolve_app, resolve_window,
)
from prometheus.computer.driver import FixtureDriver
from prometheus.computer.loop import ComputerUseLoop
from prometheus.computer.types import Element, Observation
from prometheus.permissions.checker import PermissionMode, SecurityGate

APPS = [
    AppRecord(pid=10, name="gnome-text-editor", running=True, active=True,
              bundle_id=None, launch_path="/usr/bin/gnome-text-editor"),
    AppRecord(pid=20, name="Firefox", running=True, active=False,
              bundle_id="org.mozilla.firefox", launch_path="/usr/lib/firefox/firefox"),
    AppRecord(pid=30, name="gedit", running=True, active=False,
              bundle_id=None, launch_path="/usr/bin/gedit"),
    AppRecord(pid=40, name="gedit", running=True, active=False,
              bundle_id=None, launch_path="/usr/bin/gedit"),
    AppRecord(pid=50, name="Calculator", running=False, active=False,
              bundle_id=None, launch_path="/usr/bin/gnome-calculator"),
]
WINDOWS = [
    WindowRecord(window_id=101, pid=10, app_name="gnome-text-editor",
                 title="notes.txt", is_on_screen=True, z_index=3, minimized=False),
    WindowRecord(window_id=102, pid=10, app_name="gnome-text-editor",
                 title="todo.txt", is_on_screen=True, z_index=7, minimized=False),
    WindowRecord(window_id=103, pid=10, app_name="gnome-text-editor",
                 title="hidden", is_on_screen=True, z_index=9, minimized=True),
    WindowRecord(window_id=201, pid=20, app_name="Firefox", title="Bank",
                 is_on_screen=True, z_index=5, minimized=False),
    WindowRecord(window_id=301, pid=30, app_name="gedit", title="a",
                 is_on_screen=True, z_index=1, minimized=False),
    WindowRecord(window_id=401, pid=40, app_name="gedit", title="b",
                 is_on_screen=True, z_index=2, minimized=False),
]


# ── RESOLVING A PHRASE ──────────────────────────────────────────────────────

def test_an_alias_resolves_to_one_window():
    r = resolve_app("my editor", APPS, WINDOWS,
                    aliases={"my editor": ["gnome-text-editor"]})
    assert r.status == "match"
    assert r.app.name == "gnome-text-editor"
    assert r.window.window_id == 102, "not the frontmost on-screen window"


@pytest.mark.parametrize("phrase", [
    "firefox", "FIREFOX", "org.mozilla.firefox", "firefox ",
])
def test_name_bundle_id_and_launch_path_match_case_insensitively(phrase):
    r = resolve_app(phrase, APPS, WINDOWS)
    assert r.status == "match" and r.app.pid == 20


def test_a_launch_path_basename_matches():
    r = resolve_app("gnome-text-editor", APPS, WINDOWS)
    assert r.status == "match" and r.app.pid == 10


def test_several_matches_ask_listing_names_only():
    r = resolve_app("gedit", APPS, WINDOWS)
    assert r.status == "ask"
    assert r.app is None and r.window is None
    assert r.options == ["gedit", "gedit"]
    text = r.question + " ".join(r.options)
    for leak in ("30", "40", "301", "401"):
        assert leak not in text, f"a pid or window id reached the question: {text!r}"


def test_no_match_asks_and_lists_what_is_running():
    r = resolve_app("photoshop", APPS, WINDOWS)
    assert r.status == "ask"
    assert set(r.options) == {"gnome-text-editor", "Firefox", "gedit"}
    assert "Calculator" not in r.options, "an app that is not running was offered"


def test_a_running_app_with_no_on_screen_window_asks():
    windows = [w for w in WINDOWS if w.pid != 20]
    r = resolve_app("firefox", APPS, windows)
    assert r.status == "ask"
    assert "on-screen" in r.question


def test_an_empty_on_screen_list_asks():
    r = resolve_app("firefox", APPS, [])
    assert r.status == "ask"
    assert r.options == []


def test_a_not_running_app_is_never_proposed_and_never_launched():
    driver = FixtureDriver([_obs()], apps=APPS, windows=WINDOWS)
    r = resolve_app("calculator", driver.list_apps(), driver.list_windows())
    assert r.status == "ask"
    assert not hasattr(driver, "launch_app")
    assert not hasattr(cua.CuaDriverAdapter, "launch_app")
    assert "launch_app" not in cua._BUILDERS


# ── THE WINDOW ──────────────────────────────────────────────────────────────

def test_the_frontmost_on_screen_unminimised_window_wins():
    assert frontmost_window(WINDOWS, pid=10).window_id == 102


def test_a_window_is_re_resolved_and_a_vanished_one_is_none():
    driver = FixtureDriver([_obs()], apps=APPS, windows=WINDOWS)
    assert resolve_window(driver, pid=10).window_id == 102
    driver.windows = [w for w in WINDOWS if w.window_id != 102]
    assert resolve_window(driver, pid=10).window_id == 101
    driver.windows = [w for w in WINDOWS if w.pid != 10]
    assert resolve_window(driver, pid=10) is None


def test_the_fixture_lists_only_on_screen_windows_when_asked():
    off = WindowRecord(window_id=999, pid=10, app_name="gnome-text-editor",
                       title="x", is_on_screen=False, z_index=99,
                       minimized=False)
    driver = FixtureDriver([_obs()], apps=APPS, windows=[*WINDOWS, off])
    assert 999 not in {w.window_id for w in driver.list_windows(pid=10)}
    assert 999 in {w.window_id for w in
                   driver.list_windows(pid=10, on_screen_only=False)}


# ── THE ADAPTER ─────────────────────────────────────────────────────────────

class _Sdk:
    ListAppsInput = staticmethod(lambda **kw: SimpleNamespace(kind="apps", **kw))
    ListWindowsInput = staticmethod(
        lambda **kw: SimpleNamespace(kind="windows", **kw))
    GetWindowStateInput = staticmethod(lambda **kw: SimpleNamespace(**kw))


class _Driver:
    def __init__(self, app_name="gedit"):
        self.calls: list = []
        self.app_name = app_name

    def is_available(self):
        return True

    async def list_apps(self, inp):
        self.calls.append(("list_apps", inp))
        return SimpleNamespace(apps=[SimpleNamespace(
            pid=30, name="gedit", running=True, active=True, bundle_id=None,
            launch_path="/usr/bin/gedit", kind=None, last_used=None)])

    async def list_windows(self, inp):
        self.calls.append(("list_windows", inp))
        return SimpleNamespace(current_space_id=None, windows=[SimpleNamespace(
            window_id=301, pid=30, app_name="gedit", title="a", bounds=None,
            is_on_screen=True, z_index=4, layer=0, minimized=False,
            current_space_id=None, on_current_space=None, space_ids=None)])

    async def get_window_state(self, inp):
        el = SimpleNamespace(element_index=0, element_token="t", role="push button",
                             label="Save", value=None, actions=None)
        return SimpleNamespace(snapshot_id="s1", app_name=self.app_name,
                               elements=[el])

    async def shutdown(self):
        pass


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(cua, "_require_sdk", lambda: _Sdk)

    def make(app_name="gedit"):
        fake = _Driver(app_name)
        monkeypatch.setattr(_Sdk, "CuaDriver",
                            SimpleNamespace(create=lambda: fake), raising=False)
        a = cua.CuaDriverAdapter(target="box")
        a.start()
        return a, fake

    made = []
    yield lambda **kw: made.append(make(**kw)) or made[-1]
    for a, _ in made:
        a.shutdown()


def test_the_adapter_lists_apps_and_on_screen_windows(adapter):
    a, fake = adapter()
    apps = a.list_apps()
    windows = a.list_windows(pid=30)
    assert apps == [AppRecord(pid=30, name="gedit", running=True, active=True,
                              bundle_id=None, launch_path="/usr/bin/gedit")]
    assert windows == [WindowRecord(window_id=301, pid=30, app_name="gedit",
                                    title="a", is_on_screen=True, z_index=4,
                                    minimized=False)]
    assert fake.calls[1][1].pid == 30
    assert fake.calls[1][1].on_screen_only is True


def test_the_driver_reports_the_app_or_there_is_no_app(adapter):
    """D19. The caller's claim is never the consent term."""
    a, _ = adapter(app_name=None)
    assert a.observe("box", "my-claim", 30, 301).app == ""


def test_the_driver_app_name_still_wins(adapter):
    a, _ = adapter(app_name="gedit")
    assert a.observe("box", "my-claim", 30, 301).app == "gedit"


def test_the_discovery_inputs_build_with_the_pinned_sdk():
    real = pytest.importorskip("cua_driver")
    assert type(cua._list_apps_input(real)) is real.ListAppsInput
    inp = cua._list_windows_input(real, pid=30, on_screen_only=True)
    assert type(inp) is real.ListWindowsInput
    assert inp.pid == 30 and inp.on_screen_only is True


def test_real_sdk_records_translate():
    real = pytest.importorskip("cua_driver")
    app = real.AppInfo(pid=1, name="gedit", running=True, active=False,
                       bundle_id=None, launch_path="/usr/bin/gedit", kind=None,
                       last_used=None)
    win = real.WindowInfo(
        window_id=7, pid=1, app_name="gedit", title="t",
        bounds=real.WindowBounds(x=0.0, y=0.0, width=1.0, height=1.0),
        is_on_screen=True, z_index=2, layer=0, minimized=False,
        current_space_id=None, on_current_space=None, space_ids=None)
    assert cua._app_record(app) == AppRecord(
        pid=1, name="gedit", running=True, active=False, bundle_id=None,
        launch_path="/usr/bin/gedit")
    assert cua._window_record(win) == WindowRecord(
        window_id=7, pid=1, app_name="gedit", title="t", is_on_screen=True,
        z_index=2, minimized=False)


# ── THE LOOP: A WINDOW WITH NO DRIVER-REPORTED APP NEVER ACTS ──────────────

def _obs(app="gedit") -> Observation:
    return Observation(
        target="box", app=app, pid=30, window_id=301, snapshot_id="s1",
        elements=(Element(0, "tok-save", "push button", "Save"),))


def _step(obs, *, asked_for="gedit"):
    prompted: list = []

    async def approver(tool_name, reason, arguments=None):
        prompted.append(reason)
        return True

    driver = FixtureDriver([obs, obs])
    loop = ComputerUseLoop(
        driver=driver, chooser=RuleChooser(prefer=("save",)),
        gate=SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None),
        approve=approver, skip_preconditions=True)
    result = asyncio.run(loop.step("save", "box", asked_for, 30, 301))
    return result, driver, prompted


def test_a_window_with_no_reported_app_is_blocked():
    result, driver, prompted = _step(_obs(app=""))
    assert result.status == "blocked"
    assert "reported no application" in result.reason
    assert driver.dispatched == [] and not prompted


def test_a_window_of_another_app_is_blocked():
    result, driver, prompted = _step(_obs(app="Firefox"))
    assert result.status == "blocked"
    assert "Firefox" in result.reason and "gedit" in result.reason
    assert driver.dispatched == [] and not prompted


def test_the_app_comparison_uses_the_extents_own_spelling():
    result, driver, _ = _step(_obs(app="GEdit"), asked_for="gedit")
    assert result.ok and driver.dispatched
