"""PR 6b — the driver leg's invariants, as far as they can be tested here.

``CuaDriverAdapter.capture`` cannot be exercised without a display (see
``cua.py``'s docstring) — that is the on-box check. What IS coverable, and is
the point of this file:

* the SDK→``WindowCapture`` translation is what the real adapter's shape
  promises, using the pinned 0.28.2 ``WindowStateOutput`` fields (the
  real-SDK half of ``test_cua_adapter.py`` builds the input the same way);
* ``FixtureDriver.capture`` enforces the REAL invariant — a capture mints a
  new snapshot and invalidates the tokens the loop holds. A fixture that let
  the old snapshot stand would let a test pass on driver behaviour that does
  not exist, which is the same lie the observe/act staleness rule prevents.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from prometheus.computer.cua import _window_state_input
from prometheus.computer.driver import FixtureDriver, StaleSnapshot
from prometheus.computer.types import Element, Observation, WindowCapture

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"p" * 32).decode()


def _obs(snapshot="s1") -> Observation:
    return Observation(
        target="box", app="scratchapp", pid=1, window_id=2,
        snapshot_id=snapshot,
        elements=(Element(0, f"tok-send-{snapshot}", "push button", "Send"),),
    )


def _cap(app_name="scratchapp", **over) -> WindowCapture:
    fields = dict(target="box", app="scratchapp", pid=1, window_id=2,
                  app_name=app_name, roles=("push button",),
                  image_mime="image/png", image_base64=PNG,
                  image_width=480, image_height=300)
    fields.update(over)
    return WindowCapture(**fields)


# ── the input builder still asks for no screenshot on the observe path ─────

class _PinSDK:
    """Records the GetWindowStateInput kwargs, like the real-SDK test does."""

    def __init__(self):
        self.last = None

    def GetWindowStateInput(self, **kw):  # noqa: N802 - mirrors the SDK name
        self.last = kw
        return SimpleNamespace(**kw)


def test_observe_input_still_asks_for_no_screenshot():
    """The pin that already exists must hold: turning screenshots on for
    observe would put a frame in the candidate-table path, where the capture
    ruling says none belongs."""
    sdk = _PinSDK()
    _window_state_input(sdk, pid=1, window_id=2, session=None)
    assert sdk.last["include_screenshot"] is False
    assert sdk.last["max_dimension"] is None


def test_capture_input_turns_the_screenshot_on_and_sizes_it():
    """Same builder, the 6b path: screenshot on, scaled to the thumbnail."""
    sdk = _PinSDK()
    _window_state_input(sdk, pid=1, window_id=2, session="sess",
                        include_screenshot=True, max_dimension=480)
    assert sdk.last["include_screenshot"] is True
    assert sdk.last["max_dimension"] == 480
    # The tree still comes with it — one call, so the password check and the
    # pixels describe the same moment.
    assert sdk.last["include_accessibility_tree"] is True
    # And it never writes the screenshot to a file we do not own.
    assert sdk.last["screenshot_out_file"] is None


# ── FixtureDriver.capture enforces the real invalidation ───────────────────

def test_a_capture_is_recorded_with_its_arguments():
    """Assert on what reached the driver, not a return string."""
    driver = FixtureDriver([_obs()], captures=[_cap()])
    out = driver.capture("box", "scratchapp", 1, 2, max_dimension=480)
    assert out.app_name == "scratchapp"
    assert driver.capture_calls == [{"target": "box", "app": "scratchapp",
                                     "pid": 1, "window_id": 2,
                                     "max_dimension": 480}]


def test_a_capture_invalidates_the_snapshot_the_loop_holds():
    """THE invariant. A capture mints a new snapshot, so an action built from
    the old one must be refused — not silently dispatched against a screen
    that has since been re-observed by the capture itself."""
    driver = FixtureDriver([_obs("s1")], captures=[_cap()])
    driver.observe("box", "scratchapp", 1, 2)          # current = s1
    driver.capture("box", "scratchapp", 1, 2, max_dimension=480)
    with pytest.raises(StaleSnapshot):
        driver.act("click", {"snapshot_id": "s1", "element_token": "tok-send-s1"})


def test_an_unscripted_capture_is_honest_and_skips_as_window_changed():
    """No scripted capture → app_name None, so the identity gate skips it as
    window_changed rather than a test silently getting a picture it never set
    up. A fabricated success here would hide a missing fixture."""
    driver = FixtureDriver([_obs()])
    out = driver.capture("box", "scratchapp", 1, 2, max_dimension=480)
    assert out.app_name is None
    assert out.image_base64 is None


def test_scripted_captures_are_consumed_in_order_and_held_at_the_last():
    """Mirrors observations: each capture returns the next scripted frame, and
    the last one repeats, so a task with more steps than fixtures still has a
    defined answer."""
    driver = FixtureDriver(
        [_obs()], captures=[_cap(app_name="a"), _cap(app_name="b")])
    assert driver.capture("box", "a", 1, 2, max_dimension=1).app_name == "a"
    assert driver.capture("box", "b", 1, 2, max_dimension=1).app_name == "b"
    assert driver.capture("box", "b", 1, 2, max_dimension=1).app_name == "b"
