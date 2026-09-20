"""The Cua adapter's TRANSLATION — which is the only part CI can cover.

⚠ READ THIS BEFORE TRUSTING A GREEN RUN ON THIS FILE.

The adapter's real job needs a display, an accessibility bus and a live
desktop. CI has none and cannot acquire them, so **the on-box outcome check
is the only evidence the driver works** — the same shape as the MCP stdio
transport, whose only proof is boot logs, and which is how ``mcp>=1.0``
resolved to 2.x with a green suite.

What IS covered here: that a Prometheus action dict becomes the right SDK
call with the right arguments, and that an SDK result becomes the right
verdict. Worth having. Not the same claim.

These tests run WITHOUT ``cua-driver`` installed — the extra is deliberately
absent from CI — by injecting a fake SDK with the same shapes. That is a
translation test by construction: it can only ever prove the adapter speaks
the shape it believes in, never that the shape is right. The shape was
confirmed by introspecting the real 0.28.2 package and by the on-box run.
"""

from __future__ import annotations

import enum
from types import SimpleNamespace

import pytest

from prometheus.computer import cua
from prometheus.computer.driver import DriverUnavailable, StaleSnapshot


class _Effect(enum.Enum):
    CONFIRMED = "CONFIRMED"
    PARTIAL = "PARTIAL"
    UNVERIFIABLE = "UNVERIFIABLE"
    SUSPECTED_NOOP = "SUSPECTED_NOOP"
    REFUSED = "REFUSED"


class _Delivery(enum.Enum):
    BACKGROUND = "BACKGROUND"
    FOREGROUND = "FOREGROUND"


class _Button(enum.Enum):
    LEFT = "LEFT"


class _ScrollDir(enum.Enum):
    UP = "UP"
    DOWN = "DOWN"


def _rec(**kw):
    return SimpleNamespace(**kw)


class _FakeSdk:
    """Mirrors the 0.28.2 shapes this adapter uses. Records what it is given."""

    InputDeliveryMode = _Delivery
    ClickButton = _Button
    ScrollDirection = _ScrollDir

    class ActionTarget:
        @staticmethod
        def WINDOW(pid, window_id):
            return ("WINDOW", pid, window_id)

    class ClickPosition:
        @staticmethod
        def ELEMENT(element_token):
            return ("ELEMENT", element_token)

    GetWindowStateInput = staticmethod(lambda **kw: _rec(kind="gws", **kw))
    ClickInput = staticmethod(lambda **kw: _rec(kind="click", **kw))
    PressKeyInput = staticmethod(lambda **kw: _rec(kind="press_key", **kw))
    ScrollInput = staticmethod(lambda **kw: _rec(kind="scroll", **kw))
    TypeTextInput = staticmethod(lambda **kw: _rec(kind="type_text", **kw))
    InvokeMenuInput = staticmethod(lambda **kw: _rec(kind="invoke_menu", **kw))


class _FakeDriver:
    """A CuaDriver stand-in. Async, like the real one."""

    def __init__(self, window_state=None, effect=_Effect.CONFIRMED):
        self.window_state = window_state
        self.effect = effect
        self.calls: list[tuple[str, object]] = []
        self._available = True

    def is_available(self):
        return self._available

    async def get_window_state(self, inp):
        self.calls.append(("get_window_state", inp))
        return self.window_state

    async def _action(self, name, inp):
        self.calls.append((name, inp))
        if isinstance(self.effect, Exception):
            raise self.effect
        return _rec(effect=self.effect)

    async def click(self, inp):
        return await self._action("click", inp)

    async def press_key(self, inp):
        return await self._action("press_key", inp)

    async def scroll(self, inp):
        return await self._action("scroll", inp)

    async def type_text(self, inp):
        return await self._action("type_text", inp)

    async def invoke_menu(self, inp):
        return await self._action("invoke_menu", inp)

    async def shutdown(self):
        self.calls.append(("shutdown", None))


@pytest.fixture
def adapter(monkeypatch):
    """An adapter wired to the fake SDK, with its real async seam running."""
    monkeypatch.setattr(cua, "_require_sdk", lambda: _FakeSdk)

    def make(window_state=None, effect=_Effect.CONFIRMED, target="mini"):
        a = cua.CuaDriverAdapter(target=target)
        fake = _FakeDriver(window_state=window_state, effect=effect)
        # start() spins the real loop thread; substitute the driver only.
        monkeypatch.setattr(_FakeSdk, "CuaDriver",
                            SimpleNamespace(create=lambda: fake), raising=False)
        a.start()
        return a, fake

    return make


def _window_state(snapshot_id="s1", elements=(), app_name="Scratch"):
    return _rec(pid=1, window_id=2, snapshot_id=snapshot_id,
                app_name=app_name, window_title="t",
                elements=list(elements), element_count=len(elements))


def _el(idx=0, token="tok", role="push button", label="Increment"):
    return _rec(element_index=idx, element_token=token, role=role,
                label=label, value=None, actions=None, editable=False,
                depth=0)


def _args(**over):
    a = {"target": "mini", "app": "scratch", "pid": 1, "window_id": 2,
         "snapshot_id": "s1", "delivery_mode": "background",
         "element_token": "tok"}
    a.update(over)
    return a


# ── SUSPECTED_NOOP IS NOT SUCCESS ───────────────────────────────────────────

def test_suspected_noop_raises_rather_than_returning_success(adapter):
    """THE ONE THAT MATTERS.

    Cua tells us the action appears to have done nothing. Returning success
    because the call did not raise would be trusting the success message at
    the one place this whole subsystem was built to distrust it.
    """
    a, fake = adapter(_window_state(), effect=_Effect.SUSPECTED_NOOP)
    a.observe("mini", "scratch", 1, 2)
    with pytest.raises(DriverUnavailable) as exc:
        a.act("click", _args())
    assert "SUSPECTED_NOOP" in str(exc.value)
    assert "no evidence it took effect" in str(exc.value)


def test_refused_raises(adapter):
    a, _ = adapter(_window_state(), effect=_Effect.REFUSED)
    a.observe("mini", "scratch", 1, 2)
    with pytest.raises(DriverUnavailable):
        a.act("click", _args())


@pytest.mark.parametrize("effect,landed,confirmed", [
    (_Effect.CONFIRMED, True, True),
    (_Effect.PARTIAL, True, False),
    (_Effect.UNVERIFIABLE, False, False),
])
def test_the_effect_vocabulary_is_carried_not_collapsed(
        adapter, effect, landed, confirmed):
    a, _ = adapter(_window_state(), effect=effect)
    a.observe("mini", "scratch", 1, 2)
    out = a.act("click", _args())
    assert out["effect"] == effect.name
    assert out["landed"] is landed
    assert out["confirmed_by_driver"] is confirmed


def test_the_result_has_no_key_called_ok(adapter):
    """Observed live: a click that demonstrably landed came back
    UNVERIFIABLE, so `status: executed` sat beside `ok: False` in one
    payload. Two fields named ok answering different questions."""
    a, _ = adapter(_window_state(), effect=_Effect.UNVERIFIABLE)
    a.observe("mini", "scratch", 1, 2)
    assert "ok" not in a.act("click", _args())


# ── THE TRANSLATION ─────────────────────────────────────────────────────────

def test_a_click_becomes_an_element_targeted_background_click(adapter):
    a, fake = adapter(_window_state(elements=[_el()]))
    a.observe("mini", "scratch", 1, 2)
    a.act("click", _args())
    name, inp = fake.calls[-1]
    assert name == "click"
    assert inp.target == ("WINDOW", 1, 2)
    assert inp.position == ("ELEMENT", "tok")
    assert inp.delivery_mode is _Delivery.BACKGROUND
    assert inp.button is _Button.LEFT
    assert inp.count == 1


def test_observation_never_requests_a_screenshot(adapter):
    """A frame we do not need is a frame that could end up somewhere it
    should not — see the capture ruling."""
    a, fake = adapter(_window_state(elements=[_el()]))
    a.observe("mini", "scratch", 1, 2)
    _, inp = fake.calls[0]
    assert inp.include_screenshot is False
    assert inp.include_accessibility_tree is True
    assert inp.screenshot_out_file is None


def test_elements_without_a_token_are_dropped(adapter):
    """An element with no token cannot be addressed, so offering it would
    build a candidate that can only fail."""
    a, _ = adapter(_window_state(elements=[
        _el(0, "tok"), _rec(element_index=1, element_token=None, role="x",
                            label="", value=None, actions=None,
                            editable=False, depth=0)]))
    obs = a.observe("mini", "scratch", 1, 2)
    assert len(obs.elements) == 1


def test_no_snapshot_id_is_UNUSABLE_not_empty(adapter):
    """Empty-because-unusable and empty-because-idle must not look alike."""
    a, _ = adapter(_window_state(snapshot_id=None, elements=[_el()]))
    obs = a.observe("mini", "scratch", 1, 2)
    assert obs.unusable_reason
    assert "snapshot id" in obs.unusable_reason
    assert obs.elements == ()


def test_the_driver_app_name_wins_over_the_callers(adapter):
    """The extent's app term comes from the driver's view, not the caller's.

    Worth pinning because the consent key is derived from it: an operator's
    grant is keyed on what the DRIVER calls the app.
    """
    a, _ = adapter(_window_state(app_name="Scratch_window.py"))
    assert a.observe("mini", "whatever", 1, 2).app == "Scratch_window.py"


# ── REFUSALS ────────────────────────────────────────────────────────────────

def test_an_action_for_another_machine_is_refused(adapter):
    a, fake = adapter(_window_state())
    a.observe("mini", "scratch", 1, 2)
    with pytest.raises(DriverUnavailable) as exc:
        a.act("click", _args(target="laptop"))
    assert "bound to target" in str(exc.value)
    assert not [c for c in fake.calls if c[0] == "click"]


def test_a_stale_snapshot_is_refused_BEFORE_the_driver(adapter):
    """The third independent refusal, so the guarantee does not depend on
    the driver's error text staying the same across versions."""
    a, fake = adapter(_window_state(snapshot_id="s2"))
    a.observe("mini", "scratch", 1, 2)
    with pytest.raises(StaleSnapshot):
        a.act("click", _args(snapshot_id="s1"))
    assert not [c for c in fake.calls if c[0] == "click"]


def test_acting_with_NO_observation_on_record_is_refused(adapter):
    """A missing snapshot record is "cannot determine", not "fresh".

    The guard read ``if snapshot and current and snapshot != current``, so
    when ``current`` was None it did not fire AT ALL and the call went to the
    driver. Reachable whenever the acting adapter is not the one that
    observed: a fresh adapter, a restart, a second adapter on the same
    target. Guard 3 (the driver's own error text) would still be there, but
    guard 2 exists precisely so the guarantee does NOT depend on that text
    staying stable across versions -- and in this case it was absent.

    The loop always observes and acts through one adapter instance
    (``ComputerUseLoop._driver`` is assigned once and used for observe, act
    and the post-action verify; ``build_computer_tools`` binds one driver to
    every tool). So "this adapter has no record of that window" means the
    snapshot cannot be vouched for, and the honest answer is to refuse.
    """
    a, fake = adapter(_window_state(snapshot_id="s1"))
    # NO observe -- nothing was ever recorded for (pid, window_id).
    assert a._snapshots == {}
    with pytest.raises(StaleSnapshot) as exc:
        a.act("click", _args(snapshot_id="s1"))
    assert "no observation on record" in str(exc.value)
    assert not [c for c in fake.calls if c[0] == "click"], (
        "the action reached the driver despite the adapter never having "
        "observed that window"
    )


def test_a_refusal_with_no_record_does_not_need_the_runtime(adapter):
    """Refused BEFORE start(), so an undeterminable snapshot costs nothing.

    Pinned because the check used to sit after ``self.start()``, which means
    a call that was always going to be refused first span up the Cua runtime.
    """
    a, fake = adapter(_window_state(snapshot_id="s1"))
    a.shutdown()
    a._driver = None
    with pytest.raises(StaleSnapshot):
        a.act("click", _args(snapshot_id="s1"))
    assert a._driver is None, "the runtime was started for a refused call"


def test_an_unimplemented_verb_is_refused_not_dispatched(adapter):
    """The narrowness IS the boundary — there must be no generic passthrough
    that could reach clipboard_read or the browser surface."""
    a, fake = adapter(_window_state())
    a.observe("mini", "scratch", 1, 2)
    with pytest.raises(DriverUnavailable) as exc:
        a.act("clipboard_read", _args())
    assert "not implemented" in str(exc.value)
    assert len(fake.calls) == 1  # only the observe


def test_the_implemented_set_is_exactly_the_v1_verbs():
    assert set(cua._BUILDERS) == {
        "click", "press_key", "scroll", "type_text", "invoke_menu"}


def test_a_driver_stale_error_maps_onto_our_exception(adapter):
    a, _ = adapter(_window_state(), effect=RuntimeError("element token is stale"))
    a.observe("mini", "scratch", 1, 2)
    with pytest.raises(StaleSnapshot):
        a.act("click", _args())


def test_a_missing_sdk_says_what_to_install(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_cua(name, *a, **k):
        if name == "cua_driver":
            raise ImportError("no module")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_cua)
    with pytest.raises(DriverUnavailable) as exc:
        cua._require_sdk()
    assert "--extra computer" in str(exc.value)
    assert "cannot be exercised without a display" in str(exc.value)


def test_an_unavailable_runtime_refuses_at_start(monkeypatch):
    monkeypatch.setattr(cua, "_require_sdk", lambda: _FakeSdk)
    fake = _FakeDriver()
    fake._available = False
    monkeypatch.setattr(_FakeSdk, "CuaDriver",
                        SimpleNamespace(create=lambda: fake), raising=False)
    a = cua.CuaDriverAdapter(target="mini")
    with pytest.raises(DriverUnavailable) as exc:
        a.start()
    assert "unavailable" in str(exc.value)


# ── THE HONESTY GUARD ───────────────────────────────────────────────────────

def test_this_module_says_it_cannot_cover_the_driver():
    """A green run here must not read as "the driver works"."""
    import re
    from pathlib import Path

    # Whitespace-normalised: a guard that breaks when a docstring is
    # re-wrapped teaches people to delete the guard.
    doc = re.sub(r"\s+", " ", Path(cua.__file__).read_text())
    assert "STRUCTURALLY UNCOVERABLE BY CI" in doc
    assert "on-box outcome check is the only evidence" in doc
    assert "same shape as the MCP stdio transport" in doc
