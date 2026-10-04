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


class _DeliveryMode(enum.Enum):
    """``ActionDeliveryMode`` — what the driver REPORTS it did."""

    BACKGROUND = 0
    FOREGROUND = 1
    NOT_APPLICABLE = 2
    UNKNOWN = 3


def _action_result(effect=_Effect.CONFIRMED, delivery=None, escalation=None):
    """``ActionResult``'s shape: the effect is a top-level field."""
    return _rec(effect=effect, route=None, delivery=delivery, evidence=None,
                escalation=escalation)


def _tool_result(effect=_Effect.CONFIRMED, *, is_error=False, error_code=None,
                 text="", degraded=False, action=True):
    """``ToolResult``'s shape: the effect sits at ``.action.effect``, beside
    ``is_error``/``error_code``. press_key, scroll, type_text and invoke_menu
    return THIS, not an ActionResult (0.28.2, ``_native.py:5598-5652``)."""
    return _rec(text=text, images=[], structured_json=None, is_error=is_error,
                error_code=error_code,
                action=_action_result(effect) if action else None,
                verification=None, degraded=degraded, raw_json="{}")


class _FakeDriver:
    """A CuaDriver stand-in. Async, like the real one, and returning the
    real result SHAPE per verb — a fake that returned ``ActionResult`` for
    every verb is what hid D3."""

    def __init__(self, window_state=None, effect=_Effect.CONFIRMED):
        self.window_state = window_state
        self.effect = effect
        #: When set, every action returns THIS object as-is.
        self.result = None
        #: When set, every action waits on it before returning.
        self.gate = None
        self.calls: list[tuple[str, object]] = []
        self._available = True

    def is_available(self):
        return self._available

    async def get_window_state(self, inp):
        self.calls.append(("get_window_state", inp))
        return self.window_state

    async def _action(self, name, inp):
        self.calls.append((name, inp))
        if self.gate is not None:
            await self.gate.wait()
        if isinstance(self.effect, Exception):
            raise self.effect
        if self.result is not None:
            return self.result
        if name == "click":
            return _action_result(self.effect)
        return _tool_result(self.effect)

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


def _window_state(snapshot_id="s1", elements=None, app_name="Scratch",
                  **over):
    """``WindowStateOutput``'s fields, every one of them (0.28.2).

    One element by default: an empty tree is UNUSABLE (D10), so a test
    about something else must not trip over it by accident.
    """
    elements = [_el()] if elements is None else list(elements)
    fields = dict(
        pid=1, window_id=2, snapshot_id=snapshot_id, app_name=app_name,
        window_title="t", tree_markdown=None, elements=elements,
        element_count=len(elements), total_element_count=len(elements),
        returned_element_count=len(elements), filtered_element_count=None,
        # Hard-coded false by 0.28.2 on Linux — carried, never relied on.
        elements_complete=False,
        degraded=False, degraded_reason=None,
        truncated=False, truncation_reason=None,
        screenshot_width=None, screenshot_height=None, screenshot_scale=None,
        screenshot_mime_type=None, screenshot_file_path=None,
        screenshot_frame_valid=None, window_bounds=None, images=[],
    )
    fields.update(over)
    return _rec(**fields)


def _el(idx=0, token="tok", role="push button", label="Increment", **over):
    """``WindowElement``'s fields, every one of them (0.28.2).

    ⚠ NO ``editable``. The real type has no such field; this fake used to
    supply one, which is how the adapter's read of it (always False) went
    unnoticed (D4). ``test_the_fakes_mirror_the_real_sdk_shapes`` pins it.
    """
    fields = dict(
        element_index=idx, role=role, depth=0, element_token=token,
        label=label, value=None, value_description=None, enabled=True,
        selected=False, in_web_content=False, actions=None, parent_index=None,
        frame=None, min=None, max=None,
    )
    fields.update(over)
    return _rec(**fields)


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
        _el(0, "tok"), _el(1, token=None, role="x", label="")]))
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


# ── THE ADAPTER TELLS THE TRUTH (computer-use v1.1, PR 1) ─────────────────
#
# Each test below names the defect it pins (docs/design/computer-use-v1.1.md
# §3). They were written red against origin/main 856ebb8 first.

_NON_CLICK = [
    ("press_key", {"key": "return"}),
    ("scroll", {"direction": "down", "amount": 1}),
    ("type_text", {"text": "hello"}),
    ("invoke_menu", {"path": ["File", "Save"]}),
]
_ALL_VERBS = [("click", {}), *_NON_CLICK]


@pytest.mark.parametrize("effect", [_Effect.SUSPECTED_NOOP, _Effect.REFUSED])
@pytest.mark.parametrize("verb,extra", _ALL_VERBS)
def test_a_no_op_raises_for_every_verb(adapter, verb, extra, effect):
    """D3. Only ``click`` returns an ``ActionResult``; the other four return a
    ``ToolResult`` whose effect sits at ``.action.effect``. Reading only
    ``result.effect`` turned every non-click SUSPECTED_NOOP into
    UNVERIFIABLE, which does not raise — on four verbs of five."""
    a, _ = adapter(_window_state(), effect=effect)
    a.observe("mini", "scratch", 1, 2)
    with pytest.raises(DriverUnavailable) as exc:
        a.act(verb, _args(**extra))
    assert effect.name in str(exc.value)


@pytest.mark.parametrize("verb,extra", _NON_CLICK)
def test_a_tool_result_error_raises_and_names_its_code(adapter, verb, extra):
    """D3. ``is_error`` with an ``error_code`` and no action is a failure the
    driver stated outright. It used to read as UNVERIFIABLE."""
    a, fake = adapter(_window_state())
    fake.result = _tool_result(
        is_error=True, error_code="background_unavailable",
        text="cannot deliver without raising the window", action=False)
    a.observe("mini", "scratch", 1, 2)
    with pytest.raises(DriverUnavailable) as exc:
        a.act(verb, _args(**extra))
    assert "background_unavailable" in str(exc.value)


@pytest.mark.parametrize("verb,extra", _NON_CLICK)
def test_a_confirmed_tool_result_lands(adapter, verb, extra):
    a, _ = adapter(_window_state(), effect=_Effect.CONFIRMED)
    a.observe("mini", "scratch", 1, 2)
    out = a.act(verb, _args(**extra))
    assert out["effect"] == "CONFIRMED"
    assert out["landed"] is True
    assert out["confirmed_by_driver"] is True


def test_a_tool_result_stale_error_maps_onto_our_exception(adapter):
    a, fake = adapter(_window_state())
    fake.result = _tool_result(is_error=True, error_code="stale_snapshot",
                               text="element token is stale", action=False)
    a.observe("mini", "scratch", 1, 2)
    with pytest.raises(StaleSnapshot):
        a.act("press_key", _args(key="return"))


def test_the_delivery_the_driver_reports_is_carried_and_a_mismatch_logged(
        adapter, caplog):
    """D5. We ask for background; the driver says what it actually did. A
    foreground answer to a background request is logged, with the driver's
    escalation, rather than silently reported as the background extent."""
    import logging

    a, fake = adapter(_window_state())
    fake.result = _action_result(
        _Effect.CONFIRMED,
        delivery=_rec(mode=_DeliveryMode.FOREGROUND, delivered_count=1),
        escalation=_rec(target=_rec(name="FOREGROUND"),
                        reason=_rec(name="ROUTE_UNAVAILABLE")),
    )
    a.observe("mini", "scratch", 1, 2)
    with caplog.at_level(logging.WARNING, logger=cua.logger.name):
        out = a.act("click", _args())
    assert out["delivery_requested"] == "background"
    assert out["delivery_reported"] == "foreground"
    assert out["delivery_matches"] is False
    assert out["escalation"] == "foreground: route_unavailable"
    assert "foreground" in caplog.text and "background" in caplog.text


def test_a_matching_delivery_is_reported_as_matching(adapter):
    a, fake = adapter(_window_state())
    fake.result = _action_result(
        _Effect.CONFIRMED,
        delivery=_rec(mode=_DeliveryMode.BACKGROUND, delivered_count=1))
    a.observe("mini", "scratch", 1, 2)
    out = a.act("click", _args())
    assert out["delivery_reported"] == "background"
    assert out["delivery_matches"] is True
    assert out["escalation"] is None


@pytest.mark.parametrize("verb,extra", _ALL_VERBS)
def test_delivery_is_marked_unenforced_where_the_input_cannot_carry_it(
        adapter, verb, extra):
    """D5. In 0.28.2 only ``ClickInput`` takes a delivery mode. For the other
    four verbs the extent's ``:background`` names a property the driver was
    never asked to honour, so the result says so."""
    a, _ = adapter(_window_state())
    a.observe("mini", "scratch", 1, 2)
    out = a.act(verb, _args(**extra))
    assert out["delivery_enforced"] is (verb == "click")


def test_the_driver_completeness_report_reaches_the_observation(adapter):
    """D6. A truncated tree is CARRIED (big apps truncate at MAX_ELEMENTS and
    must still work), never mistaken for a complete one."""
    a, _ = adapter(_window_state(
        truncated=True, truncation_reason="max_elements",
        total_element_count=500, returned_element_count=200,
        elements_complete=False))
    obs = a.observe("mini", "scratch", 1, 2)
    assert obs.truncated is True
    assert obs.truncation_reason == "max_elements"
    assert obs.total_element_count == 500
    assert obs.returned_element_count == 200
    assert obs.elements_complete is False
    assert obs.degraded is False
    assert obs.unusable_reason is None


def test_a_degraded_tree_is_unusable(adapter):
    """D6. The driver telling us its own tree is degraded is the loudest
    "shaped like success" warning there is, and it was dropped."""
    a, _ = adapter(_window_state(degraded=True,
                                 degraded_reason="accessibility walk timed out"))
    obs = a.observe("mini", "scratch", 1, 2)
    assert obs.degraded is True
    assert obs.degraded_reason == "accessibility walk timed out"
    assert obs.unusable_reason
    assert "accessibility walk timed out" in obs.unusable_reason


def test_element_fields_the_driver_reports_pass_through(adapter):
    """D6. ``in_web_content`` is the signal the site term needs (§5.4)."""
    a, _ = adapter(_window_state(elements=[
        _el(in_web_content=True, enabled=False, selected=True,
            parent_index=4)]))
    el = a.observe("mini", "scratch", 1, 2).elements[0]
    assert el.in_web_content is True
    assert el.enabled is False
    assert el.selected is True
    assert el.parent_index == 4


@pytest.mark.parametrize("node,seen", [
    (_el(1, token=None, role="document web"), True),
    (_el(1, token=None, role="embedded"), True),
    (_el(1, token=None, role="panel", in_web_content=True), True),
    (_el(1, token="t1", role="push button", in_web_content=True), True),
    (_el(1, token=None, role="panel"), False),
])
def test_web_content_is_judged_over_the_whole_walk(adapter, node, seen):
    """The site term needs "is there web content ANYWHERE in this window?"
    A tokenless document node is dropped from ``elements`` (it cannot be
    addressed), so the answer is computed before that filter."""
    a, _ = adapter(_window_state(elements=[_el(0), node]))
    obs = a.observe("mini", "scratch", 1, 2)
    assert obs.web_content_seen is seen
    assert [e.element_index for e in obs.elements] == (
        [0, 1] if node.element_token else [0])


def test_no_nodes_means_no_evidence_about_web_content(adapter):
    a, _ = adapter(_window_state(elements=[]))
    assert a.observe("mini", "scratch", 1, 2).web_content_seen is None


def test_an_empty_tree_is_unusable_not_empty(adapter):
    """D10. A snapshot with an id and no elements used to be usable, and the
    candidate table then offered Return, Tab and Escape into a window we
    could not see."""
    a, _ = adapter(_window_state(elements=[]))
    obs = a.observe("mini", "scratch", 1, 2)
    assert obs.elements == ()
    assert obs.unusable_reason
    assert "no elements" in obs.unusable_reason


@pytest.mark.parametrize("state", [
    {"elements": []},
    {"degraded": True, "degraded_reason": "walk failed"},
])
def test_nothing_acts_on_an_unusable_observation(adapter, state):
    """D10/D6, at the adapter. The wrapped-tool path calls ``act`` without a
    candidate table (D17), so the refusal cannot live only in the table."""
    a, fake = adapter(_window_state(**state))
    a.observe("mini", "scratch", 1, 2)
    with pytest.raises(DriverUnavailable) as exc:
        a.act("press_key", _args(key="return"))
    assert "unusable" in str(exc.value)
    assert not [c for c in fake.calls if c[0] == "press_key"]


def test_a_usable_observation_clears_an_earlier_unusable_one(adapter):
    a, fake = adapter(_window_state(elements=[]))
    a.observe("mini", "scratch", 1, 2)
    fake.window_state = _window_state(snapshot_id="s2")
    a.observe("mini", "scratch", 1, 2)
    assert a.act("click", _args(snapshot_id="s2"))["landed"] is True


def test_a_failed_health_check_is_not_skipped_on_the_next_start(monkeypatch):
    """D11. ``start()`` kept the driver after ``is_available()`` said no, so
    the SECOND start returned early and reported nothing."""
    monkeypatch.setattr(cua, "_require_sdk", lambda: _FakeSdk)
    fake = _FakeDriver()
    fake._available = False
    monkeypatch.setattr(_FakeSdk, "CuaDriver",
                        SimpleNamespace(create=lambda: fake), raising=False)
    a = cua.CuaDriverAdapter(target="mini")
    try:
        with pytest.raises(DriverUnavailable):
            a.start()
        with pytest.raises(DriverUnavailable):
            a.start()
        assert a._driver is None
    finally:
        a.shutdown()


def _wait_until(predicate, timeout=5.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def test_a_timed_out_action_is_outcome_unknown_and_forces_a_reobserve(
        adapter, monkeypatch):
    """D13. ``.result(timeout=…)`` does not cancel the SDK coroutine: probed,
    the timeout fired at 1.9 s and the action landed at 2.7 s. So a timeout
    is "outcome unknown", never "failed", and the snapshot it acted on can no
    longer be vouched for."""
    import asyncio

    from prometheus.computer.driver import ActionOutcomeUnknown

    monkeypatch.setattr(cua, "OPERATION_TIMEOUT_SECONDS", 0.2)
    a, fake = adapter(_window_state())
    a.observe("mini", "scratch", 1, 2)
    fake.gate = asyncio.Event()
    with pytest.raises(ActionOutcomeUnknown) as exc:
        a.act("click", _args())
    assert isinstance(exc.value, DriverUnavailable)
    assert "may still land" in str(exc.value)
    a._loop.call_soon_threadsafe(fake.gate.set)
    fake.gate = None
    _wait_until(lambda: not a._call_lock.locked())
    with pytest.raises(StaleSnapshot):
        a.act("click", _args())
    a.observe("mini", "scratch", 1, 2)
    assert a.act("click", _args())["landed"] is True


def test_one_driver_call_at_a_time(adapter, monkeypatch):
    """D13. Two steps on one adapter would interleave on its loop. While a
    call is in flight — even one we stopped waiting for — the next call
    waits, and if it cannot get in it is refused WITHOUT reaching the
    driver."""
    import asyncio

    from prometheus.computer.driver import DriverBusy

    monkeypatch.setattr(cua, "OPERATION_TIMEOUT_SECONDS", 0.2)
    a, fake = adapter(_window_state())
    a.observe("mini", "scratch", 1, 2)
    fake.gate = asyncio.Event()
    with pytest.raises(DriverUnavailable):
        a.act("click", _args())
    reached = len(fake.calls)
    with pytest.raises(DriverBusy):
        a.observe("mini", "scratch", 1, 2)
    assert len(fake.calls) == reached, (
        "a second call reached the driver while the first was in flight")
    a._loop.call_soon_threadsafe(fake.gate.set)
    fake.gate = None
    _wait_until(lambda: not a._call_lock.locked())
    assert a.observe("mini", "scratch", 1, 2).snapshot_id == "s1"


# ── THE REAL SDK TYPES (skipped where cua-driver is not installed, as in CI) ─


@pytest.fixture
def real_sdk():
    return pytest.importorskip("cua_driver")


def _params(cls) -> set[str]:
    import inspect

    return set(inspect.signature(cls.__init__).parameters) - {"self"}


def test_the_fakes_mirror_the_real_sdk_shapes(real_sdk):
    """The fakes are only worth something if they have the real fields and
    no others. A fake ``editable`` is how D4 hid; a fake ``ActionResult``
    for every verb is how D3 hid."""
    from cua_driver._native import ToolResult

    assert set(vars(_el())) == _params(real_sdk.WindowElement)
    assert set(vars(_window_state())) == _params(real_sdk.WindowStateOutput)
    assert set(vars(_action_result())) == _params(real_sdk.ActionResult)
    assert set(vars(_tool_result())) == _params(ToolResult)
    assert {m.name for m in _DeliveryMode} == set(
        real_sdk.ActionDeliveryMode.__members__)
    assert {m.name for m in _Effect} == set(real_sdk.ActionEffect.__members__)


def test_the_observe_input_builds_with_the_pinned_sdk(real_sdk):
    """D12's other half. From 0.28.3 ``max_image_dimension`` is a REQUIRED
    keyword, so this construction is what breaks on an unpinned install."""
    inp = cua._window_state_input(real_sdk, pid=1, window_id=2, session=None)
    assert type(inp) is real_sdk.GetWindowStateInput
    assert inp.include_screenshot is False
    assert inp.max_elements == cua.MAX_ELEMENTS


@pytest.mark.parametrize("verb,extra", _ALL_VERBS)
def test_each_action_input_builds_with_the_pinned_sdk(real_sdk, verb, extra):
    _, make = cua._BUILDERS[verb]
    inp = make(real_sdk, _args(**extra), None)
    assert type(inp).__module__.startswith("cua_driver")


def _real_action(real_sdk, effect, delivery=None):
    return real_sdk.ActionResult(
        effect=getattr(real_sdk.ActionEffect, effect),
        route=real_sdk.ActionRoute.ACCESSIBILITY,
        delivery=(None if delivery is None else real_sdk.ActionDelivery(
            mode=getattr(real_sdk.ActionDeliveryMode, delivery),
            delivered_count=1)),
        evidence=None, escalation=None)


def _real_tool_result(action, *, is_error=False, error_code=None):
    from cua_driver._native import ToolResult

    return ToolResult(text="", images=[], structured_json=None,
                      is_error=is_error, error_code=error_code, action=action,
                      verification=None, degraded=False, raw_json="{}")


@pytest.mark.parametrize("verb,extra", _NON_CLICK)
def test_a_real_tool_result_no_op_raises(real_sdk, verb, extra):
    result = _real_tool_result(_real_action(real_sdk, "SUSPECTED_NOOP"))
    with pytest.raises(DriverUnavailable) as exc:
        cua._verdict(verb, result, _args(**extra))
    assert "SUSPECTED_NOOP" in str(exc.value)


def test_a_real_tool_result_error_raises(real_sdk):
    result = _real_tool_result(None, is_error=True,
                               error_code="background_unavailable")
    with pytest.raises(DriverUnavailable) as exc:
        cua._verdict("press_key", result, _args(key="return"))
    assert "background_unavailable" in str(exc.value)


def test_a_real_action_result_reports_its_delivery(real_sdk):
    out = cua._verdict(
        "click", _real_action(real_sdk, "CONFIRMED", delivery="FOREGROUND"),
        _args())
    assert out["landed"] is True
    assert out["delivery_reported"] == "foreground"
    assert out["delivery_matches"] is False


def test_a_real_window_state_translates(real_sdk, adapter):
    """The whole observe translation over the real dataclasses."""
    el = real_sdk.WindowElement(
        element_index=3, role="push button", depth=1, element_token="t3",
        label="Save", value=None, value_description=None, enabled=True,
        selected=False, in_web_content=False, actions=["press"],
        parent_index=0, frame=None, min=None, max=None)
    state = real_sdk.WindowStateOutput(
        pid=1, window_id=2, snapshot_id="s9", app_name="gedit",
        window_title="x", tree_markdown=None, elements=[el], element_count=1,
        total_element_count=1, returned_element_count=1,
        filtered_element_count=None, elements_complete=False, degraded=False,
        degraded_reason=None, truncated=False, truncation_reason=None,
        screenshot_width=None, screenshot_height=None, screenshot_scale=None,
        screenshot_mime_type=None, screenshot_file_path=None,
        screenshot_frame_valid=None, window_bounds=None, images=[])
    a, _ = adapter(state)
    obs = a.observe("mini", "gedit", 1, 2)
    assert obs.unusable_reason is None
    assert obs.app == "gedit"
    assert obs.elements[0].element_token == "t3"
    assert obs.elements[0].in_web_content is False
    assert obs.elements[0].editable is False
    assert obs.web_content_seen is False


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
