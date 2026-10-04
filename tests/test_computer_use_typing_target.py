"""Text goes into THE field the prompt names (computer-use v1.1, D2).

THE DEFECT
----------
A ``type-N`` candidate said "Type the prepared text into the entry 'Search'"
and carried that element's token — and the adapter dropped the token.
cua-driver 0.28.2's ``TypeTextInput`` takes only an ``ActionTarget`` whose
variants are ``WINDOW`` and ``DESKTOP``, so the text went to WHATEVER HAD
FOCUS. The approval sentence described an action the driver was never asked
to perform: a consent-honesty defect, not a corner case.

THE FIX (Appendix A of the design: "set_value by element token, else
withhold type-N")
------------------------------------------------------------------------
The pinned driver's ``set_value`` tool addresses ONE element by its token
(``platform-linux/src/tools/impl_.rs:5467-5545`` at the 0.28.2 tag) and
replaces its contents. It is reached through ``call_tool``, the SDK's generic
surface, because 0.28.2 has no typed method for it. So the table offers
``set-N`` — "Set the entry 'Search' to the prepared text, replacing what it
holds" — and NO LONGER offers focus-typed ``type-N`` at all. The prompt never
names a field the driver will not target.

``set_value`` carries the text as a PAYLOAD, so it is approve-once, every
time, exactly as ``type_text`` was.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from prometheus.computer import cua
from prometheus.computer.actions import (
    ACTION_MODELS, SetValueInput, schema_for,
)
from prometheus.computer.candidates import (
    build_candidates, build_choice_request,
)
from prometheus.computer.chooser import RuleChooser
from prometheus.computer.driver import FixtureDriver
from prometheus.computer.loop import ComputerUseLoop
from prometheus.computer.tools import TOOL_CLASSES
from prometheus.computer.types import Element, Observation
from prometheus.permissions.checker import PermissionMode, SecurityGate
from prometheus.permissions.computer_extent import computer_extent_for
from prometheus.permissions.computer_schema import declared_payload_params


def _obs(snapshot="s1") -> Observation:
    return Observation(
        target="box", app="scratchapp", pid=1, window_id=2,
        snapshot_id=snapshot,
        elements=(
            Element(0, f"tok-send-{snapshot}", "push button", "Send"),
            Element(1, f"tok-field-{snapshot}", "entry", "Search"),
        ),
    )


# ── THE TABLE ───────────────────────────────────────────────────────────────

def test_the_table_offers_setting_the_named_field_never_typing_to_focus():
    rows = build_candidates(_obs(), text_to_type="hello")
    ids = [c.candidate_id for c in rows]
    assert "set-1" in ids
    assert not any(i.startswith("type-") for i in ids), (
        "a focus-typed row is still offered — its sentence names a field the "
        "driver would not target")
    row = next(c for c in rows if c.candidate_id == "set-1")
    assert row.tool_name == "computer_set_value"
    assert row.arguments["element_token"] == "tok-field-s1"
    assert row.arguments["text"] == "hello"
    assert row.description == (
        "Set the entry 'Search' to the prepared text, replacing what it holds")


def test_no_text_means_no_set_row():
    assert not [c for c in build_candidates(_obs())
                if c.tool_name == "computer_set_value"]


def test_the_chooser_never_sees_the_text():
    rows = build_candidates(_obs(), text_to_type="transfer 500")
    blob = str(build_choice_request("fill search", _obs(), rows).candidates)
    assert "transfer 500" not in blob
    assert "tok-field" not in blob


# ── THE VERB AND ITS CONSENT ────────────────────────────────────────────────

def test_set_value_is_a_declared_payload_verb():
    assert ACTION_MODELS["set_value"] is SetValueInput
    assert "set_value" in TOOL_CLASSES
    assert declared_payload_params(schema_for("set_value")) == ("text",)


def test_set_value_is_never_rememberable_and_says_what_it_does():
    row = next(c for c in build_candidates(_obs(), text_to_type="x")
               if c.candidate_id == "set-1")
    extent, unknown = computer_extent_for(
        row.tool_name, row.arguments, schema=schema_for("set_value"))
    assert unknown is None and extent is not None
    assert not extent.rememberable
    assert "set any field value" in extent.describe()


# ── THE ADAPTER: THE TOKEN REACHES THE DRIVER ───────────────────────────────

class _Sdk:
    GetWindowStateInput = staticmethod(lambda **kw: SimpleNamespace(**kw))


class _Driver:
    def __init__(self):
        self.calls: list = []

    def is_available(self):
        return True

    async def get_window_state(self, inp):
        el = SimpleNamespace(
            element_index=1, element_token="tok-field", role="entry",
            label="Search", value=None, actions=None)
        return SimpleNamespace(snapshot_id="s1", app_name="scratchapp",
                               elements=[el])

    async def call_tool(self, name, arguments_json):
        self.calls.append((name, json.loads(arguments_json)))
        return SimpleNamespace(text="", is_error=False, error_code=None,
                               action=None, degraded=False)

    async def type_text(self, inp):  # pragma: no cover - must not be reached
        self.calls.append(("type_text", inp))

    async def shutdown(self):
        pass


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(cua, "_require_sdk", lambda: _Sdk)
    fake = _Driver()
    monkeypatch.setattr(_Sdk, "CuaDriver", SimpleNamespace(create=lambda: fake),
                        raising=False)
    a = cua.CuaDriverAdapter(target="box")
    a.start()
    yield a, fake
    a.shutdown()


#: The 0.28.2 ``set_value`` input schema, read from the tag's source
#: (platform-linux/src/tools/impl_.rs:5480-5490): required pid and value,
#: ``additionalProperties: false``.
_SET_VALUE_KEYS = {"session", "pid", "window_id", "element_index",
                   "element_token", "snapshot_id", "value"}


def test_set_value_names_the_approved_element_by_its_token(adapter):
    a, fake = adapter
    a.observe("box", "scratchapp", 1, 2)
    a.act("set_value", {
        "target": "box", "app": "scratchapp", "pid": 1, "window_id": 2,
        "snapshot_id": "s1", "element_token": "tok-field", "text": "hello",
        "delivery_mode": "background"})
    assert fake.calls == [("set_value", {
        "pid": 1, "window_id": 2, "element_token": "tok-field",
        "snapshot_id": "s1", "value": "hello"})]
    name, payload = fake.calls[0]
    assert set(payload) <= _SET_VALUE_KEYS
    assert {"pid", "value"} <= set(payload)


def test_set_value_without_a_token_is_refused_before_the_driver(adapter):
    a, fake = adapter
    a.observe("box", "scratchapp", 1, 2)
    with pytest.raises(cua.DriverUnavailable):
        a.act("set_value", {
            "target": "box", "app": "scratchapp", "pid": 1, "window_id": 2,
            "snapshot_id": "s1", "text": "hello"})
    assert fake.calls == []


def test_the_real_sdk_exposes_the_generic_call_surface():
    cua_driver = pytest.importorskip("cua_driver")
    from cua_driver._native import CuaDriverProtocol

    assert hasattr(CuaDriverProtocol, "call_tool")
    assert not hasattr(cua_driver, "SetValueInput"), (
        "0.28.2 grew a typed set_value — switch the adapter to it")


# ── END TO END ──────────────────────────────────────────────────────────────

def test_the_loop_sets_the_field_the_operator_approved():
    prompted: list = []

    async def approver(tool_name, reason, arguments=None):
        prompted.append((tool_name, reason, arguments))
        return True

    driver = FixtureDriver([_obs("s1"), _obs("s2")])
    loop = ComputerUseLoop(
        driver=driver, chooser=RuleChooser(prefer=("search",)),
        gate=SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None),
        approve=approver, skip_preconditions=True)
    result = asyncio.run(loop.step(
        "fill search", "box", "scratchapp", 1, 2, text_to_type="hello"))
    assert result.ok, result.reason
    tool, reason, args = prompted[0]
    assert tool == "computer_set_value"
    assert "no lasting grant is offered" in reason
    assert driver.dispatched == [("set_value", {
        **args, "element_token": "tok-field-s1", "text": "hello"})]
