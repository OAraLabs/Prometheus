"""Named machines: local vs remote is a table entry, not a second feature.

A desktop driver is a process the daemon talks to; whether that process runs
here or elsewhere is a property of the CONNECTION. The registry is what makes
that true for the rest of the system — every other module takes a target name
and never branches on where it resolves to.

The safety property these tests hold is narrow and important: **an undeclared
or unbound target is REFUSED, never resolved to another machine.** A typo must
be an error, not a quiet redirection to the nearest desktop.
"""

from __future__ import annotations

import pytest

from prometheus.computer.driver import DriverUnavailable, FixtureDriver
from prometheus.computer.targets import (
    KIND_LOCAL,
    KIND_REMOTE,
    Target,
    TargetRegistry,
    UnknownTarget,
    registry_from_config,
)
from prometheus.computer.types import Element, Observation


def _driver(target="box"):
    return FixtureDriver([Observation(
        target=target, app="a", pid=1, window_id=2, snapshot_id="s",
        elements=(Element(0, "tok", "push button", "Go"),),
    )])


# ── NAMES ARE CONSENT TERMS, SO THEY ARE CONSTRAINED ───────────────────────

@pytest.mark.parametrize("bad", [
    "has:colon",      # would forge a different extent
    "Has-Capital",    # would split one machine's consent in two
    "has space",
    "",
    "-leading",
    "x" * 40,
])
def test_an_invalid_target_name_is_refused(bad):
    with pytest.raises(ValueError):
        Target(name=bad)


@pytest.mark.parametrize("ok", ["box", "laptop", "vm-1", "build_host", "b"])
def test_reasonable_names_are_accepted(ok):
    assert Target(name=ok).name == ok


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        Target(name="box", kind="teleport")


# ── RESOLUTION NEVER FALLS BACK ─────────────────────────────────────────────

def test_an_undeclared_target_raises_rather_than_resolving():
    reg = TargetRegistry()
    reg.declare(Target(name="box"), _driver())
    with pytest.raises(UnknownTarget):
        reg.resolve("laptop")


def test_a_typo_does_not_silently_hit_the_only_declared_machine():
    """The failure this whole term exists to prevent, stated as a test."""
    reg = TargetRegistry()
    reg.declare(Target(name="laptop", kind=KIND_REMOTE), _driver("laptop"))
    with pytest.raises(UnknownTarget):
        reg.resolve("laptopp")


def test_a_declared_but_unbound_target_refuses_rather_than_borrowing_a_driver():
    reg = TargetRegistry()
    reg.declare(Target(name="box"), _driver())
    reg.declare(Target(name="laptop", kind=KIND_REMOTE))  # no driver bound
    with pytest.raises(DriverUnavailable) as exc:
        reg.resolve("laptop")
    assert "laptop" in str(exc.value)


def test_binding_later_works():
    reg = TargetRegistry()
    reg.declare(Target(name="box"))
    d = _driver()
    reg.bind("box", d)
    assert reg.resolve("box") is d


def test_binding_an_undeclared_target_raises():
    reg = TargetRegistry()
    with pytest.raises(UnknownTarget):
        reg.bind("ghost", _driver())


def test_declaring_twice_is_an_error():
    reg = TargetRegistry()
    reg.declare(Target(name="box"))
    with pytest.raises(ValueError):
        reg.declare(Target(name="box"))


# ── CONFIG ──────────────────────────────────────────────────────────────────

def test_targets_come_from_config():
    reg = registry_from_config({"targets": {
        "box": {"kind": KIND_LOCAL, "description": "this machine"},
        "laptop": {"kind": KIND_REMOTE, "connection": {"via": "opaque"}},
    }})
    assert reg.names() == ("box", "laptop")
    assert reg.get("laptop").kind == KIND_REMOTE


def test_no_config_yields_no_implicit_local_target():
    """An implicit target is a machine nobody named.

    Its grants would read ':app:verb:delivery' — a blank first term that any
    other unnamed machine would also produce.
    """
    assert registry_from_config(None).names() == ()
    assert registry_from_config({}).names() == ()


def test_connection_details_never_reach_the_extent_terms():
    """The name is the grant term; the connection is not part of it."""
    reg = registry_from_config({"targets": {
        "laptop": {"kind": KIND_REMOTE, "connection": {"endpoint": "opaque-value"}},
    }})
    assert "opaque-value" not in "".join(reg.names())
    assert reg.get("laptop").name == "laptop"


def test_a_driver_refuses_an_observation_labelled_for_another_machine():
    """A mis-bound registry must not produce actions labelled for the wrong
    machine — the driver checks rather than trusting its binding."""
    d = _driver("box")
    with pytest.raises(DriverUnavailable):
        d.observe("laptop", "a", 1, 2)
