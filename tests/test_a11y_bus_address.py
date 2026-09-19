"""The a11y bus address is ASKED FOR, not assembled from a guess.

THE DEFECT
----------
``_check_observe_half`` probed ``$XDG_RUNTIME_DIR/at-spi/bus``. AT-SPI
suffixes that socket with the DISPLAY NUMBER: on an X11 session at ``:1`` it
is ``at-spi/bus_1``. Only a session that happened to be display ``:0``
produced the unsuffixed name the code was written against.

Measured on the live box 2026-09-19, after a reboot changed the session from
Wayland/``:0`` to X11/``:1``: ``/api/status`` reported *"the accessibility bus
socket is absent — observation would return an empty tree"* while the tree was
fully readable, 13 applications enumerated through it. A false ``act_only``.

It failed CLOSED, so nothing was damaged. It was still a false statement on an
endpoint whose whole job is to be believed when something is wrong.

⚠ WHY THE EXISTING HERMETIC SUITE COULD NOT HAVE CAUGHT IT
-----------------------------------------------------------
``test_computer_status_block.py`` builds ``tmp_path/at-spi/bus`` and asserts
the probe finds ``tmp_path/at-spi/bus``. Both sides carry the SAME WRONG NAME,
so they agree perfectly and agree with nothing real — the suite would have
stayed green forever with the production path unreachable.

Hermeticity protects against the ENVIRONMENT varying. It is structurally
blind to a CONSTANT the test and the code share. That is a different defect
from the host-reading one those tests were already fixed for, and it survives
that fix untouched.

So this file has two halves:

* hermetic tests for the BEHAVIOUR (both spellings, the visible fallback);
* **one non-hermetic confirmation that the constant is right**, run against
  the real session bus — not for the assertion, but so the name is confirmed
  by something other than itself.
"""

from __future__ import annotations

import os
import shutil

import pytest

from prometheus.computer import driver
from prometheus.computer.driver import (
    HALF_OK,
    HALF_UNAVAILABLE,
    HALF_UNKNOWN,
    _a11y_bus_address,
    _ask_the_bus,
    _check_observe_half,
)

GUESS_MARK = "GUESSED"


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """A runtime dir, and a bus that refuses to answer (so the fallback runs)."""
    monkeypatch.setattr(driver, "_ask_the_bus", lambda env: None)
    d = tmp_path / "rt"
    (d / "at-spi").mkdir(parents=True)
    return d


def _env(runtime_dir, display=":1"):
    return {"XDG_RUNTIME_DIR": str(runtime_dir), "DISPLAY": display}


# ── THE REGRESSION: the display-suffixed socket ─────────────────────────────

def test_a_display_suffixed_socket_is_found(runtime):
    """THE BUG. `bus_1` exists, `bus` does not, and the probe must see it."""
    (runtime / "at-spi" / "bus_1").write_text("")
    result = _check_observe_half(_env(runtime, ":1"))
    assert result.state == HALF_OK, (
        f"a display-suffixed a11y socket was reported as {result.state}: "
        f"{result.detail} — this is the live defect, reproduced"
    )


def test_the_unsuffixed_socket_still_works(runtime):
    """The `:0` spelling must keep working; the fix adds a name, not swaps it."""
    (runtime / "at-spi" / "bus").write_text("")
    assert _check_observe_half(_env(runtime, ":0")).state == HALF_OK


def test_a_genuinely_absent_bus_is_still_unavailable(runtime):
    """The control must not become "always ok" — that is the other failure."""
    result = _check_observe_half(_env(runtime, ":1"))
    assert result.state == HALF_UNAVAILABLE
    assert "absent" in result.detail


@pytest.mark.parametrize("display,socket_name", [
    (":0", "bus_0"), (":1", "bus_1"), (":7", "bus_7"), (":1.0", "bus_1"),
])
def test_the_suffix_follows_the_display_number(runtime, display, socket_name):
    (runtime / "at-spi" / socket_name).write_text("")
    assert _check_observe_half(_env(runtime, display)).state == HALF_OK


# ── THE FALLBACK IS VISIBLE WHEN IT FIRES ───────────────────────────────────

def test_a_guessed_address_says_so(runtime):
    """A fallback that fires SILENTLY is the same defect one level down.

    The guess would be back, just harder to see.
    """
    (runtime / "at-spi" / "bus_1").write_text("")
    result = _check_observe_half(_env(runtime, ":1"))
    assert GUESS_MARK in result.detail, (
        f"the probe fell back to guessing the path and did not say so: "
        f"{result.detail!r}"
    )


def test_an_ASKED_address_carries_no_guess_marker(tmp_path, monkeypatch):
    sock = tmp_path / "answered"
    sock.write_text("")
    monkeypatch.setattr(driver, "_ask_the_bus", lambda env: str(sock))
    result = _check_observe_half({"XDG_RUNTIME_DIR": str(tmp_path)})
    assert result.state == HALF_OK
    assert GUESS_MARK not in (result.detail or ""), (
        "an address obtained from the bus was labelled a guess"
    )


def test_no_bus_and_no_runtime_dir_is_unknown_not_unavailable(monkeypatch):
    """Nothing to ask and nothing to probe = cannot tell. Not "absent"."""
    monkeypatch.setattr(driver, "_ask_the_bus", lambda env: None)
    result = _check_observe_half({})
    assert result.state == HALF_UNKNOWN
    assert "not established" in result.detail


# ── PARSING WHAT THE BUS RETURNS ────────────────────────────────────────────

def test_the_address_is_extracted_from_the_gdbus_tuple(monkeypatch, tmp_path):
    """gdbus prints ('unix:path=/run/user/1000/at-spi/bus_1,guid=...',)."""
    import subprocess

    sock = tmp_path / "bus_1"
    sock.write_text("")
    payload = f"('unix:path={sock},guid=b490272f1643',)\n"

    class _P:
        returncode = 0
        stdout = payload

    monkeypatch.setattr(shutil, "which", lambda n: "/usr/bin/gdbus")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _P())
    got = _ask_the_bus({"DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus"})
    assert got == str(sock), f"the guid tail was not stripped: {got!r}"


def test_a_failing_gdbus_falls_back_rather_than_raising(monkeypatch, tmp_path):
    import subprocess

    def boom(*a, **k):
        raise OSError("no gdbus here")

    monkeypatch.setattr(shutil, "which", lambda n: "/usr/bin/gdbus")
    monkeypatch.setattr(subprocess, "run", boom)
    assert _ask_the_bus({"DBUS_SESSION_BUS_ADDRESS": "x"}) is None


def test_no_session_bus_means_do_not_even_try(monkeypatch):
    called = []
    monkeypatch.setattr(shutil, "which", lambda n: called.append(n) or "/usr/bin/gdbus")
    assert _ask_the_bus({}) is None
    assert not called, "the probe shelled out with no session bus to talk to"


# ── THE NON-HERMETIC HALF: confirm the CONSTANT against the real system ─────

@pytest.mark.skipif(
    not os.environ.get("DBUS_SESSION_BUS_ADDRESS") or shutil.which("gdbus") is None,
    reason="no session bus or no gdbus — this confirmation needs a real one",
)
def test_the_real_session_bus_agrees_with_the_probe():
    """⚠ THE ONE TEST HERE THAT IS DELIBERATELY NOT HERMETIC.

    Every other test in this file builds its own world, which is right for
    BEHAVIOUR and is exactly why the original defect survived: a hermetic test
    and the code can share a wrong constant and agree forever.

    This one asks the REAL session bus and requires the answer to be a socket
    that actually exists. It confirms the constant by something other than
    itself.

    It is a CONFIRMATION, not a gate: it skips where there is no session bus,
    so it contributes nothing in CI and must never be relied on there. That
    is stated rather than hidden — a skipped test claiming coverage is its own
    small lie.
    """
    address = _ask_the_bus(dict(os.environ))
    if address is None:
        pytest.skip("the session bus did not answer GetAddress here")
    assert os.path.exists(address), (
        f"the session bus named {address!r} and nothing is there — the "
        f"address is being parsed wrongly"
    )
    result = _check_observe_half(dict(os.environ))
    assert result.state == HALF_OK, (
        f"the bus answered with a live socket but the probe said "
        f"{result.state}: {result.detail}"
    )
    assert GUESS_MARK not in (result.detail or "")


def test_the_hermetic_tests_do_not_all_share_one_socket_name():
    """Guard the guard: the fixtures must exercise BOTH spellings.

    If every fixture in this file wrote the same filename, this suite would
    reproduce the original defect — agreeing with the code about a name
    neither had checked.
    """
    from pathlib import Path

    text = Path(__file__).read_text()
    assert '"bus_1"' in text and '"bus"' in text, (
        "this file no longer exercises both the suffixed and unsuffixed "
        "socket names; it would share one constant with the code again"
    )
