"""``config/display_name.py`` — the one name a Prometheus shows to people who are not yet paired.

Hello, the mDNS record and the operator's Approve prompt all show it, so there is ONE function. What is
pinned: the order it is chosen in (the owner's setting, then the computer's own name, then the host
name), that it is safe to put on a screen (no control, format or line-separator characters: a name that
renders as something else is a spoofing tool) and bounded, that a missing ``scutil`` is a fallback and
not a failure, and that it is never empty.

The computer name is cached: hello is unauthenticated, and a subprocess per request would be a way for an
outsider to spawn processes on this machine.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from prometheus.config import display_name as dn


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    dn.reset_cache()
    yield
    dn.reset_cache()


def _darwin(monkeypatch, computer_name: str | None = "Jennifer's MacBook\n", host: str = "jen-mac.local"):
    """Make this a Mac whose ``scutil`` answers *computer_name* (``None`` = it is not there)."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if computer_name is None:
            raise FileNotFoundError("scutil")
        return SimpleNamespace(returncode=0, stdout=computer_name)

    monkeypatch.setattr(dn.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(dn.platform, "node", lambda: host)
    monkeypatch.setattr(dn.subprocess, "run", fake_run)
    return calls


def test_the_owners_setting_wins_and_nothing_is_spawned(monkeypatch):
    calls = _darwin(monkeypatch)
    assert dn.display_name({"pairing": {"display_name": "Will's Mac mini"}}) == "Will's Mac mini"
    assert calls == []


@pytest.mark.parametrize("value", [None, "", "   ", 7, ["x"]])
def test_an_unusable_setting_falls_through_to_the_computer_name(monkeypatch, value):
    _darwin(monkeypatch)
    assert dn.display_name({"pairing": {"display_name": value}}) == "Jennifer's MacBook"


def test_a_missing_pairing_section_or_config_is_fine(monkeypatch):
    _darwin(monkeypatch)
    assert dn.display_name({}) == "Jennifer's MacBook"
    assert dn.display_name(None) == "Jennifer's MacBook"


def test_the_computer_name_comes_from_scutil_on_a_mac(monkeypatch):
    calls = _darwin(monkeypatch)
    assert dn.display_name({}) == "Jennifer's MacBook"
    assert calls == [["scutil", "--get", "ComputerName"]]


@pytest.mark.parametrize("failure", ["missing", "nonzero", "timeout", "blank"])
def test_a_scutil_that_does_not_help_falls_back_to_the_host_name(monkeypatch, failure):
    monkeypatch.setattr(dn.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(dn.platform, "node", lambda: "jen-mac.local")

    def fake_run(cmd, **kwargs):
        if failure == "missing":
            raise FileNotFoundError("scutil")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, 2)
        if failure == "nonzero":
            return SimpleNamespace(returncode=1, stdout="")
        return SimpleNamespace(returncode=0, stdout="  \n")

    monkeypatch.setattr(dn.subprocess, "run", fake_run)
    assert dn.display_name({}) == "jen-mac.local"


def test_nothing_is_spawned_off_a_mac(monkeypatch):
    calls = _darwin(monkeypatch)
    monkeypatch.setattr(dn.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dn.platform, "node", lambda: "linux-box")
    assert dn.display_name({}) == "linux-box"
    assert calls == []


def test_the_computer_name_is_looked_up_once_not_per_call(monkeypatch):
    calls = _darwin(monkeypatch)
    for _ in range(25):
        dn.display_name({})
    assert len(calls) == 1


def test_a_renamed_computer_is_noticed_after_the_cache_expires(monkeypatch):
    calls = _darwin(monkeypatch, "Old name")
    clock = [100.0]
    monkeypatch.setattr(dn.time, "monotonic", lambda: clock[0])
    assert dn.display_name({}) == "Old name"
    _darwin_answer = "New name"
    monkeypatch.setattr(dn.subprocess, "run", lambda cmd, **kw: SimpleNamespace(returncode=0, stdout=_darwin_answer))
    assert dn.display_name({}) == "Old name"   # still cached
    clock[0] += 3600
    assert dn.display_name({}) == "New name"
    assert len(calls) == 1


@pytest.mark.parametrize("raw, shown", [
    ("Jen\nnifer", "Jennifer"),
    ("Mac\x07\x1b[31mred", "Mac[31mred"),
    ("tab\tsep", "tabsep"),
    ("nul\x00byte", "nulbyte"),
    ("evil‮gnp.exe", "evilgnp.exe"),     # right-to-left override: a format character
    ("zero​width", "zerowidth"),
    ("line sep", "linesep"),
    ("  padded  ", "padded"),
])
def test_it_is_safe_to_put_on_a_screen(monkeypatch, raw, shown):
    _darwin(monkeypatch)
    assert dn.display_name({"pairing": {"display_name": raw}}) == shown


def test_it_is_at_most_64_characters_and_never_ends_in_a_space(monkeypatch):
    _darwin(monkeypatch)
    name = dn.display_name({"pairing": {"display_name": "a" * 63 + " " + "b" * 40}})
    assert len(name) <= 64
    assert name == name.rstrip()
    assert dn.display_name({"pairing": {"display_name": "é" * 100}}) == "é" * 64


def test_it_is_never_empty(monkeypatch):
    monkeypatch.setattr(dn.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dn.platform, "node", lambda: "")
    assert dn.display_name({}) == "Prometheus"
    assert dn.display_name({"pairing": {"display_name": "\n\x07"}}) == "Prometheus"
