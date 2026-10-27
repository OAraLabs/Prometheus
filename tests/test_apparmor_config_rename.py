"""#766: the AppArmor read floor refuses renaming ~/.config, and the boot probe says when a loaded profile is older.

Measured on Linux 7.0 with ``deploy/apparmor/prometheus-bash`` loaded: ``mv ~/.config ~/cfg && cat
~/cfg/prometheus/env`` printed the file, because the env rule names the files and ``~/.config`` itself matched
nothing. ``deny @{HOME}/.config/ w,`` refused both rename routes and changed nothing else measured. The rule lives
in a root-loaded file, so an upgraded daemon can meet the OLD profile; the probe tests the rule by outcome (rename
``~/.config`` onto itself: refused under the rule, a no-op without it) and reports a stale profile at ERROR with
the reload, without dropping the floor it still has.

These run anywhere. The real-profile tests are in ``tests/test_bash_confinement.py`` (CI's security-floors job).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pytest

from prometheus.permissions import confinement as C

PROFILE_FILE = Path(__file__).resolve().parent.parent / "deploy" / "apparmor" / "prometheus-bash"


@pytest.fixture(autouse=True)
def _fresh():
    C.reset_cache()
    yield
    C.reset_cache()


def _active_rules() -> list[str]:
    return [line.split("#", 1)[0].strip() for line in PROFILE_FILE.read_text().splitlines()
            if line.split("#", 1)[0].strip()]


def test_the_shipped_profile_refuses_writes_to_config_itself():
    rules = [re.sub(r"\s+", " ", r) for r in _active_rules()]
    assert "deny @{HOME}/.config/ w," in rules, rules
    # The env rule is still there: the new line adds to the floor, it does not replace it.
    assert "deny @{HOME}/.config/*/*env rwlkmx," in rules


def test_renaming_a_directory_onto_itself_moves_nothing(tmp_path):
    """Why the probe is safe on a real home: allowed, the rename is a no-op."""
    target = tmp_path / ".config"
    (target / "prometheus").mkdir(parents=True)
    (target / "prometheus" / "env").write_text("x\n")
    os.rename(target, target)
    assert (target / "prometheus" / "env").read_text() == "x\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == [".config"]


def _label_ok(monkeypatch, rename: str):
    monkeypatch.setattr(C, "_probe_label", lambda profile: (True, f"{profile} (enforce)"))
    monkeypatch.setattr(C, "_probe_config_rename", lambda profile: rename)
    # The device-registry half is tests/test_devices_db_floor.py's; held current here.
    monkeypatch.setattr(C, "_registry_detail", lambda profile: ("; writing the device registry refused", False))
    monkeypatch.setattr(C, "_trusted_detail", lambda profile: ("; writing trusted state refused", False))


def test_a_current_profile_is_active_and_says_the_rename_is_refused(monkeypatch, caplog):
    _label_ok(monkeypatch, "refused")
    with caplog.at_level(logging.INFO, logger=C.logger.name):
        ok, detail = C.preflight(C.PROFILE, force=True)
    assert ok and "renaming ~/.config refused" in detail
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_a_stale_profile_still_confines_and_is_reported_at_error_with_the_reload(monkeypatch, caplog):
    _label_ok(monkeypatch, "renamed")
    with caplog.at_level(logging.INFO, logger=C.logger.name):
        ok, detail = C.preflight(C.PROFILE, force=True)
    assert ok, "a stale profile must keep the floor it has, not drop to unconfined (auto) or refuse (required)"
    assert "STALE PROFILE" in detail and C.RELOAD_COMMAND in detail
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors and C.RELOAD_COMMAND in errors[0], errors


def test_a_stale_profile_still_wraps_every_shell(monkeypatch):
    _label_ok(monkeypatch, "renamed")
    monkeypatch.setattr(C, "read_floor_mechanism", lambda: C.MECHANISM_APPARMOR)
    for mode in ("auto", "required"):
        C.reset_cache()
        result = C.apply_floors(["/bin/bash", "-c", "ls"], read_mode=mode, write_mode="off", writable=[], cwd=None)
        assert result.refusal is None and result.read_floor == C.READ_ACTIVE, (mode, result)
        assert C.PROFILE in result.argv, (mode, result.argv)


@pytest.mark.parametrize("rename, words", [("absent", "not measured"), ("unknown", "could not run")])
def test_a_rename_check_that_cannot_measure_says_so_without_failing(monkeypatch, rename, words):
    _label_ok(monkeypatch, rename)
    ok, detail = C.preflight(C.PROFILE, force=True)
    assert ok and words in detail, detail


def test_the_probe_is_not_run_when_the_profile_did_not_confine(monkeypatch):
    monkeypatch.setattr(C, "_probe_label", lambda profile: (False, "the confined process never ran"))
    monkeypatch.setattr(C, "_probe_config_rename", lambda profile: pytest.fail("probed an unconfined process"))
    ok, detail = C.preflight(C.PROFILE, force=True)
    assert not ok and detail == "the confined process never ran"
