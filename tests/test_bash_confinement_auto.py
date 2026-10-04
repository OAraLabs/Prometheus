"""``security.bash_confinement: "auto"`` — the shipped default.

The read floor shipped ``"off"``, so a fresh install had no read floor at all,
and ``off|required`` were the only modes because a fallback mode "would be
indistinguishable from a working floor in every log line". ``auto`` is shipped
only because it is NOT indistinguishable:

* where the profile verifies (by outcome, ``preflight``), ``auto`` IS
  ``required`` — and a profile lost after that start fails closed;
* on Linux where it does not verify, the model's shells run without the read
  floor AND that is said out loud: an ERROR at boot naming the fix, ``dark``
  in ``/api/status``, a WARN row in ``oara doctor``, and ``read_floor:
  unavailable`` in each call's metadata;
* on a platform with no AppArmor at all (macOS) it is its own state,
  ``unsupported``: one line, no fix offered, not ``dark``;
* an explicit ``"off"`` stays off. Only an absent key changes meaning.

The canary class needs the profile loaded and runs in CI's
``security-floors`` job, which fails on a skip.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path

import pytest
import yaml

from prometheus.permissions import confinement as C
from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin.bash import BashTool, BashToolInput
from tests.test_bash_confinement import needs_profile, require_floor_subject

ARGV = ["/bin/bash", "-lc", "echo hi"]


@pytest.fixture(autouse=True)
def _fresh():
    C.reset_cache()
    C.reset_write_cache()
    yield
    C.reset_cache()
    C.reset_write_cache()


@pytest.fixture()
def linux(monkeypatch):
    """A platform where AppArmor can exist."""
    monkeypatch.setattr(C, "apparmor_possible", lambda: True, raising=False)


@pytest.fixture()
def no_apparmor_platform(monkeypatch):
    """macOS: AppArmor cannot exist, so there is nothing to probe."""
    monkeypatch.setattr(C, "apparmor_possible", lambda: False, raising=False)

    def _never(*a, **k):
        raise AssertionError("probed AppArmor on a platform that has none")

    monkeypatch.setattr(C, "preflight", _never)


@pytest.fixture()
def verified(monkeypatch):
    monkeypatch.setattr(
        C, "preflight", lambda *a, **k: (True, "prometheus-bash (enforce)"))


@pytest.fixture()
def unverified(monkeypatch):
    monkeypatch.setattr(
        C, "preflight",
        lambda *a, **k: (False, "profile 'prometheus-bash' does not exist (test)"))


def _floors(read: str):
    return C.apply_floors(ARGV, read_mode=read, write_mode="off",
                          writable=(), cwd=None)


def _doctor_read_row(config: dict):
    from prometheus.cli.doctor import check_bash_floors

    return {c.name: c for c in check_bash_floors(config)}["Bash read floor"]


# --------------------------------------------------------------------------- #
# The default
# --------------------------------------------------------------------------- #


class TestTheShippedDefault:
    def test_auto_is_a_mode(self):
        assert C.normalise_mode("auto") == "auto"
        assert C.normalise_mode("AUTO") == "auto"

    def test_the_template_ships_auto(self):
        from prometheus.config.template import load_template

        assert load_template()["security"]["bash_confinement"] == "auto"

    def test_an_absent_key_means_auto(self):
        from prometheus.security.shell_floor import ShellFloor

        assert ShellFloor.from_security_config({}).read_mode == "auto"

    def test_the_registry_builds_bash_with_auto_when_the_key_is_absent(self):
        from prometheus.__main__ import create_tool_registry

        bash = create_tool_registry({}).get("bash")
        assert bash is not None
        assert bash._confinement == "auto"

    def test_explicit_off_stays_off_and_never_probes(self, monkeypatch):
        from prometheus.security.shell_floor import ShellFloor

        assert ShellFloor.from_security_config(
            {"bash_confinement": "off"}).read_mode == "off"

        def _never(*a, **k):
            raise AssertionError("an explicit off probed the profile")

        monkeypatch.setattr(C, "preflight", _never)
        res = _floors("off")
        assert res.refusal is None
        assert list(res.argv) == ARGV


# --------------------------------------------------------------------------- #
# Verified: auto is required
# --------------------------------------------------------------------------- #


class TestVerifiedAutoIsRequired:
    def test_the_shell_is_wrapped(self, linux, verified):
        res = _floors("auto")
        assert res.refusal is None
        assert res.argv[0].endswith("aa-exec"), res.argv
        assert res.read_floor == "active"

    def test_a_profile_lost_after_a_verified_start_fails_closed(
        self, linux, monkeypatch,
    ):
        """The verified result is cached for the process, so the shell keeps
        going through aa-exec — which refuses to run without the profile.
        auto never quietly drops to an unwrapped shell after a verified start."""
        answers = iter([(True, "prometheus-bash (enforce)"),
                        (False, "profile removed")])
        monkeypatch.setattr(C, "_probe_label", lambda profile: next(answers))
        assert _floors("auto").argv[0].endswith("aa-exec")
        assert _floors("auto").argv[0].endswith("aa-exec")


# --------------------------------------------------------------------------- #
# Unverified on Linux: runs, and says so in four places
# --------------------------------------------------------------------------- #


class TestUnverifiedAutoRunsAndSaysSo:
    def test_it_runs_unwrapped_rather_than_refusing(self, linux, unverified):
        res = _floors("auto")
        assert res.refusal is None
        assert list(res.argv) == ARGV
        assert res.read_floor == "unavailable"

    def test_each_call_carries_it_in_its_metadata(self, linux, unverified, tmp_path):
        res = asyncio.run(BashTool(confinement="auto").execute(
            BashToolInput(command="echo hi"), ToolExecutionContext(cwd=tmp_path)))
        assert not res.is_error, res.output
        assert res.metadata.get("read_floor") == "unavailable", res.metadata

    def test_status_reports_it_dark(self, linux, unverified):
        rep = C.floor_report(read_mode="auto", write_mode="off")
        assert rep["bash_read_floor"]["state"] == C.STATE_DARK
        assert rep["dark"] is True

    def test_doctor_warns_and_names_the_fix(self, linux, unverified):
        row = _doctor_read_row({"security": {}})
        assert row.status == "warning", row
        assert "apparmor_parser" in (row.fix or "")

    def test_boot_logs_an_error_naming_the_fix(self, linux, unverified, caplog):
        from prometheus.security.shell_floor import ShellFloor, announce

        with caplog.at_level(logging.WARNING):
            announce(ShellFloor(read_mode="auto", write_mode="off"))
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, caplog.text
        assert any("apparmor_parser" in r.getMessage() for r in errors)
        assert any("NOT in force" in r.getMessage() for r in errors)


# --------------------------------------------------------------------------- #
# No AppArmor on this platform at all
# --------------------------------------------------------------------------- #


class TestAPlatformWithoutAppArmor:
    def test_unsupported_is_its_own_state_and_nothing_is_probed(
        self, no_apparmor_platform,
    ):
        res = _floors("auto")
        assert res.refusal is None
        assert list(res.argv) == ARGV
        assert res.read_floor == "unsupported"

    def test_status_says_unsupported_not_dark(self, no_apparmor_platform):
        rep = C.floor_report(read_mode="auto", write_mode="off")
        assert rep["bash_read_floor"]["state"] == "unsupported"
        assert rep["dark"] is False

    def test_doctor_says_so_without_offering_an_apparmor_fix(
        self, no_apparmor_platform,
    ):
        row = _doctor_read_row({"security": {}})
        assert "unsupported" in row.message
        assert "apparmor_parser" not in (row.fix or "")

    def test_boot_logs_one_warning_and_no_error(self, no_apparmor_platform, caplog):
        from prometheus.security.shell_floor import ShellFloor, announce

        with caplog.at_level(logging.INFO):
            announce(ShellFloor(read_mode="auto", write_mode="off"))
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert [r for r in caplog.records if r.levelno == logging.WARNING]


# --------------------------------------------------------------------------- #
# Where the profile is loaded, the shipped default bites
# --------------------------------------------------------------------------- #


@pytest.fixture()
def canary(tmp_path):
    ssh = require_floor_subject(Path.home() / ".ssh")
    value = f"CANARY-{uuid.uuid4().hex}"
    path = ssh / f"prometheus-auto-canary-{uuid.uuid4().hex[:8]}"
    path.write_text(value + "\n")
    (tmp_path / "where.txt").write_text(str(path))
    try:
        yield value
    finally:
        path.unlink(missing_ok=True)


@needs_profile
class TestTheShippedDefaultBitesWhereTheProfileIsLoaded:
    def test_the_bash_tool_cannot_read_the_canary(self, tmp_path, canary):
        res = asyncio.run(BashTool(confinement="auto").execute(
            BashToolInput(command='echo RAN; cat "$(cat where.txt)"'),
            ToolExecutionContext(cwd=tmp_path)))
        assert "RAN" in res.output
        assert canary not in res.output, "auto READ the canary with the profile loaded"
        assert "Permission denied" in res.output
        assert res.metadata.get("read_floor") == "active"

    @pytest.mark.asyncio
    async def test_a_config_without_the_key_floors_a_model_task(
        self, tmp_path, canary,
    ):
        """No bash_confinement key at all — the fresh-install case."""
        from tests.test_model_shell_floor import _model_task, _output

        cfg_dir = Path(os.environ["PROMETHEUS_CONFIG_DIR"])
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "prometheus.yaml").write_text(
            yaml.safe_dump({"security": {"bash_write_confinement": "off"}}))

        _, task = await _model_task(
            tmp_path, command='echo RAN; cat "$(cat where.txt)"')
        out = _output(task)
        assert "RAN" in out, task
        assert canary not in out
        assert "Permission denied" in out


def test_this_file_runs_in_the_security_floors_job():
    ci = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    assert "tests/test_bash_confinement_auto.py" in ci.read_text()
