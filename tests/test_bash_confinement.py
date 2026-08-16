"""bash runs behind a kernel floor, or it refuses to run.

The permission gate cannot reach the paths inside a command string, so
``cat ~/.ssh/id_*`` is ALLOW at both origins and always has been. This suite
covers the control that closes it and — just as important — the refusal that
must happen when that control is unavailable.

TWO CLASSES OF TEST HERE, and the difference matters when reading a green run:

* The refusal/admission-wiring tests run everywhere. They need no profile.
* The tests that prove the floor actually BITES need the ``prometheus-bash``
  AppArmor profile loaded, which is a root action on the host. They SKIP
  where it is absent. **A skipped test is not a passing test** — CI green
  says the wiring refuses correctly, not that any key is protected.

Every harness that pipes sets ``pipefail``. Without it ``cat denied | wc -c``
exits 0 because the status comes from ``wc``, and a refused read reads as a
success — which is exactly how two floor patterns first looked like leaks.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from prometheus.permissions import confinement as C
from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin.bash import BashTool, BashToolInput

SENTINEL = "CONFINE_TEST_INSIDE_RAN"


@pytest.fixture(autouse=True)
def _clear_preflight():
    C.reset_cache()
    yield
    C.reset_cache()


def _run(tool: BashTool, command: str, cwd: Path | None = None):
    ctx = ToolExecutionContext(cwd=cwd or Path("/tmp"))
    return asyncio.run(tool.execute(BashToolInput(command=command), ctx))


def _confined(command: str, **kw):
    """Run through the real profile, with a sentinel and pipefail."""
    tool = BashTool(confinement="required", **kw)
    return _run(tool, f"set -o pipefail; echo {SENTINEL}; {command}")


def _profile_loaded() -> bool:
    ok, _ = C.preflight(C.PROFILE, force=True)
    C.reset_cache()
    return ok


needs_profile = pytest.mark.skipif(
    not _profile_loaded(),
    reason=(
        "AppArmor profile 'prometheus-bash' is not loaded (root action). "
        "SKIPPED IS NOT PASSED: the floor is unproven in this environment."
    ),
)


# --------------------------------------------------------------------------- #
# Fail loud — runs everywhere, needs no profile
# --------------------------------------------------------------------------- #


class TestRefusesRatherThanRunningUnconfined:
    def test_absent_profile_refuses_and_does_not_run(self):
        res = _run(
            BashTool(confinement="required", confinement_profile="no-such-profile-xyz"),
            "echo I_RAN_UNCONFINED",
        )
        assert res.is_error
        assert "I_RAN_UNCONFINED" not in res.output, (
            "the command executed despite confinement being unavailable — "
            "this is the silent-unconfined failure the mode exists to prevent"
        )
        assert "bash REFUSED" in res.output

    def test_absent_profile_message_names_the_reason_and_the_fix(self):
        res = _run(
            BashTool(confinement="required", confinement_profile="no-such-profile-xyz"),
            "echo hi",
        )
        assert "does not exist" in res.output or "never ran" in res.output
        assert "apparmor_parser" in res.output
        assert "bash_confinement" in res.output

    def test_absent_aa_exec_refuses_and_does_not_run(self, monkeypatch):
        monkeypatch.setattr(
            C.shutil, "which",
            lambda name, *a, **k: None if name == "aa-exec" else shutil.which(name),
        )
        res = _run(BashTool(confinement="required"), "echo I_RAN_UNCONFINED")
        assert res.is_error
        assert "I_RAN_UNCONFINED" not in res.output
        assert "aa-exec is not installed" in res.output

    def test_transition_that_does_not_happen_is_refused(self, monkeypatch):
        """aa-exec exiting 0 is NOT evidence of confinement.

        The preflight reads the label the confined process reports for
        itself. A stub that exits 0 while leaving the process unconfined must
        still be refused — otherwise the check is measuring the wrong thing.
        """
        class _Fake:
            returncode = 0
            stdout = "unconfined\n"
            stderr = ""

        monkeypatch.setattr(C.subprocess, "run", lambda *a, **k: _Fake())
        ok, detail = C.preflight("prometheus-bash", force=True)
        assert ok is False
        assert "transition did not happen" in detail
        assert "unconfined" in detail


class TestModeParsing:
    @pytest.mark.parametrize("value", ["off", "OFF", "", None, "false", "no"])
    def test_values_that_mean_off(self, value):
        assert C.normalise_mode(value) == C.MODE_OFF

    @pytest.mark.parametrize("value", ["required", "REQUIRED", "enforce", "true"])
    def test_values_that_mean_required(self, value):
        assert C.normalise_mode(value) == C.MODE_REQUIRED

    def test_unknown_value_falls_to_off_loudly(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger=C.logger.name):
            assert C.normalise_mode("sort-of") == C.MODE_OFF
        assert "not one of" in caplog.text
        assert "NOT in force" in caplog.text

    def test_off_does_not_wrap_the_argv(self):
        """The other direction: mode off must leave bash exactly as it was."""
        res = _run(BashTool(confinement="off"), f"echo {SENTINEL}")
        assert not res.is_error
        assert SENTINEL in res.output


class TestWrapping:
    def test_wrap_argv_prefixes_the_transition(self):
        argv = C.wrap_argv(["/bin/bash", "-lc", "echo hi"], "p")
        assert argv[1:4] == ["-p", "p", "--"]
        assert argv[4:] == ["/bin/bash", "-lc", "echo hi"]
        assert argv[0].endswith("aa-exec")


# --------------------------------------------------------------------------- #
# The floor actually bites — needs the profile loaded
# --------------------------------------------------------------------------- #


@needs_profile
class TestFloorHolds:
    @pytest.fixture()
    def decoy(self):
        p = Path.home() / ".ssh" / "aa-pytest-decoy.txt"
        p.write_text("DECOY\n")
        yield p
        p.unlink(missing_ok=True)

    def test_the_process_really_is_confined(self):
        res = _confined("cat /proc/self/attr/current")
        assert SENTINEL in res.output, "the confined process never ran"
        assert "prometheus-bash" in res.output

    def test_confined_read_under_ssh_is_refused(self, decoy):
        res = _confined(f"cat {decoy}")
        assert SENTINEL in res.output, "never ran — not containment"
        assert res.is_error
        assert "Permission denied" in res.output
        assert "DECOY" not in res.output

    def test_confined_write_under_ssh_is_refused(self):
        target = Path.home() / ".ssh" / "aa-pytest-write.txt"
        target.unlink(missing_ok=True)
        try:
            res = _confined(f"echo LEAK > {target}")
            assert SENTINEL in res.output
            assert res.is_error
            assert not target.exists(), "the write landed — the floor leaked"
        finally:
            target.unlink(missing_ok=True)

    @pytest.mark.parametrize("wrapper", ["sh -c", "env sh -c"])
    def test_wrapper_does_not_escape_the_profile(self, decoy, wrapper):
        """ix inheritance: children stay confined.

        These are the wrappers that defeat any command-string check, so they
        are the ones that matter most.
        """
        res = _confined(f"{wrapper} 'cat {decoy}'")
        assert SENTINEL in res.output
        assert res.is_error
        assert "DECOY" not in res.output

    def test_gnupg_is_refused(self):
        res = _confined("ls ~/.gnupg")
        assert SENTINEL in res.output
        assert res.is_error

    def test_config_env_pattern_is_refused(self):
        res = _confined("wc -c < ~/.config/prometheus/env")
        assert SENTINEL in res.output
        assert res.is_error


@needs_profile
class TestAdmissionHalf:
    """A floor that breaks the loop is not a win."""

    def test_git_network_operation_succeeds(self):
        """A real authenticated git network op, under the profile.

        Was `push --dry-run origin main`, which fails with rc=1 whenever the
        deploy clone is BEHIND origin — so it measured the clone's sync state
        as much as confinement, and went red during an ordinary deploy window
        with nothing wrong. `ls-remote` exercises the same thing that matters
        here (git reaching the network with credentials, unprompted, inside
        the profile) and is independent of local repo state.
        """
        res = _confined(
            "git -C ~/prometheus-deploy ls-remote origin HEAD "
            ">/dev/null 2>&1 && echo GIT_NET_OK")
        assert "GIT_NET_OK" in res.output
        assert not res.is_error

    def test_package_install_succeeds(self, tmp_path):
        venv = tmp_path / "v"
        res = _confined(
            f"uv venv {venv} >/dev/null 2>&1 && "
            f"uv pip install --python {venv}/bin/python idna >/dev/null 2>&1 && "
            "echo INSTALL_OK")
        assert "INSTALL_OK" in res.output
        assert not res.is_error

    def test_ordinary_repo_work_succeeds(self, tmp_path):
        res = _confined(
            f"cd {tmp_path} && mkdir -p a/b && echo x > a/b/f.txt && "
            "cat a/b/f.txt && rm -rf a && echo REPO_OK")
        assert "REPO_OK" in res.output
        assert not res.is_error

    def test_a_pipeline_still_reports_its_own_failure(self, tmp_path):
        """pipefail sanity: the harness must not mask a failing stage."""
        res = _confined(f"cat {tmp_path}/nope | wc -c")
        assert res.is_error, "pipefail is not in force; refusals would read as passes"
