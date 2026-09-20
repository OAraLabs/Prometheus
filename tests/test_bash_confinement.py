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
# THE TWO CONTROLS — a precondition guard and an errno assertion
#
# These are separate on purpose and they fail differently.
#
#   PRECONDITION GUARD -> ERROR. "This test did not run", naming which
#   precondition was missing. Never a pass and never a FAIL: a test that
#   cannot reach its subject has measured nothing, and reporting that as
#   either verdict is a lie in one direction or the other. Same shape as
#   ``computer/driver.check_preconditions`` — refuse loudly, do not degrade.
#
#   ERRNO ASSERTION -> FAIL. Reached only once the subject exists. Asserts
#   the refusal SPECIFICALLY: EACCES from the profile passes, ENOENT fails.
#
# WHY BOTH, AND WHY THIS IS NOT PEDANTRY
# ---------------------------------------
# ``@{HOME}`` expands to ``@{HOMEDIRS}/*/`` and AppArmor's ``*`` does NOT
# cross ``/`` — so the denied set is exactly ``/home/<one-component>/.ssh``.
# A test home one level deeper (``~/.verify-home``) is OUTSIDE the profile
# entirely, and a subject that does not exist there fails with ENOENT. A test
# asserting only "the command errored" cannot tell that from a refusal, so it
# passes while proving nothing.
#
# That is not hypothetical. ``test_gnupg_is_refused`` and
# ``test_config_env_pattern_is_refused`` asserted exactly that and had never
# once exercised their subject — on this box OR in CI, where nothing creates
# ``$HOME/.gnupg`` either — since fb73b28 (PR #237, 2026-08-16).
#
# The fix is NOT to widen the profile to cover a test home. Making a security
# guard broader so a test can pass is the inversion of the fix.
# --------------------------------------------------------------------------- #

#: Parsed from the profile's own tunable rather than hardcoded — the naming
#: scheme is the thing under test, so reading it from somewhere else is how the
#: shared-wrong-constant defect gets rebuilt.
_TUNABLE_FILES = (
    "/etc/apparmor.d/tunables/home",
    "/etc/apparmor.d/tunables/home.d/site.local",
)


def _apparmor_homedirs() -> list[str]:
    """The directories ``@{HOMEDIRS}`` names. Empty when it cannot be read."""
    for path in _TUNABLE_FILES:
        try:
            for line in Path(path).read_text().splitlines():
                line = line.strip()
                if line.startswith("@{HOMEDIRS}="):
                    return [
                        d.rstrip("/") or "/"
                        for d in line.split("=", 1)[1].split()
                    ]
        except OSError:
            continue
    return []


def _home_inside_apparmor_home() -> tuple[bool, str]:
    """Is the effective ``HOME`` inside ``@{HOME}``'s ONE-level glob?

    ``@{HOME}=@{HOMEDIRS}/*/ /root/``. A single ``*`` in AppArmor matches one
    path component, so ``$HOME`` qualifies only when its PARENT is a homedir
    (or when it is ``/root``).
    """
    home = Path.home().resolve()
    if home == Path("/root"):
        return True, ""
    homedirs = _apparmor_homedirs()
    if not homedirs:
        return False, (
            "could not read @{HOMEDIRS} from the AppArmor tunables "
            f"({', '.join(_TUNABLE_FILES)}) — the profile's own definition of "
            "which paths it guards is unavailable, so this test cannot know "
            "whether its subject is inside it"
        )
    if str(home.parent) in homedirs:
        return True, ""
    return False, (
        f"HOME={home} is NOT inside @{{HOME}} (={'/*/ '.join(homedirs)}/*/ or "
        f"/root/). AppArmor's '*' matches ONE path component, so a home nested "
        f"below a real home is outside the profile's denied set and every "
        f"subject under it fails with ENOENT instead of being refused. "
        f"Re-run with a HOME whose parent is one of: {homedirs}"
    )


def require_floor_subject(subject: Path) -> Path:
    """Guard: ERROR unless this test can actually reach its subject.

    Two preconditions, reported separately because they have different
    remedies. Raises rather than skipping: a skip is a quiet "not measured"
    that accumulates unread, and the whole point here is that a floor claim
    nobody can see is worse than a red one.
    """
    inside, why = _home_inside_apparmor_home()
    if not inside:
        raise RuntimeError(
            f"PRECONDITION ABSENT — this test did not run. {why}"
        )
    if not subject.exists():
        raise RuntimeError(
            f"PRECONDITION ABSENT — this test did not run. Its subject "
            f"{subject} does not exist, so any failure would be ENOENT rather "
            f"than a refusal by the profile. Create it (CI does this "
            f"deliberately rather than skipping) and re-run."
        )
    return subject


#: What a refusal by the profile looks like, as opposed to an absent file.
_REFUSED = "Permission denied"
_NOT_FOUND = "No such file or directory"


def assert_refused_not_missing(res, what: str) -> None:
    """The errno assertion. EACCES passes; ENOENT FAILS.

    Copied from ``test_confined_read_under_ssh_is_refused``, which already had
    this shape and is the reason it was the only one of the seven that could
    not silently pass on a missing file.
    """
    assert SENTINEL in res.output, f"{what}: never ran — not containment"
    assert res.is_error, f"{what}: the command SUCCEEDED — the floor leaked"
    assert _NOT_FOUND not in res.output, (
        f"{what}: failed with ENOENT, not a refusal. The subject was absent, "
        f"so this proves nothing about the floor:\n{res.output}"
    )
    assert _REFUSED in res.output, (
        f"{what}: errored without {_REFUSED!r}, so the reason is unknown and "
        f"may not be the profile:\n{res.output}"
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
    def ssh_dir(self):
        """The guarded directory itself — the precondition for every ssh case.

        Does NOT create it. A test that manufactures its own subject inside a
        directory the profile may not cover is measuring the directory it just
        made, not the floor.
        """
        return require_floor_subject(Path.home() / ".ssh")

    @pytest.fixture()
    def decoy(self, ssh_dir):
        p = ssh_dir / "aa-pytest-decoy.txt"
        p.write_text("DECOY\n")
        yield p
        p.unlink(missing_ok=True)

    def test_the_process_really_is_confined(self):
        res = _confined("cat /proc/self/attr/current")
        assert SENTINEL in res.output, "the confined process never ran"
        assert "prometheus-bash" in res.output

    def test_confined_read_under_ssh_is_refused(self, decoy):
        res = _confined(f"cat {decoy}")
        assert_refused_not_missing(res, "read under ~/.ssh")
        assert "DECOY" not in res.output, "the read landed — the floor leaked"

    def test_confined_write_under_ssh_is_refused(self, ssh_dir):
        target = ssh_dir / "aa-pytest-write.txt"
        target.unlink(missing_ok=True)
        try:
            res = _confined(f"echo LEAK > {target}")
            assert_refused_not_missing(res, "write under ~/.ssh")
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
        assert_refused_not_missing(res, f"{wrapper} read under ~/.ssh")
        assert "DECOY" not in res.output, "the read landed — the floor leaked"

    @pytest.fixture()
    def gnupg_dir(self):
        return require_floor_subject(Path.home() / ".gnupg")

    @pytest.fixture()
    def config_env_file(self):
        return require_floor_subject(
            Path.home() / ".config" / "prometheus" / "env"
        )

    def test_gnupg_is_refused(self, gnupg_dir):
        res = _confined(f"ls {gnupg_dir}")
        assert_refused_not_missing(res, "ls ~/.gnupg")

    def test_config_env_pattern_is_refused(self, config_env_file):
        res = _confined(f"wc -c < {config_env_file}")
        assert_refused_not_missing(res, "read ~/.config/prometheus/env")


@needs_profile
class TestAdmissionHalf:
    @pytest.fixture()
    def deploy_clone(self):
        """Precondition guard ONLY — this test's subject is the clone, not a
        denied path, so there is no errno to assert. It must still never pass
        while unable to reach what it measures."""
        return require_floor_subject(Path.home() / "prometheus-deploy")

    """A floor that breaks the loop is not a win."""

    def test_git_network_operation_succeeds(self, deploy_clone):
        """A real authenticated git network op, under the profile.

        Was `push --dry-run origin main`, which fails with rc=1 whenever the
        deploy clone is BEHIND origin — so it measured the clone's sync state
        as much as confinement, and went red during an ordinary deploy window
        with nothing wrong. `ls-remote` exercises the same thing that matters
        here (git reaching the network with credentials, unprompted, inside
        the profile) and is independent of local repo state.
        """
        res = _confined(
            f"git -C {deploy_clone} ls-remote origin HEAD "
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


# --------------------------------------------------------------------------- #
# The controls, tested against a built failing state — runs everywhere
# --------------------------------------------------------------------------- #


class _FakeResult:
    """Just enough of a ToolResult to exercise the assertion's discrimination."""

    def __init__(self, output: str, is_error: bool = True) -> None:
        self.output = output
        self.is_error = is_error


class TestTheControlsThemselves:
    """A control nobody has watched fail is a control nobody has tested.

    The precondition guard stops a missing subject before the errno assertion
    is reached, which is correct — and it means the assertion's discrimination
    has to be proven here, against a failing state built on purpose, or it is
    never exercised at all.
    """

    def test_enoent_fails_the_assertion(self):
        """THE WHOLE POINT. This is the shape that was passing for five weeks."""
        res = _FakeResult(
            f"{SENTINEL}\nls: cannot access '/h/x/.gnupg': {_NOT_FOUND}"
        )
        with pytest.raises(AssertionError, match="ENOENT"):
            assert_refused_not_missing(res, "built-on-purpose ENOENT")

    def test_eacces_passes_the_assertion(self):
        res = _FakeResult(
            f"{SENTINEL}\nls: cannot open directory '/h/x/.gnupg': {_REFUSED}"
        )
        assert_refused_not_missing(res, "built-on-purpose EACCES")

    def test_a_successful_command_fails_the_assertion(self):
        """is_error False means the floor leaked, whatever the output says."""
        res = _FakeResult(f"{SENTINEL}\nid_rsa", is_error=False)
        with pytest.raises(AssertionError, match="floor leaked"):
            assert_refused_not_missing(res, "built-on-purpose success")

    def test_an_error_with_no_stated_reason_fails_the_assertion(self):
        """Errored, but not identifiably BY THE PROFILE. Unknown is not pass."""
        res = _FakeResult(f"{SENTINEL}\nsomething went wrong")
        with pytest.raises(AssertionError, match="reason is unknown"):
            assert_refused_not_missing(res, "built-on-purpose vague error")

    def test_a_command_that_never_ran_fails_the_assertion(self):
        res = _FakeResult("nothing at all")
        with pytest.raises(AssertionError, match="never ran"):
            assert_refused_not_missing(res, "built-on-purpose no-run")

    def test_the_guard_names_which_precondition_was_missing(
        self, tmp_path, monkeypatch
    ):
        """ERROR text must say WHICH one, because the remedies differ.

        Builds BOTH halves of the world in tmp_path rather than reading the
        host's HOME. The first draft of this test did read it, and so reported
        the HOME failure under ~/.verify-home and the subject failure under the
        real home — a test whose expectation depends on where it runs, which is
        the entire defect this file is fixing.
        """
        fake_home = tmp_path / "someuser"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(
            "tests.test_bash_confinement._apparmor_homedirs",
            lambda: [str(tmp_path)],
        )
        # HOME is now one level under a "homedir", so the first check passes
        # and the SUBJECT check is the one that must fire.
        with pytest.raises(RuntimeError, match="does not exist"):
            require_floor_subject(fake_home / "definitely-absent")

    def test_the_guard_refuses_when_the_tunable_cannot_be_read(
        self, tmp_path, monkeypatch
    ):
        """Unknown is not OK. If the profile's own definition of what it guards
        is unreadable, the test cannot know whether its subject is inside it —
        so it refuses rather than assuming the favourable answer."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(
            "tests.test_bash_confinement._apparmor_homedirs", lambda: []
        )
        inside, why = _home_inside_apparmor_home()
        assert not inside
        assert "could not read" in why

    def test_the_guard_rejects_a_home_outside_the_profile_glob(self, monkeypatch):
        """A home one level too deep is outside @{HOME} — the original defect.

        Asserted through the real resolver, not a mocked one: the thing under
        test is a naming scheme, and a mock would encode the same assumption
        the code is being checked for.
        """
        nested = Path.home() / ".verify-home-probe"
        monkeypatch.setenv("HOME", str(nested))
        inside, why = _home_inside_apparmor_home()
        assert not inside, (
            f"{nested} was accepted as inside @{{HOME}}, but AppArmor's '*' "
            f"matches ONE component — this is the defect that let three tests "
            f"pass on ENOENT for five weeks"
        )
        assert "ONE path component" in why or "one path component" in why.lower()
