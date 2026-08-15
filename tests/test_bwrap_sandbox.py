"""BwrapSandbox — kernel namespace containment for coding runs.

Two tiers, deliberately separated:

  TestArgvConstruction / TestConstructionFailureDetection / TestSelfCheck
      Pure-Python or single-invocation checks that hold regardless of whether
      this host can actually complete a bwrap run. These run unconditionally.

  TestAcceptance / TestRunBehaviour
      The actual containment claims — the shell-redirect escape must fail,
      timeouts must kill the tree, env must scrub, denied_paths must hold
      even from a shell. Gated on ``BwrapSandbox.self_check().ok``, skipped
      LOUDLY (with the self-check's own detail string, not a bare "skipped")
      when this host cannot run a namespaced process at all — see
      audits/20260813T193000Z-bwrap-sandbox-host-finding.md and the
      BwrapSandbox class docstring. A silent, generic skip here would be
      exactly Standing-Principles §2b's "a check that answers a different
      question" — "bwrap is installed" is not "bwrap contains anything".
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from prometheus.coding.sandbox import (
    BwrapSandbox,
    BwrapSelfCheck,
    SandboxConstructionError,
    SandboxViolation,
)

_CHECK: BwrapSelfCheck = BwrapSandbox.self_check() if BwrapSandbox.is_available() else None

_SKIP_REASON = (
    "bwrap not installed"
    if _CHECK is None
    else f"bwrap present but cannot run a namespaced process here: {_CHECK.detail}"
)

requires_working_bwrap = pytest.mark.skipif(
    _CHECK is None or not _CHECK.ok, reason=_SKIP_REASON
)


@pytest.fixture
def box(tmp_path: Path) -> BwrapSandbox:
    root = tmp_path / "jail"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("x = 1\n")
    # Match whichever network policy this host's self-check proved works, so
    # the behavioural tests exercise a REAL working configuration rather than
    # always requesting the default and skipping everywhere isolated-only
    # hosts differ from networked-only ones.
    isolate = bool(_CHECK and _CHECK.isolated_ok)
    return BwrapSandbox(root=root, isolate_network=isolate)


# --------------------------------------------------------------------------- #
# Always run: these hold regardless of whether bwrap can complete here
# --------------------------------------------------------------------------- #


class TestAvailability:
    def test_is_available_checks_path_only(self):
        # True on any box with the package installed — this repo's dev/CI
        # hosts and the deployment host all have it (confirmed 2026-08-13).
        assert BwrapSandbox.is_available() is True

    def test_missing_binary_raises_at_construction(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "prometheus.coding.sandbox.shutil.which", lambda *_: None
        )
        with pytest.raises(RuntimeError, match="bubblewrap"):
            BwrapSandbox(root=tmp_path)


class TestSelfCheck:
    """The self-check IS the finding from the 2026-08-13 host investigation,
    turned into reusable code. These assert its SHAPE, not a specific verdict
    — the verdict is allowed to differ (and improve) once the host is fixed.
    """

    def test_returns_both_network_policy_results(self):
        r = BwrapSandbox.self_check()
        assert isinstance(r.isolated_ok, bool)
        assert isinstance(r.networked_ok, bool)
        assert isinstance(r.ok, bool)
        # ok is true iff at least one policy actually worked.
        assert r.ok == (r.isolated_ok or r.networked_ok)

    def test_absent_binary_reports_cleanly(self, monkeypatch):
        monkeypatch.setattr(
            "prometheus.coding.sandbox.shutil.which", lambda *_: None
        )
        r = BwrapSandbox.self_check()
        assert r == BwrapSelfCheck(False, "bwrap not found on PATH", False, False)

    def test_detail_is_never_empty(self):
        # Whatever the verdict, there must be a human-readable reason —
        # this is the string a skip message and a diagnosis both lean on.
        assert BwrapSandbox.self_check().detail


class TestArgvConstruction:
    """Pure argument-list construction — no subprocess involved."""

    def test_default_isolates_network(self, tmp_path: Path):
        sb = BwrapSandbox(root=tmp_path)
        argv = sb._bwrap_argv("true", "S")
        assert "--unshare-net" in argv

    def test_isolate_network_false_omits_the_flag(self, tmp_path: Path):
        sb = BwrapSandbox(root=tmp_path, isolate_network=False)
        argv = sb._bwrap_argv("true", "S")
        assert "--unshare-net" not in argv

    def test_root_is_bound_read_write(self, tmp_path: Path):
        sb = BwrapSandbox(root=tmp_path)
        argv = sb._bwrap_argv("true", "S")
        i = argv.index("--bind")
        assert argv[i + 1] == argv[i + 2] == str(sb.root)

    def test_denied_path_inside_root_shadowed_read_only_after_root_bind(
        self, tmp_path: Path
    ):
        """bwrap applies binds in argv order; the RO shadow of a denied path
        must come AFTER the RW root bind, or the root bind would win instead.
        """
        root = tmp_path / "jail"
        root.mkdir()
        denied = root / "secrets"
        sb = BwrapSandbox(root=root, denied_paths=(denied,))
        argv = sb._bwrap_argv("true", "S")
        root_bind_idx = argv.index("--bind")
        denied_idx = next(
            i
            for i, a in enumerate(argv)
            if a == "--ro-bind-try" and argv[i + 1] == str(denied)
        )
        assert denied_idx > root_bind_idx

    def test_denied_path_outside_root_not_bound_at_all(self, tmp_path: Path):
        """A denied path that is not under root has no reason to appear in
        the argv — it is simply absent from the namespace's mount table
        already (assuming it is not also on the RO base-dir list)."""
        root = tmp_path / "jail"
        root.mkdir()
        outside_denied = tmp_path / "elsewhere"
        outside_denied.mkdir()
        sb = BwrapSandbox(root=root, denied_paths=(outside_denied,))
        argv = sb._bwrap_argv("true", "S")
        assert str(outside_denied) not in argv

    def test_sentinel_is_printed_before_the_real_command(self, tmp_path: Path):
        sb = BwrapSandbox(root=tmp_path)
        argv = sb._bwrap_argv("echo REAL", "MARK")
        inner = argv[-1]
        assert inner.index("MARK") < inner.index("REAL")

    def test_env_is_cleared_then_set_from_the_scrub_allowlist(self, tmp_path: Path):
        sb = BwrapSandbox(root=tmp_path)
        argv = sb._bwrap_argv("true", "S")
        assert "--clearenv" in argv
        assert "--setenv" in argv
        # PYTHONUNBUFFERED is always injected by _scrubbed_env (inherited
        # from ProcessSandbox unchanged) — confirms the same scrub path runs.
        setenv_pairs = {
            argv[i + 1]: argv[i + 2]
            for i, a in enumerate(argv)
            if a == "--setenv"
        }
        assert setenv_pairs.get("PYTHONUNBUFFERED") == "1"

    def test_base_ro_dirs_present_as_try_binds(self, tmp_path: Path):
        sb = BwrapSandbox(root=tmp_path)
        argv = sb._bwrap_argv("true", "S")
        assert "--ro-bind-try" in argv
        assert "/usr" in argv  # at least the one base dir every distro has

    def test_inherits_resolve_unchanged(self, tmp_path: Path):
        """resolve() is pure Python path logic — BwrapSandbox must not
        reimplement or diverge from ProcessSandbox's version."""
        root = tmp_path / "jail"
        root.mkdir()
        sb = BwrapSandbox(root=root)
        with pytest.raises(SandboxViolation, match="escapes the sandbox"):
            sb.resolve("/etc/passwd")


class TestConstructionFailureDetection:
    """The sentinel mechanism must distinguish 'bwrap could not start the
    process' from 'the process ran and exited nonzero' — conflating them
    would misreport bwrap's own exit code as the command's."""

    def test_bwrap_setup_failure_raises_not_returns(self, tmp_path: Path):
        """Force a guaranteed bwrap-level failure (nonexistent bind source
        with a HARD --bind, which is not -try) and confirm it raises
        SandboxConstructionError naming bwrap, never a SandboxResult."""
        sb = BwrapSandbox(root=tmp_path)

        async def go():
            return await sb.run("echo should-never-run")

        # Monkeypatch-free: point a REQUIRED (non -try) bind at a path that
        # cannot exist, guaranteeing bwrap itself fails before exec.
        orig = sb._bwrap_argv

        def broken_argv(command, sentinel):
            argv = orig(command, sentinel)
            argv.insert(1, "--ro-bind")
            argv.insert(2, "/this/path/does/not/exist/__probe__")
            argv.insert(3, "/this/path/does/not/exist/__probe__")
            return argv

        sb._bwrap_argv = broken_argv  # type: ignore[method-assign]
        with pytest.raises(SandboxConstructionError, match="bwrap"):
            asyncio.run(go())


# --------------------------------------------------------------------------- #
# Gated on a WORKING bwrap: the actual containment claims
# --------------------------------------------------------------------------- #


@requires_working_bwrap
class TestAcceptance:
    """The acceptance test the 2026-08-13 audit specified verbatim."""

    def test_shell_redirect_to_outside_path_fails(
        self, box: BwrapSandbox, tmp_path: Path
    ):
        outside = tmp_path / "escaped.txt"
        outside.unlink(missing_ok=True)
        r = asyncio.run(box.run(f"echo pwned > {outside}"))
        assert not outside.exists(), (
            "the redirect target must not exist on the host — this is the "
            "exact failure ProcessSandbox has (rc=0, file lands outside)"
        )
        # The write attempt inside the namespace fails at open(2) — a
        # nonzero shell exit, not a crash, not silently swallowed.
        assert r.exit_code != 0

    def test_absolute_path_read_of_a_real_host_secret_style_file_fails(
        self, box: BwrapSandbox, tmp_path: Path
    ):
        secret = tmp_path / "not-in-the-jail.secret"
        secret.write_text("sk-shouldnotleak")
        r = asyncio.run(box.run(f"cat {secret}"))
        assert "sk-shouldnotleak" not in r.output

    def test_ordinary_work_inside_root_still_succeeds(self, box: BwrapSandbox):
        """The containment must not be so tight it breaks legitimate use —
        reading/writing inside the jail, and normal exit-code passthrough."""
        r = asyncio.run(box.run("cat src/app.py && echo INSIDE_OK"))
        assert r.exit_code == 0
        assert "x = 1" in r.output
        assert "INSIDE_OK" in r.output


@requires_working_bwrap
class TestRunBehaviour:
    """Parity checks against ProcessSandbox's existing test_coding_sandbox.py
    — BwrapSandbox must keep every guarantee ProcessSandbox had, not just add
    the new one."""

    def test_nonzero_exit_reported_not_raised(self, box: BwrapSandbox):
        r = asyncio.run(box.run("exit 7"))
        assert r.exit_code == 7
        assert not r.timed_out

    def test_env_scrub_drops_secrets(self, box: BwrapSandbox, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_API_TOKEN", "sekrit-token-123")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-sekrit")
        r = asyncio.run(box.run("env"))
        assert "sekrit-token-123" not in r.output
        assert "sk-sekrit" not in r.output
        assert "PROMETHEUS_API_TOKEN" not in r.output
        assert "PATH=" in r.output

    def test_timeout_kills_the_whole_tree(self, box: BwrapSandbox, tmp_path: Path):
        marker = tmp_path / "child-survived.txt"
        cmd = f"(sleep 3 && touch {marker}) & sleep 30"
        r = asyncio.run(box.run(cmd, timeout_seconds=1.0))
        assert r.timed_out
        assert r.exit_code is None
        import time as _t

        _t.sleep(3.5)
        assert not marker.exists(), "child escaped the timeout kill"

    def test_denied_path_inside_root_refuses_a_shell_write(
        self, tmp_path: Path
    ):
        """The improvement over ProcessSandbox: denied_paths now holds for
        run(), not just resolve() — a shell command cannot write there
        either, because it is bind-mounted read-only inside the namespace."""
        root = tmp_path / "jail3"
        (root / "secrets").mkdir(parents=True)
        isolate = bool(_CHECK and _CHECK.isolated_ok)
        sb = BwrapSandbox(
            root=root, denied_paths=(root / "secrets",), isolate_network=isolate
        )
        r = asyncio.run(sb.run("echo leaked > secrets/k.pem"))
        assert not (root / "secrets" / "k.pem").exists()
        assert r.exit_code != 0

    def test_long_output_head_tail_truncated(self, box: BwrapSandbox):
        r = asyncio.run(box.run("python3 -c \"print('x' * 50000)\""))
        assert r.exit_code == 0
        assert "truncated" in r.output
        assert len(r.output) < 50_000

    def test_sentinel_never_leaks_into_reported_output(self, box: BwrapSandbox):
        """The internal marker is implementation detail — callers must never
        see it in the output they read."""
        r = asyncio.run(box.run("echo hello"))
        assert "__bwrap_run_" not in r.output
