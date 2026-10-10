"""The read floor on macOS: Seatbelt, the same three families AppArmor denies on Linux.

docs/design/master-key-out-of-agent-shell.md. ``~/.config/prometheus/env`` holds the master key, the provider keys
and the gateway bot tokens. The path-declaring tools cannot open it (``denied_paths``), but a shell command is a
string the gate never resolves, so on Linux the kernel keeps every model-written shell out of ``~/.ssh``,
``~/.gnupg`` and ``~/.config/*/*env`` (AppArmor, ``deploy/apparmor/prometheus-bash``). On a Mac there was nothing:
``auto`` reported ``unsupported`` and ``cat ~/.config/prometheus/env`` worked.

Now ``security.bash_confinement`` on a Mac means the same floor through ``sandbox-exec`` (unprivileged): allow
everything, deny reads and writes of those three families. Same modes, same verify-by-outcome preflight, same
refusal when ``required`` cannot be had, same reporting. ``unsupported`` is left for a platform with neither.

The first half runs anywhere (the platform is faked); the second half runs the real ``sandbox-exec`` and exists
only on macOS, so CI's macOS job proves it by outcome.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from prometheus.permissions import confinement as C


@pytest.fixture(autouse=True)
def _fresh_cache():
    C.reset_cache()
    yield
    C.reset_cache()


def _as(monkeypatch, platform: str, *, sandbox_exec: bool = True):
    monkeypatch.setattr(C.sys, "platform", platform)
    monkeypatch.setattr(C, "seatbelt_possible", lambda: platform == "darwin" and sandbox_exec)


# ── which mechanism, where ───────────────────────────────────────────────────

def test_linux_uses_apparmor_a_mac_uses_seatbelt_and_neither_is_unsupported(monkeypatch):
    _as(monkeypatch, "linux")
    assert C.read_floor_mechanism() == C.MECHANISM_APPARMOR
    _as(monkeypatch, "darwin")
    assert C.read_floor_mechanism() == C.MECHANISM_SEATBELT
    _as(monkeypatch, "darwin", sandbox_exec=False)
    assert C.read_floor_mechanism() is None
    _as(monkeypatch, "win32")
    assert C.read_floor_mechanism() is None


def test_the_profile_denies_the_three_families_apparmor_denies_and_allows_the_rest():
    profile = C.seatbelt_profile("/Users/someone")
    assert "(allow default)" in profile
    assert '(subpath "/Users/someone/.ssh")' in profile
    assert '(subpath "/Users/someone/.gnupg")' in profile
    assert r'(regex #"^/Users/someone/\.config/[^/]+/[^/]*env$")' in profile
    assert "file-read*" in profile and "file-write*" in profile


def test_a_home_path_cannot_break_out_of_the_profile():
    profile = C.seatbelt_profile('/Users/odd "name" (x)')
    assert '(subpath "/Users/odd \\"name\\" (x)/.ssh")' in profile
    assert r'^/Users/odd \"name\" \(x\)/\.config/' in profile, "a quote would end the regex literal"


# ── apply_floors and floor_report on a Mac ───────────────────────────────────

def test_on_a_mac_a_verified_floor_wraps_every_shell_with_sandbox_exec(monkeypatch):
    _as(monkeypatch, "darwin")
    monkeypatch.setattr(C, "seatbelt_preflight", lambda *, force=False: (True, "verified"))
    result = C.apply_floors(["/bin/bash", "-c", "ls"], read_mode="auto", write_mode="off",
                            writable=[], cwd=None)
    assert result.refusal is None and result.read_floor == C.READ_ACTIVE
    assert list(result.argv[:2]) == [C.SANDBOX_EXEC, "-p"]
    assert result.argv[2] == C.seatbelt_profile()
    assert list(result.argv[3:]) == ["/bin/bash", "-c", "ls"]


def test_on_a_mac_required_that_cannot_verify_refuses_and_names_sandbox_exec(monkeypatch):
    _as(monkeypatch, "darwin")
    monkeypatch.setattr(C, "seatbelt_preflight", lambda *, force=False: (False, "it did not hold"))
    result = C.apply_floors(["/bin/bash", "-c", "ls"], read_mode="required", write_mode="off",
                            writable=[], cwd=None)
    assert result.argv == () and result.read_floor == "refused"
    assert "sandbox-exec" in result.refusal and "it did not hold" in result.refusal
    assert "apparmor_parser" not in result.refusal, "a Mac was told to load an AppArmor profile"


def test_on_a_mac_auto_that_cannot_verify_runs_and_says_unavailable(monkeypatch):
    _as(monkeypatch, "darwin")
    monkeypatch.setattr(C, "seatbelt_preflight", lambda *, force=False: (False, "it did not hold"))
    result = C.apply_floors(["/bin/bash", "-c", "ls"], read_mode="auto", write_mode="off",
                            writable=[], cwd=None)
    assert result.read_floor == C.READ_UNAVAILABLE and list(result.argv) == ["/bin/bash", "-c", "ls"]


def test_floor_report_on_a_mac_names_seatbelt_and_is_active_not_unsupported(monkeypatch):
    _as(monkeypatch, "darwin")
    monkeypatch.setattr(C, "seatbelt_preflight", lambda *, force=False: (True, "verified"))
    read = C.floor_report(read_mode="auto", write_mode="off")["bash_read_floor"]
    assert read["state"] == C.STATE_ACTIVE and read["mechanism"] == "seatbelt:sandbox-exec"


def test_unsupported_is_only_where_neither_mechanism_exists(monkeypatch):
    _as(monkeypatch, "darwin", sandbox_exec=False)
    report = C.floor_report(read_mode="auto", write_mode="off")
    assert report["bash_read_floor"]["state"] == C.STATE_UNSUPPORTED
    assert C.apply_floors(["sh"], read_mode="auto", write_mode="off", writable=[],
                          cwd=None).read_floor == C.READ_UNSUPPORTED


def test_linux_still_uses_aa_exec(monkeypatch):
    _as(monkeypatch, "linux")
    monkeypatch.setattr(C, "preflight", lambda profile=C.PROFILE, *, force=False: (True, "prometheus-bash"))
    result = C.apply_floors(["/bin/bash", "-c", "ls"], read_mode="auto", write_mode="off",
                            writable=[], cwd=None)
    assert "-p" in result.argv and C.PROFILE in result.argv and C.SANDBOX_EXEC not in result.argv


# ── the real thing (macOS only) ──────────────────────────────────────────────

# Decided here, not by the module under test: a floor that wrongly says "not possible" must not skip the tests
# that would catch it.
real = pytest.mark.skipif(not (sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists()),
                          reason="needs macOS with /usr/bin/sandbox-exec")


def _scratch_home(tmp_path: Path) -> Path:
    home = tmp_path.resolve() / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_ed25519").write_text("PRIVATE\n")
    (home / ".gnupg").mkdir()
    (home / ".config" / "prometheus").mkdir(parents=True)
    (home / ".config" / "prometheus" / "env").write_text("PROMETHEUS_API_TOKEN=master\n")
    (home / "notes.txt").write_text("ordinary\n")
    return home


def _sh(home: Path, script: str) -> subprocess.CompletedProcess:
    return subprocess.run([C.SANDBOX_EXEC, "-p", C.seatbelt_profile(home), "/bin/bash", "-c", script],
                          capture_output=True, text=True, timeout=60, env={**os.environ, "HOME": str(home)})


@real
def test_the_real_floor_refuses_the_env_file_and_ssh_however_they_are_reached(tmp_path):
    home = _scratch_home(tmp_path)
    for script in ('cat "$HOME/.config/prometheus/env"',
                   'cd "$HOME/.config/prometheus" && cat env',
                   'cd "$HOME/.ssh" && cat id_*',
                   'cat "$HOME"/.config/*/env',
                   f'{sys.executable} -c "import os; print(open(os.path.expanduser(\'~/.config/prometheus/env\')).read())"',
                   'echo x > "$HOME/.ssh/authorized_keys"'):
        done = _sh(home, script)
        assert "master" not in done.stdout and "PRIVATE" not in done.stdout and done.returncode != 0, script
    assert not (home / ".ssh" / "authorized_keys").exists()


@real
def test_the_real_floor_leaves_ordinary_work_alone(tmp_path):
    home = _scratch_home(tmp_path)
    done = _sh(home, 'cat "$HOME/notes.txt" && echo made > "$HOME/out.txt" && cat "$HOME/out.txt" && ls "$HOME/.config"')
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["ordinary", "made", "prometheus"]


@real
def test_the_real_preflight_verifies_by_outcome():
    ok, detail = C.seatbelt_preflight(force=True)
    assert ok, detail


@real
def test_through_apply_floors_a_real_shell_cannot_read_the_env_file(tmp_path, monkeypatch):
    home = _scratch_home(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    result = C.apply_floors(["/bin/bash", "-c", 'cat "$HOME/.config/prometheus/env"; cat "$HOME/notes.txt"'],
                            read_mode="required", write_mode="off", writable=[], cwd=None)
    assert result.refusal is None and result.read_floor == C.READ_ACTIVE
    done = subprocess.run(list(result.argv), capture_output=True, text=True, timeout=60)
    assert "master" not in done.stdout and "ordinary" in done.stdout, (done.stdout, done.stderr)
