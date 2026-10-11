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
    assert r'(require-all (subpath "/Users/someone/.config") (regex #"/\.config/[^/]+/[^/]*env$"))' in profile
    assert "file-read*" in profile and "file-write*" in profile


def test_a_home_path_cannot_break_out_of_the_profile():
    profile = C.seatbelt_profile('/Users/odd "name" (x)')
    assert '(subpath "/Users/odd \\"name\\" (x)/.ssh")' in profile
    # The home never enters the regex literal, where a quote would end it whatever escapes it.
    assert profile.count('#"') == 1 and r'#"/\.config/[^/]+/[^/]*env$"' in profile


# ── apply_floors and floor_report on a Mac ───────────────────────────────────

def test_on_a_mac_a_verified_floor_wraps_every_shell_with_sandbox_exec(monkeypatch):
    _as(monkeypatch, "darwin")
    monkeypatch.setattr(C, "seatbelt_preflight", lambda **_: (True, "verified"))
    result = C.apply_floors(["/bin/bash", "-c", "ls"], read_mode="auto", write_mode="off",
                            writable=[], cwd=None)
    assert result.refusal is None and result.read_floor == C.READ_ACTIVE
    assert list(result.argv[:2]) == [C.SANDBOX_EXEC, "-p"]
    assert result.argv[2] == C.seatbelt_profile()
    assert list(result.argv[3:]) == ["/bin/bash", "-c", "ls"]


def test_on_a_mac_required_that_cannot_verify_refuses_and_names_sandbox_exec(monkeypatch):
    _as(monkeypatch, "darwin")
    monkeypatch.setattr(C, "seatbelt_preflight", lambda **_: (False, "it did not hold"))
    result = C.apply_floors(["/bin/bash", "-c", "ls"], read_mode="required", write_mode="off",
                            writable=[], cwd=None)
    assert result.argv == () and result.read_floor == "refused"
    assert "sandbox-exec" in result.refusal and "it did not hold" in result.refusal
    assert "apparmor_parser" not in result.refusal, "a Mac was told to load an AppArmor profile"


def test_on_a_mac_auto_that_cannot_verify_runs_and_says_unavailable(monkeypatch):
    _as(monkeypatch, "darwin")
    monkeypatch.setattr(C, "seatbelt_preflight", lambda **_: (False, "it did not hold"))
    result = C.apply_floors(["/bin/bash", "-c", "ls"], read_mode="auto", write_mode="off",
                            writable=[], cwd=None)
    assert result.read_floor == C.READ_UNAVAILABLE and list(result.argv) == ["/bin/bash", "-c", "ls"]


def test_floor_report_on_a_mac_names_seatbelt_and_is_active_not_unsupported(monkeypatch):
    _as(monkeypatch, "darwin")
    monkeypatch.setattr(C, "seatbelt_preflight", lambda **_: (True, "verified"))
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
@pytest.mark.parametrize("name", ['quote "q"', 'mixed "a" (b) [c] {d}+e', "plain"])
def test_the_real_floor_holds_for_a_home_path_with_awkward_characters(tmp_path, name):
    home = (tmp_path.resolve() / name)
    (home / ".config" / "prometheus").mkdir(parents=True)
    (home / ".config" / "prometheus" / "env").write_text("PROMETHEUS_API_TOKEN=master\n")
    (home / ".config" / "prometheus" / "settings.yaml").write_text("not a secret\n")
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id").write_text("PRIVATE\n")
    done = _sh(home, 'cat "$HOME/.config/prometheus/env" "$HOME/.ssh/id" 2>/dev/null; '
                     'cat "$HOME/.config/prometheus/settings.yaml"')
    assert "master" not in done.stdout and "PRIVATE" not in done.stdout, done.stdout
    assert "not a secret" in done.stdout, ("the profile did not parse, or denied too much", done.stderr[-300:])


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


# ── security.bash_allow_ssh: let ~/.ssh through, keep the master key out ──────
#
# `bash_confinement: off` was the only way to let a model-run `git push` over SSH use the keys, and it also opened
# ~/.config/prometheus/env. The narrow opt-out drops ONLY the ~/.ssh rule; ~/.gnupg and ~/.config/*/*env stay
# denied. On Linux the AppArmor profile is a file root loaded, so the key cannot change it, and the boot says so.

def test_allow_ssh_drops_only_the_ssh_rule():
    profile = C.seatbelt_profile("/Users/someone", allow_ssh=True)
    assert '(subpath "/Users/someone/.ssh")' not in profile
    assert '(subpath "/Users/someone/.gnupg")' in profile
    assert '(subpath "/Users/someone/.config")' in profile and "env$" in profile


def test_apply_floors_uses_the_ssh_profile_when_allowed(monkeypatch):
    _as(monkeypatch, "darwin")
    monkeypatch.setattr(C, "seatbelt_preflight", lambda **_: (True, "verified"))
    result = C.apply_floors(["/bin/bash", "-c", "ls"], read_mode="auto", write_mode="off", writable=[], cwd=None,
                            allow_ssh=True)
    assert result.argv[2] == C.seatbelt_profile(allow_ssh=True)


def test_the_key_is_a_strict_boolean_read_into_the_shell_floor():
    from prometheus.security.shell_floor import ShellFloor

    assert ShellFloor.from_security_config({}).allow_ssh is False
    assert ShellFloor.from_security_config({"bash_allow_ssh": True}).allow_ssh is True
    assert ShellFloor.from_security_config({"bash_allow_ssh": "yes"}).allow_ssh is False


def test_the_bash_tool_carries_it():
    from prometheus.tools.builtin.bash import BashTool

    assert BashTool(confinement="auto", allow_ssh=True)._allow_ssh is True
    assert BashTool(confinement="auto")._allow_ssh is False


def test_on_linux_the_boot_says_the_key_cannot_change_the_apparmor_profile(monkeypatch, caplog):
    import logging

    from prometheus.security.shell_floor import ShellFloor, announce

    _as(monkeypatch, "linux")
    monkeypatch.setattr(C, "preflight", lambda *a, **k: (True, "prometheus-bash (enforce)"))
    with caplog.at_level(logging.WARNING):
        announce(ShellFloor(read_mode="auto", write_mode="off", allow_ssh=True))
    assert any("bash_allow_ssh" in r.getMessage() and "AppArmor" in r.getMessage() for r in caplog.records)


@real
def test_the_real_floor_with_allow_ssh_reads_ssh_and_still_refuses_the_env_file(tmp_path):
    home = _scratch_home(tmp_path)
    done = subprocess.run(
        [C.SANDBOX_EXEC, "-p", C.seatbelt_profile(home, allow_ssh=True), "/bin/bash", "-c",
         'cat "$HOME/.ssh/id_ed25519"; cat "$HOME/.config/prometheus/env"'],
        capture_output=True, text=True, timeout=60, env={**os.environ, "HOME": str(home)})
    assert "PRIVATE" in done.stdout, "the opt-out did not let ~/.ssh through"
    assert "master" not in done.stdout, "the opt-out let the master key through"


@real
def test_the_real_preflight_verifies_the_ssh_profile_by_outcome():
    ok, detail = C.seatbelt_preflight(force=True, allow_ssh=True)
    assert ok, detail
