"""The same-Mac pairing secret is a credential, and the floors now treat it as one.

``<pairing dir>/pair.secret`` (``config/local_pairing.py``) is what ``POST /api/pair/local`` (and
``POST /api/setup/pair`` in setup mode) trades for an OWNER device token, operator-equivalent, when the request
comes from loopback with a loopback Host and no Origin. Every model-written shell on the same Mac satisfies all
three, so whatever can read the file, or WRITE one with a value it chose, can mint itself owner credentials
during the window the secret exists (setup mode before Beacon pairs, or after ``Prometheus --pair`` until used).

Measured before this change, on macOS 15 against a scratch home: under the Seatbelt read floor
``cat .../pairing/pair.secret`` printed the secret, and ``read_file`` passed the gate, because neither floor named
the directory. The same measurement found an older hole in the shell floor: ``mv ~/.config ~/cfg`` was allowed
(``~/.config`` itself is not ``~/.config/*/*env``) and ``cat ~/cfg/prometheus/env`` then printed the env file.
Renaming a parent out from under a denied path escapes a path-matched rule, so the profile now also refuses writes
to the directories ABOVE each denied root, up to the home. docs/design/master-key-out-of-agent-shell.md.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from prometheus.__main__ import create_security_gate
from prometheus.permissions import confinement as C
from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin.glob import GlobTool, GlobToolInput
from prometheus.tools.builtin.grep import GrepTool, GrepToolInput

MAC_PAIRING = "Library/Application Support/Prometheus/pairing"


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.delenv("PROMETHEUS_LOCAL_PAIRING_DIR", raising=False)
    C.reset_cache()
    yield
    C.reset_cache()


# ── the shell floor: the profile text ────────────────────────────────────────

def test_the_profile_denies_reading_and_writing_the_pairing_directory():
    profile = C.seatbelt_profile("/Users/someone")
    deny_rw = profile.split("(deny file-read* file-write*", 1)[1].split("))", 1)[0]
    assert f'(subpath "/Users/someone/{MAC_PAIRING}")' in deny_rw, profile


def test_the_profile_refuses_renaming_the_parents_of_what_it_denies():
    profile = C.seatbelt_profile("/Users/someone")
    write_only = profile.split("(deny file-write*\n", 1)[1]
    for parent in ("/Users/someone/.config", "/Users/someone/Library",
                   "/Users/someone/Library/Application Support",
                   "/Users/someone/Library/Application Support/Prometheus"):
        assert f'(literal "{parent}")' in write_only, (parent, profile)
    # The home itself is not one of them: a user cannot rename their own home (its parent is root's), and
    # refusing `touch ~` or a chmod of it would cost ordinary work for nothing.
    assert '(literal "/Users/someone")' not in profile


def test_allow_ssh_keeps_the_pairing_directory_denied():
    assert f'(subpath "/Users/someone/{MAC_PAIRING}")' in C.seatbelt_profile("/Users/someone", allow_ssh=True)


def test_a_pairing_directory_named_by_the_environment_is_denied_too(monkeypatch, tmp_path):
    elsewhere = tmp_path.resolve() / "wired" / "pairing"
    monkeypatch.setenv("PROMETHEUS_LOCAL_PAIRING_DIR", str(elsewhere))
    profile = C.seatbelt_profile("/Users/someone")
    assert f'(subpath "{elsewhere}")' in profile
    assert f'(literal "{elsewhere.parent}")' in profile


# ── the file tools: the gate's floor, and grep/glob pruning ──────────────────

def _home_secret(spelling: str = MAC_PAIRING) -> str:
    return str(Path.home() / spelling / "pair.secret")


@pytest.mark.parametrize("cfg", [{}, {"permission_mode": "default", "denied_paths": []}],
                         ids=["shipped", "explicit-empty-list"])
@pytest.mark.parametrize("tool, args", [
    ("read_file", {"file_path": _home_secret(), "is_read_only": True}),
    ("write_file", {"file_path": _home_secret(), "path_is_write": True}),
    ("read_file", {"file_path": _home_secret(MAC_PAIRING.lower()), "is_read_only": True}),
    ("read_file", {"file_path": "/home/someone/.local/share/prometheus/pairing/pair.secret", "is_read_only": True}),
    # grep and glob reach the gate with their root as the path (permissions/tool_paths.py).
    ("grep", {"file_path": str(Path.home() / MAC_PAIRING), "is_read_only": True}),
    ("glob", {"file_path": str(Path.home() / MAC_PAIRING), "is_read_only": True}),
], ids=["read", "write", "read-case-folded", "read-linux-default", "grep-root", "glob-root"])
def test_the_gate_refuses_the_pairing_directory_on_every_config_shape(cfg, tool, args):
    if args["file_path"] == _home_secret(MAC_PAIRING.lower()) and sys.platform != "darwin":
        pytest.skip("case folding applies on macOS's case-insensitive volumes")
    decision = create_security_gate(cfg).evaluate(tool, **args)
    assert not decision.allowed, (tool, args, decision)


def _plant(tmp_path: Path) -> Path:
    pairing = tmp_path / MAC_PAIRING
    pairing.mkdir(parents=True)
    (pairing / "pair.secret").write_text("PAIRSECRETVALUE\n")
    (tmp_path / "notes.txt").write_text("ordinary\n")
    return pairing


@pytest.mark.parametrize("denied_paths", [None, []], ids=["no-list", "explicit-empty-list"])
def test_grep_over_a_parent_withholds_the_secret(tmp_path, denied_paths):
    _plant(tmp_path)
    result = asyncio.run(GrepTool(denied_paths=denied_paths).execute(
        GrepToolInput(pattern="PAIRSECRET|ordinary", root=str(tmp_path)),
        ToolExecutionContext(cwd=tmp_path, metadata={})))
    assert "PAIRSECRETVALUE" not in result.output and "pair.secret" not in result.output, result.output
    assert "ordinary" in result.output


@pytest.mark.parametrize("denied_paths", [None, []], ids=["no-list", "explicit-empty-list"])
def test_glob_over_a_parent_withholds_the_secret(tmp_path, denied_paths):
    _plant(tmp_path)
    result = asyncio.run(GlobTool(denied_paths=denied_paths).execute(
        GlobToolInput(pattern="**/*", root=str(tmp_path)),
        ToolExecutionContext(cwd=tmp_path, metadata={})))
    assert "pair.secret" not in result.output, result.output
    assert "notes.txt" in result.output


# ── the real thing (macOS only) ──────────────────────────────────────────────

real = pytest.mark.skipif(not (sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists()),
                          reason="needs macOS with /usr/bin/sandbox-exec")


def _scratch_home(tmp_path: Path) -> Path:
    home = tmp_path.resolve() / "home"
    (home / ".config" / "prometheus").mkdir(parents=True)
    (home / ".config" / "prometheus" / "env").write_text("PROMETHEUS_API_TOKEN=master\n")
    _plant(home)
    return home


def _sh(home: Path, script: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run([C.SANDBOX_EXEC, "-p", C.seatbelt_profile(home), "/bin/bash", "-c", script],
                          capture_output=True, text=True, timeout=60,
                          env={**os.environ, "HOME": str(home), **env})


@real
def test_the_real_floor_refuses_reading_the_secret_however_it_is_reached(tmp_path):
    home = _scratch_home(tmp_path)
    for script in (f'cat "$HOME/{MAC_PAIRING}/pair.secret"',
                   f'cd "$HOME/{MAC_PAIRING}" && cat pair.secret',
                   f'cat "$HOME/{MAC_PAIRING.lower()}/pair.secret"',
                   f'{sys.executable} -c "import os, urllib.parse; '
                   f'print(open(os.path.expanduser(\'~/{MAC_PAIRING}/pair.secret\')).read())"',
                   f'ln -s "$HOME/{MAC_PAIRING}" "$HOME/l" && cat "$HOME/l/pair.secret"',
                   f'ln "$HOME/{MAC_PAIRING}/pair.secret" "$HOME/hard" && cat "$HOME/hard"'):
        done = _sh(home, script)
        assert "PAIRSECRETVALUE" not in done.stdout, script


@real
def test_the_real_floor_refuses_planting_a_secret_of_the_shells_choosing(tmp_path):
    home = _scratch_home(tmp_path)
    _sh(home, f'echo chosen > "$HOME/{MAC_PAIRING}/pair.secret"; rm -f "$HOME/{MAC_PAIRING}/pair.secret"')
    assert (home / MAC_PAIRING / "pair.secret").read_text() == "PAIRSECRETVALUE\n"


@real
def test_the_real_floor_refuses_renaming_a_parent_out_from_under_a_denied_path(tmp_path):
    home = _scratch_home(tmp_path)
    for script in ('mv "$HOME/.config" "$HOME/cfg" && cat "$HOME/cfg/prometheus/env"',
                   'mv "$HOME/Library/Application Support/Prometheus" "$HOME/P" && cat "$HOME/P/pairing/pair.secret"',
                   'mv "$HOME/Library/Application Support" "$HOME/AS" && cat "$HOME/AS/Prometheus/pairing/pair.secret"',
                   'mv "$HOME/Library" "$HOME/L" && cat "$HOME/L/Application Support/Prometheus/pairing/pair.secret"'):
        done = _sh(home, script)
        assert "master" not in done.stdout and "PAIRSECRETVALUE" not in done.stdout, script
    assert (home / ".config" / "prometheus" / "env").exists() and (home / MAC_PAIRING / "pair.secret").exists()


@real
def test_the_real_floor_refuses_moving_a_prepared_directory_into_place(tmp_path):
    home = tmp_path.resolve() / "home"
    (home / "Library" / "Application Support").mkdir(parents=True)
    _sh(home, 'mkdir -p "$HOME/stage/pairing" && echo chosen > "$HOME/stage/pairing/pair.secret" && '
              'mv "$HOME/stage" "$HOME/Library/Application Support/Prometheus"')
    assert not (home / MAC_PAIRING).exists()


@real
def test_the_real_floor_leaves_the_neighbours_of_what_it_protects_alone(tmp_path):
    home = _scratch_home(tmp_path)
    done = _sh(home, 'mkdir "$HOME/.config/newapp" && echo a > "$HOME/.config/newapp/settings" && '
                     'mkdir -p "$HOME/Library/Application Support/Other" && '
                     'echo b > "$HOME/Library/Application Support/Other/f" && '
                     'echo c > "$HOME/Library/Application Support/Prometheus/beside" && '
                     'cat "$HOME/.config/newapp/settings" "$HOME/Library/Application Support/Other/f" '
                     '"$HOME/Library/Application Support/Prometheus/beside" "$HOME/notes.txt"')
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["a", "b", "c", "ordinary"]


@real
def test_the_real_floor_denies_a_pairing_directory_named_by_the_environment(tmp_path, monkeypatch):
    home = _scratch_home(tmp_path)
    wired = tmp_path.resolve() / "wired" / "pairing"
    wired.mkdir(parents=True)
    (wired / "pair.secret").write_text("WIREDSECRET\n")
    monkeypatch.setenv("PROMETHEUS_LOCAL_PAIRING_DIR", str(wired))
    done = _sh(home, f'cat "{wired}/pair.secret"; mv "{wired.parent}" "{tmp_path.resolve()}/moved" && '
                     f'cat "{tmp_path.resolve()}/moved/pairing/pair.secret"')
    assert "WIREDSECRET" not in done.stdout, done.stdout


@real
def test_the_real_preflight_proves_the_pairing_directory_and_the_rename_by_outcome():
    ok, detail = C.seatbelt_preflight(force=True)
    assert ok, detail
    assert "pairing" in detail and "renaming" in detail, detail
