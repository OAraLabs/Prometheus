"""scripts/install_oara_command.sh puts a working `oara` on PATH for a deploy that does not install the package.

``scripts/deploy.sh`` builds its venvs with ``uv sync --no-install-project``, so they hold no ``oara`` console script, and
``oara pair --owner`` (the owner step for a second computer) was "command not found" on the box that runs the daemon.
pip, pipx, uv tool and Homebrew get ``oara`` from the package's entry point; a deploy that runs the code from a checkout
gets it from this launcher. It runs the deployed venv's interpreter on the deployed code with the service's environment,
and it follows ``current``, the link a deploy moves, so a deploy or a rollback needs no second step.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INSTALLER = REPO / "scripts" / "install_oara_command.sh"
DEPLOY = REPO / "scripts" / "deploy.sh"

#: A fake interpreter: it prints what the launcher handed it, so a test can see the environment and the arguments.
FAKE_PYTHON = """#!/bin/sh
printf 'VENV=%s\\n' "$PROMETHEUS_VENV"
printf 'NOUSER=%s\\n' "$PYTHONNOUSERSITE"
printf 'PYTHONPATH=%s\\n' "$PYTHONPATH"
printf 'CONFIG=%s\\n' "${PROMETHEUS_CONFIG_DIR:-unset}"
printf 'DATA=%s\\n' "${PROMETHEUS_DATA_DIR:-unset}"
printf 'TOKEN=%s\\n' "${PROMETHEUS_API_TOKEN:-unset}"
printf 'TAG=%s\\n' "__TAG__"
for a in "$@"; do printf 'ARG=[%s]\\n' "$a"; done
"""


def _venv(path: Path, tag: str = "one") -> Path:
    (path / "bin").mkdir(parents=True)
    python = path / "bin" / "python"
    python.write_text(FAKE_PYTHON.replace("__TAG__", tag))
    python.chmod(0o755)
    return path


def _code(path: Path) -> Path:
    (path / "prometheus").mkdir(parents=True)
    (path / "prometheus" / "__init__.py").write_text("")
    return path


def _install(tmp_path, venv, src, *extra, path_dirs=(), env=None):
    bin_dir = tmp_path / "bin"
    run_env = {"PATH": os.pathsep.join([*map(str, path_dirs), "/usr/bin", "/bin"]), "HOME": str(tmp_path / "home")}
    run_env.update(env or {})
    result = subprocess.run(["bash", str(INSTALLER), "--bin-dir", str(bin_dir), *extra, str(venv), str(src)],
                            capture_output=True, text=True, env=run_env, timeout=60)
    return result, bin_dir / "oara"


def _run(command: Path, *args, env=None):
    return subprocess.run([str(command), *args], capture_output=True, text=True, timeout=60,
                          env={"PATH": "/usr/bin:/bin", **(env or {})})


@pytest.fixture
def world(tmp_path):
    venv = _venv(tmp_path / "venvs" / "current")
    src = _code(tmp_path / "clone" / "src")
    return tmp_path, venv, src


# ── what the command does ────────────────────────────────────────────────────

def test_it_runs_the_deployed_interpreter_on_the_deployed_code(world):
    tmp, venv, src = world
    result, command = _install(tmp, venv, src)
    assert result.returncode == 0, result.stderr
    out = _run(command, "pair", "--owner").stdout
    assert f"VENV={venv}\n" in out and "NOUSER=1\n" in out
    assert f"PYTHONPATH={src}\n" in out
    args = [line for line in out.splitlines() if line.startswith("ARG=")]
    assert args == ["ARG=[-m]", "ARG=[prometheus]", "ARG=[pair]", "ARG=[--owner]"]


def test_it_keeps_a_pythonpath_the_caller_set_after_the_code(world):
    tmp, venv, src = world
    _, command = _install(tmp, venv, src)
    assert f"PYTHONPATH={src}:/elsewhere\n" in _run(command, env={"PYTHONPATH": "/elsewhere"}).stdout


def test_arguments_pass_through_exactly(world):
    tmp, venv, src = world
    _, command = _install(tmp, venv, src)
    out = _run(command, "say", "two words", "it's", '"quoted"', "$HOME", "").stdout
    assert [l for l in out.splitlines() if l.startswith("ARG=")][2:] == [
        "ARG=[say]", "ARG=[two words]", "ARG=[it's]", 'ARG=["quoted"]', "ARG=[$HOME]", "ARG=[]"]


def test_it_follows_the_current_link_a_deploy_moves(world):
    tmp, venv, src = world
    root = tmp / "roots"
    root.mkdir()
    first, second = _venv(root / "aaa", "first"), _venv(root / "bbb", "second")
    (root / "current").symlink_to(first)
    _, command = _install(tmp, root / "current", src)
    assert "TAG=first\n" in _run(command).stdout
    (root / "current.tmp").symlink_to(second)
    os.replace(root / "current.tmp", root / "current")            # what deploy.sh B2 does
    assert "TAG=second\n" in _run(command).stdout, "a deploy or a rollback needed a second step"


def test_paths_with_spaces_and_quotes_work(tmp_path):
    venv = _venv(tmp_path / "my venvs" / "it's current")
    src = _code(tmp_path / "the clone" / "src")
    result, command = _install(tmp_path, venv, src)
    assert result.returncode == 0, result.stderr
    out = _run(command, "x").stdout
    assert f"VENV={venv}\n" in out and f"PYTHONPATH={src}\n" in out


# ── what it leaves behind ────────────────────────────────────────────────────

def test_it_is_executable_and_leaves_nothing_else_behind(world):
    tmp, venv, src = world
    _, command = _install(tmp, venv, src)
    assert command.stat().st_mode & stat.S_IXUSR
    assert sorted(p.name for p in command.parent.iterdir()) == ["oara"], "a temporary file was left in the bin directory"


def test_a_second_install_replaces_its_own_launcher(world):
    tmp, venv, src = world
    _install(tmp, venv, src)
    other = _venv(tmp / "venvs" / "other", "other")
    result, command = _install(tmp, other, src)
    assert result.returncode == 0 and "TAG=other\n" in _run(command).stdout


def test_an_oara_that_is_not_ours_is_left_alone(world):
    tmp, venv, src = world
    (tmp / "bin").mkdir()
    foreign = tmp / "bin" / "oara"
    foreign.write_text("#!/bin/sh\necho a real entry point\n")
    foreign.chmod(0o755)
    result, command = _install(tmp, venv, src)
    assert result.returncode == 0
    assert command.read_text() == "#!/bin/sh\necho a real entry point\n", "it overwrote a command it did not write"
    assert "left it alone" in result.stdout + result.stderr


def test_a_symlinked_oara_is_left_alone(world):
    tmp, venv, src = world
    (tmp / "bin").mkdir()
    target = tmp / "real-oara"
    target.write_text("#!/bin/sh\necho linked\n")
    target.chmod(0o755)
    (tmp / "bin" / "oara").symlink_to(target)
    result, command = _install(tmp, venv, src)
    assert result.returncode == 0 and command.is_symlink() and "linked" in _run(command).stdout


def test_a_venv_without_an_interpreter_or_code_without_the_package_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    result, command = _install(tmp_path, tmp_path / "empty", _code(tmp_path / "src"))
    assert result.returncode == 1 and not command.exists()
    venv = _venv(tmp_path / "v")
    (tmp_path / "nocode").mkdir()
    result, command = _install(tmp_path, venv, tmp_path / "nocode")
    assert result.returncode == 1 and not command.exists()


# ── the service's environment ────────────────────────────────────────────────

def _fake_systemctl(tmp_path, environment: str) -> Path:
    directory = tmp_path / "fakebin"
    directory.mkdir()
    fake = directory / "systemctl"
    fake.write_text(f"#!/bin/sh\necho '{environment}'\n")
    fake.chmod(0o755)
    return directory


def test_the_services_prometheus_settings_are_used_but_never_a_secret(world):
    tmp, venv, src = world
    fake = _fake_systemctl(tmp, "PYTHONPATH=/old PROMETHEUS_CONFIG_DIR=/home/u/.prometheus PROMETHEUS_DATA_DIR=/home/u/.prometheus/data "
                                "PROMETHEUS_API_TOKEN=sekrit PROMETHEUS_VENV=/old/venv PATH=/bin")
    result, command = _install(tmp, venv, src, "--unit", "prometheus.service", path_dirs=[fake])
    assert result.returncode == 0, result.stderr
    text = command.read_text()
    assert "sekrit" not in text and "PROMETHEUS_API_TOKEN" not in text, "a secret was written into a world-readable launcher"
    assert "/old/venv" not in text and "PYTHONPATH=/old" not in text
    out = _run(command).stdout
    assert "CONFIG=/home/u/.prometheus\n" in out and "DATA=/home/u/.prometheus/data\n" in out and "TOKEN=unset\n" in out
    assert f"VENV={venv}\n" in out, "the launcher's own venv was overridden by the service's"


def test_an_explicit_setting_is_used_and_a_bad_one_is_refused(world):
    tmp, venv, src = world
    result, command = _install(tmp, venv, src, "--env", "PROMETHEUS_CONFIG_DIR=/x/.prometheus")
    assert result.returncode == 0 and "CONFIG=/x/.prometheus\n" in _run(command).stdout
    for bad in ("PROMETHEUS_CONFIG_DIR=/x; rm -rf /", "NOT_OURS=1", "PROMETHEUS_API_TOKEN=abc", "PROMETHEUS_X=$(id)"):
        result, command = _install(tmp, venv, src, "--env", bad)
        assert result.returncode == 2, bad


# ── telling the person ───────────────────────────────────────────────────────

def test_a_bin_directory_that_is_not_on_path_is_said_plainly(world):
    tmp, venv, src = world
    result, _ = _install(tmp, venv, src)
    assert "not on your PATH" in result.stdout + result.stderr and "export PATH=" in result.stdout + result.stderr


def test_a_bin_directory_that_is_on_path_gets_no_warning(world):
    tmp, venv, src = world
    (tmp / "bin").mkdir()
    result, _ = _install(tmp, venv, src, path_dirs=[tmp / "bin"])
    assert result.returncode == 0 and "not on your PATH" not in result.stdout + result.stderr


# ── the wiring ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("script", [INSTALLER, DEPLOY], ids=lambda p: p.name)
def test_the_scripts_are_valid_bash(script):
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True).returncode == 0


def test_deploy_sh_installs_the_command_after_it_has_verified_and_never_fails_the_deploy_over_it():
    text = DEPLOY.read_text()
    assert "install_oara_command.sh" in text
    assert text.index("B5 verify") < text.index("install_oara_command.sh"), "the command is installed before the deploy is verified"
    line = next(l for l in text.splitlines() if "install_oara_command.sh" in l and "bash" in l)
    assert "||" in line, "a failure to install the command must be a warning, not a failed deploy"


def test_the_installer_never_asks_for_root():
    assert "sudo" not in INSTALLER.read_text()
