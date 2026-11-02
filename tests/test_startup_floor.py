"""What starts Prometheus is under the floor, and its unconfined shells read no login file (GHSA-9ff8-vhh2-r2cw).

The independent review of the first fix found two ways a confined shell could still change what runs unconfined:

* The daemon's own STARTUP FILES were writable: its LaunchAgent plist, its systemd unit and the unit's drop-in
  folder (``deploy.sh`` writes ``10-venv.conf`` there, ``ExecStart=`` included), the deploy clone's ``scripts/``
  (``deploy_guard.sh`` is the unit's ``ExecStartPre``) and ``src/``, and the install itself: ``sys.prefix``, the venv
  root ``deploy.sh`` switches with its ``current`` link (``~/prometheus-venvs``), and the folder the ``prometheus``
  package is imported from. Now refused for writes in the gate, the Seatbelt profile and the AppArmor profile, with
  the folders above each, so a rename does not get around it. Reading stays allowed.
* Three unconfined doors (an operator's cron job, a daemon-built task, a command hook) ran ``bash -lc``, and a login
  bash runs the user's login files, which are the user's and not floored. They now run ``bash --noprofile --norc
  -c`` with their environment minus ``BASH_ENV`` and ``ENV`` (security/unconfined_shell.py).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from prometheus.__main__ import create_security_gate
from prometheus.permissions import confinement as C
from prometheus.security.unconfined_shell import UNCONFINED_SHELL, unconfined_shell_env

PROFILE_FILE = Path(__file__).resolve().parent.parent / "deploy" / "apparmor" / "prometheus-bash"
HOME = "/Users/someone"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for name in ("PROMETHEUS_DATA_DIR", "PROMETHEUS_CONFIG_DIR", "PROMETHEUS_NODE_DIR", "PROMETHEUS_VENV",
                 "PROMETHEUS_SYSTEMD_USER_DIR", "PROMETHEUS_LAUNCH_AGENTS_DIR"):
        monkeypatch.delenv(name, raising=False)
    C.reset_cache()
    yield
    C.reset_cache()


# ── the startup set ──────────────────────────────────────────────────────────

def test_the_startup_set_names_the_service_files_and_the_install(monkeypatch, tmp_path):
    prefix = (tmp_path / "venvs" / "abc123").resolve()
    prefix.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setenv("PROMETHEUS_VENV", str(tmp_path / "venvs" / "current"))
    files, dirs = C.startup_files(HOME), C.startup_dirs(HOME)
    assert f"{HOME}/Library/LaunchAgents/{C.LAUNCH_AGENT_NAME}" in files
    assert f"{HOME}/.config/systemd/user/{C.SYSTEMD_UNIT}" in files
    assert f"{HOME}/.config/systemd/user/{C.SYSTEMD_UNIT}.d" in dirs
    for d in ("prometheus-venvs", "prometheus-deploy/scripts", "prometheus-deploy/src",
              ".local/share/pipx/venvs/oara-prometheus", ".local/share/uv/tools/oara-prometheus"):
        assert f"{HOME}/{d}" in dirs, d
    assert str(prefix) in dirs, "sys.prefix"
    assert str((tmp_path / "venvs").resolve()) in dirs, "the folder PROMETHEUS_VENV's link lives in"
    assert os.path.realpath(os.path.dirname(C.__file__).rsplit("/permissions", 1)[0]) in dirs, "the package dir"


def test_the_names_match_the_installers():
    from prometheus.cli import launchd, service

    assert C.LAUNCH_AGENT_NAME == f"{launchd.LABEL}.plist"
    assert C.SYSTEMD_UNIT == service.UNIT_NAME


def test_a_moved_systemd_dir_is_covered(monkeypatch, tmp_path):
    moved = (tmp_path / "units").resolve()
    monkeypatch.setenv("PROMETHEUS_SYSTEMD_USER_DIR", str(moved))
    assert str(moved / C.SYSTEMD_UNIT) in C.startup_files(HOME)
    assert str(moved / f"{C.SYSTEMD_UNIT}.d") in C.startup_dirs(HOME)


def test_a_homebrew_keg_and_an_app_bundle_are_covered_whole(monkeypatch, tmp_path):
    keg = (tmp_path / "Cellar" / "oara-prometheus" / "0.9.18").resolve()
    (keg / "libexec").mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(keg / "libexec"))
    assert str(keg) in C.startup_dirs(HOME)
    app = (tmp_path / "Prometheus.app").resolve()
    (app / "Contents" / "Resources" / "python").mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(app / "Contents" / "Resources" / "python"))
    assert str(app) in C.startup_dirs(HOME)


# ── the gate: the file tools ─────────────────────────────────────────────────

GATE_CFGS = [{}, {"permission_mode": "autonomous"}, {"workspace_root": str(Path.home())}]
GATE_IDS = ["default", "autonomous", "workspace-covers-it"]

STARTUP = [
    f"Library/LaunchAgents/{C.LAUNCH_AGENT_NAME}",
    f".config/systemd/user/{C.SYSTEMD_UNIT}",
    f".config/systemd/user/{C.SYSTEMD_UNIT}.d/10-venv.conf",
    f".config/systemd/user/{C.SYSTEMD_UNIT}.d/99-planted.conf",
    "prometheus-venvs/current",
    "prometheus-venvs/abc123/lib/python3.12/site-packages/planted.pth",
    "prometheus-deploy/scripts/deploy_guard.sh",
    "prometheus-deploy/src/prometheus/daemon.py",
    ".local/share/pipx/venvs/oara-prometheus/bin/oara",
    ".local/share/uv/tools/oara-prometheus/lib/python3.12/site-packages/planted.pth",
]


@pytest.mark.parametrize("cfg", GATE_CFGS, ids=GATE_IDS)
@pytest.mark.parametrize("rel", STARTUP)
def test_the_gate_refuses_writing_what_starts_prometheus(cfg, rel):
    gate = create_security_gate(cfg)
    path = str(Path.home() / rel)
    d = gate.evaluate("write_file", file_path=path, path_is_write=True)
    assert not d.allowed and not d.requires_confirmation, (cfg, rel, d)
    assert "starts Prometheus" in d.reason, d.reason
    assert gate.evaluate("read_file", file_path=path, is_read_only=True).allowed, "reading stays allowed"


@pytest.mark.parametrize("cfg", GATE_CFGS, ids=GATE_IDS)
def test_the_gate_refuses_writing_this_processs_own_install(cfg, monkeypatch, tmp_path):
    prefix = (tmp_path / "somewhere" / "venv").resolve()
    prefix.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    gate = create_security_gate(cfg)
    for path in (prefix / "lib" / "site-packages" / "planted.pth",
                 Path(C._package_dir()) / "daemon.py"):
        d = gate.evaluate("write_file", file_path=str(path), path_is_write=True)
        assert not d.allowed and not d.requires_confirmation, (path, d)


def test_the_gate_still_allows_an_ordinary_write():
    gate = create_security_gate({"workspace_root": str(Path.home())})
    for rel in ("projects/app/main.py", ".config/systemd/user/other.service", "Library/LaunchAgents/other.plist"):
        d = gate.evaluate("write_file", file_path=str(Path.home() / rel), path_is_write=True)
        assert d.allowed, (rel, d)


def test_a_checkpoint_restore_never_writes_what_starts_prometheus():
    from prometheus.permissions.checker import own_state_entries
    from prometheus.security.path_guard import denying_entry

    entries = own_state_entries()
    for rel in STARTUP:
        assert denying_entry(str(Path.home() / rel), entries) is not None, rel


# ── Seatbelt profile text ────────────────────────────────────────────────────

def _write_block(profile: str) -> str:
    return "\n".join(profile.split("(deny file-write*")[1:])


def test_the_seatbelt_profile_refuses_writing_what_starts_prometheus():
    w = _write_block(C.seatbelt_profile(HOME))
    assert f'(literal "{HOME}/Library/LaunchAgents/{C.LAUNCH_AGENT_NAME}")' in w, w
    assert f'(literal "{HOME}/.config/systemd/user/{C.SYSTEMD_UNIT}")' in w, w
    assert f'(subpath "{HOME}/.config/systemd/user/{C.SYSTEMD_UNIT}.d")' in w, w
    for d in ("prometheus-venvs", "prometheus-deploy/scripts", "prometheus-deploy/src"):
        assert f'(subpath "{HOME}/{d}")' in w, d
    assert f'(subpath "{os.path.realpath(sys.prefix)}")' in w
    assert f'(subpath "{C._package_dir()}")' in w
    for parent in ("Library/LaunchAgents", ".config/systemd/user", ".config/systemd", "prometheus-deploy"):
        assert f'(literal "{HOME}/{parent}")' in w, parent


def test_the_seatbelt_profile_does_not_floor_reads_of_the_install():
    rw = C.seatbelt_profile(HOME).split("(deny file-read* file-write*", 1)[1].split("(deny file-write*", 1)[0]
    assert "prometheus-venvs" not in rw and "LaunchAgents" not in rw and "systemd" not in rw, rw


# ── AppArmor profile text ────────────────────────────────────────────────────

def _active_rules() -> list[str]:
    return [re.sub(r"\s+", " ", line.split("#", 1)[0].strip()) for line in PROFILE_FILE.read_text().splitlines()
            if line.split("#", 1)[0].strip()]


@pytest.mark.parametrize("rule", [
    "deny @{HOME}/.config/systemd/user/prometheus.service wl,",
    "deny @{HOME}/.config/systemd/user/prometheus.service.d/{,**} wl,",
    "deny @{HOME}/.config/systemd/ w,",
    "deny @{HOME}/.config/systemd/user/ w,",
    "deny @{HOME}/prometheus-venvs/{,**} wl,",
    "deny @{HOME}/prometheus-deploy/{scripts,src}/{,**} wl,",
    "deny @{HOME}/prometheus-deploy/ w,",
    "deny @{HOME}/.local/share/pipx/venvs/oara-prometheus/{,**} wl,",
    "deny @{HOME}/.local/share/uv/tools/oara-prometheus/{,**} wl,",
])
def test_the_apparmor_profile_has_the_rule(rule):
    assert rule in _active_rules(), rule


# ── the boot probe (AppArmor) ────────────────────────────────────────────────

def _label_ok(monkeypatch, *, startup: str = "refused", outside: list[str] | None = None):
    monkeypatch.setattr(C, "_probe_label", lambda profile: (True, f"{profile} (enforce)"))
    monkeypatch.setattr(C, "_probe_config_rename", lambda profile: "refused")
    monkeypatch.setattr(C, "_registry_detail", lambda profile: ("; writing the device registry refused", False))
    monkeypatch.setattr(C, "_trusted_detail", lambda profile: ("; writing trusted state refused", False))
    monkeypatch.setattr(C, "_cron_detail", lambda profile: ("; writing the scheduled-jobs files refused", False))
    monkeypatch.setattr(C, "_keys_and_launch_detail", lambda profile: ("; writing launch configuration refused",
                                                                      False))
    monkeypatch.setattr(C, "_probe_startup_write", lambda profile: startup)
    monkeypatch.setattr(C, "_install_outside_profile", lambda: list(outside or []))


def test_a_current_profile_says_the_startup_files_are_refused(monkeypatch, caplog):
    _label_ok(monkeypatch)
    with caplog.at_level(logging.INFO, logger=C.logger.name):
        ok, detail = C.preflight(C.PROFILE, force=True)
    assert ok and "writing what starts Prometheus refused" in detail, detail
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_a_profile_without_the_startup_rules_is_stale(monkeypatch, caplog):
    _label_ok(monkeypatch, startup="renamed")
    with caplog.at_level(logging.INFO, logger=C.logger.name):
        ok, detail = C.preflight(C.PROFILE, force=True)
    assert ok and "STALE PROFILE" in detail and C.RELOAD_COMMAND in detail, detail
    assert [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_an_install_outside_the_profile_is_named(monkeypatch):
    _label_ok(monkeypatch, outside=["/opt/elsewhere/venv"])
    ok, detail = C.preflight(C.PROFILE, force=True)
    assert ok and "/opt/elsewhere/venv" in detail and "outside the profile's rules" in detail, detail


def test_the_install_outside_check_skips_what_the_profile_covers(monkeypatch, tmp_path):
    home = tmp_path.resolve()
    monkeypatch.setenv("HOME", str(home))
    covered = home / "prometheus-venvs" / "abc123"
    covered.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(covered))
    monkeypatch.setattr(C, "_package_dir", lambda: str(home / "prometheus-deploy" / "src" / "prometheus"))
    assert C._install_outside_profile() == []
    loose = home / "my-venv"
    loose.mkdir()
    monkeypatch.setattr(sys, "prefix", str(loose))
    assert C._install_outside_profile() == [str(loose)]


# ── the real thing (macOS only) ──────────────────────────────────────────────

real = pytest.mark.skipif(not (sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists()),
                          reason="needs macOS with /usr/bin/sandbox-exec")


def _seed(home: Path) -> None:
    (home / "Library" / "LaunchAgents").mkdir(parents=True)
    (home / "Library" / "LaunchAgents" / C.LAUNCH_AGENT_NAME).write_text("<plist/>\n")
    units = home / ".config" / "systemd" / "user"
    (units / f"{C.SYSTEMD_UNIT}.d").mkdir(parents=True)
    (units / C.SYSTEMD_UNIT).write_text("[Service]\n")
    (units / f"{C.SYSTEMD_UNIT}.d" / "10-venv.conf").write_text("[Service]\n")
    (home / "prometheus-venvs" / "abc123" / "bin").mkdir(parents=True)
    (home / "prometheus-venvs" / "current").symlink_to(home / "prometheus-venvs" / "abc123")
    (home / "prometheus-deploy" / "scripts").mkdir(parents=True)
    (home / "prometheus-deploy" / "scripts" / "deploy_guard.sh").write_text("#!/bin/sh\n")
    (home / "prometheus-deploy" / "src" / "prometheus").mkdir(parents=True)


def _sh(home: Path, command: str) -> int:
    return subprocess.run([C.SANDBOX_EXEC, "-p", C.seatbelt_profile(home), "/bin/sh", "-c", command],
                          capture_output=True, text=True, timeout=30).returncode


@real
@pytest.mark.parametrize("command", [
    f'printf x >> "$H/Library/LaunchAgents/{C.LAUNCH_AGENT_NAME}"',
    f'printf x >> "$H/.config/systemd/user/{C.SYSTEMD_UNIT}"',
    f'printf x > "$H/.config/systemd/user/{C.SYSTEMD_UNIT}.d/99-planted.conf"',
    'mv "$H/.config/systemd/user" "$H/moved-units"',
    'mv "$H/Library/LaunchAgents" "$H/moved-agents"',
    'ln -sfn /tmp "$H/prometheus-venvs/current"',
    'printf x > "$H/prometheus-venvs/abc123/bin/planted"',
    'printf x >> "$H/prometheus-deploy/scripts/deploy_guard.sh"',
    'printf x > "$H/prometheus-deploy/src/prometheus/planted.py"',
    'mv "$H/prometheus-deploy" "$H/moved-clone"',
], ids=["plist", "unit", "drop-in", "rename-units", "rename-agents", "venv-link", "venv", "deploy-guard",
        "clone-src", "rename-clone"])
def test_a_shell_cannot_write_what_starts_prometheus(tmp_path, command):
    home = tmp_path.resolve()
    _seed(home)
    before = sorted(str(p) + (p.read_text() if p.is_file() and not p.is_symlink() else "") for p in home.rglob("*"))
    assert _sh(home, f'H="{home}"; {command}') != 0, command
    after = sorted(str(p) + (p.read_text() if p.is_file() and not p.is_symlink() else "") for p in home.rglob("*"))
    assert after == before, command
    assert os.readlink(home / "prometheus-venvs" / "current") == str(home / "prometheus-venvs" / "abc123")


@real
def test_a_shell_cannot_write_this_processs_own_install():
    """Rename each onto itself: a no-op where it is allowed, so a failing floor changes nothing."""
    rename = "import os, sys; [os.rename(p, p) for p in sys.argv[1:]]"
    for path in (os.path.realpath(sys.prefix), C._package_dir()):
        rc = subprocess.run([C.SANDBOX_EXEC, "-p", C.seatbelt_profile(), sys.executable, "-I", "-c", rename, path],
                            capture_output=True, text=True, timeout=30).returncode
        assert rc != 0, path


@real
def test_a_shell_can_still_read_and_run_the_install_and_write_beside_it(tmp_path):
    home = tmp_path.resolve()
    _seed(home)
    assert _sh(home, f'cat "{home}/prometheus-deploy/scripts/deploy_guard.sh" >/dev/null') == 0
    assert _sh(home, f'printf x > "{home}/.config/systemd/user/other.service"') == 0
    assert _sh(home, f'printf x > "{home}/Library/LaunchAgents/other.plist"') == 0
    rc = subprocess.run([C.SANDBOX_EXEC, "-p", C.seatbelt_profile(home), sys.executable, "-I", "-c", "print(1)"],
                        capture_output=True, text=True, timeout=30)
    assert rc.returncode == 0 and rc.stdout.strip() == "1", rc.stderr


@real
def test_the_seatbelt_preflight_measures_the_startup_rules():
    ok, detail = C.seatbelt_preflight(force=True)
    assert ok, detail
    assert "writing what starts Prometheus refused" in detail, detail


# ── the unconfined shells read no login file ─────────────────────────────────

@pytest.fixture()
def planted_home(tmp_path, monkeypatch):
    """A HOME whose login files and startup-file variables would each leave a marker if they ran."""
    home = (tmp_path / "home").resolve()
    home.mkdir()
    markers = tmp_path / "markers"
    markers.mkdir()
    for name in (".bash_profile", ".bash_login", ".profile", ".bashrc"):
        (home / name).write_text(f'touch "{markers}/{name}"\n')
    for var in ("BASH_ENV", "ENV"):
        script = tmp_path / f"{var.lower()}.sh"
        script.write_text(f'touch "{markers}/{var}"\n')
        monkeypatch.setenv(var, str(script))
    monkeypatch.setenv("HOME", str(home))
    return markers


def test_the_shell_is_not_a_login_shell():
    assert UNCONFINED_SHELL == ("/bin/bash", "--noprofile", "--norc", "-c")


def test_the_env_keeps_everything_but_the_startup_file_variables():
    env = unconfined_shell_env({"PATH": "/bin", "BASH_ENV": "/x", "ENV": "/y", "FOO": "1"})
    assert env == {"PATH": "/bin", "FOO": "1"}


def test_the_shell_runs_no_login_file(planted_home):
    out = subprocess.run([*UNCONFINED_SHELL, "echo RAN"], capture_output=True, text=True, timeout=30,
                         env=unconfined_shell_env())
    assert out.stdout.strip() == "RAN", out.stderr
    assert sorted(p.name for p in planted_home.iterdir()) == []


async def test_an_operators_cron_job_runs_no_login_file(planted_home, tmp_path, monkeypatch):
    from prometheus.gateway import cron_scheduler as cs

    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("KEPT_FOR_THE_JOB", "yes")
    monkeypatch.setattr(cs, "vet_cron_command", lambda command, cwd=None: (True, ""))
    monkeypatch.setattr(cs, "mark_job_run", lambda *a, **k: None)
    job = {"name": "op-job", "command": 'echo "RAN $KEPT_FOR_THE_JOB"; command -v python3',
           "cwd": str(tmp_path), "enabled": True}
    entry = await cs.execute_job(job)
    assert entry["returncode"] == 0, entry["stderr"]
    lines = entry["stdout"].strip().splitlines()
    assert lines[0] == "RAN yes", "the job keeps the daemon's environment"
    assert "job-python" in lines[1], "an operator's python3 is still the daemon's"
    assert sorted(p.name for p in planted_home.iterdir()) == []


async def test_a_daemon_built_task_runs_no_login_file(planted_home, tmp_path):
    from prometheus.tasks.manager import BackgroundTaskManager
    from prometheus.tasks.store import TaskStore
    from tests.test_managed_tasks import AllowGate, _wait_terminal

    mgr = BackgroundTaskManager(store=TaskStore(), security_gate=AllowGate())
    rec = await mgr.create_shell_task(command="echo RAN", description="daemon-built", cwd=str(tmp_path),
                                      floored=False)
    done = await _wait_terminal(mgr, rec.id)
    assert done.return_code == 0 and "RAN" in Path(done.output_file).read_text()
    assert sorted(p.name for p in planted_home.iterdir()) == []


async def test_a_command_hook_runs_no_login_file_even_with_bash_env_allowlisted(planted_home, tmp_path):
    from prometheus.hooks.events import HookEvent
    from prometheus.hooks.executor import HookExecutionContext, HookExecutor
    from prometheus.hooks.registry import HookRegistry
    from prometheus.hooks.schemas import CommandHookDefinition

    hook = CommandHookDefinition(command='echo "RAN BASH_ENV=${BASH_ENV-unset}"', timeout_seconds=10,
                                 env_allowlist=["BASH_ENV", "ENV"])
    registry = HookRegistry()
    registry.add(HookEvent.PRE_TOOL_USE, hook)
    executor = HookExecutor(registry, HookExecutionContext(cwd=tmp_path, provider=None, default_model="m"))
    result = await asyncio.wait_for(executor.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"}), 30)
    [one] = result.results
    assert one.success and one.output == "RAN BASH_ENV=unset", one
    assert sorted(p.name for p in planted_home.iterdir()) == []


# ── through the loop ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("rel", [f"units/{C.SYSTEMD_UNIT}", f"units/{C.SYSTEMD_UNIT}.d/10-venv.conf",
                                 f"agents/{C.LAUNCH_AGENT_NAME}", "venvs/abc123/planted.pth"])
async def test_a_model_s_write_file_on_a_startup_file_is_refused_through_the_loop(monkeypatch, tmp_path, rel):
    """The real path: a scripted model asks write_file for the file, the loop asks the gate, nothing lands. The
    installers' overrides and PROMETHEUS_VENV point into one temporary folder that is also the workspace, so a
    failing floor writes nothing real. A write beside them is the control."""
    from prometheus.engine.agent_loop import LoopContext, run_loop
    from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
    from prometheus.engine.usage import UsageSnapshot
    from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
    from prometheus.tools.base import ToolRegistry
    from prometheus.tools.builtin.file_write import FileWriteTool

    root = tmp_path.resolve()
    monkeypatch.setenv("PROMETHEUS_SYSTEMD_USER_DIR", str(root / "units"))
    monkeypatch.setenv("PROMETHEUS_LAUNCH_AGENTS_DIR", str(root / "agents"))
    monkeypatch.setenv("PROMETHEUS_VENV", str(root / "venvs" / "current"))
    target = root / rel
    target.parent.mkdir(parents=True)
    target.write_bytes(b"REAL")

    class _Scripted(ModelProvider):
        def __init__(self, path: Path) -> None:
            self.path = path
            self.n = 0

        async def stream_message(self, request):  # noqa: ANN001
            self.n += 1
            content = ([ToolUseBlock(id="t1", name="write_file", input={"path": str(self.path), "content": "evil"})]
                       if self.n == 1 else [TextBlock(text="done")])
            yield ApiMessageCompleteEvent(message=ConversationMessage(role="assistant", content=content),
                                          usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop")

    async def drive(path: Path) -> None:
        tools = ToolRegistry()
        tools.register(FileWriteTool())
        gate = create_security_gate({"workspace_root": str(root), "permission_mode": "autonomous"})
        ctx = LoopContext(provider=_Scripted(path), model="stub", system_prompt="S", max_tokens=64,
                          tool_registry=tools, permission_checker=gate, cwd=root)
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")], session_id="cli"):
            pass

    await drive(target)
    assert target.read_bytes() == b"REAL", f"write_file changed {rel} through the loop"
    await drive(root / "notes.txt")
    assert (root / "notes.txt").read_text() == "evil", "control: an ordinary write in the workspace lands"
