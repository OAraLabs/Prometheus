"""The scheduled-jobs files are under the shell floor: a model's shell cannot plant a job that runs as the owner.

cron_jobs.json (config/paths.py) is the schedule, re-read every scheduler tick (gateway/cron_scheduler.py), so a
job written into it runs on its next due time with no restart, as the owner, through the shell floor — persistence
beyond the session. cron_history.jsonl and the resend queue cron_undelivered.jsonl are its append-only logs.

All three live in the data directory, like the device registry, and are floored the same way: Seatbelt and
AppArmor deny a shell's writes, the gate refuses the file tools, reads still work, and the daemon (unconfined)
still writes them. Same class as GHSA-wq93.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from prometheus.__main__ import create_security_gate
from prometheus.permissions import confinement as C

PROFILE_FILE = Path(__file__).resolve().parent.parent / "deploy" / "apparmor" / "prometheus-bash"
HOME = "/Users/someone"
DATA = f"{HOME}/.prometheus/data"
CRON_FILES = ("cron_jobs.json", "cron_history.jsonl", "cron_undelivered.jsonl")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for name in ("PROMETHEUS_DATA_DIR", "PROMETHEUS_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)
    C.reset_cache()
    yield
    C.reset_cache()


def _write_denies(profile: str) -> str:
    return "\n".join(block for block in profile.split("(deny file-write*")[1:])


# ── Seatbelt profile text ────────────────────────────────────────────────────

def test_the_profile_refuses_writes_to_each_scheduled_jobs_file():
    denies = _write_denies(C.seatbelt_profile(HOME))
    for name in CRON_FILES:
        assert f'(literal "{DATA}/{name}")' in denies, name


@pytest.mark.parametrize("variable, value", [
    ("PROMETHEUS_DATA_DIR", "wired/data"),
    ("PROMETHEUS_CONFIG_DIR", "conf"),
], ids=["data-dir", "config-dir"])
def test_a_moved_data_directory_is_covered(monkeypatch, tmp_path, variable, value):
    root = tmp_path.resolve()
    monkeypatch.setenv(variable, str(root / value))
    data = root / ("wired/data" if variable == "PROMETHEUS_DATA_DIR" else "conf/data")
    denies = _write_denies(C.seatbelt_profile(HOME))
    for name in CRON_FILES:
        assert f'(literal "{data}/{name}")' in denies, (variable, name)


# ── the gate ─────────────────────────────────────────────────────────────────

def _home_cron(name: str) -> str:
    return str(Path.home() / ".prometheus" / "data" / name)


@pytest.mark.parametrize("cfg", [{}, {"permission_mode": "autonomous"},
                                 {"workspace_root": str(Path.home() / ".prometheus")}],
                         ids=["default", "gate-off", "workspace-covers-it"])
@pytest.mark.parametrize("name", CRON_FILES)
def test_the_gate_refuses_writing_a_scheduled_jobs_file(cfg, name):
    decision = create_security_gate(cfg).evaluate("write_file", file_path=_home_cron(name), path_is_write=True)
    assert not decision.allowed and not decision.requires_confirmation, (cfg, name, decision)


def test_the_gate_refuses_a_cron_file_under_any_home():
    # The always-denied glob /*/.prometheus/data/cron_jobs.json covers every user's home, not only this process's.
    # The workspace is set to cover that home, so without the glob the write would be allowed — the denial is the
    # floor, not the boundary.
    other = "/home/someone-else"
    gate = create_security_gate({"workspace_root": other})
    inside = gate.evaluate("write_file", file_path=f"{other}/notes.txt", path_is_write=True)
    assert inside.allowed, ("control: an ordinary write in that workspace is allowed", inside)
    decision = gate.evaluate("write_file", file_path=f"{other}/.prometheus/data/cron_jobs.json", path_is_write=True)
    assert not decision.allowed and not decision.requires_confirmation, decision


def test_the_gate_follows_a_moved_data_directory_for_cron(monkeypatch, tmp_path):
    # A data dir moved by PROMETHEUS_DATA_DIR is outside the /*/.prometheus/... glob, so the gate must register
    # this process's own cron files too.
    moved = tmp_path.resolve() / "elsewhere"
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(moved))
    decision = create_security_gate({}).evaluate(
        "write_file", file_path=str(moved / "cron_jobs.json"), path_is_write=True)
    assert not decision.allowed and not decision.requires_confirmation, decision


def test_the_gate_refuses_a_file_tool_read_too():
    # Like the device registry: the floor denies the file tools outright (read and write). The daemon reads the
    # schedule itself, and a shell can still read it for backups (the real macOS test below); the agent manages
    # cron through cron_create / cron_list, never by reading or writing this file.
    gate = create_security_gate({"workspace_root": str(Path.home() / ".prometheus")})
    assert not gate.evaluate("read_file", file_path=_home_cron("cron_jobs.json"), is_read_only=True).allowed


# ── AppArmor profile text ────────────────────────────────────────────────────

def _active_rules() -> list[str]:
    return [re.sub(r"\s+", " ", line.split("#", 1)[0].strip()) for line in PROFILE_FILE.read_text().splitlines()
            if line.split("#", 1)[0].strip()]


def test_the_apparmor_profile_refuses_the_scheduled_jobs_files():
    rules = _active_rules()
    assert "deny @{HOME}/.prometheus/data/{cron_jobs.json,cron_history.jsonl,cron_undelivered.jsonl} wl," in rules, \
        rules


# ── the real thing (macOS only) ──────────────────────────────────────────────

real = pytest.mark.skipif(not (sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists()),
                          reason="needs macOS with /usr/bin/sandbox-exec")


def _seed(home: Path) -> Path:
    data = home / ".prometheus" / "data"
    data.mkdir(parents=True)
    (data / "cron_jobs.json").write_text('{"jobs": []}\n')
    (data / "cron_history.jsonl").write_text("")
    (data / "cron_undelivered.jsonl").write_text("")
    return data


def _sh(home: Path, command: str) -> int:
    return subprocess.run([C.SANDBOX_EXEC, "-p", C.seatbelt_profile(home), "/bin/sh", "-c", command],
                          capture_output=True, text=True, timeout=30).returncode


@real
@pytest.mark.parametrize("name", CRON_FILES)
def test_a_shell_cannot_plant_or_change_a_scheduled_jobs_file(tmp_path, name):
    home = tmp_path.resolve()
    data = _seed(home)
    f = data / name
    before = f.read_text()
    assert _sh(home, f'printf x >> "{f}"') != 0
    assert _sh(home, f'cp "{f}" "{home}/c" && printf y >> "{home}/c" && mv "{home}/c" "{f}"') != 0
    assert _sh(home, f'rm "{f}"') != 0
    assert f.read_text() == before


@real
def test_a_shell_cannot_create_a_new_cron_jobs_file(tmp_path):
    home = tmp_path.resolve()
    data = home / ".prometheus" / "data"
    data.mkdir(parents=True)
    assert _sh(home, f'printf "{{}}" > "{data}/cron_jobs.json"') != 0
    assert not (data / "cron_jobs.json").exists()


@real
def test_a_shell_can_still_read_the_schedule(tmp_path):
    home = tmp_path.resolve()
    data = _seed(home)
    assert _sh(home, f'cat "{data}/cron_jobs.json" "{data}/cron_history.jsonl" >/dev/null') == 0
