"""The scheduled-jobs files are under the shell floor: a model's shell cannot plant a job that runs as the owner.

cron_jobs.json (config/paths.py) is the schedule, re-read every scheduler tick (gateway/cron_scheduler.py), so a
job written into it runs on its next due time with no restart, as the owner, through the shell floor — persistence
beyond the session. cron_history.jsonl and the resend queue cron_undelivered.jsonl are its logs (the queue is sent
to the owner verbatim), and cron_scheduler.pid says whether the scheduler runs.

The daemon writes the schedule through a sibling cron_jobs.json.tmp and then os.replace, and moves a corrupt file to
cron_jobs.json.corrupt (gateway/cron_service.py). Those side files are covered too: a shell that planted
cron_jobs.json.tmp as a symlink would make the daemon write job JSON through it, to wherever it points.

All of them live in the data directory, like the device registry, and are floored the same way, by prefix: Seatbelt
and AppArmor deny a shell's writes and links, the gate refuses the file tools, reads still work, and the daemon
(unconfined) still writes them. Same class as GHSA-wq93.
"""

from __future__ import annotations

import logging
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
CRON_BASES = ("cron_jobs.json", "cron_history.jsonl", "cron_undelivered.jsonl", "cron_scheduler.pid")
#: The daemon's own side files for the schedule: the atomic-write temp and the corrupt-file backup.
CRON_SIDE = ("cron_jobs.json.tmp", "cron_jobs.json.corrupt")
CRON_ALL = CRON_BASES + CRON_SIDE


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for name in ("PROMETHEUS_DATA_DIR", "PROMETHEUS_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)
    C.reset_cache()
    yield
    C.reset_cache()


def _write_denies(profile: str) -> str:
    return "\n".join(block for block in profile.split("(deny file-write*")[1:])


def _cron_rx() -> re.Pattern[str]:
    """The Seatbelt regex the profile uses, compiled, so its reach can be measured on paths."""
    match = re.search(r'regex #"(.*)"\)$', C._CRON_REGEX)
    assert match, C._CRON_REGEX
    return re.compile(match.group(1))


# ── Seatbelt profile text ────────────────────────────────────────────────────

def test_the_profile_refuses_writes_to_every_scheduled_jobs_file():
    denies = _write_denies(C.seatbelt_profile(HOME))
    assert f'(require-all (subpath "{DATA}") {C._CRON_REGEX})' in denies, denies
    rx = _cron_rx()
    for name in CRON_ALL:
        assert rx.search(f"{DATA}/{name}"), name


def test_the_cron_rule_is_narrow():
    # Files beside the schedule stay the shell's: only the cron_* names the daemon owns are matched.
    rx = _cron_rx()
    for neighbour in ("notes.txt", "cron.log", "memories.db", "my_cron_jobs.json", "cron_jobs", "cron_notes.json"):
        assert not rx.search(f"{DATA}/{neighbour}"), neighbour


@pytest.mark.parametrize("variable, value", [
    ("PROMETHEUS_DATA_DIR", "wired/data"),
    ("PROMETHEUS_CONFIG_DIR", "conf"),
], ids=["data-dir", "config-dir"])
def test_a_moved_data_directory_is_covered(monkeypatch, tmp_path, variable, value):
    root = tmp_path.resolve()
    monkeypatch.setenv(variable, str(root / value))
    data = root / ("wired/data" if variable == "PROMETHEUS_DATA_DIR" else "conf/data")
    denies = _write_denies(C.seatbelt_profile(HOME))
    assert f'(require-all (subpath "{data}") {C._CRON_REGEX})' in denies, (variable, denies)


# ── the gate ─────────────────────────────────────────────────────────────────

def _home_cron(name: str) -> str:
    return str(Path.home() / ".prometheus" / "data" / name)


@pytest.mark.parametrize("cfg", [{}, {"permission_mode": "autonomous"},
                                 {"workspace_root": str(Path.home() / ".prometheus")}],
                         ids=["default", "gate-off", "workspace-covers-it"])
@pytest.mark.parametrize("name", CRON_ALL)
def test_the_gate_refuses_writing_a_scheduled_jobs_file(cfg, name):
    decision = create_security_gate(cfg).evaluate("write_file", file_path=_home_cron(name), path_is_write=True)
    assert not decision.allowed and not decision.requires_confirmation, (cfg, name, decision)


@pytest.mark.parametrize("name", ("cron_jobs.json", "cron_jobs.json.tmp", "cron_scheduler.pid"))
def test_the_gate_refuses_a_cron_file_under_any_home(name):
    # The always-denied globs (/*/.prometheus/data/cron_jobs.json* …) cover every user's home, not only this
    # process's. The workspace is set to cover that home, so without the glob the write would be allowed — the
    # denial is the floor, not the boundary.
    other = "/home/someone-else"
    gate = create_security_gate({"workspace_root": other})
    inside = gate.evaluate("write_file", file_path=f"{other}/notes.txt", path_is_write=True)
    assert inside.allowed, ("control: an ordinary write in that workspace is allowed", inside)
    decision = gate.evaluate("write_file", file_path=f"{other}/.prometheus/data/{name}", path_is_write=True)
    assert not decision.allowed and not decision.requires_confirmation, (name, decision)


@pytest.mark.parametrize("name", ("cron_jobs.json", "cron_jobs.json.tmp", "cron_jobs.json.corrupt",
                                  "cron_scheduler.pid"))
def test_the_gate_follows_a_moved_data_directory_for_cron(monkeypatch, tmp_path, name):
    # A data dir moved by PROMETHEUS_DATA_DIR is outside the /*/.prometheus/... glob, so the gate must register
    # this process's own cron files too — by prefix, so the side files are covered.
    moved = tmp_path.resolve() / "elsewhere"
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(moved))
    decision = create_security_gate({}).evaluate("write_file", file_path=str(moved / name), path_is_write=True)
    assert not decision.allowed and not decision.requires_confirmation, (name, decision)


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
    rule = "deny @{HOME}/.prometheus/data/cron_{jobs.json,history.jsonl,undelivered.jsonl,scheduler.pid}* wl,"
    assert rule in rules, rules


# ── the boot probe (AppArmor) ────────────────────────────────────────────────

def _label_ok(monkeypatch, *, cron: str = "refused"):
    monkeypatch.setattr(C, "_probe_label", lambda profile: (True, f"{profile} (enforce)"))
    monkeypatch.setattr(C, "_probe_config_rename", lambda profile: "refused")
    monkeypatch.setattr(C, "_registry_detail", lambda profile: ("; writing the device registry refused", False))
    monkeypatch.setattr(C, "_trusted_detail", lambda profile: ("; writing trusted state refused", False))
    monkeypatch.setattr(C, "_probe_cron_rename", lambda profile: cron)


def test_a_current_profile_says_the_scheduled_jobs_files_are_refused(monkeypatch, caplog):
    _label_ok(monkeypatch)
    with caplog.at_level(logging.INFO, logger=C.logger.name):
        ok, detail = C.preflight(C.PROFILE, force=True)
    assert ok and "writing the scheduled-jobs files refused" in detail, detail
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_a_profile_without_the_cron_rule_is_stale_and_says_how_to_reload(monkeypatch, caplog):
    _label_ok(monkeypatch, cron="renamed")
    with caplog.at_level(logging.INFO, logger=C.logger.name):
        ok, detail = C.preflight(C.PROFILE, force=True)
    assert ok, "a stale profile keeps the floor it has"
    assert "STALE PROFILE" in detail and "scheduled" in detail and C.RELOAD_COMMAND in detail, detail
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors and C.RELOAD_COMMAND in errors[0], errors


def test_a_moved_data_directory_names_the_scheduled_jobs_files(monkeypatch, tmp_path):
    """The profile names @{HOME}/.prometheus/data; a data directory moved elsewhere leaves the cron files outside it
    as well as the registry, and the report says so."""
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.setattr(C, "_probe_registry_rename", lambda profile: "refused")
    detail, stale = C._registry_detail(C.PROFILE)
    assert stale and "outside" in detail and "scheduled" in detail and "PROMETHEUS_DATA_DIR" in detail, detail


# ── the real thing (macOS only) ──────────────────────────────────────────────

real = pytest.mark.skipif(not (sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists()),
                          reason="needs macOS with /usr/bin/sandbox-exec")


def _seed(home: Path) -> Path:
    data = home / ".prometheus" / "data"
    data.mkdir(parents=True)
    (data / "cron_jobs.json").write_text('{"jobs": []}\n')
    (data / "cron_jobs.json.corrupt").write_text("not json\n")
    (data / "cron_history.jsonl").write_text("")
    (data / "cron_undelivered.jsonl").write_text("")
    (data / "cron_scheduler.pid").write_text("12345\n")
    return data


def _sh(home: Path, command: str) -> int:
    return subprocess.run([C.SANDBOX_EXEC, "-p", C.seatbelt_profile(home), "/bin/sh", "-c", command],
                          capture_output=True, text=True, timeout=30).returncode


@real
@pytest.mark.parametrize("name", [n for n in CRON_ALL if n != "cron_jobs.json.tmp"])
def test_a_shell_cannot_change_a_scheduled_jobs_file(tmp_path, name):
    home = tmp_path.resolve()
    data = _seed(home)
    f = data / name
    before = f.read_text()
    assert _sh(home, f'printf x >> "{f}"') != 0
    assert _sh(home, f'cp "{f}" "{home}/c" && printf y >> "{home}/c" && mv "{home}/c" "{f}"') != 0
    assert _sh(home, f'rm "{f}"') != 0
    assert f.read_text() == before


@real
@pytest.mark.parametrize("name", ("cron_jobs.json", "cron_jobs.json.tmp", "cron_undelivered.jsonl",
                                  "cron_scheduler.pid"))
def test_a_shell_cannot_create_a_scheduled_jobs_file(tmp_path, name):
    home = tmp_path.resolve()
    data = home / ".prometheus" / "data"
    data.mkdir(parents=True)
    assert _sh(home, f'printf "{{}}" > "{data}/{name}"') != 0
    assert not (data / name).exists()


@real
@pytest.mark.parametrize("how", ["ln -s", "ln"], ids=["symlink", "hard-link"])
def test_a_shell_cannot_plant_a_link_as_the_schedule_temp(tmp_path, how):
    # save_cron_jobs opens cron_jobs.json.tmp with open(tmp, "w"), which follows a symlink: a planted link would make
    # the daemon overwrite its target with job JSON, then os.replace would move the link into place.
    home = tmp_path.resolve()
    data = _seed(home)
    target = home / "outside.txt"
    target.write_text("keep me\n")
    link = data / "cron_jobs.json.tmp"
    assert _sh(home, f'{how} "{target}" "{link}"') != 0
    assert not link.exists() and not link.is_symlink()
    assert target.read_text() == "keep me\n"


@real
def test_a_shell_cannot_plant_a_link_as_the_resend_queue(tmp_path):
    home = tmp_path.resolve()
    data = home / ".prometheus" / "data"
    data.mkdir(parents=True)
    target = home / "outside.txt"
    target.write_text("keep me\n")
    assert _sh(home, f'ln -s "{target}" "{data}/cron_undelivered.jsonl"') != 0
    assert not (data / "cron_undelivered.jsonl").is_symlink()


@real
def test_a_shell_can_still_read_the_schedule(tmp_path):
    home = tmp_path.resolve()
    data = _seed(home)
    names = " ".join(f'"{data / n}"' for n in CRON_BASES + ("cron_jobs.json.corrupt",))
    assert _sh(home, f"cat {names} >/dev/null") == 0


@real
def test_a_shell_can_still_write_beside_the_schedule(tmp_path):
    home = tmp_path.resolve()
    data = _seed(home)
    assert _sh(home, f'printf x > "{data}/notes.txt"') == 0
