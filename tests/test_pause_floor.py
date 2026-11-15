"""pause.json sits in the trusted-state floor (security review of #918, MEDIUM-2; Will, 2026-11-12).

The pause state is read at boot, so a shell that deleted or rewrote it would undo an emergency stop at the next
restart. It is floored like the scheduled-jobs files: by the gate (file tools), by Seatbelt (macOS bash) and by
the AppArmor profile (Linux bash), by PREFIX, so the daemon's own ``pause.json.<hex>.tmp`` (written, then renamed
over) is covered too. Reading it is not refused.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from prometheus.__main__ import create_security_gate
from prometheus.config.shipped_defaults import DENIED_PATHS_DEFAULTS_SHIPPED, SHIPPED_DENIED_PATHS
from prometheus.engine import pause
from prometheus.permissions import confinement as C

ROOT = Path(__file__).resolve().parent.parent
PROFILE_FILE = ROOT / "deploy" / "apparmor" / "prometheus-bash"
NAMES = ("pause.json", "pause.json.0a1b2c3d.tmp")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for name in ("PROMETHEUS_DATA_DIR", "PROMETHEUS_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)
    C.reset_cache()
    yield
    C.reset_cache()


def test_it_is_one_of_the_trusted_data_files():
    assert "pause.json" in C.DATA_TRUSTED_FILES


def test_the_seatbelt_rule_covers_it_and_its_temp_file_and_nothing_beside_it():
    match = re.search(r'regex #"(.*)"\)$', C._CRON_REGEX)
    assert match, C._CRON_REGEX
    rx = re.compile(match.group(1))
    for name in NAMES:
        assert rx.search(f"/Users/someone/.prometheus/data/{name}"), name
    assert not rx.search("/Users/someone/.prometheus/data/notes.json")
    assert not rx.search("/Users/someone/.prometheus/data/my-pause.json")


def test_the_apparmor_profile_refuses_it():
    rules = [line.strip() for line in PROFILE_FILE.read_text().splitlines() if not line.strip().startswith("#")]
    assert "deny @{HOME}/.prometheus/data/pause.json* wl," in rules


def test_the_shipped_denied_paths_name_it_and_an_older_shipped_list_still_upgrades():
    assert "/*/.prometheus/data/pause.json*" in SHIPPED_DENIED_PATHS
    previous = DENIED_PATHS_DEFAULTS_SHIPPED[-2]
    assert "/*/.prometheus/data/pause.json*" not in previous
    assert set(previous) < set(SHIPPED_DENIED_PATHS)
    template = (ROOT / "config" / "prometheus.yaml.default").read_text()
    assert '- "/*/.prometheus/data/pause.json*"' in template


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("cfg", [{}, {"permission_mode": "autonomous"},
                                 {"workspace_root": str(Path.home() / ".prometheus")}],
                         ids=["default", "gate-off", "workspace-covers-it"])
def test_the_gate_refuses_writing_it(cfg, name):
    target = str(Path.home() / ".prometheus" / "data" / name)
    decision = create_security_gate(cfg).evaluate("write_file", file_path=target, path_is_write=True)
    assert not decision.allowed and not decision.requires_confirmation, (cfg, name, decision)


def test_the_daemon_writes_its_temp_file_under_the_floored_prefix(tmp_path, monkeypatch):
    made: list[str] = []
    real_open = pause.os.open

    def spy(path, *a, **kw):
        made.append(Path(path).name)
        return real_open(path, *a, **kw)

    monkeypatch.setattr(pause.os, "open", spy)
    pause.PauseState(tmp_path / "pause.json").pause(by="test")
    temps = [n for n in made if n != tmp_path.name]
    assert temps and all(n.startswith("pause.json.") and n.endswith(".tmp") for n in temps), made


real = pytest.mark.skipif(not (sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists()),
                          reason="needs macOS with /usr/bin/sandbox-exec")


def _sh(home: Path, command: str) -> int:
    return subprocess.run([C.SANDBOX_EXEC, "-p", C.seatbelt_profile(home), "/bin/sh", "-c", command],
                          capture_output=True, text=True, timeout=30).returncode


@real
def test_a_shell_cannot_change_delete_or_create_it(tmp_path):
    home = tmp_path.resolve()
    data = home / ".prometheus" / "data"
    data.mkdir(parents=True)
    f = data / "pause.json"
    f.write_text('{"paused": true, "since": 1.0, "by": "test"}')
    before = f.read_text()
    assert _sh(home, f'printf x >> "{f}"') != 0
    assert _sh(home, f'rm "{f}"') != 0
    assert _sh(home, f'cp "{f}" "{home}/c" && mv "{home}/c" "{f}"') != 0
    assert f.read_text() == before
    f.unlink()
    assert _sh(home, f'printf "{{}}" > "{f}"') != 0 and not f.exists()
    assert _sh(home, f'cat "{data}/../data/pause.json" 2>/dev/null; true') == 0


@real
def test_a_shell_can_still_read_it(tmp_path):
    home = tmp_path.resolve()
    data = home / ".prometheus" / "data"
    data.mkdir(parents=True)
    (data / "pause.json").write_text('{"paused": false}')
    assert _sh(home, f'cat "{data}/pause.json" > /dev/null') == 0
