"""deploy.sh installs the ``computer`` extra ONLY when computer use is on.

cua-driver is a native desktop driver. A box that has not switched computer
use on should not carry it at all: not imported, not audited as part of
what runs, not something a later change can reach by accident. So the deploy
reads the LIVE config (the one the daemon will start with) through the same
literal-``true`` rule the daemon uses, and:

* off  → the extra is not installed, and if ``PROMETHEUS_DEPLOY_EXTRAS``
         names it anyway it is dropped, with a line saying so;
* on   → the extra is installed, G2 audits it, and G4 refuses a venv whose
         cua-driver is not the exact version the adapter was validated on.

The G-checks still pass when it is off: G4 then asserts cua-driver is
ABSENT from the venv.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "scripts" / "deploy.sh"


def _write(tmp_path, text: str) -> Path:
    path = tmp_path / "prometheus.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("text, want", [
    ("computer_use:\n  enabled: true\n", True),
    ("computer_use:\n  enabled: false\n", False),
    ('computer_use:\n  enabled: "true"\n', False),
    ("computer_use:\n  enabled: yes_please\n", False),
    ("other: 1\n", False),
    ("", False),
])
def test_the_extra_follows_the_literal_true_rule(tmp_path, text, want):
    from prometheus.computer.deploy import computer_extra_wanted

    assert computer_extra_wanted(_write(tmp_path, text)) is want


def test_a_missing_or_broken_config_means_off(tmp_path):
    from prometheus.computer.deploy import computer_extra_wanted

    assert computer_extra_wanted(tmp_path / "nope.yaml") is False
    assert computer_extra_wanted(_write(tmp_path, "computer_use: [\n")) is False


@pytest.mark.parametrize("enabled, extras, want", [
    (False, "anthropic mcp push voice", "anthropic mcp push voice"),
    (False, "anthropic computer mcp", "anthropic mcp"),
    (True, "anthropic mcp", "anthropic mcp computer"),
    (True, "anthropic computer", "anthropic computer"),
])
def test_the_cli_rewrites_the_extras(tmp_path, enabled, extras, want):
    path = _write(tmp_path, f"computer_use:\n  enabled: {str(enabled).lower()}\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run(
        [sys.executable, "-m", "prometheus.computer.deploy", "extras",
         str(path), extras],
        capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == want.split()


def test_the_driver_check_passes_off_and_refuses_a_stray_driver(tmp_path):
    from prometheus.computer.deploy import driver_check

    assert driver_check(enabled=False, installed=None) == (True, "computer use is off; cua-driver is not installed")
    ok, why = driver_check(enabled=False, installed="0.28.2")
    assert not ok and "installed although computer use is off" in why
    ok, why = driver_check(enabled=True, installed=None)
    assert not ok and "not installed" in why
    ok, why = driver_check(enabled=True, installed="0.33.1")
    assert not ok and "0.28.2" in why
    assert driver_check(enabled=True, installed="0.28.2")[0] is True


def test_deploy_sh_decides_the_extra_from_the_live_config():
    text = DEPLOY.read_text(encoding="utf-8")
    assert "prometheus.computer.deploy extras" in text, (
        "the extras must be rewritten from the live config before uv sync")
    assert '$CLONE/config/prometheus.yaml' in text, (
        "the config the daemon starts with (ExecStart --config) decides it")
    assert "G4" in text and "prometheus.computer.deploy driver" in text
    r = subprocess.run(["bash", "-n", str(DEPLOY)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
