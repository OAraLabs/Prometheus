"""What a CLIENT checks before it trusts a downloaded Prometheus.app, and what it must never run.

A client (Beacon, or packaging/macos/clean_mac_check.sh on a stock Mac) runs on a Mac without the developer
tools, where ``/usr/bin/stapler`` and ``xcrun`` are Xcode shims that open the "install the command line developer
tools" dialog. Its checks are the codesign requirement (signed, intact, and by OUR team), Gatekeeper on the
bundle (``spctl -a -t exec``) and ``syspolicy_check`` on macOS 14 or later. ``stapler validate`` belongs to the
build machine (verify_app.py --notarized).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN_MAC_CHECK = ROOT / "packaging" / "macos" / "clean_mac_check.sh"
DESIGN = ROOT / "docs" / "design" / "macos-app-installer.md"
REQUIREMENT = 'anchor apple generic and certificate leaf[subject.OU] = "53JM8W47RL"'


def _commands(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")]


def test_the_clean_mac_check_never_runs_an_xcode_shim():
    for line in _commands(CLEAN_MAC_CHECK):
        assert "stapler" not in line and "xcrun" not in line, line.strip()


def test_the_clean_mac_check_asks_the_three_client_questions():
    commands = "\n".join(_commands(CLEAN_MAC_CHECK))
    assert f"-R={REQUIREMENT}" in commands, "the signature must be checked as OURS, not only as valid"
    assert "spctl -a -t exec" in commands and "syspolicy_check distribution" in commands


def test_the_contract_tells_a_client_the_requirement_and_not_stapler():
    text = DESIGN.read_text(encoding="utf-8")
    contract = text[text.index("## The contract for a client"):]
    checks = contract[contract.index("Check, in this order"):contract.index("**Launcher modes**")]
    assert REQUIREMENT in checks and "spctl -a -t exec" in checks and "syspolicy_check distribution" in checks
    numbered = [line for line in checks.splitlines() if line[:3].strip().rstrip(".").isdigit()]
    assert numbered and not any("stapler" in line for line in numbered), numbered
