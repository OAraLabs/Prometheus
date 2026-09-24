"""scripts/deploy.sh — the static half of its contract.

The script's phases need uv, a network, systemd and /proc, so they are
exercised by running it (phase A end to end, and B0/B6 against a live
daemon, in the PR that added it), not here. What CAN rot silently is
checked here: the script parses, refuses bad usage, and the drop-in it
writes still sets PROMETHEUS_VENV — the variable that makes
scripts/deploy_guard.sh compare the venv with the checkout's uv.lock. Drop
that line and the guard's lock check quietly stops applying.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
DEPLOY = SCRIPTS / "deploy.sh"
GUARD = SCRIPTS / "deploy_guard.sh"


def test_the_script_is_present_executable_and_parses():
    assert os.access(DEPLOY, os.X_OK), f"{DEPLOY} is not executable"
    r = subprocess.run(["bash", "-n", str(DEPLOY)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_bad_usage_exits_2_and_help_exits_0():
    for args, want in (([], 2), (["--bogus"], 2), (["a", "b"], 2), (["--help"], 0)):
        r = subprocess.run(["bash", str(DEPLOY), *args], capture_output=True,
                           text=True, timeout=30)
        assert r.returncode == want, (args, r.returncode, r.stderr)
        assert "USAGE" in r.stderr


def test_the_drop_in_arms_the_guards_lock_check():
    text = DEPLOY.read_text(encoding="utf-8")
    dropin = text[text.index('cat > "$DROPIN" <<EOF'):]
    dropin = dropin[:dropin.index("\nEOF\n")]
    assert "Environment=PROMETHEUS_VENV=$ROOT/current" in dropin
    assert "ExecStart=\n" in dropin, "the base ExecStart must be cleared first"
    assert "PROMETHEUS_VENV" in GUARD.read_text(encoding="utf-8")


def test_the_venv_records_the_lock_the_guard_reads():
    assert 'BUILT_FROM_UV_LOCK"' in DEPLOY.read_text(encoding="utf-8")
    assert "/BUILT_FROM_UV_LOCK" in GUARD.read_text(encoding="utf-8")


def test_the_embedded_python_compiles():
    """B0/B6 and G1 are heredocs; a syntax error there would only show up
    mid-deploy, after the daemon was already being switched."""
    text = DEPLOY.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'EOF'[^\n]*\n(.*?)\nEOF\n", text, re.S)
    assert len(blocks) == 3, f"expected G1, B0 and B6 heredocs, found {len(blocks)}"
    for i, src in enumerate(blocks):
        compile(src, f"deploy.sh heredoc #{i}", "exec")
