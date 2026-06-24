"""Tests for scripts/sync_skills.py — Lane-1 skill sync + leak guard.

The script is pure-stdlib (so the git hook can call it with any python3); it is
not part of the package, so it's loaded by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "sync_skills.py"
_spec = importlib.util.spec_from_file_location("sync_skills", _SCRIPT)
sync_skills = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_skills)


# --- lint_text -------------------------------------------------------------

def test_lint_flags_absolute_home_path():
    findings = sync_skills.lint_text("run `cd /home/will/projects/foo`")
    assert any("/home/will/" in f for f in findings)


def test_lint_allows_placeholder_home_path():
    assert sync_skills.lint_text("Worktree at /home/user/myproject/.worktrees/x") == []


def test_lint_flags_tailscale_ip():
    assert any("Tailscale" in f for f in sync_skills.lint_text("http://100.110.140.39:11434"))


def test_lint_flags_private_domain():
    assert any("private domain" in f for f in sync_skills.lint_text("POST capstonerecruiting.com"))


def test_lint_flags_oara_host_but_not_package_name():
    assert any("OAra" in f for f in sync_skills.lint_text("ssh oara-mini"))
    assert sync_skills.lint_text("the oara-prometheus distribution name") == []


def test_lint_flags_token_shapes():
    assert sync_skills.lint_text("token 1234567890:ABCdefABCdefABCdefABCdefABCdefABCdef12")
    assert sync_skills.lint_text("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")


def test_lint_clean_on_env_references():
    # Env-var references are the safe pattern and must NOT trip the guard.
    assert sync_skills.lint_text('curl -H "Authorization: token $GITHUB_TOKEN"') == []
    assert sync_skills.lint_text("client uses ANTHROPIC_API_KEY env var") == []
    assert sync_skills.lint_text('curl ".../bot${TELEGRAM_BOT_TOKEN}/sendMessage"') == []


# --- lane classification ---------------------------------------------------

def test_printing_press_is_not_lane1():
    body = "---\nname: espn\n---\n  module: github.com/x/printing-press-library/y\n"
    assert sync_skills.is_printing_press(body)
    assert not sync_skills.is_lane1("espn", body)


def test_wordpress_name_excluded_from_lane1():
    assert not sync_skills.is_lane1("wordpress", "any body")


def test_plain_skill_is_lane1():
    assert sync_skills.is_lane1("docker", "---\nname: docker\n---\nbody")


# --- compute / install / check ---------------------------------------------

def _write(d: Path, name: str, body: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.md").write_text(body, encoding="utf-8")


def test_install_converges_userdir(tmp_path):
    repo, user = tmp_path / "repo", tmp_path / "user"
    _write(repo, "alpha", "---\nname: alpha\n---\nNEW")
    _write(repo, "beta", "---\nname: beta\n---\nB")
    _write(user, "alpha", "---\nname: alpha\n---\nOLD")          # drift -> updated
    _write(user, "gamma", "---\nname: gamma\n---\nG")           # not in repo -> removed
    _write(user, "espn", "  module: printing-press-library/x")  # Lane 2 -> kept

    assert sync_skills.cmd_install(repo, user) == 0
    assert (user / "alpha.md").read_text().endswith("NEW")      # updated
    assert (user / "beta.md").exists()                          # installed
    assert not (user / "gamma.md").exists()                     # pruned
    assert (user / "espn.md").exists()                          # Lane 2 untouched


def test_install_refuses_to_sync_a_leak(tmp_path):
    repo, user = tmp_path / "repo", tmp_path / "user"
    _write(repo, "leaky", "ssh into 100.1.2.3 via /home/will/.ssh/id_rsa")
    assert sync_skills.cmd_install(repo, user) == 1
    assert not (user / "leaky.md").exists()                     # nothing written


def test_check_detects_missing_in_userdir(tmp_path):
    repo, user = tmp_path / "repo", tmp_path / "user"
    _write(repo, "alpha", "A")
    user.mkdir()
    assert sync_skills.cmd_check(repo, user) == 1               # drift


def test_check_ok_when_synced(tmp_path):
    repo, user = tmp_path / "repo", tmp_path / "user"
    _write(repo, "alpha", "---\nname: alpha\n---\nA")
    _write(user, "alpha", "---\nname: alpha\n---\nA")
    assert sync_skills.cmd_check(repo, user) == 0
