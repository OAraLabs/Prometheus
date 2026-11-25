"""The Environment section's "- Working directory:" line follows the run's cwd.

The boot prompt carries the daemon's own cwd. Under launchd that is ``/``,
so every Beacon conversation was told its working directory was the
filesystem root while its tools ran in the bound session workspace — the
prompt contradicted its own "# Workspace" section. Item W swapped the
project section per run but never touched this line.

Pinned here, beside the model-identity rewrite's tests (same defect class,
same discipline: the rewriter lives beside its producer and matches its
exact format):

* the rewrite replaces the line the environment section writes, once;
* a prompt without the line is returned unchanged (custom base prompts);
* the rewrite never stacks a second line.
"""

from __future__ import annotations

from prometheus.context.environment import get_environment_info
from prometheus.context.system_prompt import (
    _format_environment_section,
    rewrite_working_directory,
)


def test_rewrites_the_line_the_environment_section_writes(tmp_path) -> None:
    boot = tmp_path / "boot"; boot.mkdir()
    ws = tmp_path / "ws"; ws.mkdir()
    prompt = "BASE\n\n" + _format_environment_section(get_environment_info(cwd=str(boot)))
    out = rewrite_working_directory(prompt, ws)
    assert f"- Working directory: {ws}" in out
    assert f"- Working directory: {boot}" not in out
    assert out.count("- Working directory:") == 1


def test_no_line_is_left_alone() -> None:
    prompt = "You are a custom agent.\n\n# Environment\n- OS: macOS"
    assert rewrite_working_directory(prompt, "/somewhere") == prompt


def test_only_the_first_line_is_rewritten() -> None:
    prompt = "- Working directory: /a\nbody\n- Working directory: /b"
    out = rewrite_working_directory(prompt, "/ws")
    assert out.startswith("- Working directory: /ws\n")
    assert "- Working directory: /b" in out  # a second occurrence is not ours to touch


def test_repeated_rewrites_do_not_stack(tmp_path) -> None:
    boot = tmp_path / "boot"; boot.mkdir()
    prompt = _format_environment_section(get_environment_info(cwd=str(boot)))
    out = rewrite_working_directory(prompt, "/one")
    out = rewrite_working_directory(out, "/two")
    assert out.count("- Working directory:") == 1
    assert "- Working directory: /two" in out