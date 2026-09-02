"""The command is `oara`; `prometheus` is an alias for the deprecation window.

Item C of the 2026-09-01 roadmap: `pip install oara` (and the Homebrew
formula named oara) must put an `oara` on PATH — a package named oara that
installs a `prometheus` binary collides with the CNCF project in every
shell. The import path and the package stay `prometheus`.
"""

from __future__ import annotations

import io
import sys
import tomllib
from pathlib import Path

import pytest

from prometheus import __main__ as cli

REPO = Path(__file__).resolve().parents[1]


def test_entry_points_name_oara_and_keep_the_alias() -> None:
    scripts = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["oara"] == "prometheus.__main__:main"
    # The daemon is `oara daemon`, a subcommand — no second binary.
    assert not any(k.startswith("oara") and k != "oara" for k in scripts)
    # The alias is the deprecation window: same target, old name.
    assert scripts["prometheus"] == scripts["oara"]
    assert scripts["prometheus-daemon"] == "prometheus.daemon:main"


@pytest.mark.parametrize("argv0,expected", [
    ("/venv/bin/oara", "oara"),
    ("/venv/bin/prometheus", "prometheus"),
    ("/usr/lib/python3.12/site-packages/prometheus/__main__.py", "oara"),  # python -m prometheus
    ("", "oara"),
])
def test_prog_follows_the_invoked_name(monkeypatch, argv0, expected) -> None:
    """`--help` must never tell a user to type a name they did not use."""
    monkeypatch.setattr(sys, "argv", [argv0])
    assert cli.invoked_command_name() == expected


def _run_help(monkeypatch, argv0: str) -> str:
    monkeypatch.setattr(sys, "argv", [argv0, "--help"])
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    with pytest.raises(SystemExit):
        cli.main()
    return out.getvalue()


def test_usage_line_uses_the_invoked_name(monkeypatch) -> None:
    assert _run_help(monkeypatch, "/venv/bin/oara").startswith("usage: oara")
    assert _run_help(monkeypatch, "/venv/bin/prometheus").startswith("usage: prometheus")


def test_default_exec_start_prefers_oara(monkeypatch) -> None:
    from prometheus.cli import service

    seen = []

    def fake_which(name):
        seen.append(name)
        return {"oara": "/opt/venv/bin/oara"}.get(name)

    monkeypatch.setattr(service.shutil, "which", fake_which)
    assert service.resolve_exec_start() == "/opt/venv/bin/oara daemon"
    assert seen[0] == "oara"                                   # asked for the new name first

    monkeypatch.setattr(service.shutil, "which", lambda name: {"prometheus": "/old/bin/prometheus"}.get(name))
    assert service.resolve_exec_start() == "/old/bin/prometheus daemon"   # alias-only installs still work
    assert service.DEFAULT_EXEC_START == "/usr/bin/env oara daemon"
