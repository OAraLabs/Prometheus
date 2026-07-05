"""``prometheus install-service`` — install the systemd user unit.

Onboarding Phase 0, item 3: the README has promised
``systemctl --user enable --now prometheus`` for months without shipping
a unit. This writes ``packaging/prometheus.service`` (with ExecStart
resolved to the installed ``prometheus`` binary) to
``~/.config/systemd/user/``, runs ``daemon-reload``, and enables it.

Safety properties (tested):
- REFUSES to overwrite an existing unit unless ``--force`` is given —
  a machine already running a hand-rolled prometheus.service is never
  clobbered.
- Idempotent: re-running when the installed unit is byte-identical is a
  no-op success.
- Never starts/restarts anything unless ``--now`` is passed; enabling
  only wires the unit for the next login/boot.
- ``--systemd-dir`` / ``PROMETHEUS_SYSTEMD_USER_DIR`` override the target
  directory (tests point this at a tmp dir).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

UNIT_NAME = "prometheus.service"

# Kept in sync with packaging/prometheus.service (test-enforced):
# {exec_start} is the only render-time substitution.
UNIT_TEMPLATE = """\
[Unit]
Description=Prometheus AI Agent Daemon
After=network.target

[Service]
Type=simple
ExecStart={exec_start}
ExecStop=/bin/kill -SIGTERM $MAINPID
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Secrets (PROMETHEUS_API_TOKEN, gateway tokens, provider keys) live in
# the env file, written by `prometheus setup` / `prometheus token rotate`.
# The leading "-" makes it optional so a fresh install still boots.
EnvironmentFile=-%h/.config/prometheus/env

[Install]
WantedBy=default.target
"""

DEFAULT_EXEC_START = "/usr/bin/env prometheus daemon"


def get_systemd_user_dir() -> Path:
    """Target directory for the user unit (env-overridable for tests)."""
    override = os.environ.get("PROMETHEUS_SYSTEMD_USER_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "systemd" / "user"


def resolve_exec_start() -> str:
    """ExecStart line resolving the installed ``prometheus`` binary.

    Falls back to a PATH lookup at unit start when the binary isn't
    findable right now (e.g. editable install outside PATH).
    """
    binary = shutil.which("prometheus")
    if binary:
        return f"{binary} daemon"
    return DEFAULT_EXEC_START


def render_unit(exec_start: str | None = None) -> str:
    """Render the unit file content."""
    return UNIT_TEMPLATE.format(exec_start=exec_start or resolve_exec_start())


def install_service(
    *,
    systemd_dir: Path | None = None,
    force: bool = False,
    now: bool = False,
    runner=subprocess.run,
) -> int:
    """Install the user unit. Returns an exit code.

    ``runner`` is injectable so tests never invoke the real systemctl.
    """
    systemd_dir = systemd_dir or get_systemd_user_dir()
    target = systemd_dir / UNIT_NAME
    content = render_unit()

    if target.exists():
        existing = target.read_text(encoding="utf-8")
        if existing == content:
            print(f"Already installed and up to date: {target}")
            return 0
        if not force:
            print(f"REFUSING to overwrite existing unit: {target}")
            print("It differs from what install-service would write.")
            print("Inspect it, then re-run with --force to replace it.")
            return 1
        backup = target.with_suffix(".service.bak")
        backup.write_text(existing, encoding="utf-8")
        print(f"Existing unit backed up to {backup}")

    systemd_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"Wrote {target}")

    for cmd in (
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", UNIT_NAME],
    ):
        try:
            result = runner(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            print("systemctl not found — unit written, but you must enable it "
                  "manually on this system.")
            return 0
        if result.returncode != 0:
            print(f"WARNING: {' '.join(cmd)} failed: "
                  f"{(result.stderr or result.stdout or '').strip()}")
            return 1
        print(f"Ran: {' '.join(cmd)}")

    if now:
        result = runner(["systemctl", "--user", "start", UNIT_NAME],
                        capture_output=True, text=True)
        if result.returncode != 0:
            print(f"WARNING: start failed: {(result.stderr or '').strip()}")
            return 1
        print(f"Started {UNIT_NAME}")
    else:
        print(f"Enabled. Start it with: systemctl --user start prometheus")
    return 0


def run_install_service_command(args: argparse.Namespace) -> int:
    """Entry point for ``prometheus install-service``."""
    systemd_dir = (
        Path(args.systemd_dir).expanduser() if getattr(args, "systemd_dir", None)
        else None
    )
    return install_service(
        systemd_dir=systemd_dir,
        force=bool(getattr(args, "force", False)),
        now=bool(getattr(args, "now", False)),
    )


def add_install_service_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``install-service`` subcommand."""
    p = subparsers.add_parser(
        "install-service",
        help="Install the systemd user unit (daemon-reload + enable)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing prometheus.service (backs it up first)",
    )
    p.add_argument(
        "--now", action="store_true",
        help="Also start the service immediately after enabling",
    )
    p.add_argument(
        "--systemd-dir", default=None,
        help="Target directory (default: ~/.config/systemd/user)",
    )


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(prog="prometheus install-service")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--now", action="store_true")
    parser.add_argument("--systemd-dir", default=None)
    sys.exit(run_install_service_command(parser.parse_args()))
