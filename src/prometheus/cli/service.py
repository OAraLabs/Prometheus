"""``oara install-service`` — install the daemon as a user service.

Onboarding Phase 0, item 3: the README has promised
``systemctl --user enable --now prometheus`` for months without shipping
a unit. On Linux this writes ``packaging/prometheus.service`` (with ExecStart
resolved to the installed ``oara`` binary) to ``~/.config/systemd/user/``,
runs ``daemon-reload``, and enables it. On macOS there is no systemd: it
writes a LaunchAgent plist (``cli/launchd.py``) to ``~/Library/LaunchAgents/``
instead, which launchd loads at the next login — the analogue of
``systemctl enable``.

Safety properties (tested), the same on both platforms:
- REFUSES to overwrite an existing unit/plist unless ``--force`` is given —
  a machine already running a hand-rolled one is never clobbered, and
  ``--force`` backs the old one up first.
- Idempotent: re-running when the installed file is byte-identical is a
  no-op success, and touches neither systemctl nor launchctl.
- Never starts/restarts anything unless ``--now`` is passed; installing
  only wires the service for the next login/boot.
- Never exits 0 having enabled nothing: a missing systemctl/launchctl, or a
  failing one, is exit 1 with the reason. (This used to print "enable it
  manually" and return 0.)
- ``--systemd-dir`` / ``PROMETHEUS_SYSTEMD_USER_DIR`` and ``--launch-agents-dir``
  / ``PROMETHEUS_LAUNCH_AGENTS_DIR`` override the target directory (tests
  point these at a tmp dir). Each is meaningless on the other platform and
  exits 2 there rather than being silently ignored.

macOS only: if the job label is already loaded in launchd but there is no
plist of ours, another supervisor (an app that registered it, or a
hand-rolled plist) owns the daemon. That is refused even with ``--force``,
which replaces OUR plist and nothing else.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from prometheus.cli import launchd
from prometheus.cli.launchd import Runner

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
# the env file, written by `oara setup` / `oara token rotate`.
# The leading "-" makes it optional so a fresh install still boots.
EnvironmentFile=-%h/.config/prometheus/env

[Install]
WantedBy=default.target
"""

DEFAULT_EXEC_START = "/usr/bin/env oara daemon"


def get_systemd_user_dir() -> Path:
    """Target directory for the user unit (env-overridable for tests)."""
    override = os.environ.get("PROMETHEUS_SYSTEMD_USER_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "systemd" / "user"


def _host_platform() -> str:
    """``sys.platform``, behind a function so tests can fake a host."""
    return sys.platform


def resolve_binary() -> str | None:
    """The installed ``oara`` (or its ``prometheus`` alias), if on PATH now."""
    return shutil.which("oara") or shutil.which("prometheus")


def resolve_exec_start() -> str:
    """ExecStart line resolving the installed ``prometheus`` binary.

    Falls back to a PATH lookup at unit start when the binary isn't
    findable right now (e.g. editable install outside PATH).
    """
    binary = resolve_binary()
    if binary:
        return f"{binary} daemon"
    return DEFAULT_EXEC_START


def render_unit(exec_start: str | None = None) -> str:
    """Render the unit file content."""
    return UNIT_TEMPLATE.format(exec_start=exec_start or resolve_exec_start())


def install_service(
    *,
    systemd_dir: Path | None = None,
    launch_agents_dir: Path | None = None,
    force: bool = False,
    now: bool = False,
    runner: Runner = subprocess.run,
    platform: str | None = None,
) -> int:
    """Install the user service for this platform. Returns an exit code.

    0 installed (or already up to date); 1 refused or failed; 2 an option that
    does not apply on this platform. ``runner`` is injectable so tests never
    invoke the real systemctl/launchctl, and ``platform`` (default: the host's
    ``sys.platform``) so they can exercise either path on any machine.
    """
    platform = platform or _host_platform()
    if platform == "darwin":
        if systemd_dir is not None:
            print("--systemd-dir is not applicable on macOS: the service is a "
                  "LaunchAgent, not a systemd unit. Use --launch-agents-dir to "
                  "choose where its plist goes.")
            return 2
        return _install_launchagent(
            agents_dir=launch_agents_dir, force=force, now=now, runner=runner,
        )
    if launch_agents_dir is not None:
        print("--launch-agents-dir is only for macOS. On this system the service "
              "is a systemd user unit: use --systemd-dir to choose where it goes.")
        return 2
    return _install_systemd(
        systemd_dir=systemd_dir, force=force, now=now, runner=runner,
    )


def _install_systemd(
    *,
    systemd_dir: Path | None,
    force: bool,
    now: bool,
    runner: Runner,
) -> int:
    """Linux: write the user unit, ``daemon-reload``, ``enable``."""
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
            # The unit is on disk but nothing is enabled. That is a failure,
            # not a success with a footnote: a script that checks the exit
            # code must not conclude the daemon will start.
            print(f"ERROR: systemctl not found, so nothing was enabled. The unit "
                  f"was written to {target} and is left in place. This system "
                  f"may not use systemd; once it does, enable it with: "
                  f"systemctl --user daemon-reload && "
                  f"systemctl --user enable {UNIT_NAME}")
            return 1
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
        print("Enabled. Start it with: systemctl --user start prometheus")
    return 0


def _install_launchagent(
    *,
    agents_dir: Path | None,
    force: bool,
    now: bool,
    runner: Runner,
) -> int:
    """macOS: write the LaunchAgent plist; with ``--now``, load it."""
    binary = resolve_binary()
    if binary is None:
        # The systemd unit can fall back to `/usr/bin/env oara` and hope for
        # the PATH at start time; a LaunchAgent that cannot find its program
        # would just crash-loop every ten seconds. Better to say so now.
        print("ERROR: could not find the `oara` binary on PATH, so there is "
              "nothing for the LaunchAgent to run. Put `oara` on PATH and "
              "re-run. Nothing was written.")
        return 1
    binary = os.path.abspath(binary)

    agents_dir = agents_dir or launchd.launch_agents_dir()
    target = launchd.plist_path(agents_dir)
    logs = launchd.log_dir()
    content = launchd.render_plist(
        [binary, "daemon"],
        environment={"PATH": launchd.daemon_path(binary)},
        stdout_path="/dev/null",
        stderr_path=logs / launchd.ERR_LOG_NAME,
    )

    if target.exists():
        existing = target.read_bytes()
        if existing == content:
            print(f"Already installed and up to date: {target}")
            return 0
        if not force:
            print(f"REFUSING to overwrite existing plist: {target}")
            print("It differs from what install-service would write.")
            print("Inspect it, then re-run with --force to replace it.")
            return 1
        backup = target.with_suffix(".plist.bak")
        backup.write_bytes(existing)
        print(f"Existing plist backed up to {backup}")
    else:
        # No plist of ours. If launchd nonetheless has the label loaded,
        # someone else registered it (an app, or a hand-rolled plist). Writing
        # ours would put two supervisors on one daemon, so stop here — even
        # with --force, which only ever replaces a plist this command wrote.
        try:
            taken = launchd.is_loaded(runner)
        except FileNotFoundError:
            print("ERROR: launchctl not found, so this cannot check whether "
                  f"{launchd.LABEL} is already loaded. Nothing was written.")
            return 1
        if taken:
            print(f"REFUSING to install: {launchd.LABEL} is already loaded in "
                  f"launchd, but there is no {target.name} of ours in "
                  f"{agents_dir}. Another supervisor (an app that registered it, "
                  "or a hand-rolled plist) owns the daemon, and two supervisors "
                  "for one daemon is a bug. Nothing was written, and --force "
                  "does not change that: it replaces this command's own plist "
                  "only. Stop or unregister the other supervisor, then re-run.")
            return 1

    agents_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    print(f"Wrote {target}")

    if not now:
        print("Installed, not started. launchd loads the plists in "
              "~/Library/LaunchAgents at your next login. To load it now: "
              f"launchctl bootstrap {launchd.gui_domain()} {target}")
        return 0

    try:
        enabled = launchd.enable(runner)
        if enabled.returncode != 0:
            print(f"ERROR: launchctl enable failed: "
                  f"{(enabled.stderr or enabled.stdout or '').strip()}. "
                  f"The plist is left in place at {target}.")
            return 1
        loaded = launchd.bootstrap(runner, target)
    except FileNotFoundError:
        print(f"ERROR: launchctl not found, so nothing was loaded. The plist "
              f"was written to {target} and is left in place; launchd will "
              f"load it at your next login.")
        return 1
    if loaded.returncode != 0:
        print(f"ERROR: launchctl bootstrap failed: "
              f"{(loaded.stderr or loaded.stdout or '').strip()}. "
              f"The plist is left in place at {target}.")
        return 1
    print(f"Loaded {launchd.LABEL}; launchd is starting the daemon.")
    return 0


def _dir_arg(args: argparse.Namespace, name: str) -> Path | None:
    value = getattr(args, name, None)
    return Path(value).expanduser() if value else None


def run_install_service_command(
    args: argparse.Namespace, *, runner: Runner = subprocess.run,
) -> int:
    """Entry point for ``oara install-service``."""
    return install_service(
        systemd_dir=_dir_arg(args, "systemd_dir"),
        launch_agents_dir=_dir_arg(args, "launch_agents_dir"),
        force=bool(getattr(args, "force", False)),
        now=bool(getattr(args, "now", False)),
        runner=runner,
    )


def add_install_service_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``install-service`` subcommand."""
    p = subparsers.add_parser(
        "install-service",
        help="Install the daemon as a systemd user unit (Linux) or LaunchAgent "
             "(macOS)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing unit (Linux) or plist (macOS), backing it "
             "up first",
    )
    p.add_argument(
        "--now", action="store_true",
        help="Also start the service immediately after enabling (macOS: "
             "launchctl enable + bootstrap; without it the plist is only "
             "written and launchd loads it at next login)",
    )
    p.add_argument(
        "--systemd-dir", default=None,
        help="Linux: target directory (default: ~/.config/systemd/user). "
             "Not applicable on macOS (exit 2)",
    )
    p.add_argument(
        "--launch-agents-dir", default=None,
        help="macOS: target directory (default: ~/Library/LaunchAgents). "
             "Not applicable on Linux (exit 2)",
    )


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(prog="oara install-service")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--now", action="store_true")
    parser.add_argument("--systemd-dir", default=None)
    parser.add_argument("--launch-agents-dir", default=None)
    sys.exit(run_install_service_command(parser.parse_args()))
