"""The macOS half of ``oara install-service``: a LaunchAgent plist.

Pure on purpose — bytes in, bytes out, and an injectable ``runner`` for the
three launchctl calls — so it is testable without launchd and so a second
caller (an app bundle that registers the daemon itself) renders the SAME job
from the same code.

Why the label is final
----------------------
``LABEL`` is the job's identity in launchd. The app registers a plist with the
SAME label, so launchd itself refuses to run two supervisors for one daemon;
that refusal is the safety net under ``install-service``'s own check. Changing
the label would turn "second registration fails" into "second daemon starts".

The restart policy is the systemd unit's, restated
--------------------------------------------------
``Restart=on-failure`` / ``RestartSec=10`` in ``cli/service.py``'s unit means
"restart when the daemon exits non-zero, wait ten seconds between starts".
launchd spells that ``KeepAlive={SuccessfulExit: false}`` and
``ThrottleInterval=10``. Both plist variants get those keys from the one
place below, so they cannot drift from each other; tests pin them against the
unit too.
"""

from __future__ import annotations

import os
import plistlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

LABEL = "com.oaralabs.prometheus.daemon"

# launchd's seconds-between-starts. Equal to RestartSec in the systemd unit.
THROTTLE_INTERVAL = 10

# launchd hands a user agent a minimal PATH (/usr/bin:/bin:/usr/sbin:/sbin),
# which finds neither Homebrew nor a venv. The binary's own directory is put in
# front of this so `oara` and anything it shells out to resolve.
_PATH_TAIL = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

# The one log launchd keeps for the job (stderr). The daemon rotates its own
# log under ~/.prometheus/logs; this one exists so a failure that happens
# before that logging starts (a bad interpreter, a missing module) is visible.
ERR_LOG_NAME = "launchd.err.log"

# subprocess.run's shape, as the rest of the CLI injects it: called with an
# argv list plus capture_output/text, returning something with returncode,
# stdout and stderr. Raises FileNotFoundError when the program is absent.
Runner = Callable[..., Any]


def render_plist(
    program_arguments: Sequence[str],
    *,
    bundle_program: str | None = None,
    associated_bundle_ids: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
    working_directory: str | os.PathLike[str] | None = None,
    stdout_path: str | os.PathLike[str] | None = None,
    stderr_path: str | os.PathLike[str] | None = None,
) -> bytes:
    """The LaunchAgent plist as XML bytes.

    ``program_arguments`` is the argv launchd runs. For the CLI install that is
    the absolute path of the resolved ``oara`` plus ``daemon``. An app that
    ships the launcher inside its bundle also passes ``bundle_program``, the
    launcher's path RELATIVE to the bundle (launchd's ``BundleProgram``).

    ``Label``, ``RunAtLoad``, ``KeepAlive`` and ``ThrottleInterval`` are set
    here and nowhere else, so the two forms always carry identical values.
    """
    if not program_arguments:
        raise ValueError("program_arguments must not be empty")

    job: dict[str, Any] = {
        "Label": LABEL,
        "ProgramArguments": list(program_arguments),
        # Start when the agent is loaded (login, or an explicit bootstrap)...
        "RunAtLoad": True,
        # ...and restart only after a failure, never after a clean exit.
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": THROTTLE_INTERVAL,
    }
    if bundle_program is not None:
        job["BundleProgram"] = bundle_program
    if associated_bundle_ids:
        job["AssociatedBundleIdentifiers"] = list(associated_bundle_ids)
    if environment:
        job["EnvironmentVariables"] = dict(environment)
    if working_directory is not None:
        job["WorkingDirectory"] = os.fspath(working_directory)
    if stdout_path is not None:
        job["StandardOutPath"] = os.fspath(stdout_path)
    if stderr_path is not None:
        job["StandardErrorPath"] = os.fspath(stderr_path)
    # plistlib sorts keys by default, so the bytes are a pure function of the
    # inputs: "is the installed plist identical?" can compare them directly.
    return plistlib.dumps(job, fmt=plistlib.FMT_XML)


def daemon_path(binary: str) -> str:
    """The PATH the job runs with: the binary's directory, then the usual ones."""
    return f"{os.path.dirname(binary)}:{_PATH_TAIL}"


def launch_agents_dir() -> Path:
    """Where per-user agents live (env-overridable for tests)."""
    override = os.environ.get("PROMETHEUS_LAUNCH_AGENTS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library" / "LaunchAgents"


def plist_path(agents_dir: Path | None = None) -> Path:
    """The plist this command owns inside ``agents_dir`` (default: the user's)."""
    return (agents_dir or launch_agents_dir()) / f"{LABEL}.plist"


def log_dir() -> Path:
    """Where launchd writes the job's stderr."""
    return Path.home() / "Library" / "Logs" / "Prometheus"


def gui_domain(uid: int | None = None) -> str:
    """launchd's per-user GUI domain, where LaunchAgents are bootstrapped."""
    return f"gui/{os.getuid() if uid is None else uid}"


def is_loaded(runner: Runner) -> bool:
    """Whether launchd already has the label loaded (read-only probe).

    ``launchctl print`` exits 0 for a loaded service and non-zero (113,
    "Could not find service") otherwise.
    """
    result = runner(
        ["launchctl", "print", f"{gui_domain()}/{LABEL}"],
        capture_output=True, text=True,
    )
    return bool(result.returncode == 0)


def enable(runner: Runner) -> Any:
    """Clear any "disabled" override for the label, so bootstrap will load it."""
    return runner(
        ["launchctl", "enable", f"{gui_domain()}/{LABEL}"],
        capture_output=True, text=True,
    )


def bootstrap(runner: Runner, plist: Path) -> Any:
    """Load ``plist`` into the user's GUI domain; RunAtLoad starts the daemon."""
    return runner(
        ["launchctl", "bootstrap", gui_domain(), str(plist)],
        capture_output=True, text=True,
    )
