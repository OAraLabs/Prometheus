"""Kernel-enforced floors for the bash tool.

TWO FLOORS, TWO MECHANISMS, ONE DOCTRINE
----------------------------------------
* The READ floor (AppArmor, ``aa-exec -p prometheus-bash``) keeps bash out
  of three secret path families. Needs root to load a profile. On macOS the
  same floor is a Seatbelt profile run through ``sandbox-exec`` (no root, no
  profile to load); see "THE READ FLOOR ON macOS" below.
* The WRITE floor (bubblewrap) keeps bash's writes inside the workspace.
  Needs no root, and is the second half of this file.

They are independent, composable, and each is verified BY OUTCOME rather
than by an exit code. Everything below the next heading is the read floor;
the write floor starts at "THE WRITE FLOOR".

WHY THIS EXISTS
---------------
``SecurityGate._check_denied_path`` is nested under ``if file_path:``, and
bash is handed a command string, so the denied-path floor covers the
path-declaring tools and misses bash at BOTH origins. Measured, not
inferred: at ``origin="user"`` and at ``origin="system"``,
``cat /home/<user>/.gnupg/x`` and ``echo x > /home/<user>/.ssh/x`` are both
ALLOW, and three such writes landed on a live daemon.

Every check on the command string is defeated by ordinary shell —
``cd ~/.ssh && cat id_*``, ``$HOME`` indirection, globs, ``sh -c``, heredocs
— so the floor has to be enforced below the tool layer, by the kernel, on
the path as the kernel resolves it.

WHAT THIS IS NOT
----------------
Not a sandbox, and not a gate. Gating (whether to ask the operator) and the
floor (what is never allowed) are different mechanisms; this touches only
the second. Nothing here introduces a prompt: the command runs exactly as
before and ``open()`` returns EACCES on three paths. ``rm``, ``git push``,
package installs, migrations and repo edits are untouched — verified by
outcome under the profile before this was written.

FAIL LOUD
---------
When confinement is ``required`` and the environment cannot provide it,
bash REFUSES to run and says why. It must never silently execute
unconfined: a floor that quietly isn't there is worse than no floor, because
everything downstream is written as though it is. That is the same
false-assurance defect #235 removed from a docstring.

The verification is BY OUTCOME. ``aa-exec`` exiting 0 is not evidence that a
transition happened — it is evidence that ``aa-exec`` ran. So the preflight
launches a process through ``aa-exec`` and reads the label that process
reports for itself from ``/proc/self/attr/current``.
"""

from __future__ import annotations

import glob as _glob
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from prometheus.config import local_pairing
from prometheus.config.paths import config_dir_path, data_dir_path, lcm_db_path, node_dir_path

logger = logging.getLogger(__name__)

PROFILE: Final[str] = "prometheus-bash"

MODE_OFF: Final[str] = "off"
MODE_AUTO: Final[str] = "auto"
MODE_REQUIRED: Final[str] = "required"
VALID_MODES: Final[tuple[str, ...]] = (MODE_OFF, MODE_AUTO, MODE_REQUIRED)

# "auto" — THE SHIPPED DEFAULT — AND WHY IT IS NOT A SILENT FALLBACK.
#
# This module refused a fallback mode for a long time: one that runs
# unconfined when the profile is missing "would be indistinguishable from a
# working floor in every log line". That objection was right, and "auto" is
# shipped only because it is now answered rather than ignored:
#
#   * VERIFIED (preflight, by outcome): auto IS required for the life of the
#     process. The success is cached and the shell keeps going through
#     aa-exec, which refuses to run without the profile — so a profile lost
#     after a verified start fails CLOSED. auto never quietly degrades.
#   * UNVERIFIED on Linux: the shell runs without the read floor, and that is
#     said in four places — an ERROR at boot naming the fix
#     (security/shell_floor.announce), `dark` in /api/status (floor_report),
#     a WARN row in `oara doctor`, and `read_floor: unavailable` in each
#     call's metadata.
#   * NO APPARMOR ON THIS PLATFORM (macOS): its own state, "unsupported". No
#     probe, one line, no fix offered — there is none to offer.
#
# "required" remains for an operator who wants refusal, and an explicit "off"
# stays off: only a config with no key at all changes meaning.

#: The per-call state of the read floor, as each call's metadata reports it.
READ_ACTIVE: Final[str] = "active"
READ_UNAVAILABLE: Final[str] = "unavailable"
READ_UNSUPPORTED: Final[str] = "unsupported"

_preflight_cache: dict[str, tuple[bool, str]] = {}


def normalise_mode(value: object) -> str:
    """Map a config value onto a known mode. Unknown values fail SAFE-LOUD.

    An unrecognised mode returns ``off`` rather than ``required`` so a typo
    cannot brick every bash call, but it is logged at WARNING so it cannot
    pass unnoticed either. ``None`` (a key written with no value) is also
    ``off``; an ABSENT key never reaches here as None — readers pass the
    shipped default, ``"auto"``.
    """
    text = str(value or MODE_OFF).strip().lower()
    if text in VALID_MODES:
        return text
    if text in ("true", "yes", "on", "enforce", "enforced"):
        return MODE_REQUIRED
    if text in ("false", "no", "none", ""):
        return MODE_OFF
    logger.warning(
        "security.bash_confinement=%r is not one of %s — treating as %r. "
        "The bash floor is NOT in force.", value, list(VALID_MODES), MODE_OFF,
    )
    return MODE_OFF


# --------------------------------------------------------------------------- #
# DELEGATION TO A ROOT DAEMON: THE DOCKER SOCKET
# --------------------------------------------------------------------------- #
#
# A floor confines the process TREE it starts. A command that hands the work to
# a separate, unconfined service is not inside that tree — the same gap
# docs/design/master-key-out-of-agent-shell.md records for `systemd-run --user`
# over D-Bus, which the AppArmor profile allows. The Docker control socket is
# that pattern with a sharper edge: a daemon user in the `docker` group can
# write /run/docker.sock, and `docker run -v /:/host …` mounts the host root
# into a container it controls as uid 0 — passwordless ROOT, which reads and
# writes every path the floor denies (the env file, the MCP store, node keys).
#
# This is NOT closable in the profile: the profile is broad by design, denying
# the socket would break legitimate docker use AND would not remove the group
# member's other roads to dockerd, and once the shell is root in a container no
# per-exec user profile binds it. So it is a DEPLOYMENT RULE (keep the
# Prometheus user out of the docker group, or run dockerd rootless), and the
# floor's doctrine — verify by outcome, fail loud — makes it a boot warning.
#
# The probe is by outcome and from THIS process, whose uid/gids a model-written
# shell inherits through aa-exec: os.access() here is exactly what that shell
# can do. Linux only — on a Mac, Docker Desktop's socket reaches a VM, not the
# host, so `-v /:/host` does not mount the Mac's filesystem.
DOCKER_SOCKETS: Final[tuple[str, ...]] = (
    "/run/docker.sock",
    "/var/run/docker.sock",
)


def docker_socket_reachable(
    sockets: Sequence[str] | None = None,
) -> tuple[bool, str]:
    """Whether a model-written shell can reach a Docker control socket.

    Returns ``(True, path)`` for the first socket this process can read and
    write (connecting to a stream socket needs both), else ``(False, reason)``.
    Writing the socket is root on the host by Docker's own model, and it is
    outside every bash floor. Linux only.
    """
    if not apparmor_possible():
        return (False, "not Linux")
    for path in (DOCKER_SOCKETS if sockets is None else sockets):
        try:
            if os.path.exists(path) and os.access(path, os.R_OK | os.W_OK):
                return (True, path)
        except OSError:
            continue
    return (False, "no reachable Docker socket")


def apparmor_possible() -> bool:
    """Can AppArmor exist on this platform at all? Linux only.

    Where it cannot (macOS), "auto" reports ``unsupported`` without probing:
    there is no profile to load, so a probe would only produce an ERROR line
    and a fix that cannot be applied.
    """
    return sys.platform.startswith("linux")


def _probe_label(profile: str) -> tuple[bool, str]:
    """Launch a process through aa-exec and read the label it reports.

    Returns ``(ok, detail)``. ``ok`` is True only when the confined process
    actually ran AND names ``profile`` as its own confinement — the three-way
    never-ran / refused / succeeded distinction, collapsed to a boolean at
    the boundary, with the reason preserved in ``detail``.
    """
    exe = shutil.which("aa-exec")
    if exe is None:
        return False, "aa-exec is not installed (apparmor userspace missing)"
    try:
        proc = subprocess.run(
            [exe, "-p", profile, "--", "cat", "/proc/self/attr/current"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"aa-exec could not be run: {exc}"

    label = (proc.stdout or "").replace("\x00", "").strip()
    stderr = (proc.stderr or "").strip()[:200]
    if not label:
        # aa-exec refuses to exec when the profile is absent, so nothing ran.
        return False, (
            f"the confined process never ran (rc={proc.returncode})"
            + (f": {stderr}" if stderr else "")
        )
    if not label.startswith(profile):
        return False, (
            f"aa-exec ran but the transition did not happen: the process "
            f"reports its label as {label!r}, expected {profile!r}"
        )
    return True, label


#: What an operator runs after the shipped profile changes. Named wherever a stale profile is reported.
RELOAD_COMMAND: Final[str] = f"sudo apparmor_parser -r -W /etc/apparmor.d/{PROFILE}"

# Rename ~/.config onto itself. The kernel asks the LSM about a rename BEFORE
# it notices source and target are the same, so under a profile that denies
# `w` on ~/.config/ this is refused, and anywhere else it is a no-op: the boot
# probe can test the rule on the real home without moving anything (#766).
_CONFIG_RENAME_PROBE: Final[str] = (
    "import os\n"
    "p = os.path.join(os.path.expanduser('~'), '.config')\n"
    "try:\n"
    "    os.rename(p, p)\n"
    "except FileNotFoundError:\n"
    "    print('ABSENT')\n"
    "except PermissionError:\n"
    "    print('REFUSED')\n"
    "else:\n"
    "    print('RENAMED')\n"
)


def _probe_config_rename(profile: str) -> str:
    """Under *profile*, may ``~/.config`` be renamed? ``refused``, ``renamed``, ``absent`` or ``unknown``.

    Run only once the label check has passed, so the process is known to be confined.
    """
    exe = shutil.which("aa-exec")
    if exe is None:
        return "unknown"
    try:
        proc = subprocess.run([exe, "-p", profile, "--", sys.executable, "-c", _CONFIG_RENAME_PROBE],
                              capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    out = (proc.stdout or "").strip()
    return {"REFUSED": "refused", "RENAMED": "renamed", "ABSENT": "absent"}.get(out, "unknown")


# Rename the device registry onto itself, the same trick: refused under a profile that denies `w` on it, a no-op
# anywhere else, so the boot probe tests the rule on the real file without changing it.
_REGISTRY_RENAME_PROBE: Final[str] = (
    "import os, sys\n"
    "p = sys.argv[1]\n"
    "try:\n"
    "    os.rename(p, p)\n"
    "except FileNotFoundError:\n"
    "    print('ABSENT')\n"
    "except PermissionError:\n"
    "    print('REFUSED')\n"
    "else:\n"
    "    print('RENAMED')\n"
)


def _registry_path() -> str:
    return os.path.join(os.path.expanduser(str(data_dir_path())), REGISTRY_NAME)


def _registry_outside_profile() -> bool:
    """The AppArmor profile names ``@{HOME}/.prometheus/data/devices.db*``: a data directory moved elsewhere is not
    covered by it, whatever the probe says."""
    default = os.path.realpath(os.path.join(os.path.expanduser("~"), ".prometheus", "data"))
    return os.path.realpath(os.path.dirname(_registry_path())) != default


def _probe_registry_rename(profile: str) -> str:
    """Under *profile*, may the device registry be written? ``refused``, ``renamed``, ``absent`` or ``unknown``."""
    exe = shutil.which("aa-exec")
    if exe is None:
        return "unknown"
    try:
        proc = subprocess.run([exe, "-p", profile, "--", sys.executable, "-c", _REGISTRY_RENAME_PROBE,
                               _registry_path()], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    out = (proc.stdout or "").strip()
    return {"REFUSED": "refused", "RENAMED": "renamed", "ABSENT": "absent"}.get(out, "unknown")


def _registry_detail(profile: str) -> tuple[str, bool]:
    """The device-registry half of the preflight detail, and whether it makes the profile STALE."""
    if _registry_outside_profile():
        return (f"; the device registry and the scheduled-jobs files are outside the profile's rule "
                f"(PROMETHEUS_DATA_DIR or PROMETHEUS_CONFIG_DIR moved them to {os.path.dirname(_registry_path())}), "
                f"so a shell can write them"), True
    registry = _probe_registry_rename(profile)
    if registry == "refused":
        return "; writing the device registry refused", False
    if registry == "renamed":
        return ("; STALE PROFILE: the device registry (devices.db) is writable, so a shell can add a device, an "
                f"owner or a desktop mark. Reload the shipped profile: {RELOAD_COMMAND}"), True
    if registry == "absent":
        return "; the device registry does not exist yet, so its rule was not measured", False
    return "; the device-registry check could not run", False


#: The loaded configs the shipped AppArmor profile names: the default config dir's, and the deploy clone's
#: (scripts/deploy.sh's default ``$HOME/prometheus-deploy``).
_PROFILE_CONFIGS: Final[tuple[str, ...]] = (
    os.path.join("~", ".prometheus", "prometheus.yaml"),
    os.path.join("~", "prometheus-deploy", "config", "prometheus.yaml"),
)


def _trusted_outside_profile() -> list[str]:
    """Trusted state the static profile cannot name: a moved config dir, or a loaded config elsewhere."""
    named = {os.path.realpath(os.path.expanduser(p)) for p in _PROFILE_CONFIGS}
    outside = [p for p in _loaded_configs if p not in named]
    default_dir = os.path.realpath(os.path.join(os.path.expanduser("~"), ".prometheus"))
    wired = os.path.realpath(os.path.expanduser(str(config_dir_path())))
    if wired != default_dir:
        outside.append(wired)
    return outside


def _probe_trusted_rename(profile: str) -> str:
    """Under *profile*, may trusted state be written? Renames the first that exists onto itself (skills/, then each
    trusted file): ``refused``, ``renamed``, ``absent`` or ``unknown``."""
    exe = shutil.which("aa-exec")
    if exe is None:
        return "unknown"
    base = os.path.expanduser(str(config_dir_path()))
    candidates = [os.path.join(base, n) for n in (*TRUSTED_DIRS, *TRUSTED_FILES)]
    target = next((c for c in candidates if os.path.exists(c)), None)
    if target is None:
        return "absent"
    try:
        proc = subprocess.run([exe, "-p", profile, "--", sys.executable, "-c", _REGISTRY_RENAME_PROBE, target],
                              capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    out = (proc.stdout or "").strip()
    return {"REFUSED": "refused", "RENAMED": "renamed", "ABSENT": "absent"}.get(out, "unknown")


def _trusted_detail(profile: str) -> tuple[str, bool]:
    """The trusted-state half of the preflight detail, and whether it makes the profile incomplete."""
    outside = _trusted_outside_profile()
    note = ""
    if outside:
        note = ("; the loaded config or config directory is outside the profile's rule (" + ", ".join(outside)
                + "), so a shell can write it unless the write floor covers it")
    trusted = _probe_trusted_rename(profile)
    if trusted == "refused":
        return f"; writing trusted state refused{note}", bool(outside)
    if trusted == "renamed":
        return ("; STALE PROFILE: trusted state (the config, identity files, memory or skills) is writable, so a "
                f"shell can rewrite what Prometheus trusts. Reload the shipped profile: {RELOAD_COMMAND}{note}"), True
    if trusted == "absent":
        return f"; no trusted state exists yet, so its rule was not measured{note}", bool(outside)
    return f"; the trusted-state check could not run{note}", bool(outside)


def _probe_cron_rename(profile: str) -> str:
    """Under *profile*, may the scheduled-jobs files be written? Renames the first that exists onto itself:
    ``refused``, ``renamed``, ``absent`` or ``unknown``. A profile with the registry rule but not this one (loaded
    before the cron rule shipped) would otherwise pass the registry probe and leave the schedule writable."""
    exe = shutil.which("aa-exec")
    if exe is None:
        return "unknown"
    base = os.path.expanduser(str(data_dir_path()))
    target = next((p for p in (os.path.join(base, n) for n in DATA_TRUSTED_FILES) if os.path.exists(p)), None)
    if target is None:
        return "absent"
    try:
        proc = subprocess.run([exe, "-p", profile, "--", sys.executable, "-c", _REGISTRY_RENAME_PROBE, target],
                              capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    out = (proc.stdout or "").strip()
    return {"REFUSED": "refused", "RENAMED": "renamed", "ABSENT": "absent"}.get(out, "unknown")


def _cron_detail(profile: str) -> tuple[str, bool]:
    """The scheduled-jobs half of the preflight detail, and whether it makes the profile STALE. A moved data
    directory is reported by :func:`_registry_detail`, which shares it."""
    cron = _probe_cron_rename(profile)
    if cron == "refused":
        return "; writing the scheduled-jobs files refused", False
    if cron == "renamed":
        return ("; STALE PROFILE: the scheduled-jobs files (cron_jobs.json and its logs) are writable, so a shell can "
                f"plant a job that runs as the owner. Reload the shipped profile: {RELOAD_COMMAND}"), True
    if cron == "absent":
        return "; no scheduled-jobs file exists yet, so its rule was not measured", False
    return "; the scheduled-jobs check could not run", False


# Open each path for reading: REFUSED when every one that exists is refused, READABLE when any opens. The probe
# prints only the verdict, never a byte of what it read.
_READ_PROBE: Final[str] = (
    "import sys\n"
    "seen = False\n"
    "for p in sys.argv[1:]:\n"
    "    try:\n"
    "        open(p, 'rb').close()\n"
    "    except FileNotFoundError:\n"
    "        continue\n"
    "    except PermissionError:\n"
    "        seen = True\n"
    "        continue\n"
    "    print('READABLE')\n"
    "    sys.exit()\n"
    "print('REFUSED' if seen else 'ABSENT')\n"
)

# Rename each path onto itself (the registry probe's trick, on several): RENAMED when any one is writable.
_RENAME_ALL_PROBE: Final[str] = (
    "import os, sys\n"
    "seen = False\n"
    "for p in sys.argv[1:]:\n"
    "    try:\n"
    "        os.rename(p, p)\n"
    "    except FileNotFoundError:\n"
    "        continue\n"
    "    except PermissionError:\n"
    "        seen = True\n"
    "        continue\n"
    "    print('RENAMED')\n"
    "    sys.exit()\n"
    "print('REFUSED' if seen else 'ABSENT')\n"
)


def _run_probe(profile: str, script: str, paths: Sequence[str]) -> str:
    exe = shutil.which("aa-exec")
    if exe is None:
        return "unknown"
    try:
        proc = subprocess.run([exe, "-p", profile, "--", sys.executable, "-c", script, *paths],
                              capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    out = (proc.stdout or "").strip()
    return {"REFUSED": "refused", "READABLE": "readable", "RENAMED": "renamed", "ABSENT": "absent"}.get(out, "unknown")


def _node_dir() -> str:
    return os.path.expanduser(str(node_dir_path()))


def _probe_node_read(profile: str) -> str:
    """Under *profile*, may the node keys or the MCP store be read? ``refused``, ``readable``, ``absent`` or
    ``unknown``. A profile loaded before these rules shipped passes every other probe and leaves them readable."""
    node = _node_dir()
    paths = [os.path.join(node, n) for n in ("node.key", "instance.key")]
    paths.append(os.path.join(os.path.expanduser(str(data_dir_path())), MCP_STORE_NAME))
    paths.append(os.path.join(os.path.expanduser(str(config_dir_path())), XAI_OAUTH_NAME))
    return _run_probe(profile, _READ_PROBE, paths)


def _probe_launch_write(profile: str) -> str:
    """Under *profile*, may launch configuration be written? Renames each that exists onto itself: ``refused``,
    ``renamed``, ``absent`` or ``unknown``."""
    data = os.path.expanduser(str(data_dir_path()))
    cfg = os.path.expanduser(str(config_dir_path()))
    paths = [os.path.join(data, n) for n in (MCP_STORE_NAME, JOB_SHIM_DIR_NAME)]
    paths.append(os.path.expanduser(str(lcm_db_path())))
    paths += [os.path.join(cfg, n) for n in (*CONFIG_LAUNCH_FILES, *CONFIG_LAUNCH_DIRS, "profiles")]
    paths.append(_node_dir())
    return _run_probe(profile, _RENAME_ALL_PROBE, paths)


def _node_outside_profile() -> bool:
    """The AppArmor profile names ``@{HOME}/.prometheus/node``: a node directory PROMETHEUS_NODE_DIR or
    PROMETHEUS_CONFIG_DIR moved is not covered by it, whatever the probe says."""
    default = os.path.realpath(os.path.join(os.path.expanduser("~"), ".prometheus", NODE_DIR_NAME))
    return os.path.realpath(_node_dir()) != default


def _keys_and_launch_detail(profile: str) -> tuple[str, bool]:
    """The node-key and launch-configuration half of the preflight detail, and whether it makes the profile STALE.
    A moved data or config directory is reported by the registry and trusted-state halves."""
    parts: list[str] = []
    stale = False
    if _node_outside_profile():
        parts.append(f"; the node keys are outside the profile's rule (PROMETHEUS_NODE_DIR or PROMETHEUS_CONFIG_DIR "
                     f"moved them to {_node_dir()}), so a shell can read them")
        stale = True
    keys = _probe_node_read(profile)
    if keys == "refused":
        parts.append("; reading the node keys refused, and the MCP store and the xAI tokens")
    elif keys == "readable":
        parts.append("; STALE PROFILE: the node keys, the MCP store or the xAI tokens are readable, so a shell can "
                     f"copy this machine's identity or a credential. Reload the shipped profile: {RELOAD_COMMAND}")
        stale = True
    elif keys == "absent":
        parts.append("; no node key exists yet, so its rule was not measured")
    else:
        parts.append("; the node-key check could not run")
    launch = _probe_launch_write(profile)
    if launch == "refused":
        parts.append("; writing launch configuration refused")
    elif launch == "renamed":
        parts.append("; STALE PROFILE: launch configuration (the MCP store, the cron jobs' PATH shims, lcm.db, "
                     "config_pins.yaml, checkpoints, profiles or the node keys) is writable, so a shell can change "
                     f"what the daemon starts at boot. Reload the shipped profile: {RELOAD_COMMAND}")
        stale = True
    elif launch == "absent":
        parts.append("; no launch configuration exists yet, so its rule was not measured")
    else:
        parts.append("; the launch-configuration check could not run")
    return "".join(parts), stale


def _probe_startup_write(profile: str) -> str:
    """Under *profile*, may what starts Prometheus be written? Renames each service file and install folder that
    exists onto itself: ``refused``, ``renamed``, ``absent`` or ``unknown``. Only the places the profile names are
    asked; :func:`_install_outside_profile` reports the rest."""
    home = os.path.realpath(os.path.expanduser("~"))
    covered = [p for p in (*startup_files(home), *startup_dirs(home)) if _under_profile_startup(p, home)]
    return _run_probe(profile, _RENAME_ALL_PROBE, covered)


def _under_profile_startup(path: str, home: str) -> bool:
    """Whether the shipped AppArmor profile names *path* (the default unit, its drop-ins, the default installs)."""
    units = os.path.join(home, ".config", "systemd", "user")
    named = [os.path.join(units, SYSTEMD_UNIT), os.path.join(units, f"{SYSTEMD_UNIT}.d"),
             *(os.path.join(home, d) for d in _STARTUP_HOME_DIRS)]
    return any(path == n or path.startswith(n + "/") for n in named)


def _install_outside_profile() -> list[str]:
    """This process's own install folders that the shipped profile does not name and this user can write, so a shell
    can write them whatever the profile says. A root-owned install (a system package) is not listed."""
    home = os.path.realpath(os.path.expanduser("~"))
    return [d for d in _install_dirs()
            if not _under_profile_startup(d, home) and os.path.isdir(d) and os.access(d, os.W_OK)]


def _startup_detail(profile: str) -> tuple[str, bool]:
    """The what-starts-Prometheus half of the preflight detail, and whether it makes the profile STALE. An install
    the profile does not name is reported, not called stale: the profile cannot name every venv."""
    parts: list[str] = []
    stale = False
    startup = _probe_startup_write(profile)
    if startup == "refused":
        parts.append("; writing what starts Prometheus refused")
    elif startup == "renamed":
        parts.append("; STALE PROFILE: what starts Prometheus (its systemd unit or drop-ins, the deploy's venvs, "
                     "clone or scripts) is writable, so a shell can change what the daemon runs at boot. Reload the "
                     f"shipped profile: {RELOAD_COMMAND}")
        stale = True
    elif startup == "absent":
        parts.append("; no service file or default install exists, so the startup rule was not measured")
    else:
        parts.append("; the startup-file check could not run")
    outside = _install_outside_profile()
    if outside:
        parts.append(f"; this install ({', '.join(outside)}) is outside the profile's rules, so a shell can write "
                     "it (file tools are still refused)")
    return "".join(parts), stale


def _with_rename_check(profile: str, label: str) -> tuple[str, bool]:
    """The preflight detail for a confined *label*, and whether the profile is STALE.

    A profile loaded before the ``~/.config`` rule (#766) still confines, and still denies everything else, so a
    stale one is not treated as no profile: ``auto`` would then run shells with no floor at all and ``required``
    would refuse every one. It is reported, at ERROR, with the reload that fixes it.
    """
    rename = _probe_config_rename(profile)
    registry, registry_stale = _registry_detail(profile)
    trusted, trusted_stale = _trusted_detail(profile)
    cron, cron_stale = _cron_detail(profile)
    keys, keys_stale = _keys_and_launch_detail(profile)
    startup, startup_stale = _startup_detail(profile)
    registry = registry + trusted + cron + keys + startup
    registry_stale = registry_stale or trusted_stale or cron_stale or keys_stale or startup_stale
    if rename == "refused":
        return f"{label}; renaming ~/.config refused{registry}", registry_stale
    if rename == "renamed":
        return (f"{label}; STALE PROFILE: ~/.config can be renamed out from under the env rule (#766), so a shell "
                f"can still reach ~/.config/*/*env. Reload the shipped profile: {RELOAD_COMMAND}{registry}"), True
    if rename == "absent":
        return f"{label}; ~/.config does not exist, so the rename rule was not measured{registry}", registry_stale
    return f"{label}; the ~/.config rename check could not run{registry}", registry_stale


def preflight(profile: str = PROFILE, *, force: bool = False) -> tuple[bool, str]:
    """Cached per-process check that confinement genuinely works.

    Caching is safe in the only direction that matters. If the profile is
    unloaded after a cached success, the next call still runs through
    ``aa-exec``, which fails closed — verified by outcome::

        aa-exec -p no-such-profile -- bash -c 'echo I_RAN'
        rc=1, "aa-exec: ERROR: profile 'no-such-profile' does not exist"

    and ``I_RAN`` is never printed. So a stale cache can degrade to a refused
    command, never to an unconfined one.
    """
    if force or profile not in _preflight_cache:
        ok, detail = _probe_label(profile)
        stale = False
        if ok:
            detail, stale = _with_rename_check(profile, detail)
        _preflight_cache[profile] = (ok, detail)
        # Record the effective state. A runtime that auto-detects but leaves
        # no record is how an operator ends up believing in a floor that is
        # not there (CROSS-CUTTING §12).
        if ok and stale:
            logger.error("bash confinement ACTIVE but INCOMPLETE: profile %s (%s)", profile, detail)
        elif ok:
            logger.info("bash confinement ACTIVE: profile %s (%s)", profile, detail)
        else:
            logger.error("bash confinement UNAVAILABLE: %s", detail)
    return _preflight_cache[profile]


def reset_cache() -> None:
    """Drop the memoised preflight. For tests and for a profile reload."""
    _preflight_cache.clear()


def wrap_argv(argv: list[str], profile: str = PROFILE) -> list[str]:
    """Prefix an argv with the aa-exec transition."""
    exe = shutil.which("aa-exec") or "aa-exec"
    return [exe, "-p", profile, "--", *argv]


def refusal_message(profile: str, detail: str) -> str:
    """What the operator sees instead of an unconfined shell."""
    return (
        "bash REFUSED: kernel confinement is required but unavailable.\n"
        f"Reason: {detail}\n"
        f"Profile: {profile}\n"
        "\n"
        "The command was NOT run. Confinement is what keeps bash out of "
        "~/.ssh, ~/.gnupg and ~/.config/*/*env — the permission gate cannot "
        "reach those paths for bash, so running unconfined would remove the "
        "floor entirely rather than degrade it.\n"
        "\n"
        "Fix: load the profile —\n"
        f"  sudo apparmor_parser -r -W /etc/apparmor.d/{profile}\n"
        "or set security.bash_confinement: off to run without the floor, "
        "knowingly."
    )


# --------------------------------------------------------------------------- #
# THE READ FLOOR ON macOS: SEATBELT
# --------------------------------------------------------------------------- #
#
# There is no AppArmor on a Mac, and ``auto`` used to report "unsupported" there,
# so every model-written shell could read ~/.config/prometheus/env (the master
# key, the provider keys, the bot tokens), ~/.ssh and ~/.gnupg. The same floor,
# with the same three families, modes and doctrine, is a Seatbelt profile run
# through ``sandbox-exec``. It allows everything, then denies reads and writes of
# those families. It is unprivileged, every child inherits it, and it is verified
# BY OUTCOME at boot against a scratch home. ``sandbox-exec`` is deprecated by
# Apple and still ships: if it goes, ``auto`` says so loudly and ``required``
# refuses, as with a missing AppArmor profile.
#
# On a Mac it denies one thing more: the same-Mac pairing directory, whose
# pair.secret is traded for an OWNER device token over loopback (read it, or
# plant one, and the shell can pair itself as owner). It also refuses renaming
# the directories above what it denies, because a path-matched rule is escaped
# by moving a parent (``mv ~/.config ~/cfg`` was allowed, measured).
# docs/design/master-key-out-of-agent-shell.md.

SANDBOX_EXEC: Final[str] = "/usr/bin/sandbox-exec"
MECHANISM_APPARMOR: Final[str] = "apparmor"
MECHANISM_SEATBELT: Final[str] = "seatbelt"
_SEATBELT_CACHE_KEY: Final[str] = "seatbelt:sandbox-exec"

def seatbelt_possible() -> bool:
    """A Mac with ``sandbox-exec``. Nothing to load: the profile travels on each command line."""
    return sys.platform == "darwin" and os.path.exists(SANDBOX_EXEC)


def read_floor_mechanism() -> str | None:
    """``apparmor`` on Linux, ``seatbelt`` on a Mac with sandbox-exec, None where neither can exist."""
    if apparmor_possible():
        return MECHANISM_APPARMOR
    if seatbelt_possible():
        return MECHANISM_SEATBELT
    return None


def _sb_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def normalise_allow_ssh(value: object) -> bool:
    """``security.bash_allow_ssh``: ``true`` or ``false``. Anything else is false, and said once."""
    if value is None or isinstance(value, bool):
        return bool(value)
    logger.warning("security.bash_allow_ssh must be true or false, got %r: ~/.ssh stays denied", value)
    return False


#: Where the Mac app keeps the same-Mac pairing secret, under the home (config/local_pairing.py).
MAC_PAIRING_DIR: Final[str] = "Library/Application Support/Prometheus/pairing"


def _pairing_dirs(home: str) -> list[str]:
    """The pairing directories to deny, real paths: the Mac app's under *home*, and the one
    ``PROMETHEUS_LOCAL_PAIRING_DIR`` names when this process was given one."""
    dirs = [os.path.realpath(os.path.join(home, MAC_PAIRING_DIR))]
    override = os.environ.get(local_pairing.ENV_DIR)
    if override:
        wired = os.path.realpath(os.path.expanduser(override))
        if wired not in dirs:
            dirs.append(wired)
    return dirs


#: The device registry's file name in the data directory (config/paths.py), and the Seatbelt match for it and its
#: -wal / -shm / -journal files. A constant: paths never go inside a regex literal (see :func:`seatbelt_profile`).
REGISTRY_NAME: Final[str] = "devices.db"
_REGISTRY_REGEX: Final[str] = '(regex #"/devices\\.db[^/]*$")'

#: Other trusted files in the data directory: the scheduled-jobs schedule (re-read every tick, so a planted job
#: runs as the owner with no restart), its logs (the resend queue is sent to the owner verbatim) and the
#: scheduler's PID file. The daemon writes them; a shell must not. Each is floored BY PREFIX, like the registry:
#: the daemon also writes ``cron_jobs.json.tmp`` (then ``os.replace``) and ``cron_jobs.json.corrupt``
#: (gateway/cron_service.py), and ``open(tmp, "w")`` follows a symlink, so a planted side file would make the
#: daemon write job JSON through it.
DATA_TRUSTED_FILES: Final[tuple[str, ...]] = (
    "cron_jobs.json", "cron_history.jsonl", "cron_undelivered.jsonl", "cron_scheduler.pid",
    # The pause-everything state (engine/pause.py), read at boot: a shell that deleted or rewrote it would undo an
    # emergency stop at the next restart. The daemon writes ``pause.json.<hex>.tmp`` and renames it over.
    "pause.json",
)
#: The Seatbelt match for :data:`DATA_TRUSTED_FILES` and anything they are a prefix of (``.tmp``, ``.corrupt``).
_CRON_REGEX: Final[str] = (
    '(regex #"/(cron_(jobs\\.json|history\\.jsonl|undelivered\\.jsonl|scheduler\\.pid)|pause\\.json)[^/]*$")'
)


#: Prometheus's TRUSTED STATE in the config directory: what the daemon reads as its own. The config file (the
#: permission mode, the floors' modes, allowed commands and persisted grants), the identity files that go into every
#: prompt, MEMORY.md and USER.md (every prompt, refreshed per turn), and the skills. The daemon's own writers (the
#: memory tool, the skill writers and their scanners) are unconfined; a model's shell must not write any of it.
#: The gate asks the owner before a file tool does (permissions/checker.py reads these same names).
TRUSTED_FILES: Final[tuple[str, ...]] = (
    "prometheus.yaml", "SOUL.md", "AGENTS.md", "ANATOMY.md", "MEMORY.md", "USER.md",
)
TRUSTED_DIRS: Final[tuple[str, ...]] = ("skills", "profiles")

#: The machine's private keys (config/paths.py ``node_dir_path``: ``<config dir>/node``, or PROMETHEUS_NODE_DIR):
#: ``node.key``, which signs the identity challenge a paired client checks, and ``instance.key``, the TLS key a paired
#: client pins. A copy of both answers as this Prometheus. Floored for READS and writes, like the pairing secret; the
#: daemon reads them itself, unconfined, and nothing a shell or a file tool does needs them.
NODE_DIR_NAME: Final[str] = "node"

#: Launch configuration in the data directory: what the daemon RUNS or binds a conversation to. A shell must not
#: write any of it; the daemon (unconfined) still does.
#: - ``mcp_servers.json`` (mcp/store.py): MCP servers the daemon starts at boot, outside the floor, and their own
#:   credentials, so it is floored for reads too. By prefix, for the store's ``.tmp`` side file.
#: - ``job-python/`` (utils/job_python.py): first on PATH for the owner's cron jobs, which run outside the floor.
#: - ``lcm.db`` (config/paths.py): each conversation's workspace binding (the write floor's writable set) and profile
#:   binding (its tool limits), and the summaries fed back into prompts. By prefix: -wal, -shm, -journal.
MCP_STORE_NAME: Final[str] = "mcp_servers.json"
JOB_SHIM_DIR_NAME: Final[str] = "job-python"
LCM_NAME: Final[str] = lcm_db_path().name  # the one resolver names it (tests/test_lcm_db_path_resolution.py)
_MCP_STORE_REGEX: Final[str] = '(regex #"/mcp_servers\\.json[^/]*$")'
#: The xAI OAuth tokens in the config directory (providers/xai_oauth.py), and their .tmp side file: a credential,
#: floored for reads and writes like the MCP store.
XAI_OAUTH_NAME: Final[str] = "xai_oauth.json"
_XAI_OAUTH_REGEX: Final[str] = '(regex #"/xai_oauth\\.json[^/]*$")'
_LCM_REGEX: Final[str] = '(regex #"/lcm\\.db[^/]*$")'

#: Launch configuration in the config directory: ``config_pins.yaml`` (daemon.py ``apply_config_pins``, re-applied over
#: the config at every boot, any key) and the file checkpoints (checkpoints/store.py: an owner's restore writes what a
#: checkpoint names). The custom agent profiles are TRUSTED_DIRS above (a file tool's write asks the owner).
CONFIG_LAUNCH_FILES: Final[tuple[str, ...]] = ("config_pins.yaml",)
CONFIG_LAUNCH_DIRS: Final[tuple[str, ...]] = ("checkpoints",)

#: WHAT STARTS PROMETHEUS: refused for writes, never for reads. The service files that say what to run at boot (the
#: LaunchAgent plist, the systemd unit and its drop-in folder, where ``scripts/deploy.sh`` writes ``ExecStart=``), and
#: the install it runs: ``sys.prefix``, the venv root a deploy switches with its ``current`` link, the folder the
#: ``prometheus`` package is imported from (a deploy clone's ``src/`` runs from there), and that clone's ``scripts/``
#: (``deploy_guard.sh`` is the unit's ``ExecStartPre``). The owner's installers write them, unconfined; no shell or file
#: tool a model drives needs to. The names are the installers' own (cli/launchd.py, cli/service.py; test-pinned).
LAUNCH_AGENT_NAME: Final[str] = "com.oaralabs.prometheus.daemon.plist"
SYSTEMD_UNIT: Final[str] = "prometheus.service"
#: The default places, relative to the home, that the static AppArmor profile names too: the deploy's venv root and
#: clone (scripts/deploy.sh), and a pipx or ``uv tool`` install of the distribution.
_STARTUP_HOME_DIRS: Final[tuple[str, ...]] = (
    "prometheus-venvs", "prometheus-deploy/scripts", "prometheus-deploy/src",
    ".local/share/pipx/venvs/oara-prometheus", ".local/share/uv/tools/oara-prometheus",
)


def _package_dir() -> str:
    """The folder the ``prometheus`` package is imported from, real path."""
    import prometheus

    return os.path.realpath(os.path.dirname(prometheus.__file__))


def _install_dirs() -> list[str]:
    """This process's own install, real paths: ``sys.prefix`` (the whole Homebrew keg when it is one's ``libexec``,
    the whole ``.app`` when it is inside one), the folder ``PROMETHEUS_VENV``'s link lives in (the deploy's venv
    root), the package folder, and the ``scripts/`` beside a source tree's ``src/``."""
    prefix = os.path.realpath(sys.prefix)
    dirs = [prefix]
    if os.path.basename(prefix) == "libexec":
        dirs.append(os.path.dirname(prefix))
    bundle = next((p for p in [prefix, *_parents_below(prefix, "/")] if p.endswith(".app")), None)
    if bundle:
        dirs.append(bundle)
    venv = os.environ.get("PROMETHEUS_VENV")
    if venv:
        dirs.append(os.path.realpath(os.path.dirname(os.path.abspath(os.path.expanduser(venv)))))
    pkg = _package_dir()
    dirs.append(pkg)
    src = os.path.dirname(pkg)
    if os.path.basename(src) == "src" and os.path.isdir(os.path.join(os.path.dirname(src), "scripts")):
        dirs.append(os.path.join(os.path.dirname(src), "scripts"))
    return list(dict.fromkeys(dirs))


def _service_dirs(home: str) -> tuple[list[str], list[str]]:
    """The LaunchAgents and systemd user directories: the defaults under *home*, and any the installers' overrides
    (``PROMETHEUS_LAUNCH_AGENTS_DIR``, ``PROMETHEUS_SYSTEMD_USER_DIR``) name, real paths."""
    agents = [os.path.join(home, "Library", "LaunchAgents")]
    units = [os.path.join(home, ".config", "systemd", "user")]
    for env, out in (("PROMETHEUS_LAUNCH_AGENTS_DIR", agents), ("PROMETHEUS_SYSTEMD_USER_DIR", units)):
        wired = os.environ.get(env)
        if wired:
            real = os.path.realpath(os.path.expanduser(wired))
            if real not in out:
                out.append(real)
    return agents, units


def startup_files(home: str | os.PathLike[str] | None = None) -> list[str]:
    """The service files that start Prometheus (see LAUNCH_AGENT_NAME above), real paths."""
    real_home = os.path.realpath(os.fspath(home) if home is not None else str(Path.home()))
    agents, units = _service_dirs(real_home)
    return [*(os.path.join(d, LAUNCH_AGENT_NAME) for d in agents), *(os.path.join(d, SYSTEMD_UNIT) for d in units)]


def startup_dirs(home: str | os.PathLike[str] | None = None) -> list[str]:
    """The folders whose contents start Prometheus: the unit's drop-in folder, the default install places, and this
    process's own install. Real paths."""
    real_home = os.path.realpath(os.fspath(home) if home is not None else str(Path.home()))
    _agents, units = _service_dirs(real_home)
    dirs = [os.path.join(d, f"{SYSTEMD_UNIT}.d") for d in units]
    dirs += [os.path.join(real_home, d) for d in _STARTUP_HOME_DIRS]
    dirs += [d for d in _install_dirs() if d not in dirs]
    return dirs

#: The config file(s) this process actually loaded, wherever they are (a repo-local ``config/prometheus.yaml`` in a
#: checkout or a deploy clone). Registered by the gate when it is built (``SecurityGate.__init__``).
_loaded_configs: list[str] = []


def protect_loaded_config(path: str | os.PathLike[str]) -> None:
    """Put the config file this process loaded under the shell floor too, wherever it lives."""
    real = os.path.realpath(os.path.expanduser(os.fspath(path)))
    if real not in _loaded_configs:
        _loaded_configs.append(real)


def reset_loaded_config() -> None:
    _loaded_configs.clear()


def loaded_configs() -> tuple[str, ...]:
    return tuple(_loaded_configs)


def _config_dirs(home: str) -> list[str]:
    """The config directories whose trusted state to protect, real paths: the default under *home*, and the one
    this process uses when ``PROMETHEUS_CONFIG_DIR`` moved it."""
    dirs = [os.path.realpath(os.path.join(home, ".prometheus"))]
    wired = os.path.realpath(os.path.expanduser(str(config_dir_path())))
    if wired not in dirs:
        dirs.append(wired)
    return dirs


def _data_dirs(home: str) -> list[str]:
    """The data directories whose device registry to protect, real paths: the default under *home*, and the one
    this process uses when ``PROMETHEUS_DATA_DIR`` or ``PROMETHEUS_CONFIG_DIR`` moved it."""
    dirs = [os.path.realpath(os.path.join(home, ".prometheus", "data"))]
    wired = os.path.realpath(os.path.expanduser(str(data_dir_path())))
    if wired not in dirs:
        dirs.append(wired)
    return dirs


def _node_dirs(home: str) -> list[str]:
    """The node directories whose keys to protect, real paths: the default under *home*, the one under a config
    directory ``PROMETHEUS_CONFIG_DIR`` moved, and the one ``PROMETHEUS_NODE_DIR`` names."""
    dirs = [os.path.realpath(os.path.join(home, ".prometheus", NODE_DIR_NAME))]
    for wired in (os.path.join(os.path.expanduser(str(config_dir_path())), NODE_DIR_NAME),
                  os.path.expanduser(str(node_dir_path()))):
        real = os.path.realpath(wired)
        if real not in dirs:
            dirs.append(real)
    return dirs


def _parents_below(path: str, home: str) -> list[str]:
    """Every directory above *path*, nearest first, stopping BEFORE *home* when *path* is under it (a user cannot
    rename their own home: its parent is root's) and before ``/`` otherwise."""
    stop = home if path.startswith(home.rstrip("/") + "/") else "/"
    out: list[str] = []
    parent = os.path.dirname(path)
    while parent not in (stop, "/", ""):
        out.append(parent)
        parent = os.path.dirname(parent)
    return out


def seatbelt_profile(home: str | os.PathLike[str] | None = None, *, allow_ssh: bool = False) -> str:
    """The Seatbelt profile: everything allowed except reading or writing the three families
    ``deploy/apparmor/prometheus-bash`` denies, ``~/.ssh``, ``~/.gnupg`` and ``~/.config/*/*env``, and the same-Mac
    pairing directory (``~/Library/Application Support/Prometheus/pairing``, plus ``PROMETHEUS_LOCAL_PAIRING_DIR``
    when set), whose ``pair.secret`` mints an owner device token over loopback. Writes count as much as reads there:
    a secret the shell PLANTED would be traded just the same.

    *home* defaults to ``Path.home()`` (``$HOME``, as the daemon and the app's launcher resolve it), resolved,
    because Seatbelt matches the real path a file is opened at.

    A rule matched on a path is escaped by renaming a directory above it: measured, ``mv ~/.config ~/cfg`` was
    allowed (``~/.config`` itself is not ``~/.config/*/*env``) and ``cat ~/cfg/prometheus/env`` printed the env
    file. So writes to the directories ABOVE each denied root are refused as well, up to but not including the
    home: ``~/.config``, ``~/Library``, ``~/Library/Application Support``, ``.../Prometheus``. ``literal`` matches
    that directory alone: creating, writing or deleting things INSIDE it is unaffected (measured), but renaming,
    replacing or re-moding the directory itself is not allowed. ``~/.ssh`` and ``~/.gnupg`` have no such parent,
    and renaming them is renaming a denied ``subpath`` itself.

    The paths only ever appear inside ordinary string literals, where ``\\"`` escapes a quote. They are never put
    inside the regex literal (``#"..."``): there a quote ends the literal whatever precedes it, so a home
    containing one would break the profile (measured: ``unbound variable``, rc 65). ``~/.config/*/*env`` is
    therefore "under ~/.config" AND "ends in /.config/<one segment>/<name>env", the AppArmor ``*`` that does not
    cross ``/``.

    *allow_ssh* (``security.bash_allow_ssh``) drops ONLY the ``~/.ssh`` rule, for ``git`` over SSH:
    ``~/.gnupg``, ``~/.config/*/*env`` (the master key, the provider keys, the bot tokens) and the pairing
    directory stay denied.
    """
    real_home = os.path.realpath(os.fspath(home) if home is not None else str(Path.home()))
    base = _sb_string(real_home)
    pairing = _pairing_dirs(real_home)
    data_dirs = _data_dirs(real_home)
    parents: list[str] = [os.path.join(real_home, ".config")]
    for directory in pairing:
        parents += [p for p in _parents_below(directory, real_home) if p not in parents]
    for directory in data_dirs:
        # The data directory itself and every one above it: renaming any of them would move the registry out from
        # under its rule. Files beside the registry stay writable (``literal`` is the directory alone).
        parents += [p for p in _parents_below(os.path.join(directory, REGISTRY_NAME), real_home) if p not in parents]
    config_dirs = _config_dirs(real_home)
    trusted_files = [os.path.join(d, name) for d in config_dirs for name in (*TRUSTED_FILES, *CONFIG_LAUNCH_FILES)]
    trusted_files += [p for p in _loaded_configs if p not in trusted_files]
    trusted_dirs = [os.path.join(d, name) for d in config_dirs for name in (*TRUSTED_DIRS, *CONFIG_LAUNCH_DIRS)]
    trusted_dirs += [os.path.join(d, JOB_SHIM_DIR_NAME) for d in data_dirs]
    nodes = _node_dirs(real_home)
    for directory in nodes:
        # The node directory itself is a denied subpath (renaming it is writing it); the folders above it too.
        parents += [p for p in _parents_below(directory, real_home) if p not in parents]
    for path in trusted_files:
        # Each file's folders too, for the same reason as the registry's.
        parents += [p for p in _parents_below(path, real_home) if p not in parents]
    boot_files, boot_dirs = startup_files(real_home), startup_dirs(real_home)
    for path in (*boot_files, *boot_dirs):
        # What starts Prometheus, and the folders above it: moving the drop-in folder or the deploy clone aside and
        # putting another in its place is a write to the folder's own path.
        parents += [p for p in _parents_below(path, real_home) if p not in parents]
    return (
        "(version 1)\n"
        "(allow default)\n"
        "(deny file-read* file-write*\n"
        + ("" if allow_ssh else f'  (subpath "{base}/.ssh")\n') +
        f'  (subpath "{base}/.gnupg")\n'
        + "".join(f'  (subpath "{_sb_string(d)}")\n' for d in pairing)
        + "".join(f'  (subpath "{_sb_string(d)}")\n' for d in nodes)
        + "".join(f'  (require-all (subpath "{_sb_string(d)}") {_MCP_STORE_REGEX})\n' for d in data_dirs)
        + "".join(f'  (require-all (subpath "{_sb_string(d)}") {_XAI_OAUTH_REGEX})\n' for d in config_dirs) +
        f'  (require-all (subpath "{base}/.config") (regex #"/\\.config/[^/]+/[^/]*env$")))\n'
        "(deny file-write*\n"
        + "".join(f'  (require-all (subpath "{_sb_string(d)}") {_REGISTRY_REGEX})\n' for d in data_dirs)
        + "".join(f'  (require-all (subpath "{_sb_string(d)}") {_CRON_REGEX})\n' for d in data_dirs)
        + "".join(f'  (require-all (subpath "{_sb_string(d)}") {_LCM_REGEX})\n' for d in data_dirs)
        + "".join(f'  (literal "{_sb_string(p)}")\n' for p in trusted_files)
        + "".join(f'  (subpath "{_sb_string(d)}")\n' for d in trusted_dirs)
        + "".join(f'  (literal "{_sb_string(p)}")\n' for p in boot_files)
        + "".join(f'  (subpath "{_sb_string(d)}")\n' for d in boot_dirs)
        + "".join(f'  (literal "{_sb_string(p)}")\n' for p in parents) +
        ")\n"
    )


def _probe_seatbelt(allow_ssh: bool = False) -> tuple[bool, str]:
    """Run a shell under the profile against a scratch home and see what it could read.

    Verified BY OUTCOME, as the AppArmor probe is: an ordinary file must be readable, a ``.ssh`` file, a
    ``.config/*/env`` file and a pairing secret must be refused, and so must renaming ``~/.config`` out from under
    the env rule. ``sandbox-exec`` exiting 0 alone proves only that it ran.
    """
    if not os.path.exists(SANDBOX_EXEC):
        return False, f"{SANDBOX_EXEC} is not on this Mac"
    ssh_check = ('cat "$1/.ssh/probe" >/dev/null 2>&1 || exit 6; ' if allow_ssh
                 else 'cat "$1/.ssh/probe" >/dev/null 2>&1 && exit 4; ')
    script = ('cat "$1/ordinary" >/dev/null || exit 3; '
              + ssh_check +
              'cat "$1/.config/prometheus/env" >/dev/null 2>&1 && exit 5; '
              f'cat "$1/{MAC_PAIRING_DIR}/pair.secret" >/dev/null 2>&1 && exit 7; '
              'mv "$1/.config" "$1/moved" >/dev/null 2>&1 && exit 8; '
              f'printf x >> "$1/.prometheus/data/{REGISTRY_NAME}" 2>/dev/null && exit 9; '
              'mv "$1/.prometheus/data" "$1/moved-data" >/dev/null 2>&1 && exit 10; '
              'printf x > "$1/.prometheus/data/beside" || exit 11; '
              'printf x >> "$1/.prometheus/AGENTS.md" 2>/dev/null && exit 12; '
              'printf x > "$1/.prometheus/skills/planted.md" 2>/dev/null && exit 13; '
              'printf x >> "$1/.prometheus/data/cron_jobs.json" 2>/dev/null && exit 14; '
              'cat "$1/.prometheus/node/node.key" >/dev/null 2>&1 && exit 15; '
              'cat "$1/.prometheus/data/mcp_servers.json" >/dev/null 2>&1 && exit 16; '
              'mv "$1/.prometheus/node" "$1/moved-node" >/dev/null 2>&1 && exit 17; '
              'printf x > "$1/.prometheus/data/job-python/curl" 2>/dev/null && exit 18; '
              'printf x >> "$1/.prometheus/data/lcm.db" 2>/dev/null && exit 19; '
              'printf x >> "$1/.prometheus/config_pins.yaml" 2>/dev/null && exit 20; '
              'printf x > "$1/.prometheus/checkpoints/planted" 2>/dev/null && exit 21; '
              'printf x > "$1/.prometheus/profiles/planted.yaml" 2>/dev/null && exit 22; '
              f'cat "$1/.prometheus/{XAI_OAUTH_NAME}" >/dev/null 2>&1 && exit 23; '
              f'printf x >> "$1/Library/LaunchAgents/{LAUNCH_AGENT_NAME}" 2>/dev/null && exit 24; '
              f'printf x > "$1/.config/systemd/user/{SYSTEMD_UNIT}.d/planted.conf" 2>/dev/null && exit 25; '
              'ln -sfn /tmp "$1/prometheus-venvs/current" >/dev/null 2>&1 && exit 26; '
              'printf x >> "$1/prometheus-deploy/scripts/deploy_guard.sh" 2>/dev/null && exit 27; '
              'echo FLOOR_HELD')
    try:
        with tempfile.TemporaryDirectory(prefix="prometheus-read-floor-") as tmp:
            home = Path(tmp).resolve()
            (home / ".ssh").mkdir()
            (home / ".ssh" / "probe").write_text("x\n")
            (home / ".config" / "prometheus").mkdir(parents=True)
            (home / ".config" / "prometheus" / "env").write_text("x\n")
            (home / MAC_PAIRING_DIR).mkdir(parents=True)
            (home / MAC_PAIRING_DIR / "pair.secret").write_text("x\n")
            (home / ".prometheus" / "data").mkdir(parents=True)
            (home / ".prometheus" / "data" / REGISTRY_NAME).write_text("x\n")
            (home / ".prometheus" / "AGENTS.md").write_text("x\n")
            (home / ".prometheus" / "skills").mkdir()
            (home / ".prometheus" / "data" / "cron_jobs.json").write_text("{}\n")
            (home / ".prometheus" / NODE_DIR_NAME).mkdir()
            (home / ".prometheus" / NODE_DIR_NAME / "node.key").write_text("x\n")
            (home / ".prometheus" / "data" / MCP_STORE_NAME).write_text("{}\n")
            (home / ".prometheus" / "data" / JOB_SHIM_DIR_NAME).mkdir()
            (home / ".prometheus" / "data" / LCM_NAME).write_text("x\n")
            (home / ".prometheus" / "config_pins.yaml").write_text("{}\n")
            (home / ".prometheus" / "checkpoints").mkdir()
            (home / ".prometheus" / "profiles").mkdir()
            (home / ".prometheus" / XAI_OAUTH_NAME).write_text("{}\n")
            (home / "Library" / "LaunchAgents").mkdir(parents=True)
            (home / "Library" / "LaunchAgents" / LAUNCH_AGENT_NAME).write_text("<plist/>\n")
            (home / ".config" / "systemd" / "user" / f"{SYSTEMD_UNIT}.d").mkdir(parents=True)
            (home / "prometheus-venvs" / "abc").mkdir(parents=True)
            (home / "prometheus-venvs" / "current").symlink_to(home / "prometheus-venvs" / "abc")
            (home / "prometheus-deploy" / "scripts").mkdir(parents=True)
            (home / "prometheus-deploy" / "scripts" / "deploy_guard.sh").write_text("#!/bin/sh\n")
            (home / "ordinary").write_text("x\n")
            proc = subprocess.run(
                [SANDBOX_EXEC, "-p", seatbelt_profile(home, allow_ssh=allow_ssh), "/bin/sh", "-c", script, "probe",
                 str(home)],
                capture_output=True, text=True, timeout=15,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"sandbox-exec could not be run: {exc}"
    if proc.returncode == 0 and "FLOOR_HELD" in (proc.stdout or ""):
        return True, ("seatbelt via sandbox-exec: ~/.ssh, ~/.config/*/*env and the pairing secret refused, "
                      "renaming ~/.config refused, writing the device registry refused, writing trusted state "
                      "(config, identity, memory, skills) refused, reading the node keys refused, and the MCP store and the "
                      "xAI tokens, "
                      "writing launch configuration refused, writing what starts Prometheus refused, "
                      "ordinary reads allowed")
    why = {
        3: "an ordinary file could not be read under the profile",
        4: "~/.ssh stayed readable under the profile",
        5: "~/.config/*/*env stayed readable under the profile",
        6: "~/.ssh was refused although security.bash_allow_ssh is on",
        7: "the pairing secret stayed readable under the profile",
        8: "~/.config could be renamed out from under the env rule",
        9: "the device registry stayed writable under the profile",
        10: "the data directory could be renamed out from under the device-registry rule",
        11: "a file beside the device registry could not be written under the profile",
        12: "the trusted state (the identity files) stayed writable under the profile",
        13: "a skill could be planted in the trusted state's skills/ under the profile",
        14: "the scheduled-jobs schedule stayed writable under the profile",
        15: "the node keys stayed readable under the profile",
        16: "the MCP store stayed readable under the profile",
        17: "the node directory could be renamed out from under its rule",
        18: "a command could be planted in the cron jobs' PATH shims (job-python/) under the profile",
        19: "lcm.db (each conversation's workspace and profile binding) stayed writable under the profile",
        20: "config_pins.yaml stayed writable under the profile",
        21: "the file checkpoints stayed writable under the profile",
        22: "a custom agent profile could be planted under the profile",
        23: "the xAI OAuth tokens stayed readable under the profile",
        24: "the LaunchAgent plist stayed writable under the profile",
        25: "a systemd drop-in could be planted for the daemon's unit under the profile",
        26: "the deploy's current-venv link could be re-pointed under the profile",
        27: "the deploy clone's scripts/ stayed writable under the profile",
    }.get(proc.returncode)
    return False, why or (f"sandbox-exec did not run the probe (rc={proc.returncode})"
                          + (f": {(proc.stderr or '').strip()[:200]}" if (proc.stderr or "").strip() else ""))


def _seatbelt_key(allow_ssh: bool) -> str:
    return _SEATBELT_CACHE_KEY + (":allow-ssh" if allow_ssh else "")


def seatbelt_preflight(*, force: bool = False, allow_ssh: bool = False) -> tuple[bool, str]:
    """Cached per-process check that the Seatbelt floor genuinely holds (see :func:`preflight`), for the profile
    that will run: with *allow_ssh* it must let ``~/.ssh`` through and still refuse ``~/.config/*/*env``."""
    key = _seatbelt_key(allow_ssh)
    if force or key not in _preflight_cache:
        ok, detail = _probe_seatbelt(allow_ssh)
        _preflight_cache[key] = (ok, detail)
        if ok:
            logger.info("bash confinement ACTIVE: %s", detail)
        else:
            logger.error("bash confinement UNAVAILABLE: %s", detail)
    return _preflight_cache[key]


def seatbelt_refusal_message(detail: str) -> str:
    """What the operator sees instead of an unconfined shell, on a Mac."""
    return (
        "bash REFUSED: the read floor is required but unavailable.\n"
        f"Reason: {detail}\n"
        f"Mechanism: Seatbelt ({SANDBOX_EXEC})\n"
        "\n"
        "The command was NOT run. The floor is what keeps bash out of ~/.ssh, ~/.gnupg, ~/.config/*/*env "
        "(the master key, the provider keys and the bot tokens live in ~/.config/prometheus/env) and the "
        "same-Mac pairing secret, which mints an owner device token. The permission "
        "gate cannot reach those paths for bash, so running unconfined would remove the floor entirely rather "
        "than degrade it.\n"
        "\n"
        "sandbox-exec ships with macOS and needs nothing loaded. If it is missing or no longer holds, set "
        "security.bash_confinement: auto (runs without the floor and says so) or off, knowingly."
    )


def _read_preflight(profile: str, allow_ssh: bool = False) -> tuple[bool, str]:
    mechanism = read_floor_mechanism()
    if mechanism == MECHANISM_SEATBELT:
        return seatbelt_preflight(allow_ssh=allow_ssh)
    if mechanism is None:
        return False, "this platform has neither AppArmor (Linux) nor sandbox-exec (macOS)"
    return preflight(profile)


def _read_cache_key(profile: str, allow_ssh: bool = False) -> str:
    return _seatbelt_key(allow_ssh) if read_floor_mechanism() == MECHANISM_SEATBELT else profile


def _read_prefix(profile: str, allow_ssh: bool = False) -> list[str]:
    if read_floor_mechanism() == MECHANISM_SEATBELT:
        return [SANDBOX_EXEC, "-p", seatbelt_profile(allow_ssh=allow_ssh)]
    return wrap_argv([], profile)


def _read_refusal(profile: str, detail: str) -> str:
    if read_floor_mechanism() == MECHANISM_SEATBELT:
        return seatbelt_refusal_message(detail)
    return refusal_message(profile, detail)


def read_mechanism_label(profile: str = PROFILE) -> str:
    """The mechanism as /api/status and ``oara doctor`` name it."""
    if read_floor_mechanism() == MECHANISM_SEATBELT:
        return _SEATBELT_CACHE_KEY
    return f"apparmor:{profile}"


# =========================================================================== #
# THE WRITE FLOOR
# =========================================================================== #
#
# WHAT IT CLOSES
# --------------
# ``SecurityGate`` raises an approval when ``write_file``/``edit_file`` name a
# ``file_path`` outside the workspace (checker.py, ``_APPROVE_TOOLS``). bash
# has no ``file_path`` — it is handed a command string — so the one tool that
# can write anywhere is the one tool that check cannot see. ``printf 'x' >
# ~/outside.txt`` lands with no prompt. ``exfiltration.py`` reads ``<`` into
# network commands; nothing reads ``>``.
#
# WHY NOT PARSE THE COMMAND
# -------------------------
# Because the list has no end. ``>``, ``>>``, ``>|``, ``tee``, ``dd of=``,
# ``sed -i``, ``cp``, ``mv``, ``install``, ``truncate``, ``python -c``,
# heredocs, ``$(…)``, ``sh -c`` of any of those, and whatever the next shell
# adds. Every one of them is the same EFFECT — a write at a path outside the
# workspace — reached by a different syntax, and a parser has to win every
# round while a shell has to win once. So the control is placed where the
# effect happens: the kernel resolves the path, and the mount it lands on is
# read-only. Seven of those syntaxes are acceptance tests
# (tests/test_bash_write_floor.py) precisely because none of them is special.
#
# HOW
# ---
# ``bwrap --ro-bind / /`` gives the command the host filesystem, entire and
# readable, mounted read-only; the workspace roots (plus scratch, below) are
# then bind-mounted read-write ON TOP. A write anywhere else fails EROFS
# inside the shell's own ``open(2)``. This is a mount namespace, not a
# sandbox: the network, the PID namespace, the environment and the device
# nodes a GPU box needs are all deliberately left alone, because the goal is
# a write boundary and every extra restriction is a way for ordinary work to
# break for reasons the operator will not connect to this.
#
# READS ARE UNCHANGED, deliberately: ``--ro-bind / /`` means bash can still
# read anything it could read before, including ~/.ssh. That is the READ
# floor's job (AppArmor, above), and conflating the two would let a green
# write-floor test be read as protection it does not provide.
#
# THE SCRATCH ALLOWANCE, stated plainly because it is a real hole in the
# boundary: /tmp, /var/tmp and ~/.cache are writable. Without them ``uv``,
# ``pip``, ``git``, ``pytest`` and most of the toolchain fail in ways that
# would get this floor switched off within a day. They are scratch and cache
# — not credentials, not config, not autostart — and the trade is made here,
# once, visibly, rather than discovered later in a bug report.

BWRAP_BIN: Final[str] = "bwrap"

WRITE_MODE_OFF: Final[str] = "off"
WRITE_MODE_AUTO: Final[str] = "auto"
WRITE_MODE_REQUIRED: Final[str] = "required"
VALID_WRITE_MODES: Final[tuple[str, ...]] = (
    WRITE_MODE_OFF, WRITE_MODE_AUTO, WRITE_MODE_REQUIRED,
)

#: Writable in addition to the workspace roots. See "THE SCRATCH ALLOWANCE".
SCRATCH_DIRS: Final[tuple[str, ...]] = ("/tmp", "/var/tmp")

#: Device nodes bound read-WRITE when present. A GPU box's bash runs
#: ``nvidia-smi`` and CUDA; under the synthetic ``--dev`` those nodes vanish
#: and every GPU command breaks. Writing to a character device is not a
#: filesystem escape, so this costs the boundary nothing.
DEVICE_BIND_GLOBS: Final[tuple[str, ...]] = (
    "/dev/nvidia*", "/dev/dri", "/dev/kfd",
)

_write_preflight_cache: dict[tuple[str, ...], tuple[bool, str]] = {}


def normalise_write_mode(value: object) -> str:
    """Map a config value onto a known write-floor mode.

    Unknown values fail SAFE-LOUD in the same direction as
    :func:`normalise_mode`: they land on ``auto`` (the default, which never
    refuses) rather than ``required``, and they are logged at WARNING.
    """
    text = str(value if value is not None else WRITE_MODE_AUTO).strip().lower()
    if text in VALID_WRITE_MODES:
        return text
    if text in ("true", "yes", "on", "enforce", "enforced"):
        return WRITE_MODE_REQUIRED
    if text in ("false", "no", "none"):
        return WRITE_MODE_OFF
    logger.warning(
        "security.bash_write_confinement=%r is not one of %s — treating as "
        "%r.", value, list(VALID_WRITE_MODES), WRITE_MODE_AUTO,
    )
    return WRITE_MODE_AUTO


def writable_roots(
    workspaces: Iterable[Path | str],
    extra: Iterable[Path | str] = (),
) -> tuple[Path, ...]:
    """The read-write set: workspace roots + scratch + cache + operator extras.

    Order is preserved and duplicates are dropped, because bwrap applies
    binds in argv order and a later bind at a nested path shadows an earlier
    one — the same ordering rule ``BwrapSandbox._bwrap_argv`` documents.
    """
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    candidates = [
        *(Path(w).expanduser() for w in workspaces if w),
        *(Path(s) for s in SCRATCH_DIRS),
        Path(cache),
        *(Path(e).expanduser() for e in extra if e),
    ]
    seen: list[Path] = []
    for c in candidates:
        try:
            resolved = c.resolve()
        except OSError:  # pragma: no cover — unresolvable path
            continue
        if resolved not in seen:
            seen.append(resolved)
    return tuple(seen)


def write_wrap_argv(
    argv: Sequence[str],
    *,
    writable: Sequence[Path],
    cwd: Path | str | None = None,
) -> list[str]:
    """Prefix an argv with the bwrap write floor.

    ``--new-session`` is NOT passed, deliberately. It calls ``setsid``, which
    would put the inner shell in its own process group — and
    ``BashTool._kill_process_group`` kills the OUTER group, so a timed-out
    command's real work would survive the kill and reparent to init. That is
    the orphaned-``find`` bug the killpg handling exists to prevent, and
    re-introducing it here would be invisible until a runaway command was
    already thrashing the disk.
    """
    exe = shutil.which(BWRAP_BIN) or BWRAP_BIN
    wrapped = [
        exe,
        "--die-with-parent",   # a killed daemon must not orphan a namespace
        "--unshare-user",      # the only unprivileged way in
        # The whole host, readable, read-only. Reads are NOT this floor's
        # business; writes are, and this is what makes every path outside the
        # rw set fail in open(2) rather than in a parser.
        "--ro-bind", "/", "/",
        "--proc", "/proc",
        # Synthetic /dev. A read-only bind of the host's /dev is not an
        # option: it makes `> /dev/null` fail with EACCES, which breaks
        # essentially every shell command including the login profile.
        "--dev", "/dev",
    ]
    for pattern in DEVICE_BIND_GLOBS:
        for node in sorted(_glob.glob(pattern)):
            wrapped += ["--dev-bind-try", node, node]
    # ⚠ ORDER IS LOAD-BEARING: this MUST stay after every --dev-bind-try above.
    #
    # `--dev /dev` mounts a synthetic tmpfs that is WRITABLE, so a command
    # aimed at a device node the namespace does not carry does not fail — it
    # creates a regular file of that name, exits 0 and prints nothing. An
    # agent asked to write a device is told the write worked. Remounting that
    # tmpfs read-only turns the lie into EROFS. The device nodes bound above
    # are separate mounts layered on top and are unaffected: measured
    # writable after the remount on a host with /dev/nvidia0 and
    # /dev/dri/renderD128, as is /dev/null.
    #
    # Emitted BEFORE those binds, bwrap cannot mkdir the bind's mountpoint on
    # a now-read-only /dev and dies before running anything:
    #
    #     bwrap: Can't mkdir /dev/dri: Read-only file system
    #
    # That is every bash call failing to start on any host with a GPU. CI has
    # no GPU, so no behavioural test can catch the reorder — the argv order
    # itself is pinned in
    # tests/test_bash_write_floor.py::TestDevRemountOrderIsPinned.
    wrapped += ["--remount-ro", "/dev"]
    for path in writable:
        # bind-TRY: a workspace root or cache dir that does not exist yet is
        # a missing bind source, and must not abort every bash call.
        wrapped += ["--bind-try", str(path), str(path)]
    if cwd is not None:
        wrapped += ["--chdir", str(cwd)]
    return [*wrapped, "--", *argv]


def _probe_write_floor(inner_prefix: Sequence[str] = ()) -> tuple[bool, str]:
    """Run the real wrapper and check the real effect.

    Verification is BY OUTCOME, for the same reason ``_probe_label`` is:
    ``bwrap`` exiting 0 proves that bwrap ran, not that anything was
    contained. So this builds an argv through :func:`write_wrap_argv` — the
    exact function the tool uses, including ``inner_prefix`` when the
    AppArmor floor is composed inside it — and asserts three things about
    what actually happened:

    * the sentinel was printed, so the inner shell started at all;
    * a write OUTSIDE the writable set failed;
    * a write INSIDE it succeeded.

    The third matters as much as the second. A namespace where nothing is
    writable would pass a blocked-write check and break every command.

    Scope note: this proves the MECHANISM binds — ro-bind holds, rw bind
    works — using a purpose-built pair of directories. It does not and
    cannot prove the POLICY (which roots the caller made writable); that is
    what tests/test_bash_write_floor.py asserts against the real set.
    """
    if shutil.which(BWRAP_BIN) is None:
        return False, (
            "bwrap (bubblewrap) is not installed — no unprivileged way to "
            "mount the read-only floor"
        )
    sentinel = f"__write_floor_probe_{uuid.uuid4().hex}__"
    try:
        with tempfile.TemporaryDirectory(prefix="prometheus-write-floor-") as tmp:
            root = Path(tmp)
            inside = root / "inside"
            inside.mkdir()
            outside = root / "outside.txt"
            script = (
                f'printf "%s\\n" {sentinel}; '
                f'if printf x > "{outside}" 2>/dev/null; then echo LEAK; '
                f'else echo BLOCKED; fi; '
                f'if printf y > "{inside}/probe" 2>/dev/null; then echo INSIDE_OK; '
                f'else echo INSIDE_BLOCKED; fi'
            )
            argv = write_wrap_argv(
                [*inner_prefix, "/bin/sh", "-c", script],
                writable=(inside,),
                cwd=inside,
            )
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"the write-floor probe could not be run: {exc}"

    out = proc.stdout or ""
    stderr = (proc.stderr or "").strip()[:200]
    if sentinel not in out:
        return False, (
            f"the confined process never ran (rc={proc.returncode})"
            + (f": {stderr}" if stderr else "")
        )
    if "LEAK" in out:
        return False, (
            "bwrap ran but the read-only bind did not hold: a write outside "
            "the writable set SUCCEEDED"
        )
    if "INSIDE_OK" not in out:
        return False, (
            "bwrap ran and blocked the outside write, but the workspace bind "
            "is not writable either — the floor would break ordinary work"
        )
    return True, "bwrap mount namespace: / read-only, writable set bound rw"


def write_preflight(
    *, inner_prefix: Sequence[str] = (), force: bool = False,
) -> tuple[bool, str]:
    """Cached per-process check that the write floor genuinely contains.

    Cached on ``inner_prefix``, so a composed stack (write floor OUTSIDE,
    AppArmor read floor INSIDE) is probed as the stack it will actually be,
    and never inherits a bare probe's success. There is no claim here that
    the two compose on any given host — the probe finds out, and a stack
    that does not compose fails closed exactly like a missing profile.

    Caching is safe in the same one direction as :func:`preflight`: if bwrap
    is removed after a cached success, the next call execs a missing binary
    and the command fails, rather than running unconfined.
    """
    key = tuple(inner_prefix)
    if force or key not in _write_preflight_cache:
        ok, detail = _probe_write_floor(inner_prefix)
        _write_preflight_cache[key] = (ok, detail)
        if ok:
            logger.info("bash WRITE floor ACTIVE: %s", detail)
        else:
            logger.error("bash WRITE floor UNAVAILABLE: %s", detail)
    return _write_preflight_cache[key]


def reset_write_cache() -> None:
    """Drop the memoised write preflight. For tests."""
    _write_preflight_cache.clear()


def write_refusal_message(detail: str) -> str:
    """What the operator sees instead of an unconfined shell, in ``required``."""
    return (
        "bash REFUSED: the kernel write floor is required but unavailable.\n"
        f"Reason: {detail}\n"
        "\n"
        "The command was NOT run. Without the floor, bash can write anywhere "
        "on the filesystem with no approval: the outside-workspace check "
        "reads a tool's file_path argument, and bash has none.\n"
        "\n"
        "Fix: install bubblewrap —\n"
        "  sudo apt install bubblewrap        # or your platform's package\n"
        "or set security.bash_write_confinement: auto to run without the "
        "floor where it is unavailable, knowingly."
    )


# =========================================================================== #
# REPORTING — the enforced state, not the configured one
# =========================================================================== #
#
# Both floors were invisible: /api/status carried nothing about either, and
# `doctor` reported the CODING sandbox backend while saying nothing about the
# floor around bash. The only evidence a floor was in force was a log line at
# preflight, so "is bash confined right now" was answered by reading config
# and inferring — which is the inference the config-dark law exists to remove
# (the same reason /api/status grew its `compaction` block).
#
# A configured mode is NOT a state. The five states below are what an operator
# actually needs to tell apart, and only two of them are visible from config:
#
#   off       no floor asked for.
#   active    asked for AND verified working, by outcome.
#   dark      asked for, NOT working, and bash runs anyway. Only "auto" can
#             reach this — either floor's, both of them shipped defaults — it
#             is the documented degradation, and it is the state that most
#             needs saying out loud, because nothing else about it is visible.
#   unsupported  the read floor in "auto" on a platform with no AppArmor
#             (macOS). Nothing that could exist is missing, so not dark.
#   refusing  asked for as "required", NOT working, so every bash call now
#             fails. Loud by construction, but an operator staring at a wall
#             of refusals should find the reason here.
#   no-workspace  a floor was asked for with no workspace root to bound. The
#             write floor has no boundary to enforce and does nothing.

STATE_OFF: Final[str] = "off"
STATE_ACTIVE: Final[str] = "active"
STATE_DARK: Final[str] = "dark"
STATE_REFUSING: Final[str] = "refusing"
STATE_NO_WORKSPACE: Final[str] = "no-workspace"
#: The read floor in "auto" on a platform with neither AppArmor nor sandbox-exec.
#: Not dark: nothing that could exist is missing. Reported, never alerted on.
STATE_UNSUPPORTED: Final[str] = READ_UNSUPPORTED


def floor_report(
    *,
    read_mode: object = MODE_OFF,
    write_mode: object = WRITE_MODE_AUTO,
    has_workspace: bool = True,
    probe: bool = True,
    allow_ssh: bool = False,
) -> dict[str, object]:
    """What both bash floors are configured to do, and what they actually do.

    Deliberately probes rather than reporting config back. Both probes are
    memoised per process, so the cost is paid once and every later call is a
    dict lookup — and ``probe=False`` reports from the cache alone (``None``
    availability where nothing has probed yet) for callers that must not
    spawn a subprocess.

    The write floor is probed through the SAME composition the bash tool
    builds — read floor inside, write floor outside — so a stack that does
    not compose reports the state the next bash call will actually get,
    rather than a bare probe's optimism.

    Config knowledge stays out of here on purpose: callers resolve their own
    modes and workspace roots, and this stays a permissions primitive.
    """
    read = normalise_mode(read_mode)
    write = normalise_write_mode(write_mode)

    read_ok: bool | None = None
    read_detail = "not probed"
    inner_prefix: list[str] = []
    unsupported = read == MODE_AUTO and read_floor_mechanism() is None
    if unsupported:
        read_detail = "this platform has neither AppArmor nor sandbox-exec"
    elif read in (MODE_REQUIRED, MODE_AUTO):
        read_key = _read_cache_key(PROFILE, allow_ssh)
        if probe:
            read_ok, read_detail = _read_preflight(PROFILE, allow_ssh)
        elif read_key in _preflight_cache:
            read_ok, read_detail = _preflight_cache[read_key]
        # The tool wraps with the read floor (aa-exec, or sandbox-exec on a Mac)
        # whenever it is required, and in auto once it verified, so the write
        # probe carries the same prefix: that is the argv the next call gets.
        if read == MODE_REQUIRED or read_ok:
            inner_prefix = _read_prefix(PROFILE, allow_ssh)

    write_ok: bool | None = None
    write_detail = "not probed"
    if write != WRITE_MODE_OFF and has_workspace:
        key = tuple(inner_prefix)
        if probe:
            write_ok, write_detail = write_preflight(inner_prefix=inner_prefix)
        elif key in _write_preflight_cache:
            write_ok, write_detail = _write_preflight_cache[key]

    def _state(mode: str, ok: bool | None, *, refusing_mode: str) -> str:
        # has_workspace is NOT consulted here. It bounds the write floor only
        # (the read floor's boundary is the profile, not a root), and the
        # write caller handles it before reaching this.
        if mode == MODE_OFF:
            return STATE_OFF
        if ok is None:
            return "unknown"
        if ok:
            return STATE_ACTIVE
        return STATE_REFUSING if mode == refusing_mode else STATE_DARK

    read_state = (STATE_UNSUPPORTED if unsupported
                  else _state(read, read_ok, refusing_mode=MODE_REQUIRED))
    if write == WRITE_MODE_OFF:
        write_state = STATE_OFF
    elif not has_workspace:
        write_state = STATE_NO_WORKSPACE
    else:
        write_state = _state(write, write_ok, refusing_mode=WRITE_MODE_REQUIRED)

    return {
        "bash_read_floor": {
            "mode": read,
            "state": read_state,
            "available": read_ok,
            "detail": read_detail,
            "mechanism": read_mechanism_label(PROFILE),
            "covers": ("reads of ~/.gnupg, ~/.config/*/*env (~/.ssh allowed: security.bash_allow_ssh)"
                       if allow_ssh and read_floor_mechanism() == MECHANISM_SEATBELT
                       else "reads of ~/.ssh, ~/.gnupg, ~/.config/*/*env"),
            "allow_ssh": allow_ssh,
        },
        "bash_write_floor": {
            "mode": write,
            "state": write_state,
            "available": write_ok,
            "detail": write_detail,
            "mechanism": "bwrap",
            "covers": "writes outside the workspace roots",
            "composed_with_read_floor": bool(inner_prefix),
        },
        # THE ONE FIELD A MONITOR SHOULD ALERT ON. True when something was
        # asked for and is not happening while bash runs anyway — the state
        # that is invisible everywhere else.
        "dark": read_state == STATE_DARK or write_state == STATE_DARK,
        # SEPARATE from dark on purpose: this is loud (every bash call fails),
        # so it needs an explanation, not an alert.
        "refusing": read_state == STATE_REFUSING or write_state == STATE_REFUSING,
        # Named so a reader does not generalise it. Every shell a MODEL wrote
        # goes through apply_floors below; command hooks are the operator's
        # own commands from prometheus.yaml and are deliberately not floored
        # (their environment is allowlisted instead, hooks/executor.py).
        "scope": SHELL_FLOOR_SCOPE,
    }


#: Which shells the two floors reach — the doors that hand a model-written
#: command to a shell, all of them through :func:`apply_floors`.
SHELL_FLOOR_SCOPE: Final[str] = (
    "every shell a model writes: the bash tool, background shell tasks and "
    "poll predicates (task_create), cron jobs the model created (cron_create, "
    "origin \"model\"), and a coding run's commands (process backend). Not "
    "command hooks, and not the operator's own cron jobs: those run the "
    "operator's own commands as before"
)


# =========================================================================== #
# ONE COMPOSITION FOR EVERY MODEL-WRITTEN SHELL
# =========================================================================== #
#
# The bash tool built this stack inline, so it was the only door that had it.
# A background task, a poll predicate, a cron job and a coding run's code_run
# each take a command string from the model and start a shell at their own
# call site, and the floor that keeps bash out of ~/.ssh was one task_create
# away from not existing. The stack now lives here, once, and every one of
# those sites calls it — so a fix to the composition (an ordering bug, a new
# refusal) lands at every door at the same time instead of at one.


@dataclass(frozen=True)
class FloorResult:
    """What :func:`apply_floors` decided for one shell call.

    ``argv`` is what to exec when ``refusal`` is None. ``write_floor`` is the
    per-call state the bash tool already reports in its result metadata:
    ``off``, ``active``, ``unavailable`` (auto, degraded), ``no-workspace``,
    ``refused`` or ``not-reached`` (the read floor refused first).
    """

    argv: tuple[str, ...]
    write_floor: str
    refusal: str | None = None
    #: ``off``, ``active``, ``unavailable`` (auto, unverified), ``unsupported``
    #: (auto, neither AppArmor nor sandbox-exec here) or ``refused`` (required).
    read_floor: str = MODE_OFF


def apply_floors(
    argv: Sequence[str],
    *,
    read_mode: object,
    write_mode: object,
    writable: Sequence[Path],
    cwd: Path | str | None,
    profile: str = PROFILE,
    allow_ssh: bool = False,
) -> FloorResult:
    """Wrap a shell argv in the read floor (inside) and write floor (outside).

    Moved here verbatim from ``BashTool.execute`` so every model-written
    shell gets the same stack and the same failure policy:

    * READ floor required and unavailable -> REFUSE. Never fall through to an
      unconfined shell; that removes the floor rather than degrading it.
    * WRITE floor: ``required`` refuses when unavailable, ``auto`` degrades to
      no floor and says so (``unavailable``), and with no writable root there
      is no boundary to enforce (``no-workspace``, logged).

    Ordering: the write floor goes OUTSIDE the AppArmor transition, so aa-exec
    runs inside the mount namespace, and the write probe is run against that
    composed stack — a stack that does not compose fails closed.
    """
    read = normalise_mode(read_mode)
    write = normalise_write_mode(write_mode)
    out = list(argv)

    aa_prefix: list[str] = []
    read_floor = MODE_OFF
    if read == MODE_AUTO and read_floor_mechanism() is None:
        read_floor = READ_UNSUPPORTED
    elif read in (MODE_REQUIRED, MODE_AUTO):
        ok, detail = _read_preflight(profile, allow_ssh)
        if ok:
            # Kept as a PREFIX rather than applied once, so the write floor
            # below can probe the composed stack as the stack it will be.
            # AppArmor's aa-exec on Linux, Seatbelt's sandbox-exec on a Mac.
            aa_prefix = _read_prefix(profile, allow_ssh)
            out = [*aa_prefix, *out]
            read_floor = READ_ACTIVE
        elif read == MODE_REQUIRED:
            return FloorResult((), "not-reached", _read_refusal(profile, detail),
                               read_floor="refused")
        else:
            # auto, unverified: run without the read floor. preflight() has
            # logged why at ERROR, once; announce() said it at boot with the
            # fix; this records it on the call that ran without it.
            read_floor = READ_UNAVAILABLE

    write_floor = WRITE_MODE_OFF
    if write != WRITE_MODE_OFF:
        if not writable:
            # Loud, not silent: a floor asked for with no root to bound.
            write_floor = STATE_NO_WORKSPACE
            logger.warning(
                "the shell write floor is %r but no workspace root is "
                "configured — there is no boundary to enforce and the "
                "command may write anywhere.", write,
            )
        else:
            ok, detail = write_preflight(inner_prefix=aa_prefix)
            if ok:
                out = write_wrap_argv(out, writable=writable, cwd=cwd)
                write_floor = STATE_ACTIVE
            elif write == WRITE_MODE_REQUIRED:
                return FloorResult((), "refused", write_refusal_message(detail),
                                   read_floor=read_floor)
            else:
                # "auto" degrades to the unfloored shell. write_preflight()
                # has logged the reason at ERROR, once; the per-call state
                # records it as well, so a floor that is not there is legible
                # from the call that ran without it.
                write_floor = "unavailable"
    return FloorResult(tuple(out), write_floor, read_floor=read_floor)
