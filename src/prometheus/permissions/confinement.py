"""Kernel-enforced floor for the bash tool.

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

import logging
import shutil
import subprocess
from typing import Final

logger = logging.getLogger(__name__)

PROFILE: Final[str] = "prometheus-bash"

MODE_OFF: Final[str] = "off"
MODE_REQUIRED: Final[str] = "required"
VALID_MODES: Final[tuple[str, ...]] = (MODE_OFF, MODE_REQUIRED)

# No "preferred"/"best-effort" mode, deliberately. A mode that falls back to
# unconfined execution when the profile is missing is precisely the silent
# degradation this module exists to prevent, and it would be indistinguishable
# from a working floor in every log line.

_preflight_cache: dict[str, tuple[bool, str]] = {}


def normalise_mode(value: object) -> str:
    """Map a config value onto a known mode. Unknown values fail SAFE-LOUD.

    An unrecognised mode returns ``off`` rather than ``required`` so a typo
    cannot brick every bash call, but it is logged at WARNING so it cannot
    pass unnoticed either.
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
        _preflight_cache[profile] = (ok, detail)
        # Record the effective state. A runtime that auto-detects but leaves
        # no record is how an operator ends up believing in a floor that is
        # not there (CROSS-CUTTING §12).
        if ok:
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
