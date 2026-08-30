"""Kernel-enforced floors for the bash tool.

TWO FLOORS, TWO MECHANISMS, ONE DOCTRINE
----------------------------------------
* The READ floor (AppArmor, ``aa-exec -p prometheus-bash``) keeps bash out
  of three secret path families. Needs root to load a profile.
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
import tempfile
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
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
