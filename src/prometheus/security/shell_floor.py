"""shell_floor — the floors a MODEL-WRITTEN shell command runs under, at every door.

WHY THIS EXISTS
---------------
The bash tool runs its command behind two kernel floors (the AppArmor read
floor, the bubblewrap write floor — ``permissions/confinement.py``) and with a
scrubbed environment (``security/env_scrub.py``). Four other doors take a
command string from the model and start a shell at their own call site:

    task_create type=local_bash   tasks/manager.py           _start_process
    task_create type=poll         tasks/watchers.py          _run_predicate
    cron_create                   gateway/cron_scheduler.py  execute_job
    a coding run's code_run       coding/sandbox.py          ProcessSandbox.run

and none of them had either floor, and the first three inherited the daemon's
whole environment. So ``cat ~/.config/prometheus/env`` — the D15 route to the
API token — was refused through ``bash`` on a host with the floor and allowed
through ``task_create``.

This module is the one place those doors get their floor from, so they cannot
be configured apart from the bash tool:

* ``create_tool_registry`` — where the daemon AND the CLI build the bash tool —
  wires the same ``security`` section here (:func:`set_shell_floor`).
* A process that never wired it (tests, a coding run's child, a script) reads
  the floor from the config, the way cron's gate builds itself when nobody
  wired one. A model-written shell is never left without the floor the
  config asks for because some caller forgot a line.

WHAT IS NOT FLOORED, AND WHY
----------------------------
* Commands the DAEMON built — a sub-agent launch (``python -m prometheus``)
  and a coding run's launcher. They go through the same spawn site but carry
  no model-written shell, need the daemon's environment (provider keys) and
  write its state directories. The model's shells inside them are floored
  where they run: the child's own bash tool, and the coding sandbox.
* Command hooks. The command is the operator's, from ``prometheus.yaml``; the
  model only fires the hook and its payload travels as data, never as code.
  Their environment is allowlisted (``hooks/executor.py``).

The composition itself — read floor inside, write floor outside, and the
refusal policy — is ``confinement.apply_floors``. Nothing here re-implements
it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prometheus.permissions import confinement as _CONFINE

log = logging.getLogger(__name__)

#: How a model-written command is handed to a shell at the task, poll and cron
#: doors — the same login shell the bash tool uses.
LOGIN_SHELL: tuple[str, ...] = ("/bin/bash", "-lc")


class ShellFloorRefused(RuntimeError):
    """The floor is required and this host cannot provide it. Nothing ran.

    ``str(exc)`` is the operator-facing refusal (it names the reason and the
    fix), the same text the bash tool returns.
    """

    def __init__(self, message: str, write_floor: str) -> None:
        super().__init__(message)
        self.write_floor = write_floor


@dataclass(frozen=True)
class ShellFloor:
    """The configured floors, resolved. No probing happens here."""

    read_mode: str = _CONFINE.MODE_AUTO
    write_mode: str = _CONFINE.WRITE_MODE_AUTO
    workspaces: tuple[Path, ...] = ()
    write_allow: tuple[str, ...] = field(default=())
    profile: str = _CONFINE.PROFILE

    @classmethod
    def from_security_config(cls, security_cfg: dict[str, Any] | None) -> "ShellFloor":
        """Read the keys the bash tool reads, through the same resolvers."""
        from prometheus.config.shipped_defaults import resolve_workspace_root

        sec = security_cfg or {}
        roots = resolve_workspace_root(sec)
        if isinstance(roots, str):
            roots = [roots]
        return cls(
            read_mode=_CONFINE.normalise_mode(sec.get("bash_confinement", "auto")),
            write_mode=_CONFINE.normalise_write_mode(
                sec.get("bash_write_confinement", "auto")),
            workspaces=tuple(
                Path(r).expanduser().resolve() for r in roots if r),
            write_allow=tuple(str(p) for p in (sec.get("bash_write_allow") or ())),
        )


_WIRED: ShellFloor | None = None
#: (config path, mtime) -> floor, so an unwired process reads the file once
#: per change rather than once per shell (and logs an absent file once).
_FROM_CONFIG: dict[tuple[str, int | None], ShellFloor] = {}


def set_shell_floor(floor: ShellFloor | None) -> None:
    """Wire the floor every model-written shell runs under. ``None`` unwires,
    which means "read it from the config", never "no floor"."""
    global _WIRED
    _WIRED = floor


def current_shell_floor() -> ShellFloor:
    """The wired floor, else the config's — never an unconfigured default
    because a caller forgot to wire one."""
    if _WIRED is not None:
        return _WIRED
    from prometheus.config.defaults import resolve_config_path
    from prometheus.config.load import load_config_file

    path = resolve_config_path()
    try:
        mtime: int | None = path.stat().st_mtime_ns
    except OSError:
        mtime = None
    key = (str(path), mtime)
    floor = _FROM_CONFIG.get(key)
    if floor is None:
        load = load_config_file(
            path,
            subsystem="shell_floor",
            substituting=(
                "the shipped shell floors (read floor off, write floor auto "
                "over the shipped workspace root)"),
            explicit=False,
        )
        floor = ShellFloor.from_security_config(load.section("security"))
        _FROM_CONFIG.clear()
        _FROM_CONFIG[key] = floor
    return floor


def floored_argv(
    command: str,
    *,
    cwd: Path | str,
    workspaces: Iterable[Path | str] | None = None,
    shell: Sequence[str] = LOGIN_SHELL,
    floor: ShellFloor | None = None,
) -> tuple[list[str], _CONFINE.FloorResult]:
    """The argv that runs *command* behind the floors, and what each floor did
    (``.read_floor``, ``.write_floor``) — for the call's own record.

    *workspaces* replaces the configured roots for this call — a session with a
    workspace of its own, or a coding run whose root is its clone — exactly as
    the bash tool lets a session's roots replace the configured ones.

    Raises :class:`ShellFloorRefused` when a floor is required and
    unavailable. The caller must not start anything in that case.
    """
    floor = floor or current_shell_floor()
    roots = tuple(
        Path(r).expanduser().resolve() for r in (workspaces or ()) if r
    ) or floor.workspaces
    writable = _CONFINE.writable_roots(roots, floor.write_allow) if roots else ()
    result = _CONFINE.apply_floors(
        [*shell, command],
        read_mode=floor.read_mode,
        write_mode=floor.write_mode,
        writable=writable,
        cwd=cwd,
        profile=floor.profile,
    )
    if result.refusal is not None:
        raise ShellFloorRefused(result.refusal, result.write_floor)
    return list(result.argv), result


def model_shell_env(overlay: dict[str, str] | None = None) -> dict[str, str]:
    """The environment a model-written shell gets: the bash tool's scrub.

    An *overlay* is applied on top — a caller's explicit, in-memory grant
    (none of the floored doors passes one today).
    """
    from prometheus.security.env_scrub import scrubbed_env

    env = scrubbed_env()
    if overlay:
        env.update(overlay)
    return env


def announce(floor: ShellFloor | None = None) -> dict[str, object]:
    """Say ONCE, at boot, what the read floor of every model-written shell is.

    The loud half of "auto": a host where the profile did not verify runs
    every model shell without the read floor, and an operator must not have to
    infer that from a missing line. ERROR with the one-line fix when it is not
    in force on Linux; one WARNING on a platform with no AppArmor; INFO when it
    is active or deliberately off. Returns the report it logged from.
    """
    floor = floor or current_shell_floor()
    rep = _CONFINE.floor_report(
        read_mode=floor.read_mode,
        write_mode=floor.write_mode,
        has_workspace=bool(floor.workspaces),
    )
    read = rep["bash_read_floor"]
    assert isinstance(read, dict)
    state, detail = read["state"], read["detail"]
    fix = (f"sudo apparmor_parser -r -W /etc/apparmor.d/{_CONFINE.PROFILE} "
           f"— or set security.bash_confinement: off to run without it "
           f"knowingly")
    exposed = ("every model-written shell (bash, background tasks, poll "
               "predicates, model-created cron jobs, coding runs) can read "
               "~/.ssh, ~/.gnupg and ~/.config/*/*env")
    if state == _CONFINE.STATE_DARK:
        log.error(
            "the READ floor is NOT in force: security.bash_confinement is %r and the "
            "AppArmor profile did not verify (%s), so %s. Fix: %s.",
            floor.read_mode, detail, exposed, fix,
        )
    elif state == _CONFINE.STATE_REFUSING:
        log.error(
            "READ FLOOR REQUIRED BUT UNAVAILABLE (%s): every model-written "
            "shell will be REFUSED until it is. Fix: %s.", detail, fix,
        )
    elif state == _CONFINE.STATE_UNSUPPORTED:
        log.warning(
            "read floor unsupported on this platform (no AppArmor): %s. The "
            "denied_paths list protects the path-declaring tools only.",
            exposed,
        )
    elif state == _CONFINE.STATE_ACTIVE:
        log.info("read floor ACTIVE for every model-written shell (%s)", detail)
    else:
        log.info("read floor %s (security.bash_confinement: %r)",
                 state, floor.read_mode)
    return rep
