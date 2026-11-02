"""The shell for a command the DAEMON runs outside the floor: not a login shell, and no startup file.

WHY. Three doors run a shell that is not behind the bash tool's floors:

    an operator's cron job       gateway/cron_scheduler.py  execute_job
    a daemon-built task          tasks/manager.py           _start_process (a worker, a spawned agent, a coding run)
    a command hook               hooks/executor.py          _command_launch

They ran ``bash -lc``, and a login bash runs the user's login files (``~/.bash_profile``, ``~/.bash_login``,
``~/.profile``) first. Those files are not Prometheus's to floor (they are the user's), and a model-written shell
can write them. So a line planted there by a confined shell ran UNCONFINED the next time any of the three started
(GHSA-9ff8-vhh2-r2cw). The same goes for ``$BASH_ENV`` (sourced by a non-interactive bash) and ``$ENV`` (sourced
by sh): each names a file to run before the command.

WHAT. ``bash --noprofile --norc -c``: no login file and no ``~/.bashrc`` (bash sources it for a ``-c`` shell whose
stdin is a socket). The environment is each door's own, unchanged, minus ``BASH_ENV`` and ``ENV``. PATH is the
daemon's own; an operator's cron job still gets the job-python shims in front (utils/job_python.py).

THE COST (Will, 2026-10-30). A job or hook that relied on its login profile to set PATH or export a variable must
do that itself now (``source ~/.profile`` in the command, an absolute path, or the hook's ``env_allowlist``). A
clean per-job environment waits for a later design.

The bash tool and the other model-written shells still run ``bash -lc`` (security/shell_floor.LOGIN_SHELL): they
are inside the floor, so a login file they read runs confined.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

#: How the daemon hands an unconfined command to a shell: never a login shell, never ~/.bashrc.
UNCONFINED_SHELL: tuple[str, ...] = ("/bin/bash", "--noprofile", "--norc", "-c")

#: Variables that name a file a shell runs before its command.
STARTUP_FILE_VARS: tuple[str, ...] = ("BASH_ENV", "ENV")


def unconfined_shell_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """*env* (default: this process's environment), complete, without ``BASH_ENV`` or ``ENV``."""
    src = os.environ if env is None else env
    return {k: v for k, v in src.items() if k not in STARTUP_FILE_VARS}
