"""A cron job's ``python3`` is the daemon's own interpreter.

WHY. The deployed daemon runs from a venv built from ``uv.lock``, and its
systemd drop-in sets ``PYTHONNOUSERSITE=1`` (``scripts/deploy.sh`` B3, gate G1:
nothing from ``~/.local``). Every job shell the daemon spawns inherits that.
A job that runs bare ``python3`` therefore gets the SYSTEM interpreter WITHOUT
the user site its packages lived in. ``daily_news_briefing_pm``
(``python3 -m prometheus.jobs.daily_briefing``) died on
``No module named 'pydantic'`` every night from 2026-09-24 until it was edited
by hand on 2026-10-03.

WHAT. ``job_shell_command`` wraps a job's command so that, inside the job's
shell, ``python3`` and ``python`` resolve to ``sys.executable``: the
interpreter the daemon itself runs, with the packages ``uv.lock`` describes.
The wrappers are two small scripts in ``<data dir>/job-python``, put first on
PATH AFTER the login shell has read its profile (the jobs run under
``bash -lc``, and ``~/.profile`` prepends to PATH), so a command finds them
wherever it says ``python3``: ``cd x && python3``, ``timeout 60 python3``,
``env python3``. An absolute interpreter path is the job author's choice and
is left alone.

CRON ONLY (Will, 2026-10-04). Used by the cron scheduler
(``gateway/cron_scheduler.execute_job``) and nowhere else. The command is
vetted, logged and recorded as written; only the shell that runs it changes.

* NOT background tasks (``tasks/manager``). The model can start those, and
  model-run code must never get the daemon's own venv interpreter: a
  ``python3 -m pip install`` would install into the production venv. A task's
  ``python3`` stays whatever its PATH finds.
* NOT the agent's ``bash`` tool, for the same reason; it is not a job.
* NOT a cron job the MODEL created (``cron_create`` stores ``origin:
  "model"``; see ``cron_service.is_model_job``), for the same reason again:
  it is model-run code, and it runs behind the shell floor instead. Only the
  operator's jobs — created through ``POST /api/cron``, or stored before jobs
  recorded who wrote them — get the daemon's interpreter.

ACCEPTED TRADE-OFF (Will, 2026-10-04). A cron job that needs a package
installed only for the SYSTEM Python (an apt ``python3-*`` package) and not in
the venv will no longer find it under bare ``python3``. Such a job names its
interpreter with an absolute path (``/usr/bin/python3``), which this leaves
alone. The alternative, giving cron children back the user site, would put the
user-site packages that sit below the repo's security floors (the reason for
gate G1) back under every job.
"""

from __future__ import annotations

import logging
import os
import shlex
import sys
from pathlib import Path

from prometheus.config.paths import get_data_dir

log = logging.getLogger(__name__)

#: The names a job may call the interpreter by.
SHIM_NAMES = ("python3", "python")

_SHIM_DIR = "job-python"


def _wrapper(interpreter: str) -> str:
    return (
        "#!/bin/sh\n"
        "# Written by Prometheus (utils/job_python.py): in a cron job,\n"
        "# `python3` is the daemon's own interpreter. Rewritten when it changes.\n"
        f"exec {shlex.quote(interpreter)} \"$@\"\n"
    )


def ensure_python_shims() -> Path:
    """Write (or refresh) the wrappers for the current ``sys.executable``; return their dir.

    Idempotent and cheap when nothing changed: each wrapper is rewritten only
    when its content differs, through a temp file and an atomic rename, so a
    job starting at the same moment never sees half a script.
    """
    shim_dir = get_data_dir() / _SHIM_DIR
    shim_dir.mkdir(parents=True, exist_ok=True)
    body = _wrapper(sys.executable)
    for name in SHIM_NAMES:
        path = shim_dir / name
        try:
            if path.read_text() == body and os.access(path, os.X_OK):
                continue
        except OSError:
            pass
        tmp = shim_dir / f".{name}.{os.getpid()}.tmp"
        tmp.write_text(body)
        tmp.chmod(0o755)
        os.replace(tmp, path)
    return shim_dir


def job_shell_command(command: str) -> str:
    """The script a cron job's ``bash -lc`` runs: ``command``, with ``python3`` = the daemon's.

    If the wrappers cannot be written the command runs unchanged and the miss
    is a WARNING: a job is never refused over this.
    """
    try:
        shim_dir = ensure_python_shims()
    except Exception:
        log.warning("job python3: could not write the interpreter wrappers; the job's "
                    "python3 stays whatever PATH finds", exc_info=True)
        return command
    return f'export PATH={shlex.quote(str(shim_dir))}:"$PATH"\n{command}'
