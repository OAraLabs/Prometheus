# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/tools/bash_tool.py
# License: MIT
# Modified: renamed imports (openharness → prometheus);
#           added workspace_root locking (refuses commands outside allowed dir);
#           added configurable timeout (default 30s);
#           added output truncation (default 10000 chars)

"""Shell command execution tool."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.permissions import confinement as _CONFINE
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_OUTPUT = 10_000


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class BashToolInput(BaseModel):
    """Arguments for the bash tool."""

    command: str = Field(description="Shell command to execute")
    cwd: str | None = Field(default=None, description="Working directory override")
    timeout_seconds: int = Field(
        default=_DEFAULT_TIMEOUT, ge=1, le=600,
        description=(
            "Seconds before the command is killed. Raise it for a known-slow "
            "command rather than retrying after a timeout."
        ),
    )


class BashTool(BaseTool):
    """Execute a shell command with stdout/stderr capture.

    Optionally locked to a workspace_root: commands whose resolved cwd falls
    outside the workspace are refused before execution.
    """

    name = "bash"
    description = (
        "Run a short shell command in the local repository. For work that "
        "outlives this turn (long builds, downloads, file watches) use "
        "task_create instead of 'nohup … &' — a backgrounded Bash job is killed "
        "at the timeout and left untracked."
    )
    input_model = BashToolInput
    example_call = {"command": "ls -la"}

    def __init__(
        self,
        workspace: str | Path | None = None,
        max_output: int = _DEFAULT_MAX_OUTPUT,
        confinement: str = "off",
        confinement_profile: str = _CONFINE.PROFILE,
        write_confinement: str = _CONFINE.WRITE_MODE_AUTO,
        write_allow: Sequence[str | Path] = (),
    ) -> None:
        # Multi-root, mirroring SecurityGate: security.workspace_root may be
        # a list. A single root did not survive contact on a real box.
        if not workspace:
            self._workspaces: tuple[Path, ...] = ()
        elif isinstance(workspace, (str, Path)):
            self._workspaces = (Path(workspace).expanduser().resolve(),)
        else:
            self._workspaces = tuple(
                Path(w).expanduser().resolve() for w in workspace if w)
        self._workspace = self._workspaces[0] if self._workspaces else None
        self._max_output = max_output
        self._confinement = _CONFINE.normalise_mode(confinement)
        self._confinement_profile = confinement_profile
        # The write floor's boundary IS the workspace lock, one level down:
        # the lock decides where a command may start, the floor decides where
        # its writes may land. With no workspace root there is no boundary to
        # enforce, so the floor has nothing to do — the same "no roots
        # configured = no confinement" rule SecurityGate applies.
        self._write_confinement = _CONFINE.normalise_write_mode(write_confinement)
        self._write_allow = write_allow
        self._writable = (
            _CONFINE.writable_roots(self._workspaces, write_allow)
            if self._workspaces else ()
        )

    async def execute(self, arguments: BashToolInput, context: ToolExecutionContext) -> ToolResult:
        cwd = Path(arguments.cwd).expanduser().resolve() if arguments.cwd else context.cwd.resolve()

        # Item W: a session with a workspace of its own carries it in the
        # execution context, and for that call it REPLACES the configured
        # roots — the lock and the write floor both follow the session.
        session_roots = tuple(
            Path(r).expanduser().resolve()
            for r in (context.metadata.get("workspace_roots") or ())
        )
        workspaces = session_roots or self._workspaces
        primary = workspaces[0] if workspaces else None
        writable = (
            _CONFINE.writable_roots(workspaces, self._write_allow)
            if session_roots else self._writable
        )

        if workspaces:
            if any(_is_under(cwd, root) for root in workspaces):
                pass
            else:
                if not arguments.cwd:
                    # No explicit cwd requested — fall back to workspace root
                    # instead of blocking (daemon often runs from repo dir)
                    cwd = primary
                else:
                    return ToolResult(
                        output=(
                            f"Workspace lock violation: {cwd} is outside "
                            f"allowed workspace {primary}"
                        ),
                        is_error=True,
                    )

        # start_new_session=True puts the shell in its own process group, so the
        # ENTIRE pipeline (e.g. ``find … | grep …``) can be killed as a unit. A
        # bare process.kill() only signals /bin/bash; its children get reparented
        # to init and keep running — that is how a timed-out ``find`` orphaned
        # itself and thrashed the disk for minutes after the turn moved on.
        argv = ["/bin/bash", "-lc", arguments.command]

        # The READ floor, below the tool layer. The permission gate cannot see
        # the paths inside a command string, so this is the only place a bash
        # call can be stopped from reading a private key. When it is required
        # and unavailable we REFUSE — never fall through to an unconfined
        # shell, which would silently remove the floor rather than degrade it.
        aa_prefix: list[str] = []
        if self._confinement == _CONFINE.MODE_REQUIRED:
            ok, detail = _CONFINE.preflight(self._confinement_profile)
            if not ok:
                return ToolResult(
                    output=_CONFINE.refusal_message(
                        self._confinement_profile, detail),
                    is_error=True,
                    metadata={"write_floor": "not-reached"},
                )
            # Kept as a PREFIX rather than applied here, so the write floor
            # below can probe the composed stack as the stack it will be.
            aa_prefix = _CONFINE.wrap_argv([], self._confinement_profile)
            argv = [*aa_prefix, *argv]

        # The WRITE floor. Same doctrine, different mechanism and different
        # failure policy: bubblewrap needs no root, so it can be attempted
        # everywhere, but it does not EXIST everywhere (no macOS equivalent),
        # and refusing every bash call on a host that cannot provide it would
        # be a worse outcome than the hole. Hence three modes rather than two.
        #
        # Ordering: the write floor goes OUTSIDE the AppArmor transition, so
        # aa-exec runs inside the mount namespace. Whether those two compose
        # on a given host is not asserted here — write_preflight() probes the
        # composed argv and a stack that does not compose fails closed.
        write_floor = "off"
        if self._write_confinement != _CONFINE.WRITE_MODE_OFF:
            if not writable:
                # Loud, not silent: "required" with no workspace root is a
                # config that asks for a boundary without defining one.
                write_floor = "no-workspace"
                logger.warning(
                    "bash write floor is %r but no workspace root is "
                    "configured — there is no boundary to enforce and bash "
                    "may write anywhere.", self._write_confinement,
                )
            else:
                ok, detail = _CONFINE.write_preflight(inner_prefix=aa_prefix)
                if ok:
                    argv = _CONFINE.write_wrap_argv(
                        argv, writable=writable, cwd=cwd)
                    write_floor = "active"
                elif self._write_confinement == _CONFINE.WRITE_MODE_REQUIRED:
                    return ToolResult(
                        output=_CONFINE.write_refusal_message(detail),
                        is_error=True,
                        metadata={"write_floor": "refused"},
                    )
                else:
                    # "auto" degrades to today's behaviour. write_preflight()
                    # has already logged the reason at ERROR, once; this
                    # records it per call as well, because a floor that is not
                    # there must be legible from the call that ran without it
                    # and not only from a log line at startup.
                    write_floor = "unavailable"

        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=arguments.timeout_seconds,
            )
        except asyncio.TimeoutError:
            await self._kill_process_group(process)
            return ToolResult(
                output=f"Command timed out after {arguments.timeout_seconds} seconds",
                is_error=True,
                metadata={"write_floor": write_floor},
            )
        except asyncio.CancelledError:
            # The agent loop wraps tool.execute() in its own (longer) timeout;
            # when THAT fires it cancels us instead of raising TimeoutError here.
            # Without this handler the subprocess and its children would keep
            # running after the turn was abandoned (the original freeze: an
            # orphaned ``find`` still scanning $HOME). Kill the whole group,
            # then let the cancellation propagate.
            await self._kill_process_group(process)
            raise

        parts = []
        if stdout:
            parts.append(stdout.decode("utf-8", errors="replace").rstrip())
        if stderr:
            parts.append(stderr.decode("utf-8", errors="replace").rstrip())

        text = "\n".join(part for part in parts if part).strip()
        if not text:
            text = "(no output)"

        if len(text) > self._max_output:
            text = f"{text[:self._max_output]}\n...[truncated]..."

        return ToolResult(
            output=text,
            is_error=process.returncode != 0,
            metadata={"returncode": process.returncode, "write_floor": write_floor},
        )

    @staticmethod
    async def _kill_process_group(process: asyncio.subprocess.Process) -> None:
        """SIGKILL the shell's whole process group, then reap it.

        The shell is launched with ``start_new_session=True`` so it leads its own
        group; killing the group takes the entire pipeline (``find``, ``grep``,
        …) with it rather than leaving orphans. Best-effort and idempotent: the
        process or group may already be gone.
        """
        if process.returncode is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # Already dead, or we couldn't address the group — fall back to
            # signalling the shell directly so we don't leak it.
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            await process.wait()
        except Exception:  # noqa: BLE001 — reaping must never raise
            pass
