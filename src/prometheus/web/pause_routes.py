"""``PUT /api/pause`` and ``DELETE /api/pause``: pause everything, and resume (Beacon iOS "Piece 17").

The state is :mod:`prometheus.engine.pause`. These routes add who may do what, and the pause itself:

  * **Pausing only takes away**, so the operator may: the API token or an owner device (``route_access``:
    operator only; a scoped device gets ``403 operator_only``).
  * **Resuming gives everything back**, so only a person may: an owner device. Never the bare API token, which
    sits in a file a model with a shell can read; an agent that could lift its own emergency stop has not been
    stopped. ``403 person_required`` otherwise. (The owner in Telegram resumes through ``/resume``.)

Pausing, in this order: the state is written to disk (nothing new starts from here); every pending approval is
withdrawn, so nothing approved later runs a tool; every turn running in this process is stopped, the bridge's
through its own Stop and every other one by cancelling its task; desktop tasks are stopped; running tasks and
workers are stopped (their process group goes); coding runs are PAUSED through their control file, which keeps
the work; then one ``paused_changed`` frame goes to every client. Pausing again sweeps again and is still one
pause (the same ``since``).

Source: novel code for Prometheus, 2026-11-12.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from prometheus.engine import pause

logger = logging.getLogger(__name__)


def _identity(request: Request) -> Any:
    return getattr(getattr(request, "state", None), "device_identity", None)


def _who(identity: Any) -> str:
    if identity is None:
        return "open daemon"
    if getattr(identity, "is_global", False):
        return "API token"
    kind = "owner device" if getattr(identity, "owner", False) else "device"
    name = str(getattr(identity, "name", "") or getattr(identity, "id", ""))
    return f"{kind} {name}".strip()


def is_person(identity: Any) -> bool:
    """An owner device. Not the global token, not an open daemon, not a scoped device."""
    return identity is not None and not getattr(identity, "is_global", False) and bool(getattr(identity, "owner", False))


def paused_response() -> JSONResponse | None:
    """The 423 a front door answers while paused, else None. The message is not kept to run on resume."""
    info = pause.current()
    if info is None:
        return None
    return JSONResponse(status_code=423, content={
        "error": "paused", **info.to_dict(),
        "detail": "Prometheus is paused; nothing new runs until a person resumes it. This message was not kept."})


def _state() -> pause.PauseState:
    state = pause.installed()
    if state is None:
        state = pause.PauseState()
        pause.install(state)
    return state


async def _broadcast(app: Any, payload: dict) -> None:
    bridge = getattr(app.state, "ws_bridge", None)
    if bridge is None:
        return
    try:
        await bridge.broadcast({"type": "paused_changed", "timestamp": time.time(), "payload": payload})
    except Exception:                                          # noqa: BLE001 - telling clients is best effort
        logger.warning("pause: paused_changed broadcast failed", exc_info=True)


def _pause_coding_run(record: Any) -> bool:
    from prometheus.coding import managed
    from prometheus.coding.control import control_path, read_state_for_write, with_paused, write_state

    coding_id = (getattr(record, "session_id", None) or "").removeprefix("coding:") or record.id
    path = control_path(managed.coding_control_dir(coding_id))
    state = read_state_for_write(path)
    if not state.paused:
        write_state(path, with_paused(state, True))
    return True


async def pause_everything(app: Any, *, by: str, reason: str) -> dict:
    """Pause (or confirm the pause), then stop everything that runs. Returns the answer's body."""
    info, _new = _state().pause(by=by, reason=reason)
    stopped = {"turns": 0, "desktop_tasks": 0, "tasks": 0, "coding_runs": 0, "approvals_withdrawn": 0}

    queue = getattr(app.state, "approval_queue", None)
    if queue is not None and hasattr(queue, "withdraw_all"):
        try:
            stopped["approvals_withdrawn"] = len(await queue.withdraw_all(why="paused"))
        except Exception:                                      # noqa: BLE001 - keep stopping the rest
            logger.warning("pause: withdrawing approvals failed", exc_info=True)

    bridge: Any = getattr(app.state, "ws_bridge", None)
    bridge_tasks = set()
    for session_id, task in list((getattr(bridge, "_turn_tasks", None) or {}).items()):
        bridge_tasks.add(task)
        try:
            if bridge.interrupt_turn(session_id):
                stopped["turns"] += 1
        except Exception:                                      # noqa: BLE001
            logger.warning("pause: interrupting %s failed", session_id, exc_info=True)
    me = asyncio.current_task()
    for task in pause.running_turn_tasks():
        if task is me or task in bridge_tasks:
            continue
        task.cancel()
        stopped["turns"] += 1

    runner = getattr(app.state, "computer_runner", None)
    if runner is not None:
        for task in list(getattr(runner, "running", lambda: [])()):
            try:
                if runner.stop(task.task_id):
                    stopped["desktop_tasks"] += 1
            except Exception:                                  # noqa: BLE001
                logger.warning("pause: stopping desktop task failed", exc_info=True)

    manager = getattr(app.state, "task_manager", None)
    if manager is None:
        try:
            from prometheus.tasks.manager import get_task_manager

            manager = get_task_manager()
        except Exception:                                      # noqa: BLE001
            manager = None
    if manager is not None:
        try:
            running = list(manager.list_tasks(status="running"))
        except Exception:                                      # noqa: BLE001
            logger.warning("pause: listing tasks failed", exc_info=True)
            running = []
        for record in running:
            try:
                if (getattr(record, "session_id", None) or "").startswith("coding:"):
                    if _pause_coding_run(record):
                        stopped["coding_runs"] += 1
                else:
                    await manager.stop_task(record.id)
                    stopped["tasks"] += 1
            except Exception:                                  # noqa: BLE001
                logger.warning("pause: stopping task %s failed", getattr(record, "id", "?"), exc_info=True)

    logger.warning("Prometheus PAUSED by %s%s; stopped %s", info.by, f" ({info.reason})" if info.reason else "",
                   stopped)
    await _broadcast(app, {"paused": True, **info.to_dict()})
    return {"paused": True, **info.to_dict(), "stopped": stopped}


def register_pause_routes(app: Any) -> None:
    @app.put("/api/pause")
    async def put_pause(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        reason = body.get("reason") if isinstance(body, dict) else None
        if reason is not None and not isinstance(reason, str):
            return JSONResponse(status_code=400, content={"error": "bad_request", "detail": "reason is a string"})
        return await pause_everything(app, by=_who(_identity(request)), reason=reason or "")

    @app.delete("/api/pause")
    async def delete_pause(request: Request):
        identity = _identity(request)
        if not is_person(identity):
            return JSONResponse(status_code=403, content={
                "error": "person_required",
                "detail": "only a person resumes Prometheus: an owner device, or the owner in Telegram (/resume). "
                          "The API token cannot, because a model with a shell can read it."})
        cleared = _state().resume(by=_who(identity))
        logger.warning("Prometheus RESUMED by %s", _who(identity))
        await _broadcast(app, {"paused": False})
        return {"paused": False, "was": cleared.to_dict() if cleared else None}
