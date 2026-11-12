"""``PUT /api/pause`` and ``DELETE /api/pause``: pause everything, and resume (Beacon iOS "Piece 17").

The state is :mod:`prometheus.engine.pause`. These routes add who may do what, and the pause itself:

  * **Pausing only takes away**, so the operator may: the API token or an owner device (``route_access``:
    operator only; a scoped device gets ``403 operator_only``).
  * **Resuming gives everything back**, so only a person may, by the repo's one definition of a person
    (``computer/door.py PersonCheck``, the bar hold lifts use): a device a person MARKED (``/computer allow`` from
    Telegram, or a marked device; never the API token) and not held, or the owner in Telegram (``/resume``).
    NOT "any owner device": the API token sits in a file a model with a shell can read, and with it a model can
    mint itself an owner device from this machine. An agent that could lift its own emergency stop has not been
    stopped. ``403 person_required`` otherwise.

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
    kind = "owner device" if getattr(identity, "owner", False) else "scoped device"
    name = str(getattr(identity, "name", "") or getattr(identity, "id", ""))
    return f"{kind} {name}".strip()


def person_check(identity: Any, devices: Any) -> tuple[bool, str]:
    """``PersonCheck`` over the request's credential: a marked, unheld device; never the token or an open daemon."""
    from prometheus.computer.door import PersonCheck
    from prometheus.permissions.approver import from_device_identity

    try:
        return PersonCheck(device_store=devices).check(from_device_identity(identity))
    except Exception:                                          # noqa: BLE001 - any doubt is not a person
        logger.warning("pause: the person check failed", exc_info=True)
        return False, "the person check could not be completed"


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
    state = _state()
    info, _new = state.pause(by=by, reason=reason)
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

    persisted = state.persisted
    logger.warning("Prometheus PAUSED by %s%s; stopped %s%s", info.by, f" ({info.reason})" if info.reason else "",
                   stopped, "" if persisted else "; NOT written to disk, so it will not survive a restart")
    # Every client hears THAT it is paused; who paused and why are the operator's (GET /api/status).
    await _broadcast(app, {"paused": True, "since": info.since})
    return {"paused": True, **info.to_dict(), "persisted": persisted, "stopped": stopped}


def register_pause_routes(app: Any, *, devices: Any = None) -> None:
    """*devices*: a callable returning the daemon's device store (the one the marks live in)."""
    devices_of = devices if callable(devices) else (lambda: devices)

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
        ok, why = person_check(identity, devices_of())
        if not ok:
            return JSONResponse(status_code=403, content={
                "error": "person_required",
                "detail": "only a person resumes Prometheus: a device you marked (/computer allow <device id> from "
                          "Telegram), or the owner in Telegram (/resume). " + why})
        cleared = _state().resume(by=_who(identity))
        logger.warning("Prometheus RESUMED by %s", _who(identity))
        await _broadcast(app, {"paused": False, "since": None})
        return {"paused": False, "was": cleared.to_dict() if cleared else None}
