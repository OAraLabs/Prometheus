"""Whether anything can answer an approval prompt here, in words an operator can act on.

``security.approval_queue.enabled`` ships ``false``, and the daemon builds the queue only when the Telegram gateway
is running. With no queue, ``SecurityGate.request_approval`` returns False at once, so every call that needs an
approval (a third-party MCP tool, a write or a shell command in ``strict`` or ``cautious`` mode, a network or install
command in a background turn) is refused within seconds as "Permission denied" and nothing is ever listed to approve.
``cautious`` reads as "asks the owner" and behaves as "refuses everything". ``oara doctor`` and the daemon's startup
warning both read this, so they cannot disagree about it (the shape of ``gateway/owners.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The one mode that never reaches an approval tier. Every other mode can.
AUTONOMOUS = "autonomous"


@dataclass(frozen=True)
class ApprovalGap:
    message: str   # what is true now, plainly
    fix: str       # what to do, naming the keys


def approval_channel_gap(config: dict[str, Any] | None, *, queue_built: bool | None = None) -> ApprovalGap | None:
    """Why no approval prompt could be answered, or None when something can.

    *queue_built* is the daemon's own knowledge at boot (was the queue actually constructed?). ``oara doctor``
    has only the config, so it passes None and the answer is what the daemon WOULD build from it.
    """
    from prometheus.config.shipped_defaults import resolve_telegram_enabled

    security = (config or {}).get("security") or {}
    mode = str(security.get("permission_mode", "default")).strip().lower()
    if mode == AUTONOMOUS:
        return None  # asks nothing, so there is nothing to answer
    queue_cfg = security.get("approval_queue") or {}
    enabled = bool(queue_cfg.get("enabled", False)) if isinstance(queue_cfg, dict) else False
    telegram = resolve_telegram_enabled((config or {}).get("gateway") or {})

    if queue_built is True or (queue_built is None and enabled and telegram):
        return None

    if not enabled:
        why = "security.approval_queue.enabled is false, which is how Prometheus ships"
    elif not telegram:
        why = ("security.approval_queue.enabled is true, but the queue is only built together with the Telegram "
               "gateway, and gateway.telegram_enabled is off")
    else:
        why = "security.approval_queue.enabled is true, but the Telegram gateway did not start, so the queue was not built"
    return ApprovalGap(
        message=(f"nothing here can answer an approval prompt ({why}). Every call that needs an approval is "
                 f"refused within seconds as \"Permission denied\" and none is ever listed to approve: third-party "
                 f"MCP tools, writes and shell commands in cautious or strict mode ({mode!r} here), and network or "
                 f"install commands in background turns"),
        fix=("Set security.approval_queue.enabled: true and run the Telegram gateway with an owner named "
             "(gateway.owner_user_ids), so you can answer with /approve; or set security.permission_mode: "
             "autonomous if you accept that nothing asks."),
    )
