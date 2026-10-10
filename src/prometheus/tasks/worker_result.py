"""The one line a worker ends with: what happened, machine-readable.

A worker (``python -m prometheus --worker``) prints, as its LAST line::

    PROMETHEUS_WORKER_RESULT {"status": "ok", "reason": null, "route": "local", ...}

``status`` is ``ok`` (the turn finished) or ``failed`` (it did not). ``reason``
names why not: a stop the loop chose (``context_preflight_refusal``,
``max_iterations_hit``, ...) or the provider error's kind from
:func:`prometheus.api.turn_errors.classify_turn_error`. The other fields are
public facts about the route — never message content, never a credential —
because the manager copies the line onto the task record and the completion
signal, where the parent and the operator read it.

Source: novel code for Prometheus, 2026-10-10.
"""

from __future__ import annotations

import json
from typing import Any

RESULT_PREFIX = "PROMETHEUS_WORKER_RESULT "

#: Fields a result may carry; anything else is dropped when it is parsed.
_FIELDS = ("status", "reason", "route", "model", "provider", "window", "tokens_needed")


def format_result(result: dict[str, Any]) -> str:
    return RESULT_PREFIX + json.dumps({k: result.get(k) for k in _FIELDS})


def parse_result(output: str) -> dict[str, Any] | None:
    """The LAST result line in *output*, or None when there is none."""
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                data = json.loads(line[len(RESULT_PREFIX):])
            except ValueError:
                return None
            if not isinstance(data, dict) or data.get("status") not in ("ok", "failed"):
                return None
            return {k: data.get(k) for k in _FIELDS}
    return None
