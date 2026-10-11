"""A spawned worker asks its PARENT conversation for approval (PR-6 of #704).

A worker (``python -m prometheus --worker``) has no approval channel of its
own: under ``permission_mode: default`` anything that needs confirmation was
refused outright. The task manager gives each worker:

* in its environment (not secret): ``PROMETHEUS_WORKER_APPROVAL_URL`` (the
  daemon's ``/api/workers/approvals``) and ``PROMETHEUS_WORKER_TASK_ID``;
* on stdin, as the first line ahead of the brief: its per-worker TOKEN
  (tasks/worker_auth.py says why it is not in the environment).

:func:`approval_prompt_from_env` turns them into the loop's
``permission_prompt``: each request is POSTed to the daemon, which files it in
its approval queue under the conversation that spawned the worker and waits
for a person. The answer is True only when the daemon says ``approved``.
Anything else — a denial, an expiry, a refusal, a daemon that cannot be
reached, a reply that cannot be read, no token — is a denial. Nothing here
approves anything.

Source: novel code for Prometheus, 2026-10-10.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from typing import Any

log = logging.getLogger(__name__)

#: How long a worker waits for its answer: the daemon's own approval window
#: (30 minutes, permissions/approval_queue.py) plus headroom for the reply.
WAIT_SECONDS = 1800 + 120


_TOKEN: list[str | None] = [None]


def set_worker_token(token: str | None) -> None:
    """The token this worker asks with (read from its stdin by cli/worker.py)."""
    _TOKEN[0] = token or None


def split_token_line(stdin_text: str) -> tuple[str | None, str]:
    """(token, brief): the manager's first stdin line, then the brief itself."""
    from prometheus.tasks.worker_auth import TOKEN_LINE_PREFIX

    if stdin_text.startswith(TOKEN_LINE_PREFIX):
        first, _, rest = stdin_text.partition("\n")
        return first[len(TOKEN_LINE_PREFIX):].strip() or None, rest
    return None, stdin_text


def approval_prompt_from_env(
    env: MutableMapping[str, str] | Mapping[str, str],
) -> Callable[..., Awaitable[bool]] | None:
    """The worker's permission prompt, or None when it has nobody to ask.

    A token found in *env* is taken and REMOVED from it, so no child process
    inherits it. (The manager does not put one there; see worker_auth.py.)
    """
    from prometheus.tasks.worker_auth import HEADER_TASK, HEADER_TOKEN

    if isinstance(env, MutableMapping) and env.get("PROMETHEUS_WORKER_TOKEN"):
        set_worker_token(env.pop("PROMETHEUS_WORKER_TOKEN"))
    url = env.get("PROMETHEUS_WORKER_APPROVAL_URL")
    task_id = env.get("PROMETHEUS_WORKER_TASK_ID")
    if not (url and task_id):
        return None

    async def ask(tool_name: str, reason: str, arguments: dict[str, Any] | None = None) -> bool:
        import httpx

        from prometheus.permissions.argument_view import redact_arguments

        token = _TOKEN[0]
        if not token:
            log.warning("worker approval for %s: no token to ask with — denied", tool_name)
            return False
        # Redacted and truncated here, as the queue would: the card shows no
        # more than this, and a whole file body would not fit the daemon's cap.
        body = {"tool_name": tool_name, "reason": reason,
                "arguments": redact_arguments(arguments)}
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, read=WAIT_SECONDS),
            ) as client:
                resp = await client.post(
                    url, json=body, headers={HEADER_TASK: task_id, HEADER_TOKEN: token})
            answer = resp.json().get("result") if resp.status_code == 200 else None
        except Exception as exc:  # noqa: BLE001 — no answer is a denial
            log.warning("worker approval for %s could not be asked: %s", tool_name, exc)
            return False
        if answer != "approved":
            log.info("worker approval for %s: %s", tool_name, answer or resp.status_code)
        return answer == "approved"

    return ask
