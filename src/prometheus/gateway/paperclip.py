"""Paperclip fleet-orchestration gateway — Prometheus as a hireable agent.

Paperclip (github.com/paperclipai/paperclip) manages fleets of AI agents:
org charts, budgets, audit trails, heartbeat-based work assignment. Its
generic ``http`` adapter is fire-and-forget — it POSTs a wake payload
(``{runId, agentId, companyId?, context: {taskId, wakeReason, commentId}}``)
to a webhook and ignores the response body, so EVERY observable outcome
must flow back through the Paperclip REST API.

This module is that heartbeat client. ``POST /api/paperclip/wake`` (mounted
in web/server.py, bearer-protected by the standard /api/ middleware) parses
the wake and spawns ``run_heartbeat``, which drives the documented protocol
(docs/guides/agent-developer/heartbeat-protocol.md, verified 2026-07-17):

    identity -> resolve issue -> checkout -> agent turn -> report -> cost event

The deterministic protocol steps live HERE in Python; only the actual work
step runs through the agent loop — via ``WebSocketBridge.run_turn_awaited``
so a heartbeat is an ordinary web session (``paperclip:issue:{id}``): it
persists to LCM, streams live to Beacon, and keeps context across wakes.

Auth out: a long-lived Paperclip agent API key (create with
``POST /api/agents/{agentId}/keys``), sourced from ``gateway.paperclip.api_key``
or ``PROMETHEUS_PAPERCLIP_API_KEY``. Empty key = Paperclip local trusted mode.

Trust model: issue titles/descriptions/comments are EXTERNAL input injected
into the agent prompt — same provenance class as an inbound Telegram message.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Final-line status marker the work prompt asks the agent to emit. Last match
# wins; absence degrades to in_progress (safe: never falsely closes an issue).
_STATUS_LINE = re.compile(
    r"^\s*STATUS:\s*(done|in_progress|blocked)\s*$", re.IGNORECASE | re.MULTILINE
)

# Checkout preconditions straight from the heartbeat protocol doc. in_progress
# is absent by design: an issue you already own checks out idempotently.
_CHECKOUT_EXPECTED_STATUSES = ["todo", "backlog", "blocked", "in_review"]


@dataclass
class WakeEvent:
    """Parsed Paperclip ``http``-adapter wake payload."""

    run_id: str
    agent_id: str
    company_id: str | None = None
    issue_id: str | None = None
    wake_reason: str | None = None
    comment_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _clean(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


class PaperclipGateway:
    """Runs Paperclip heartbeats against a configured Paperclip server."""

    def __init__(
        self,
        cfg: dict[str, Any],
        bridge: Any,
        daemon_config: dict[str, Any] | None = None,
    ) -> None:
        self.api_url = (_clean(cfg.get("api_url")) or "").rstrip("/")
        if not self.api_url:
            # FAIL LOUD at boot: an enabled-but-unconfigured gateway must not
            # silently no-op (config-dark trap).
            raise ValueError(
                "gateway.paperclip.enabled is true but api_url is empty — "
                "set gateway.paperclip.api_url to the Paperclip server URL"
            )
        self.api_key = _clean(cfg.get("api_key")) or os.environ.get(
            "PROMETHEUS_PAPERCLIP_API_KEY", ""
        )
        if not self.api_key:
            logger.warning(
                "Paperclip gateway: no api_key configured — assuming Paperclip "
                "local trusted mode (unauthenticated API)"
            )
        self.timeout = float(cfg.get("timeout_seconds", 30))
        self.comment_max_chars = int(cfg.get("comment_max_chars", 4000))
        self.session_prefix = _clean(cfg.get("session_prefix")) or "paperclip"
        self.bridge = bridge
        self.daemon_config = daemon_config or {}
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ #
    # Wake intake
    # ------------------------------------------------------------------ #

    def parse_wake(self, body: dict[str, Any]) -> WakeEvent:
        """Validate + normalize an ``http``-adapter POST body.

        Raises ``ValueError`` on a payload that cannot be a Paperclip wake
        (the route turns that into a 400).
        """
        run_id = _clean(body.get("runId"))
        agent_id = _clean(body.get("agentId"))
        if not run_id:
            raise ValueError("runId is required")
        if not agent_id:
            raise ValueError("agentId is required")
        context = body.get("context") if isinstance(body.get("context"), dict) else {}
        return WakeEvent(
            run_id=run_id,
            agent_id=agent_id,
            company_id=_clean(body.get("companyId")) or _clean(context.get("companyId")),
            issue_id=_clean(context.get("taskId")) or _clean(context.get("issueId")),
            wake_reason=_clean(context.get("wakeReason")),
            comment_id=_clean(context.get("commentId"))
            or _clean(context.get("wakeCommentId")),
            raw=body,
        )

    def start_heartbeat(self, wake: WakeEvent) -> asyncio.Task:
        """Spawn the heartbeat as a tracked background task.

        Fire-and-forget mirrors the ``http`` adapter's own semantics (it never
        reads our response), and keeps the wake webhook immune to Paperclip's
        ``timeoutSec``. The done-callback logs crashes so nothing dies silently.
        """
        task = asyncio.create_task(self.run_heartbeat(wake))
        self._tasks.add(task)
        task.add_done_callback(self._on_heartbeat_done)
        return task

    def _on_heartbeat_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Paperclip heartbeat task crashed: %s", exc, exc_info=exc)

    # ------------------------------------------------------------------ #
    # Heartbeat protocol
    # ------------------------------------------------------------------ #

    async def run_heartbeat(self, wake: WakeEvent) -> dict[str, Any]:
        """Run one full heartbeat. Returns a summary dict (also used by tests)."""
        headers = {}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        try:
            async with httpx.AsyncClient(
                base_url=self.api_url, timeout=self.timeout, headers=headers
            ) as client:
                return await self._run_heartbeat_inner(client, wake)
        except Exception as exc:
            logger.exception(
                "Paperclip heartbeat failed (run=%s agent=%s): %s",
                wake.run_id,
                wake.agent_id,
                exc,
            )
            return {"outcome": "error", "error": str(exc)}

    async def _run_heartbeat_inner(
        self, client: httpx.AsyncClient, wake: WakeEvent
    ) -> dict[str, Any]:
        run_header = {"X-Paperclip-Run-Id": wake.run_id}

        # Step 1 — identity (also fills company id when the wake omitted it).
        me = await self._get_json(client, "/api/agents/me")
        agent_id = _clean(me.get("id")) or wake.agent_id
        company_id = wake.company_id or _clean(me.get("companyId"))

        # Step 2 — resolve the issue to work: the triggering task, else inbox.
        issue_id = wake.issue_id
        if not issue_id:
            issue_id = await self._pick_from_inbox(client, company_id, agent_id, wake)
        if not issue_id:
            logger.info("Paperclip heartbeat %s: no assigned work — idle", wake.run_id)
            return {"outcome": "idle"}

        # Step 3 — atomic checkout. 409 = another agent owns it; NEVER retry.
        resp = await client.post(
            f"/api/issues/{issue_id}/checkout",
            json={"agentId": agent_id, "expectedStatuses": _CHECKOUT_EXPECTED_STATUSES},
            headers=run_header,
        )
        if resp.status_code == 409:
            logger.info(
                "Paperclip heartbeat %s: issue %s checked out elsewhere — skipping",
                wake.run_id,
                issue_id,
            )
            return {"outcome": "checkout_conflict", "issue_id": issue_id}
        resp.raise_for_status()

        # Step 4 — context.
        issue = await self._get_json(client, f"/api/issues/{issue_id}")
        comments = await self._get_json(client, f"/api/issues/{issue_id}/comments")
        prompt = self.build_prompt(issue_id, issue, comments, wake)

        # Step 5 — the work: one awaited agent turn in a durable per-issue
        # session (visible in Beacon, persisted to LCM across wakes).
        session_id = f"{self.session_prefix}:issue:{issue_id}"
        try:
            text, usage = await self.bridge.run_turn_awaited(session_id, prompt)
        except Exception as exc:
            logger.exception(
                "Paperclip heartbeat %s: agent turn failed on issue %s",
                wake.run_id,
                issue_id,
            )
            await self._comment_best_effort(
                client,
                issue_id,
                run_header,
                f"Prometheus heartbeat failed before completing work: {exc}",
            )
            return {"outcome": "agent_error", "issue_id": issue_id, "error": str(exc)}

        # Step 6 — report. in_progress = comment-only PATCH (checkout already
        # moved the issue to in_progress; the status field is for done/blocked).
        status, comment = self.parse_status(text)
        patch_body: dict[str, Any] = {"comment": comment[: self.comment_max_chars]}
        if status != "in_progress":
            patch_body["status"] = status
        patch = await client.patch(
            f"/api/issues/{issue_id}", json=patch_body, headers=run_header
        )
        patch.raise_for_status()

        # Step 7 — cost event (best-effort; budgets stay meaningful).
        tokens = await self._report_cost(client, company_id, agent_id, usage)

        logger.info(
            "Paperclip heartbeat %s: issue %s -> %s (%s tokens)",
            wake.run_id,
            issue_id,
            status,
            tokens or "untracked",
        )
        return {
            "outcome": "completed",
            "issue_id": issue_id,
            "status": status,
            "tokens": tokens,
        }

    # ------------------------------------------------------------------ #
    # Protocol helpers
    # ------------------------------------------------------------------ #

    async def _get_json(
        self, client: httpx.AsyncClient, path: str, params: dict[str, str] | None = None
    ) -> Any:
        resp = await client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    async def _pick_from_inbox(
        self,
        client: httpx.AsyncClient,
        company_id: str | None,
        agent_id: str,
        wake: WakeEvent,
    ) -> str | None:
        """Untargeted wake: pick from the assignment inbox per protocol order —
        in_progress first, in_review only when woken by a comment, then todo.
        Blocked issues are skipped in v1 (we can't unblock deterministically)."""
        if not company_id:
            logger.warning(
                "Paperclip heartbeat %s: no companyId available — cannot list inbox",
                wake.run_id,
            )
            return None
        data = await self._get_json(
            client,
            f"/api/companies/{company_id}/issues",
            params={
                "assigneeAgentId": agent_id,
                "status": "todo,in_progress,in_review,blocked",
            },
        )
        if isinstance(data, dict):
            issues = data.get("issues") or data.get("items") or []
        else:
            issues = data or []
        issues = [i for i in issues if isinstance(i, dict)]

        preference = ["in_progress", "in_review", "todo"] if wake.comment_id else [
            "in_progress",
            "todo",
        ]
        for wanted in preference:
            for issue in issues:  # server returns priority order — keep it
                if issue.get("status") == wanted and _clean(issue.get("id")):
                    return _clean(issue.get("id"))
        return None

    async def _comment_best_effort(
        self,
        client: httpx.AsyncClient,
        issue_id: str,
        run_header: dict[str, str],
        body: str,
    ) -> None:
        try:
            resp = await client.post(
                f"/api/issues/{issue_id}/comments",
                json={"body": body[: self.comment_max_chars]},
                headers=run_header,
            )
            resp.raise_for_status()
        except Exception:
            logger.warning(
                "Paperclip: failure comment on issue %s could not be posted",
                issue_id,
                exc_info=True,
            )

    async def _report_cost(
        self,
        client: httpx.AsyncClient,
        company_id: str | None,
        agent_id: str,
        usage: Any,
    ) -> int | None:
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        if not company_id or (input_tokens <= 0 and output_tokens <= 0):
            return None
        model_cfg = self.daemon_config.get("model") or {}
        try:
            resp = await client.post(
                f"/api/companies/{company_id}/cost-events",
                json={
                    "agentId": agent_id,
                    "provider": model_cfg.get("provider") or "prometheus",
                    "model": model_cfg.get("model") or "local",
                    "inputTokens": input_tokens,
                    "outputTokens": output_tokens,
                },
            )
            resp.raise_for_status()
        except Exception:
            logger.warning("Paperclip: cost event report failed", exc_info=True)
        return input_tokens + output_tokens

    # ------------------------------------------------------------------ #
    # Prompt + status parsing
    # ------------------------------------------------------------------ #

    def build_prompt(
        self,
        issue_id: str,
        issue: dict[str, Any],
        comments: Any,
        wake: WakeEvent,
    ) -> str:
        title = _clean(issue.get("title")) or "(untitled)"
        description = _clean(issue.get("description")) or _clean(issue.get("body")) or "(none)"
        status = _clean(issue.get("status")) or "unknown"
        priority = _clean(issue.get("priority")) or "unset"

        if isinstance(comments, dict):
            comment_items = comments.get("comments") or comments.get("items") or []
        else:
            comment_items = comments or []
        rendered: list[str] = []
        for entry in comment_items[-10:]:
            if not isinstance(entry, dict):
                continue
            body = _clean(entry.get("body")) or _clean(entry.get("text"))
            if not body:
                continue
            author = _clean(entry.get("authorName")) or _clean(entry.get("authorAgentId")) or "unknown"
            rendered.append(f"- {author}: {body[:1000]}")

        lines = [
            "You are working as a Paperclip-managed agent employee.",
            f"Wake reason: {wake.wake_reason or 'scheduled heartbeat'}.",
            "",
            f"Issue {issue_id}: {title}",
            f"Current status: {status} | Priority: {priority}",
            "",
            "Description:",
            description[:4000],
            "",
            "Recent comments (as provided by the tracker):",
            *(rendered or ["(none)"]),
            "",
            "Do this work now, using your tools as needed.",
            "Your entire reply will be posted to the Paperclip issue as your "
            "progress comment, so write it for the issue thread: what you did, "
            "what remains, and the concrete next action.",
            "End your reply with exactly one final line:",
            "STATUS: done — the work is complete",
            "STATUS: in_progress — you made progress but more remains",
            "STATUS: blocked — you cannot proceed (say why and who unblocks it)",
        ]
        return "\n".join(lines)

    @staticmethod
    def parse_status(text: str) -> tuple[str, str]:
        """Extract the trailing ``STATUS:`` marker; default in_progress.

        Returns ``(status, comment)`` where the comment is the reply with the
        marker line removed.
        """
        text = text or ""
        matches = list(_STATUS_LINE.finditer(text))
        if matches:
            last = matches[-1]
            status = last.group(1).lower()
            comment = (text[: last.start()] + text[last.end() :]).strip()
        else:
            status = "in_progress"
            comment = text.strip()
        if not comment:
            comment = f"(heartbeat produced no textual output; status={status})"
        return status, comment
