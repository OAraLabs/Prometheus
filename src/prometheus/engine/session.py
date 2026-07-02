"""Gateway-agnostic conversation session management.

Provides per-chat conversation history that any gateway adapter
(Telegram, Slack, Discord, etc.) can use to maintain multi-turn
context when dispatching to the agent loop.

SPRINT-2 WS1 — Durability & Steering. ChatSession now also holds two
per-session queues:

- ``queued_steers``: drained on every loop iteration as a *system-prompt
  addendum* for the upcoming model call. Lets a user mid-conversation
  inject "actually focus on X" without restarting the turn.
- ``queued_prompts``: drained at end of turn as the next user message.
  Lets a user line up follow-up tasks while the current one runs.

Pattern adapted from Hermes Agent (NousResearch/hermes-agent)
Original: run_agent.py — AIAgent.steer / _drain_pending_steer +
  agent/agent_runtime_helpers.py:apply_pending_steer_to_tool_results
License: MIT

Differences from Hermes:
- Hermes appends the steer text to the LAST tool result's content
  ("\\n\\nUser guidance: …"). Prometheus injects it as a system-prompt
  addendum on the upcoming model call so the steer is lexically distinct
  from tool output. Both preserve role alternation; the system-prompt
  approach keeps the conversation messages immutable and makes the steer
  recognisably "out of band" to the model.
- Hermes triggers steer via "busy_input_mode" config (any user text
  during a running turn becomes a steer). Prometheus uses an explicit
  ``/steer`` slash command — matches Telegram's discrete-command UX.
- Prometheus exposes ``/unqueue`` + ``/clear-steers`` as separate
  cancellation surfaces; Hermes drops pending steers implicitly via
  ``interrupt()``.

The two queues live on the existing ChatSession to avoid spinning up a
parallel "agent state" object — the gateway already looks up ChatSession
by session_id, so the queues ride along with the same handle.
"""

from __future__ import annotations

import logging
import time

from prometheus.engine.messages import ConversationMessage

log = logging.getLogger(__name__)

MAX_SESSION_MESSAGES = 50


class ChatSession:
    """Per-chat conversation state.

    Sprint 2: also carries ``queued_steers`` and ``queued_prompts`` —
    see the module docstring for the contract.

    Concurrency: Prometheus runs the gateway + agent loop in the same
    asyncio event loop. Append / pop / clear on the underlying lists is
    atomic with respect to ``await`` checkpoints — no explicit lock is
    required because there's no thread crossing. Hermes uses a
    ``threading.Lock`` because their gateway runs on a separate thread;
    that's a deliberate divergence (see module docstring).
    """

    __slots__ = (
        "session_id", "messages", "created_at",
        "queued_steers", "queued_prompts",
        "_lcm_engine",
    )

    def __init__(
        self,
        session_id: str,
        *,
        lcm_engine: object | None = None,
    ) -> None:
        self.session_id = session_id
        self.messages: list[ConversationMessage] = []
        self.created_at: float = time.time()
        # SPRINT-2 WS1 — see module docstring for semantics.
        self.queued_steers: list[str] = []
        self.queued_prompts: list[str] = []
        # PR fix/memory-lcm-full-rewire (2026-05-26) — LCM persistence
        # handle, set by SessionManager when the daemon has wired LCM.
        # ``None`` when the session was created before LCM was available
        # (e.g. tests, CLI without LCM) — persistence becomes a no-op.
        self._lcm_engine = lcm_engine

    # ------------------------------------------------------------------
    # SPRINT-2 WS1 — /steer and /queue plumbing
    # ------------------------------------------------------------------

    def enqueue_steer(self, text: str) -> bool:
        """Append ``text`` to the steer queue. Returns False if empty."""
        cleaned = (text or "").strip()
        if not cleaned:
            return False
        self.queued_steers.append(cleaned)
        return True

    def drain_steers(self) -> str | None:
        """Return all pending steers concatenated with blank lines, then
        clear the queue. Returns ``None`` when nothing is pending.

        Hermes parity: multiple steers concatenate before drain — the
        loop sees them as one combined addendum on the next model call.
        """
        if not self.queued_steers:
            return None
        text = "\n\n".join(self.queued_steers)
        self.queued_steers.clear()
        return text

    def clear_steers(self) -> int:
        """Drop all pending steers without surfacing them. Returns the
        number dropped — useful for ``/clear-steers`` confirmation."""
        n = len(self.queued_steers)
        self.queued_steers.clear()
        return n

    def enqueue_prompt(self, text: str) -> bool:
        """Append ``text`` to the queued-prompts list. Returns False
        if empty."""
        cleaned = (text or "").strip()
        if not cleaned:
            return False
        self.queued_prompts.append(cleaned)
        return True

    def drain_prompt(self) -> str | None:
        """Pop and return the next queued prompt (FIFO), or ``None`` if
        the queue is empty. Used by the gateway after a turn ends to
        dispatch the next queued user message."""
        if not self.queued_prompts:
            return None
        return self.queued_prompts.pop(0)

    def clear_prompts(self) -> int:
        """Drop all queued prompts. Returns the number dropped."""
        n = len(self.queued_prompts)
        self.queued_prompts.clear()
        return n

    def add_user_message(
        self,
        text: str,
        *,
        provenance: str = "user",
        is_trusted: bool = True,
    ) -> int:
        """Append a user-role message to the conversation. Returns its ``turn_index``.

        The returned turn_index is the durable per-session ordinal the message
        is persisted under — callers use it as the ``msg-{turn_index}`` wire id
        (e.g. the WS user-echo correlates a client_msg_id to it).

        Managed-tasks sprint: ``provenance`` + ``is_trusted`` let the shared
        ``inject_turn`` primitive record a non-user, untrusted turn (e.g. a task
        result) on the same path. Defaults keep the human-user behavior
        (``provenance="user"``, ``is_trusted=True``) for all existing callers.

        PR fix/memory-lcm-full-rewire (2026-05-26): also persists to
        LCM (best-effort) when an engine is wired. Without this hook,
        the user-message half of every turn would be invisible to
        LCM/MemoryExtractor — only the loop-appended tail would land
        in the durable store.
        """
        # turn_index = position the message will occupy in self.messages
        # AFTER the append (matches what add_result_messages will use
        # for downstream turns).
        new_turn_index = len(self.messages)
        if provenance == "user" and is_trusted:
            message = ConversationMessage.from_user_text(text)
        else:
            message = ConversationMessage.from_injected(
                text, provenance=provenance, is_trusted=is_trusted
            )
        self.messages.append(message)
        if self._lcm_engine is not None:
            self._persist_to_lcm(
                [self.messages[-1]],
                base_turn_index=new_turn_index,
            )
        return new_turn_index

    def add_result_messages(
        self,
        result_messages: list[ConversationMessage],
        original_len: int,
    ) -> None:
        """Append new messages produced by the agent loop.

        *result_messages* is ``RunResult.messages`` — the full messages list
        after the agent turn (which includes the user message we already
        added plus any assistant / tool-call / tool-result messages the loop
        appended).  *original_len* is the index into *result_messages* at
        which the new content starts (i.e. ``len(session.messages) - 1``
        before the call, since the user message was already appended).

        PR fix/memory-lcm-full-rewire (2026-05-26): after the in-memory
        append, persist the new messages to LCM (when wired). LCM is the
        durable conversation store that MemoryExtractor and future LCM
        compaction read from. Persistence is best-effort and never
        raises into the agent's path — failures are surfaced via
        ``telemetry.record_silent_failure``.
        """
        new = result_messages[original_len:]
        if new:
            self.messages.extend(new)
            if self._lcm_engine is not None:
                self._persist_to_lcm(new, base_turn_index=original_len)

    def _schedule_lcm_compaction(self) -> None:
        """Fire-and-forget LCM ``maybe_compact`` after an ingest batch.

        Runs as a background task so the summarizer LLM call never sits in the
        turn's critical path. No-op when there is no running event loop (the
        CLI path awaits ``maybe_compact`` itself). Failures are LOUD:
        journal error + silent_failure telemetry — a session crossing the
        threshold without compaction firing must never be silent again.
        """
        if self._lcm_engine is None:
            return
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # sync/CLI context — __main__ owns its own maybe_compact call
        loop.create_task(self._run_lcm_compaction())

    async def _run_lcm_compaction(self) -> None:
        try:
            result = await self._lcm_engine.maybe_compact(self.session_id)
            if result is not None:
                log.info(
                    "LCM compaction ran for %s: %s", self.session_id, result
                )
        except Exception as exc:
            log.error(
                "LCM compaction FAILED for %s: %s", self.session_id, exc,
                exc_info=True,
            )
            try:
                from prometheus.telemetry.tracker import get_telemetry_handle

                tel = get_telemetry_handle()
                if tel is not None:
                    tel.record_silent_failure(
                        subsystem="lcm_compaction",
                        operation="maybe_compact",
                        exc=exc,
                        context={"session_id": self.session_id},
                    )
            except Exception:
                log.warning("telemetry unavailable for lcm_compaction failure")

    def persist_loop_result(self, original_len: int) -> None:
        """Persist messages that ``run_loop`` appended IN PLACE to LCM.

        The streaming WS path (``web/ws_server.py:_run_agent``) passes
        ``session.get_messages()`` straight into ``run_loop``, which appends the
        assistant + tool-result messages directly onto ``self.messages``. So
        unlike :meth:`add_result_messages` (which extends from a *separate*
        RunResult list), the rows are ALREADY in ``self.messages`` — we persist
        the new tail WITHOUT re-appending. Without this, web/Beacon assistant
        turns stream but never reach LCM (the gateway adapters call
        ``add_result_messages``; the WS bridge had no equivalent).

        ``original_len`` is ``len(self.messages)`` captured before the loop ran.
        Best-effort, same contract as :meth:`_persist_to_lcm` — never raises.
        """
        new = self.messages[original_len:]
        if new and self._lcm_engine is not None:
            self._persist_to_lcm(new, base_turn_index=original_len)

    def _persist_to_lcm(
        self,
        new_messages: list[ConversationMessage],
        *,
        base_turn_index: int,
    ) -> None:
        """Persist new messages to LCM. Best-effort — never raises.

        ``turn_index`` is set to ``base_turn_index + i`` so it matches
        the position in ``self.messages`` after the extend.
        """
        try:
            for i, msg in enumerate(new_messages):
                self._lcm_engine.ingest_sync(  # type: ignore[union-attr]
                    session_id=self.session_id,
                    role=msg.role,
                    content=msg.text,
                    content_json=msg.content_json,
                    turn_index=base_turn_index + i,
                    # Persist the turn's trust tag so an injected (untrusted)
                    # task result survives the LCM round-trip rather than being
                    # silently dropped to the trusted default.
                    provenance=msg.provenance,
                    is_trusted=msg.is_trusted,
                )
            # Sprint 2 (OAra): the durable-DAG relief valve. maybe_compact was
            # CLI-only for the daemon's whole life — every gateway ingested
            # messages forever and nothing ever summarized. This is the one
            # choke point all paths share (telegram/slack via
            # add_result_messages, web/Beacon/Bridge via persist_loop_result).
            self._schedule_lcm_compaction()
        except Exception as exc:
            # Memory persistence MUST NOT be in the agent's critical
            # path. Surface to silent_failures and continue. The
            # nested try around the telemetry call covers the rare
            # case where telemetry itself is unavailable — we log a
            # warning rather than a bare pass so the primary error
            # still leaves a trace.
            try:
                from prometheus.telemetry.tracker import get_telemetry_handle
                tel = get_telemetry_handle()
                if tel is not None:
                    tel.record_silent_failure(
                        subsystem="chat_session",
                        operation="persist_to_lcm",
                        exc=exc,
                        context={
                            "session_id": self.session_id,
                            "new_msgs": len(new_messages),
                            "base_turn_index": base_turn_index,
                        },
                    )
            except Exception as nested_exc:
                log.warning(
                    "ChatSession: record_silent_failure ALSO failed "
                    "(%r); primary error was: %r",
                    nested_exc, exc,
                )
            log.warning(
                "ChatSession: LCM persist failed for session=%s "
                "(%d new messages) — agent loop unaffected",
                self.session_id, len(new_messages),
                exc_info=True,
            )

    def rollback_last(self) -> None:
        """Remove the most recently appended message (error recovery)."""
        if self.messages:
            self.messages.pop()

    def get_messages(self) -> list[ConversationMessage]:
        """Return the conversation history."""
        return self.messages

    def last_persisted_row_id(self) -> int:
        """Durable LCM rowid of this session's most-recently persisted message, or 0
        when no LCM engine is wired.

        Used as the canonical ``message_id`` on the WS user echo — the SAME id
        ``GET /api/sessions/{id}/messages`` reports for that row. Correct in the daemon's
        single-event-loop model: persistence is synchronous, so right after
        ``add_user_message`` the session's max rowid IS the just-inserted message.
        """
        if self._lcm_engine is None:
            return 0
        try:
            return self._lcm_engine.conversation_store.max_rowid(self.session_id)
        except Exception:
            return 0

    def clear(self) -> None:
        """Reset conversation history."""
        self.messages = []

    def trim(self, max_messages: int = MAX_SESSION_MESSAGES) -> None:
        """Truncate from the front if history exceeds *max_messages*."""
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]


class SessionManager:
    """Shared session store passed to all gateway adapters."""

    MAX_SESSION_MESSAGES = MAX_SESSION_MESSAGES

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}
        # PR fix/memory-lcm-full-rewire — set by the daemon after LCM
        # init. Sessions created after this is wired get LCM persistence;
        # sessions created before (none, in practice — daemon order
        # guarantees this) silently no-op. Public attribute (not a setter
        # method) keeps the wire site terse: ``session_manager.lcm_engine
        # = lcm_engine``.
        self.lcm_engine: object | None = None

    def get_or_create(self, session_id: str) -> ChatSession:
        """Return the existing session or create a new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatSession(
                session_id, lcm_engine=self.lcm_engine
            )
        return self._sessions[session_id]

    def get(self, session_id: str) -> "ChatSession | None":
        """Return the existing session for ``session_id``, or None.

        Non-creating lookup — used by session-mutating commands (/steer,
        /unqueue, /clearsteers) that target an in-flight run and no-op when
        there's no active session, unlike get_or_create.
        """
        return self._sessions.get(session_id)

    def clear(self, session_id: str) -> None:
        """Clear conversation history for a session (keeps the object)."""
        if session_id in self._sessions:
            self._sessions[session_id].clear()

    def remove(self, session_id: str) -> None:
        """Forget a session: drop its in-memory entry from ``self._sessions``.

        This is what makes a junk/test session disappear for good: ``GET
        /api/sessions`` enumerates ``self._sessions``, so once the entry is
        popped the session stops being listed (``clear`` only empties the
        message list, leaving the entry — and thus the listing — in place
        until a daemon restart).

        Deliberately does NOT touch the durable LCM conversation store: that
        store is append-only and is the system of record for history, so we
        only forget the live in-memory handle, never the persisted rows.
        No-op (safe) when the id is unknown.
        """
        self._sessions.pop(session_id, None)
