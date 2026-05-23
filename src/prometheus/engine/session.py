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

import time

from prometheus.engine.messages import ConversationMessage

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
    )

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.messages: list[ConversationMessage] = []
        self.created_at: float = time.time()
        # SPRINT-2 WS1 — see module docstring for semantics.
        self.queued_steers: list[str] = []
        self.queued_prompts: list[str] = []

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

    def add_user_message(self, text: str) -> None:
        """Append a user message to the conversation."""
        self.messages.append(ConversationMessage.from_user_text(text))

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
        """
        new = result_messages[original_len:]
        if new:
            self.messages.extend(new)

    def rollback_last(self) -> None:
        """Remove the most recently appended message (error recovery)."""
        if self.messages:
            self.messages.pop()

    def get_messages(self) -> list[ConversationMessage]:
        """Return the conversation history."""
        return self.messages

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

    def get_or_create(self, session_id: str) -> ChatSession:
        """Return the existing session or create a new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatSession(session_id)
        return self._sessions[session_id]

    def clear(self, session_id: str) -> None:
        """Clear conversation history for a session (keeps the object)."""
        if session_id in self._sessions:
            self._sessions[session_id].clear()

    def remove(self, session_id: str) -> None:
        """Delete a session entirely."""
        self._sessions.pop(session_id, None)
