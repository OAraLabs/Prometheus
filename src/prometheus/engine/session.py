"""Gateway-agnostic conversation session management.

Provides per-chat conversation history that any gateway adapter
(Telegram, Slack, Discord, etc.) can use to maintain multi-turn
context when dispatching to the agent loop.
"""

from __future__ import annotations

import time

from prometheus.engine.messages import ConversationMessage

MAX_SESSION_MESSAGES = 50


class ChatSession:
    """Per-chat conversation state."""

    __slots__ = ("session_id", "messages", "created_at")

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.messages: list[ConversationMessage] = []
        self.created_at: float = time.time()

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
