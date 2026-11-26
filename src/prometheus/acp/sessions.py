"""The mapping between ACP sessions and Prometheus sessions.

An ACP sessionId is opaque to the client. Prometheus session ids are
``<surface>:<uuid>`` (engine/session.py); an ACP session maps to one of those
under the ``acp:`` surface, so its rows, titles and tombstones live in the same
LCM store as every other conversation and NOTHING internal changes shape.

The map is durable-by-construction: the ACP sessionId IS the Prometheus
session id (``acp:<uuid>``), so a daemon restart needs no mapping table — the
store already remembers the session. ``session/resume`` rehydrates through the
same ``SessionManager.rehydrate_if_cold`` path every surface uses.

What this module owns:
- new/list/resume/delete bookkeeping against the store (titles, tombstones);
- prompt idempotency: (sessionId, idempotencyKey) -> the message it inserted,
  durable in the store's ``message_client_ids`` side table (the same mechanism
  the REST send path uses — one dedup vocabulary, not two);
- replay: history pages from the store, ending with the schema's
  ``ReplayComplete`` marker, live updates held while a replay runs.

What it deliberately does NOT own: running turns, permissions, tool gating.
Those stay in the bridge and the gate; wiring an ACP prompt to a turn is later
work, blocked on the undecided approval rule. This core is the shared vocabulary
both directions will use.

Source: novel code for Prometheus, 2026-11-25.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from prometheus.acp import types
from prometheus.acp.framing import INVALID_PARAMS, FrameError

#: The surface prefix for ACP sessions. Same ``<surface>:<id>`` grammar the
#: gateway ids use, so list_sessions, tombstones and scoping all just work.
SURFACE = "acp"

#: Schema IdempotencyKey: opaque, 1..255 chars, session-scoped. The claim is
#: stored WITH the "acp-key:" prefix; the store's writer guard caps the stored
#: claim at CLIENT_ID_STORAGE_MAX = 263 (memory/lcm_conversation_store, D15 —
#: a plain TEXT column, the guard IS the migration), so the full 255-char
#: schema key round-trips: 255 + 8 = 263. A longer key is refused by name at
#: the edge — never truncated (truncation would merge distinct submissions),
#: and never accepted-then-silently-dropped by the store's own length guard.
IDEMPOTENCY_KEY_MAX = 255
_CLAIM_PREFIX = "acp-key:"


def new_session_id() -> str:
    return f"{SURFACE}:{uuid.uuid4()}"


def is_acp_session(session_id: str) -> bool:
    return session_id.startswith(f"{SURFACE}:")


class SessionNotFound(FrameError):
    """The sessionId names no session this agent knows (schema -32002)."""

    def __init__(self, session_id: str) -> None:
        super().__init__(-32002, f"session not found: {session_id}", None)
        self.session_id = session_id


@dataclass
class PromptInsertion:
    """What ``session/prompt`` answers with: the INSERTED user message's id.

    Per the schema, PromptResponse acknowledges insertion, not completion.
    Processing is reported through state updates; this record is the durable
    half of that contract.
    """

    session_id: str
    message_id: str
    idempotency_key: str | None = None


@dataclass
class SessionMap:
    """ACP session bookkeeping over the LCM conversation store.

    ``store`` is a ``memory.lcm_conversation_store.LCMConversationStore`` (or a
    test double with the same surface). No daemon globals are read here.
    """

    store: Any
    #: Sessions created or resumed through THIS connection, with their cwd.
    #: The store is the durable truth; this is the connection's working set.
    _known: dict[str, str] = field(default_factory=dict)

    # -- session/new ---------------------------------------------------------

    def new_session(self, params: dict[str, Any]) -> types.NewSessionResponse:
        try:
            req = types.NewSessionRequest.model_validate(params)
        except ValidationError as exc:
            raise FrameError(INVALID_PARAMS, f"invalid session/new params: {exc}") from exc
        session_id = new_session_id()
        self._known[session_id] = req.cwd
        # The cwd is durable too (session_workspaces), so session/list can
        # report a real cwd after a restart — SessionInfo requires it.
        self.store.set_session_workspace(session_id, req.cwd, set_by="acp")
        return types.NewSessionResponse(session_id=session_id)

    # -- session/list --------------------------------------------------------

    def list_sessions(self, params: dict[str, Any]) -> types.ListSessionsResponse:
        try:
            req = types.ListSessionsRequest.model_validate(params)
        except ValidationError as exc:
            raise FrameError(INVALID_PARAMS, f"invalid session/list params: {exc}") from exc
        infos: list[types.SessionInfo] = []
        for row in self.store.list_sessions():
            sid = row.get("session_id", "")
            if not is_acp_session(sid):
                continue
            cwd = self._known.get(sid) or self.store.get_session_workspace(sid) or req.cwd or ""
            infos.append(types.SessionInfo(
                session_id=sid,
                cwd=cwd,
                title=row.get("title") or None,
            ))
        return types.ListSessionsResponse(sessions=infos)

    # -- session/prompt (insertion + idempotency only; running the turn is later work) -

    def insert_prompt(self, params: dict[str, Any]) -> PromptInsertion:
        """Validate a prompt and INSERT its user message, idempotently.

        A repeat of (sessionId, idempotencyKey) returns the FIRST insertion's
        message id and inserts nothing — the schema's retry contract. The
        durable claim lives in ``message_client_ids`` under the key
        ``acp-key:<idempotencyKey>``. Lookup, insert and claim record run in
        ONE store transaction: two concurrent prompts with the same key
        cannot both pass the lookup, and a crash mid-unit rolls back the
        insert with it — a restart cannot double-insert.
        """
        try:
            req = types.PromptRequest.model_validate(params)
        except ValidationError as exc:
            raise FrameError(INVALID_PARAMS, f"invalid session/prompt params: {exc}") from exc
        session_id = req.session_id
        if not is_acp_session(session_id):
            raise SessionNotFound(session_id)
        key_obj = req.idempotency_key
        # The generated type is a RootModel[str]; the wire value is .root.
        key: str | None = key_obj.root if key_obj is not None else None
        if key is not None:
            if not (1 <= len(key) <= IDEMPOTENCY_KEY_MAX):
                raise FrameError(INVALID_PARAMS, "idempotencyKey must be 1..255 characters")
        text = "\n".join(b.text for b in req.prompt if getattr(b, "text", None))
        with self.store.transaction():
            if key is not None:
                stored = self.store.get_message_client_ids(session_id)
                for row_id, cid in stored.items():
                    if cid == f"{_CLAIM_PREFIX}{key}":
                        return PromptInsertion(session_id, str(row_id), key)
            row_id = self._insert_user_row(session_id, text)
            if key is not None:
                self.store.set_message_client_id(session_id, row_id, f"{_CLAIM_PREFIX}{key}")
        return PromptInsertion(session_id, str(row_id), key)

    def _insert_user_row(self, session_id: str, text: str) -> int:
        """One user row through the store's own insert (ids, redaction, FTS)."""
        from prometheus.memory.lcm_types import MessagePart

        part = MessagePart(
            session_id=session_id,
            role="user",
            content=text,
            timestamp=time.time(),
            provenance="user",
            is_trusted=True,
        )
        self.store.insert_message(part, append=True)
        return int(part.row_id)

    # -- session/resume (replay; live updates held by the caller) ------------

    def resume_session(self, params: dict[str, Any]) -> tuple[types.ResumeSessionResponse, list[dict[str, Any]]]:
        """Validate a resume and produce the replay payload.

        Returns the response and the ordered history updates (user/agent
        message updates keyed by durable rowid). The CALLER sends the updates,
        then the ``replay_complete`` marker, then the response — and holds live
        updates for the session until the marker is out (schema ReplayComplete).
        """
        try:
            req = types.ResumeSessionRequest.model_validate(params)
        except ValidationError as exc:
            raise FrameError(INVALID_PARAMS, f"invalid session/resume params: {exc}") from exc
        session_id = req.session_id
        if not is_acp_session(session_id) or not self.store.has_session_rows(session_id):
            raise SessionNotFound(session_id)
        self._known[session_id] = req.cwd
        self.store.set_session_workspace(session_id, req.cwd, set_by="acp")
        replay: list[dict[str, Any]] = []
        if req.replay_from is not None:
            # Only the "start" cursor exists in the pinned tag; anything else
            # was already refused by the generated union's open-variant guard
            # unless _-prefixed, and an unsupported cursor MUST be -32602.
            if getattr(req.replay_from, "type", None) != "start":
                raise FrameError(INVALID_PARAMS, "unsupported replayFrom cursor")
            # messages_after_id SELECTs rowid AS row_id — the durable wire id.
            # (get_all_messages deliberately returns row_id=0; never key on it.)
            parts = self.store.messages_after_id(0, session_id=session_id)
            for part in sorted(parts, key=lambda p: (p.turn_index, p.row_id)):
                replay.append(message_update(part))
        return types.ResumeSessionResponse(), replay

    # -- session/delete ------------------------------------------------------

    def delete_session(self, params: dict[str, Any]) -> types.DeleteSessionResponse:
        try:
            req = types.DeleteSessionRequest.model_validate(params)
        except ValidationError as exc:
            raise FrameError(INVALID_PARAMS, f"invalid session/delete params: {exc}") from exc
        session_id = req.session_id
        if not is_acp_session(session_id):
            raise SessionNotFound(session_id)
        self.store.tombstone_session(session_id)
        self._known.pop(session_id, None)
        return types.DeleteSessionResponse()


def message_update(part: Any) -> dict[str, Any]:
    """One stored row as an ACP session update (user_message / agent_message).

    Keyed by the DURABLE rowid — never a positional id — so a client's upserts
    reconcile with live updates and later replays alike (A1 rule 1/2).
    """
    role = getattr(part, "role", "user")
    kind = "user_message" if role == "user" else "agent_message"
    return {
        "sessionUpdate": kind,
        "messageId": str(part.row_id),
        "content": [{"type": "text", "text": getattr(part, "content", "")}],
    }


def replay_complete_update() -> dict[str, Any]:
    """The schema's end-of-replay marker, as a session/update payload.

    The wire shape is a SessionUpdate variant with the ``replay_complete``
    discriminator wrapping the (empty) ReplayComplete body.
    """
    return {"sessionUpdate": "replay_complete",
            **types.ReplayComplete().model_dump(by_alias=True, exclude_none=True)}