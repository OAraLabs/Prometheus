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

import asyncio
import logging
import time

from prometheus.config.ephemeral import is_session_ephemeral
from prometheus.engine.messages import ConversationMessage

log = logging.getLogger(__name__)

MAX_SESSION_MESSAGES = 50

# feat/session-rehydrate: the cold-start restore window. Rows first (one
# newest-anchored page), then an estimated-token cap applied newest-first —
# the compactor's relief valve is single-pass and off by default, so the
# restored set must arrive already inside a sane budget rather than rely on
# being rescued.
_REHYDRATE_WINDOW = 40
_REHYDRATE_TOKEN_BUDGET = 8_000


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
        "_lcm_engine", "_compaction_tasks",
        "_lcm_persisted_len", "_lcm_persisted_ahead",
        "_turn_index_offset",
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
        # Strong refs for fire-and-forget compaction tasks (GC guard).
        self._compaction_tasks: set = set()
        # PR fix/memory-lcm-full-rewire (2026-05-26) — LCM persistence
        # handle, set by SessionManager when the daemon has wired LCM.
        # ``None`` when the session was created before LCM was available
        # (e.g. tests, CLI without LCM) — persistence becomes a no-op.
        self._lcm_engine = lcm_engine
        # Exact-once persistence watermark (fix: duplicate LCM rows from
        # overlapping tail persists — see the 2026-08-11 mapping survey).
        # Every index < _lcm_persisted_len is SETTLED: durably written, or
        # deliberately skipped (ephemeral interval). _lcm_persisted_ahead
        # holds indices >= the watermark that were written out of band —
        # a user row landing (and persisting) while a turn is still
        # appending its unpersisted tail below it.
        self._lcm_persisted_len: int = 0
        self._lcm_persisted_ahead: set[int] = set()
        # feat/session-rehydrate: added to each row's list index when
        # stamping turn_index at persist time. Zero for a cold session
        # (unchanged contract); set by restore() so rows written AFTER a
        # rehydrate continue the durable numbering instead of colliding
        # with the historical rows the restored tail came from — the
        # ORDER BY turn_index readers (LCM compactor/assembler) would
        # otherwise interleave new rows into old history.
        self._turn_index_offset: int = 0

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
        # seal=False: this writes ONE row that may sit ABOVE a running
        # turn's still-unpersisted tail (Beacon sends mid-turn; the echo
        # needs the rowid immediately). Sealing here would mark that tail
        # settled and the turn's own persist would then skip it — loss.
        self._persist_to_lcm(new_turn_index, seal=False)
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
        appended). *original_len* is the index into *result_messages* at
        which the new content starts: ``len(session.get_messages())``
        captured AFTER ``add_user_message`` appended the user turn — the
        value every gateway adapter captures as ``pre_len``. (An earlier
        revision of this docstring said ``len(session.messages) - 1``; that
        formula is off by one and, implemented literally on the ``/api/chat``
        route, re-appended and re-persisted the user row every turn.)

        PR fix/memory-lcm-full-rewire (2026-05-26): after the in-memory
        append, persist the new messages to LCM (when wired). LCM is the
        durable conversation store that MemoryExtractor and future LCM
        compaction read from. Persistence is best-effort and never
        raises into the agent's path — failures are surfaced via
        ``telemetry.record_silent_failure``.
        """
        pre_len = len(self.messages)
        if pre_len != original_len:
            # Caller drift: the index does not match the session length, so
            # the caller's idea of "where the new content starts" is stale
            # (the /api/chat off-by-one duplicated the user row this way).
            # The identity skip below corrects the low case; say so, loudly.
            log.warning(
                "add_result_messages: original_len=%d but session has "
                "%d messages (session=%s) — caller passed a stale index",
                original_len, pre_len, self.session_id,
            )
        start = original_len
        # Identity prefix-skip: run_async hands the loop a SHALLOW copy of
        # the session list, so any entry of result_messages that is the very
        # object already sitting at the same position of self.messages is
        # prefix, not new content — regardless of what the caller claimed
        # via original_len. A stale/low index therefore re-appends and
        # re-persists nothing.
        while (
            start < len(result_messages)
            and start < pre_len
            and result_messages[start] is self.messages[start]
        ):
            start += 1
        new = result_messages[start:]
        if new:
            self.messages.extend(new)
            self._persist_to_lcm(pre_len, seal=True)

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
        # Hold a strong reference: a bare create_task can be garbage-collected
        # mid-await and die SILENTLY — the exact failure mode this hook exists
        # to eliminate.
        task = loop.create_task(self._run_lcm_compaction())
        self._compaction_tasks.add(task)
        task.add_done_callback(self._compaction_tasks.discard)

    async def _run_lcm_compaction(self) -> None:
        try:
            log.info("LCM maybe_compact check for %s", self.session_id)
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

    def persist_loop_result(self, original_len: int) -> int | None:
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

        Exact-once (2026-08-11 duplicate-rows fix): the tail
        ``self.messages[original_len:]`` may contain rows that are ALREADY
        durable — a user message that landed (and persisted) mid-turn, or,
        when turns overlap on one session, rows an earlier-finishing turn's
        tail persist covered. The watermark inside :meth:`_persist_to_lcm`
        skips those, so calling this with a conservative (low) or stale
        ``original_len`` re-writes nothing. This is what stops the N-way
        row fan the 2026-08-11 survey found in the ``desktop:s4-*``
        sessions (rowids 461/466/481/496 — one copy per in-flight turn).

        Returns the durable rowid of the assistant turn just persisted (or
        None) — the wire cursor id the streaming path forwards on ``chat_done``
        (GRAFT-MOBILE-BRIDGE 3b). Callers that only persist ignore it.
        """
        return self._persist_to_lcm(original_len, seal=True)

    def restore(
        self,
        messages: list["ConversationMessage"],
        *,
        next_turn_index: int,
    ) -> None:
        """Seed a COLD session with already-durable history (rehydrate).

        Cold-start only, and the refusals are the contract: a session that
        has messages, or has persisted anything, must not be restored over —
        that is how double-rehydrate and history clobbering become
        structurally impossible rather than merely unlikely.

        The three assignments after the guard are the entire double-ingest
        defence (2026-08-11 mapping survey's shape): the watermark is seeded
        to the restored length so every restored row reads as SETTLED — the
        next turn's persists write only genuinely new rows — and the
        turn-index offset makes those new rows continue the durable
        numbering from ``next_turn_index`` instead of colliding with the
        history the tail was loaded from.

        Deliberately does NOT persist (the rows are already durable) and
        does NOT schedule compaction (nothing new was written).
        """
        if self.messages or self._lcm_persisted_len:
            raise RuntimeError(
                f"restore() on a non-cold session {self.session_id!r} "
                f"({len(self.messages)} messages, watermark "
                f"{self._lcm_persisted_len}) — refusing to clobber"
            )
        self.messages = list(messages)
        self._lcm_persisted_len = len(self.messages)
        self._lcm_persisted_ahead = set()
        self._turn_index_offset = next_turn_index - len(self.messages)

    def _note_persisted(self, idx: int) -> None:
        """Record that ``self.messages[idx]`` is durably written.

        Contiguous writes advance the watermark (draining any ahead-marks
        the advance reaches); a write above the watermark — over a still-
        unpersisted in-flight tail — is remembered in the ahead-set so the
        eventual tail persist skips it instead of re-writing it.
        """
        if idx == self._lcm_persisted_len:
            self._lcm_persisted_len += 1
            while self._lcm_persisted_len in self._lcm_persisted_ahead:
                self._lcm_persisted_ahead.discard(self._lcm_persisted_len)
                self._lcm_persisted_len += 1
        elif idx > self._lcm_persisted_len:
            self._lcm_persisted_ahead.add(idx)

    def _persist_to_lcm(self, start: int, *, seal: bool) -> int | None:
        """Persist the not-yet-persisted rows of ``self.messages[start:]``
        to LCM. Best-effort — never raises. No-op when no engine is wired.

        Returns the durable rowid of the LAST assistant-role row written in
        this span, or None when the span wrote no assistant row (or the engine
        exposes no rowid — an older engine or a test fake). This is the wire
        cursor id the streaming path puts on ``chat_done`` so a client learns
        the assistant turn's durable id without a REST re-read
        (GRAFT-MOBILE-BRIDGE 3b). It never affects persistence itself.

        ``turn_index`` for each row is its index in ``self.messages`` —
        unchanged from the original contract, but now computed per row so a
        span with skips (an already-durable user row in the middle of a
        turn's tail) still stamps every row with its true position.

        Exact-once: rows below the watermark, and rows in the ahead-set,
        are skipped — persisting an overlapping span is a safe no-op. This
        is the choke point that makes EVERY caller idempotent, so a stale
        ``original_len`` (or two turns racing on one session) can inflate
        nothing. On a mid-span ingest failure the rows already written stay
        marked, the rest stay pending, and a later overlapping persist
        resumes where this one failed (previously the whole remainder was
        silently lost).

        ``seal=True`` (turn-tail persists) afterwards marks everything below
        ``len(self.messages)`` settled — including rows deliberately never
        written, e.g. an ephemeral interval — so the watermark cannot wedge
        below a permanent hole. ``seal=False`` (single user-row persists)
        must NOT do that: the row may sit above a running turn's unpersisted
        tail, and sealing would silently drop that tail from the store.
        """
        if self._lcm_engine is None:
            return None
        wrote_any = False
        last_assistant_row_id: int | None = None
        try:
            end = len(self.messages)
            pending = [
                i
                for i in range(max(start, self._lcm_persisted_len), end)
                if i not in self._lcm_persisted_ahead
            ]
            for i in pending:
                msg = self.messages[i]
                self._lcm_engine.ingest_sync(
                    session_id=self.session_id,
                    role=msg.role,
                    content=msg.text,
                    content_json=msg.content_json,
                    turn_index=i + self._turn_index_offset,
                    # Persist the turn's trust tag so an injected (untrusted)
                    # task result survives the LCM round-trip rather than being
                    # silently dropped to the trusted default.
                    provenance=msg.provenance,
                    is_trusted=msg.is_trusted,
                )
                wrote_any = True
                if msg.role == "assistant":
                    # The row the client's streamed bubble reconciles to. Take
                    # the LAST one: a tool-using turn persists several assistant
                    # rows, and the final text is where the reply settles.
                    row_id = getattr(self._lcm_engine, "last_ingested_row_id", None)
                    if row_id:
                        last_assistant_row_id = row_id
                self._note_persisted(i)
            if seal:
                if end > self._lcm_persisted_len:
                    self._lcm_persisted_len = end
                self._lcm_persisted_ahead = {
                    i for i in self._lcm_persisted_ahead
                    if i >= self._lcm_persisted_len
                }
            # Sprint 2 (OAra): the durable-DAG relief valve. maybe_compact was
            # CLI-only for the daemon's whole life — every gateway ingested
            # messages forever and nothing ever summarized. This is the one
            # choke point all paths share (telegram/slack via
            # add_result_messages, web/Beacon/Bridge via persist_loop_result).
            # Skipped when the span was a pure dedup no-op — nothing new to
            # compact.
            if wrote_any:
                self._schedule_lcm_compaction()
            return last_assistant_row_id
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
                            "span_start": start,
                            "persisted_len": self._lcm_persisted_len,
                            "messages_len": len(self.messages),
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
                "(span from %d, watermark %d) — agent loop unaffected",
                self.session_id, start, self._lcm_persisted_len,
                exc_info=True,
            )

    def rollback_last(self) -> None:
        """Remove the most recently appended message (error recovery).

        If the popped row was already durable it stays in LCM (append-only
        store; unchanged behavior) — but the watermark must retreat so the
        NEXT message at this position persists instead of being skipped as
        already-written.
        """
        if self.messages:
            self.messages.pop()
            idx = len(self.messages)
            self._lcm_persisted_ahead.discard(idx)
            if self._lcm_persisted_len > idx:
                self._lcm_persisted_len = idx

    def rollback_to(self, length: int) -> int:
        """Discard every message appended past ``length``. Returns the count.

        The span twin of :meth:`rollback_last`, for the in-place ``run_loop``
        contract :meth:`persist_loop_result` describes: a turn that dies
        mid-flight has ALREADY appended its assistant and tool-result rows
        onto ``self.messages``. Leaving them is not neutral — the next message
        rebuilds a prompt containing whatever killed this one.

        That is not hypothetical. On 2026-08-17 a ``bash`` result carrying
        llama.cpp's per-process media marker (curl of the inference server's
        ``/props``) made the backend reject the prompt; the rows stayed, every
        later message re-sent the marker and took the same 400, and the web
        surface has no ``/reset`` to clear it. Microcompaction would have
        truncated the offending result away, but it cannot fire before round
        ``microcompact_after_turns`` — and the turn dies on round 0.

        Durable rows stay in LCM (append-only, unchanged) but the watermark
        retreats for each freed position, so the NEXT message written there
        persists instead of being skipped as already-written.
        """
        length = max(0, length)
        discarded = len(self.messages) - length
        if discarded <= 0:
            return 0
        del self.messages[length:]
        for idx in range(length, length + discarded):
            self._lcm_persisted_ahead.discard(idx)
        if self._lcm_persisted_len > length:
            self._lcm_persisted_len = length
        return discarded

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
        """Reset conversation history.

        IN-MEMORY ONLY. This is the whole of what ``/reset`` does: the durable
        LCM rows, their FTS entries, any summary, and anything already mined
        into ``memory.db`` or the wiki all survive it, and the MemoryExtractor
        watermark is untouched — so a message cleared here is still mined on
        the next cadence tick. If you want a turn not to be remembered, that
        decision has to be made BEFORE it is sent: see
        ``prometheus.config.ephemeral``.
        """
        self.messages = []
        # Positions restart at 0, so the watermark must too — otherwise every
        # post-reset message would look already-persisted and be dropped.
        self._lcm_persisted_len = 0
        self._lcm_persisted_ahead.clear()

    def set_lcm_engine(self, engine: object | None) -> None:
        """Point this session at an LCM engine, or at ``None`` for no durable
        persistence at all.

        ``None`` is what makes a session ephemeral: every persist site in this
        class is already guarded by ``if self._lcm_engine is not None``, so a
        null engine turns the whole durable path — message rows, FTS index,
        the compaction that produces summaries, and therefore every downstream
        consumer (MemoryExtractor → memory.db → wiki) — into a no-op without a
        second branch anywhere. :class:`SessionManager` re-applies this on
        every ``get_or_create`` so a mid-conversation toggle takes effect on
        the next turn.
        """
        self._lcm_engine = engine

    @property
    def lcm_engine(self) -> object | None:
        """The wired LCM engine, or None for an ephemeral session. Read-only
        public access for callers (e.g. the session-title hook) that need the
        engine's stores without reaching into a private field."""
        return self._lcm_engine

    def trim(self, max_messages: int = MAX_SESSION_MESSAGES) -> None:
        """Truncate from the front if history exceeds *max_messages*.

        Every surviving message shifts down by ``dropped`` positions, so the
        persistence bookkeeping shifts with them (turn_index has always been
        list position; that contract is unchanged).
        """
        if len(self.messages) > max_messages:
            dropped = len(self.messages) - max_messages
            self.messages = self.messages[-max_messages:]
            self._lcm_persisted_len = max(0, self._lcm_persisted_len - dropped)
            self._lcm_persisted_ahead = {
                i - dropped for i in self._lcm_persisted_ahead if i >= dropped
            }


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
        # feat/session-rehydrate: gate for rehydrate_if_cold(). Default OFF
        # (matching the compaction.enabled precedent for a new context-
        # shaping behaviour); the daemon wires it from sessions.rehydrate.
        self.rehydrate_enabled: bool = False
        # One lock per session id — THE cross-surface turn serializer.
        # Every surface that runs agent turns (telegram gateway, web/WS
        # bridge) resolves its per-session lock here via turn_lock_for(),
        # so a turn injected through one surface cannot interleave with a
        # live turn on the same session running through another.
        self._turn_locks: dict[str, asyncio.Lock] = {}

    def _effective_lcm_engine(self, session_id: str) -> object | None:
        """The engine this session may persist through — ``None`` if ephemeral.

        Resolved through the real config-backed reader on every call (no
        cache), which is what lets ``/ephemeral on`` take effect on the very
        next turn and survive a daemon restart.
        """
        if is_session_ephemeral(session_id):
            return None
        return self.lcm_engine

    def get_or_create(self, session_id: str) -> ChatSession:
        """Return the existing session or create a new one.

        The ephemeral flag is re-applied on EVERY call, not just at creation.
        A chat that toggles mid-conversation already has a cached ChatSession
        holding a live engine reference; without the re-apply, ``/ephemeral
        on`` would do nothing until the daemon restarted — the precise shape
        of "the setting appears to take effect because a value with that name
        exists".
        """
        session = self._sessions.get(session_id)
        if session is None:
            session = ChatSession(
                session_id, lcm_engine=self._effective_lcm_engine(session_id)
            )
            self._sessions[session_id] = session
        else:
            session.set_lcm_engine(self._effective_lcm_engine(session_id))
        return session

    def rehydrate_if_cold(self, session_id: str) -> int:
        """Restore a cold session's recent durable history into the live
        working set. Returns how many messages were restored (0 = nothing to
        do, not eligible, or disabled).

        Closes the two-stores/no-rehydrate gap: after a daemon restart a
        session's full history was servable to CLIENTS over REST while the
        MODEL started blind. Called from the send paths (not get_or_create,
        so switch_session and POST /api/sessions stay structural no-ops).

        Eligibility and boundaries, each load-bearing:

        - ``list_sessions()`` is the gate because it is the ONLY
          tombstone-aware read in the store — every other reader would
          resurrect a forgotten chat on its next message, turning DELETE
          into "forget it for one round trip".
        - the ephemeral reader is consulted the same way persistence
          consults it: an ephemeral session reads nothing back, or
          ``/ephemeral on`` would write nothing while silently restoring
          everything.
        - the restored tail STARTS at a clean human turn (role user,
          provenance user, text-only). Cutting at a count or rowid can
          orphan a ToolResultBlock from its tool_use — a hard 400 from
          every provider. No clean turn in the window → restore nothing
          (fail closed: exactly today's behaviour).
        - the window is capped by rows AND estimated tokens, so the loop
          is never handed an initial set the compactor structurally cannot
          rescue (single-pass, and off by default).
        """
        if not self.rehydrate_enabled:
            return 0
        existing = self._sessions.get(session_id)
        if existing is not None and existing.messages:
            return 0  # warm — the live set is authoritative
        engine = self._effective_lcm_engine(session_id)
        store = getattr(engine, "conversation_store", None)
        if store is None:
            return 0
        try:
            if not any(
                row.get("session_id") == session_id
                for row in store.list_sessions()
            ):
                return 0  # no durable history, or tombstoned
            parts, _more = store.messages_page(
                limit=_REHYDRATE_WINDOW, session_id=session_id
            )
        except Exception:
            log.warning(
                "rehydrate_if_cold: durable read failed for %s — starting "
                "cold, exactly as before the feature", session_id,
                exc_info=True,
            )
            return 0
        if not parts:
            return 0

        # Token budget, newest-first: keep the most recent rows that fit.
        kept: list = []
        budget = _REHYDRATE_TOKEN_BUDGET
        for part in reversed(parts):
            cost = max(1, len(part.content_json or part.content or "") // 4)
            if kept and budget - cost < 0:
                break
            budget -= cost
            kept.append(part)
        kept.reverse()

        converted = [
            ConversationMessage.from_stored(
                role=p.role,
                content=p.content,
                content_json=p.content_json,
                provenance=getattr(p, "provenance", "user"),
                is_trusted=getattr(p, "is_trusted", True),
            )
            for p in kept
        ]
        start = next(
            (
                i for i, m in enumerate(converted)
                if m.role == "user"
                and m.provenance == "user"
                and all(type(b).__name__ == "TextBlock" for b in m.content)
            ),
            None,
        )
        if start is None:
            return 0
        converted = converted[start:]
        kept = kept[start:]

        session = self.get_or_create(session_id)
        try:
            session.restore(
                converted,
                # +1 past the LARGEST historical turn_index in the tail, not
                # the last row's: turn_index restarts per daemon lifetime,
                # so the last row's value is not necessarily the max.
                next_turn_index=max(p.turn_index for p in kept) + 1,
            )
        except RuntimeError:
            # Raced by a concurrent turn that warmed the session between
            # the cold check and here — its live set wins.
            return 0
        log.info(
            "rehydrate: %s restored %d message(s) (window %d rows, "
            "boundary trimmed %d)",
            session_id, len(converted), len(parts), start,
        )
        return len(converted)

    def get(self, session_id: str) -> "ChatSession | None":
        """Return the existing session for ``session_id``, or None.

        Non-creating lookup — used by session-mutating commands (/steer,
        /unqueue, /clearsteers) that target an in-flight run and no-op when
        there's no active session, unlike get_or_create.
        """
        return self._sessions.get(session_id)

    def turn_lock_for(self, session_id: str) -> asyncio.Lock:
        """Return the per-session TURN lock, creating it on first use.

        The single serialization point for agent turns on one session,
        shared by every surface wired to this manager. The telegram
        gateway (audit M6) and the web/WS bridge (2026-08-11
        duplicate-rows fix) each kept a private map of this exact shape,
        which serialized turns within a surface but not across surfaces:
        a managed-task re-engagement (``telegram.inject_turn`` targeting
        e.g. a ``desktop:*`` session) ran under telegram's lock while a
        live Beacon turn on the SAME session ran under the bridge's lock,
        and both appended to the shared ``ChatSession.messages``
        concurrently — interleaved model rounds, scrambled order. (The
        ChatSession persistence watermark caps the durable damage at zero
        duplicate LCM rows; it does not prevent the in-memory interleave.)
        Both surfaces now delegate here.

        Turns on one session serialize; different sessions never contend.
        The map lazily initializes for managers built via ``__new__`` in
        tests — the same resilience the surface-local helpers had.
        """
        locks = getattr(self, "_turn_locks", None)
        if locks is None:
            locks = {}
            self._turn_locks = locks
        lock = locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            locks[session_id] = lock
        return lock

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
