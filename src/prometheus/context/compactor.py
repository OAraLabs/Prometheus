"""ContextCompactor — single-layer, assembly-time context compaction.

When the estimated assembled prompt (system prompt + history + tool schemas)
approaches the model's effective context, summarize the oldest contiguous
span of turns via the SAME local model and substitute the summary into the
RENDER VIEW handed to the provider. The loop's message list, the session,
and lcm.db are never mutated — compaction is a prompt-assembly concern,
never a storage mutation. The LCM DAG stays the untouched source of truth.

SINGLE-LAYER BY CONSTRUCTION (the spec's hard constraint): the synthetic
summary message carries ``provenance="compactor"``, spans contain only
``provenance == "user"`` messages, and any other provenance is a span
BARRIER — so a summary can never be ingested into a later span. Very long
sessions accumulate *sibling* summaries, never summaries of summaries.

Ships behind ``compaction.enabled`` (default false): with the section absent
``from_config`` returns ``None`` and nothing changes anywhere.

Config (``compaction:`` section of prometheus.yaml — ABSENT by default)::

    compaction:
      enabled: true
      threshold_pct: 0.75       # of (effective_limit - reserve_tokens)
      reserve_tokens: 4096      # output headroom subtracted from n_ctx
      protect_recent_turns: 8   # recent user turns never compacted
      max_summary_tokens: 512   # hard output cap for the ONE summary call

n_ctx comes from ``context.effective_limit`` with
``context.model_overrides.<model>.effective_limit`` applied — the same
resolution as ``context/budget.py``.

Fail-loud policy: a failed summarization call (the envelope already wrote
``silent_failures``) or an insane result (empty, or not smaller than the
span it replaces) logs at ERROR, emits a ``context_compaction_failed``
signal event (durable in ``telemetry.signal_events`` → /events, Beacon),
and falls back EXPLICITLY to the pre-existing overflow behavior: the
unmodified prompt is sent and the provider/server reports the overflow.
Never a silent fallback. Every successful compaction records a
``context_compaction`` event with span size, tokens before/after, duration,
and cache state.

Inspired by the context-compactor design in the Odysseus project (MIT);
clean-room, design knowledge only. No source copied.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from prometheus.context.token_estimation import estimate_tokens

if TYPE_CHECKING:
    from prometheus.engine.messages import ConversationMessage
    from prometheus.providers.base import ModelProvider

log = logging.getLogger(__name__)

DEFAULT_THRESHOLD_PCT = 0.75
DEFAULT_RESERVE_TOKENS = 4096
DEFAULT_PROTECT_RECENT_TURNS = 8
DEFAULT_MAX_SUMMARY_TOKENS = 512

# A span smaller than this many messages isn't worth a model call.
MIN_SPAN_MESSAGES = 4

# Idempotence cache (FIFO-bounded): span content hash -> summary text.
_CACHE_MAX_ENTRIES = 16

# Caps on the summarizer's INPUT rendering — the span being summarized is by
# definition the oversized part of the conversation, and the summary call
# goes to the same context-limited local model. Per-message and total caps
# keep the one call viable; anything dropped is marked, logged, and counted
# in telemetry (no silent truncation).
_RENDER_PER_MESSAGE_CHARS = 1_000
_RENDER_TOTAL_CHARS = 18_000

# Prompt follows the repo's existing summarization voice
# (memory/lcm_summarize.py::_MESSAGE_SUMMARY_PROMPT) with the spec's
# preservation list.
_SUMMARY_PROMPT = (
    "You are a conversation compactor. Summarize the following conversation "
    "span into one concise factual digest. Preserve:\n"
    "  - The user's goals and any decisions made\n"
    "  - File paths, identifiers, commands, and technical details\n"
    "  - Unresolved questions and pending work\n\n"
    "Be terse and factual. Do NOT add commentary.\n\n"
    "Conversation:\n{conversation}"
)

SUMMARY_MARKER_PREFIX = "[Compacted summary of turns"


class ContextCompactor:
    """Single-layer assembly-time compactor. Construct via :meth:`from_config`."""

    def __init__(
        self,
        *,
        provider: "ModelProvider",
        model: str,
        effective_limit: int,
        threshold_pct: float = DEFAULT_THRESHOLD_PCT,
        reserve_tokens: int = DEFAULT_RESERVE_TOKENS,
        protect_recent_turns: int = DEFAULT_PROTECT_RECENT_TURNS,
        max_summary_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS,
        telemetry: Any | None = None,
        signal_bus: Any | None = None,
    ) -> None:
        from prometheus.learning.llm_envelope import LLMCallEnvelope

        self._provider = provider
        self._model = model
        self._effective_limit = int(effective_limit)
        self._threshold_pct = float(threshold_pct)
        self._reserve_tokens = int(reserve_tokens)
        self._protect_recent_turns = int(protect_recent_turns)
        self._max_summary_tokens = int(max_summary_tokens)
        self._telemetry = telemetry
        self._signal_bus = signal_bus
        # span-content-hash -> summary text (FIFO-bounded)
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        # session_id -> anchored span end index. Once a span is summarized we
        # keep substituting THAT span (cache hit, zero calls) until the
        # compacted estimate re-crosses the threshold; only then the span
        # extends and one new call is paid. This is the spec's idempotence:
        # the cache invalidates only when the span actually has to change.
        self._session_spans: dict[str, int] = {}
        # return_none: failures are already loud (silent_failures + failed
        # subsystem_runs); apply() then falls back explicitly.
        self._envelope = LLMCallEnvelope(
            subsystem="context_compactor",
            telemetry=telemetry,
            on_failure="return_none",
        )

    # -- construction -----------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None,
        *,
        provider: "ModelProvider",
        model: str,
        telemetry: Any | None = None,
        signal_bus: Any | None = None,
    ) -> "ContextCompactor | None":
        """Build from the loaded prometheus.yaml dict.

        Returns ``None`` unless ``compaction.enabled`` is truthy — the
        default-off contract: absent section = zero behavior change.
        """
        cfg = config or {}
        comp = cfg.get("compaction") or {}
        if not comp.get("enabled", False):
            return None

        ctx = cfg.get("context") or {}
        effective_limit = int(ctx.get("effective_limit", 24000))
        overrides = ctx.get("model_overrides") or {}
        per_model = overrides.get(model)
        if isinstance(per_model, dict) and "effective_limit" in per_model:
            effective_limit = int(per_model["effective_limit"])

        return cls(
            provider=provider,
            model=model,
            effective_limit=effective_limit,
            threshold_pct=comp.get("threshold_pct", DEFAULT_THRESHOLD_PCT),
            reserve_tokens=comp.get("reserve_tokens", DEFAULT_RESERVE_TOKENS),
            protect_recent_turns=comp.get(
                "protect_recent_turns", DEFAULT_PROTECT_RECENT_TURNS),
            max_summary_tokens=comp.get(
                "max_summary_tokens", DEFAULT_MAX_SUMMARY_TOKENS),
            telemetry=telemetry,
            signal_bus=signal_bus,
        )

    @property
    def signal_bus(self) -> Any | None:
        return self._signal_bus

    @signal_bus.setter
    def signal_bus(self, bus: Any) -> None:
        """Late wiring — the daemon constructs SignalBus after the loop."""
        self._signal_bus = bus

    # -- estimation --------------------------------------------------------

    @staticmethod
    def _message_tokens(msg: "ConversationMessage") -> int:
        """Estimated tokens for one message — content_json covers every block
        type (text, tool_use, tool_result), unlike .text."""
        try:
            return estimate_tokens(msg.content_json)
        except Exception:
            return estimate_tokens(getattr(msg, "text", "") or "")

    def estimate_total(
        self,
        system_prompt: str,
        messages: list,
        tools_chars: int = 0,
    ) -> int:
        """Estimated tokens of the assembled request."""
        total = estimate_tokens(system_prompt or "")
        total += sum(self._message_tokens(m) for m in messages)
        total += max(0, int(tools_chars)) // 4  # same chars/4 heuristic
        return total

    def _threshold_tokens(self) -> int:
        available = max(1, self._effective_limit - self._reserve_tokens)
        return int(available * self._threshold_pct)

    # -- span selection ----------------------------------------------------

    def _protected_boundary(self, messages: list) -> int:
        """Index of the first message of the protected tail: the last
        ``protect_recent_turns`` user-role messages and everything after the
        earliest of them never compact (microcompaction's fresh-window
        convention)."""
        seen = 0
        for i in range(len(messages) - 1, -1, -1):
            if getattr(messages[i], "role", "") == "user":
                seen += 1
                if seen >= self._protect_recent_turns:
                    return i
        return 0

    def _select_span_end(self, messages: list) -> int:
        """End index (exclusive) of the compactable prefix span.

        The span is the contiguous prefix of ``provenance == "user"``
        messages (which includes the loop's own assistant/tool-result
        messages — they default to "user" provenance) up to the protected
        boundary. Any other provenance (task_supervisor, cron, orchestrator,
        compactor, …) is a BARRIER: trust-tagged injections and managed-task
        context are never summarized away, and the single-layer guarantee
        holds because synthetic summaries can never enter a span.
        """
        boundary = self._protected_boundary(messages)
        end = 0
        for i in range(boundary):
            if getattr(messages[i], "provenance", "user") != "user":
                break
            end = i + 1
        return end

    @staticmethod
    def _span_key(session_id: str, span: list) -> str:
        h = hashlib.sha256()
        h.update(session_id.encode("utf-8", "replace"))
        for msg in span:
            h.update(b"\x00")
            h.update(getattr(msg, "role", "").encode("utf-8", "replace"))
            h.update(b"\x01")
            h.update(getattr(msg, "provenance", "user").encode("utf-8", "replace"))
            h.update(b"\x02")
            try:
                h.update(msg.content_json.encode("utf-8", "replace"))
            except Exception:
                h.update((getattr(msg, "text", "") or "").encode("utf-8", "replace"))
        return h.hexdigest()

    # -- summarizer input --------------------------------------------------

    @staticmethod
    def _render_message(msg: "ConversationMessage") -> str:
        parts: list[str] = []
        for block in getattr(msg, "content", None) or ():
            btype = getattr(block, "type", "")
            if btype == "text":
                parts.append(block.text)
            elif btype == "tool_use":
                parts.append(f"[called {block.name}({block.input})]")
            elif btype == "tool_result":
                flag = " ERROR" if getattr(block, "is_error", False) else ""
                parts.append(f"[result{flag}: {block.content}]")
        text = " ".join(p for p in parts if p)
        if len(text) > _RENDER_PER_MESSAGE_CHARS:
            text = text[:_RENDER_PER_MESSAGE_CHARS] + "…"
        return f"{getattr(msg, 'role', '?')}: {text}"

    def _render_span(self, span: list) -> tuple[str, int]:
        """Render the span for the summarizer. Returns (text, omitted_count).

        Bounded input (see module constants); omissions are marked in the
        text, logged, and recorded in telemetry — never silent.
        """
        lines: list[str] = []
        used = 0
        omitted = 0
        for msg in span:
            line = self._render_message(msg)
            if used + len(line) > _RENDER_TOTAL_CHARS:
                omitted += 1
                continue
            lines.append(line)
            used += len(line)
        if omitted:
            lines.append(
                f"[{omitted} additional message(s) omitted from the "
                f"summarizer input for size]")
        return "\n".join(lines), omitted

    # -- telemetry ---------------------------------------------------------

    async def _record_event(self, signal_type: str, payload: dict) -> None:
        """One durable write path (Sprint-A convention): the SignalBus
        persists to signal_events before broadcasting; without a bus, write
        the telemetry row directly. Never both."""
        try:
            if self._signal_bus is not None:
                from prometheus.sentinel.signals import ActivitySignal

                await self._signal_bus.emit(ActivitySignal(
                    kind=signal_type,
                    payload=payload,
                    source="context_compactor",
                ))
                return
            if self._telemetry is not None:
                self._telemetry.record_signal_event(
                    signal_type=signal_type,
                    payload=payload,
                    source_subsystem="context_compactor",
                )
        except Exception:
            log.exception("ContextCompactor: telemetry event write failed")

    async def _fail_loud(self, reason: str, session_id: str, span_len: int) -> None:
        log.error(
            "ContextCompactor FAILED (%s) for session %s — falling back to "
            "the pre-existing behavior: sending the UNCOMPACTED prompt "
            "(provider-side overflow possible)", reason, session_id,
        )
        if self._telemetry is not None:
            try:
                self._telemetry.record_silent_failure(
                    subsystem="context_compactor",
                    operation="summarize_span",
                    exc=RuntimeError(reason),
                    context={"session_id": session_id, "span_messages": span_len},
                )
            except Exception:
                log.warning("ContextCompactor: silent_failure write failed",
                            exc_info=True)
        await self._record_event("context_compaction_failed", {
            "session_id": session_id,
            "span_messages": span_len,
            "reason": reason,
        })

    # -- the compaction ----------------------------------------------------

    async def apply(
        self,
        messages: list,
        *,
        session_id: str = "",
        system_prompt: str = "",
        tools_chars: int = 0,
    ) -> list:
        """Return the render view: ``messages`` itself when no compaction is
        needed, or a NEW list with the oldest span replaced by one synthetic
        summary message. NEVER mutates ``messages`` or any message in it.
        """
        tokens_before = self.estimate_total(system_prompt, messages, tools_chars)
        threshold = self._threshold_tokens()
        if tokens_before <= threshold:
            return messages

        max_end = self._select_span_end(messages)
        if max_end < MIN_SPAN_MESSAGES:
            log.debug(
                "ContextCompactor: over threshold (%d > %d) but compactable "
                "span has only %d message(s) — nothing to do",
                tokens_before, threshold, max_end,
            )
            return messages

        # Anchored span: reuse the previously summarized span while the
        # compacted estimate stays under threshold (zero model calls); extend
        # only when it re-crosses.
        span_end = max_end
        anchored = self._session_spans.get(session_id)
        if anchored is not None and MIN_SPAN_MESSAGES <= anchored <= max_end:
            anchored_key = self._span_key(session_id, messages[:anchored])
            cached = self._cache.get(anchored_key)
            if cached is not None:
                anchored_tokens = (
                    tokens_before
                    - sum(self._message_tokens(m) for m in messages[:anchored])
                    + estimate_tokens(cached)
                )
                if anchored_tokens <= threshold:
                    span_end = anchored

        span = messages[:span_end]
        span_tokens = sum(self._message_tokens(m) for m in span)
        key = self._span_key(session_id, span)
        summary = self._cache.get(key)
        cache_hit = summary is not None
        omitted = 0
        duration_ms = 0.0

        if not cache_hit:
            conversation, omitted = self._render_span(span)
            started = time.time()
            raw = await self._envelope.call(
                provider=self._provider,
                model=self._model,
                prompt=_SUMMARY_PROMPT.format(conversation=conversation),
                max_tokens=self._max_summary_tokens,
                operation="summarize_span",
                context={"session_id": session_id, "span_messages": len(span)},
            )
            duration_ms = (time.time() - started) * 1000.0

            # Sanity gate: empty, or not actually smaller than what it
            # replaces → loud failure + explicit fallback.
            if raw is None or not str(raw).strip():
                await self._fail_loud(
                    "summarizer returned empty/failed", session_id, len(span))
                return messages
            summary = str(raw).strip()
            if estimate_tokens(summary) >= span_tokens:
                await self._fail_loud(
                    f"summary ({estimate_tokens(summary)} tok) not smaller "
                    f"than the span it replaces ({span_tokens} tok)",
                    session_id, len(span),
                )
                return messages

            self._cache[key] = summary
            while len(self._cache) > _CACHE_MAX_ENTRIES:
                self._cache.popitem(last=False)
            self._session_spans[session_id] = span_end

        # Build the substitution — a single clearly-marked synthetic message,
        # tagged compactor-generated (closed Provenance enum), trusted (the
        # agent's own machinery; untrusted turns can never be IN a span).
        from prometheus.engine.messages import ConversationMessage, TextBlock

        synthetic = ConversationMessage(
            role="user",
            content=[TextBlock(
                text=f"{SUMMARY_MARKER_PREFIX} 1–{len(span)}] {summary}")],
            provenance="compactor",
            is_trusted=True,
        )
        out = [synthetic] + list(messages[span_end:])
        tokens_after = self.estimate_total(system_prompt, out, tools_chars)

        log.info(
            "ContextCompactor: %d msgs → 1 summary (%s); tokens %d → %d "
            "(threshold %d)%s",
            len(span), session_id, tokens_before, tokens_after, threshold,
            " [cache]" if cache_hit else "",
        )
        await self._record_event("context_compaction", {
            "session_id": session_id,
            "span_messages": len(span),
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "threshold": threshold,
            "duration_ms": round(duration_ms, 1),
            "cache_hit": cache_hit,
            "summarizer_input_omitted": omitted,
        })
        return out
