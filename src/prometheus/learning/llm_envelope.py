"""LLMCallEnvelope — shared invocation envelope for autonomous subsystems.

Replaces the per-subsystem ``_call_model`` pattern that silently swallowed
a ``ValidationError`` for an unknown duration before it was surfaced by
PR #1 / ``ed8f1a6``. See ``docs/audits/SILENT-FAILURE-AUDIT.md`` (PR #2).

The envelope owns three responsibilities that every autonomous LLM caller
needs:

1. **Correct message-shape construction.** Callers pass a raw ``prompt:
   str``; the envelope wraps it in
   ``ConversationMessage(role="user", content=[TextBlock(text=prompt)])``.
   The bug shape from ``ed8f1a6`` — passing ``content=prompt`` (a string)
   to a field that wants ``list[ContentBlock]`` — becomes structurally
   impossible at the call site.

2. **Failure surfacing.** Every exception is written to
   ``telemetry.silent_failures`` before the configured ``on_failure``
   policy decides whether to re-raise (default), log-only, or
   return-none. The original log message is preserved.

3. **Liveness telemetry.** Every successful call writes a
   ``subsystem_runs`` row with duration, output size, and a summary so
   ``/health`` can answer "is this subsystem actually running?"

Usage::

    envelope = LLMCallEnvelope(
        subsystem="skill_creator",
        telemetry=telemetry_store,        # ToolCallTelemetry | None
        on_failure="raise",               # "raise" | "log_only" | "return_none"
    )
    text = await envelope.call(
        provider=self._provider,
        model=self._model,
        prompt=prompt,
        max_tokens=1024,
        operation="generate_skill",
    )

Why three on_failure modes? The migration from existing ``_call_model``
helpers maps cleanly:

- ``"raise"`` (default) — for future code. Lets the caller decide how to
  recover, instead of silently degrading.
- ``"return_none"`` — preserves the legacy ``maybe_create`` /
  ``maybe_refine_recent`` contract that returns ``None`` / ``False`` on
  failure. The telemetry row still lands; the caller just doesn't see
  the exception.
- ``"log_only"`` — for emit-on-success paths where the caller intends to
  fall through. Effectively re-raises ``None`` then continues.

The point is that *failure is always observable* in telemetry.db. The
caller-visible behaviour stays compatible with the legacy contract.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from prometheus.engine.messages import ConversationMessage
    from prometheus.providers.base import ModelProvider
    from prometheus.telemetry.tracker import ToolCallTelemetry

log = logging.getLogger(__name__)


OnFailure = Literal["raise", "log_only", "return_none"]


class LLMCallShapeError(TypeError):
    """Raised when a caller hands the envelope a malformed message list.

    Specifically: a ``ConversationMessage`` whose ``content`` is not a
    ``list[ContentBlock]``. The pydantic schema catches this at
    construction time; this exception type lets the envelope surface
    the same bug explicitly when it's handed a pre-built message list
    (the path some tests exercise).

    Named for ed8f1a6 — the original ValidationError shape that hid
    inside ``except Exception: log.exception(...)``.
    """


@dataclass
class LLMCallResult:
    """What :meth:`LLMCallEnvelope.call` returns when ``on_failure='log_only'``.

    On success: ``text`` populated, ``error`` is ``None``.
    On failure (log_only): ``text`` is ``""``, ``error`` is the exception.
    Callers that want a plain string should use ``on_failure="return_none"``
    (returns ``str | None``) instead.
    """

    text: str
    error: BaseException | None
    duration_ms: float


class LLMCallEnvelope:
    """Shared LLM invocation envelope. See module docstring."""

    def __init__(
        self,
        subsystem: str,
        telemetry: "ToolCallTelemetry | None" = None,
        *,
        on_failure: OnFailure = "raise",
    ) -> None:
        if on_failure not in ("raise", "log_only", "return_none"):
            raise ValueError(
                f"on_failure must be one of 'raise' | 'log_only' | 'return_none' "
                f"(got {on_failure!r})"
            )
        self._subsystem = subsystem
        self._telemetry = telemetry
        self._on_failure = on_failure

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    async def call(
        self,
        *,
        provider: "ModelProvider",
        model: str,
        prompt: str,
        max_tokens: int = 2048,
        operation: str = "_call_model",
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Invoke the model and return concatenated text on success.

        Return type depends on ``on_failure`` mode:
          - ``"raise"``       — returns ``str`` on success; raises on failure.
          - ``"return_none"`` — returns ``str | None`` (None on failure).
          - ``"log_only"``    — returns :class:`LLMCallResult` always.

        ``operation`` is recorded on both success and failure telemetry
        so a subsystem can distinguish "generate_skill" from "refine_skill"
        when multiple call paths share an envelope.
        """
        from prometheus.engine.messages import ConversationMessage, TextBlock
        from prometheus.providers.base import (
            ApiMessageRequest,
            ApiTextDeltaEvent,
        )

        # Build the request with the canonically correct content shape.
        # This is the structural barrier against ed8f1a6 — there is no
        # call site that can pass `content=prompt`.
        request = ApiMessageRequest(
            model=model,
            messages=[
                ConversationMessage(
                    role="user",
                    content=[TextBlock(text=prompt)],
                )
            ],
            max_tokens=max_tokens,
        )

        started = time.time()
        text_parts: list[str] = []
        try:
            async for event in provider.stream_message(request):
                if isinstance(event, ApiTextDeltaEvent):
                    text_parts.append(event.text)
        except BaseException as exc:  # noqa: BLE001 — we re-raise per policy
            duration_ms = (time.time() - started) * 1000.0
            self._record_failure(operation, exc, context, duration_ms)
            log.exception(
                "%s.%s: LLM call failed (on_failure=%s)",
                self._subsystem, operation, self._on_failure,
            )
            if self._on_failure == "raise":
                raise
            if self._on_failure == "return_none":
                return None
            # log_only
            return LLMCallResult(text="", error=exc, duration_ms=duration_ms)

        text = "".join(text_parts)
        duration_ms = (time.time() - started) * 1000.0
        self._record_success(operation, text, duration_ms, context)
        if self._on_failure == "log_only":
            return LLMCallResult(text=text, error=None, duration_ms=duration_ms)
        return text

    # ------------------------------------------------------------------
    # Explicit message-shape validator (used by tests + future callers)
    # ------------------------------------------------------------------

    @staticmethod
    def validate_messages(messages: "list[ConversationMessage]") -> None:
        """Raise :class:`LLMCallShapeError` if any message has the ed8f1a6 shape.

        The :class:`ConversationMessage` pydantic schema already catches
        this at construction time, but tests want a path that surfaces
        the bug explicitly without depending on pydantic's error text.
        """
        from prometheus.engine.messages import (
            ConversationMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
        )

        # `ContentBlock` is an Annotated Union with a discriminator —
        # isinstance() against it fails. Use the concrete types directly.
        _valid_block_types = (TextBlock, ToolUseBlock, ToolResultBlock)

        for i, msg in enumerate(messages):
            if not isinstance(msg, ConversationMessage):
                raise LLMCallShapeError(
                    f"messages[{i}] is {type(msg).__name__}, "
                    f"expected ConversationMessage"
                )
            content = msg.content
            if isinstance(content, str):
                raise LLMCallShapeError(
                    f"messages[{i}].content is a str — this is the ed8f1a6 "
                    f"shape. ConversationMessage.content must be "
                    f"list[ContentBlock]. Wrap your prompt: "
                    f"`content=[TextBlock(text=prompt)]`."
                )
            if not isinstance(content, list):
                raise LLMCallShapeError(
                    f"messages[{i}].content must be list[ContentBlock], "
                    f"got {type(content).__name__}"
                )
            for j, block in enumerate(content):
                if not isinstance(block, _valid_block_types):
                    raise LLMCallShapeError(
                        f"messages[{i}].content[{j}] is "
                        f"{type(block).__name__}, expected a ContentBlock "
                        f"subclass (TextBlock / ToolUseBlock / ToolResultBlock)"
                    )

    # ------------------------------------------------------------------
    # Internal — telemetry writes are best-effort, never raise
    # ------------------------------------------------------------------

    def _record_failure(
        self,
        operation: str,
        exc: BaseException,
        context: dict[str, Any] | None,
        duration_ms: float,
    ) -> None:
        if self._telemetry is None:
            return
        try:
            self._telemetry.record_silent_failure(
                subsystem=self._subsystem,
                operation=operation,
                exc=exc,
                context=context,
            )
            self._telemetry.record_run(
                subsystem=self._subsystem,
                operation=operation,
                outcome="failed",
                duration_ms=duration_ms,
                summary={"exception_type": type(exc).__name__},
            )
        except Exception:
            log.debug(
                "LLMCallEnvelope: telemetry write failed for "
                "%s.%s — failure was still surfaced via on_failure=%s",
                self._subsystem, operation, self._on_failure,
                exc_info=True,
            )

    def _record_success(
        self,
        operation: str,
        text: str,
        duration_ms: float,
        context: dict[str, Any] | None,
    ) -> None:
        if self._telemetry is None:
            return
        try:
            summary = {"output_chars": len(text)}
            if context:
                summary["context"] = context
            self._telemetry.record_run(
                subsystem=self._subsystem,
                operation=operation,
                outcome="success",
                duration_ms=duration_ms,
                summary=summary,
            )
        except Exception:
            log.debug(
                "LLMCallEnvelope: success telemetry write failed for %s.%s",
                self._subsystem, operation,
                exc_info=True,
            )
