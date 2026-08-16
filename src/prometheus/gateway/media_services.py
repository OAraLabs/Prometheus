"""Shared media services — vision, STT, and document budget helpers.

SPRINT G2: mechanical extraction of TelegramAdapter._describe_image /
_transcribe_audio / _truncate_for_context so the Discord adapter routes
attachments through the SAME services Telegram uses (images → vision
analysis, audio → Whisper STT, documents → context-budgeted extraction)
instead of duplicating them. Telegram delegates here; behaviour is pinned
byte-identical in tests/test_gateway_media_pins.py (written and run green
against the pre-refactor tree).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


async def describe_image(image_path: str, *, provider: Any = None) -> str | None:
    """Run an image through VisionTool for a text description.

    Returns None on any failure (missing multimodal model, tool error) —
    callers fall back to a plain "[The user sent a photo]" injection.

    ⚠ THE TRY IS DELIBERATELY NARROW (#78's broad-except shape, and the
    reason #230 shipped). Setup — the imports, ``VisionTool()``, the
    input and the context — is OUR code with no I/O in it. If any of that
    raises it is a programming error, not "vision unavailable", and it
    MUST propagate: the beacon copy of this function died on an
    ``AttributeError`` in exactly that setup region and the broad
    ``except`` downgraded it to a log line and a ``None``, so every image
    upload silently produced no description and CI stayed green on two
    Python versions.

    Only ``tool.execute`` is guarded, because only it can fail for
    legitimate environmental reasons (no mmproj, model down, unreadable
    file). Even that is now loud: WARNING with the traceback plus a
    ``silent_failures`` telemetry row, so a chronically-broken vision path
    is countable instead of anecdotal.
    """
    from prometheus.tools.base import ToolExecutionContext
    from prometheus.tools.builtin.vision import VisionInput, VisionTool

    tool = VisionTool()
    arguments = VisionInput(
        image_path=image_path, question="Describe this image in detail."
    )
    context = ToolExecutionContext(cwd=Path.cwd(), metadata={"provider": provider})

    try:
        result = await tool.execute(arguments, context)
    except Exception as exc:
        logger.warning(
            "Vision analysis failed for %s: %s", image_path, exc, exc_info=True
        )
        _record_media_failure("describe_image", exc, {"image_path": image_path})
        return None

    if not result.is_error and result.output:
        return result.output
    return None


def _record_media_failure(
    operation: str, exc: BaseException, ctx: dict[str, Any] | None = None
) -> None:
    """Best-effort ``silent_failures`` row for a swallowed media failure.

    Same shape as ``agent_loop``'s circuit-breaker reporting: never raises,
    so telemetry being unhappy can never turn a degraded image description
    into a crashed gateway.
    """
    try:
        from prometheus.telemetry.tracker import get_telemetry_handle

        tel = get_telemetry_handle()
        if tel is not None and hasattr(tel, "record_silent_failure"):
            tel.record_silent_failure(
                subsystem="media_services",
                operation=operation,
                exc=exc,
                context=ctx or {},
            )
    except Exception:  # pragma: no cover - telemetry must never mask the result
        logger.debug("media_services: record_silent_failure failed", exc_info=True)


async def transcribe_audio(audio_path: str) -> str | None:
    """Run audio through WhisperSTT for transcription. Returns None on failure."""
    try:
        from prometheus.tools.builtin.whisper_stt import WhisperSTTTool, WhisperSTTInput
        from prometheus.tools.base import ToolExecutionContext

        tool = WhisperSTTTool()
        result = await tool.execute(
            WhisperSTTInput(audio_path=audio_path),
            ToolExecutionContext(cwd=Path.cwd()),
        )
        if not result.is_error and result.output:
            return result.output
    except Exception as exc:
        logger.debug("Whisper STT unavailable: %s", exc)
    return None


def truncate_for_context(
    text: str,
    *,
    prometheus_config: dict[str, Any],
    system_prompt: str,
    tool_registry: Any,
    provider: Any = None,
) -> str:
    """Truncate extracted document text to fit the context window budget."""
    from prometheus.context.token_estimation import estimate_tokens

    doc_tokens = estimate_tokens(text)

    # Get server-detected context size if available, else use config
    ctx_limit = prometheus_config.get("context", {}).get(
        "effective_limit", 24000
    )
    server_ctx = getattr(provider, "server_context_size", None)
    if server_ctx:
        ctx_limit = min(ctx_limit, server_ctx)

    # Calculate actual overhead from system prompt + tool schemas
    prompt_tokens = estimate_tokens(system_prompt)
    import json
    schema_chars = len(json.dumps(tool_registry.list_schemas()))
    tool_tokens = schema_chars // 4  # rough char-to-token ratio

    reserved_output = prometheus_config.get("context", {}).get(
        "reserved_output", 2000
    )
    # overhead = system prompt + tools + output reserve + conversation buffer
    overhead = prompt_tokens + tool_tokens + reserved_output + 500
    max_doc_tokens = max(2000, ctx_limit - overhead)

    if doc_tokens > max_doc_tokens:
        char_limit = max_doc_tokens * 4
        text = text[:char_limit] + (
            f"\n\n[... truncated to fit context window "
            f"({max_doc_tokens} of {doc_tokens} tokens)]"
        )
        logger.info(
            "Truncated document from %d to ~%d tokens "
            "(context: %d, overhead: %d)",
            doc_tokens, max_doc_tokens, ctx_limit, overhead,
        )
    return text
