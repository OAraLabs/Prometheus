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
    """
    try:
        from prometheus.tools.builtin.vision import VisionTool, VisionInput
        from prometheus.tools.base import ToolExecutionContext

        tool = VisionTool()
        result = await tool.execute(
            VisionInput(image_path=image_path, question="Describe this image in detail."),
            ToolExecutionContext(cwd=Path.cwd(), metadata={"provider": provider}),
        )
        if not result.is_error and result.output:
            return result.output
    except Exception as exc:
        logger.debug("Vision analysis unavailable: %s", exc)
    return None


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
