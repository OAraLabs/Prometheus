"""AnthropicProvider — Anthropic Messages API fallback.

Uses httpx directly to avoid a hard dependency on the anthropic SDK.
Set ANTHROPIC_API_KEY in your environment or pass api_key= explicitly.

Supports:
  - Streaming responses
  - Tool use (native Anthropic format)
  - Prompt caching (cache_control headers on long system prompts)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator
from uuid import uuid4

import httpx

from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiStreamEvent,
    ApiTextDeltaEvent,
    ModelProvider,
)

log = logging.getLogger(__name__)

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_DEFAULT_MODEL = "claude-sonnet-4-6"
_ANTHROPIC_VERSION = "2023-06-01"
_MAX_RETRIES = 3
_BASE_DELAY = 1.0
_MAX_DELAY = 30.0
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}

# Enable prompt caching when system prompt exceeds this many chars
_CACHE_THRESHOLD_CHARS = 1024


class AnthropicProvider(ModelProvider):
    """Anthropic Messages API provider.

    Usage:
        provider = AnthropicProvider()                         # reads ANTHROPIC_API_KEY
        provider = AnthropicProvider(api_key="sk-ant-...")
        provider = AnthropicProvider(prompt_caching=True)      # cache long system prompts
    """

    api_enforced_structure: bool = True

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        timeout: float = 120.0,
        prompt_caching: bool = False,
        base_url: str = _ANTHROPIC_API_URL,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Anthropic API key not provided. "
                "Set ANTHROPIC_API_KEY or pass api_key= to AnthropicProvider."
            )
        self._model = model
        self._timeout = timeout
        self._prompt_caching = prompt_caching
        self._base_url = base_url.rstrip("/")

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        """Stream a response from the Anthropic API."""
        import asyncio
        import random

        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                async for event in self._call_once(request):
                    yield event
                return
            except Exception as exc:
                last_error = exc
                status = getattr(exc, "status_code", None) or (
                    exc.response.status_code if hasattr(exc, "response") else None
                )
                retryable = status in _RETRYABLE_STATUS_CODES if status else isinstance(
                    exc, (httpx.ConnectError, httpx.TimeoutException, ConnectionError)
                )
                if attempt >= _MAX_RETRIES or not retryable:
                    raise
                delay = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
                delay += random.uniform(0, delay * 0.25)
                log.warning(
                    "Anthropic request failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, _MAX_RETRIES + 1, delay, exc,
                )
                await asyncio.sleep(delay)

        if last_error is not None:
            raise last_error

    async def _call_once(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        """Single streaming call to the Anthropic Messages API."""
        messages = _build_anthropic_messages(request.messages)

        payload: dict[str, Any] = {
            "model": request.model or self._model,
            "max_tokens": request.max_tokens,
            "messages": messages,
            "stream": True,
        }

        # System prompt with optional prompt caching
        if request.system_prompt:
            if (
                self._prompt_caching
                and len(request.system_prompt) >= _CACHE_THRESHOLD_CHARS
            ):
                payload["system"] = [
                    {
                        "type": "text",
                        "text": request.system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                payload["system"] = request.system_prompt

        # Tools in Anthropic native format
        if request.tools:
            payload["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("input_schema", t.get("parameters", {})),
                }
                for t in request.tools
                if t.get("name")
            ]
            payload["tool_choice"] = {"type": "auto"}

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        if self._prompt_caching:
            headers["anthropic-beta"] = "prompt-caching-2024-07-31"

        url = f"{self._base_url}/messages" if not self._base_url.endswith("/messages") else self._base_url

        log.debug("POST %s model=%s messages=%d", url, payload["model"], len(messages))

        # Streaming state
        content_blocks: list[Any] = []
        current_block: dict[str, Any] | None = None
        input_tokens = 0
        output_tokens = 0
        stop_reason: str | None = None

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if not data:
                        continue

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type", "")

                    if etype == "message_start":
                        usage = event.get("message", {}).get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)

                    elif etype == "content_block_start":
                        current_block = event.get("content_block", {})

                    elif etype == "content_block_delta":
                        delta = event.get("delta", {})
                        dtype = delta.get("type", "")
                        if dtype == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                if current_block is not None:
                                    current_block.setdefault("text", "")
                                    current_block["text"] += text
                                yield ApiTextDeltaEvent(text=text)
                        elif dtype == "input_json_delta":
                            partial = delta.get("partial_json", "")
                            if current_block is not None:
                                current_block.setdefault("partial_json", "")
                                current_block["partial_json"] += partial

                    elif etype == "content_block_stop":
                        if current_block:
                            content_blocks.append(current_block)
                        current_block = None

                    elif etype == "message_delta":
                        delta = event.get("delta", {})
                        stop_reason = delta.get("stop_reason") or stop_reason
                        usage = event.get("usage", {})
                        output_tokens = usage.get("output_tokens", output_tokens)

        # Build final ConversationMessage from collected blocks
        msg_content: list[Any] = []
        for block in content_blocks:
            btype = block.get("type", "")
            if btype == "text":
                text = block.get("text", "")
                if text:
                    msg_content.append(TextBlock(text=text))
            elif btype == "tool_use":
                raw_input = block.get("input") or {}
                if not isinstance(raw_input, dict):
                    partial = block.get("partial_json", "{}")
                    try:
                        raw_input = json.loads(partial) if partial else {}
                    except json.JSONDecodeError:
                        raw_input = {}
                msg_content.append(
                    ToolUseBlock(
                        id=block.get("id", f"toolu_{uuid4().hex}"),
                        name=block.get("name", ""),
                        input=raw_input,
                    )
                )

        final_message = ConversationMessage(role="assistant", content=msg_content)
        yield ApiMessageCompleteEvent(
            message=final_message,
            usage=UsageSnapshot(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            stop_reason=stop_reason,
        )


def _build_anthropic_messages(
    messages: list[ConversationMessage],
) -> list[dict[str, Any]]:
    """Convert ConversationMessages to Anthropic API message format."""
    from prometheus.engine.messages import ToolResultBlock

    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "assistant":
            content_list: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    content_list.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUseBlock):
                    content_list.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            result.append({"role": "assistant", "content": content_list})

        elif msg.role == "user":
            content_list = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    content_list.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolResultBlock):
                    content_list.append({
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": block.content,
                        "is_error": block.is_error,
                    })
            result.append({"role": "user", "content": content_list})

    return result
