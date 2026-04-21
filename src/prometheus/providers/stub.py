"""StubProvider — OpenAI-compatible HTTP provider for llama.cpp and Ollama.

Connects to any server that speaks the OpenAI /v1/chat/completions API.
Default target: llama.cpp at http://localhost:8080.
"""

from __future__ import annotations

import json
import logging
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

# Retry config
MAX_RETRIES = 3
BASE_DELAY = 1.0
MAX_DELAY = 30.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503}


def _build_openai_messages(request: ApiMessageRequest) -> list[dict[str, Any]]:
    """Convert ConversationMessages to OpenAI wire format, handling tool results."""
    result: list[dict[str, Any]] = []

    if request.system_prompt:
        result.append({"role": "system", "content": request.system_prompt})

    for msg in request.messages:
        # Pass through pre-formatted dicts (e.g. multimodal vision requests)
        if isinstance(msg, dict):
            result.append(msg)
            continue

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for block in msg.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })
            else:
                # ToolResultBlock → separate tool message in OpenAI format
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.tool_use_id,
                    "content": block.content,
                })

        if msg.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                entry["content"] = " ".join(text_parts)
            if tool_calls:
                entry["tool_calls"] = tool_calls
            result.append(entry)
        elif tool_results:
            # user turn that contains tool results → emit as tool messages
            result.extend(tool_results)
            if text_parts:
                result.append({"role": "user", "content": " ".join(text_parts)})
        else:
            result.append({"role": "user", "content": " ".join(text_parts)})

    return result


def _parse_assistant_message(choice: dict[str, Any]) -> ConversationMessage:
    """Parse an OpenAI choice dict into a ConversationMessage."""
    content_blocks: list[Any] = []
    message = choice.get("message", {})

    raw_content = message.get("content") or ""
    if raw_content:
        content_blocks.append(TextBlock(text=raw_content))

    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        content_blocks.append(
            ToolUseBlock(
                id=tc.get("id", f"toolu_{uuid4().hex}"),
                name=fn.get("name", ""),
                input=args,
            )
        )

    return ConversationMessage(role="assistant", content=content_blocks)


class StubProvider(ModelProvider):
    """OpenAI-compatible provider for llama.cpp / Ollama.

    Usage:
        provider = StubProvider(base_url="http://localhost:8080")
        async for event in provider.stream_message(request):
            ...
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        """Send request to llama.cpp, yield text deltas then complete event."""
        import asyncio
        import random

        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                async for event in self._call_once(request):
                    yield event
                return
            except Exception as exc:
                last_error = exc
                status = getattr(exc, "status_code", None) or (
                    exc.response.status_code
                    if hasattr(exc, "response")
                    else None
                )
                retryable = status in RETRYABLE_STATUS_CODES if status else isinstance(
                    exc, (httpx.ConnectError, httpx.TimeoutException, ConnectionError)
                )
                if attempt >= MAX_RETRIES or not retryable:
                    raise
                delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                delay += random.uniform(0, delay * 0.25)
                log.warning(
                    "Provider request failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, MAX_RETRIES + 1, delay, exc,
                )
                await asyncio.sleep(delay)

        if last_error is not None:
            raise last_error

    async def _call_once(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        """Single non-retried call to the completions endpoint."""
        messages = _build_openai_messages(request)

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "stream": True,
        }

        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", t.get("parameters", {})),
                    },
                }
                for t in request.tools
            ]
            payload["tool_choice"] = "auto"

        url = f"{self._base_url}/v1/chat/completions"
        log.debug("POST %s model=%s messages=%d", url, request.model, len(messages))

        accumulated_text = ""
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        input_tokens = 0
        output_tokens = 0

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    log.error("HTTP %d from %s: %s",
                              response.status_code, url, response.text[:500])
                    response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # Usage may appear in the final chunk
                    if "usage" in chunk:
                        u = chunk["usage"] or {}
                        input_tokens = u.get("prompt_tokens", 0)
                        output_tokens = u.get("completion_tokens", 0)

                    for choice in chunk.get("choices", []):
                        finish_reason = choice.get("finish_reason") or finish_reason
                        delta = choice.get("delta", {})

                        text = delta.get("content") or ""
                        if text:
                            accumulated_text += text
                            yield ApiTextDeltaEvent(text=text)

                        for tc in delta.get("tool_calls") or []:
                            idx = tc.get("index", 0)
                            if idx not in accumulated_tool_calls:
                                accumulated_tool_calls[idx] = {
                                    "id": tc.get("id", f"toolu_{uuid4().hex}"),
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            fn = tc.get("function", {})
                            if fn.get("name"):
                                accumulated_tool_calls[idx]["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                accumulated_tool_calls[idx]["function"]["arguments"] += fn["arguments"]

        # Build final message from accumulated state
        final_choice: dict[str, Any] = {
            "message": {
                "content": accumulated_text or None,
                "tool_calls": list(accumulated_tool_calls.values()) if accumulated_tool_calls else None,
            }
        }
        final_message = _parse_assistant_message(final_choice)

        yield ApiMessageCompleteEvent(
            message=final_message,
            usage=UsageSnapshot(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            stop_reason=finish_reason,
        )
