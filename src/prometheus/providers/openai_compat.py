"""OpenAICompatProvider — works with any OpenAI-compatible API.

Covers: OpenAI, Google Gemini, xAI Grok, and any server exposing
/v1/chat/completions with Bearer token auth.

For local servers without auth (llama.cpp, Ollama), use the existing
LlamaCppProvider or OllamaProvider instead.
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
from prometheus.providers.stub import (
    MAX_DELAY,
    MAX_RETRIES,
    RETRYABLE_STATUS_CODES,
    BASE_DELAY,
    _build_openai_messages,
    _parse_assistant_message,
)

log = logging.getLogger(__name__)


class OpenAICompatProvider(ModelProvider):
    """Provider for any OpenAI-compatible chat completions API.

    Works with: OpenAI, Google Gemini (OpenAI compat layer), xAI Grok,
    vLLM, LiteLLM, or any server exposing /v1/chat/completions.

    Usage:
        provider = OpenAICompatProvider(
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
            model="gpt-4o",
        )
        async for event in provider.stream_message(request):
            ...
    """

    api_enforced_structure: bool = True

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "",
        default_max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._default_max_tokens = default_max_tokens
        self._timeout = timeout

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        """Stream a response with exponential-backoff retry."""
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
                    "OpenAI-compat request failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, MAX_RETRIES + 1, delay, exc,
                )
                await asyncio.sleep(delay)

        if last_error is not None:
            raise last_error

    async def _call_once(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        """Single attempt to /v1/chat/completions (or /chat/completions)."""
        messages = _build_openai_messages(request)

        payload: dict[str, Any] = {
            "model": request.model or self._model,
            "messages": messages,
            "max_tokens": request.max_tokens or self._default_max_tokens,
            "stream": True,
        }

        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.get("function", {}).get("name", t.get("name", "")),
                        "description": t.get("function", {}).get(
                            "description", t.get("description", "")
                        ),
                        "parameters": t.get("function", {}).get(
                            "parameters", t.get("input_schema", t.get("parameters", {}))
                        ),
                    },
                }
                for t in request.tools
            ]
            payload["tool_choice"] = "auto"

        # Build URL — handle base URLs that already end with /v1
        if self._base_url.endswith("/v1"):
            url = f"{self._base_url}/chat/completions"
        else:
            url = f"{self._base_url}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        log.debug(
            "POST %s model=%s messages=%d",
            url, payload["model"], len(messages),
        )

        accumulated_text = ""
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        input_tokens = 0
        output_tokens = 0

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST", url, json=payload, headers=headers
            ) as response:
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
