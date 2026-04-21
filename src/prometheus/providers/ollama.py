"""OllamaProvider — connects to the Ollama API.

Ollama exposes an OpenAI-compatible /v1/chat/completions endpoint as well
as its own /api/chat endpoint. This provider uses the OpenAI-compatible
path with `format: "json"` support for structured output.

Default target: http://localhost:11434
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


class OllamaProvider(ModelProvider):
    """Provider for Ollama's OpenAI-compatible API.

    Usage:
        provider = OllamaProvider(base_url="http://localhost:11434")
        async for event in provider.stream_message(request):
            ...

    With forced JSON output:
        provider = OllamaProvider(force_json=True)   # adds format="json"
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        force_json: bool = False,
        grammar: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._force_json = force_json
        self._grammar = grammar

    def set_grammar(self, grammar: str | None) -> None:
        """Set GBNF grammar for constrained decoding (llama.cpp extension)."""
        self._grammar = grammar

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        """Stream a response from Ollama with exponential-backoff retry."""
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
                    exc.response.status_code if hasattr(exc, "response") else None
                )
                retryable = status in RETRYABLE_STATUS_CODES if status else isinstance(
                    exc, (httpx.ConnectError, httpx.TimeoutException, ConnectionError)
                )
                if attempt >= MAX_RETRIES or not retryable:
                    raise
                delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                delay += random.uniform(0, delay * 0.25)
                log.warning(
                    "Ollama request failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, MAX_RETRIES + 1, delay, exc,
                )
                await asyncio.sleep(delay)

        if last_error is not None:
            raise last_error

    async def _call_once(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        """Single attempt to Ollama's /v1/chat/completions."""
        messages = _build_openai_messages(request)

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": True,
            "options": {"num_predict": request.max_tokens},
        }

        if self._force_json:
            payload["format"] = "json"

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
        elif self._grammar:
            # Only send grammar when tools aren't in the payload — with --jinja,
            # the server handles tool calling natively and grammar conflicts with it
            payload["grammar"] = self._grammar

        # Final safety: grammar + tools in same request = 400 from llama-server
        if "tools" in payload and payload["tools"]:
            payload.pop("grammar", None)

        url = f"{self._base_url}/v1/chat/completions"
        log.debug("POST %s model=%s messages=%d tools=%d grammar=%s",
                  url, request.model, len(messages),
                  len(payload.get("tools", [])), bool(payload.get("grammar")))

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

                    # Ollama puts usage in the final chunk under prompt_eval_count
                    if "prompt_eval_count" in chunk:
                        input_tokens = chunk.get("prompt_eval_count", 0)
                        output_tokens = chunk.get("eval_count", 0)

                    if "usage" in chunk:
                        u = chunk["usage"] or {}
                        input_tokens = u.get("prompt_tokens", input_tokens)
                        output_tokens = u.get("completion_tokens", output_tokens)

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
