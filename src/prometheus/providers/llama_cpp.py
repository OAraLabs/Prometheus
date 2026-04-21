"""LlamaCppProvider — connects to llama-server's OpenAI-compatible API.

Target: llama-server at http://localhost:8080 (or any base_url you configure).

Differences from StubProvider:
  - Accepts an optional GBNF grammar string for constrained decoding
  - Sets model to "local" when not specified (llama-server ignores the model field)
  - Passes grammar via the `grammar` request field when provided
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


class LlamaCppProvider(ModelProvider):
    """OpenAI-compatible provider targeting llama-server.

    Usage:
        provider = LlamaCppProvider(base_url="http://localhost:8080")
        async for event in provider.stream_message(request):
            ...

    With constrained grammar:
        provider = LlamaCppProvider(
            base_url="http://localhost:8080",
            grammar=enforcer.generate_grammar(tool_schemas),
        )
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 120.0,
        grammar: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._grammar = grammar
        self.detected_model: str | None = None
        self.server_context_size: int | None = None

    async def detect_context_size(self) -> int | None:
        """Query the server's actual context size via /props.

        llama.cpp's /props endpoint returns default_generation_settings.n_ctx.
        Returns the context size in tokens, or None if detection fails.
        """
        url = f"{self._base_url}/props"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                props = resp.json()

            dgs = props.get("default_generation_settings", {})
            n_ctx = dgs.get("n_ctx")
            if n_ctx and isinstance(n_ctx, int):
                self.server_context_size = n_ctx
                log.info("Detected server context size: %d tokens", n_ctx)
                return n_ctx

            log.info("Could not read n_ctx from /props")
            return None
        except Exception as exc:
            log.warning("Context size detection failed: %s", exc)
            return None

    async def detect_vision(self) -> bool:
        """Check if llama.cpp was started with --mmproj.

        The /props endpoint includes a "multimodal" field when a vision
        projector is loaded. Older versions nest it under
        "default_generation_settings".
        """
        url = f"{self._base_url}/props"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                props = resp.json()

            multimodal = props.get("multimodal", False)
            if not multimodal:
                dgs = props.get("default_generation_settings", {})
                multimodal = dgs.get("multimodal", False)

            self.supports_vision = bool(multimodal)
            log.info("Vision detection: multimodal=%s (endpoint=%s)",
                     self.supports_vision, self._base_url)
            return self.supports_vision

        except (httpx.HTTPError, httpx.ConnectError) as exc:
            log.warning("Vision detection failed (server unreachable): %s", exc)
            self.supports_vision = False
            return False
        except (KeyError, ValueError) as exc:
            log.warning("Vision detection failed (bad response): %s", exc)
            self.supports_vision = False
            return False

    async def detect_loaded_model(self) -> str | None:
        """Query /v1/models to discover the model actually loaded on the server.

        Returns the model id string, or None if the endpoint is unreachable.
        Caches the result in ``self.detected_model``.
        """
        url = f"{self._base_url}/v1/models"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                body = resp.json()
                # OpenAI-compatible: data[0].id
                models = body.get("data", [])
                if models:
                    self.detected_model = models[0].get("id")
                    log.info("Detected loaded model: %s", self.detected_model)
                    return self.detected_model
        except Exception as exc:
            log.warning("Could not detect model from %s: %s", url, exc)
        return None

    def set_grammar(self, grammar: str | None) -> None:
        """Update the GBNF grammar used for constrained decoding."""
        self._grammar = grammar

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        """Stream a response from llama-server with exponential-backoff retry."""
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
                    "llama.cpp request failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, MAX_RETRIES + 1, delay, exc,
                )
                await asyncio.sleep(delay)

        if last_error is not None:
            raise last_error

    async def _call_once(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        """Single attempt to /v1/chat/completions."""
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

        # Inject GBNF grammar only when tools aren't in the payload — with --jinja,
        # the server handles tool calling natively and grammar conflicts with it
        if self._grammar and "tools" not in payload:
            payload["grammar"] = self._grammar

        # Final safety: grammar + tools in same request = 400 from llama-server
        if "tools" in payload and payload["tools"]:
            payload.pop("grammar", None)

        url = f"{self._base_url}/v1/chat/completions"
        log.debug("POST %s model=%s messages=%d grammar=%s",
                  url, request.model, len(messages), bool(payload.get("grammar")))

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
