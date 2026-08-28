"""OpenAICompatProvider — works with any OpenAI-compatible API.

Covers: OpenAI, Google Gemini, xAI Grok, DeepSeek, Kimi (Moonshot),
GLM (Z.ai), MiMo (Xiaomi), and any server exposing /v1/chat/completions
(or a version-suffixed equivalent) with Bearer token auth.

For local servers without auth (llama.cpp, Ollama), use the existing
LlamaCppProvider or OllamaProvider instead.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator, Callable
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

# A base URL whose final path segment is a version tag (/v1, /v4, /v1beta …)
# already names the API root — appending another /v1 would 404. Matches the
# trailing segment only, so Gemini's /v1beta/openai base (final segment
# "openai") keeps the historical behavior.
_VERSIONED_BASE_RE = re.compile(r"/v\d+[a-z0-9]*$")


def _chat_completions_url(base_url: str) -> str:
    """Join a provider base URL with the chat-completions path.

    Rules (CLOUD EXPANSION — GLM's ``https://api.z.ai/api/paas/v4`` base):

    * base ends with a version segment (``/v1``, ``/v4``, ``/v1beta``) →
      append ``/chat/completions`` — the base already IS the API root.
    * anything else → append ``/v1/chat/completions`` (historical behavior,
      unchanged for bare hosts like ``https://api.deepseek.com`` and for
      Gemini's ``/v1beta/openai`` base).
    """
    base = base_url.rstrip("/")
    if _VERSIONED_BASE_RE.search(base):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _native_tool_choice(tool_choice: object) -> object:
    """Map the engine's per-call tool_choice to the OpenAI-compatible native param.

    force-search (IGNITION): replaces the historical hardcode — for "auto" it
    returns the exact same ``"auto"`` string, so the default path serializes
    byte-identically to before. ``required`` -> ``"required"``; ``{"tool": X}``
    -> ``{"type": "function", "function": {"name": X}}``.

    Fail-loud guard: any other value raises. "none" never reaches here on the
    engine path (run_loop sends no tools, so this block is skipped). A silent
    degrade-to-auto or dropped key would defeat a forced turn invisibly.
    """
    if tool_choice == "auto" or tool_choice is None:
        return "auto"
    if tool_choice == "required":
        return "required"
    if isinstance(tool_choice, dict) and isinstance(tool_choice.get("tool"), str) and tool_choice["tool"]:
        return {"type": "function", "function": {"name": tool_choice["tool"]}}
    raise ValueError(
        f"unmapped tool_choice {tool_choice!r} for the OpenAI-compatible provider — "
        "expected 'auto' | 'required' | {'tool': <name>} (with tools present; "
        "'none' is expressed by sending no tools). Refusing to degrade silently."
    )


def _coerce_int(value: object) -> int | None:
    """Non-negative int, or None. Providers occasionally send null/strings."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 else None
    if isinstance(value, str):
        try:
            n = int(value.strip())
        except ValueError:
            return None
        return n if n >= 0 else None
    return None


def _parse_cache_usage(usage: dict[str, Any]) -> tuple[int | None, int | None]:
    """Extract (cached_input_tokens, cache_write_tokens) from a usage object.

    Providers disagree on where prompt-cache accounting lives, and an
    OpenAI-*compatible* endpoint is not obliged to match OpenAI exactly — xAI,
    DeepSeek and friends each picked their own spelling. Rather than pin one
    shape, probe the known ones in order of specificity and return None when a
    provider says nothing at all.

    None is deliberately NOT folded into 0: "this provider reports no cache
    information" and "the cache was cold this round" are different findings,
    and conflating them would make an unsupported provider look like a 0%
    hit rate forever.
    """
    if not isinstance(usage, dict):
        return None, None

    cached: int | None = None
    write: int | None = None

    # OpenAI / xAI / most compat servers: nested details object.
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = _coerce_int(details.get("cached_tokens"))
    write_details = usage.get("completion_tokens_details")
    if isinstance(write_details, dict) and write is None:
        # Some servers report cache writes here; harmless when absent.
        write = _coerce_int(write_details.get("cache_write_tokens"))

    # DeepSeek-style flat keys (prompt_cache_hit_tokens / _miss_tokens).
    if cached is None:
        cached = _coerce_int(usage.get("prompt_cache_hit_tokens"))
    # Anthropic-style, in case a compat shim passes them straight through.
    if cached is None:
        cached = _coerce_int(usage.get("cache_read_input_tokens"))
    if write is None:
        write = _coerce_int(usage.get("cache_creation_input_tokens"))
    # Last resort: a flat cached_tokens on the usage object itself.
    if cached is None:
        cached = _coerce_int(usage.get("cached_tokens"))

    return cached, write


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
        api_key: "str | Callable[[], str | None]",
        model: str = "",
        default_max_tokens: int = 4096,
        timeout: float = 120.0,
        provider_name: str = "",
        vision: bool = False,
    ) -> None:
        # DECLARED by the catalog for the configured MODEL, never inferred here. One
        # class serves openai, gemini, xai, deepseek, kimi, glm, mimo and qwen, and
        # whether a given model takes pictures is a fact about that model, not about
        # the wire format they share. Default False: absence is not permission.
        self.supports_vision = bool(vision)
        # WHICH OpenAI-compatible service this instance talks to — openai,
        # gemini, xai, deepseek, kimi, glm, mimo, qwen. One class serves all
        # of them, so without this the only signal left is the class name,
        # and `_provider_name_for_telemetry`'s fallback maps every one of
        # them to "openai". That mislabelled every qwen call (the busiest
        # model on this box) as openai in telemetry, and made golden-trace
        # flagging depend on "openai" happening to be in _CLOUD_PROVIDERS
        # rather than on the real provider being classified at all.
        #
        # Anything that consumes this MUST classify every name the registry
        # can produce — see the drift guard in tests/test_wiring.py.
        self.provider_name = provider_name
        self._base_url = base_url.rstrip("/")
        # api_key may be a static string OR a zero-arg callable resolved per
        # request. The callable form powers rotating credentials (xAI SuperGrok
        # OAuth) where the bearer refreshes mid-session and must not be frozen
        # at construction — providers here are long-lived and cached.
        self._api_key = api_key
        self._model = model
        self._default_max_tokens = default_max_tokens
        self._timeout = timeout

    def _resolve_bearer(self) -> str:
        """Return the current bearer, resolving a callable credential if given."""
        key = self._api_key() if callable(self._api_key) else self._api_key
        if not key:
            raise ValueError(
                "no bearer available for this request — the credential source "
                "returned empty (expired OAuth token with no fallback key?)"
            )
        return key

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        """Stream a response with exponential-backoff retry."""
        import asyncio
        import random

        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            # Whether this attempt has handed anything to the consumer yet.
            emitted = False
            try:
                async for event in self._call_once(request):
                    emitted = True
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
                # A retry re-runs the request from scratch. Anything already yielded is
                # already on the consumer's screen, so replaying over it appends a SECOND,
                # possibly contradictory answer to the first (issue #293). Once output has
                # left, the only honest move is to fail. Retries stay fully available for
                # the common case: a failure before the first event.
                if emitted or attempt >= MAX_RETRIES or not retryable:
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

    async def _call_once(  # noqa: C901 - stream parsing is inherently branchy
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        """Single attempt to /v1/chat/completions (or /chat/completions)."""
        messages = _build_openai_messages(request, allow_images=self.supports_vision)

        payload: dict[str, Any] = {
            "model": request.model or self._model,
            "messages": messages,
            "max_tokens": request.max_tokens or self._default_max_tokens,
            "stream": True,
            # SPRINT-loop-envelope (F1): OpenAI-compat servers only emit the
            # usage chunk when asked — without this the stream parser's
            # ``"usage" in chunk`` branch never fires and every UsageSnapshot
            # is structurally 0/0. Accounting-only: generation is unaffected,
            # and servers that predate stream_options ignore unknown fields.
            "stream_options": {"include_usage": True},
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
            payload["tool_choice"] = _native_tool_choice(getattr(request, "tool_choice", "auto"))

        # Build URL — version-suffixed bases (/v1, /v4, /v1beta) already name
        # the API root; everything else gets the /v1 prefix.
        url = _chat_completions_url(self._base_url)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._resolve_bearer()}",
        }

        log.debug(
            "POST %s model=%s messages=%d",
            url, payload["model"], len(messages),
        )

        accumulated_text = ""
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        cached_input: int | None = None
        cache_write: int | None = None
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
                        cached_input, cache_write = _parse_cache_usage(u)

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
        final_message, dropped_malformed = _parse_assistant_message(
            final_choice, model=request.model
        )

        yield ApiMessageCompleteEvent(
            message=final_message,
            usage=UsageSnapshot(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input,
                cache_write_tokens=cache_write,
            ),
            stop_reason=finish_reason,
            dropped_malformed=dropped_malformed,
        )
