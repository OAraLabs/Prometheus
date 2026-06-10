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


class EmptyCompletionError(RuntimeError):
    """Raised inside the provider to mark an empty-content completion in telemetry.

    Never propagated to callers — we synthesise it solely as the
    ``exc`` argument to ``record_silent_failure`` so the row in
    ``silent_failures`` is type-tagged and findable.
    """


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
        suppress_thinking: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._grammar = grammar
        # When True (the default), every completion sends BOTH
        # chat_template_kwargs.thinking=False and .enable_thinking=False so
        # whichever toggle the loaded model's chat template responds to gets
        # suppressed. Gemma's template responds to ``thinking``, Qwen's
        # responds to ``enable_thinking`` — sending both is model-agnostic.
        # Overridable per call via ApiMessageRequest.suppress_thinking.
        # Backstory: gemma4-26b's default template hard-codes a thought-channel
        # prefill that, with no flag set, makes the model burn its whole
        # token budget thinking and emit empty ``content`` — silently
        # breaking the daily briefing, the memory extractor, and a large
        # share of tool calls until this knob shipped.
        self._suppress_thinking = suppress_thinking
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

    def _effective_suppress(self, request: ApiMessageRequest) -> bool:
        """Per-call override wins; otherwise fall back to provider default."""
        if request.suppress_thinking is None:
            return self._suppress_thinking
        return request.suppress_thinking

    def _build_request_payload(self, request: ApiMessageRequest) -> dict[str, Any]:
        """Build the /v1/chat/completions JSON body.

        Factored out of ``_call_once`` so the thinking-suppression behaviour
        is unit-testable without mocking httpx. Preserves any existing
        chat_template_kwargs in case a future field on ApiMessageRequest
        carries caller-supplied entries (today only suppress_thinking exists).
        """
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
        if "tools" in payload and payload["tools"]:
            payload.pop("grammar", None)

        # Thinking suppression — send BOTH keys so the fix is model-agnostic
        # across gemma (responds to ``thinking``) and Qwen (responds to
        # ``enable_thinking``).  See __init__ docstring for the backstory.
        if self._effective_suppress(request):
            existing = payload.get("chat_template_kwargs") or {}
            merged = dict(existing)
            merged.setdefault("thinking", False)
            merged.setdefault("enable_thinking", False)
            payload["chat_template_kwargs"] = merged

        return payload

    def _finalize_content(
        self,
        accumulated_text: str,
        accumulated_reasoning: str,
        finish_reason: str | None,
        *,
        model: str,
        output_tokens: int,
    ) -> tuple[str, bool]:
        """Decide the final content and whether a silent failure was recorded.

        Returns ``(final_text, recorded_failure)`` so the caller can yield a
        terminal text-delta if a fallback was applied. When ``content`` is
        non-empty, returns it unchanged. When it's empty:

          - logs a warning,
          - records to telemetry.silent_failures (via the module-level handle)
            so the empty completion is OBSERVABLE — previously these were
            silently returned and the memory extractor's "0 chars" runs
            looked like quiet successes,
          - falls back to ``reasoning_content`` if present (best-effort
            recovery — the answer was thought, just not emitted),
          - if even reasoning is empty AND finish_reason == "length", surfaces
            "budget exhausted on thinking" explicitly in the recorded context.
        """
        if accumulated_text and accumulated_text.strip():
            return accumulated_text, False

        budget_exhausted = (
            finish_reason == "length"
            and not (accumulated_reasoning and accumulated_reasoning.strip())
        )
        used_fallback = bool(accumulated_reasoning and accumulated_reasoning.strip())

        log.warning(
            "llama.cpp returned empty content (finish=%s, reasoning_chars=%d, "
            "output_tokens=%d) — %s",
            finish_reason,
            len(accumulated_reasoning or ""),
            output_tokens,
            "falling back to reasoning_content" if used_fallback
            else "no reasoning to recover; budget exhausted on thinking"
            if budget_exhausted
            else "no content and no reasoning to recover",
        )

        recorded = False
        try:
            from prometheus.telemetry.tracker import get_telemetry_handle

            tel = get_telemetry_handle()
            if tel is not None and hasattr(tel, "record_silent_failure"):
                exc = EmptyCompletionError(
                    f"finish_reason={finish_reason}, "
                    f"reasoning_chars={len(accumulated_reasoning or '')}"
                )
                tel.record_silent_failure(
                    "llama_cpp_provider",
                    "stream_message",
                    exc,
                    context={
                        "model": model,
                        "finish_reason": finish_reason,
                        "output_tokens": output_tokens,
                        "reasoning_chars": len(accumulated_reasoning or ""),
                        "used_reasoning_fallback": used_fallback,
                        "budget_exhausted_on_thinking": budget_exhausted,
                    },
                )
                recorded = True
        except Exception:
            log.exception("llama.cpp: telemetry record_silent_failure failed")

        if used_fallback:
            return accumulated_reasoning, recorded
        return "", recorded

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
        payload = self._build_request_payload(request)
        url = f"{self._base_url}/v1/chat/completions"
        log.debug("POST %s model=%s messages=%d grammar=%s suppress_thinking=%s",
                  url, request.model, len(payload["messages"]),
                  bool(payload.get("grammar")), self._effective_suppress(request))

        accumulated_text = ""
        accumulated_reasoning = ""
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

                        # Capture reasoning_content deltas separately. Previously
                        # dropped — that's why empty-content runs hit downstream
                        # subsystems silently. We use this both as a recovery
                        # fallback and to flag budget-exhausted-on-thinking
                        # situations in telemetry.
                        rtext = delta.get("reasoning_content") or ""
                        if rtext:
                            accumulated_reasoning += rtext

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

        # Empty-content handling: log + record + best-effort fallback to
        # reasoning_content so the call returns SOMETHING actionable instead
        # of silently swallowing the answer.
        final_text, _recorded = self._finalize_content(
            accumulated_text,
            accumulated_reasoning,
            finish_reason,
            model=request.model,
            output_tokens=output_tokens,
        )

        # When a fallback was applied, emit one terminal text delta so
        # streaming consumers (the briefing job accumulates from deltas, not
        # from the complete event) actually see the recovered text.
        if final_text and not accumulated_text:
            yield ApiTextDeltaEvent(text=final_text)

        final_choice: dict[str, Any] = {
            "message": {
                "content": final_text or None,
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
            ),
            stop_reason=finish_reason,
            dropped_malformed=dropped_malformed,
        )
