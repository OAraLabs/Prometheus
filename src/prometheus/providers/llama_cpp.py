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


# The llama-server unit on the model box runs with
# ``--reasoning-budget-message "[REASONING_BUDGET_EXHAUSTED]"`` — the server
# injects this string into the reasoning stream right before it forces the
# end-of-thinking tag when ``--reasoning-budget N`` runs out. It is the
# deterministic in-band signal that thinking was truncated; if the unit's
# message ever changes, change this constant with it.
REASONING_BUDGET_MARKER = "[REASONING_BUDGET_EXHAUSTED]"


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
        # force-search (IGNITION): the grammar SOURCE — the enforcer + tool
        # schemas the boot grammar was generated from — so per-call
        # required/{tool:X} grammars are derived through the SAME generate
        # path (never a string hack on the boot grammar). Wired at the boot
        # sites right after set_grammar(); absent => required/{tool} fail loud.
        self._grammar_enforcer: Any | None = None
        self._grammar_tool_schemas: list[dict[str, Any]] | None = None
        self._derived_grammars: dict[str, str] = {}  # "required" | "tool:<name>"

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
        self._derived_grammars = {}  # force-search: derived grammars follow the boot grammar

    def set_grammar_source(self, enforcer: Any, tool_schemas: list[dict[str, Any]]) -> None:
        """force-search (IGNITION): hand over the grammar SOURCE (the enforcer +
        the schemas the boot grammar came from) so per-call ``required`` /
        ``{"tool": X}`` grammars are derived via the same
        ``enforcer.generate_grammar`` path — never a string edit of the boot
        grammar. Called at the boot sites right after ``set_grammar``."""
        self._grammar_enforcer = enforcer
        self._grammar_tool_schemas = list(tool_schemas)
        self._derived_grammars = {}

    def can_force_via_grammar(self, tool_choice: object) -> bool:
        """FIRST-ROUND FORCING (post-IGNITION follow-up 2): True when this
        provider can DETERMINISTICALLY enforce a forced directive via GBNF —
        i.e. it has a boot grammar AND a wired grammar source to derive the
        required / {tool:X} grammar from. The engine uses this to withhold
        native tools for the forced round: the grammar cannot be ignored by
        the server, unlike the OpenAI-shape function-forcing param, which the
        live llama-server build silently drops."""
        if not self._grammar:
            return False
        if not (tool_choice == "required" or isinstance(tool_choice, dict)):
            return False
        return self._grammar_enforcer is not None and self._grammar_tool_schemas is not None

    def _derived_grammar(self, cache_key: str, *, only_tool: str | None = None) -> str:
        """Generate-and-cache a constrained grammar via the enforcer.

        Cache is keyed by tool name (bounded by the registry: unknown names were
        already 400'd at ingress, and the enforcer ASSERTS by raising on a name
        outside its schemas — no re-validation here)."""
        cached = self._derived_grammars.get(cache_key)
        if cached is not None:
            return cached
        if self._grammar_enforcer is None or self._grammar_tool_schemas is None:
            raise RuntimeError(
                "tool_choice requires the grammar source, but set_grammar_source() was "
                "never wired for this provider — refusing to degrade a forced turn."
            )
        derived = self._grammar_enforcer.generate_grammar(
            self._grammar_tool_schemas,
            require_tool_use=True,
            only_tool=only_tool,
        )
        self._derived_grammars[cache_key] = derived
        return derived

    def _grammar_for(self, request: ApiMessageRequest) -> str | None:
        """force-search: SELECT the GBNF grammar for this per-call tool_choice.

        auto/absent -> the UNTOUCHED boot grammar object (identity, no
        regeneration — the dormant path); none / suppress_tools -> None
        (dropped — same as chat); required -> enforcer-generated tool-call-only
        root; {"tool": X} -> enforcer-generated single-alternative grammar
        admitting ONLY X. Unrecognized values raise — a forced turn must never
        silently degrade (same fail-loud contract as the cloud mapping)."""
        if not self._grammar:
            return None
        tc = getattr(request, "tool_choice", "auto")
        if getattr(request, "suppress_tools", False) or tc == "none":
            return None
        if tc == "auto" or tc is None:
            return self._grammar  # the boot grammar object itself — identity, not a copy
        if tc == "required":
            return self._derived_grammar("required")
        if isinstance(tc, dict) and isinstance(tc.get("tool"), str) and tc["tool"]:
            name = tc["tool"]
            return self._derived_grammar(f"tool:{name}", only_tool=name)
        raise ValueError(
            f"unmapped tool_choice {tc!r} for the llama.cpp grammar path — "
            "expected 'auto' | 'none' | 'required' | {'tool': <name>}. "
            "Refusing to degrade silently."
        )

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
            # force-search: the native tools path (tier-light + --jinja) carries the
            # same OpenAI-shape mapping the compat provider uses — auto stays the
            # exact former "auto" hardcode (byte-identical dormant path), required /
            # {tool:X} force natively, anything else raises (never degrades).
            from prometheus.providers.openai_compat import _native_tool_choice
            payload["tool_choice"] = _native_tool_choice(getattr(request, "tool_choice", "auto"))

        # Inject GBNF grammar only when tools aren't in the payload — with --jinja,
        # the server handles tool calling natively and grammar conflicts with it. force-search:
        # _grammar_for SELECTS per the per-call tool_choice (auto=boot, none/suppress=dropped,
        # required/{tool}=prose-dropped) — auto/none stay byte-identical to before.
        _selected_grammar = self._grammar_for(request)
        if _selected_grammar and "tools" not in payload:
            payload["grammar"] = _selected_grammar
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

    def _record_reasoning_budget_exhaustion(
        self,
        *,
        model: str,
        reasoning_chars: int,
        finish_reason: str | None,
        content_chars: int,
    ) -> bool:
        """Record one ``subsystem_runs`` row when the server truncated thinking.

        Fires when the configured ``--reasoning-budget-message`` marker
        (:data:`REASONING_BUDGET_MARKER`) appears in the accumulated
        reasoning stream. Outcome is ``"partial"`` — the call itself
        succeeded (content was still generated after the forced
        end-of-thinking), but the reasoning was cut. Distinct from
        ``_finalize_content``'s empty-content path, which covers the old
        unbounded shape (thinking ate the whole token budget and content
        never arrived). Returns whether a row was written.
        """
        try:
            from prometheus.telemetry.tracker import get_telemetry_handle

            tel = get_telemetry_handle()
            if tel is not None and hasattr(tel, "record_run"):
                tel.record_run(
                    subsystem="llama_cpp_provider",
                    operation="reasoning_budget_exhausted",
                    outcome="partial",
                    summary={
                        "marker": REASONING_BUDGET_MARKER,
                        "reasoning_chars": reasoning_chars,
                        "content_chars": content_chars,
                        "finish_reason": finish_reason,
                    },
                    model=model,
                )
                return True
        except Exception:
            log.exception(
                "llama.cpp: budget-exhaustion telemetry write failed"
            )
        return False

    def _finalize_content(
        self,
        accumulated_text: str,
        accumulated_reasoning: str,
        finish_reason: str | None,
        *,
        model: str,
        output_tokens: int,
        has_tool_calls: bool = False,
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

        if has_tool_calls:
            # Empty TEXT content on a native tool-call completion is the normal
            # shape — the tool_calls were accumulated separately and ride the
            # complete event (rider triage 2026-07-02: 516/560 "silent failures"
            # were this). Not a failure; and no reasoning fallback here, which
            # would inject chain-of-thought text beside the tool calls.
            return "", False

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

        # Reasoning-budget truncation: the server injects the configured
        # marker into the reasoning stream when --reasoning-budget runs out.
        # Record it (outcome="partial") so budget pressure is queryable —
        # coding-mode turns run thinking-on and need this observable.
        if accumulated_reasoning and REASONING_BUDGET_MARKER in accumulated_reasoning:
            log.warning(
                "llama.cpp reasoning budget exhausted (reasoning_chars=%d, "
                "content_chars=%d, finish=%s)",
                len(accumulated_reasoning), len(accumulated_text), finish_reason,
            )
            self._record_reasoning_budget_exhaustion(
                model=request.model,
                reasoning_chars=len(accumulated_reasoning),
                finish_reason=finish_reason,
                content_chars=len(accumulated_text),
            )

        # Empty-content handling: log + record + best-effort fallback to
        # reasoning_content so the call returns SOMETHING actionable instead
        # of silently swallowing the answer.
        final_text, _recorded = self._finalize_content(
            accumulated_text,
            accumulated_reasoning,
            finish_reason,
            model=request.model,
            output_tokens=output_tokens,
            has_tool_calls=bool(accumulated_tool_calls),
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
