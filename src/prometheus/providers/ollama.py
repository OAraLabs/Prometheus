"""OllamaProvider — connects to the Ollama API.

Ollama exposes an OpenAI-compatible /v1/chat/completions endpoint as well
as its own /api/chat endpoint. This provider uses the OpenAI-compatible
path with `format: "json"` support for structured output.

Default target: http://localhost:11434

Thinking models (WP-X.35). Ollama turns thinking ON for any model whose
``/api/show`` capabilities list ``thinking`` unless the request says otherwise,
and its OpenAI-compatible route streams the thought as ``delta.reasoning``.
The provider asks ``/api/show`` once per model and, only for a model that lists
``thinking``, sends ``reasoning_effort: "none"`` unless the call opted in
(``ApiMessageRequest.suppress_thinking is False``). Every other model — and any
model whose capabilities could not be read — gets exactly the request it got
before. Reasoning is kept as a thinking block, never as reply text.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator
from uuid import uuid4

import httpx

from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.retry import stream_with_retry
from prometheus.providers.llama_cpp import EmptyCompletionError
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiStreamEvent,
    ApiTextDeltaEvent,
    ModelProvider,
)
from prometheus.providers.stub import (
    MAX_RETRIES,
    RETRYABLE_STATUS_CODES,
    _build_openai_messages,
    _parse_assistant_message,
)

log = logging.getLogger(__name__)

# The /api/show capability that marks a model with a thinking channel; the same
# spelling from Ollama 0.23.0 to 0.34.4 (types/model/capability.go).
_THINKING_CAPABILITY = "thinking"
# The one reasoning_effort value that turns thinking OFF on every Ollama from
# 0.23.0 on ("none" -> think=false, openai/openai.go). The levels differ by
# version ("minimal" is a 400 on 0.23.0), so no other value is ever sent. /v1
# has no ``think`` field: a top-level ``think`` is dropped without an error.
_THINKING_OFF = "none"
# gpt-oss goes through the harmony parser, which keeps its analysis channel
# whatever the request says: "none" cannot turn its reasoning off, so a retry
# with thinking off would only repeat the first attempt.
_HARMONY_FAMILIES = frozenset({"gptoss", "gpt-oss"})
_SHOW_TIMEOUT_S = 10.0


def _is_cloud_model(name: str) -> bool:
    """An Ollama cloud model by name (last ``:`` part ``cloud`` or ending in
    ``-cloud``, case-insensitive): the local server proxies its /api/show to
    ollama.com, so it is never asked (docs/contracts/hooks.md, "Ollama cloud
    models")."""
    last = name.rsplit(":", 1)[-1].lower()
    return last == "cloud" or last.endswith("-cloud")


class _ModelThinking:
    """What /api/show said about one model. Only a definitive answer is built:
    an unreadable /api/show leaves the model unknown, never 'not thinking'."""

    __slots__ = ("capable", "harmony")

    def __init__(self, *, capable: bool, harmony: bool) -> None:
        self.capable = capable
        self.harmony = harmony


class _Attempt:
    """One streamed completion's accumulated state."""

    def __init__(self) -> None:
        self.text = ""
        self.reasoning = ""
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def reasoning_only(self) -> bool:
        """Thought, but produced no answer: no text, no tool call. The budget
        went to thinking (finish "length") or the model stopped mid-thought."""
        return (not self.text.strip() and not self.tool_calls
                and bool(self.reasoning.strip()))


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
        # Definitive /api/show answers only, by model name. Not named
        # ``_suppress_thinking``: the loop's telemetry envelope reads that
        # attribute into subsystem_runs.thinking, and one Ollama serves many
        # models, so a provider-wide value would be wrong for most of them.
        self._thinking: dict[str, _ModelThinking] = {}

    def set_grammar(self, grammar: str | None) -> None:
        """Set GBNF grammar for constrained decoding (llama.cpp extension)."""
        self._grammar = grammar

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        """Stream a response from Ollama with exponential-backoff retry."""

        async for event in stream_with_retry(
            lambda: self._call_once(request),
            retryable_status=RETRYABLE_STATUS_CODES,
            label="Ollama",
            max_retries=MAX_RETRIES,
        ):
            yield event

    async def _call_once(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        """Single attempt to Ollama's /v1/chat/completions."""
        # Ollama has no mmproj probe, so `supports_vision` stays at the class
        # default (False) unless something sets it — an ImageBlock then raises
        # rather than being silently paraphrased. Same threading as llama_cpp.
        messages = _build_openai_messages(request, allow_images=self.supports_vision)

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": True,
            # The OpenAI-compatible endpoint reads the top-level max_tokens (and
            # maps it to num_predict itself); its request has no ``options``
            # field, so ``options.num_predict`` was dropped silently and every
            # turn ran unbounded (measured on 0.23.0: options.num_predict=5 ->
            # 150 tokens, finish "stop"; max_tokens=5 -> 5 tokens, "length").
            "max_tokens": request.max_tokens,
            # Ollama's OpenAI-compatible endpoint sends the usage chunk only
            # when asked (measured on 0.23.0: without this the stream carries
            # no ``usage`` at all), so every ollama round recorded 0/0 tokens.
            # The same field llama_cpp and openai_compat send. Accounting-only:
            # generation is unaffected, and a server that predates
            # stream_options ignores the unknown field and simply sends none.
            "stream_options": {"include_usage": True},
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

        # Decided per model, after the messages are built (an unsendable
        # message still refuses before any request). Nothing is added unless
        # /api/show says the model thinks, so every other request is the one
        # this provider always sent.
        thinking = await self._thinking_support(request.model)
        thinking_on = False
        if thinking is not None and thinking.capable:
            if request.suppress_thinking is False:
                # The caller opted in (coding sessions do): send nothing, which
                # is Ollama's own default for a thinking model — thinking on.
                thinking_on = True
            else:
                payload["reasoning_effort"] = _THINKING_OFF

        url = f"{self._base_url}/v1/chat/completions"
        log.debug("POST %s model=%s messages=%d tools=%d grammar=%s reasoning_effort=%s",
                  url, request.model, len(messages),
                  len(payload.get("tools", [])), bool(payload.get("grammar")),
                  payload.get("reasoning_effort"))

        attempt = _Attempt()
        async for event in self._stream(url, payload, attempt):
            yield event

        if attempt.reasoning_only:
            # The whole budget went to thinking. Retried once with thinking off
            # when it was on and "none" can turn it off (not gpt-oss). Nothing
            # but whitespace text was streamed, so the retry replays no answer.
            retry = thinking_on and thinking is not None and not thinking.harmony
            self._record_reasoning_only(request.model, attempt, retried=retry)
            if retry:
                first = attempt
                attempt = _Attempt()
                async for event in self._stream(
                    url, {**payload, "reasoning_effort": _THINKING_OFF}, attempt,
                ):
                    yield event
                if attempt.reasoning_only:
                    self._record_reasoning_only(request.model, attempt, retried=False)
                # Both attempts were really spent; the thought is kept, the
                # answer is the retry's.
                attempt.input_tokens += first.input_tokens
                attempt.output_tokens += first.output_tokens
                attempt.reasoning = "\n\n".join(
                    r for r in (first.reasoning, attempt.reasoning) if r.strip())

        final_choice: dict[str, Any] = {
            "message": {
                "content": attempt.text or None,
                "tool_calls": list(attempt.tool_calls.values()) if attempt.tool_calls else None,
            }
        }
        if attempt.reasoning.strip():
            # #333: the shared parser turns this into a ThinkingBlock — kept on
            # the message, never in its text, never sent back to the server.
            final_choice["message"]["reasoning_content"] = attempt.reasoning
        final_message, dropped_malformed = _parse_assistant_message(
            final_choice, model=request.model
        )

        yield ApiMessageCompleteEvent(
            message=final_message,
            usage=UsageSnapshot(
                input_tokens=attempt.input_tokens,
                output_tokens=attempt.output_tokens,
            ),
            stop_reason=attempt.finish_reason,
            dropped_malformed=dropped_malformed,
        )

    async def _stream(
        self, url: str, payload: dict[str, Any], attempt: _Attempt,
    ) -> AsyncIterator[ApiStreamEvent]:
        """POST one streamed completion; text deltas are yielded as they come,
        everything else accumulates on ``attempt``."""
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
                        attempt.input_tokens = chunk.get("prompt_eval_count", 0)
                        attempt.output_tokens = chunk.get("eval_count", 0)

                    if "usage" in chunk:
                        # completion_tokens already counts the thinking tokens
                        # (the runner counts every sampled token; the thought is
                        # split out afterwards) — nothing to add for reasoning.
                        u = chunk["usage"] or {}
                        attempt.input_tokens = u.get("prompt_tokens", attempt.input_tokens)
                        attempt.output_tokens = u.get("completion_tokens", attempt.output_tokens)

                    for choice in chunk.get("choices", []):
                        attempt.finish_reason = choice.get("finish_reason") or attempt.finish_reason
                        delta = choice.get("delta") or {}

                        # Ollama's field is ``reasoning`` (0.23.0 and 0.34.x);
                        # ``reasoning_content`` is llama.cpp's, accepted too.
                        thought = delta.get("reasoning") or delta.get("reasoning_content") or ""
                        if thought:
                            attempt.reasoning += thought

                        # Absent, null and "" are all no text (0.34.x omits the
                        # field on a reasoning chunk; 0.23.0 sends "").
                        text = delta.get("content") or ""
                        if text:
                            attempt.text += text
                            yield ApiTextDeltaEvent(text=text)

                        for tc in delta.get("tool_calls") or []:
                            idx = tc.get("index", 0)
                            if idx not in attempt.tool_calls:
                                attempt.tool_calls[idx] = {
                                    "id": tc.get("id", f"toolu_{uuid4().hex}"),
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            fn = tc.get("function", {})
                            if fn.get("name"):
                                attempt.tool_calls[idx]["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                attempt.tool_calls[idx]["function"]["arguments"] += fn["arguments"]

    async def _thinking_support(self, model: str) -> _ModelThinking | None:
        """Does ``model`` think, per this server's ``/api/show``? None = unknown:
        no name, a cloud name, or an /api/show that could not be read. Unknown
        is never cached (a transient failure must not become 'not thinking' for
        good) and adds nothing to the request."""
        if not model or _is_cloud_model(model):
            return None
        known = self._thinking.get(model)
        if known is not None:
            return known
        url = f"{self._base_url}/api/show"
        try:
            async with httpx.AsyncClient(timeout=_SHOW_TIMEOUT_S) as client:
                resp = await client.post(url, json={"model": model})
            if resp.status_code != 200:
                log.debug("ollama /api/show %s for %s: HTTP %d", url, model, resp.status_code)
                return None
            show = resp.json()
        except Exception as exc:  # noqa: BLE001 — unknown, not fatal
            log.debug("ollama /api/show %s for %s failed: %s", url, model, exc)
            return None
        caps = show.get("capabilities") if isinstance(show, dict) else None
        if not isinstance(caps, list):
            return None
        details = show.get("details") if isinstance(show.get("details"), dict) else {}
        family = str(details.get("family") or "").lower()
        verdict = _ModelThinking(capable=_THINKING_CAPABILITY in caps,
                                 harmony=family in _HARMONY_FAMILIES)
        self._thinking[model] = verdict
        return verdict

    @staticmethod
    def _record_reasoning_only(model: str, attempt: _Attempt, *, retried: bool) -> None:
        """A completion that thought but answered nothing — observable in
        ``silent_failures``, in the shape llama_cpp.py records an empty
        completion with. Unlike llama_cpp.py, the reasoning is NOT returned as
        the reply (``used_reasoning_fallback`` is always False here)."""
        log.warning(
            "ollama returned reasoning but no content (model=%s, finish=%s, "
            "reasoning_chars=%d, output_tokens=%d)%s",
            model, attempt.finish_reason, len(attempt.reasoning), attempt.output_tokens,
            " — retrying once with thinking off" if retried else "",
        )
        try:
            from prometheus.telemetry.tracker import get_telemetry_handle

            tel = get_telemetry_handle()
            if tel is not None and hasattr(tel, "record_silent_failure"):
                tel.record_silent_failure(
                    "ollama_provider",
                    "stream_message",
                    EmptyCompletionError(
                        f"finish_reason={attempt.finish_reason}, "
                        f"reasoning_chars={len(attempt.reasoning)}"
                    ),
                    context={
                        "model": model,
                        "finish_reason": attempt.finish_reason,
                        "output_tokens": attempt.output_tokens,
                        "reasoning_chars": len(attempt.reasoning),
                        "used_reasoning_fallback": False,
                        # llama_cpp.py's meaning: "length" with NO reasoning.
                        # A reasoning-only completion always has reasoning.
                        "budget_exhausted_on_thinking": False,
                        "reasoning_only": True,
                        "retried_with_thinking_off": retried,
                    },
                )
        except Exception:
            log.exception("ollama: telemetry record_silent_failure failed")

