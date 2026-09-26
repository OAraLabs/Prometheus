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
before. Reasoning is kept as a thinking block, never as reply text. A turn that
thought but answered nothing is recorded, and retried once with thinking off
when that can help (``stream_message``).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any, AsyncIterator
from uuid import uuid4

import httpx

from prometheus.engine.messages import ThinkingBlock
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
    an unreadable /api/show leaves the model unknown, never 'not thinking'.

    ``effort_ok`` turns False once this server refused ``reasoning_effort``
    for the model: Ollama 0.11.5-0.12.3 pass "none" through as a think level
    and answer 400 "invalid think value" (only 0.12.4 on map it to off). The
    model then gets the request it always got — thinking on, as before."""

    __slots__ = ("capable", "harmony", "effort_ok")

    def __init__(self, *, capable: bool, harmony: bool) -> None:
        self.capable = capable
        self.harmony = harmony
        self.effort_ok = True


class _ThinkingOffRefused(RuntimeError):
    """The server refused the thinking-off retry's reasoning_effort. Not an
    HTTP error, so the transport retry does not repeat it."""


class _Attempt:
    """One streamed completion's accumulated state."""

    def __init__(self) -> None:
        self.text = ""
        self.reasoning = ""
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.input_tokens = 0
        self.output_tokens = 0


def _thought(event: ApiMessageCompleteEvent) -> str:
    return "".join(b.thinking for b in event.message.content if isinstance(b, ThinkingBlock))


def _reasoning_only(event: ApiMessageCompleteEvent) -> bool:
    """Thought, but produced no answer: no text, no tool call (not even a
    malformed one). The budget went to thinking (finish "length") or the model
    stopped mid-thought."""
    return (not event.message.text.strip() and not event.message.tool_uses
            and not event.dropped_malformed and bool(_thought(event).strip()))


def _refuses_effort(exc: httpx.HTTPStatusError) -> bool:
    """A 400 about the thinking value — Ollama 0.11.5-0.12.3's "invalid think
    value", or a later server's "invalid reasoning value"."""
    if exc.response.status_code != 400:
        return False
    try:
        body = exc.response.text.lower()
    except Exception:  # noqa: BLE001 — an unread body says nothing
        return False
    return "think" in body or "reasoning" in body


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
        """Stream a response from Ollama with exponential-backoff retry.

        A completion that thought but answered nothing is recorded, and — when
        thinking was on and "none" can turn it off (not gpt-oss, not a server
        that refuses reasoning_effort) — asked once more with thinking off. The
        retry is its own request with its own transport retries, so a failure
        there never re-runs the attempt that already spent its budget; if it
        fails before producing anything, the first attempt's result stands."""
        first: ApiMessageCompleteEvent | None = None
        async for event in self._with_retry(request):
            if isinstance(event, ApiMessageCompleteEvent):
                first = event
            else:
                yield event
        if first is None:
            return
        if not _reasoning_only(first):
            yield first
            return

        thinking = self._thinking.get(request.model)
        retry = (request.suppress_thinking is False and thinking is not None
                 and thinking.capable and thinking.effort_ok and not thinking.harmony)
        self._record_reasoning_only(request.model, first, retried=retry)
        if not retry:
            yield first
            return

        second: ApiMessageCompleteEvent | None = None
        streamed = False
        try:
            async for event in self._with_retry(request, thinking_off=True):
                if isinstance(event, ApiMessageCompleteEvent):
                    second = event
                else:
                    streamed = True
                    yield event
        except Exception as exc:  # noqa: BLE001 — the retry is best effort
            if streamed:
                raise
            log.warning("ollama: the thinking-off retry for %s failed (%s); "
                        "keeping the first attempt", request.model, exc)
        if second is None:
            yield first
            return
        if _reasoning_only(second):
            self._record_reasoning_only(request.model, second, retried=False)
        # The answer is the retry's. The thought is kept; the prompt was sent
        # twice but is one prompt (input_tokens is the context size the loop's
        # meter shows), while both attempts' output was really generated.
        thoughts = [t for t in (_thought(first), _thought(second)) if t.strip()]
        content = [b for b in second.message.content if not isinstance(b, ThinkingBlock)]
        if thoughts:
            content.insert(0, ThinkingBlock(thinking="\n\n".join(thoughts)))
        yield dataclasses.replace(
            second,
            message=second.message.model_copy(update={"content": content}),
            usage=second.usage.model_copy(update={
                "output_tokens": first.usage.output_tokens + second.usage.output_tokens,
            }),
        )

    def _with_retry(
        self, request: ApiMessageRequest, *, thinking_off: bool = False,
    ) -> AsyncIterator[ApiStreamEvent]:
        # The first attempt is the plain ``_call_once(request)`` every provider
        # retries through (tests/test_no_replay_after_partial_stream.py swaps it).
        return stream_with_retry(
            (lambda: self._call_once(request, thinking_off=True)) if thinking_off
            else (lambda: self._call_once(request)),
            retryable_status=RETRYABLE_STATUS_CODES,
            label="Ollama",
            max_retries=MAX_RETRIES,
        )

    async def _call_once(
        self, request: ApiMessageRequest, *, thinking_off: bool = False,
    ) -> AsyncIterator[ApiStreamEvent]:
        """Single attempt to Ollama's /v1/chat/completions.

        ``thinking_off`` forces reasoning_effort "none" for a thinking model
        whatever the request says — the reasoning-only retry."""
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
        if (thinking is not None and thinking.capable and thinking.effort_ok
                and (thinking_off or request.suppress_thinking is not False)):
            # A call that opted in (suppress_thinking=False, as coding sessions
            # do) gets nothing: Ollama's own default for a thinking model — on.
            payload["reasoning_effort"] = _THINKING_OFF

        url = f"{self._base_url}/v1/chat/completions"
        log.debug("POST %s model=%s messages=%d tools=%d grammar=%s reasoning_effort=%s",
                  url, request.model, len(messages),
                  len(payload.get("tools", [])), bool(payload.get("grammar")),
                  payload.get("reasoning_effort"))

        attempt = _Attempt()
        try:
            async for event in self._stream(url, payload, attempt):
                yield event
        except httpx.HTTPStatusError as exc:
            # The status is checked before any line is read: nothing was
            # yielded, so sending again replays nothing.
            if thinking is None or "reasoning_effort" not in payload or not _refuses_effort(exc):
                raise
            thinking.effort_ok = False
            log.warning("ollama %s refused reasoning_effort for %s; sending %s without it "
                        "from now on (thinking stays on, as before)",
                        self._base_url, request.model, request.model)
            if thinking_off:
                raise _ThinkingOffRefused(str(exc)) from exc
            payload.pop("reasoning_effort")
            attempt = _Attempt()
            async for event in self._stream(url, payload, attempt):
                yield event

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
    def _record_reasoning_only(
        model: str, event: ApiMessageCompleteEvent, *, retried: bool,
    ) -> None:
        """A completion that thought but answered nothing — observable in
        ``silent_failures``, in the shape llama_cpp.py records an empty
        completion with. Unlike llama_cpp.py, the reasoning is NOT returned as
        the reply (``used_reasoning_fallback`` is always False here)."""
        reasoning_chars = len(_thought(event))
        log.warning(
            "ollama returned reasoning but no content (model=%s, finish=%s, "
            "reasoning_chars=%d, output_tokens=%d)%s",
            model, event.stop_reason, reasoning_chars, event.usage.output_tokens,
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
                        f"finish_reason={event.stop_reason}, reasoning_chars={reasoning_chars}"
                    ),
                    context={
                        "model": model,
                        "finish_reason": event.stop_reason,
                        "output_tokens": event.usage.output_tokens,
                        "reasoning_chars": reasoning_chars,
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
