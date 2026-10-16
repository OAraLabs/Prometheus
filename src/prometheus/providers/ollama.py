"""OllamaProvider — connects to the Ollama API.

Requests go to Ollama's native ``/api/chat`` (audit 2026-10-16). Its
OpenAI-compatible ``/v1/chat/completions`` has no ``options``, so a request
could not say which window to load: the model served Ollama's default (its
VRAM tier, or 4k) while Prometheus budgeted the window ``/api/show`` reports
as TRAINED (262,144 for qwen3.5:9b), and a prompt over the served window was
cut from the FRONT with no error. Natively, every request sends
``options.num_ctx``: the window the loop budgeted
(``ApiMessageRequest.context_window``), never above the model's trained
length; a caller that budgeted none gets the window the model is loaded at
(``/api/ps``), so it never forces a reload; nothing loaded, nothing sent.

Default target: http://localhost:11434

Thinking models (WP-X.35). Ollama turns thinking ON for any model whose
``/api/show`` capabilities list ``thinking`` unless the request says otherwise.
The provider asks ``/api/show`` once per model and, only for a model that lists
``thinking``, sends ``think: false`` unless the call opted in
(``ApiMessageRequest.suppress_thinking is False``). Every other model — and any
model whose capabilities could not be read — gets no ``think`` field. Reasoning
(``message.thinking``) is kept as a thinking block, never as reply text. A turn
that thought but answered nothing is recorded, and retried once with thinking
off when that can help (``stream_message``).
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
    _unreadable_call_ids,
)

log = logging.getLogger(__name__)

# The /api/show capability that marks a model with a thinking channel; the same
# spelling from Ollama 0.23.0 to 0.34.4 (types/model/capability.go).
_THINKING_CAPABILITY = "thinking"
# Thinking OFF on the native API: ``think: false``. The /v1 route's
# reasoning_effort "none" was mapped to exactly this (openai/openai.go).
_THINKING_OFF = False
# gpt-oss goes through the harmony parser, which keeps its analysis channel
# whatever the request says: ``think: false`` cannot turn its reasoning off, so
# a retry with thinking off would only repeat the first attempt.
_HARMONY_FAMILIES = frozenset({"gptoss", "gpt-oss"})
_SHOW_TIMEOUT_S = 10.0
# How long one /api/ps reading of a model's loaded window is reused.
_PS_TTL_S = 5.0


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

    ``effort_ok`` turns False once this server refused ``think`` for the
    model (a 400 that names it). The model then gets no ``think`` field —
    thinking on, Ollama's own default."""

    __slots__ = ("capable", "harmony", "effort_ok")

    def __init__(self, *, capable: bool, harmony: bool) -> None:
        self.capable = capable
        self.harmony = harmony
        self.effort_ok = True


class _ThinkingOffRefused(RuntimeError):
    """The server refused the thinking-off retry's ``think``. Not an HTTP
    error, so the transport retry does not repeat it."""


class OllamaStreamError(RuntimeError):
    """An ``{"error": ...}`` line inside a 200 stream: Ollama's way of failing
    after the headers were sent."""


class _Attempt:
    """One streamed completion's accumulated state."""

    def __init__(self) -> None:
        self.text = ""
        self.reasoning = ""
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.served_model: str | None = None


def _thought(event: ApiMessageCompleteEvent) -> str:
    return "".join(b.thinking for b in event.message.content if isinstance(b, ThinkingBlock))


def _reasoning_only(event: ApiMessageCompleteEvent) -> bool:
    """Thought, but produced no answer: no text, no tool call (not even a
    malformed one). The budget went to thinking (finish "length") or the model
    stopped mid-thought."""
    return (not event.message.text.strip() and not event.message.tool_uses
            and not event.dropped_malformed and bool(_thought(event).strip()))


def _refuses_effort(exc: httpx.HTTPStatusError) -> bool:
    """A 400 about the thinking value ("invalid think value", "does not
    support thinking", "invalid reasoning value")."""
    if exc.response.status_code != 400:
        return False
    try:
        body = exc.response.text.lower()
    except Exception:  # noqa: BLE001 — an unread body says nothing
        return False
    return "think" in body or "reasoning" in body


def _native_messages(request: ApiMessageRequest, *, allow_images: bool) -> list[dict[str, Any]]:
    """The request's messages in ``/api/chat``'s shape.

    Built from the OpenAI-compatible list every local provider sends, then
    moved to Ollama's own fields, as its /v1 route did (openai/openai.go):
    text parts joined into ``content``, a picture's base64 into ``images``, a
    call's arguments as an object, and a tool result naming its call and tool.
    """
    names: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for msg in _build_openai_messages(request, allow_images=allow_images):
        role = msg.get("role")
        entry: dict[str, Any] = {"role": role}
        content = msg.get("content")
        if isinstance(content, list):
            texts, images = [], []
            for part in content:
                if part.get("type") == "text":
                    texts.append(str(part.get("text") or ""))
                elif part.get("type") == "image_url":
                    url = str((part.get("image_url") or {}).get("url") or "")
                    images.append(url.split(",", 1)[1] if url.startswith("data:") and "," in url else url)
            entry["content"] = " ".join(texts)
            if images:
                entry["images"] = images
        else:
            entry["content"] = content if isinstance(content, str) else ("" if content is None else str(content))
        if msg.get("tool_calls"):
            calls = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function") or {}
                raw = fn.get("arguments")
                try:
                    args = json.loads(raw) if isinstance(raw, str) and raw else (raw or {})
                except json.JSONDecodeError:
                    args = {}
                if tc.get("id"):
                    names[str(tc["id"])] = str(fn.get("name") or "")
                calls.append({"id": tc.get("id"), "function": {"name": fn.get("name", ""), "arguments": args}})
            entry["tool_calls"] = calls
        if role == "tool":
            call_id = msg.get("tool_call_id")
            if call_id:
                entry["tool_call_id"] = call_id
                if names.get(str(call_id)):
                    entry["tool_name"] = names[str(call_id)]
        out.append(entry)
    return out


class OllamaProvider(ModelProvider):
    """Provider for Ollama's native chat API (``/api/chat``).

    Usage:
        provider = OllamaProvider(base_url="http://localhost:11434")
        async for event in provider.stream_message(request):
            ...

    With forced JSON output:
        provider = OllamaProvider(force_json=True)   # adds format="json"
    """

    provider_name = "ollama"

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
        # The trained window /api/show reports, by model: the most num_ctx can ask.
        self._trained: dict[str, int] = {}
        # model -> (read at, loaded window or None), from /api/ps.
        self._loaded: dict[str, tuple[float, int | None]] = {}
        self._warned_trained: set[tuple[str, int]] = set()

    def set_grammar(self, grammar: str | None) -> None:
        """Set GBNF grammar for constrained decoding (llama.cpp extension)."""
        self._grammar = grammar

    async def detect_tool_template(self, model_name: str | None = None) -> Any:
        """What the served model's template does with tools, from ``/api/show``.

        Ollama lists ``tools`` under ``capabilities`` and returns the Go
        ``template`` for the model asked about; the verdict — an
        ``adapter.tier.ToolTemplate`` — decides the adapter tier for a model
        ``config/model_registry.yaml`` does not list (WP-X.28), and is cached
        on the instance as ``tool_template``. The request body is the one the
        backend registry's probe already sends (``{"model": name}``).

        Ollama serves many models, so the question needs a name; without one
        the answer is "unknown", never a guess. A server that cannot be asked
        is a recorded fact, never a boot failure.
        """
        from prometheus.adapter.tier import ToolTemplate, classify_template

        if not model_name:
            self.tool_template = ToolTemplate(
                native=None, call_format=None,
                evidence="no model name to ask ollama /api/show about",
            )
            return self.tool_template
        url = f"{self._base_url}/api/show"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json={"model": model_name})
                resp.raise_for_status()
                show = resp.json()
        except Exception as exc:  # noqa: BLE001 — unreachable or unparseable: recorded, not fatal
            log.warning("Chat template detection failed (%s, %s): %s", url, model_name, exc)
            self.tool_template = ToolTemplate(
                native=None, call_format=None,
                evidence=f"/api/show could not be read ({type(exc).__name__})",
            )
            return self.tool_template
        show = show if isinstance(show, dict) else {}
        self.tool_template = classify_template(
            ollama_capabilities=show.get("capabilities"), ollama_template=show.get("template"),
        )
        log.info("Chat template: %s (%s at %s)", self.tool_template.evidence, model_name, self._base_url)
        return self.tool_template

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        """Stream a response from Ollama with exponential-backoff retry.

        A completion that thought but answered nothing is recorded, and — when
        thinking was on and "none" can turn it off (not gpt-oss, not a server
        that refuses think) — asked once more with thinking off. The
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
        """Single attempt to Ollama's native /api/chat.

        ``thinking_off`` forces ``think: false`` for a thinking model whatever
        the request says — the reasoning-only retry."""
        # Ollama has no mmproj probe, so `supports_vision` stays at the class
        # default (False) unless something sets it — an ImageBlock then raises
        # rather than being silently paraphrased. Same threading as llama_cpp.
        messages = _native_messages(request, allow_images=self.supports_vision)

        # num_predict is the reply's cap (the /v1 route's max_tokens). The
        # final line reports prompt_eval_count / eval_count unasked.
        options: dict[str, Any] = {"num_predict": request.max_tokens}
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": True,
            "options": options,
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
        # /api/show says the model thinks.
        thinking = await self._thinking_support(request.model)
        if (thinking is not None and thinking.capable and thinking.effort_ok
                and (thinking_off or request.suppress_thinking is not False)):
            # A call that opted in (suppress_thinking=False, as coding sessions
            # do) gets nothing: Ollama's own default for a thinking model — on.
            payload["think"] = _THINKING_OFF

        num_ctx = await self._num_ctx(request)
        if num_ctx:
            options["num_ctx"] = num_ctx

        url = f"{self._base_url}/api/chat"
        log.debug("POST %s model=%s messages=%d tools=%d grammar=%s think=%s num_ctx=%s",
                  url, request.model, len(messages),
                  len(payload.get("tools", [])), bool(payload.get("grammar")),
                  payload.get("think"), num_ctx)

        attempt = _Attempt()
        try:
            async for event in self._stream(url, payload, attempt):
                yield event
        except httpx.HTTPStatusError as exc:
            # The status is checked before any line is read: nothing was
            # yielded, so sending again replays nothing.
            if thinking is None or "think" not in payload or not _refuses_effort(exc):
                raise
            thinking.effort_ok = False
            log.warning("ollama %s refused think for %s; sending %s without it "
                        "from now on (thinking stays on, as before)",
                        self._base_url, request.model, request.model)
            if thinking_off:
                raise _ThinkingOffRefused(str(exc)) from exc
            payload.pop("think")
            attempt = _Attempt()
            async for event in self._stream(url, payload, attempt):
                yield event

        if num_ctx and attempt.input_tokens >= num_ctx:
            # Ollama cuts an over-window prompt from the FRONT and reports the
            # window as the prompt size (measured 2026-10-16). The budget should
            # have prevented it; say so rather than let it pass.
            log.warning(
                "ollama %s: the prompt for %s filled the whole %d-token window (prompt_eval_count %d); "
                "Ollama cuts an over-window prompt from the front, so its start was probably lost",
                self._base_url, request.model, num_ctx, attempt.input_tokens,
            )

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
            unreadable_tool_calls=_unreadable_call_ids(final_choice, attempt.finish_reason),
            served_model=attempt.served_model,
        )

    async def _num_ctx(self, request: ApiMessageRequest) -> int | None:
        """The window to ask Ollama for (see the module docstring)."""
        budget = request.context_window
        if budget:
            trained = self._trained.get(request.model)
            if trained and budget > trained:
                marker = (request.model, int(budget))
                if marker not in self._warned_trained:
                    self._warned_trained.add(marker)
                    log.warning(
                        "ollama %s: %s is budgeted at %d tokens but trained for %d; asking for %d. Prompts "
                        "between the two are cut from the front: set the backend's context_limit to %d.",
                        self._base_url, request.model, budget, trained, trained, trained,
                    )
                return trained
            return int(budget)
        return await self._loaded_ctx(request.model)

    async def _loaded_ctx(self, model: str) -> int | None:
        """The window *model* is loaded at, per /api/ps (None: not loaded, or unknown)."""
        import time

        now = time.monotonic()
        cached = self._loaded.get(model)
        if cached is not None and now - cached[0] < _PS_TTL_S:
            return cached[1]
        loaded: int | None = None
        try:
            async with httpx.AsyncClient(timeout=_SHOW_TIMEOUT_S) as client:
                resp = await client.get(f"{self._base_url}/api/ps")
            if resp.status_code == 200:
                for row in (resp.json() or {}).get("models") or []:
                    if row.get("name") == model or row.get("model") == model:
                        value = row.get("context_length")
                        loaded = int(value) if isinstance(value, int) and value > 0 else None
                        break
        except Exception as exc:  # noqa: BLE001 — unknown, not fatal
            log.debug("ollama /api/ps %s failed: %s", self._base_url, exc)
        self._loaded[model] = (now, loaded)
        return loaded

    async def _stream(
        self, url: str, payload: dict[str, Any], attempt: _Attempt,
    ) -> AsyncIterator[ApiStreamEvent]:
        """POST one streamed completion; text deltas are yielded as they come,
        everything else accumulates on ``attempt``. One JSON object per line."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    log.error("HTTP %d from %s: %s",
                              response.status_code, url, response.text[:500])
                    response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get("error"):
                        raise OllamaStreamError(f"ollama {url}: {str(chunk['error'])[:500]}")

                    # What served this call, as Ollama names it on every line
                    # (WP-X.21 T12b), the way llama_cpp.py reads it.
                    attempt.served_model = chunk.get("model") or attempt.served_model
                    message = chunk.get("message") or {}

                    thought = message.get("thinking") or ""
                    if thought:
                        attempt.reasoning += thought

                    text = message.get("content") or ""
                    if text:
                        attempt.text += text
                        yield ApiTextDeltaEvent(text=text)

                    # A call arrives whole, its arguments an object.
                    for tc in message.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        args = fn.get("arguments")
                        attempt.tool_calls[len(attempt.tool_calls)] = {
                            "id": tc.get("id") or f"toolu_{uuid4().hex}",
                            "type": "function",
                            "function": {
                                "name": str(fn.get("name") or ""),
                                "arguments": args if isinstance(args, str)
                                else json.dumps(args if args is not None else {}, ensure_ascii=False),
                            },
                        }

                    if chunk.get("done"):
                        attempt.input_tokens = int(chunk.get("prompt_eval_count") or 0)
                        # eval_count counts every sampled token, the thought included.
                        attempt.output_tokens = int(chunk.get("eval_count") or 0)
                        reason = chunk.get("done_reason") or "stop"
                        # The /v1 route reported a call's end as "tool_calls".
                        attempt.finish_reason = (
                            "tool_calls" if reason == "stop" and attempt.tool_calls else reason
                        )

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
        info = show.get("model_info") if isinstance(show, dict) else None
        for key, value in (info or {}).items() if isinstance(info, dict) else ():
            if str(key).endswith(".context_length") and isinstance(value, int) and value > 0:
                self._trained[model] = value
                break
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
