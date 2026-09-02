"""OpenAI-compatible chat surface — ``GET /v1/models``, ``POST /v1/chat/completions``.

Item 2 of the 2026-09-01 roadmap. Open WebUI, LobeChat, Continue, Zed and
every other client that speaks the OpenAI chat-completions wire can point
at this daemon and get Prometheus — memory, tools, security gate, the
local model — without a Prometheus-specific client. Beacon stays the
surface; this is the second door.

What it is, precisely:

* **Stateless per request**, like OpenAI. The client sends the whole
  conversation every call; each call runs ONE agent turn in a fresh
  ``openai:<id>`` session that persists nothing to LCM. The daemon's own
  sessions, memory extraction and retention are untouched — a compat
  client owns its history, the daemon does not learn from it. (Telemetry
  rows are still written under that session id; the ``openai:`` prefix is
  not one of retention's machine tiers, so the rows simply accumulate.)
* **Tools run server-side**, invisibly. A client that sends ``tools`` /
  ``functions`` / ``tool_choice`` is refused with 400 rather than having
  its tools silently ignored: the model's tools are Prometheus's, gated by
  the SecurityGate, and the client only ever sees text deltas. A
  non-read-only tool that needs an APPROVE therefore blocks the turn the
  same way it would for Beacon — the operator approves it there.
* **``model`` is a catalog key** from ``GET /v1/models`` (``local``, or a
  cloud preset such as ``claude`` / ``qwen:qwen3.7-max``); it applies a
  per-session override for the duration of the request and clears it after.
* ``system`` messages are appended to the daemon's own system prompt as
  client instructions — they never replace the identity, tool and safety
  text the loop is built on.
* Ignored, deliberately: ``temperature``, ``top_p``, ``max_tokens``,
  ``n``, ``stop``, ``logprobs`` — the loop owns generation settings.

Streaming is Server-Sent Events in the OpenAI chunk shape, driven by the
same ``run_loop`` iteration the WebSocket bridge uses — but per caller. The
bridge broadcasts every frame to every connected client; an HTTP response
belongs to one.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

SESSION_PREFIX = "openai"
LOCAL_MODEL_KEY = "local"
_ROLES = ("system", "user", "assistant")
_CLIENT_TOOL_KEYS = ("tools", "functions", "tool_choice", "function_call")
_CLIENT_SYSTEM_HEADER = "# Instructions from the connecting client"


def _error(status: int, message: str, code: str, etype: str = "invalid_request_error") -> JSONResponse:
    """OpenAI's error envelope, so client libraries surface it as their own error type."""
    return JSONResponse(status_code=status, content={
        "error": {"message": message, "type": etype, "code": code, "param": None},
    })


def _text_of(content: Any) -> str | None:
    """Flatten OpenAI message content: a string, or a list of ``text`` parts.
    Returns None for anything else (image parts etc.) so the caller can refuse."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                return None
            parts.append(part["text"])
        return "".join(parts)
    return None


def _assistant_text(message: Any) -> str:
    from prometheus.engine.messages import TextBlock

    return "".join(b.text for b in getattr(message, "content", []) if isinstance(b, TextBlock))


def register_openai_routes(
    app: Any,
    *,
    config: dict[str, Any],
    model_catalog: Callable[[], list[dict[str, Any]]],
    resolve_model_target: Callable[[str], dict[str, Any] | None],
) -> None:
    """Mount ``/v1`` on *app*. Called by ``create_app`` before the static
    catch-all; the bearer middleware guards ``/v1/`` like ``/api/``."""
    from prometheus.api.tool_choice import resolve_mode_to_tool_choice
    from prometheus.api.turn_errors import classify_turn_error
    from prometheus.engine.messages import ConversationMessage, TextBlock
    from prometheus.engine.stream_events import (
        AssistantTextDelta,
        AssistantTurnComplete,
    )

    router = APIRouter(prefix="/v1")

    @router.get("/models")
    async def list_models():
        now = int(time.time())
        data = [
            {
                "id": row["key"],
                "object": "model",
                "created": now,
                "owned_by": row.get("provider") or "prometheus",
                # Not OpenAI fields, but harmless to clients and useful to people.
                "label": row.get("label"),
                "model": row.get("model"),
            }
            for row in model_catalog()
            if row.get("available", False)
        ]
        return {"object": "list", "data": data}

    @router.post("/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict):
            return _error(400, "expected a JSON object body", "invalid_body")

        for key in _CLIENT_TOOL_KEYS:
            if key in body:
                return _error(
                    400,
                    "tools run inside Prometheus (gated by its security gate) and are "
                    f"not client-supplied; remove {key!r} — the reply carries only text",
                    "tools_unsupported",
                )

        raw_messages = body.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            return _error(400, "messages must be a non-empty list", "messages_required")

        client_system: list[str] = []
        history: list[ConversationMessage] = []
        for i, m in enumerate(raw_messages):
            if not isinstance(m, dict) or m.get("role") not in _ROLES:
                return _error(400, f"messages[{i}].role must be one of {_ROLES}", "invalid_role")
            text = _text_of(m.get("content"))
            if text is None:
                return _error(
                    400,
                    f"messages[{i}].content must be a string or text parts "
                    "(image and audio parts are not accepted here)",
                    "unsupported_content",
                )
            role = m["role"]
            if role == "system":
                client_system.append(text)
            elif role == "user":
                history.append(ConversationMessage.from_user_text(text))
            else:
                history.append(ConversationMessage(role="assistant", content=[TextBlock(text=text)]))
        if not history or history[-1].role != "user":
            return _error(400, "the last message must be from the user", "last_message_not_user")

        bridge = getattr(app.state, "ws_bridge", None)
        loop_context = getattr(bridge, "loop_context", None)
        if loop_context is None:
            return _error(503, "the agent loop is not available on this daemon", "loop_unavailable", "server_error")

        # Model: a catalog key. `local` (or absent) is the configured primary.
        key = str(body.get("model") or LOCAL_MODEL_KEY).strip()
        known = {row["key"] for row in model_catalog()}
        if key not in known:
            return _error(
                404, f"model {key!r} is not a key from GET /v1/models", "model_not_found",
            )
        router_obj = getattr(loop_context, "model_router", None) or getattr(app.state, "model_router", None)
        if key != LOCAL_MODEL_KEY and router_obj is None:
            return _error(503, "model overrides need the model router, which is not wired", "router_unavailable", "server_error")

        mode = body.get("mode") or "agent"          # Prometheus extension: "chat" = no tools
        if mode not in ("agent", "chat"):
            return _error(400, "mode must be 'agent' or 'chat'", "invalid_mode")
        tool_choice = resolve_mode_to_tool_choice(mode)

        session_id = f"{SESSION_PREFIX}:{uuid.uuid4().hex[:12]}"
        ctx = loop_context
        if client_system:
            ctx = dataclasses.replace(
                loop_context,
                system_prompt=f"{loop_context.system_prompt}\n\n{_CLIENT_SYSTEM_HEADER}\n"
                              + "\n\n".join(client_system),
            )
        if key != LOCAL_MODEL_KEY:
            preset = resolve_model_target(key)
            if preset is None:
                return _error(404, f"model {key!r} has no resolvable preset", "model_not_found")
            try:
                router_obj.set_override(session_id, preset)
            except ValueError as exc:
                return _error(400, str(exc), "model_override_rejected")

        from prometheus.engine.agent_loop import run_loop

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        stream = bool(body.get("stream", False))

        async def events() -> AsyncIterator[tuple[str, dict[str, int]]]:
            """Yield ("delta", text) as the reply grows; the final item is
            ("done", usage). Tool events are consumed here and never leave."""
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            round_acc = ""
            try:
                async for event, _u in run_loop(ctx, history, mode=mode, session_id=session_id, tool_choice=tool_choice):
                    if isinstance(event, AssistantTextDelta):
                        round_acc += event.text
                        yield ("delta", {"text": event.text})
                    elif isinstance(event, AssistantTurnComplete):
                        # A provider that emits no deltas delivers the text only
                        # here; a streaming one already did. Emit the remainder.
                        full = _assistant_text(event.message)
                        if full.startswith(round_acc) and len(full) > len(round_acc):
                            yield ("delta", {"text": full[len(round_acc):]})
                        round_acc = ""
                        if event.usage is not None:
                            usage["prompt_tokens"] += int(event.usage.input_tokens or 0)
                            usage["completion_tokens"] += int(event.usage.output_tokens or 0)
                    # ToolExecutionStarted / Completed / ProviderDegraded: server-side, not surfaced.
            finally:
                if key != LOCAL_MODEL_KEY and router_obj is not None:
                    router_obj.clear_override(session_id)
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            yield ("done", usage)

        if not stream:
            text = ""
            usage: dict[str, int] = {}
            try:
                async for kind, payload in events():
                    if kind == "delta":
                        text += payload["text"]
                    else:
                        usage = payload
            except Exception as exc:  # the loop died: say what, in the client's envelope
                err = classify_turn_error(exc)
                logger.error("openai-compat turn failed (%s): %s", session_id, err.get("message"), exc_info=True)
                return _error(502, err.get("message") or str(exc), err.get("kind") or "turn_failed", "server_error")
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": key,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": usage,
                "prometheus": {"session_id": session_id, "mode": mode},
            }

        def chunk(delta: dict[str, Any], finish: str | None = None, usage: dict[str, int] | None = None) -> str:
            obj: dict[str, Any] = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": key,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if usage is not None:
                obj["usage"] = usage
            return f"data: {json.dumps(obj)}\n\n"

        async def sse() -> AsyncIterator[str]:
            yield chunk({"role": "assistant", "content": ""})
            try:
                async for kind, payload in events():
                    if kind == "delta":
                        yield chunk({"content": payload["text"]})
                    else:
                        yield chunk({}, finish="stop", usage=payload)
            except Exception as exc:
                err = classify_turn_error(exc)
                logger.error("openai-compat stream failed (%s): %s", session_id, err.get("message"), exc_info=True)
                yield "data: " + json.dumps({"error": {
                    "message": err.get("message") or str(exc),
                    "type": "server_error", "code": err.get("kind") or "turn_failed",
                }}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.include_router(router)
