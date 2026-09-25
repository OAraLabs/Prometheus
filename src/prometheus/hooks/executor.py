# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/hooks/executor.py
# License: MIT
# Modified: renamed imports (openharness → prometheus);
#           replaced SupportsStreamingMessages / openharness.api.client with
#           prometheus.providers.base.ModelProvider;
#           replaced openharness.hooks.loader.HookRegistry with
#           prometheus.hooks.registry.HookRegistry (in-memory, Sprint 2);
#           removed OPENHARNESS_HOOK_* env vars → PROMETHEUS_HOOK_* naming

"""Hook execution engine."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from prometheus.engine.messages import ConversationMessage
from prometheus.hooks.events import HookEvent
from prometheus.hooks.registry import HookRegistry
from prometheus.hooks.schemas import (
    AgentHookDefinition,
    CommandHookDefinition,
    HookDefinition,
    HttpHookDefinition,
    PromptHookDefinition,
)
from prometheus.hooks.types import AggregatedHookResult, HookResult
from prometheus.providers.base import ApiMessageCompleteEvent, ApiMessageRequest, ModelProvider

log = logging.getLogger(__name__)

# What every command hook inherits from the daemon's environment, when set.
# Everything else — provider API keys, PROMETHEUS_API_TOKEN, gateway tokens —
# stays behind unless the hook names it in `env_allowlist`. See
# `_command_environment`.
BASE_ENV_NAMES = ("PATH", "HOME", "USER", "LANG", "TMPDIR", "SHELL")
BASE_ENV_PREFIXES = ("LC_",)


@dataclass
class HookExecutionContext:
    """Context passed into hook execution."""

    cwd: Path
    provider: ModelProvider
    default_model: str


class HookExecutor:
    """Execute hooks for lifecycle events."""

    def __init__(self, registry: HookRegistry, context: HookExecutionContext) -> None:
        self._registry = registry
        self._context = context

    def update_registry(self, registry: HookRegistry) -> None:
        """Replace the active hook registry."""
        self._registry = registry

    async def execute(self, event: HookEvent, payload: dict[str, Any]) -> AggregatedHookResult:
        """Execute all matching hooks for an event."""
        results: list[HookResult] = []
        for position, hook in enumerate(self._registry.get(event), start=1):
            if not _matches_hook(hook, payload):
                continue
            if isinstance(hook, CommandHookDefinition):
                results.append(await self._run_command_hook(hook, event, payload))
            elif isinstance(hook, HttpHookDefinition):
                results.append(await self._run_http_hook(hook, event, payload))
            elif isinstance(hook, PromptHookDefinition):
                results.append(await self._run_prompt_like_hook(
                    hook, event, payload, agent_mode=False, position=position))
            elif isinstance(hook, AgentHookDefinition):
                results.append(await self._run_prompt_like_hook(
                    hook, event, payload, agent_mode=True, position=position))
        return AggregatedHookResult(results=results)

    async def _run_command_hook(
        self,
        hook: CommandHookDefinition,
        event: HookEvent,
        payload: dict[str, Any],
    ) -> HookResult:
        # $ARGUMENTS IS A SHELL PARAMETER, NOT A TEXT SPLICE. See
        # _payload_environment below for why that distinction is the whole
        # fix — the command string is passed to bash EXACTLY as the operator
        # wrote it, and the model-controlled payload only ever arrives as the
        # VALUE of a variable.
        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-lc",
            hook.command,
            cwd=str(self._context.cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_command_environment(hook, event, payload),
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=hook.timeout_seconds,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return HookResult(
                hook_type=hook.type,
                success=False,
                blocked=hook.block_on_failure,
                reason=f"command hook timed out after {hook.timeout_seconds}s",
            )

        output = "\n".join(
            part for part in (
                stdout.decode("utf-8", errors="replace").strip(),
                stderr.decode("utf-8", errors="replace").strip(),
            ) if part
        )
        success = process.returncode == 0
        return HookResult(
            hook_type=hook.type,
            success=success,
            output=output,
            blocked=hook.block_on_failure and not success,
            reason=output or f"command hook failed with exit code {process.returncode}",
            metadata={"returncode": process.returncode},
        )

    async def _run_http_hook(
        self,
        hook: HttpHookDefinition,
        event: HookEvent,
        payload: dict[str, Any],
    ) -> HookResult:
        try:
            async with httpx.AsyncClient(timeout=hook.timeout_seconds) as client:
                response = await client.post(
                    hook.url,
                    json={"event": event.value, "payload": payload},
                    headers=hook.headers,
                )
            success = response.is_success
            output = response.text
            return HookResult(
                hook_type=hook.type,
                success=success,
                output=output,
                blocked=hook.block_on_failure and not success,
                reason=output or f"http hook returned {response.status_code}",
                metadata={"status_code": response.status_code},
            )
        except Exception as exc:
            return HookResult(
                hook_type=hook.type,
                success=False,
                blocked=hook.block_on_failure,
                reason=str(exc),
            )

    async def _run_prompt_like_hook(
        self,
        hook: PromptHookDefinition | AgentHookDefinition,
        event: HookEvent,
        payload: dict[str, Any],
        *,
        agent_mode: bool,
        position: int,
    ) -> HookResult:
        """Run a prompt/agent hook inside its deadline; never raise.

        These hooks await the daemon's own provider, and nothing bounded that
        wait or caught what it raised. A stalled provider stalled the tool
        call. A raise escaped `execute` into `_execute_tool_call`, where
        `_safe_execute` reported it as "Tool X raised an exception" — and at
        `post_tool_use` the tool has already run, so the model was told a call
        failed whose side effects had landed, and could run it again.

        So the deadline is `timeout_seconds`, and any failure becomes a
        HookResult that honors `block_on_failure`, with a reason naming the
        hook and one WARNING. `asyncio.CancelledError` is not an Exception and
        still propagates: a cancelled turn stays cancelled.
        """
        label = hook_label(event, position, hook)
        started = time.monotonic()
        deadline = asyncio.timeout(hook.timeout_seconds)
        try:
            async with deadline:
                return await self._ask_prompt_like_hook(hook, payload, agent_mode=agent_mode)
        except Exception as exc:  # noqa: BLE001 — a hook failure must not look like a tool failure
            # Only OUR deadline is a timeout. A TimeoutError the provider
            # raised on its own is an error like any other.
            if isinstance(exc, TimeoutError) and deadline.expired():
                outcome = "timeout"
                detail = f"timed out after {hook.timeout_seconds}s"
            else:
                outcome = "error"
                detail = f"raised {type(exc).__name__}"
                log.debug("hook %s raised", label, exc_info=True)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        # The exception's TYPE only, here and in the reason: its text can
        # quote what the provider was sent, which is the payload, and the
        # reason reaches the model and the telemetry row. The traceback is at
        # DEBUG above.
        log.warning(
            "hook %s outcome=%s (%s) after %d ms — %s",
            label, outcome, detail, elapsed_ms,
            "blocking the call (block_on_failure)" if hook.block_on_failure
            else "continuing",
        )
        return HookResult(
            hook_type=hook.type,
            success=False,
            blocked=hook.block_on_failure,
            reason=f"{label} {detail}",
            metadata={"outcome": outcome, "duration_ms": elapsed_ms},
        )

    async def _ask_prompt_like_hook(
        self,
        hook: PromptHookDefinition | AgentHookDefinition,
        payload: dict[str, Any],
        *,
        agent_mode: bool,
    ) -> HookResult:
        prompt = _inject_arguments(hook.prompt, payload)
        prefix = (
            "You are validating whether a hook condition passes in Prometheus. "
            "Return strict JSON: {\"ok\": true} or {\"ok\": false, \"reason\": \"...\"}."
        )
        if agent_mode:
            prefix += " Be more thorough and reason over the payload before deciding."
        request = ApiMessageRequest(
            model=hook.model or self._context.default_model,
            messages=[ConversationMessage.from_user_text(prompt)],
            system_prompt=prefix,
            max_tokens=512,
        )

        text_chunks: list[str] = []
        final_event: ApiMessageCompleteEvent | None = None
        async for event_item in self._context.provider.stream_message(request):
            if isinstance(event_item, ApiMessageCompleteEvent):
                final_event = event_item
            else:
                text_chunks.append(event_item.text)

        text = "".join(text_chunks)
        if final_event is not None and final_event.message.text:
            text = final_event.message.text

        parsed = _parse_hook_json(text)
        if parsed["ok"]:
            return HookResult(hook_type=hook.type, success=True, output=text)
        return HookResult(
            hook_type=hook.type,
            success=False,
            output=text,
            blocked=hook.block_on_failure,
            reason=parsed.get("reason", "hook rejected the event"),
        )


def hook_label(event: HookEvent, position: int, hook: HookDefinition) -> str:
    """How logs, block reasons and `oara doctor` name one configured hook.

    Hooks have no names in config, so this is the event, the hook's 1-based
    position in that event's list, its kind and its matcher — enough to find
    the entry in `hooks:`.
    """
    label = f"{event.value} {hook.type} hook #{position}"
    matcher = getattr(hook, "matcher", None)
    if matcher:
        label += f" (matcher {matcher!r})"
    return label


def _matches_hook(hook: HookDefinition, payload: dict[str, Any]) -> bool:
    matcher = getattr(hook, "matcher", None)
    if not matcher:
        return True
    subject = str(payload.get("tool_name") or payload.get("prompt") or payload.get("event") or "")
    return fnmatch.fnmatch(subject, matcher)


def _command_environment(
    hook: CommandHookDefinition,
    event: HookEvent,
    payload: dict[str, Any],
) -> dict[str, str]:
    """The whole environment a command hook runs with. Nothing else is inherited.

    Command hooks used to get ``{**os.environ, **payload}``: the daemon's
    entire environment, which holds every provider API key, the daemon's own
    ``PROMETHEUS_API_TOKEN`` and any gateway token. A hook script — or
    anything it runs — could read and forward all of them.

    Now a hook gets only what a shell needs to behave normally
    (`BASE_ENV_NAMES`, plus ``LC_*``), then each variable the operator names
    in the hook's ``env_allowlist``, then the payload variables. The payload
    goes last so an allowlisted name can never replace it. A name on the
    allowlist that the daemon doesn't have is left unset.

    The shell is still ``bash -l``, so the operator's own login files still
    run and can export what they like. This bounds what the daemon hands over,
    not what the operator's profile adds.
    """
    env = {
        name: value
        for name, value in os.environ.items()
        if name in BASE_ENV_NAMES or name.startswith(BASE_ENV_PREFIXES)
    }
    for name in hook.env_allowlist:
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    env.update(_payload_environment(event, payload))
    return env


def _payload_environment(event: HookEvent, payload: dict[str, Any]) -> dict[str, str]:
    """Environment for a command hook. The payload travels as DATA, not text.

    WHY THIS EXISTS
    ---------------
    Command hooks used to be built by string replacement::

        command = hook.command.replace("$ARGUMENTS", json.dumps(payload))
        await asyncio.create_subprocess_exec("/bin/bash", "-lc", command, ...)

    The payload is model-controlled. `PRE_TOOL_USE` carries
    ``{"tool_name": ..., "tool_input": ...}`` and `tool_input` is whatever the
    model decided to pass — a bash command, a file body, or text it copied out
    of a page it fetched a moment earlier. Splicing that into a string handed
    to `bash -lc` makes the data into code.

    Reproduced against this executor, using the example the loader's own
    docstring documents (``echo checking $ARGUMENTS``) and a `tool_input` of
    ``{"command": "$(touch /tmp/PWNED)"}``: the file was created. Any operator
    following the documented example got arbitrary command execution from a
    tool argument, in a login shell carrying the daemon's full environment —
    `ANTHROPIC_API_KEY`, `PROMETHEUS_API_TOKEN` and the rest.

    THE FIX, AND WHY IT IS THIS ONE
    -------------------------------
    Quoting the splice (`shlex.quote`) would be a fix for one layer and a trap
    for the next: it is correct only when the placeholder sits outside quotes
    in the operator's command, and `"$ARGUMENTS"` — the natural way to write
    it, and the way the example did — would then interpolate literal quote
    characters. The escaping would be right and the meaning wrong.

    So the payload is not spliced at all. It is exported as `ARGUMENTS`, and
    the command string reaches bash exactly as written. `$ARGUMENTS` then
    resolves through ordinary parameter expansion, and BASH DOES NOT RE-EVALUATE
    THE VALUE OF AN EXPANDED VARIABLE. Verified on this platform, both quoted
    and unquoted::

        ARGUMENTS='$(touch /tmp/M)' bash -lc 'echo "checking $ARGUMENTS"'
          -> checking $(touch /tmp/M)      /tmp/M NOT created
        ARGUMENTS='`touch /tmp/M`'  bash -lc 'echo "checking $ARGUMENTS"'
          -> checking `touch /tmp/M`       /tmp/M NOT created

    Command substitution, backticks, `;`, `&&`, redirections — all of it is
    inert, because the shell parses the command text and the payload is never
    part of the command text. This is a property of where the data goes, not
    of how carefully it was escaped, which is why no pattern list or character
    filter appears anywhere here.

    WHAT THIS CHANGES FOR HOOK AUTHORS
    ----------------------------------
    The spelling is unchanged: `echo "$ARGUMENTS"` works exactly as documented.
    One case differs deliberately — `'$ARGUMENTS'` inside SINGLE quotes no
    longer interpolates, because single quotes are how a shell author says
    "literal". That is the correct reading of the operator's own syntax, and
    the previous behaviour of substituting there anyway was part of the defect.

    `PROMETHEUS_HOOK_PAYLOAD` carries the same JSON and predates this change;
    it is kept, and is the better name to use in new hooks.
    """
    blob = json.dumps(payload, ensure_ascii=True)
    return {
        "PROMETHEUS_HOOK_EVENT": event.value,
        "PROMETHEUS_HOOK_PAYLOAD": blob,
        "ARGUMENTS": blob,
    }


def _inject_arguments(template: str, payload: dict[str, Any]) -> str:
    """Textual substitution for PROMPT hooks only.

    NEVER use this to build a shell command. A prompt is model input, where
    interpolated text stays text; a command string is parsed by bash, where it
    becomes code. `_run_command_hook` deliberately does not call this — see
    `_payload_environment`.
    """
    return template.replace("$ARGUMENTS", json.dumps(payload, ensure_ascii=True))


def _parse_hook_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("ok"), bool):
            return parsed
    except json.JSONDecodeError:
        pass
    lowered = text.strip().lower()
    if lowered in {"ok", "true", "yes"}:
        return {"ok": True}
    return {"ok": False, "reason": text.strip() or "hook returned invalid JSON"}
