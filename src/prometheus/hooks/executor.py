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
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

# httpcore is httpx's transport layer today, used here only to make a TLS
# handshake cut off by a hook's deadline close its socket (_make_cancel_safe).
# It is declared as a dependency, but it is httpx's internals: if it is missing
# or its network classes move, http hooks still run, without that wrapping,
# and each one says so in a WARNING. Nothing else here needs it.
try:
    import httpcore
    _NETWORK_STREAM = httpcore.AsyncNetworkStream
    _NETWORK_BACKEND = httpcore.AsyncNetworkBackend
except (ImportError, AttributeError):
    _NETWORK_STREAM = _NETWORK_BACKEND = None

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
                results.append(await self._run_command_hook(
                    hook, event, payload, position=position))
            elif isinstance(hook, HttpHookDefinition):
                results.append(await self._run_http_hook(
                    hook, event, payload, position=position))
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
        *,
        position: int,
    ) -> HookResult:
        # $ARGUMENTS IS A SHELL PARAMETER, NOT A TEXT SPLICE. See
        # _payload_environment below for why that distinction is the whole
        # fix — the command string is passed to bash EXACTLY as the operator
        # wrote it, and the model-controlled payload only ever arrives as the
        # VALUE of a variable.
        #
        # A hook that cannot START — a working directory that no longer
        # exists, no /bin/bash, a payload the environment cannot carry — used
        # to raise out of `execute`, and the loop reported the hook's failure
        # as the tool's ("Tool X raised an exception"). It now fails the way a
        # prompt/agent hook does: a HookResult honoring block_on_failure, a
        # reason naming the hook, one WARNING.
        started = time.monotonic()
        try:
            argv, env = _command_launch(
                hook.command, _command_environment(hook, event, payload),
            )
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(self._context.cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                # Its own process group, so a timeout can stop everything the
                # hook started, not just bash: see _kill_process_group.
                start_new_session=True,
            )
        except Exception as exc:  # noqa: BLE001 — a hook failure must not look like a tool failure
            label = hook_label(event, position, hook)
            log.debug("hook %s failed to start", label, exc_info=True)
            return _failed_hook_result(
                hook, event, label, "error",
                f"failed to start ({type(exc).__name__})", started,
            )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=hook.timeout_seconds,
            )
        except asyncio.TimeoutError:
            # process.kill() alone signalled only bash. Anything bash had
            # started (the `sleep` in `sleep 4; echo ...`, both halves of a
            # pipeline, a backgrounded job) kept the output pipes open, and
            # waiting for the process means waiting for those pipes, so a 1 s
            # timeout returned after 4 s (WP-X.26). The whole group goes now.
            label = hook_label(event, position, hook)
            detail = f"timed out after {hook.timeout_seconds}s"
            leftover = await _stop_process_group(process)
            if leftover:
                detail += f"; {leftover}"
            return _failed_hook_result(hook, event, label, "timeout", detail, started)
        except asyncio.CancelledError:
            # A cancelled turn, Ctrl-C, daemon shutdown. The hook runs in its
            # own session, so no terminal signal reaches it: if we do not stop
            # its group here, nothing will. Don't wait for it; just stop it.
            _kill_process_group(process)
            raise

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
        *,
        position: int,
    ) -> HookResult:
        # httpx's timeout is PER PHASE: connect, each read, each write. A
        # server that sends a byte every half second never trips a 1 s read
        # timeout, and such a hook ran as long as the server liked, then
        # reported success (WP-X.26). timeout_seconds is now the total.
        #
        # The deadline arrives as a cancellation, which can land anywhere,
        # including inside httpcore's TLS handshake, whose cleanup runs only on
        # an Exception. The client is built exactly as before (so environment
        # and system proxies still apply) and its pools are then made
        # cancel-safe for exactly that case (_make_cancel_safe); httpx's own
        # per-phase limits are as before.
        started = time.monotonic()
        deadline = asyncio.timeout(hook.timeout_seconds)
        try:
            async with deadline:
                async with httpx.AsyncClient(timeout=hook.timeout_seconds) as client:
                    _make_cancel_safe(client)
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
            if isinstance(exc, TimeoutError) and deadline.expired():
                return _failed_hook_result(
                    hook, event, hook_label(event, position, hook), "timeout",
                    f"timed out after {hook.timeout_seconds}s", started,
                )
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
                return _failed_hook_result(
                    hook, event, label, "timeout",
                    f"timed out after {hook.timeout_seconds}s", started,
                )
            log.debug("hook %s raised", label, exc_info=True)
            return _failed_hook_result(
                hook, event, label, "error", f"raised {type(exc).__name__}", started,
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


#: The leader of a command hook's session: a constant, non-login bash that runs
#: the operator's command in a login bash as its CHILD, then exits with its
#: status. See _command_launch.
_SESSION_LEADER_SCRIPT = (
    '{ case $# in 2) BASH_ENV=$2; export BASH_ENV;; esac; '
    '/bin/bash -lc "$1" 2>&3 3>&-; exit $?; } 3>&2 2>/dev/null'
)


def _command_launch(command: str, env: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """argv and environment for a command hook: ``bash -lc <command>``, one level down.

    The hook gets its own session so a timeout can stop everything it started.
    Run directly, ``bash -lc cmd`` execs a single simple command in place, and
    the operator's own program would become the session and group leader,
    where it was an ordinary process before: its ``setsid()`` fails with
    EPERM, and util-linux ``setsid ./gate.sh`` forks and exits 0 at once, so a
    failing gate would ALLOW the call. A constant leader keeps the command out
    of that role; ``exit $?`` hands its status through. The command reaches the
    inner bash as a positional argument, never as script text of the leader.

    The leader keeps out of the hook what it can:

    - Its own stderr goes to /dev/null (the command keeps the real one), so
      when the command dies from a signal the leader's job-status line
      ("Killed: 9 /bin/bash -lc ...") does not become the hook's output.
    - ``--norc``: bash sources ~/.bashrc for a non-login ``-c`` shell whose
      stdin is a socket (its rshd/sshd heuristic), and a daemon started with
      socket stdio would otherwise run it before every hook.
    - A non-interactive bash sources ``$BASH_ENV`` before its script runs, so
      an allowlisted BASH_ENV would run twice, the first time before the
      operator's profile. The leader gets it as an argument instead and
      exports it, unchanged, for the command's login bash only.

    Differences that remain, listed in the hooks contract (§15): the command's
    ``$PPID`` is the leader, not the daemon; ``SHLVL`` is one higher; a command
    killed by a signal reports 128+n (the leader's ``exit``), not -n; a warning
    bash prints while starting up, before the script redirects stderr (bash 5.2,
    an ``LC_ALL`` naming a missing locale), appears twice; and a fork failure
    in the leader itself leaves no message.
    """
    env = dict(env)
    argv = ["/bin/bash", "--norc", "-c", _SESSION_LEADER_SCRIPT, "prometheus-hook", command]
    bash_env = env.pop("BASH_ENV", None)
    if bash_env is not None:
        argv.append(bash_env)
    return argv, env


#: After a command hook times out, how long to keep killing its process group
#: and waiting for its output pipes to close before giving up on them.
_REAP_GRACE_SECONDS = 2.0
_KILL_INTERVAL_SECONDS = 0.05


def _cancel_safe_backend_class():
    """The cancel-safe backend class, or None when httpcore can't provide
    the bases it wraps (see the import at the top)."""
    if _NETWORK_STREAM is None or _NETWORK_BACKEND is None:
        return None

    class _CancelSafeStream(_NETWORK_STREAM):
        """A network stream that closes its socket when a TLS handshake is cut off.

        httpcore's ``start_tls`` closes the stream only on an ``Exception``. An
        http hook's total deadline cancels the request, and a cancellation that
        lands in the handshake left the socket open for as long as the server
        held it: one leaked fd per timed-out hook against a stalled TLS endpoint.
        """

        def __init__(self, stream) -> None:
            self._stream = stream

        async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
            return await self._stream.read(max_bytes, timeout)

        async def write(self, buffer: bytes, timeout: float | None = None) -> None:
            await self._stream.write(buffer, timeout)

        async def aclose(self) -> None:
            await self._stream.aclose()

        async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            try:
                tls = await self._stream.start_tls(ssl_context, server_hostname, timeout)
            except BaseException:
                try:
                    await self._stream.aclose()
                except BaseException:  # noqa: BLE001 — the original exception wins
                    pass
                raise
            return _CancelSafeStream(tls)

        def get_extra_info(self, info: str):
            return self._stream.get_extra_info(info)

    class _CancelSafeBackend(_NETWORK_BACKEND):
        """httpcore's own backend, handing out cancel-safe streams."""

        def __init__(self, inner) -> None:
            self._inner = inner

        async def connect_tcp(self, host, port, timeout=None, local_address=None,
                              socket_options=None):
            return _CancelSafeStream(await self._inner.connect_tcp(
                host, port, timeout=timeout, local_address=local_address,
                socket_options=socket_options))

        async def connect_unix_socket(self, path, timeout=None, socket_options=None):
            return _CancelSafeStream(await self._inner.connect_unix_socket(
                path, timeout=timeout, socket_options=socket_options))

        async def sleep(self, seconds: float) -> None:
            await self._inner.sleep(seconds)

    return _CancelSafeBackend


try:
    _CancelSafeBackend = _cancel_safe_backend_class()
except Exception:  # noqa: BLE001 — a changed httpcore must not stop hooks loading
    _CancelSafeBackend = None


def _make_cancel_safe(client: httpx.AsyncClient) -> None:
    """Give every connection pool ``client`` holds a cancel-safe backend.

    That is the direct transport and each proxy transport httpx mounted from
    the environment or the system settings, so a proxy's CONNECT-tunnel
    handshake is covered too. The client itself is built exactly as before:
    passing ``transport=`` instead would make httpx skip environment proxies
    altogether. httpx 0.28 has no public way to pass a network backend, but
    the httpcore pool each transport holds uses its ``_network_backend``
    lazily for every connection, so it is swapped in place. If a future httpx
    moves it, the hook still runs, and this says so (the TLS-stall test in
    tests/test_hook_timeouts.py would fail too).
    """
    try:
        transports = [client._transport, *(t for t in client._mounts.values() if t is not None)]
    except AttributeError:
        transports = [client]  # httpx moved them: warn once, below, and go on
    for transport in transports:
        pool = getattr(transport, "_pool", None)
        backend = getattr(pool, "_network_backend", None)
        if backend is None or _CancelSafeBackend is None:
            log.warning("http hooks: %s has no pool backend to wrap (%s); a hook that "
                        "times out in a TLS handshake may leak its socket",
                        type(transport).__name__,
                        "httpcore's network classes are not importable"
                        if _CancelSafeBackend is None else "httpx moved it")
        elif not isinstance(backend, _CancelSafeBackend):
            pool._network_backend = _CancelSafeBackend(backend)


def _kill_process_group(process: asyncio.subprocess.Process) -> bool:
    """SIGKILL a command hook's whole process group. False once it is empty.

    The hook runs with ``start_new_session=True``, so bash leads a group whose
    id is bash's own pid. That id stays valid while ANY member lives, even
    after bash itself has exited and left a backgrounded child holding the
    pipes, which is why it is used directly instead of ``os.getpgid``. Bash is
    a member, so it needs no separate kill (and ``process.kill()`` could reap
    it behind asyncio's back on Python 3.11/3.12).
    """
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


async def _stop_process_group(process: asyncio.subprocess.Process) -> str | None:
    """Kill the group until the hook's output pipes close. None when they do.

    The pipes, not bash's exit, are what the timeout was waiting on, so that is
    what is waited for: the rest of ``communicate()``, which ends only at EOF on
    both. The kill repeats, because a child being forked at the instant of one
    SIGKILL can miss it. After the grace, whatever still holds the pipes is
    named: a process that left the group (setsid), or one SIGKILL did not end.
    """
    deadline = time.monotonic() + _REAP_GRACE_SECONDS
    drain = asyncio.ensure_future(process.communicate())
    try:
        while True:
            group_alive = _kill_process_group(process)
            done, _ = await asyncio.wait({drain}, timeout=_KILL_INTERVAL_SECONDS)
            if done:
                return None
            if time.monotonic() >= deadline:
                if group_alive:
                    return "a process in its group did not exit after SIGKILL"
                return "a process it started left its group and was not stopped"
    finally:
        if not drain.done():
            drain.cancel()


def _failed_hook_result(
    hook: HookDefinition,
    event: HookEvent,
    label: str,
    outcome: str,
    detail: str,
    started: float,
) -> HookResult:
    """A hook that did not produce an answer: one WARNING, one failed result.

    The result honors `block_on_failure`, and its reason names the hook. The
    detail carries an exception's TYPE only, here and in the log: its text can
    quote the payload (a provider echoing its request), and the reason reaches
    the model and the telemetry row. Callers log the traceback at DEBUG.
    """
    elapsed_ms = int((time.monotonic() - started) * 1000)
    if event is HookEvent.POST_TOOL_USE:
        # The loop discards post_tool_use results: the call has already run
        # and block_on_failure cannot undo it, so don't claim it blocked.
        consequence = "the call already ran; its result is unchanged"
    elif hook.block_on_failure:
        consequence = "blocking the call (block_on_failure)"
    else:
        consequence = "continuing"
    log.warning(
        "hook %s outcome=%s (%s) after %d ms — %s",
        label, outcome, detail, elapsed_ms, consequence,
    )
    return HookResult(
        hook_type=hook.type,
        success=False,
        blocked=hook.block_on_failure,
        reason=f"{label} {detail}",
        metadata={"outcome": outcome, "duration_ms": elapsed_ms},
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
