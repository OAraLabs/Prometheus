"""Operator hook hardening (WP-X.9, WP-X.10).

PROMPT AND AGENT HOOKS HAD NO DEADLINE AND NO CATCH
---------------------------------------------------
`_run_prompt_like_hook` awaited the daemon's provider with no bound, although
`timeout_seconds` is declared for both kinds, and let anything it raised
escape `HookExecutor.execute`. The loop's `_safe_execute` then reported the
hook's failure as the tool's: "Tool X raised an exception". At
`post_tool_use` the tool has already run, so the model was told a call failed
whose side effects had landed, and could run it again.

COMMAND HOOKS INHERITED THE DAEMON'S WHOLE ENVIRONMENT
------------------------------------------------------
`{**os.environ, **payload}` handed every provider API key and
`PROMETHEUS_API_TOKEN` to the hook's shell. A hook now gets PATH, HOME, USER,
LANG, LC_*, TMPDIR and SHELL, the names in its own `env_allowlist`, and the
payload variables. Nothing else.

The environment tests plant SENTINEL values rather than asking whether a name
is set: the hook shell is `bash -l`, and an operator's login files may export
a real key under the same name. What this change bounds is what the daemon
hands over, and a sentinel only the daemon's environment holds measures
exactly that.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import AsyncIterator

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import LoopContext, _safe_execute
from prometheus.hooks.events import HookEvent
from prometheus.hooks.executor import HookExecutionContext, HookExecutor
from prometheus.hooks.registry import HookRegistry
from prometheus.hooks.schemas import (
    AgentHookDefinition,
    CommandHookDefinition,
    PromptHookDefinition,
)
from prometheus.providers.base import ApiMessageRequest, ModelProvider
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult
from tests.support.doubles import register_double

EXECUTOR_LOGGER = "prometheus.hooks.executor"
PAYLOAD_SENTINEL = "payload-sentinel-7f3a"


# ---------------------------------------------------------------------------
# Providers a prompt/agent hook can meet
# ---------------------------------------------------------------------------

@register_double("hook_hardening.HangingProvider",
                 replaces="prometheus.providers.base.ModelProvider")
class HangingProvider(ModelProvider):
    """A provider that accepts the request and never answers."""

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator:
        await asyncio.Event().wait()
        yield  # pragma: no cover — unreachable; makes this an async generator


@register_double("hook_hardening.RaisingProvider",
                 replaces="prometheus.providers.base.ModelProvider")
class RaisingProvider(ModelProvider):
    """A provider whose request fails, quoting what it was sent."""

    def __init__(self, exc: BaseException | None = None) -> None:
        self._exc = exc

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator:
        # Real provider errors can echo the request; this one does, so the
        # tests can check the payload never reaches a log line or a reason.
        raise self._exc or RuntimeError(
            f"upstream refused request: {request.messages[0].text}")
        yield  # pragma: no cover


def _executor(event: HookEvent, hook, provider: ModelProvider, tmp_path: Path) -> HookExecutor:
    registry = HookRegistry()
    registry.add(event, hook)
    return HookExecutor(
        registry,
        HookExecutionContext(cwd=tmp_path, provider=provider, default_model="stub"),
    )


def _prompt_like(kind: str, **kw):
    cls = PromptHookDefinition if kind == "prompt" else AgentHookDefinition
    return cls(prompt="Is this call safe? $ARGUMENTS", **kw)


def _warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records
            if r.name == EXECUTOR_LOGGER and r.levelno == logging.WARNING]


# ---------------------------------------------------------------------------
# (a) the deadline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["prompt", "agent"])
@pytest.mark.asyncio
async def test_hung_prompt_hook_is_cut_off_at_its_timeout(kind, tmp_path, caplog):
    hook = _prompt_like(kind, timeout_seconds=1, block_on_failure=True)
    executor = _executor(HookEvent.PRE_TOOL_USE, hook, HangingProvider(), tmp_path)

    caplog.set_level(logging.WARNING, logger=EXECUTOR_LOGGER)
    started = time.monotonic()
    # The outer bound is the test's own: without the fix the hook never
    # returns, and this is what turns that hang into a failure.
    result = await asyncio.wait_for(
        executor.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"}),
        timeout=10,
    )
    elapsed = time.monotonic() - started

    assert 0.9 <= elapsed < 5, f"cut off after {elapsed:.2f}s, not at its 1s timeout"
    [one] = result.results
    assert one.success is False
    assert one.blocked is True
    assert f"pre_tool_use {kind} hook #1 timed out after 1s" == one.reason
    assert one.metadata["outcome"] == "timeout"
    [warning] = _warnings(caplog)
    assert "outcome=timeout" in warning.getMessage()


@pytest.mark.asyncio
async def test_provider_timeout_error_is_an_error_not_our_deadline(tmp_path):
    """Only the hook's own deadline reads as a timeout. A TimeoutError the
    provider raised by itself, well inside the deadline, is an error."""
    hook = _prompt_like("prompt", timeout_seconds=30, block_on_failure=False)
    executor = _executor(HookEvent.PRE_TOOL_USE, hook,
                         RaisingProvider(TimeoutError("read timed out")), tmp_path)

    result = await executor.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"})

    [one] = result.results
    assert one.metadata["outcome"] == "error"
    assert one.reason == "pre_tool_use prompt hook #1 raised TimeoutError"
    assert one.blocked is False


# ---------------------------------------------------------------------------
# (a) a raising hook, through the loop's own tool path
# ---------------------------------------------------------------------------

class _NoteInput(BaseModel):
    text: str


class _NoteTool(BaseTool):
    """A tool with a side effect the test can count."""

    name = "write_note"
    description = "append a note"
    input_model = _NoteInput

    def __init__(self) -> None:
        self.runs = 0

    async def execute(self, arguments, context):  # noqa: ANN001
        self.runs += 1
        return ToolResult(output=f"appended {len(arguments.text)} bytes to notes.txt")


def _loop_with(hook_event: HookEvent, hook, tmp_path: Path):
    tool = _NoteTool()
    tools = ToolRegistry()
    tools.register(tool)
    context = LoopContext(
        provider=None, model="stub", system_prompt="", max_tokens=1024,
        tool_registry=tools,
        hook_executor=_executor(hook_event, hook, RaisingProvider(), tmp_path),
    )
    call = SimpleNamespace(id="toolu_1", name="write_note",
                           input={"text": PAYLOAD_SENTINEL})
    return context, call, tool


@pytest.mark.parametrize("kind", ["prompt", "agent"])
@pytest.mark.asyncio
async def test_raising_post_tool_use_hook_leaves_the_tool_result_as_returned(
        kind, tmp_path, caplog):
    # block_on_failure is the prompt/agent default (True): even a hook that
    # asked to block cannot rewrite a result whose side effects have landed.
    hook = _prompt_like(kind)
    assert hook.block_on_failure is True
    context, call, tool = _loop_with(HookEvent.POST_TOOL_USE, hook, tmp_path)

    caplog.set_level(logging.WARNING, logger=EXECUTOR_LOGGER)
    block = await _safe_execute(context, call, raw_model_output=None)

    assert tool.runs == 1
    assert block.tool_use_id == "toolu_1"
    assert block.is_error is False
    assert block.content == f"appended {len(PAYLOAD_SENTINEL)} bytes to notes.txt"
    [warning] = _warnings(caplog)
    assert f"post_tool_use {kind} hook #1" in warning.getMessage()
    assert "outcome=error" in warning.getMessage()


@pytest.mark.parametrize("kind", ["prompt", "agent"])
@pytest.mark.asyncio
async def test_raising_pre_tool_use_hook_blocks_with_a_reason_naming_the_hook(
        kind, tmp_path, caplog):
    hook = _prompt_like(kind, block_on_failure=True, matcher="write_*")
    context, call, tool = _loop_with(HookEvent.PRE_TOOL_USE, hook, tmp_path)

    caplog.set_level(logging.WARNING, logger=EXECUTOR_LOGGER)
    block = await _safe_execute(context, call, raw_model_output=None)

    assert tool.runs == 0
    assert block.is_error is True
    assert "raised an exception" not in block.content
    assert block.content == (
        f"pre_tool_use {kind} hook #1 (matcher 'write_*') raised RuntimeError")
    [warning] = _warnings(caplog)
    assert "blocking the call" in warning.getMessage()
    # The provider's error quoted the prompt, which carries the payload.
    # Neither the model (the reason) nor the log gets it.
    assert PAYLOAD_SENTINEL not in block.content
    assert PAYLOAD_SENTINEL not in warning.getMessage()


@pytest.mark.asyncio
async def test_raising_pre_tool_use_hook_without_block_lets_the_tool_run(tmp_path):
    hook = _prompt_like("prompt", block_on_failure=False)
    context, call, tool = _loop_with(HookEvent.PRE_TOOL_USE, hook, tmp_path)

    block = await _safe_execute(context, call, raw_model_output=None)

    assert tool.runs == 1
    assert block.is_error is False
    assert block.content == f"appended {len(PAYLOAD_SENTINEL)} bytes to notes.txt"


# ---------------------------------------------------------------------------
# (b) the command hook's environment
# ---------------------------------------------------------------------------

DAEMON_SECRETS = {
    "ANTHROPIC_API_KEY": "sentinel-anthropic-0b1c",
    "OPENAI_API_KEY": "sentinel-openai-2d3e",
    "GEMINI_API_KEY": "sentinel-gemini-4f5a",
    "PROMETHEUS_API_TOKEN": "sentinel-daemon-token-6b7c",
    "TELEGRAM_BOT_TOKEN": "sentinel-telegram-8d9e",
}


async def _run_command(tmp_path: Path, hook: CommandHookDefinition, payload: dict):
    executor = _executor(HookEvent.PRE_TOOL_USE, hook, RaisingProvider(), tmp_path)
    result = await executor.execute(HookEvent.PRE_TOOL_USE, payload)
    [one] = result.results
    assert one.success, f"hook failed: {one.reason}"
    return one.output


@pytest.mark.asyncio
async def test_command_hook_env_is_minimal_plus_payload(tmp_path, monkeypatch):
    for name, value in DAEMON_SECRETS.items():
        monkeypatch.setenv(name, value)
    payload = {"tool_name": "bash", "tool_input": {"command": PAYLOAD_SENTINEL},
               "event": "pre_tool_use"}

    output = await _run_command(tmp_path, CommandHookDefinition(command="env"), payload)

    for name, value in DAEMON_SECRETS.items():
        assert value not in output, f"{name} reached the hook"
    # Payload variables still arrive, with the payload as the value.
    lines = output.splitlines()
    assert "PROMETHEUS_HOOK_EVENT=pre_tool_use" in lines
    assert any(line.startswith("PROMETHEUS_HOOK_PAYLOAD=") and PAYLOAD_SENTINEL in line
               for line in lines)
    assert any(line.startswith("ARGUMENTS=") and PAYLOAD_SENTINEL in line
               for line in lines)
    # And the shell still has what it needs to run anything at all.
    assert any(line.startswith("PATH=") for line in lines)
    assert any(line.startswith("HOME=") for line in lines)


@pytest.mark.asyncio
async def test_allowlisted_variable_gets_through_and_its_unlisted_neighbor_does_not(
        tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_SERVICE_URL", "sentinel-allowed-1a2b")
    monkeypatch.setenv("NOTES_SERVICE_URL_TOKEN", "sentinel-not-listed-3c4d")
    monkeypatch.setenv("PROMETHEUS_HOOK_PAYLOAD", "sentinel-forged-payload")
    hook = CommandHookDefinition(
        command=('printf "%s|%s|%s" "$NOTES_SERVICE_URL" '
                 '"$NOTES_SERVICE_URL_TOKEN" "$PROMETHEUS_HOOK_PAYLOAD"'),
        env_allowlist=["NOTES_SERVICE_URL", "NAME_THE_DAEMON_DOES_NOT_HAVE",
                       "PROMETHEUS_HOOK_PAYLOAD"],
    )

    output = await _run_command(tmp_path, hook, {"tool_name": PAYLOAD_SENTINEL})

    allowed, neighbor, payload = output.split("|")
    # Exact names: the listed one arrives, a longer name sharing its prefix
    # does not, and a listed name the daemon lacks is simply unset.
    assert allowed == "sentinel-allowed-1a2b"
    assert neighbor == ""
    # Listing a payload variable's name cannot replace the payload.
    assert PAYLOAD_SENTINEL in payload
    assert "sentinel-forged-payload" not in payload


# ---------------------------------------------------------------------------
# (b) oara doctor lists each hook and its allowlist
# ---------------------------------------------------------------------------

def test_doctor_lists_each_hook_with_its_allowlist():
    from prometheus.cli.doctor import check_operator_hooks

    rows = check_operator_hooks({"hooks": {
        "pre_tool_use": [
            {"type": "command", "command": "true", "matcher": "bash",
             "block_on_failure": True, "env_allowlist": ["AWS_PROFILE", "NOTES_URL"]},
            {"type": "command", "command": "true"},
        ],
        "post_tool_use": [{"type": "http", "url": "http://127.0.0.1:9/hook"}],
    }})

    by_name = {r.name: r for r in rows}
    assert set(by_name) == {
        "Hook pre_tool_use command hook #1 (matcher 'bash')",
        "Hook pre_tool_use command hook #2",
        "Hook post_tool_use http hook #1",
    }
    first = by_name["Hook pre_tool_use command hook #1 (matcher 'bash')"].message
    assert "allowlist: AWS_PROFILE, NOTES_URL" in first
    assert "block_on_failure=true" in first
    assert "allowlist: none" in by_name["Hook pre_tool_use command hook #2"].message
    # An http hook spawns no process, so there is no environment to list.
    assert "allowlist" not in by_name["Hook post_tool_use http hook #1"].message
    assert all(r.status == "info" for r in rows)


def test_doctor_adds_no_hook_rows_when_none_are_configured():
    from prometheus.cli.doctor import check_operator_hooks

    assert check_operator_hooks({}) == []
    assert check_operator_hooks({"hooks": {}}) == []
