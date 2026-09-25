# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/hooks/schemas.py
# License: MIT
# Modified: renamed imports (openharness → prometheus)

"""Hook configuration schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CommandHookDefinition(BaseModel):
    """A hook that executes a shell command."""

    type: Literal["command"] = "command"
    command: str
    timeout_seconds: int = Field(default=30, ge=1, le=600)
    matcher: str | None = None
    block_on_failure: bool = False
    # Names of daemon environment variables this hook may see, beyond the
    # minimal set every command hook gets (PATH, HOME, USER, LANG, LC_*,
    # TMPDIR, SHELL) and its payload variables. Exact names, no patterns.
    # Empty by default: provider keys and PROMETHEUS_API_TOKEN stay behind.
    env_allowlist: list[str] = Field(default_factory=list)


class PromptHookDefinition(BaseModel):
    """A hook that asks the model to validate a condition."""

    type: Literal["prompt"] = "prompt"
    prompt: str
    model: str | None = None
    timeout_seconds: int = Field(default=30, ge=1, le=600)
    matcher: str | None = None
    block_on_failure: bool = True


class HttpHookDefinition(BaseModel):
    """A hook that POSTs the event payload to an HTTP endpoint."""

    type: Literal["http"] = "http"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=30, ge=1, le=600)
    matcher: str | None = None
    block_on_failure: bool = False


class AgentHookDefinition(BaseModel):
    """A hook that performs a deeper model-based validation."""

    type: Literal["agent"] = "agent"
    prompt: str
    model: str | None = None
    timeout_seconds: int = Field(default=60, ge=1, le=1200)
    matcher: str | None = None
    block_on_failure: bool = True


HookDefinition = (
    CommandHookDefinition
    | PromptHookDefinition
    | HttpHookDefinition
    | AgentHookDefinition
)
