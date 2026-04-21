"""SandboxedExecution — locked bash execution for Sprint 4.

Wraps subprocess execution with:
- Working directory locked to workspace_root
- Environment variable sanitization (strips API keys / tokens)
- Configurable timeout enforcement
- Output size limits
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from prometheus.tools.base import ToolResult

# Env var key patterns to strip from the subprocess environment
_SENSITIVE_KEY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r".*_API_KEY$", re.IGNORECASE),
    re.compile(r".*_TOKEN$", re.IGNORECASE),
    re.compile(r".*_SECRET$", re.IGNORECASE),
    re.compile(r".*_PASSWORD$", re.IGNORECASE),
    re.compile(r"AWS_.*", re.IGNORECASE),
    re.compile(r"ANTHROPIC_.*", re.IGNORECASE),
    re.compile(r"OPENAI_.*", re.IGNORECASE),
]

_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_OUTPUT = 10_000


def _sanitize_env(env: dict[str, str]) -> dict[str, str]:
    """Return a copy of env with sensitive keys removed."""
    return {
        k: v
        for k, v in env.items()
        if not any(p.match(k) for p in _SENSITIVE_KEY_PATTERNS)
    }


class SandboxedExecution:
    """Execute shell commands in a sandboxed environment.

    Example:
        sandbox = SandboxedExecution(workspace="/tmp/prometheus-workspace")
        result = await sandbox.run("ls -la")
        print(result.output)
    """

    def __init__(
        self,
        workspace: str | Path,
        timeout: int = _DEFAULT_TIMEOUT,
        max_output: int = _DEFAULT_MAX_OUTPUT,
        strip_env_keys: list[str] | None = None,
    ) -> None:
        self._workspace = Path(workspace).expanduser().resolve()
        self._timeout = timeout
        self._max_output = max_output
        # Extra literal keys to strip (in addition to pattern-matched ones)
        self._extra_strip: set[str] = set(strip_env_keys or [])

    async def run(
        self,
        command: str,
        env_override: dict[str, str] | None = None,
    ) -> ToolResult:
        """Run a shell command inside the sandbox.

        Args:
            command: Shell command string (passed to /bin/bash -c).
            env_override: Optional env vars to merge in (after sanitization).

        Returns:
            ToolResult with combined stdout+stderr, is_error=True on failure.
        """
        base_env = _sanitize_env(dict(os.environ))
        # Remove extra literal keys
        for key in self._extra_strip:
            base_env.pop(key, None)
        if env_override:
            base_env.update(env_override)

        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-c",
            command,
            cwd=str(self._workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=base_env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ToolResult(
                output=f"Command timed out after {self._timeout}s",
                is_error=True,
                metadata={"returncode": -1, "timed_out": True},
            )

        parts = []
        if stdout:
            parts.append(stdout.decode("utf-8", errors="replace").rstrip())
        if stderr:
            parts.append(stderr.decode("utf-8", errors="replace").rstrip())

        text = "\n".join(p for p in parts if p).strip() or "(no output)"

        if len(text) > self._max_output:
            text = text[: self._max_output] + f"\n...[output truncated at {self._max_output} chars]"

        return ToolResult(
            output=text,
            is_error=process.returncode != 0,
            metadata={"returncode": process.returncode},
        )

    @property
    def workspace(self) -> Path:
        return self._workspace
