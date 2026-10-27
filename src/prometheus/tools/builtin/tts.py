# Provenance: NousResearch/hermes-agent (https://github.com/NousResearch/hermes-agent)
# Original: tools/tts_tool.py
# License: MIT
# Modified: Rewritten as Prometheus BaseTool; local TTS engines (espeak-ng, piper),
#           with an optional OpenAI-compatible remote (tts.url) tried first

"""Text-to-speech using local TTS engines (espeak-ng or piper)."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.permissions.path_schema import PATH_FIELD_WRITE
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class TTSInput(BaseModel):
    """Arguments for text-to-speech."""

    text: str = Field(description="Text to convert to speech")
    voice: str | None = Field(
        default=None,
        description="Voice name (engine-specific, e.g. 'en-us+f3' for espeak-ng)",
    )
    output_path: str | None = Field(
        json_schema_extra=PATH_FIELD_WRITE,
        default=None, description="Output file path; auto-generates a temp WAV if omitted"
    )
    engine: str | None = Field(
        default=None,
        description="TTS engine to use: 'espeak' or 'piper'. Auto-detects if omitted.",
    )


class TTSTool(BaseTool):
    """Convert text to speech using a local TTS engine."""

    name = "tts"
    description = (
        "Convert text to speech audio using a local engine (espeak-ng or piper). "
        "Returns the path to the generated audio file."
    )
    input_model = TTSInput
    find_me = ("convert text to speech", "read text aloud")

    async def execute(
        self, arguments: TTSInput, context: ToolExecutionContext
    ) -> ToolResult:
        text = arguments.text[:4000]  # cap at 4000 chars

        # A relative output_path resolves against the tool context's cwd (the
        # session workspace in a workspace session), like write_file. Handed
        # on as given, the engine subprocess resolved it against the daemon
        # PROCESS's directory instead.
        if arguments.output_path:
            out_path = str(_resolve_path(context.cwd, arguments.output_path))
        else:
            out_path = tempfile.mktemp(suffix=".wav")

        # tts.url, when set, is tried first unless a local engine was asked
        # for by name. Any failure falls through to the local engines below.
        if arguments.engine is None:
            from prometheus.infra.voice_remote import remote_settings, synthesize_remote

            remote = remote_settings("tts")
            if remote is not None and await synthesize_remote(
                remote, text, out_path, arguments.voice,
            ):
                return ToolResult(output=f"Audio saved to {out_path}")

        engine = arguments.engine or _detect_engine()
        if engine is None:
            return ToolResult(
                output=(
                    "No TTS engine found. Install espeak-ng, or piper with: "
                    "pip install 'oara-prometheus[voice]'"
                ),
                is_error=True,
            )
        if engine == "piper" and not shutil.which("piper"):
            # Asked for by name: without this the exec below raises
            # FileNotFoundError instead of saying what to install.
            return ToolResult(
                output=(
                    "piper is not installed. "
                    "Fix: pip install 'oara-prometheus[voice]'"
                ),
                is_error=True,
            )

        if engine == "espeak":
            return await _run_espeak(text, arguments.voice, out_path)
        elif engine == "piper":
            return await _run_piper(text, arguments.voice, out_path)
        else:
            return ToolResult(
                output=f"Unknown TTS engine: {engine}. Use 'espeak' or 'piper'.",
                is_error=True,
            )


def _resolve_path(base: Path, candidate: str) -> Path:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = Path(base) / path
    return path.resolve()


def _detect_engine() -> str | None:
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        return "espeak"
    if shutil.which("piper"):
        return "piper"
    return None


async def _run_espeak(text: str, voice: str | None, out_path: str) -> ToolResult:
    cmd = shutil.which("espeak-ng") or shutil.which("espeak") or "espeak-ng"
    args = [cmd, "-w", out_path]
    if voice:
        args.extend(["-v", voice])
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await asyncio.wait_for(proc.communicate(text.encode()), timeout=30)
    if proc.returncode != 0:
        return ToolResult(
            output=f"espeak failed (rc={proc.returncode}): {stderr.decode()[:500]}",
            is_error=True,
        )
    return ToolResult(output=f"Audio saved to {out_path}")


async def _run_piper(text: str, voice: str | None, out_path: str) -> ToolResult:
    args = ["piper", "--output_file", out_path]
    if voice:
        args.extend(["--model", voice])
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await asyncio.wait_for(proc.communicate(text.encode()), timeout=60)
    if proc.returncode != 0:
        return ToolResult(
            output=f"piper failed (rc={proc.returncode}): {stderr.decode()[:500]}",
            is_error=True,
        )
    return ToolResult(output=f"Audio saved to {out_path}")
