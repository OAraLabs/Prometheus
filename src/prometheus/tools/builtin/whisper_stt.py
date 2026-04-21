"""WhisperSTT — speech-to-text via local Whisper binary.

Counterpart to the existing TTSTool (tts.py).  Accepts an audio file path
(typically .ogg from Telegram voice memos), converts to WAV if needed via
ffmpeg, then transcribes via whisper CLI or faster-whisper.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)


class WhisperSTTInput(BaseModel):
    audio_path: str = Field(description="Path to audio file (.ogg, .wav, .mp3)")
    language: str = Field(default="en", description="Language code (e.g. 'en', 'de')")
    model: str = Field(default="base", description="Whisper model size (tiny, base, small, medium, large)")


class WhisperSTTTool(BaseTool):
    """Transcribe audio to text using local Whisper."""

    name = "whisper_stt"
    description = (
        "Transcribe an audio file to text using Whisper speech-to-text. "
        "Supports .ogg, .wav, .mp3 formats."
    )
    input_model = WhisperSTTInput

    async def execute(self, arguments: WhisperSTTInput, context: ToolExecutionContext) -> ToolResult:
        audio_path = Path(arguments.audio_path)
        if not audio_path.is_file():
            return ToolResult(output=f"Audio file not found: {arguments.audio_path}", is_error=True)

        # Convert to WAV if not already WAV (Whisper prefers WAV)
        wav_path = audio_path
        tmp_wav = None
        if audio_path.suffix.lower() not in (".wav",):
            wav_path, tmp_wav = await self._convert_to_wav(audio_path)
            if wav_path is None:
                return ToolResult(output="Failed to convert audio to WAV (is ffmpeg installed?)", is_error=True)

        try:
            result = await self._transcribe(wav_path, arguments.language, arguments.model)
            return result
        finally:
            if tmp_wav and Path(tmp_wav).exists():
                Path(tmp_wav).unlink(missing_ok=True)

    async def _convert_to_wav(self, audio_path: Path) -> tuple[Path | None, str | None]:
        """Convert audio to WAV via ffmpeg. Returns (wav_path, tmp_path_to_clean)."""
        if not shutil.which("ffmpeg"):
            return None, None

        tmp = tempfile.mktemp(suffix=".wav")
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", str(audio_path), "-ar", "16000", "-ac", "1", "-y", tmp,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return None, None

        if proc.returncode != 0:
            logger.warning("ffmpeg failed: %s", stderr.decode(errors="replace")[:200])
            return None, None

        return Path(tmp), tmp

    async def _transcribe(self, wav_path: Path, language: str, model: str) -> ToolResult:
        """Run Whisper on a WAV file."""
        # Try faster-whisper first, then whisper CLI
        engine = _detect_whisper_engine()
        if engine is None:
            return ToolResult(
                output="No Whisper engine found. Install 'whisper' or 'faster-whisper-xxl'.",
                is_error=True,
            )

        if engine == "faster-whisper":
            return await self._run_faster_whisper(wav_path, language, model)
        return await self._run_whisper_cli(wav_path, language, model)

    async def _run_whisper_cli(self, wav_path: Path, language: str, model: str) -> ToolResult:
        """Run openai/whisper CLI."""
        proc = await asyncio.create_subprocess_exec(
            "whisper", str(wav_path),
            "--model", model,
            "--language", language,
            "--output_format", "txt",
            "--output_dir", str(wav_path.parent),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult(output="Whisper transcription timed out (120s)", is_error=True)

        if proc.returncode != 0:
            return ToolResult(
                output=f"Whisper failed: {stderr.decode(errors='replace')[:500]}",
                is_error=True,
            )

        # Whisper writes a .txt file alongside the input
        txt_path = wav_path.with_suffix(".txt")
        if txt_path.exists():
            text = txt_path.read_text(encoding="utf-8").strip()
            txt_path.unlink(missing_ok=True)
            return ToolResult(output=text)

        # Fallback: try stdout
        text = stdout.decode(errors="replace").strip()
        return ToolResult(output=text or "(empty transcription)")

    async def _run_faster_whisper(self, wav_path: Path, language: str, model: str) -> ToolResult:
        """Run faster-whisper-xxl or faster-whisper CLI."""
        cmd = shutil.which("faster-whisper-xxl") or shutil.which("faster-whisper")
        proc = await asyncio.create_subprocess_exec(
            cmd, str(wav_path),
            "--model", model,
            "--language", language,
            "--output_format", "txt",
            "--output_dir", str(wav_path.parent),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult(output="faster-whisper transcription timed out (120s)", is_error=True)

        if proc.returncode != 0:
            return ToolResult(
                output=f"faster-whisper failed: {stderr.decode(errors='replace')[:500]}",
                is_error=True,
            )

        txt_path = wav_path.with_suffix(".txt")
        if txt_path.exists():
            text = txt_path.read_text(encoding="utf-8").strip()
            txt_path.unlink(missing_ok=True)
            return ToolResult(output=text)

        text = stdout.decode(errors="replace").strip()
        return ToolResult(output=text or "(empty transcription)")

    def is_read_only(self, arguments: BaseModel) -> bool:
        return True


def _detect_whisper_engine() -> str | None:
    """Detect which Whisper engine is available."""
    if shutil.which("faster-whisper-xxl") or shutil.which("faster-whisper"):
        return "faster-whisper"
    if shutil.which("whisper"):
        return "whisper"
    return None
