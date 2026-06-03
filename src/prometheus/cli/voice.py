"""CLI voice-mode primitives — mic capture, playback, STT, TTS.

A counterpart to the Telegram gateway's voice plumbing (see
``gateway/telegram.py`` :class:`TelegramAdapter` voice section). The
Telegram side encodes opus-in-ogg for Telegram's voice-message format;
the CLI side stays in raw 16-bit PCM WAV because we play it back
directly through the host's audio output, no re-encoding needed.

Shared design choices (kept in sync with the Telegram impl):

- **STT** via the existing :class:`WhisperSTTTool` — same engine, same
  config. The tool already handles ogg↔wav conversion and chooses
  faster-whisper vs whisper CLI based on what's installed.
- **TTS** via piper directly (subprocess), not via the ``TTSTool`` — we
  need the raw WAV output path, and the tool's interface returns a
  string message instead. Half a dozen lines of subprocess is cheaper
  than adapting the tool's return shape.
- **Text stripping** matches ``TelegramAdapter._strip_text_for_tts``
  so the spoken reply sounds the same regardless of channel.

Usage from the REPL::

    text = await cli_voice_capture()                # mic → Whisper → text
    # ... feed text to agent loop, get response ...
    await cli_voice_speak(response_text, voice_config)

Both helpers degrade gracefully — missing mic, missing piper binary,
or a bad model path returns ``None`` / ``False`` and the caller falls
back to plain typing.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 16 kHz mono matches Whisper's expected input — no resampling needed
# downstream and the WAV stays small (32 KB/sec, ~2 MB/min).
SAMPLE_RATE = 16_000
CHANNELS = 1


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def get_voice_config(prometheus_config: dict[str, Any] | None) -> dict[str, Any]:
    """Pull the ``gateway.voice`` block out of the loaded yaml config.

    Mirrors :meth:`TelegramAdapter._voice_config` so CLI and Telegram
    share the same engine/model/max_chars/bitrate settings — toggling
    a voice in one place reflects in both.
    """
    if not isinstance(prometheus_config, dict):
        return {}
    gw = prometheus_config.get("gateway", {})
    if not isinstance(gw, dict):
        return {}
    voice = gw.get("voice", {})
    return voice if isinstance(voice, dict) else {}


def strip_text_for_tts(text: str) -> str:
    """Strip markdown/code/URLs before TTS so piper doesn't read symbols.

    Kept in sync with :meth:`TelegramAdapter._strip_text_for_tts`.
    Code fences are dropped wholesale (TTS reading source code is
    unintelligible), URLs collapse to "link", everything else has its
    formatting markers removed but the prose preserved.
    """
    text = re.sub(r"```[\s\S]*?```", " [code omitted] ", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"^\s*[\*\-•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"https?://\S+", "link", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Mic capture (push-to-talk)
# ---------------------------------------------------------------------------

async def record_push_to_talk(max_seconds: int = 60) -> Path | None:
    """Capture mic audio until the user presses Enter (or *max_seconds*).

    Push-to-talk variant: caller arranges for the prompt to be visible,
    we open the stream and block on stdin. Returns the path to a
    16 kHz mono 16-bit PCM WAV file, or ``None`` if no audio was
    captured or the device couldn't be opened.

    The Enter-to-stop signal arrives via ``input()`` in an executor
    thread so the sounddevice callback (which fires on the audio
    thread) can keep accumulating buffers in parallel. Without the
    executor hop, blocking on ``input()`` would freeze the asyncio
    loop and the audio thread would still record fine — but the *next*
    coroutine wouldn't run until the user hit Enter, which is fine
    here, just stylistically inconsistent.
    """
    try:
        import numpy as np
        import sounddevice as sd
        from scipy.io import wavfile
    except ImportError as exc:
        print(f"\n[voice] missing audio dependency: {exc}")
        return None

    buffer: list[Any] = []

    def callback(indata, frames, time_info, status):
        if status:
            # Underruns/overflows show up here. Log once per status flag.
            logger.debug("sounddevice status: %s", status)
        buffer.append(indata.copy())

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            callback=callback,
        )
    except Exception as exc:
        print(f"\n[voice] failed to open mic: {exc}")
        return None

    try:
        with stream:
            print("🎤 Recording... press Enter to stop", flush=True)
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline,
                )
            except (KeyboardInterrupt, EOFError):
                print(" (cancelled)")
                return None
    except Exception as exc:
        print(f"\n[voice] capture failed: {exc}")
        return None

    if not buffer:
        print("  (no audio captured)")
        return None

    audio = np.concatenate(buffer, axis=0)
    duration = len(audio) / SAMPLE_RATE
    if duration < 0.2:
        print("  (too short — under 0.2s)")
        return None

    out_path = tempfile.mktemp(suffix=".wav")
    try:
        wavfile.write(out_path, SAMPLE_RATE, audio)
    except Exception as exc:
        print(f"\n[voice] failed to write WAV: {exc}")
        return None

    print(f"  captured {duration:.1f}s")
    return Path(out_path)


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

def play_wav(wav_path: str | Path) -> bool:
    """Play a WAV file through the default audio output. Blocks until done.

    Returns ``True`` if playback completed without error. Falls back
    silently to ffplay if sounddevice can't open the file (rare —
    happens when scipy chokes on a non-standard WAV format).
    """
    p = Path(wav_path)
    if not p.is_file():
        return False

    try:
        import sounddevice as sd
        from scipy.io import wavfile

        sample_rate, data = wavfile.read(str(p))
        sd.play(data, sample_rate)
        sd.wait()
        return True
    except Exception as exc:
        logger.debug("sounddevice playback failed: %s — trying ffplay", exc)

    # Fallback: ffplay (autoexit, no display, quiet)
    if shutil.which("ffplay"):
        try:
            import subprocess
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(p)],
                check=True,
            )
            return True
        except Exception as exc:
            logger.debug("ffplay playback failed: %s", exc)
    return False


# ---------------------------------------------------------------------------
# STT (Whisper) — reuses the existing tool
# ---------------------------------------------------------------------------

async def transcribe_wav(wav_path: str | Path) -> str | None:
    """Transcribe a WAV via :class:`WhisperSTTTool`. Returns text or ``None``."""
    try:
        from prometheus.tools.base import ToolExecutionContext
        from prometheus.tools.builtin.whisper_stt import (
            WhisperSTTInput,
            WhisperSTTTool,
        )
    except ImportError as exc:
        logger.warning("Whisper STT module unavailable: %s", exc)
        return None

    tool = WhisperSTTTool()
    result = await tool.execute(
        WhisperSTTInput(audio_path=str(wav_path)),
        ToolExecutionContext(cwd=Path.cwd()),
    )
    if result.is_error or not result.output:
        return None
    text = result.output.strip()
    return text or None


# ---------------------------------------------------------------------------
# TTS (Piper) — direct subprocess for raw-WAV output
# ---------------------------------------------------------------------------

async def synthesize_wav(
    text: str, voice_config: dict[str, Any],
) -> Path | None:
    """Synthesize *text* to a WAV via piper. Returns path or ``None``."""
    piper_bin = shutil.which("piper")
    if piper_bin is None:
        logger.debug("piper binary not on PATH — voice output disabled")
        return None
    model_path = voice_config.get("model_path")
    if not model_path:
        logger.debug("No piper model_path configured in gateway.voice")
        return None
    model = Path(model_path).expanduser()
    if not model.is_file():
        logger.warning("Piper model not found: %s", model)
        return None

    out_wav = tempfile.mktemp(suffix=".wav")
    proc = await asyncio.create_subprocess_exec(
        piper_bin, "-m", str(model), "-f", out_wav,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(
            proc.communicate(text.encode("utf-8")), timeout=60,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        logger.warning("piper timed out after 60s")
        return None
    if proc.returncode != 0:
        logger.warning(
            "piper failed (rc=%s): %s",
            proc.returncode, stderr.decode(errors="replace")[:300],
        )
        return None
    out = Path(out_wav)
    if not out.is_file() or out.stat().st_size == 0:
        return None
    return out


# ---------------------------------------------------------------------------
# High-level convenience: one voice turn end-to-end
# ---------------------------------------------------------------------------

async def cli_voice_capture(max_seconds: int = 60) -> str | None:
    """Record one mic turn and return the transcribed text (or None)."""
    wav_path = await record_push_to_talk(max_seconds=max_seconds)
    if wav_path is None:
        return None
    try:
        text = await transcribe_wav(wav_path)
        if text:
            print(f"  You: {text}")
        else:
            print("  (transcription empty — try again)")
        return text
    finally:
        wav_path.unlink(missing_ok=True)


async def cli_voice_speak(
    text: str, voice_config: dict[str, Any],
) -> bool:
    """Synthesize *text* via TTS and play it. Returns True on success."""
    clean = strip_text_for_tts(text)
    if not clean:
        return False
    max_chars = int(voice_config.get("max_chars", 800))
    if len(clean) > max_chars:
        clean = clean[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    wav_path = await synthesize_wav(clean, voice_config)
    if wav_path is None:
        return False
    try:
        return play_wav(wav_path)
    finally:
        wav_path.unlink(missing_ok=True)
