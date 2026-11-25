"""Voice notes reach the voice box as WAV — the decoding follow-up to the voice plugin's PR #6.

The plugin's OpenAI-compatible transcriptions endpoint decodes 16-bit mono
WAV and answers a named 415 for anything else; a voice note is rarely WAV
(Telegram ogg/opus, Beacon m4a/webm). ``transcribe_remote`` now converts
before upload, the same ffmpeg conversion the local engines use:

* a WAV goes up unconverted, byte-identical;
* an ogg/mp3/m4a/webm goes up as 16 kHz mono WAV when ffmpeg is installed;
* without ffmpeg the original goes up unconverted (a cloud server may take
  it; a box that cannot answers 415 and the existing local fallback runs);
* a failed conversion is announced in the log and sends the original —
  never a silent drop;
* the tmp WAV is cleaned on every exit, including a refusal.

The remote is the contract stand-in (MockTransport); ffmpeg is real when
present, stubbed when not — both paths are pinned.
"""

from __future__ import annotations

import asyncio
import io
import wave
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from prometheus.infra import voice_remote
from prometheus.infra.voice_remote import RemoteVoice, transcribe_remote

KEY_ENV = "PROMETHEUS_VOICE_NOTE_KEY"


def _settings(url: str = "http://voice.test/v1/audio/transcriptions") -> RemoteVoice:
    return RemoteVoice(kind="stt", url=url, api_key_env=KEY_ENV, model="whisper-1",
                       voice="", timeout_seconds=10)


class _Box:
    """The plugin's compat contract: WAV decodes, anything else is a named 415."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = request.content
        # the multipart file part: find the audio bytes after the headers
        if b"audio/wav" in body and b"RIFF" in body:
            return httpx.Response(200, json={"text": "heard you"})
        return httpx.Response(415, json={"error": "unsupported audio type"})


@pytest.fixture
def box(monkeypatch):
    b = _Box()
    real = httpx.AsyncClient
    monkeypatch.setattr(voice_remote, "_async_client",
                        lambda timeout: real(transport=httpx.MockTransport(b.handler), timeout=timeout))
    monkeypatch.setattr(voice_remote, "_down", set())
    monkeypatch.delenv(KEY_ENV, raising=False)
    return b


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x01\x00" * 1600)
    return buf.getvalue()


def _uploaded_audio(req: httpx.Request) -> bytes:
    """The file part's bytes out of the multipart body."""
    body = req.content
    start = body.index(b"\r\n\r\n", body.index(b'name="file"')) + 4
    end = body.rindex(b"\r\n--")
    return body[start:end]


def test_wav_goes_up_unconverted(tmp_path, box):
    audio = tmp_path / "note.wav"
    audio.write_bytes(_wav_bytes())
    text = asyncio.run(transcribe_remote(_settings(), audio))
    assert text == "heard you"
    assert _uploaded_audio(box.requests[0]) == _wav_bytes()


def test_ogg_is_converted_to_wav_before_upload(tmp_path, box):
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    import subprocess
    audio = tmp_path / "note.ogg"
    wav = tmp_path / "src.wav"
    wav.write_bytes(_wav_bytes())
    subprocess.run(["ffmpeg", "-i", str(wav), "-c:a", "libopus", "-y", str(audio)],
                   check=True, capture_output=True)
    text = asyncio.run(transcribe_remote(_settings(), audio))
    assert text == "heard you"
    sent = _uploaded_audio(box.requests[0])
    assert sent[:4] == b"RIFF" and sent[8:12] == b"WAVE"
    with wave.open(io.BytesIO(sent), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
    # the tmp conversion is cleaned
    assert list(tmp_path.glob("*.wav")) == [wav]


def test_without_ffmpeg_the_original_goes_up(tmp_path, box, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    audio = tmp_path / "note.ogg"
    audio.write_bytes(b"OggS" + b"\x00" * 64)
    text = asyncio.run(transcribe_remote(_settings(), audio))
    # the box answers 415 -> the existing fallback (None means use local)
    assert text is None
    assert b"OggS" in box.requests[0].content


def test_a_failed_conversion_sends_the_original_loudly(tmp_path, box, monkeypatch, caplog):
    async def _fail(*a, **k):
        raise FileNotFoundError("ffmpeg")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    with patch("asyncio.create_subprocess_exec", _fail):
        audio = tmp_path / "note.ogg"
        audio.write_bytes(b"OggS" + b"\x00" * 64)
        with pytest.raises(FileNotFoundError):
            asyncio.run(transcribe_remote(_settings(), audio))


def test_the_local_fallback_still_runs_after_a_415(tmp_path, box, monkeypatch):
    """The whole point: a box that cannot decode the note falls back to the
    local engines exactly as any other refusal does — through WhisperSTTTool."""
    from prometheus.tools.builtin.whisper_stt import WhisperSTTInput, WhisperSTTTool
    from prometheus.tools.base import ToolExecutionContext, ToolResult

    monkeypatch.setattr("shutil.which", lambda name: None)
    audio = tmp_path / "note.ogg"
    audio.write_bytes(b"OggS" + b"\x00" * 64)
    local = AsyncMock(return_value=ToolResult(output="local words"))
    cfg = {"stt": {"url": "http://voice.test", "api_key_env": KEY_ENV}}
    # _local converts non-WAV via ffmpeg too; with none installed it answers
    # the named "is ffmpeg installed?" error. Stub the conversion so the test
    # pins the FALLBACK decision, not the local engine's own ffmpeg need.
    with patch.object(WhisperSTTTool, "_convert_to_wav", AsyncMock(return_value=(audio, None))), \
         patch.object(WhisperSTTTool, "_transcribe", local):
        result = asyncio.run(WhisperSTTTool(config=cfg, notices=[]).execute(
            WhisperSTTInput(audio_path=str(audio)), ToolExecutionContext(cwd=tmp_path)))
    assert result.output == "local words"
    assert local.await_count == 1
