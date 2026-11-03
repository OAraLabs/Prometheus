"""When a remote voice server goes away, voice falls back to this computer and SAYS SO, once; and once when it is back.

Contract: docs/contracts/pieces.md, section 9. The per-outage state is the smallest that does it: which remotes are
down, by kind and url. The call that finds a remote down (when it was up) gets one line for the person; the call that
finds it answering again (when it was down) gets one more. Every failed request still leaves its own log line, exactly
as before: that request-level logging is not changed here.

Where the line goes: a gateway sends it to the chat that used voice, the terminal prints it, and a tool the model
called carries it in its result.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from prometheus.tools.base import ToolExecutionContext, ToolResult

LOGGER = "prometheus.infra.voice_remote"
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt remote-audio"


class _Server:
    def __init__(self) -> None:
        self.status = 200
        self.down = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "nope"})
        if request.url.path.endswith("/audio/transcriptions"):
            return httpx.Response(200, json={"text": "remote words"})
        return httpx.Response(200, content=WAV, headers={"content-type": "audio/wav"})


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    from prometheus.infra import voice_remote

    monkeypatch.setattr(voice_remote, "_down", set(), raising=False)
    for name in ("PROMETHEUS_STT_API_KEY", "PROMETHEUS_TTS_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def server(monkeypatch):
    srv = _Server()
    real = httpx.AsyncClient

    def _async(timeout):
        return real(transport=httpx.MockTransport(srv.handler), timeout=timeout)

    monkeypatch.setattr("prometheus.infra.voice_remote._async_client", _async)
    return srv


def settings(kind: str, url: str = "http://voice.example:8001"):
    from prometheus.infra.voice_remote import remote_settings

    remote = remote_settings(kind, {kind: {"url": url}})
    assert remote is not None
    return remote


def speak(remote, tmp_path, notices):
    from prometheus.infra.voice_remote import synthesize_remote

    return asyncio.run(synthesize_remote(remote, "hello", tmp_path / "out.wav", notices=notices))


def listen(remote, tmp_path, notices):
    from prometheus.infra.voice_remote import transcribe_remote

    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 64)
    return asyncio.run(transcribe_remote(remote, audio, notices=notices))


# ── the transitions ──────────────────────────────────────────────────────────

def test_one_line_when_it_goes_down_and_none_while_it_stays_down(server, tmp_path):
    remote = settings("tts")
    server.down = True
    first: list[str] = []
    second: list[str] = []
    assert speak(remote, tmp_path, first) is False
    assert speak(remote, tmp_path, second) is False
    assert len(first) == 1
    assert "isn't answering" in first[0]
    assert "this computer's own voice" in first[0]
    assert second == []


def test_one_line_when_it_is_back_and_none_while_it_stays_up(server, tmp_path):
    remote = settings("tts")
    server.down = True
    speak(remote, tmp_path, [])
    server.down = False
    back: list[str] = []
    again: list[str] = []
    assert speak(remote, tmp_path, back) is True
    assert speak(remote, tmp_path, again) is True
    assert len(back) == 1
    assert "answering again" in back[0]
    assert again == []


def test_a_healthy_remote_says_nothing(server, tmp_path):
    notices: list[str] = []
    assert listen(settings("stt"), tmp_path, notices) == "remote words"
    assert notices == []


def test_a_refused_key_says_so_in_its_own_words(server, tmp_path):
    server.status = 401
    notices: list[str] = []
    assert listen(settings("stt"), tmp_path, notices) is None
    assert len(notices) == 1
    assert "refused" in notices[0] and "key" in notices[0]
    assert "speech-to-text" in notices[0]


def test_speech_to_text_and_voice_are_tracked_apart(server, tmp_path):
    server.down = True
    stt: list[str] = []
    tts: list[str] = []
    listen(settings("stt"), tmp_path, stt)
    speak(settings("tts"), tmp_path, tts)
    assert len(stt) == 1 and len(tts) == 1


def test_the_state_moves_even_when_the_caller_cannot_relay_a_line(server, tmp_path):
    remote = settings("tts")
    server.down = True
    speak(remote, tmp_path, None)
    later: list[str] = []
    speak(remote, tmp_path, later)
    assert later == []                      # already known to be down


def test_every_failed_request_still_logs_its_own_line(server, tmp_path, caplog):
    remote = settings("tts")
    server.down = True
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        speak(remote, tmp_path, [])
        speak(remote, tmp_path, [])
    assert len([r for r in caplog.records if r.name == LOGGER]) == 2


# ── the call sites ───────────────────────────────────────────────────────────

def _stt_tool(tmp_path, notices=None):
    from prometheus.tools.builtin.whisper_stt import WhisperSTTInput, WhisperSTTTool

    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 64)
    local = AsyncMock(return_value=ToolResult(output="local words"))
    with patch.object(WhisperSTTTool, "_transcribe", local):
        return asyncio.run(WhisperSTTTool(config={"stt": {"url": "http://voice.example:8001"}}, notices=notices)
                           .execute(WhisperSTTInput(audio_path=str(audio)), ToolExecutionContext(cwd=tmp_path)))


def test_a_gateways_transcription_stays_clean_and_the_line_goes_to_its_list(server, tmp_path):
    server.down = True
    notices: list[str] = []
    result = _stt_tool(tmp_path, notices)
    assert result.output == "local words"
    assert len(notices) == 1


def test_the_model_called_stt_tool_carries_the_line_in_its_result(server, tmp_path):
    server.down = True
    result = _stt_tool(tmp_path)
    assert result.output.startswith("local words")
    assert "isn't answering" in result.output


def test_the_tts_tool_carries_the_line_in_its_result(server, tmp_path, monkeypatch):
    import yaml

    from prometheus.tools.builtin.tts import TTSInput, TTSTool

    cfg_dir = tmp_path / "prom-config"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "prometheus.yaml").write_text(yaml.safe_dump({"tts": {"url": "http://voice.example:8003"}}))
    server.down = True

    async def _espeak(text, voice, out_path):
        Path(out_path).write_bytes(b"local-audio")
        return ToolResult(output=f"Audio saved to {out_path}")

    with patch("prometheus.tools.builtin.tts._detect_engine", return_value="espeak"), \
            patch("prometheus.tools.builtin.tts._run_espeak", AsyncMock(side_effect=_espeak)):
        result = asyncio.run(TTSTool().execute(
            TTSInput(text="hello", output_path=str(tmp_path / "out.wav")), ToolExecutionContext(cwd=tmp_path)))
    assert not result.is_error
    assert "Audio saved to" in result.output
    assert "isn't answering" in result.output


def test_the_terminal_prints_the_line(server, tmp_path, capsys):
    from prometheus.cli import voice

    server.down = True
    local_wav = tmp_path / "local.wav"
    local_wav.write_bytes(b"local-audio")
    with patch.object(voice, "_synthesize_piper", AsyncMock(return_value=local_wav)):
        asyncio.run(voice.synthesize_wav("hello", {"model_path": "x.onnx"},
                                         config={"tts": {"url": "http://voice.example:8003"}}))
    assert "isn't answering" in capsys.readouterr().out


def test_telegram_voice_replies_hand_back_the_line(server, tmp_path):
    from prometheus.gateway.telegram import TelegramAdapter

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._prometheus_config = {"tts": {"url": "http://voice.example:8003"},
                                  "gateway": {"voice": {"engine": "piper", "model_path": "x.onnx"}}}
    server.down = True

    async def _piper(text, out_wav, model_path):
        Path(out_wav).write_bytes(b"local-audio")
        return True

    async def _opus(wav_path, bitrate):
        ogg = tmp_path / "out.ogg"
        ogg.write_bytes(b"ogg")
        return str(ogg)

    notices: list[str] = []
    with patch.object(adapter, "_run_piper", AsyncMock(side_effect=_piper)), \
            patch.object(adapter, "_wav_to_opus_ogg", side_effect=_opus):
        assert asyncio.run(adapter._synthesize_voice("hello", notices=notices)) is not None
    assert len(notices) == 1


def test_gateway_transcription_hands_back_the_line(server, tmp_path):
    from prometheus.gateway import media_services
    from prometheus.tools.builtin.whisper_stt import WhisperSTTTool

    audio = tmp_path / "in.ogg"
    audio.write_bytes(b"OggS" + b"\x00" * 64)
    server.down = True
    notices: list[str] = []
    with patch.object(WhisperSTTTool, "_convert_to_wav", AsyncMock(return_value=(audio, None))), \
            patch.object(WhisperSTTTool, "_transcribe", AsyncMock(return_value=ToolResult(output="local words"))):
        text = asyncio.run(media_services.transcribe_audio(
            str(audio), config={"stt": {"url": "http://voice.example:8001"}}, notices=notices))
    assert text == "local words"
    assert len(notices) == 1
