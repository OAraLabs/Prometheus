"""Optional remote voice servers: ``stt.url`` and ``tts.url``.

Each path is pinned at every call site that speaks or listens:

* unset        — the local engines run exactly as before; no request is made
* reachable    — the remote answers, the local engine is never touched, and
                 the bearer key comes from the env file (never the yaml)
* unreachable  — falls back to local with ONE log line, no ERROR
* 401 / 403    — falls back to local, but the refusal is logged at ERROR
* doctor       — one row per configured remote saying whether it answers

The call sites: ``WhisperSTTTool`` (every STT caller goes through it), the
``tts`` tool, CLI voice mode (``cli.voice.synthesize_wav``) and Telegram voice
replies (``TelegramAdapter._synthesize_voice``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import yaml

from prometheus.tools.base import ToolExecutionContext, ToolResult

LOGGER = "prometheus.infra.voice_remote"
KEY_ENV = "PROMETHEUS_VOICE_TEST_KEY"
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt remote-audio"


def _closed_port() -> int:
    """A loopback port nothing listens on: connecting is refused at once."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _cfg(kind: str, url: str, **extra: Any) -> dict[str, Any]:
    return {kind: {"url": url, "api_key_env": KEY_ENV, **extra}}


@pytest.fixture
def env_key(tmp_path, monkeypatch):
    """The key lives in the env file only (conftest points it at tmp)."""
    monkeypatch.delenv(KEY_ENV, raising=False)
    Path(tmp_path / "prom-env").write_text(f"{KEY_ENV}=sekrit-from-env-file\n")
    return "sekrit-from-env-file"


class _Server:
    """A MockTransport stand-in for the remote voice server."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "nope"})
        if request.url.path.endswith("/audio/transcriptions"):
            return httpx.Response(200, json={"text": "remote words"})
        if request.url.path.endswith("/audio/speech"):
            return httpx.Response(200, content=WAV,
                                  headers={"content-type": "audio/wav"})
        return httpx.Response(200, json={"data": []})


@pytest.fixture
def server(monkeypatch):
    srv = _Server()
    real_async, real_sync = httpx.AsyncClient, httpx.Client

    def _async(timeout):
        return real_async(transport=httpx.MockTransport(srv.handler), timeout=timeout)

    def _sync(timeout):
        return real_sync(transport=httpx.MockTransport(srv.handler), timeout=timeout)

    monkeypatch.setattr("prometheus.infra.voice_remote._async_client", _async)
    monkeypatch.setattr("prometheus.infra.voice_remote._sync_client", _sync)
    return srv


def _records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER]


# ---------------------------------------------------------------------------
# Call-site drivers. Each returns (result, local_was_used).
# ---------------------------------------------------------------------------

def _run_stt(config: dict[str, Any] | None, tmp_path: Path):
    from prometheus.tools.builtin.whisper_stt import WhisperSTTInput, WhisperSTTTool

    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 64)
    local = AsyncMock(return_value=ToolResult(output="local words"))
    with patch.object(WhisperSTTTool, "_transcribe", local):
        result = asyncio.run(WhisperSTTTool(config=config).execute(
            WhisperSTTInput(audio_path=str(audio)),
            ToolExecutionContext(cwd=tmp_path),
        ))
    return result.output, local.await_count > 0


def _run_tts_tool(config: dict[str, Any] | None, tmp_path: Path):
    """The tts tool reads the config FILE: it is built without one."""
    from prometheus.tools.builtin.tts import TTSInput, TTSTool

    if config is not None:
        cfg_dir = tmp_path / "prom-config"
        cfg_dir.mkdir(exist_ok=True)
        (cfg_dir / "prometheus.yaml").write_text(yaml.safe_dump(config))
    out = tmp_path / "out.wav"

    async def _espeak(text, voice, out_path):
        Path(out_path).write_bytes(b"local-audio")
        return ToolResult(output=f"Audio saved to {out_path}")

    local = AsyncMock(side_effect=_espeak)
    with patch("prometheus.tools.builtin.tts._detect_engine", return_value="espeak"), \
            patch("prometheus.tools.builtin.tts._run_espeak", local):
        result = asyncio.run(TTSTool().execute(
            TTSInput(text="hello there", output_path=str(out)),
            ToolExecutionContext(cwd=tmp_path),
        ))
    assert not result.is_error, result.output
    return out.read_bytes(), local.await_count > 0


def _run_cli(config: dict[str, Any] | None, tmp_path: Path):
    from prometheus.cli import voice

    local_wav = tmp_path / "local.wav"
    local_wav.write_bytes(b"local-audio")
    local = AsyncMock(return_value=local_wav)
    with patch.object(voice, "_synthesize_piper", local):
        path = asyncio.run(voice.synthesize_wav(
            "hello there", {"model_path": "x.onnx"}, config=config))
    return Path(path).read_bytes(), local.await_count > 0


def _run_telegram(config: dict[str, Any] | None, tmp_path: Path):
    from prometheus.gateway.telegram import TelegramAdapter

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._prometheus_config = {
        **(config or {}),
        "gateway": {"voice": {"engine": "piper", "model_path": "x.onnx"}},
    }
    seen: dict[str, bytes] = {}

    async def _piper(text, out_wav, model_path):
        Path(out_wav).write_bytes(b"local-audio")
        return True

    async def _opus(wav_path, bitrate):
        seen["wav"] = Path(wav_path).read_bytes()
        ogg = tmp_path / "out.ogg"
        ogg.write_bytes(b"ogg")
        return str(ogg)

    local = AsyncMock(side_effect=_piper)
    with patch.object(adapter, "_run_piper", local), \
            patch.object(adapter, "_wav_to_opus_ogg", side_effect=_opus):
        ogg = asyncio.run(adapter._synthesize_voice("hello there"))
    assert ogg is not None
    return seen["wav"], local.await_count > 0


TTS_SITES = [
    pytest.param(_run_tts_tool, id="tts-tool"),
    pytest.param(_run_cli, id="cli-voice"),
    pytest.param(_run_telegram, id="telegram"),
]


# ---------------------------------------------------------------------------
# Unset → local, exactly as today
# ---------------------------------------------------------------------------

class TestUnset:

    def test_stt_unset_runs_local(self, tmp_path, server):
        text, used_local = _run_stt({}, tmp_path)
        assert (text, used_local) == ("local words", True)
        assert server.requests == []

    def test_stt_empty_url_runs_local(self, tmp_path, server):
        text, used_local = _run_stt({"stt": {"url": ""}}, tmp_path)
        assert (text, used_local) == ("local words", True)
        assert server.requests == []

    @pytest.mark.parametrize("site", TTS_SITES)
    def test_tts_unset_runs_local(self, site, tmp_path, server):
        audio, used_local = site(None, tmp_path)
        assert (audio, used_local) == (b"local-audio", True)
        assert server.requests == []

    def test_settings_none_when_unset(self):
        from prometheus.infra.voice_remote import remote_settings

        assert remote_settings("stt", {}) is None
        assert remote_settings("tts", {"tts": {"url": "  "}}) is None


# ---------------------------------------------------------------------------
# Reachable → remote, key from the env file only
# ---------------------------------------------------------------------------

class TestReachable:

    def test_stt_uses_remote(self, tmp_path, server, env_key):
        url = "http://voice.test/v1/audio/transcriptions"
        text, used_local = _run_stt(_cfg("stt", url, model="whisper-large"), tmp_path)
        assert (text, used_local) == ("remote words", False)
        [req] = server.requests
        assert req.method == "POST" and str(req.url) == url
        assert req.headers["authorization"] == f"Bearer {env_key}"
        body = req.content
        assert b'name="model"' in body and b"whisper-large" in body
        assert b'name="file"' in body

    @pytest.mark.parametrize("site", TTS_SITES)
    def test_tts_uses_remote(self, site, tmp_path, server, env_key):
        url = "http://voice.test/v1/audio/speech"
        audio, used_local = site(_cfg("tts", url, voice="nova"), tmp_path)
        assert (audio, used_local) == (WAV, False)
        [req] = server.requests
        assert req.method == "POST" and str(req.url) == url
        assert req.headers["authorization"] == f"Bearer {env_key}"
        payload = json.loads(req.content)
        assert payload["input"] == "hello there"
        assert payload["voice"] == "nova"
        assert payload["response_format"] == "wav"

    def test_key_in_yaml_is_never_sent(self, tmp_path, server, monkeypatch):
        monkeypatch.delenv(KEY_ENV, raising=False)
        cfg = {"stt": {"url": "http://voice.test/v1/audio/transcriptions",
                       "api_key": "pasted-into-yaml"}}
        text, _ = _run_stt(cfg, tmp_path)
        assert text == "remote words"
        [req] = server.requests
        assert "authorization" not in req.headers
        assert b"pasted-into-yaml" not in req.content

    def test_process_env_wins_over_env_file(self, tmp_path, server, env_key, monkeypatch):
        monkeypatch.setenv(KEY_ENV, "from-process-env")
        _run_stt(_cfg("stt", "http://voice.test/v1/audio/transcriptions"), tmp_path)
        assert server.requests[0].headers["authorization"] == "Bearer from-process-env"


# ---------------------------------------------------------------------------
# Unreachable → local, one quiet log line
# ---------------------------------------------------------------------------

class TestUnreachable:

    def test_stt_falls_back_quietly(self, tmp_path, caplog):
        url = f"http://127.0.0.1:{_closed_port()}/v1/audio/transcriptions"
        caplog.set_level(logging.DEBUG, logger=LOGGER)
        text, used_local = _run_stt({"stt": {"url": url}}, tmp_path)
        assert (text, used_local) == ("local words", True)
        [rec] = [r for r in _records(caplog) if r.levelno >= logging.INFO]
        assert rec.levelno == logging.WARNING
        assert "falling back" in rec.getMessage()

    @pytest.mark.parametrize("site", TTS_SITES)
    def test_tts_falls_back_quietly(self, site, tmp_path, caplog):
        url = f"http://127.0.0.1:{_closed_port()}/v1/audio/speech"
        caplog.set_level(logging.DEBUG, logger=LOGGER)
        audio, used_local = site({"tts": {"url": url}}, tmp_path)
        assert (audio, used_local) == (b"local-audio", True)
        [rec] = [r for r in _records(caplog) if r.levelno >= logging.INFO]
        assert rec.levelno == logging.WARNING

    def test_server_error_also_falls_back(self, tmp_path, server, caplog):
        server.status = 503
        caplog.set_level(logging.DEBUG, logger=LOGGER)
        text, used_local = _run_stt(
            {"stt": {"url": "http://voice.test/v1/audio/transcriptions"}}, tmp_path)
        assert (text, used_local) == ("local words", True)
        assert [r.levelno for r in _records(caplog) if r.levelno >= logging.INFO] == [
            logging.WARNING]


# ---------------------------------------------------------------------------
# 401 / 403 → local, but loud
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", [401, 403])
class TestRefused:

    def test_stt_refusal_logs_error(self, status, tmp_path, server, env_key, caplog):
        server.status = status
        caplog.set_level(logging.DEBUG, logger=LOGGER)
        text, used_local = _run_stt(
            _cfg("stt", "http://voice.test/v1/audio/transcriptions"), tmp_path)
        assert (text, used_local) == ("local words", True)
        [rec] = [r for r in _records(caplog) if r.levelno >= logging.INFO]
        assert rec.levelno == logging.ERROR
        assert str(status) in rec.getMessage()
        assert KEY_ENV in rec.getMessage()
        assert env_key not in rec.getMessage()

    @pytest.mark.parametrize("site", TTS_SITES)
    def test_tts_refusal_logs_error(self, status, site, tmp_path, server, env_key, caplog):
        server.status = status
        caplog.set_level(logging.DEBUG, logger=LOGGER)
        audio, used_local = site(_cfg("tts", "http://voice.test/v1/audio/speech"), tmp_path)
        assert (audio, used_local) == (b"local-audio", True)
        [rec] = [r for r in _records(caplog) if r.levelno >= logging.INFO]
        assert rec.levelno == logging.ERROR
        assert env_key not in rec.getMessage()


# ---------------------------------------------------------------------------
# oara doctor
# ---------------------------------------------------------------------------

class TestDoctorRow:

    def test_no_row_when_unset(self):
        from prometheus.cli.doctor import check_voice_remotes

        assert check_voice_remotes({}) == []

    def test_reachable_rows(self, server, env_key):
        from prometheus.cli.doctor import check_voice_remotes

        cfg = {**_cfg("stt", "http://voice.test/v1/audio/transcriptions"),
               **_cfg("tts", "http://voice.test/v1/audio/speech")}
        rows = check_voice_remotes(cfg)
        assert [(r.name, r.status) for r in rows] == [
            ("Remote STT", "ok"), ("Remote TTS", "ok")]
        assert "reachable" in rows[0].message
        # The probe is a read, sent with the key the daemon would send.
        assert [r.method for r in server.requests] == ["GET", "GET"]
        assert {str(r.url) for r in server.requests} == {"http://voice.test/v1/models"}
        assert all(r.headers["authorization"] == f"Bearer {env_key}"
                   for r in server.requests)

    def test_unreachable_row_warns(self):
        from prometheus.cli.doctor import check_voice_remotes

        url = f"http://127.0.0.1:{_closed_port()}/v1/audio/speech"
        [row] = check_voice_remotes({"tts": {"url": url}}, timeout=2.0)
        assert (row.name, row.status) == ("Remote TTS", "warning")
        assert "not reachable" in row.message and "local" in row.message
        assert row.fix

    @pytest.mark.parametrize("status", [401, 403])
    def test_refused_row_errors(self, status, server, env_key):
        from prometheus.cli.doctor import check_voice_remotes

        server.status = status
        [row] = check_voice_remotes(_cfg("stt", "http://voice.test/v1/audio/transcriptions"))
        assert (row.name, row.status) == ("Remote STT", "error")
        assert str(status) in row.message
        assert KEY_ENV in (row.fix or "")
        assert env_key not in row.message + (row.fix or "")

    def test_url_userinfo_is_not_displayed(self, server):
        from prometheus.cli.doctor import check_voice_remotes

        [row] = check_voice_remotes(
            {"stt": {"url": "http://user:pw@voice.test/v1/audio/transcriptions"}})
        assert "pw" not in row.message

    def test_row_is_part_of_the_report(self):
        import inspect

        from prometheus.cli import doctor

        assert "check_voice_remotes(config" in inspect.getsource(doctor.run_extended_checks)
