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


@pytest.fixture(autouse=True)
def _no_default_keys(monkeypatch):
    """The default key names must come from the test, never the shell."""
    for name in ("PROMETHEUS_STT_API_KEY", "PROMETHEUS_TTS_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_outage_carried_over(monkeypatch):
    """Which remotes are down is module state; each test starts with none (the down/back sentence is
    tests/test_voice_remote_says_so.py's)."""
    from prometheus.infra import voice_remote

    monkeypatch.setattr(voice_remote, "_down", set())


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
        # path -> status for a route that answers differently from the rest
        # (a server with only /health, one without /health, ...).
        self.routes: dict[str, int] = {}
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path in self.routes:
            code = self.routes[request.url.path]
            return httpx.Response(code, json={"status": "ok"} if code == 200 else {})
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
        # A list for the down/back sentence, as every gateway and the terminal pass: the output is the text.
        result = asyncio.run(WhisperSTTTool(config=config, notices=[]).execute(
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
        # Health first (at the server root, beside /v1), then the models list
        # to check the key. Every probe is a GET sent with the daemon's key.
        assert [str(r.url) for r in server.requests] == [
            "http://voice.test/health", "http://voice.test/v1/models"] * 2
        assert all(r.method == "GET" for r in server.requests)
        assert all(r.headers["authorization"] == f"Bearer {env_key}"
                   for r in server.requests)

    def test_health_only_server_is_reachable(self, server, env_key):
        """A voice server that serves /health and no /v1/models is up."""
        from prometheus.cli.doctor import check_voice_remotes

        server.status = 404
        server.routes = {"/health": 200}
        cfg = {**_cfg("stt", "http://voice.test/v1/audio/transcriptions"),
               **_cfg("tts", "http://voice.test/v1/audio/speech")}
        rows = check_voice_remotes(cfg)
        assert [(r.name, r.status) for r in rows] == [
            ("Remote STT", "ok"), ("Remote TTS", "ok")]
        assert all("reachable" in r.message for r in rows)
        assert server.requests[0].url.path == "/health"

    def test_server_without_health_falls_back_to_models(self, server, env_key):
        from prometheus.cli.doctor import check_voice_remotes

        server.routes = {"/health": 404}
        [row] = check_voice_remotes(_cfg("tts", "http://voice.test/v1/audio/speech"))
        assert (row.name, row.status) == ("Remote TTS", "ok")
        assert [r.url.path for r in server.requests] == ["/health", "/v1/models"]

    def test_unhealthy_server_warns(self, server):
        from prometheus.cli.doctor import check_voice_remotes

        server.routes = {"/health": 503}
        [row] = check_voice_remotes({"stt": {"url": "http://voice.test/v1/audio/transcriptions"}})
        assert (row.name, row.status) == ("Remote STT", "warning")
        assert "503" in row.message

    def test_health_ok_but_key_refused_errors(self, server, env_key):
        from prometheus.cli.doctor import check_voice_remotes

        server.routes = {"/health": 200, "/v1/models": 401}
        [row] = check_voice_remotes(_cfg("stt", "http://voice.test/v1/audio/transcriptions"))
        assert (row.name, row.status) == ("Remote STT", "error")
        assert "401" in row.message

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


# ---------------------------------------------------------------------------
# The server contract: base URLs, the server's own default voice, 400 as a
# configuration error, the default key variable, and whisper.url
# ---------------------------------------------------------------------------

STT_ROUTE = "http://voice.test/v1/audio/transcriptions"
TTS_ROUTE = "http://voice.test/v1/audio/speech"


class TestBaseUrl:
    """A url may name the server (root or /v1) instead of the full route."""

    @pytest.mark.parametrize("url", [
        "http://voice.test", "http://voice.test/", "http://voice.test/v1",
        "http://voice.test/v1/", STT_ROUTE,
    ])
    def test_stt_base_url_gets_the_route(self, url, tmp_path, server):
        text, used_local = _run_stt({"stt": {"url": url}}, tmp_path)
        assert (text, used_local) == ("remote words", False)
        assert [str(r.url) for r in server.requests] == [STT_ROUTE]

    @pytest.mark.parametrize("site", TTS_SITES)
    def test_tts_base_url_gets_the_route(self, site, tmp_path, server):
        audio, used_local = site({"tts": {"url": "http://voice.test"}}, tmp_path)
        assert (audio, used_local) == (WAV, False)
        assert [str(r.url) for r in server.requests] == [TTS_ROUTE]

    def test_doctor_probes_the_root_of_a_base_url(self, server):
        from prometheus.cli.doctor import check_voice_remotes

        [row] = check_voice_remotes({"stt": {"url": "http://voice.test:8001"}})
        assert row.status == "ok"
        assert [str(r.url) for r in server.requests] == [
            "http://voice.test:8001/health", "http://voice.test:8001/v1/models"]


class TestVoice:

    @pytest.mark.parametrize("site", TTS_SITES)
    def test_no_voice_configured_lets_the_server_choose(self, site, tmp_path, server):
        site({"tts": {"url": TTS_ROUTE}}, tmp_path)
        [req] = server.requests
        assert "voice" not in json.loads(req.content)

    def test_tool_voice_argument_stays_with_the_local_engines(self, tmp_path, server):
        """The tts tool's voice is engine-specific ('en-us+f3' for espeak);
        the remote gets tts.voice instead."""
        from prometheus.tools.builtin.tts import TTSInput, TTSTool

        cfg_dir = tmp_path / "prom-config"
        cfg_dir.mkdir(exist_ok=True)
        (cfg_dir / "prometheus.yaml").write_text(
            yaml.safe_dump({"tts": {"url": TTS_ROUTE, "voice": "nova"}}))
        result = asyncio.run(TTSTool().execute(
            TTSInput(text="hi", voice="en-us+f3", output_path=str(tmp_path / "o.wav")),
            ToolExecutionContext(cwd=tmp_path),
        ))
        assert not result.is_error
        assert json.loads(server.requests[0].content)["voice"] == "nova"


class TestBadRequest:
    """400 is a configuration error (unknown voice or model): ERROR, then local."""

    def test_stt_400_logs_error(self, tmp_path, server, caplog):
        server.status = 400
        caplog.set_level(logging.DEBUG, logger=LOGGER)
        text, used_local = _run_stt({"stt": {"url": STT_ROUTE}}, tmp_path)
        assert (text, used_local) == ("local words", True)
        [rec] = [r for r in _records(caplog) if r.levelno >= logging.INFO]
        assert rec.levelno == logging.ERROR
        assert "400" in rec.getMessage()

    @pytest.mark.parametrize("site", TTS_SITES)
    def test_tts_400_logs_error(self, site, tmp_path, server, caplog):
        server.status = 400
        caplog.set_level(logging.DEBUG, logger=LOGGER)
        audio, used_local = site({"tts": {"url": TTS_ROUTE, "voice": "nope"}}, tmp_path)
        assert (audio, used_local) == (b"local-audio", True)
        [rec] = [r for r in _records(caplog) if r.levelno >= logging.INFO]
        assert rec.levelno == logging.ERROR
        assert "400" in rec.getMessage()


class TestDefaultKeyVariable:
    """With api_key_env unset, the key is looked for under a default name."""

    @pytest.mark.parametrize(("kind", "url", "default"), [
        ("stt", STT_ROUTE, "PROMETHEUS_STT_API_KEY"),
        ("tts", TTS_ROUTE, "PROMETHEUS_TTS_API_KEY"),
    ])
    def test_default_name_is_read(self, kind, url, default, tmp_path, server, monkeypatch):
        from prometheus.infra.voice_remote import remote_settings

        monkeypatch.delenv(default, raising=False)
        (tmp_path / "prom-env").write_text(f"{default}=from-default-name\n")
        settings = remote_settings(kind, {kind: {"url": url}})
        assert settings is not None and settings.api_key_env == default
        assert settings.headers() == {"Authorization": "Bearer from-default-name"}

    def test_stt_request_carries_the_default_key(self, tmp_path, server, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_STT_API_KEY", "k-default")
        _run_stt({"stt": {"url": STT_ROUTE}}, tmp_path)
        assert server.requests[0].headers["authorization"] == "Bearer k-default"

    def test_named_variable_wins(self, tmp_path, server, env_key, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_STT_API_KEY", "k-default")
        _run_stt(_cfg("stt", STT_ROUTE), tmp_path)
        assert server.requests[0].headers["authorization"] == f"Bearer {env_key}"

    def test_no_key_anywhere_sends_none(self, tmp_path, server, monkeypatch):
        monkeypatch.delenv("PROMETHEUS_STT_API_KEY", raising=False)
        _run_stt({"stt": {"url": STT_ROUTE}}, tmp_path)
        assert "authorization" not in server.requests[0].headers


class TestWhisperUrlAlias:
    """whisper.url means stt.url; stt.url wins when both are set."""

    def test_whisper_url_reaches_the_remote(self, tmp_path, server):
        text, used_local = _run_stt({"whisper": {"url": "http://voice.test"}}, tmp_path)
        assert (text, used_local) == ("remote words", False)
        assert [str(r.url) for r in server.requests] == [STT_ROUTE]

    def test_stt_url_wins_over_whisper_url(self, tmp_path, server):
        _run_stt({"stt": {"url": STT_ROUTE},
                  "whisper": {"url": "http://other.test"}}, tmp_path)
        assert [str(r.url) for r in server.requests] == [STT_ROUTE]

    def test_stt_settings_still_apply_with_whisper_url(self, tmp_path, server):
        _run_stt({"whisper": {"url": "http://voice.test"},
                  "stt": {"model": "large-v3"}}, tmp_path)
        assert b"large-v3" in server.requests[0].content

    def test_doctor_row_for_whisper_url(self, server):
        from prometheus.cli.doctor import check_voice_remotes

        [row] = check_voice_remotes({"whisper": {"url": "http://voice.test"}})
        assert (row.name, row.status) == ("Remote STT", "ok")
