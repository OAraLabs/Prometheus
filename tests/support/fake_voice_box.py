"""A fake voice box: the speech-to-text and text-to-speech servers Prometheus reaches over the OpenAI audio API.

Built from the voice box's own server contract (the one ``infra/voice_remote.py`` and ``docs/contracts/pieces.md``
were written against), not from what Prometheus happens to send:

* ``GET /health``: no auth, ``{"status": "healthy"}``.
* ``GET /v1/models``: bearer key when the box has one.
* ``POST /v1/audio/transcriptions``: multipart ``file``, ``model``, ``language``, ``response_format`` (``json``
  default); returns ``{"text": ...}``. Uploads over 25 MB are a 413.
* ``POST /v1/audio/speech``: JSON ``model``, ``input`` (at most 4096 characters, else 400), ``voice``,
  ``response_format`` ``wav`` (default) or ``pcm`` (anything else is 400); returns 16-bit mono 24 kHz audio.
* A bearer key, when set, is required on everything but ``/health``: 401 without it or with the wrong one.

It is an ``httpx.MockTransport`` handler: no socket. ``down`` makes every request a connection error, ``status``
forces one answer on a path, and ``requests`` records what arrived.
"""

from __future__ import annotations

import io
import json
import wave

import httpx

SAMPLE_RATE = 24_000
MAX_INPUT = 4096
MAX_UPLOAD = 25 * 1024 * 1024


def wav_of(text: str) -> bytes:
    """The WAV the box returns for *text*: silence, 10 ms per character, 16-bit mono 24 kHz."""
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * (SAMPLE_RATE // 100) * max(1, len(text)))
    return out.getvalue()


class FakeVoiceBox:
    def __init__(self, *, key: str = "", transcript: str = "hello from the box") -> None:
        self.key = key
        self.transcript = transcript
        self.down = False
        #: path -> (status, body) answered instead of the real route.
        self.status: dict[str, tuple[int, bytes]] = {}
        self.requests: list[httpx.Request] = []
        #: The ``input`` of every speech request, in order.
        self.spoken: list[str] = []
        #: The voice-settings surface (docs/design/voice-settings.md). Off by default: an older box that
        #: predates the reads answers 404 on them, which is exactly what discovery must tolerate.
        self.settings_surface = False
        self.voices: list[str] = ["af_heart", "af_bella", "am_michael"]
        self.stt_models: list[dict] = [
            {"id": "base", "tier": "cpu", "active": False},
            {"id": "small", "tier": "cpu", "active": False},
            {"id": "large-v3-turbo", "tier": "gpu", "active": True},
        ]
        self.engines: list[dict] = [
            {"id": "kokoro", "available": True, "active": True},
            {"id": "chatterbox", "available": False, "reason": "not installed"},
            {"id": "system", "available": False, "reason": "macOS only"},
        ]
        self.runtime: dict = {"voice": "af_heart", "speed": 1.0, "engine": "kokoro",
                              "silence_gate_rms": 32.0, "vad_sensitivity": 0.5}
        #: The body of every PUT /v1/settings, in order.
        self.settings_pushes: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        if path in self.status:
            code, body = self.status[path]
            return httpx.Response(code, content=body)
        if path == "/health":
            return httpx.Response(200, json={"status": "healthy"})
        routes = {("GET", "/v1/models"), ("POST", "/v1/audio/speech"), ("POST", "/v1/audio/transcriptions")}
        if self.settings_surface:
            routes |= {("GET", "/v1/voices"), ("GET", "/v1/settings"), ("PUT", "/v1/settings")}
        if (request.method, path) not in routes:
            return httpx.Response(404, json={"detail": "not found"})  # routing comes before auth
        if self.key and request.headers.get("authorization") != f"Bearer {self.key}":
            return httpx.Response(401, json={"detail": "unauthorized"})
        if path == "/v1/models":
            if self.settings_surface:
                data = [{"id": m["id"], "kind": "stt", "tier": m["tier"], "active": m["active"]}
                        for m in self.stt_models]
                data.append({"id": "kokoro-v1.0", "kind": "tts", "active": True})
                return httpx.Response(200, json={"object": "list", "data": data})
            return httpx.Response(200, json={"data": [{"id": "whisper-1"}, {"id": "tts-1"}]})
        if path == "/v1/voices":
            return httpx.Response(200, json={"engine": self.runtime["engine"],
                                             "active": self.runtime["voice"], "voices": self.voices})
        if path == "/v1/settings":
            if request.method == "PUT":
                return self._put_settings(request)
            return httpx.Response(200, json=self._settings_body())
        if path == "/v1/audio/speech":
            return self._speech(request)
        return self._transcription(request)

    def _settings_body(self) -> dict:
        return {
            "stt": {"model": "large-v3-turbo", "tier": "gpu",
                    "silence_gate_rms": self.runtime["silence_gate_rms"]},
            "tts": {"engine": self.runtime["engine"], "voice": self.runtime["voice"],
                    "speed": self.runtime["speed"], "engines": self.engines},
            "vad": {"sensitivity": self.runtime["vad_sensitivity"], "health": "ok"},
            "ranges": {"speed": [0.5, 2.0], "vad_sensitivity": [0.1, 0.9], "silence_gate_rms": [0.0, 1000.0]},
        }

    def _put_settings(self, request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content)
        except ValueError:
            return httpx.Response(422, json={"detail": "invalid json"})
        self.settings_pushes.append(body)
        if "voice" in body and body["voice"] not in self.voices:
            return httpx.Response(400, json={"error": "unknown_voice"})
        if "engine" in body:
            row = next((e for e in self.engines if e["id"] == body["engine"]), None)
            if row is None or not row.get("available"):
                return httpx.Response(409, json={"error": "engine_not_running"})
            self.runtime["engine"] = body["engine"]
        for field in ("voice", "speed", "silence_gate_rms", "vad_sensitivity"):
            if field in body:
                self.runtime[field] = body[field]
        return httpx.Response(200, json=self._settings_body())

    def _speech(self, request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content)
        except ValueError:
            return httpx.Response(422, json={"detail": "invalid json"})
        text = body.get("input")
        if not isinstance(text, str) or not text or len(text) > MAX_INPUT:
            return httpx.Response(400, json={"detail": "input must be 1-4096 characters"})
        if body.get("response_format", "wav") not in ("wav", "pcm"):
            return httpx.Response(400, json={"detail": "response_format must be wav or pcm"})
        self.spoken.append(text)
        if body.get("response_format") == "pcm":
            return httpx.Response(200, content=wav_of(text)[44:], headers={"content-type": "audio/pcm"})
        return httpx.Response(200, content=wav_of(text), headers={"content-type": "audio/wav"})

    def _transcription(self, request: httpx.Request) -> httpx.Response:
        if len(request.content) > MAX_UPLOAD:
            return httpx.Response(413, json={"detail": "file too large"})
        if b'name="file"' not in request.content:
            return httpx.Response(422, json={"detail": "file is required"})
        return httpx.Response(200, json={"text": self.transcript})
