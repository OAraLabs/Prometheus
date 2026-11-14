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
        if self.key and request.headers.get("authorization") != f"Bearer {self.key}":
            return httpx.Response(401, json={"detail": "unauthorized"})
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "whisper-1"}, {"id": "tts-1"}]})
        if path == "/v1/audio/speech" and request.method == "POST":
            return self._speech(request)
        if path == "/v1/audio/transcriptions" and request.method == "POST":
            return self._transcription(request)
        return httpx.Response(404, json={"detail": "not found"})

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
        return httpx.Response(200, content=wav_of(text), headers={"content-type": "audio/wav"})

    def _transcription(self, request: httpx.Request) -> httpx.Response:
        if len(request.content) > MAX_UPLOAD:
            return httpx.Response(413, json={"detail": "file too large"})
        if b'name="file"' not in request.content:
            return httpx.Response(400, json={"detail": "file is required"})
        return httpx.Response(200, json={"text": self.transcript})
