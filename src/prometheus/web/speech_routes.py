"""The speech routes: a reply read aloud, phrase by phrase, through the voice box (docs/design/speech-route.md).

* ``GET /api/speech``: what works right now (each half configured, and whether it answers), and the limits.
* ``POST /api/speech/transcribe``: audio in, text out. The app sends the text as the person's message itself.
* ``GET /api/sessions/{session_id}/speech``: a session's reply as an NDJSON stream of spoken phrases.
* ``POST /api/speech/speak``: any text as the same stream.

**Who may call:** the same callers as chat. All four are in ``route_access.SCOPED_ALLOWED``, so a chat-only device
may use them; the session path guard in ``web/server.py`` answers 404 for a session that is not the caller's before
the stream route runs (a GET only reads, so it never claims an id).

**Limits** (design section 8): the upload is read as a stream and refused the moment it passes the smaller of the
``/api`` body cap and ``gateway.media.max_file_size_mb`` (a chunked upload has no ``Content-Length`` for the
middleware to check); ``speak`` text is at most :data:`SPEAK_MAX` characters; a caller (a device, or the global token)
has at most :data:`PER_CALLER` streams, devices other than the owner's at most :data:`DEVICE_STREAMS` in all, and the
owner is never refused because of them; a newer reply stream from the same caller for the same session replaces the
older one (``done`` / ``replaced``). A revoked token's streams end at once (``done`` / ``revoked``).

**When the voice box is down** the route says so (503 ``speech_unavailable``, or a final ``done`` / ``unavailable``)
and never falls back to the daemon's own voice engines: the app reads the rest with its own voice.

Never in an answer: the box's address, its key, or the name of the variable that holds it.

Source: novel code for Prometheus, 2026-11-13.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from prometheus.speech.phrases import MAX_CHARS, SPOKEN_CAP
from prometheus.speech.replies import ReplyTap
from prometheus.speech.stream import MIME, speak, text_source
from prometheus.speech.voice import Voice, VoiceUnavailable

VERSION = 1
#: The ``/api`` body cap (``web/server.py``); an upload is also held to ``gateway.media.max_file_size_mb``.
BODY_CAP = 2 * 1024 * 1024
SPEAK_MAX = 20_000
#: The most a ``speak`` body may be: 20,000 characters, each up to 12 bytes (an emoji escaped as two \\uXXXX, as
#: many JSON encoders write non-ASCII), and the JSON around them.
SPEAK_BODY_CAP = 12 * SPEAK_MAX + 1024
PER_CALLER = 2
DEVICE_STREAMS = 4
WAIT_DEFAULT = 30.0
WAIT_MAX = 120.0

_LANGUAGE = re.compile(r"[A-Za-z-]{2,8}")
_MESSAGE_ID = re.compile(r"[A-Za-z0-9:_.-]{1,64}")
_LEADING = re.compile(r"^[\s/]+")
#: Audio types with a known file signature. ``media_guard`` admits unrecognised bytes on their declared type (for a
#: format with no signature); for these the bytes must carry it.
_SIGNED_AUDIO = frozenset({"audio/ogg", "audio/mpeg", "audio/wav", "audio/mp4", "audio/webm"})
_NO_STORE = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}


@dataclass(eq=False)
class _Stream:
    caller: str
    owner: bool
    session_id: str | None
    loop: asyncio.AbstractEventLoop
    stop: asyncio.Future[str] = field(repr=False)

    def end(self, reason: str) -> None:
        """Ask the stream to end with *reason*. Safe from any thread (a revoke runs on the revoking one)."""
        def _set() -> None:
            if not self.stop.done():
                self.stop.set_result(reason)
        if self.loop.is_closed():
            return
        self.loop.call_soon_threadsafe(_set)


class SpeechService:
    """The reply holder, the voice box client and the open streams. One per app, on ``app.state.speech``."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.tap = ReplyTap()
        self.voice = Voice(lambda: self.config)
        self._streams: list[_Stream] = []
        self._lock = threading.Lock()

    def admit(self, caller: str, owner: bool, session_id: str | None) -> _Stream | None:
        """A new stream, or None when a limit says no. An older reply stream of the caller's for the session ends."""
        loop = asyncio.get_running_loop()
        with self._lock:
            if session_id is not None:
                for old in [s for s in self._streams if s.caller == caller and s.session_id == session_id]:
                    self._streams.remove(old)
                    old.end("replaced")
            if sum(s.caller == caller for s in self._streams) >= PER_CALLER:
                return None
            if not owner and sum(not s.owner for s in self._streams) >= DEVICE_STREAMS:
                return None
            stream = _Stream(caller, owner, session_id, loop, loop.create_future())
            self._streams.append(stream)
            return stream

    def release(self, stream: _Stream) -> None:
        with self._lock:
            if stream in self._streams:
                self._streams.remove(stream)

    def on_revoked(self, device_ids: list[str]) -> None:
        """The device store's word that these tokens were revoked: their streams end now."""
        wanted = set(device_ids)
        with self._lock:
            gone = [s for s in self._streams if s.caller in wanted]
            for stream in gone:
                self._streams.remove(stream)
        for stream in gone:
            stream.end("revoked")


def attach_bridge(app: Any, bridge: Any) -> None:
    """Let the reply holder hear every frame the bridge broadcasts (the launcher calls this once)."""
    bridge.frame_observers.append(app.state.speech.tap.observe)


def _unavailable(exc: VoiceUnavailable) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": "speech_unavailable", "voice": exc.reason,
                                                  "message": exc.message})


def _error(status: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": error, "detail": detail})


class _ReleasingResponse(StreamingResponse):
    """Releases the stream's slot however the response ends, even if its body never started.

    Under a server that speaks ASGI 2.4 or later, Starlette notices a client that left only at its next write, so a
    stream that is waiting for a turn keeps its slot until the next keepalive (at most 15 s). uvicorn today speaks 2.3,
    where the disconnect is noticed at once.
    """

    def __init__(self, *args: Any, release: Callable[[], None], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._release = release

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._release()


async def _read_capped(request: Request, cap: int) -> bytes | None:
    """The body, read as it arrives; None the moment it passes *cap* (a chunked body has no Content-Length)."""
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > cap:
            return None
    return bytes(body)


def register_speech_routes(
    app: Any, *, config: dict[str, Any], devices: Callable[[], Any], auth_on: Callable[[], bool] = lambda: False,
) -> SpeechService:
    """Register the four routes on *app* and return the service (also on ``app.state.speech``)."""
    service = SpeechService(config)
    app.state.speech = service
    listening: set[int] = set()

    def _listen() -> None:
        """Hear revocations from the device store in use, whenever it came to exist."""
        store = devices()
        add_listener = getattr(store, "add_revoke_listener", None)
        if callable(add_listener) and id(store) not in listening:
            listening.add(id(store))
            add_listener(service.on_revoked)

    _listen()

    def _caller(request: Request) -> tuple[str, bool] | None:
        """(caller, is the owner); None when auth is on and the caller resolved to nobody (fail closed)."""
        _listen()
        identity = getattr(request.state, "device_identity", None)
        if identity is None:
            return None if auth_on() else ("global", True)  # auth off: open, like every other route
        return str(identity.id), bool(identity.is_operator)

    def _admit(request: Request, session_id: str | None) -> _Stream | JSONResponse:
        caller = _caller(request)
        if caller is None:
            return _error(401, "unauthorized", "no identity")
        stream = service.admit(*caller, session_id)
        if stream is None:
            return _error(429, "speech_busy", "too many speech streams are open")
        return stream

    def _upload_cap() -> int:
        from prometheus.gateway.media_guard import policy_from_config

        return min(BODY_CAP, policy_from_config(config).max_bytes)

    def _ndjson(lines: AsyncIterator[dict[str, Any]], stream: _Stream) -> StreamingResponse:
        async def body() -> AsyncIterator[str]:
            try:
                async for line in lines:
                    yield json.dumps(line, separators=(",", ":")) + "\n"
            finally:
                service.release(stream)
                await lines.aclose()  # type: ignore[attr-defined]
        return _ReleasingResponse(body(), media_type="application/x-ndjson", headers=_NO_STORE,
                                  release=lambda: service.release(stream))

    @app.get("/api/speech")
    async def speech_status() -> dict[str, Any]:
        halves = await service.voice.status()
        return {
            "version": VERSION,
            "stt": halves["stt"],
            "tts": halves["tts"],
            "audio": {"mime": MIME},
            "limits": {"upload_bytes": _upload_cap(), "phrase_chars": MAX_CHARS, "spoken_chars": SPOKEN_CAP,
                       "speak_chars": SPEAK_MAX, "streams": DEVICE_STREAMS},
        }

    @app.post("/api/speech/transcribe")
    async def speech_transcribe(request: Request, language: str | None = None) -> JSONResponse:
        # Query parameters are declared (strict_query refuses any other) and validated here, as strings.
        from prometheus.gateway.media_guard import (
            MediaRejected,
            canonical_mime,
            check_declared_mime,
            policy_from_config,
        )

        policy = policy_from_config(config)
        declared = canonical_mime(request.headers.get("content-type"))
        try:
            if not declared:
                raise MediaRejected("speech", "no Content-Type")
            check_declared_mime(declared, "audio", policy)
        except MediaRejected:
            return _error(415, "type_not_allowed", "send audio of a type on gateway.media's audio list")
        if language is not None and not _LANGUAGE.fullmatch(language):
            return _error(400, "bad_language", "language is 2 to 8 letters or '-'")
        # A transcription holds a slot like a stream: one paired device must not queue hundreds on the voice box.
        slot = _admit(request, None)
        if isinstance(slot, JSONResponse):
            return slot
        try:
            return await _transcribe(request, declared, language, policy, slot)
        finally:
            service.release(slot)

    async def _transcribe(request: Request, declared: str, language: str | None, policy: Any,
                          slot: _Stream) -> JSONResponse:
        from prometheus.gateway.media_guard import MediaRejected, MediaTooLarge, sniff_mime, validate_inbound

        cap = _upload_cap()
        data = await _read_capped(request, cap)
        if data is None:
            return _error(413, "too_large", f"at most {cap} bytes")
        if not data:
            return _error(400, "bad_audio", "the body is empty")
        try:
            validate_inbound(data=data, declared_mime=declared, kind="audio", policy=policy)
        except MediaTooLarge:
            return _error(413, "too_large", f"at most {cap} bytes")
        except MediaRejected:
            return _error(400, "bad_audio", "the bytes are not the declared audio type")
        if declared in _SIGNED_AUDIO and sniff_mime(data) != declared:
            return _error(400, "bad_audio", "the bytes are not the declared audio type")
        if slot.stop.done():
            return _error(401, "unauthorized", "the token was revoked")
        try:
            await service.voice.ready("stt")
            text = await service.voice.transcribe(data, declared, language)
        except VoiceUnavailable as exc:
            return _unavailable(exc)
        if slot.stop.done() and slot.stop.result() == "revoked":
            return _error(401, "unauthorized", "the token was revoked")
        return JSONResponse({"text": _LEADING.sub("", text)}, headers={"Cache-Control": "no-store"})

    @app.post("/api/speech/speak", response_model=None)
    async def speech_speak(request: Request) -> JSONResponse | StreamingResponse:
        raw = await _read_capped(request, SPEAK_BODY_CAP)
        if raw is None:
            return _error(413, "too_long", f"the body is over {SPEAK_BODY_CAP} bytes")
        try:
            body = json.loads(raw)
        except (ValueError, RecursionError):
            body = None
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip():
            return _error(400, "text_required", 'send {"text": "..."}')
        if len(text) > SPEAK_MAX:
            return _error(413, "too_long", f"at most {SPEAK_MAX} characters")
        try:
            await service.voice.ready("tts")
        except VoiceUnavailable as exc:
            return _unavailable(exc)
        stream = _admit(request, None)
        if isinstance(stream, JSONResponse):
            return stream
        return _ndjson(speak(text_source(text), service.voice, stop=stream.stop), stream)

    @app.get("/api/sessions/{session_id}/speech", response_model=None)
    async def speech_reply(session_id: str, request: Request, message_id: str | None = None,
                           wait: str | None = None) -> JSONResponse | StreamingResponse:
        # The session path guard has already answered 404 for a session that is not the caller's.
        if message_id is not None and not _MESSAGE_ID.fullmatch(message_id):
            return _error(400, "bad_query", "message_id is the id from a chat_delta frame")
        try:
            seconds = WAIT_DEFAULT if wait is None else float(wait)
        except ValueError:
            return _error(400, "bad_query", "wait is a number of seconds")
        if not 1 <= seconds <= WAIT_MAX:
            return _error(400, "bad_query", f"wait is 1 to {int(WAIT_MAX)} seconds")
        try:
            await service.voice.ready("tts")
        except VoiceUnavailable as exc:
            return _unavailable(exc)
        stream = _admit(request, session_id)
        if isinstance(stream, JSONResponse):
            return stream

        async def source() -> AsyncIterator[tuple[str, Any]]:
            reply = await service.tap.find(session_id, message_id=message_id, wait=seconds)
            if reply is None:
                yield ("end", "no_reply")
                return
            async for event in reply.follow():
                yield event

        return _ndjson(speak(source(), service.voice, stop=stream.stop), stream)

    return service
