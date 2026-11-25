"""The voice event route — one door for wake boxes and PTT nodes (OAra Voice, Phase 1).

``POST /api/speech/event`` accepts the envelope every front end honors
(``speech/envelope.py``): ``wake`` with trailing audio, ``ptt_start`` /
``ptt_end`` from a button. The daemon does all the thinking:

* ``ptt_start`` opens a capture for the node (no audio yet) and answers at
  once — nothing in the daemon's event loop waits on the edge.
* ``wake`` / ``ptt_end`` carry the utterance as raw samples (the plugin's
  wire format). The daemon transcribes through the SAME voice box the
  speech route uses (``speech/voice.py`` — its health record, its refusal
  semantics, its "said once" notices), then dispatches ONE agent turn whose
  origin is derived from the authenticated credential
  (``permissions/origins.py``), never from the envelope. The reply streams
  over the bridge as any turn's does; the node reads it through the session
  speech stream the speech route already serves.

The pre-roll rides the same ``audio`` field, so the start of the utterance
is not clipped between the wake word and the request. ``ts`` is recorded
for latency tracking against the TTFA baseline and trusted for nothing.

Failure is loud, never absorbed: no voice box → 503 ``speech_unavailable``
(the same answer the speech route gives); the box down mid-turn → the turn
is not dispatched and the reason is named; a node that dies costs nothing —
the capture is a dict entry, and the turn, once dispatched, belongs to the
session, not to the socket.

Who may call: the same callers as chat (``route_access.SCOPED_ALLOWED``).
An unmarked scoped credential fails closed to the ``voice_box`` capability
set; the operator's own credential is ``terminal``.

Source: novel code for Prometheus, 2026-11-24.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from prometheus.speech.envelope import EnvelopeRefused, parse_envelope, rms_pcm16, to_pcm16
from prometheus.speech.voice import VoiceUnavailable

logger = logging.getLogger(__name__)

#: The silence floor, the plugin's measured one (~-60 dBFS of 32768 full
#: scale): whisper hallucinates a phrase from digital silence, so quiet
#: audio is refused BEFORE the ears, and the level is reported so a too-quiet
#: mic is distinguishable from a dead one.
SILENCE_RMS_FLOOR = 32.0

_NO_STORE = {"Cache-Control": "no-store"}


def register_voice_event_route(app: Any, *, config: dict[str, Any], devices: Any) -> None:
    """Register ``POST /api/speech/event`` on *app*. The speech service (voice box, streams) is already on
    ``app.state.speech``; the bridge is read per request, like ``/api/chat/send`` reads it."""
    # node_id -> (opened_at, device_id): an open PTT capture. A node that dies leaves an entry; the next
    # ptt_start from the same node replaces it, and a stale one is refused by name, never waited on.
    captures: dict[str, tuple[float, str]] = {}

    def _error(status: int, error: str, detail: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": error, "detail": detail}, headers=_NO_STORE)

    @app.post("/api/speech/event")
    async def speech_event(request: Request) -> JSONResponse:
        from prometheus.engine import pause as _pause

        if _pause.current() is not None:
            return _error(503, "paused", "Prometheus is paused; nothing new runs until a person resumes it.")
        try:
            body = await request.json()
        except Exception:
            return _error(400, "bad_body", "body must be a JSON object")
        try:
            event = parse_envelope(body)
        except EnvelopeRefused as exc:
            return _error(400, exc.kind, exc.message)

        identity = getattr(request.state, "device_identity", None)
        caller = str(identity.id) if identity is not None else "global"
        from prometheus.permissions.origins import voice_origin_for

        marked = None
        if identity is not None and not identity.is_operator:
            store = devices()
            if store is not None and hasattr(store, "device_origin"):
                marked = store.device_origin(identity.id)
        origin = voice_origin_for(identity, marked_origin=marked)

        if event.event == "ptt_start":
            captures[event.node_id] = (time.time(), caller)
            return JSONResponse({"status": "listening", "node_id": event.node_id, "origin": origin},
                                headers=_NO_STORE)

        opened = captures.pop(event.node_id, None)
        if event.event == "ptt_end" and opened is None:
            return _error(409, "no_capture", "ptt_end without a ptt_start from this node")
        if opened is not None and opened[1] != caller:
            return _error(403, "not_your_capture", "another device opened this node's capture")

        # The silence gate, BEFORE the ears: the same measured floor the plugin uses.
        pcm = to_pcm16(event.audio, event.audio_format)
        level = rms_pcm16(pcm)
        if level < SILENCE_RMS_FLOOR:
            return JSONResponse({"status": "no_speech", "node_id": event.node_id,
                                 "level_rms": round(level, 1), "silence_floor_rms": SILENCE_RMS_FLOOR},
                                headers=_NO_STORE)

        service = getattr(app.state, "speech", None)
        if service is None:
            return _error(503, "speech_unavailable", "the speech service is not wired")
        # Wrap the raw samples as WAV for the box: the speech route's client sends a container, and the
        # box's OpenAI-compatible endpoint decodes it. 16-bit mono at the event's own honest rate.
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(event.sample_rate)
            w.writeframes(pcm)
        try:
            await service.voice.ready("stt")
            text = await service.voice.transcribe(buf.getvalue(), "audio/wav", None)
        except VoiceUnavailable as exc:
            return JSONResponse(status_code=503,
                                content={"error": "speech_unavailable", "voice": exc.reason, "message": exc.message},
                                headers=_NO_STORE)
        text = text.strip()
        if not text.strip(" .,!?\n"):
            return JSONResponse({"status": "no_speech", "node_id": event.node_id,
                                 "level_rms": round(level, 1), "silence_floor_rms": SILENCE_RMS_FLOOR},
                                headers=_NO_STORE)

        bridge = getattr(app.state, "ws_bridge", None)
        if bridge is None or not hasattr(bridge, "dispatch_user_message"):
            return _error(503, "chat_unavailable", "chat dispatch unavailable — ws_bridge not wired")

        # The node's session: its own, named by its credential — never an id from the envelope. A fresh id
        # per node is claimable by the device (SessionAccess.admit on the chat path does the same claim).
        session_id = f"voice:{caller}:{event.node_id}"
        import uuid as _uuid

        run_id = _uuid.uuid4().hex
        # The transcript is the person's message. A leading slash in a transcript is NOT a command: the
        # marker leads, the same shape the voice-upload path uses.
        message = f"[Voice event: {event.event} from {event.node_id}]\n{text}"
        await bridge.dispatch_user_message(
            session_id, message, client_msg_id=None, mode="agent",
            operator=bool(identity is None or identity.is_operator),
            origin=origin,
        )
        answer: dict[str, Any] = {
            "status": "dispatched", "node_id": event.node_id, "origin": origin,
            "session_id": session_id, "run_id": run_id, "transcript": text,
            "level_rms": round(level, 1), "silence_floor_rms": SILENCE_RMS_FLOOR,
        }
        if event.ts is not None:
            # Descriptive only: the node's clock against the daemon's, for the TTFA baseline.
            answer["node_ts"] = event.ts
            answer["received_at"] = time.time()
        return JSONResponse(answer, headers=_NO_STORE)
