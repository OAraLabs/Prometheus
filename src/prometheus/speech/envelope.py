"""The voice event envelope — one contract every front end honors (OAra Voice, Phase 1).

A node is an ear, a mouth, a screen or a trigger; it never runs the agent.
Everything smart stays on this side of one event message:

    event     ``wake`` | ``ptt_start`` | ``ptt_end`` — what fired
    node_id   which box fired; tells the daemon where to send TTS back
    audio     the utterance (or pre-roll buffer), RAW samples: 16 kHz mono,
              pcm16 or f32 — the plugin's existing wire format, unchanged
    ts        the node's timestamp (seconds), for latency tracking against
              the TTFA baseline. DESCRIPTIVE: recorded, never trusted for
              behaviour — a node's clock is not the daemon's.

**Origin is NOT a field.** It is derived server-side from the authenticated
device credential (``permissions/origins.py``), never from anything in the
envelope or the transcript. A body that carries an ``origin`` key is refused
by name: the field must not exist, so no client can grow the habit of
sending one.

``ptt_start`` opens a capture (no audio yet); ``ptt_end`` closes it with the
utterance. ``wake`` carries the trailing audio in one message. The pre-roll
rides the same ``audio`` field — the format is the plugin's, so a front end
that already streams VAD segments needs no second encoder.

Additive by construction: nothing here reshapes an existing request or
response body. The envelope is a NEW route's NEW body; the speech routes,
the bridge contract and the chat routes are untouched.

Source: novel code for Prometheus, 2026-11-24.
"""

from __future__ import annotations

import base64
import math
import re
import struct
from dataclasses import dataclass

#: The events a front end may send. One vocabulary for wake boxes and PTT nodes.
EVENTS = ("wake", "ptt_start", "ptt_end")

#: Raw-sample formats, the plugin's wire vocabulary (no container).
FORMATS = ("pcm16", "f32")

SAMPLE_RATE = 16000
#: The most audio one event may carry: 60 s at 16 kHz mono pcm16 is ~1.9 MB,
#: under the /api body cap; f32 of the same length is refused by the cap.
MAX_AUDIO_BYTES = 2 * 1024 * 1024
#: A node id names a box, not a path: letters, digits, dot, dash, underscore.
_NODE_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


class EnvelopeRefused(Exception):
    """The envelope is malformed — a 400 with ``kind`` naming the field."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass(frozen=True)
class VoiceEvent:
    event: str
    node_id: str
    audio_format: str
    sample_rate: int
    audio: bytes          # raw samples, decoded; b"" for ptt_start
    ts: float | None      # the node's clock, descriptive only


def parse_envelope(body: object) -> VoiceEvent:
    """Validate one event body. Refuses by name; never guesses.

    An ``origin`` key is refused whatever its value: origin comes from the
    credential, and a field that must not exist should not parse.
    """
    if not isinstance(body, dict):
        raise EnvelopeRefused("bad_body", "body must be a JSON object")
    if "origin" in body:
        raise EnvelopeRefused("origin_not_a_field",
                              "origin is derived from the device credential, never sent")
    event = body.get("event")
    if event not in EVENTS:
        raise EnvelopeRefused("bad_event", f"event is one of {', '.join(EVENTS)}")
    node_id = body.get("node_id")
    if not isinstance(node_id, str) or not _NODE_ID.fullmatch(node_id):
        raise EnvelopeRefused("bad_node_id", "node_id is 1 to 64 letters, digits, dot, dash or underscore")
    ts = body.get("ts")
    if ts is not None and (not isinstance(ts, (int, float)) or isinstance(ts, bool) or not math.isfinite(ts)):
        raise EnvelopeRefused("bad_ts", "ts is a finite number of seconds")

    audio_format = body.get("audio_format", "pcm16")
    if audio_format not in FORMATS:
        raise EnvelopeRefused("bad_audio_format", f"audio_format is one of {', '.join(FORMATS)}")
    sample_rate = body.get("sample_rate", SAMPLE_RATE)
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or not 8000 <= sample_rate <= 48000:
        raise EnvelopeRefused("bad_sample_rate", "sample_rate is 8000 to 48000")

    raw = body.get("audio_base64", "")
    if event == "ptt_start":
        if raw:
            raise EnvelopeRefused("audio_on_ptt_start", "ptt_start opens a capture; audio rides ptt_end or wake")
        audio = b""
    else:
        if not isinstance(raw, str) or not raw:
            raise EnvelopeRefused("audio_required", f"{event} carries audio_base64")
        try:
            audio = base64.b64decode(raw, validate=True)
        except Exception as exc:
            raise EnvelopeRefused("bad_audio", "audio_base64 is not valid base64") from exc
        if not audio:
            raise EnvelopeRefused("bad_audio", "audio_base64 decoded to nothing")
        if len(audio) > MAX_AUDIO_BYTES:
            raise EnvelopeRefused("too_large", f"at most {MAX_AUDIO_BYTES} audio bytes")
        unit = 4 if audio_format == "f32" else 2
        if len(audio) % unit:
            raise EnvelopeRefused("bad_audio", f"audio length is not a whole number of {audio_format} samples")
    return VoiceEvent(event=event, node_id=node_id, audio_format=audio_format,
                      sample_rate=sample_rate, audio=audio, ts=float(ts) if ts is not None else None)


def to_pcm16(raw: bytes, fmt: str) -> bytes:
    """f32 -> pcm16 (clamped). The plugin's conversion, byte-for-byte: the
    samples measured for silence must be the samples the ears hear."""
    if fmt == "f32":
        n = len(raw) // 4
        floats = struct.unpack(f"<{n}f", raw[: n * 4])
        return struct.pack(f"<{n}h", *(max(-32768, min(32767, int(x * 32767.0))) for x in floats))
    return raw


def rms_pcm16(pcm: bytes) -> float:
    """RMS of 16-bit mono samples; 0.0 for an empty buffer. The plugin's
    measured silence gate uses this same arithmetic."""
    import array

    a = array.array("h")
    a.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not a:
        return 0.0
    return math.sqrt(sum(x * x for x in a) / len(a))
