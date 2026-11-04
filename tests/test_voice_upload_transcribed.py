"""A voice message uploaded through ``chat_upload`` is transcribed, and a client is told whether to offer voice.

THE RULE
--------
* ``chat_upload`` with an ``audio/*`` type goes through the same inbound media checks as every other surface
  (``gateway/media_guard``): the declared type against ``gateway.media.allowed_audio_types``, the size against
  ``gateway.media.max_file_size_mb``, and the bytes' own signature against the declared type. A refusal is a
  ``system`` chat message and nothing else: no transcription, no turn.
* Admitted audio is transcribed by the configured speech-to-text (``gateway/media_services.transcribe_audio``:
  ``stt.url`` or an approved voice box first, this computer's Whisper as the fallback), and the transcript becomes
  the user's message. It is marked as a voice message, so neither the transcript nor a caption is ever read as a
  slash command.
* ``voice_input_available`` says whether that can work here: a remote speech-to-text server is set, or a local
  Whisper engine and ffmpeg are installed. The ``subscribed`` ack lists ``voice_input`` in ``server_capabilities``
  when it can, so a client offers voice only then. An audio upload that arrives anyway is refused in the chat.
"""

from __future__ import annotations

import base64
import json
import sys
import types

if "prometheus.memory" not in sys.modules:  # pragma: no cover - import plumbing
    try:
        import prometheus.memory  # noqa: F401
    except Exception:
        sys.modules["prometheus.memory"] = types.ModuleType("prometheus.memory")

import pytest  # noqa: E402

from prometheus.gateway import media_cache, media_services  # noqa: E402
from prometheus.web.ws_server import WebSocketBridge  # noqa: E402

OGG = b"OggS" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
SID = "beacon:voice"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


class Rig:
    def __init__(self, monkeypatch, tmp_path, config: dict | None = None, *, available: bool = True,
                 transcript: str | None = "turn the lights off", notices: tuple[str, ...] = ()) -> None:
        self.bridge = WebSocketBridge(session_mgr=None, loop_context=None, config=config or {})
        self.sent: list[tuple[str, dict]] = []
        self.system: list[str] = []
        self.transcribed: list[str] = []

        async def send(session_id, text, **kw):
            self.sent.append((text, kw))

        async def broadcast(frame):
            if frame.get("type") == "chat_message" and frame["payload"].get("role") == "system":
                self.system.append(frame["payload"]["content"])

        async def transcribe(path, config=None, notices=None):
            self.transcribed.append(path)
            if notices is not None:
                notices.extend(rig_notices)
            return transcript

        rig_notices = list(notices)
        monkeypatch.setattr(self.bridge, "_handle_send_message", send)
        monkeypatch.setattr(self.bridge, "broadcast", broadcast)
        monkeypatch.setattr(media_services, "transcribe_audio", transcribe)
        monkeypatch.setattr(media_services, "voice_input_available", lambda config=None: available)
        monkeypatch.setattr(media_cache, "audio_cache_dir", lambda: tmp_path)

    async def upload(self, data: bytes, mime: str = "audio/ogg", *, filename: str = "memo.ogg",
                     caption: str = "", **kw) -> None:
        await self.bridge._handle_file_upload(SID, filename, b64(data), mime, caption, **kw)


# ── admitted audio is transcribed and becomes the turn ──────────────────────

@pytest.mark.asyncio
async def test_an_audio_upload_is_transcribed_and_the_transcript_is_the_turn(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.upload(OGG)
    assert len(rig.transcribed) == 1, "the configured speech-to-text was never asked"
    assert len(rig.sent) == 1 and "turn the lights off" in rig.sent[0][0]
    assert rig.system == []


@pytest.mark.asyncio
async def test_the_caption_rides_with_the_transcript(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.upload(OGG, caption="for the kitchen")
    text = rig.sent[0][0]
    assert "turn the lights off" in text and "for the kitchen" in text


@pytest.mark.asyncio
async def test_neither_a_transcript_nor_a_caption_is_a_slash_command(monkeypatch, tmp_path):
    """Speech can come back as "/gate off", and a caption can be typed as one. An upload is never a command."""
    rig = Rig(monkeypatch, tmp_path, transcript="/gate off")
    await rig.upload(OGG, caption="/revoke all")
    assert not rig.sent[0][0].lstrip().startswith("/"), rig.sent[0][0]


@pytest.mark.asyncio
async def test_a_scoped_devices_voice_turn_stays_a_scoped_devices(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.upload(OGG, operator=False)
    assert rig.sent[0][1].get("operator") is False


@pytest.mark.asyncio
async def test_a_failed_transcription_still_reaches_the_turn_and_says_so(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, transcript=None)
    await rig.upload(OGG)
    assert len(rig.sent) == 1 and "could not be transcribed" in rig.sent[0][0]


@pytest.mark.asyncio
async def test_a_speech_server_going_down_is_said_in_the_chat(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, notices=("Your speech-to-text server isn't answering.",))
    await rig.upload(OGG)
    assert rig.system == ["Your speech-to-text server isn't answering."]


# ── the same hardening as every other inbound surface ───────────────────────

@pytest.mark.asyncio
async def test_bytes_that_are_not_the_declared_audio_are_refused(monkeypatch, tmp_path):
    """The renamed-extension case: a PNG declared as audio/ogg."""
    rig = Rig(monkeypatch, tmp_path)
    await rig.upload(PNG)
    assert rig.transcribed == [] and rig.sent == []
    assert rig.system and "do not match" in rig.system[0]


@pytest.mark.asyncio
async def test_an_audio_type_outside_the_allowlist_is_refused(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, {"gateway": {"media": {"allowed_audio_types": ["audio/wav"]}}})
    await rig.upload(OGG)
    assert rig.transcribed == [] and rig.sent == []
    assert rig.system and "Unsupported file type" in rig.system[0]


@pytest.mark.asyncio
async def test_audio_over_the_configured_size_is_refused(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, {"gateway": {"media": {"max_file_size_mb": 1}}})
    await rig.upload(OGG + b"\x00" * (1024 * 1024))
    assert rig.transcribed == [] and rig.sent == []
    assert rig.system and "too large" in rig.system[0]


@pytest.mark.asyncio
async def test_audio_with_no_signature_and_an_unlisted_type_is_refused(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.upload(b"\x00" * 64, "audio/x-made-up", filename="memo.bin")
    assert rig.transcribed == [] and rig.sent == []


# ── voice offered only where it works ───────────────────────────────────────

@pytest.mark.asyncio
async def test_audio_that_arrives_where_voice_is_unavailable_is_refused_in_the_chat(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, available=False)
    await rig.upload(OGG)
    assert rig.transcribed == [] and rig.sent == []
    assert rig.system and "voice" in rig.system[0].lower()


class Sock:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send(self, raw: str) -> None:
        self.frames.append(json.loads(raw))


@pytest.mark.parametrize("available", [True, False])
@pytest.mark.asyncio
async def test_the_subscribed_ack_says_whether_voice_input_works_here(monkeypatch, available):
    bridge = WebSocketBridge(session_mgr=None, loop_context=None, config={})
    monkeypatch.setattr(media_services, "voice_input_available", lambda config=None: available)
    ws = Sock()
    bridge._clients.add(ws)
    await bridge._handle_client_message(ws, json.dumps({"type": "subscribe", "payload": {}}))
    ack = [f for f in ws.frames if f["type"] == "subscribed"][-1]["payload"]
    assert ("voice_input" in ack["server_capabilities"]) is available


def _no_local_engine(monkeypatch, *, engine: str | None, ffmpeg: bool) -> None:
    from prometheus.infra import pieces
    from prometheus.tools.builtin import whisper_stt

    monkeypatch.setattr(pieces, "approved_offer", lambda kind: None)
    monkeypatch.setattr(whisper_stt, "whisper_engine_installed", lambda: engine is not None)
    monkeypatch.setattr(media_services.shutil, "which", lambda name: "/usr/bin/ffmpeg" if ffmpeg else None)


def test_a_remote_speech_server_makes_voice_input_available(monkeypatch):
    _no_local_engine(monkeypatch, engine=None, ffmpeg=False)
    assert media_services.voice_input_available({"stt": {"url": "http://127.0.0.1:9000"}}) is True


def test_a_local_engine_with_ffmpeg_makes_voice_input_available(monkeypatch):
    _no_local_engine(monkeypatch, engine="faster-whisper-python", ffmpeg=True)
    assert media_services.voice_input_available({}) is True


@pytest.mark.parametrize(("engine", "ffmpeg"), [(None, True), ("whisper", False), (None, False)])
def test_without_a_server_or_a_working_local_engine_voice_input_is_unavailable(monkeypatch, engine, ffmpeg):
    _no_local_engine(monkeypatch, engine=engine, ffmpeg=ffmpeg)
    assert media_services.voice_input_available({}) is False


def test_asking_never_imports_the_python_engine(monkeypatch):
    """faster_whisper loads its native runtime on import; the subscribe ack only looks for it."""
    import importlib.util

    from prometheus.tools.builtin import whisper_stt

    monkeypatch.setattr(whisper_stt.shutil, "which", lambda name: None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "faster_whisper" else None)
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)
    assert whisper_stt.whisper_engine_installed() is True
    assert "faster_whisper" not in sys.modules
