"""m4a (``audio/mp4``) and WebM (``audio/webm``) voice recordings are admitted, by their signatures.

Phones and browsers record in these two: an iPhone's voice memo is m4a, a browser's MediaRecorder is WebM/Opus (or
MP4 on Safari). The shipped audio allowlist named only ogg, mpeg and wav, so every one of them was refused.

THE RULE, the same as for the types already admitted (``gateway/media_guard``)
------------------------------------------------------------------------------
* Each has a signature, and the sniff decides: an ISO-media ``ftyp`` box with an audio or generic MP4 brand is
  ``audio/mp4``; an EBML header whose DocType is ``webm`` is ``audio/webm``. Bytes that sniff as one and are declared
  as another are refused, as a PNG declared ``audio/ogg`` is.
* An ``ftyp`` that is not audio (HEIC, AVIF, QuickTime) is NOT claimed as audio: it stays "no signature", as it was.
  So is a Matroska file that is not WebM.
* The spellings recorders send fold onto the sniffer's names: ``audio/x-m4a`` and ``audio/m4a`` are ``audio/mp4``, and
  a declared type's parameters (``audio/webm;codecs=opus``) are not part of the type.
"""

from __future__ import annotations

import base64
import sys
import types
from pathlib import Path

if "prometheus.memory" not in sys.modules:  # pragma: no cover - import plumbing
    try:
        import prometheus.memory  # noqa: F401
    except Exception:
        sys.modules["prometheus.memory"] = types.ModuleType("prometheus.memory")

import pytest  # noqa: E402
import yaml  # noqa: E402

from prometheus.gateway.media_guard import (  # noqa: E402
    MediaPolicy,
    MediaRejected,
    canonical_mime,
    check_declared_mime,
    sniff_mime,
    validate_inbound,
)

# Real leading bytes: an iPhone voice memo's ftyp box, and a browser MediaRecorder's EBML header.
M4A = b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00M4A isommp42" + b"\x00" * 32
MP4_GENERIC = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 32
SAFARI_ISO5 = b"\x00\x00\x00\x18ftypiso5\x00\x00\x02\x00iso5iso6mp41" + b"\x00" * 32
HEIC = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 32
MOV = b"\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00qt  " + b"\x00" * 32
WEBM = (b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01\x42\xf2\x81\x04\x42\xf3\x81\x08"
        b"\x42\x82\x84webm\x42\x87\x81\x04\x42\x85\x81\x02" + b"\x00" * 32)
MKV = (b"\x1a\x45\xdf\xa3\xa3\x42\x86\x81\x01\x42\xf7\x81\x01\x42\xf2\x81\x04\x42\xf3\x81\x08"
       b"\x42\x82\x88matroska\x42\x87\x81\x04\x42\x85\x81\x02" + b"\x00" * 32)
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 32
OGG = b"OggS\x00\x02\x00\x00" + b"\x00" * 32

TEMPLATE = Path(__file__).resolve().parents[1] / "config" / "prometheus.yaml.default"


def _shipped() -> MediaPolicy:
    media = yaml.safe_load(TEMPLATE.read_text())["gateway"]["media"]
    return MediaPolicy(
        allowed_image_types=tuple(media["allowed_image_types"]),
        allowed_audio_types=tuple(media["allowed_audio_types"]),
        allowed_document_types=tuple(media["allowed_document_types"]),
        max_file_size_mb=media.get("max_file_size_mb", 20),
    )


POLICY = _shipped()


# ── the signatures ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("data", [M4A, MP4_GENERIC, SAFARI_ISO5], ids=["m4a", "mp42", "iso5"])
def test_an_audio_or_generic_mp4_ftyp_sniffs_as_audio_mp4(data):
    assert sniff_mime(data) == "audio/mp4"


def test_a_webm_ebml_header_sniffs_as_audio_webm():
    assert sniff_mime(WEBM) == "audio/webm"


@pytest.mark.parametrize("data", [HEIC, MOV, MKV], ids=["heic", "quicktime", "matroska"])
def test_an_ftyp_or_ebml_file_that_is_not_one_of_them_is_not_claimed(data):
    assert sniff_mime(data) is None


def test_a_truncated_ftyp_is_not_claimed():
    assert sniff_mime(b"\x00\x00\x00\x20ftyp") is None


# ── the shipped allowlist and the spellings ─────────────────────────────────

def test_the_shipped_allowlist_admits_both():
    from prometheus.config.shipped_defaults import SHIPPED_ALLOWED_AUDIO_TYPES

    for mime in ("audio/mp4", "audio/webm"):
        assert mime in POLICY.allowed_audio_types
        assert mime in SHIPPED_ALLOWED_AUDIO_TYPES


@pytest.mark.parametrize(("declared", "data", "admitted"), [
    ("audio/mp4", M4A, "audio/mp4"),
    ("audio/x-m4a", M4A, "audio/mp4"),
    ("audio/m4a", M4A, "audio/mp4"),
    ("audio/mp4; codecs=mp4a.40.2", SAFARI_ISO5, "audio/mp4"),
    ("audio/webm", WEBM, "audio/webm"),
    ("audio/webm;codecs=opus", WEBM, "audio/webm"),
    (None, M4A, "audio/mp4"),
    (None, WEBM, "audio/webm"),
], ids=["mp4", "x-m4a", "m4a", "mp4-codecs", "webm", "webm-codecs", "undeclared-m4a", "undeclared-webm"])
def test_a_real_recording_is_admitted(declared, data, admitted):
    check_declared_mime(declared, "audio", POLICY)
    assert validate_inbound(data=data, declared_mime=declared, kind="audio", policy=POLICY) == admitted


def test_parameters_are_not_part_of_the_type():
    assert canonical_mime("Audio/WebM; codecs=opus") == "audio/webm"
    assert canonical_mime("audio/x-m4a") == "audio/mp4"


# ── the refusals that stay ──────────────────────────────────────────────────

@pytest.mark.parametrize(("declared", "data"), [
    ("audio/webm", M4A),     # an m4a presented as WebM
    ("audio/mp4", WEBM),     # and the other way round
    ("audio/ogg", WEBM),
    ("audio/mp4", PNG),      # a picture presented as a recording
    ("audio/webm", OGG),
], ids=["m4a-as-webm", "webm-as-mp4", "webm-as-ogg", "png-as-mp4", "ogg-as-webm"])
def test_bytes_that_disagree_with_the_declared_type_are_refused(declared, data):
    with pytest.raises(MediaRejected):
        validate_inbound(data=data, declared_mime=declared, kind="audio", policy=POLICY)


def test_an_operator_who_lists_only_the_old_three_still_refuses_m4a():
    old = MediaPolicy(allowed_image_types=(), allowed_audio_types=("audio/ogg", "audio/mpeg", "audio/wav"),
                      allowed_document_types=(), max_file_size_mb=20)
    with pytest.raises(MediaRejected):
        check_declared_mime("audio/x-m4a", "audio", old)
    with pytest.raises(MediaRejected):
        validate_inbound(data=M4A, declared_mime=None, kind="audio", policy=old)


def test_a_heic_an_operator_allowed_is_admitted_as_before():
    """Not claiming non-audio ftyp brands keeps HEIC on the declared-type path it was on."""
    heic = MediaPolicy(allowed_image_types=("image/heic",), allowed_audio_types=(), allowed_document_types=(),
                       max_file_size_mb=20)
    assert validate_inbound(data=HEIC, declared_mime="image/heic", kind="image", policy=heic) == "image/heic"


# ── through Beacon's chat_upload ────────────────────────────────────────────

@pytest.mark.parametrize(("data", "mime", "ext"), [(M4A, "audio/x-m4a", ".m4a"),
                                                  (WEBM, "audio/webm;codecs=opus", ".webm")], ids=["m4a", "webm"])
@pytest.mark.asyncio
async def test_a_phone_or_browser_recording_uploaded_from_beacon_is_transcribed(monkeypatch, tmp_path, data, mime,
                                                                                 ext):
    from prometheus.gateway import media_cache, media_services
    from prometheus.web.ws_server import WebSocketBridge

    bridge = WebSocketBridge(session_mgr=None, loop_context=None, config={})
    sent, paths, system = [], [], []

    async def send(session_id, text, **kw):
        sent.append(text)

    async def broadcast(frame):
        system.append(frame["payload"]["content"])

    async def transcribe(path, config=None, notices=None):
        paths.append(path)
        return "remind me at six"

    monkeypatch.setattr(bridge, "_handle_send_message", send)
    monkeypatch.setattr(bridge, "broadcast", broadcast)
    monkeypatch.setattr(media_services, "transcribe_audio", transcribe)
    monkeypatch.setattr(media_services, "voice_input_available", lambda config=None: True)
    monkeypatch.setattr(media_cache, "audio_cache_dir", lambda: tmp_path)
    await bridge._handle_file_upload("beacon:v", f"memo{ext}", base64.b64encode(data).decode(), mime, "")
    assert system == [], system
    assert paths and paths[0].endswith(ext), "the cached file must keep the container's extension for ffmpeg"
    assert sent and "remind me at six" in sent[0]
