"""ADMISSION tests — the missing half of the media guard's mutation matrix.

WHY THIS FILE EXISTS
--------------------
PR #140's 15-mutation matrix is thorough and every mutation asks the same
question: *"does disabling this control let something bad through?"* **None
asks "does this control let the PERMITTED things through?"**

A guard suite built entirely from refusal cases is structurally blind to
**over-refusal** — and over-refusal is how a security change becomes an
outage. #140 shipped with the Telegram document surface reduced to
**PDF-only**: 19 of the 20 advertised document types were refused, including
``text/plain`` and ``text/markdown``, which the allowlist explicitly permitted.
Every existing test passed, because none of them ever asserted that an
*allowed* type is actually admissible.

The rule this file enforces: **for every type the shipped config permits,
a legitimate file of that type must be ADMITTED.** Parametrized, so adding an
allowlist entry without a working path fails immediately.

These load the SHIPPED template, not a hand-built policy, so config drift is
what they are watching.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from prometheus.gateway.media_cache import SUPPORTED_DOCUMENT_TYPES
from prometheus.gateway.media_guard import (
    MediaPolicy,
    MediaRejected,
    check_declared_mime,
    validate_inbound,
)

TEMPLATE = Path(__file__).resolve().parents[1] / "config" / "prometheus.yaml.default"


def _shipped_policy() -> MediaPolicy:
    media = (yaml.safe_load(TEMPLATE.read_text())["gateway"]["media"]) or {}
    return MediaPolicy(
        allowed_image_types=tuple(media.get("allowed_image_types") or []),
        allowed_audio_types=tuple(media.get("allowed_audio_types") or []),
        allowed_document_types=tuple(media.get("allowed_document_types") or []),
        max_file_size_mb=media.get("max_file_size_mb", 20),
    )


POLICY = _shipped_policy()

# Real magic bytes. A "legitimate file" means one whose leading bytes are what
# the format actually starts with — anything less would test the test.
SAMPLES: dict[str, bytes] = {
    "image/jpeg": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01" + b"\x00" * 32,
    "image/png": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 32,
    "image/gif": b"GIF89a\x01\x00\x01\x00" + b"\x00" * 32,
    "image/webp": b"RIFF\x24\x08\x00\x00WEBPVP8 " + b"\x00" * 32,
    "audio/ogg": b"OggS\x00\x02\x00\x00" + b"\x00" * 32,
    "audio/mpeg": b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 32,
    "audio/wav": b"RIFF\x24\x08\x00\x00WAVEfmt " + b"\x00" * 32,
    "application/pdf": b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n",
}
# Signature-less formats: real content, no magic bytes. These are exactly the
# types the PDF-only regression refused.
TEXTUAL = b"# notes\n\nhello, this is real file content.\n"


def _sample_for(mime: str) -> bytes:
    return SAMPLES.get(mime, TEXTUAL)


# ── The agreement the regression violated ────────────────────────────────────

def test_extension_gate_and_document_allowlist_agree():
    """The 20-entry extension gate and the MIME allowlist must name one set.

    When the gate advertised 20 types and the allowlist named 3, the other 17
    passed the gate and were rejected a few lines later. Two lists describing
    the same policy must not be allowed to drift apart silently.
    """
    advertised = set(SUPPORTED_DOCUMENT_TYPES.values())
    allowed = set(POLICY.allowed_document_types)
    unreachable = advertised - allowed
    assert not unreachable, (
        "these types pass the extension gate but are refused by the media "
        f"allowlist: {sorted(unreachable)}"
    )


# ── Admission: every permitted type must actually get through ────────────────

@pytest.mark.parametrize(
    "ext,declared",
    sorted(SUPPORTED_DOCUMENT_TYPES.items()),
    ids=lambda v: v if isinstance(v, str) else str(v),
)
def test_every_advertised_document_extension_is_admitted(ext, declared):
    """For all 20 advertised document extensions, a real file gets through.

    This is the test whose absence let the PDF-only regression ship.
    """
    data = _sample_for(declared)
    check_declared_mime(declared, "document", POLICY)  # pre-download half
    admitted = validate_inbound(
        data=data, declared_mime=declared, kind="document", policy=POLICY
    )
    assert admitted, f"{ext} ({declared}) produced no admitted type"


@pytest.mark.parametrize("declared", sorted(POLICY.allowed_image_types))
def test_every_allowed_image_type_is_admitted(declared):
    check_declared_mime(declared, "image", POLICY)
    assert (
        validate_inbound(
            data=_sample_for(declared),
            declared_mime=declared,
            kind="image",
            policy=POLICY,
        )
        == declared
    )


@pytest.mark.parametrize("declared", sorted(POLICY.allowed_audio_types))
def test_every_allowed_audio_type_is_admitted(declared):
    check_declared_mime(declared, "audio", POLICY)
    assert (
        validate_inbound(
            data=_sample_for(declared),
            declared_mime=declared,
            kind="audio",
            policy=POLICY,
        )
        == declared
    )


@pytest.mark.parametrize("declared", sorted(POLICY.allowed_image_types))
def test_allowed_image_types_are_admitted_without_a_declaration(declared):
    """The PhotoSize case: no declared type, so the sniff alone must suffice.

    Images all have signatures, so this branch must work for every one of them
    — a photo arriving with no ``mime_type`` is the normal case, not an edge.
    """
    assert (
        validate_inbound(
            data=_sample_for(declared), declared_mime=None, kind="image", policy=POLICY
        )
        == declared
    )


# ── Regression pins for the three specific defects found ─────────────────────

@pytest.mark.parametrize("declared", ["text/plain", "text/markdown"])
def test_signature_less_text_is_admitted_on_its_declared_type(declared):
    """THE regression: sniff returns None for text, which is UNKNOWN not DENIED.

    Text has no magic bytes and never will. Refusing on ``sniffed is None``
    refused the two text types the allowlist explicitly permitted.
    """
    assert (
        validate_inbound(
            data=TEXTUAL, declared_mime=declared, kind="document", policy=POLICY
        )
        == declared
    )


def test_mp3_is_admitted_despite_the_audio_mp3_spelling():
    """``audio/mp3`` looks right and is wrong — real MP3s sniff as audio/mpeg.

    The shipped allowlist previously said ``audio/mp3``, so every MP3 was
    refused by the allowlist. Both spellings must now reach admission.
    """
    assert "audio/mpeg" in POLICY.allowed_audio_types, "allowlist must use audio/mpeg"
    for declared in ("audio/mpeg", "audio/mp3"):
        assert (
            validate_inbound(
                data=SAMPLES["audio/mpeg"],
                declared_mime=declared,
                kind="audio",
                policy=POLICY,
            )
            == "audio/mpeg"
        )


# ── The refusals that must SURVIVE this change ───────────────────────────────

def test_no_declaration_and_no_signature_still_fails_closed():
    """The sticker case keeps its fail-closed direction.

    Admitting here would admit arbitrary bytes: neither signal said anything.
    This is the one branch the UNKNOWN-is-not-DENIED change must not relax.
    """
    with pytest.raises(MediaRejected) as exc:
        validate_inbound(
            data=b"\x00\x01\x02 not any known format",
            declared_mime=None,
            kind="document",
            policy=POLICY,
        )
    assert exc.value.guard_name == "media.mime_sniffed"


def test_a_signature_that_disagrees_with_the_declaration_is_still_refused():
    """Renamed-extension detection is the check that actually matters."""
    with pytest.raises(MediaRejected) as exc:
        validate_inbound(
            data=SAMPLES["application/pdf"],
            declared_mime="image/png",
            kind="image",
            policy=POLICY,
        )
    assert exc.value.guard_name == "media.mime_sniffed"


def test_a_declared_type_outside_the_allowlist_is_still_refused():
    """Admitting on the declared type does NOT mean admitting any declaration."""
    with pytest.raises(MediaRejected) as exc:
        validate_inbound(
            data=TEXTUAL,
            declared_mime="application/x-msdownload",
            kind="document",
            policy=POLICY,
        )
    assert exc.value.guard_name == "media.allowlist"


def test_unknown_bytes_declared_as_an_allowlisted_type_are_bounded_by_the_allowlist():
    """The honest cost of the fallback, pinned so it is a decision not a drift.

    A signature-less format is admitted on its declared type. That is the
    documented trade-off; what must NOT happen is the allowlist ceasing to
    bound which declarations are acceptable.
    """
    assert (
        validate_inbound(
            data=b"\x00\xff\xfe arbitrary bytes",
            declared_mime="text/plain",
            kind="document",
            policy=POLICY,
        )
        == "text/plain"
    )
    with pytest.raises(MediaRejected):
        validate_inbound(
            data=b"\x00\xff\xfe arbitrary bytes",
            declared_mime="application/x-sh-not-allowed",
            kind="document",
            policy=POLICY,
        )
