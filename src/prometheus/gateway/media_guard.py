"""Inbound media validation for the public gateway surface.

WHY THIS EXISTS
---------------
``gateway.media.allowed_{image,audio,document}_types`` and ``max_file_size_mb``
were declared in config with **no enforcement anywhere on the Telegram inbound
path**. The four handlers went ``get_file()`` -> ``download_as_bytearray()``
with no type check and — except one hardcoded 20 MB on documents — no size
check. A reader who greps the config finds an allowlist and a size cap and
stops looking; that is worse than having neither.

ORDER OF CHECKS, and why it is this order
-----------------------------------------
1. **declared MIME** — free, arrives before any transfer
2. **size pre-check** — from ``file_size``, still before any transfer
3. **download under a hard byte ceiling** — because ``file_size`` is supplied
   by the peer and step 2 believed it
4. **sniff magic bytes** — the only type signal the peer does not control
5. **agreement + allowlist**

Cheapest and earliest first. Nothing is transferred before its declared size
has been refused.

⚠ THE PHOTO/STICKER ASYMMETRY — read this before "fixing" it
-------------------------------------------------------------
Telegram's ``PhotoSize`` and ``Sticker`` objects carry **no ``mime_type``
field**; only ``Voice``, ``Document`` and ``Audio`` do. So for photos and
stickers there is no declared type to compare the sniffed type against.

The rule is therefore asymmetric BY NECESSITY, not by oversight:

* declared type present -> it must be allowlisted, and it must AGREE with the
  sniffed type when there IS one
* declared type absent  -> the **sniffed** type alone must be allowlisted

This drops the comparison that has no second operand. Making photos fail for
lack of a declaration would reject all photos; synthesising a declaration from
the filename would compare a sniff against a string the peer also controls,
which proves nothing.

⚠ UNKNOWN IS NOT DISALLOWED — the distinction this file got wrong once
----------------------------------------------------------------------
``sniff_mime`` returns ``None`` for "I have no signature for this", which is
NOT the same claim as "this type is not permitted". Conflating them made the
document surface **PDF-only**: 19 of the 20 advertised document types were
refused, including ``text/plain`` and ``text/markdown``, which the allowlist
explicitly permitted. Text formats have no magic bytes — there is no signature
to add — so "no signature" can never be a refusal reason for them.

The resolution, and why each branch fails the way it does:

* sniff returns a type, and it DISAGREES with the declared type -> **refuse**.
  This is the renamed-extension attack and the check that actually matters.
* sniff returns ``None`` and a declared type exists -> **admit on the declared
  type if it is allowlisted**, and log it. The declared type is peer-supplied,
  but for a signature-less format it is the only evidence there is, and the
  allowlist still bounds which declarations are acceptable.
* sniff returns ``None`` and NO declared type exists -> **refuse**. Nothing to
  fall back to; admitting would admit arbitrary bytes.

⚠ The honest cost: for signature-less formats the declared type is trusted.
A payload can be presented as ``text/plain`` and be admitted as long as no
signature in the table contradicts it. That is strictly weaker than verifying
magic bytes — and strictly stronger than the behaviour it replaces, which
refused the type outright and so protected nothing while breaking everything.
Adding signatures for common container formats would narrow the gap further
(a ZIP presented as ``text/plain`` would then be caught by disagreement);
the table currently has none.

Source: novel code for Prometheus, 2026-08-03.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from prometheus.gateway.guards import (
    MEDIA_ALLOWLIST,
    MEDIA_BYTE_CEILING,
    MEDIA_MIME_DECLARED,
    MEDIA_MIME_SNIFFED,
    MEDIA_SIZE_PRECHECK,
)

log = logging.getLogger(__name__)


class MediaRejected(Exception):
    """A control refused this media. Carries sender-safe text."""

    def __init__(self, guard_name: str, message: str) -> None:
        super().__init__(message)
        self.guard_name = guard_name
        self.message = message


class MediaTooLarge(MediaRejected):
    """The byte ceiling tripped mid-transfer — i.e. file_size lied."""


# ── Magic-byte signatures ───────────────────────────────────────────────────
# A small in-repo table rather than a dependency: `magic`/`filetype`/`puremagic`
# are not installed, and `imghdr` is deprecated and removed in Python 3.13.
# Covers exactly the types the shipped allowlists name. Longest prefixes are
# checked first so subtypes are not shadowed by their container.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"%PDF-", "application/pdf"),
    (b"OggS", "audio/ogg"),
    (b"ID3", "audio/mpeg"),
    (b"\xff\xfb", "audio/mpeg"),
    (b"\xff\xf3", "audio/mpeg"),
    (b"\xff\xf2", "audio/mpeg"),
)
# Container formats needing a second probe at a fixed offset.
_RIFF_KINDS = {b"WEBP": "image/webp", b"WAVE": "audio/wav"}


def sniff_mime(data: bytes) -> str | None:
    """Return the MIME type implied by *data*'s magic bytes, or None.

    None means "unrecognised", not "safe" — callers must treat it as a
    refusal, because an allowlist that admits unknown bytes is not one.
    """
    if not data:
        return None
    for prefix, mime in _SIGNATURES:
        if data.startswith(prefix):
            return mime
    if data[:4] == b"RIFF" and len(data) >= 12:
        return _RIFF_KINDS.get(data[8:12])
    return None


# Spelling variants a client may declare for a format the signature table emits
# under its IANA name. Both halves of a disagreement check must speak the same
# vocabulary: without this, a client declaring the legacy "audio/mp3" for bytes
# that sniff as "audio/mpeg" is refused for DISAGREEMENT — two spellings of one
# format reported as a mismatch. Aliases are folded, never widened: this maps
# names onto each other, it does not admit anything the allowlist excludes.
_MIME_ALIASES: dict[str, str] = {
    "audio/mp3": "audio/mpeg",
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "image/jpg": "image/jpeg",
}


def canonical_mime(mime: str | None) -> str | None:
    """Fold known spelling variants onto the name the sniffer emits."""
    if mime is None:
        return None
    return _MIME_ALIASES.get(mime.strip().lower(), mime.strip().lower())


@dataclass(frozen=True)
class MediaPolicy:
    """Effective media policy, resolved from config (never from defaults alone)."""

    allowed_image_types: tuple[str, ...]
    allowed_audio_types: tuple[str, ...]
    allowed_document_types: tuple[str, ...]
    max_file_size_mb: int

    @property
    def max_bytes(self) -> int:
        return int(self.max_file_size_mb) * 1024 * 1024

    def allowlist_for(self, kind: str) -> tuple[str, ...]:
        return {
            "image": self.allowed_image_types,
            "audio": self.allowed_audio_types,
            "document": self.allowed_document_types,
        }.get(kind, ())


def check_size_precheck(file_size: object, policy: MediaPolicy) -> None:
    """Refuse before any transfer. ``file_size`` is peer-supplied — see below.

    A non-numeric ``file_size`` is treated as UNKNOWN, not as a refusal and not
    as a crash. Absent-or-unknown already means "proceed"; the byte ceiling is
    the control that does not trust this number anyway, so an unparseable value
    degrades to exactly the case the ceiling exists for.

    Comparing directly (``file_size > max``) raised TypeError on a non-int —
    a CONTROL failing in an *undefined* direction rather than a declared one.
    Found by a pre-existing pin test passing a MagicMock, not by these tests.
    """
    try:
        size = int(file_size)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return
    if size > policy.max_bytes:
        raise MediaRejected(
            MEDIA_SIZE_PRECHECK.name,
            f"File too large (max {policy.max_file_size_mb} MB).",
        )


def check_declared_mime(declared: str | None, kind: str, policy: MediaPolicy) -> None:
    """Free, pre-download. Absent is NOT a failure — see the module docstring."""
    if declared is None:
        return
    declared = canonical_mime(declared)
    allowed = policy.allowlist_for(kind)
    if allowed and declared not in allowed:
        raise MediaRejected(
            MEDIA_MIME_DECLARED.name,
            f"Unsupported file type ({declared}).",
        )


def check_sniffed_mime(
    data: bytes, declared: str | None, kind: str, policy: MediaPolicy
) -> str:
    """The unforgeable half. Returns the sniffed type; raises on any refusal.

    Fails CLOSED on a sniff error: a control that degrades to "allow" when its
    detector breaks is not a control.
    """
    declared = canonical_mime(declared)
    try:
        sniffed = sniff_mime(data)
    except Exception as exc:  # pragma: no cover - defensive
        if not MEDIA_MIME_SNIFFED.on_error(exc):
            raise MediaRejected(
                MEDIA_MIME_SNIFFED.name, "Could not verify file type."
            ) from exc
        raise

    if sniffed is None:
        # UNKNOWN is not DISALLOWED — see the module docstring. Treating the
        # two as the same thing is what made this surface PDF-only.
        if declared is None:
            # Nothing to fall back to (the PhotoSize/Sticker case). Stays
            # FAIL-CLOSED: admitting here would admit arbitrary bytes, since
            # neither signal said anything at all.
            raise MediaRejected(
                MEDIA_MIME_SNIFFED.name,
                "Could not verify file type — refusing.",
            )
        # A declared type exists and no signature contradicts it. The
        # allowlist still decides; the declared type is the only evidence
        # available for a format that has no magic bytes.
        allowed = policy.allowlist_for(kind)
        if allowed and declared not in allowed:
            raise MediaRejected(
                MEDIA_ALLOWLIST.name, f"Unsupported file type ({declared})."
            )
        log.info(
            "media: no signature for declared type %r (%s) — admitted on the "
            "declared type; signature-less format",
            declared,
            kind,
        )
        return declared

    if declared is not None and sniffed != declared:
        # The renamed-extension case: a .pdf presented as image/png.
        raise MediaRejected(
            MEDIA_MIME_SNIFFED.name,
            f"File contents ({sniffed}) do not match declared type ({declared}).",
        )

    allowed = policy.allowlist_for(kind)
    if allowed and sniffed not in allowed:
        raise MediaRejected(
            MEDIA_ALLOWLIST.name, f"Unsupported file type ({sniffed})."
        )
    return sniffed


def enforce_byte_ceiling(data: bytes, policy: MediaPolicy) -> bytes:
    """Fail loudly when the transferred size exceeds the cap.

    ⚠ HONEST LABEL: this is a POST-DOWNLOAD assertion, not a stream cap.
    ``python-telegram-bot``'s ``download_as_bytearray()`` is all-or-nothing —
    it returns the complete payload or raises, exposing no incremental hook —
    so the bytes have already crossed the wire by the time this runs.

    What it DOES buy: a peer that lies in ``file_size`` to slip past the
    pre-check cannot get the oversized payload *processed, cached or fed to a
    model*. What it does NOT buy: protection from the transfer itself. A true
    stream cap needs a different download API and is a separate change.

    Labelled rather than implied, because "we believed the attacker's number"
    is a bad line in a post-mortem and "we implied a stream cap we did not
    build" is a worse one.
    """
    if len(data) > policy.max_bytes:
        raise MediaTooLarge(
            MEDIA_BYTE_CEILING.name,
            f"File too large (max {policy.max_file_size_mb} MB).",
        )
    return data


def validate_inbound(
    *,
    data: bytes,
    declared_mime: str | None,
    kind: str,
    policy: MediaPolicy,
) -> str:
    """Post-transfer half of the pipeline: ceiling, sniff, agreement, allowlist.

    The pre-transfer half (``check_declared_mime`` then
    ``check_size_precheck``) runs at the call site *before* the download, which
    is the whole point of splitting them.
    """
    enforce_byte_ceiling(data, policy)
    return check_sniffed_mime(data, declared_mime, kind, policy)
