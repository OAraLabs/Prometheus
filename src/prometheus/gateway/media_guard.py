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

* declared type present -> it must be allowlisted, it must AGREE with the
  sniffed type, and the sniffed type must be allowlisted
* declared type absent  -> the **sniffed** type alone must be allowlisted

This drops the comparison that has no second operand. It does not weaken the
control: in both branches the sniffed (unforgeable) type must be in the
allowlist. Making photos fail for lack of a declaration would reject all
photos; synthesising a declaration from the filename would compare a sniff
against a string the peer also controls, which proves nothing.

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
    try:
        sniffed = sniff_mime(data)
    except Exception as exc:  # pragma: no cover - defensive
        if not MEDIA_MIME_SNIFFED.on_error(exc):
            raise MediaRejected(
                MEDIA_MIME_SNIFFED.name, "Could not verify file type."
            ) from exc
        raise

    if sniffed is None:
        raise MediaRejected(
            MEDIA_MIME_SNIFFED.name,
            "Could not verify file type — refusing.",
        )

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
