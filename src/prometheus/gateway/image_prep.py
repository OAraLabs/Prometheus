"""Turn a cached image file into an :class:`ImageBlock`, once, for every provider.

ONE PLACE ON PURPOSE (spec Q3). If each provider learned its own size limits we
would have four subtly different answers to "how big is too big" and no way to
tell which one truncated a screenshot. This runs before the block is built, so
every provider gets identically prepared bytes.

WHY DOWNSCALE AT ALL, when hosted providers downscale server-side anyway:

  * cost — vision is billed per visual token, and resolution above the
    provider's tiling threshold buys literally nothing
  * payload — server-side downscale does not bound what crosses the WS gateway
    and sits in the request body on the way there

WHY Pillow: it resamples at any ratio, so a shrunk image lands exactly on the
cap. Until 0.9.9 this used PyMuPDF (AGPL, absent from the Mac app), whose
``Pixmap.shrink`` reduces only by powers of two: a page rendered at 4896x6336
came out 2448x3168, 23% over the cap (docs/audits/pdf-extraction/).

THE ORIGINAL DIMENSIONS ARE LOGGED. "Why did it not see the detail" is a
question that gets asked weeks later, and the only way to answer it is to know
what was sent versus what was captured.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

from prometheus.engine.messages import ImageBlock

log = logging.getLogger(__name__)

# Long-edge cap in pixels. 2576 is the largest long edge any current hosted tier
# uses before it downscales for you, so at this size nothing is thrown away that
# the model would have looked at, and the payload stays bounded.
MAX_LONG_EDGE = 2576

# Shrink only ABOVE this, not above the cap itself.
#
# WHY, measured. When this shrank by powers of two (PyMuPDF), shrinking anything
# merely over the cap overshot hard: the wire test's 2638x1646 screenshot was 62px
# past 2576 and came out 1319x823 — HALF the resolution discarded to clear a 2%
# overshoot. It cost real legibility: Claude read the sidebar chip "Chats 118"
# as "Chats 115" and the status pill "LVX" as "LYX". Small UI text is exactly
# what a screenshot is sent for.
#
# Between the cap and this trigger, send as captured and let the provider's own
# resampler take it down: the original bytes are never re-encoded for a 2% gain,
# and an image in this band is close enough that the payload is not the problem.
# Above the trigger the payload IS the problem (a 4x-oversized image crossing a
# WS gateway), and it is resampled to exactly the cap.
SHRINK_ABOVE = MAX_LONG_EDGE * 2

_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
    (b"RIFF", "image/webp"),
)


def sniff_media_type(data: bytes) -> str | None:
    """Media type from magic bytes, or None for something that is not an image.

    From the bytes, never the file extension: the extension is attacker- and
    typo-controlled, and a mislabelled media_type is a provider-side 400 that
    reads like an outage.
    """
    for magic, media_type in _MAGIC:
        if data.startswith(magic):
            return media_type
    return None


def prepare_image_block(path: str | Path) -> ImageBlock | None:
    """Read a cached image and return the block to send, or None if unreadable.

    None means "fall back to the description path" — it is not an error, and it
    must never be mistaken for "sent successfully". The caller decides.
    """
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:
        log.warning("image_prep: cannot read %s: %s", p, exc)
        return None

    media_type = sniff_media_type(data)
    if media_type is None:
        log.warning("image_prep: %s is not a recognised image (magic bytes)", p)
        return None

    data, media_type = _downscale_if_needed(data, media_type, p)
    return ImageBlock(
        media_type=media_type,
        data=base64.b64encode(data).decode("ascii"),
        source_path=str(p),
    )


def _downscale_if_needed(data: bytes, media_type: str, p: Path) -> tuple[bytes, str]:
    """Resample to exactly ``MAX_LONG_EDGE`` above the trigger. Original bytes if it fits.

    Returning the ORIGINAL bytes untouched when no shrink is needed matters: a
    re-encode would re-compress a screenshot for no reason, and JPEG artifacts
    on small text are exactly the thing that makes a model misread a UI.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - declared dependency
        log.warning("image_prep: Pillow unavailable, sending %s at full size", p)
        return data, media_type

    try:
        img = Image.open(io.BytesIO(data))
        width, height = img.size
    except Exception as exc:
        # Unreadable by Pillow but valid by magic bytes (an exotic PNG variant,
        # say). Send it as captured rather than dropping the picture.
        log.warning("image_prep: %s not decodable for resize (%s), sending as-is", p, exc)
        return data, media_type

    if max(width, height) <= SHRINK_ABOVE:
        # Includes everything between the cap and 2x it: as-captured beats a
        # re-encode the provider's own resampler makes unnecessary.
        log.info(
            "image_prep: %s %dx%d sent as captured (%d bytes, cap=%d, shrink above %d)",
            p.name, width, height, len(data), MAX_LONG_EDGE, SHRINK_ABOVE,
        )
        return data, media_type

    try:
        img.load()
        img.thumbnail((MAX_LONG_EDGE, MAX_LONG_EDGE), Image.Resampling.LANCZOS)
        out_fmt = "JPEG" if media_type == "image/jpeg" else "PNG"
        if out_fmt == "JPEG" and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format=out_fmt, **({"quality": 90} if out_fmt == "JPEG" else {}))
        resized = buf.getvalue()
    except Exception as exc:
        log.warning("image_prep: resize of %s failed (%s), sending as-is", p, exc)
        return data, media_type

    log.info(
        "image_prep: %s downscaled %dx%d -> %dx%d (%d -> %d bytes)",
        p.name, width, height, img.width, img.height, len(data), len(resized),
    )
    return resized, ("image/jpeg" if out_fmt == "JPEG" else "image/png")
