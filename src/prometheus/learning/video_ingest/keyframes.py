"""Keyframe extraction — SSIM near-duplicate filtering of extracted frames.

Ported from skillforge-engine ``core/capture.py::extract_keyframes``:
duration-scaled keyframe budget, forced keyframes when a visually static
stretch exceeds ``force_interval``, a start pad so the SSIM diff engine
has a baseline before the first user action, and a blank-frame skip.
Dropped: the cursor-event enrichment (``enrich_with_events``) — there
are no cursor events in the video-ingestion path.

The SSIM stack (Pillow + numpy + scikit-image) is import-guarded. When
unavailable, the fallback keeps every Nth frame so the count stays under
``max_keyframes`` (the original kept *all* frames, which would flood the
vision digest on long videos).
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Frames smaller than this are treated as blank/corrupt and skipped.
_MIN_FRAME_BYTES = 5000


def _load_ssim_deps() -> tuple[Any, Any, Any] | None:
    """Import guard for the SSIM stack.

    Returns ``(Image, numpy, ssim)`` or ``None`` when any piece is
    missing. Split out so tests can monkeypatch the no-SSIM fallback.
    """
    try:
        import numpy as np
        from PIL import Image
        from skimage.metrics import structural_similarity as ssim

        return Image, np, ssim
    except ImportError:
        return None


def _frame_timestamp(frame_name: str, frame_fps: float) -> float | None:
    """Derive seconds from a frame filename (frame_0042.png -> 21.0s at 2fps)."""
    m = re.search(r"(\d+)", frame_name)
    if m and frame_fps > 0:
        return int(m.group(1)) / frame_fps
    return None


def _scaled_max_keyframes(frame_count: int, frame_fps: float, config_cap: int) -> tuple[int, float | None]:
    """Scale the keyframe budget with video duration.

    Short (<=5 min): 60 baseline. Medium (5-20 min): ~15/minute. Long
    (>20 min): hard cap 400. The configured cap is always respected as a
    hard maximum. Returns ``(max_keyframes, tightened_threshold)`` where
    the threshold is ``0.92`` for videos over five minutes, else None.
    """
    duration_minutes = (frame_count / frame_fps) / 60 if frame_fps > 0 else 0.0
    if duration_minutes <= 5:
        scaled = 60
    elif duration_minutes <= 20:
        scaled = int(duration_minutes * 15)
    else:
        scaled = 400
    tightened = 0.92 if duration_minutes > 5 else None
    return min(scaled, config_cap), tightened


def _fallback_every_nth(
    frames: list[str],
    screenshots_dir: Path,
    keyframes_dir: Path,
    max_keyframes: int,
    frame_fps: float,
) -> list[dict[str, Any]]:
    """No-SSIM fallback: keep every Nth frame to stay under the budget."""
    stride = max(1, math.ceil(len(frames) / max_keyframes))
    keyframes: list[dict[str, Any]] = []
    for i in range(0, len(frames), stride):
        if len(keyframes) >= max_keyframes:
            break
        frame_name = frames[i]
        src_path = screenshots_dir / frame_name
        if src_path.stat().st_size < _MIN_FRAME_BYTES:
            log.warning("Skipping %s (likely blank frame)", frame_name)
            continue
        dst_path = keyframes_dir / f"keyframe_{len(keyframes):04d}.png"
        shutil.copy2(src_path, dst_path)
        keyframes.append({
            "index": len(keyframes),
            "source_frame": frame_name,
            "path": str(dst_path),
            "forced": False,
            "timestamp": _frame_timestamp(frame_name, frame_fps),
        })
    return keyframes


def extract_keyframes(
    session_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    similarity_threshold: float = 0.95,
    frame_fps: float = 2.0,
    force_interval: float = 10.0,
    start_pad_frames: int = 3,
    max_keyframes: int = 60,
) -> list[dict[str, Any]]:
    """Extract keyframes from ``session_dir/screenshots/``, filtering near-duplicates.

    Uses structural similarity (SSIM) to detect significant visual
    changes; keeps every Nth frame when the SSIM stack is unavailable.
    Selected frames are copied to ``output_dir/keyframes/`` and a
    ``keyframes.json`` manifest is written next to them.

    Args:
        session_dir: Directory containing ``screenshots/frame_*.png``.
        output_dir: Where to write ``keyframes/`` + ``keyframes.json``
            (defaults to ``session_dir``).
        similarity_threshold: SSIM score below which a frame is a keyframe.
        frame_fps: Extraction rate the frames were produced at.
        force_interval: Max seconds allowed between keyframes — prevents
            blind spots when critical actions happen during visually
            static stretches.
        start_pad_frames: Frames at the start always kept, giving SSIM a
            baseline before the first user action fires.
        max_keyframes: Hard cap (scaled with duration, never exceeded).

    Returns:
        Keyframe manifest entries:
        ``{index, source_frame, path, forced, timestamp}``.
    """
    session = Path(session_dir)
    screenshots_dir = session / "screenshots"
    if not screenshots_dir.is_dir():
        log.error("No screenshots directory found at %s", screenshots_dir)
        return []

    out = Path(output_dir) if output_dir is not None else session
    keyframes_dir = out / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(f.name for f in screenshots_dir.glob("*.png"))
    if not frames:
        log.warning("No screenshots found in %s", screenshots_dir)
        return []

    max_keyframes, tightened = _scaled_max_keyframes(len(frames), frame_fps, max_keyframes)
    if tightened is not None:
        log.info(
            "SSIM threshold tightened from %.2f to %.2f for longer video",
            similarity_threshold, tightened,
        )
        similarity_threshold = tightened
    log.info(
        "Processing %d frames for keyframe extraction (force every %.0fs, max %d keyframes)",
        len(frames), force_interval, max_keyframes,
    )

    deps = _load_ssim_deps()
    if deps is None:
        log.warning(
            "scikit-image/Pillow/numpy not available — falling back to "
            "every-Nth-frame sampling (install scikit-image for SSIM dedup)"
        )
        keyframes = _fallback_every_nth(
            frames, screenshots_dir, keyframes_dir, max_keyframes, frame_fps,
        )
        _write_manifest(out, keyframes)
        return keyframes

    Image, np, ssim = deps

    keyframes: list[dict[str, Any]] = []
    prev_array = None
    last_keyframe_ts: float | None = None
    forced_count = 0

    for i, frame_name in enumerate(frames):
        src_path = screenshots_dir / frame_name
        is_keyframe = False
        forced = False

        curr_ts = _frame_timestamp(frame_name, frame_fps)

        # Start pad: always include the first N frames — establishes the
        # SSIM baseline so the trigger click is never cut off.
        if i < start_pad_frames:
            is_keyframe = True
            forced = True

        # Force-keyframe: gap since last captured keyframe exceeded the window
        if not is_keyframe and (
            last_keyframe_ts is not None
            and curr_ts is not None
            and (curr_ts - last_keyframe_ts) >= force_interval
        ):
            is_keyframe = True
            forced = True

        if not is_keyframe:
            if prev_array is None:
                is_keyframe = True
            else:
                try:
                    img = Image.open(src_path).convert("L").resize((320, 240))
                    curr_array = np.array(img)
                    score = ssim(prev_array, curr_array)
                    if score < similarity_threshold:
                        is_keyframe = True
                except Exception as exc:  # noqa: BLE001 — keep frame on SSIM failure
                    log.warning("SSIM failed for %s: %s", frame_name, exc)
                    is_keyframe = True

        if not is_keyframe:
            continue

        # Hard cap: stop adding keyframes once we hit the maximum
        if len(keyframes) >= max_keyframes:
            log.warning("Reached maximum keyframe limit (%d). Stopping.", max_keyframes)
            break

        # Quality check: skip blank/corrupt frames
        if src_path.stat().st_size < _MIN_FRAME_BYTES:
            log.warning("Skipping %s (likely blank frame)", frame_name)
            continue

        dst_path = keyframes_dir / f"keyframe_{len(keyframes):04d}.png"
        shutil.copy2(src_path, dst_path)

        if forced:
            forced_count += 1

        keyframes.append({
            "index": len(keyframes),
            "source_frame": frame_name,
            "path": str(dst_path),
            "forced": forced,
            "timestamp": curr_ts,
        })
        last_keyframe_ts = curr_ts

        try:
            img = Image.open(src_path).convert("L").resize((320, 240))
            prev_array = np.array(img)
        except Exception:  # noqa: BLE001 — baseline update is best-effort
            pass

    if forced_count:
        log.info("Force-included %d keyframes (pad or gap > %.0fs)", forced_count, force_interval)
    log.info("Extracted %d keyframes from %d frames", len(keyframes), len(frames))

    _write_manifest(out, keyframes)
    return keyframes


def _write_manifest(output_dir: Path, keyframes: list[dict[str, Any]]) -> None:
    """Write the keyframes.json manifest. Best-effort."""
    manifest_path = output_dir / "keyframes.json"
    try:
        manifest_path.write_text(
            json.dumps(keyframes, indent=2, default=str), encoding="utf-8",
        )
        log.info("Keyframe manifest -> %s", manifest_path)
    except OSError:
        log.warning("Failed to write %s", manifest_path, exc_info=True)
