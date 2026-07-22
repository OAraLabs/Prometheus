"""Video acquisition, frame extraction, and narration transcription.

Ported from skillforge-engine ``ingest_video.py`` and
``core/audio_transcribe.py``. Frame extraction uses the ffmpeg subprocess
invocation from ``core/capture.py::extract_frames_from_video`` (the
imageio frame-iterator approach in the original ``ingest_video.py``
assumed a 30fps source; ffmpeg's ``fps=`` filter is exact).

Optional dependencies are import-guarded with actionable errors:

- ``yt-dlp`` — only needed for :func:`download_video` (URL sources).
- ``imageio-ffmpeg`` — preferred ffmpeg source; falls back to a system
  ``ffmpeg`` on PATH.
- ``faster-whisper`` — only needed for :func:`transcribe`; already part
  of Prometheus's ``voice`` extra.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Narration segments within this many seconds of a keyframe's timestamp
# are attached to it (port of align_to_frames' WINDOW).
_NARRATION_WINDOW_SECONDS = 3.0


def _find_ffmpeg() -> str:
    """Return a path to an ffmpeg binary.

    Prefers the bundled imageio-ffmpeg binary (no system install needed),
    then falls back to ``ffmpeg`` on PATH.
    """
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 — ImportError or missing-binary RuntimeError
        pass

    exe = shutil.which("ffmpeg")
    if exe:
        return exe

    raise RuntimeError(
        "ffmpeg not found. Install the bundled binary with "
        "`pip install imageio-ffmpeg`, or install ffmpeg system-wide "
        "(e.g. `brew install ffmpeg`)."
    )


# ── Video download ───────────────────────────────────────────────────


def download_video(url: str, dest_dir: Path | str) -> Path:
    """Download a video from a URL (YouTube etc.) via yt-dlp.

    Prefers mp4 and caps resolution at 1080p — frames are downscaled to
    1280x720 for the vision model anyway, so higher source resolution is
    wasted bandwidth.

    Returns:
        Path to the downloaded video file inside ``dest_dir``.
    """
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is required to download videos from URLs. "
            "Install it with: pip install yt-dlp"
        ) from exc

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": (
            "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]"
            "/best[height<=1080][ext=mp4]"
            "/best[height<=1080]"
            "/best"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(dest / "%(title).100s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    log.info("Downloading video from %s", url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))

    if not path.exists():
        # Post-merge the container extension may differ from the template.
        merged = path.with_suffix(".mp4")
        if merged.exists():
            path = merged
        else:
            raise RuntimeError(f"yt-dlp reported success but no file found at {path}")

    log.info("Downloaded %s (%.1f MB)", path.name, path.stat().st_size / 1_000_000)
    return path


# ── Frame extraction ─────────────────────────────────────────────────


def extract_frames(video_path: Path | str, session_dir: Path | str, fps: float = 2.0) -> int:
    """Extract PNG frames at ``fps`` into ``session_dir/screenshots/``.

    Frames are scaled to 1280x720 (lanczos) — the resolution the
    keyframe SSIM filter and the vision prompts were tuned against.

    Returns:
        Number of frames written.
    """
    shots_dir = Path(session_dir) / "screenshots"
    shots_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, "-i", str(video_path),
        "-vf", f"fps={fps},scale=1280:720:flags=lanczos",
        "-q:v", "2",
        str(shots_dir / "frame_%04d.png"),
        "-y",
    ]
    log.info("Extracting frames at %.1ffps from %s", fps, Path(video_path).name)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    count = len(list(shots_dir.glob("*.png")))
    if count == 0:
        stderr_tail = (result.stderr or "")[-300:]
        raise RuntimeError(f"ffmpeg frame extraction produced no frames: {stderr_tail}")

    log.info("Extracted %d frames -> %s", count, shots_dir)
    return count


# ── Audio transcription ──────────────────────────────────────────────


def _extract_audio(video_path: Path | str, session_dir: Path | str) -> Path | None:
    """Extract mono 16kHz WAV audio (Whisper's expected input). Best-effort."""
    try:
        ffmpeg = _find_ffmpeg()
    except RuntimeError as exc:
        log.warning("Audio extraction skipped: %s", exc)
        return None

    out_dir = Path(session_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / "audio.wav"

    cmd = [
        ffmpeg, "-y", "-i", str(video_path),
        "-vn",                    # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",           # 16 kHz — Whisper requirement
        "-ac", "1",               # mono
        str(audio_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        log.warning("Audio extraction timed out")
        return None

    if result.returncode != 0:
        log.warning("Audio extraction failed: %s", (result.stderr or "")[-300:])
        return None
    if not audio_path.exists() or audio_path.stat().st_size < 1000:
        log.info("No usable audio track (video may be silent)")
        return None
    return audio_path


def transcribe(
    video_path: Path | str,
    session_dir: Path | str,
    *,
    model_size: str = "base",
) -> dict[str, Any] | None:
    """Transcribe the video's narration with faster-whisper (local, CPU).

    Saves ``session_dir/transcription.json`` on success.

    Returns:
        ``{"text": str, "segments": [{"start", "end", "text"}, ...],
        "language": str}`` or ``None`` if there is no audio, or
        faster-whisper is not installed.
    """
    audio_path = _extract_audio(video_path, session_dir)
    if audio_path is None:
        return None

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.warning(
            "faster-whisper not installed — narration will be skipped. "
            "Install Prometheus's voice extra or: pip install faster-whisper"
        )
        return None

    try:
        log.info("Transcribing narration with Whisper (%s)...", model_size)
        # device="cpu", compute_type="int8" — works anywhere without GPU
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments_gen, info = model.transcribe(str(audio_path), language="en", beam_size=5)

        segments: list[dict[str, Any]] = []
        full_text_parts: list[str] = []
        for seg in segments_gen:
            segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })
            full_text_parts.append(seg.text.strip())

        transcription = {
            "text": " ".join(full_text_parts),
            "segments": segments,
            "language": info.language,
        }
    except Exception as exc:  # noqa: BLE001 — narration is best-effort
        log.warning("Whisper transcription failed: %s", exc)
        return None

    out_path = Path(session_dir) / "transcription.json"
    try:
        out_path.write_text(json.dumps(transcription, indent=2), encoding="utf-8")
        log.info(
            "Transcription complete: %d segments, %d chars -> %s",
            len(segments), len(transcription["text"]), out_path,
        )
    except OSError:
        log.warning("Failed to write %s", out_path, exc_info=True)

    return transcription


# ── Narration alignment ──────────────────────────────────────────────


def align_narration(
    transcription: dict[str, Any] | None,
    keyframes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach narration text to each keyframe by timestamp overlap.

    Port of ``align_to_frames``: every transcript segment within a ±3s
    window of the keyframe's timestamp contributes to that keyframe's
    narration string.

    Returns:
        List of ``{frame_index, timestamp, narration}`` dicts (empty if
        there is no transcription).
    """
    if not transcription or not transcription.get("segments"):
        return []

    segments = transcription["segments"]
    result = []

    for kf in keyframes:
        # Prefer the explicit timestamp; fall back to index (1fps assumption)
        ts = kf.get("timestamp")
        if ts is None:
            ts = kf.get("estimated_timestamp")
        if ts is None:
            ts = float(kf.get("index", 0))

        narration_parts = [
            seg["text"] for seg in segments
            if seg["start"] - _NARRATION_WINDOW_SECONDS <= ts <= seg["end"] + _NARRATION_WINDOW_SECONDS
        ]

        result.append({
            "frame_index": kf.get("index", 0),
            "timestamp": ts,
            "narration": " ".join(narration_parts).strip(),
        })

    return result
