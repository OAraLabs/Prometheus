"""Video-ingestion pipeline — screen recording to skill draft.

Orchestrates the vision leg of the record-a-skill funnel: (optional URL
download) -> ffmpeg frame extraction -> optional narration transcription
-> SSIM keyframe dedup -> registry gate -> VLM digest -> action
extraction/parameterization -> funnel mapping -> the SAME quality gate
and synthesizer the live recorder (DOM path) uses.

Trust policy: vision output is inference, not ground truth, so this
pipeline NEVER persists to ``skills/auto/``. The returned ``skill``
carries SKILL.md ``content`` for the drafts subsystem to hold for human
review (see ``live_recorder.service`` for the two-tier policy).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from prometheus.learning.live_recorder.quality_gate import gate_actions
from prometheus.learning.live_recorder.synthesizer import build_skill_content
from prometheus.learning.video_ingest.action_extractor import (
    extract_actions,
    parameterize_actions,
)
from prometheus.learning.video_ingest.funnel import (
    to_funnel_actions,
    to_funnel_parameters,
)
from prometheus.learning.video_ingest.ingest import (
    align_narration,
    download_video,
    extract_frames,
    transcribe,
)
from prometheus.learning.video_ingest.keyframes import extract_keyframes
from prometheus.learning.video_ingest.vision_digest import (
    digest_keyframes,
    validate_vision_model,
)

log = logging.getLogger(__name__)


def _title_from_filename(video_path: Path) -> str:
    """Infer a workflow title from the video filename."""
    base = video_path.stem.replace("_", " ").replace("-", " ").strip()
    return base.title() if base else "Recorded Workflow"


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


async def ingest_video_to_skill(
    source: str,
    *,
    vision_model_cfg: dict[str, Any],
    work_dir: Path | None = None,
    fps: float = 2.0,
    transcribe_audio: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Run the full video-ingestion pipeline on a recording.

    Args:
        source: Local video path or an http(s) URL (downloaded via yt-dlp).
        vision_model_cfg: A Prometheus ``model:``-shaped config block for
            the vision model (``{provider, model, base_url?, ...}``).
        work_dir: Session directory (created if missing); a temp
            directory is created when omitted.
        fps: Frame extraction rate.
        transcribe_audio: Transcribe narration with faster-whisper when
            available (best-effort — silently skipped if not installed).
        force: Run the digest even when the model registry says the
            configured model has no vision support.

    Returns:
        A JSON-serializable dict::

            {
              "status": "ok" | "rejected" | "error",
              "skill": {name, title, description, content,
                        step_count, parameter_count} | None,
              "quality_gate": {...} | None,
              "session_dir": str,
              "digest_stats": {...},
              "error": str,          # only when status == "error"
              "reason": str,         # only when status == "rejected"
            }

        The skill content is a draft — the caller decides what to do
        with it; nothing is persisted to skills/auto/ here.
    """
    session_dir = Path(work_dir) if work_dir is not None else Path(
        tempfile.mkdtemp(prefix="prometheus-video-ingest-")
    )
    session_dir.mkdir(parents=True, exist_ok=True)

    def _error(message: str) -> dict[str, Any]:
        return {
            "status": "error",
            "error": message,
            "skill": None,
            "quality_gate": None,
            "session_dir": str(session_dir),
            "digest_stats": {},
        }

    try:
        # (a) Resolve the source video
        if _is_url(source):
            video_path = download_video(source, session_dir)
        else:
            video_path = Path(source)
            if not video_path.is_file():
                return _error(f"video not found: {source}")

        # (b) Extract frames
        frame_count = extract_frames(video_path, session_dir, fps=fps)
        duration_seconds = int(round(frame_count / fps)) if fps > 0 else 0

        # (c) Optional narration transcription (best-effort)
        transcription = transcribe(video_path, session_dir) if transcribe_audio else None

        # (d) Keyframe extraction (SSIM dedup)
        keyframes = extract_keyframes(session_dir, session_dir, frame_fps=fps)
        if not keyframes:
            return _error("no keyframes could be extracted from the recording")
        narration = align_narration(transcription, keyframes) if transcription else []

        # (e) Registry gate — refuse to digest with a known text-only model
        supported, explanation = validate_vision_model(vision_model_cfg)
        if not supported:
            if not force:
                return _error(
                    f"vision model check failed: {explanation} "
                    "(pass force=True to override)"
                )
            log.warning("Registry says no vision support (%s) — proceeding, force=True", explanation)

        title = _title_from_filename(video_path)

        # (f) Vision digest (checkpointed; resumes on rerun with same work_dir)
        digests = await digest_keyframes(
            keyframes, narration, vision_model_cfg, session_dir,
            prompt_context=title,
        )

        # (g) Action extraction + parameterization
        actions = extract_actions(digests)
        actions, parameters = parameterize_actions(actions)

        # (h) Map into the shared funnel shape
        funnel_actions = to_funnel_actions(actions)
        funnel_parameters = to_funnel_parameters(parameters)

        digest_stats = {
            "frames_extracted": frame_count,
            "keyframes": len(keyframes),
            "digested_frames": len(digests),
            "actions_extracted": len(actions),
            "parameters": len(parameters),
            "narration_segments": len((transcription or {}).get("segments", [])),
        }

        # (i) Deterministic quality gate — authoritative
        gate = gate_actions(funnel_actions, funnel_parameters)
        if gate.overall == "fail":
            log.info(
                "Video ingestion: quality gate rejected %s (%d/%d checks failed)",
                video_path.name, gate.failed, gate.total,
            )
            return {
                "status": "rejected",
                "reason": "quality_gate",
                "skill": None,
                "quality_gate": gate.to_dict(),
                "session_dir": str(session_dir),
                "digest_stats": digest_stats,
            }

        # (j) Synthesize SKILL.md content (draft — never persisted here)
        metadata = {
            "start_url": source if _is_url(source) else "",
            "title": title,
            "app": "",
            "duration_seconds": duration_seconds,
            "source": "video_ingestion",
        }
        draft = build_skill_content(funnel_actions, funnel_parameters, metadata)

        log.info(
            "Video ingestion: synthesized draft '%s' (%d steps, %d parameters)",
            draft.name, draft.step_count, draft.parameter_count,
        )
        return {
            "status": "ok",
            "skill": {
                "name": draft.name,
                "title": draft.title,
                "description": draft.description,
                "content": draft.content,
                "step_count": draft.step_count,
                "parameter_count": draft.parameter_count,
            },
            "quality_gate": gate.to_dict(),
            "session_dir": str(session_dir),
            "digest_stats": digest_stats,
        }

    except Exception as exc:  # noqa: BLE001 — pipeline boundary: report, don't crash
        log.exception("Video ingestion failed for %s", source)
        return _error(str(exc))
