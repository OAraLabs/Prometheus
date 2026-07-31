"""Video ingestion — the vision leg of the record-a-skill funnel.

Turns a screen recording (local file or YouTube/URL download) into a
skill *draft*: frames are extracted with ffmpeg, near-duplicates are
filtered with SSIM, a vision-language model digests each keyframe
(optionally guided by narration transcribed with faster-whisper), and
the extracted actions are mapped into the same funnel the live recorder
(DOM path) uses — ``live_recorder.quality_gate.gate_actions`` then
``live_recorder.synthesizer.build_skill_content``.

Transplanted from the shelved SkillForge project's video pipeline
(``ingest_video.py``, ``core/capture.py``, ``core/audio_transcribe.py``,
``core/vision_digest.py``, ``core/action_extractor.py``), rewritten to
speak Prometheus's provider layer instead of SkillForge's bespoke vision
clients.

Trust policy: vision output is inference, not ground truth. This package
NEVER persists to ``skills/auto/`` — the pipeline returns SKILL.md
content for the drafts subsystem to hold for human review.

Entry point: :func:`prometheus.learning.video_ingest.pipeline.ingest_video_to_skill`.
"""

from prometheus.learning.video_ingest.pipeline import ingest_video_to_skill

__all__ = [
    "ingest_video_to_skill",
]
