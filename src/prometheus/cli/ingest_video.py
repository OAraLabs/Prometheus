"""`prometheus ingest-video` — turn a screen recording into a skill draft.

Runs the video-ingestion pipeline (learning/video_ingest/) on a local
recording or YouTube URL and files the result as a DRAFT in
~/.prometheus/skills/drafts/ for human review in Beacon — vision-derived
skills never enter skills/auto/ directly (two-tier trust).

The vision model comes from config `learning.video_ingest.vision_model`
unless overridden with --model/--base-url/--provider.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

log = logging.getLogger(__name__)


async def _run_ingest(args, config: dict) -> int:
    from prometheus.learning.skill_drafts import SkillDraftStore
    from prometheus.learning.video_ingest.pipeline import ingest_video_to_skill

    vi_cfg = (config.get("learning", {}) or {}).get("video_ingest", {}) or {}
    vision_model_cfg = dict(vi_cfg.get("vision_model") or {})
    if args.model:
        vision_model_cfg["model"] = args.model
    if args.base_url:
        vision_model_cfg["base_url"] = args.base_url
    if args.provider:
        vision_model_cfg["provider"] = args.provider
    if not vision_model_cfg.get("model"):
        print("error: no vision model configured — set "
              "learning.video_ingest.vision_model in prometheus.yaml "
              "or pass --model/--base-url")
        return 2

    print(f"Ingesting {args.source} with {vision_model_cfg.get('provider', 'llama_cpp')}"
          f"/{vision_model_cfg['model']} ...")

    outcome = await ingest_video_to_skill(
        args.source,
        vision_model_cfg=vision_model_cfg,
        work_dir=Path(args.work_dir).expanduser() if args.work_dir else None,
        fps=args.fps,
        transcribe_audio=not args.no_audio,
        force=args.force,
    )

    status = outcome.get("status")
    stats = outcome.get("digest_stats") or {}
    if stats:
        print(f"  frames={stats.get('frames_extracted')} "
              f"keyframes={stats.get('keyframes')} "
              f"actions={stats.get('actions_extracted')} "
              f"parameters={stats.get('parameters')}")

    if status == "error":
        print(f"error: {outcome.get('error')}")
        print(f"session dir (for resume/debugging): {outcome.get('session_dir')}")
        return 1
    if status == "rejected":
        gate = outcome.get("quality_gate") or {}
        print(f"rejected by quality gate ({gate.get('failed')}/{gate.get('total')} checks failed):")
        for check in gate.get("checks", []):
            if not check.get("passed"):
                print(f"  - {check.get('name')}: {check.get('detail')}")
        return 1

    skill = outcome["skill"]
    store = SkillDraftStore()
    sidecar = store.create(
        skill["content"],
        source="video_ingestion",
        provenance={
            "video_source": args.source,
            "session_dir": outcome.get("session_dir"),
            "vision_model": {k: v for k, v in vision_model_cfg.items() if k != "api_key"},
            "quality_gate": outcome.get("quality_gate"),
            "digest_stats": stats,
            "step_count": skill.get("step_count"),
            "parameter_count": skill.get("parameter_count"),
        },
    )

    print(f"\nDraft filed: {sidecar['draft_id']}  ({skill['title']}, "
          f"{skill['step_count']} steps, {skill['parameter_count']} parameters)")
    print("Review it in Beacon, or via:")
    print(f"  GET  /api/learning/skill-drafts/{sidecar['draft_id']}")
    print(f"  POST /api/learning/skill-drafts/{sidecar['draft_id']}/accept | /reject")
    return 0


def run_ingest_video_command(args, config: dict) -> int:
    """Entry point for `prometheus ingest-video`."""
    return asyncio.run(_run_ingest(args, config))
