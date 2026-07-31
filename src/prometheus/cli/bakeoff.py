"""`prometheus bakeoff-vlm` — score a vision model against the annotated corpus.

The new-VLM ritual: before a vision model is enabled for video ingestion,
run it over the annotated session corpus (screen-recording videos +
hand-written golden SKILL.md files) and score every generated skill with
the hallucination-penalized golden diff (learning/skill_scoring.py).

Corpus layout (the SkillForge annotated corpus, e.g.
oara-4090:~/projects/skillforge):

    <corpus>/
        <videos or sessions dir>/*.mp4      (any of: ., videos/, sessions/, test-videos/)
        ground_truth/<video_stem>.md        golden skills

Ground-truth lookup mirrors run_daily_validation.py: the video stem is
lowercased, spaces -> underscores, and matched against ground_truth/*.md
(exact stem first, then prefix match). Videos without a golden skill are
reported as skipped, never silently dropped.

Usage:
    prometheus bakeoff-vlm --corpus ~/projects/skillforge \
        --model gemma-4 --base-url http://localhost:8080 [--provider llama_cpp]
        [--limit N] [--fps 2.0] [--force] [--output report.json]
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
_VIDEO_SUBDIRS = (".", "videos", "sessions", "test-videos")


def _find_videos(corpus: Path) -> list[Path]:
    videos: list[Path] = []
    for sub in _VIDEO_SUBDIRS:
        d = corpus / sub if sub != "." else corpus
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in _VIDEO_EXTS:
                videos.append(f)
        # Session dirs one level down (sessions/<name>/<name>.mp4)
        if sub == "sessions":
            for session_dir in sorted(p for p in d.iterdir() if p.is_dir()):
                for f in sorted(session_dir.iterdir()):
                    if f.suffix.lower() in _VIDEO_EXTS:
                        videos.append(f)
    return videos


def _golden_for(video: Path, ground_truth_dir: Path) -> Path | None:
    stem = video.stem.lower().replace(" ", "_")
    exact = ground_truth_dir / f"{stem}.md"
    if exact.exists():
        return exact
    # Prefix match, longest golden stem first (run_daily_validation
    # normalized stems by dropping suffixes like _smooth/_tracker)
    candidates = sorted(
        (p for p in ground_truth_dir.glob("*.md")
         if stem.startswith(p.stem) or p.stem.startswith(stem)),
        key=lambda p: -len(p.stem),
    )
    return candidates[0] if candidates else None


async def _run_bakeoff(args) -> int:
    from prometheus.learning.skill_scoring import score_skill
    from prometheus.learning.video_ingest.pipeline import ingest_video_to_skill

    corpus = Path(args.corpus).expanduser()
    ground_truth_dir = corpus / "ground_truth"
    if not ground_truth_dir.is_dir():
        print(f"error: no ground_truth/ directory under {corpus}")
        return 2

    videos = _find_videos(corpus)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print(f"error: no videos found under {corpus} (looked in {', '.join(_VIDEO_SUBDIRS)})")
        return 2

    vision_model_cfg = {
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url,
    }

    print(f"VLM bakeoff: {args.provider}/{args.model} over {len(videos)} video(s)\n")

    results = []
    scored: list[float] = []
    for video in videos:
        golden = _golden_for(video, ground_truth_dir)
        if golden is None:
            print(f"  SKIP  {video.name}  (no golden skill in ground_truth/)")
            results.append({"video": str(video), "status": "skipped",
                            "reason": "no ground truth"})
            continue

        started = time.time()
        try:
            outcome = await ingest_video_to_skill(
                str(video),
                vision_model_cfg=vision_model_cfg,
                fps=args.fps,
                transcribe_audio=not args.no_audio,
                force=args.force,
            )
        except Exception as exc:  # noqa: BLE001 — one bad video must not kill the run
            log.warning("bakeoff: pipeline failed on %s", video, exc_info=True)
            print(f"  FAIL  {video.name}  pipeline error: {exc}")
            results.append({"video": str(video), "status": "error", "error": str(exc)})
            continue
        duration = time.time() - started

        skill = outcome.get("skill") or {}
        content = skill.get("content", "")
        if not content:
            print(f"  FAIL  {video.name}  no skill generated "
                  f"(status={outcome.get('status')})")
            results.append({"video": str(video), "status": "no_skill",
                            "pipeline": {k: v for k, v in outcome.items() if k != "skill"}})
            continue

        score = score_skill(content, golden.read_text(encoding="utf-8"))
        scored.append(score.accuracy)
        print(
            f"  {score.accuracy:6.1%}  {video.name}  "
            f"(matched {score.matched_steps}/{score.expected_steps}, "
            f"hallucinated {score.hallucinated_steps}, missing {score.missing_steps}, "
            f"gate {outcome.get('quality_gate', {}).get('overall', '?')}, "
            f"{duration:.0f}s)"
        )
        results.append({
            "video": str(video),
            "golden": str(golden),
            "status": "scored",
            "score": score.to_dict(),
            "quality_gate": outcome.get("quality_gate"),
            "duration_seconds": round(duration, 1),
        })

    print()
    if scored:
        mean = sum(scored) / len(scored)
        print(f"Mean ground-truth accuracy: {mean:.1%} over {len(scored)} scored video(s)")
        verdict = "PASS" if mean >= args.threshold else "BELOW THRESHOLD"
        print(f"Threshold {args.threshold:.0%}: {verdict}")
    else:
        mean = 0.0
        print("No videos were scored.")

    if args.output:
        report = {
            "model": vision_model_cfg,
            "corpus": str(corpus),
            "video_count": len(videos),
            "scored_count": len(scored),
            "mean_accuracy": round(mean, 4),
            "threshold": args.threshold,
            "results": results,
        }
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {out}")

    return 0 if scored and mean >= args.threshold else 1


def run_bakeoff_command(args) -> int:
    """Entry point for `prometheus bakeoff-vlm`."""
    return asyncio.run(_run_bakeoff(args))
