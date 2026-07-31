# Record a Skill

Demonstrate a workflow once; Prometheus writes the skill. Two producers
feed one funnel, with different trust levels:

| Producer | Signal | Model needed | Trust |
|---|---|---|---|
| **Browser recording** (Live DOM path) | DOM events with full element context — deterministic ground truth | none | auto-persists to `skills/auto/` after the quality gate |
| **Video ingestion** (screen recording / YouTube) | frames + narration interpreted by a vision model | a VLM from `config/model_registry.yaml` | always lands as a **draft** requiring human review in Beacon |

Both producers converge on the same funnel: structured workflow actions →
deterministic quality gate → SKILL.md synthesis → persistence through
`SkillCreator.persist_skill_content()` (frontmatter validation,
no-overwrite, `skill_created` signal). Beacon's Activity feed and
Config→Skills tab pick new skills up with no extra wiring, and the live
`SkillRegistry` is reloaded so the running agent can use them immediately.

## The Live DOM path (phase 1)

1. Install the recorder extension (SkillForge Live, `chrome://extensions`
   → Load unpacked). Its upload endpoint defaults to the local daemon:
   `http://127.0.0.1:8005/api/learning/live-upload`. If your daemon has a
   web API token, paste it in the extension's settings panel (gear icon).
2. Record: Start Recording → do the workflow → Stop → Send to Prometheus.
3. The daemon runs the deterministic pipeline
   (`prometheus/learning/live_recorder/`): noise filtering, input
   merging, click+type grouping, event→action mapping, parameter
   detection (typed values become parameters; passwords are masked at
   capture time and never leave the browser unmasked).
4. The **quality gate** (five structural checks: app consistency, action
   distribution, consecutive duplicates, 3–50 viable steps, parameter
   sanity) rejects broken recordings with a 422 the extension surfaces.
5. Optional **step verification** (`learning.live_recorder.verify_steps`,
   default on): the configured local model reviews the extracted steps
   for gaps/duplicates/misordering via the provider layer. Advisory —
   it blocks persistence only on `poor` quality with critical issues,
   and degrades to a no-op when no model is reachable.

Nothing leaves the machine: the trace goes browser → your daemon, and the
DOM path needs no model at all.

Every upload is archived under `~/.prometheus/data/recordings/<id>/`
(events, screenshots, extracted actions, gate + verification results)
for provenance and future training-data use.

## Video ingestion (phase 2)

```bash
# local recording or YouTube URL
prometheus daemon &  # or call the pipeline from your own code
python -c "..."      # see prometheus/learning/video_ingest/pipeline.py
```

The pipeline: download (yt-dlp) → frame extraction (ffmpeg, 2fps,
1280×720) → optional narration transcript (faster-whisper, the `voice`
extra) → SSIM keyframe selection → per-keyframe vision digest → action
extraction (tooltip filtering, target normalization) → the same funnel
as the DOM path.

**The VLM is a config dial, not a code path.** All model calls go through
the provider layer; the model comes from `learning.video_ingest.
vision_model` (same shape as the top-level `model:` block) and must be a
`config/model_registry.yaml` entry whose `capabilities.vision.supported`
is true (e.g. `gemma-4` with an mmproj). The pipeline refuses to run a
text-only model unless forced. As open VLMs improve, quality improves by
editing config — not code.

Vision output is *interpretation, not ground truth* (SkillForge's real
corpus accuracy was 56%, dominated by hallucinated steps), so
vision-derived skills **never** auto-persist. They land in
`~/.prometheus/skills/drafts/` and appear in Beacon for redline review:

- `GET  /api/learning/skill-drafts` — list pending drafts
- `GET  /api/learning/skill-drafts/{id}` — content + provenance
- `POST /api/learning/skill-drafts/{id}/accept` — persist through the
  standard auto-skill path (optionally with edited content)
- `POST /api/learning/skill-drafts/{id}/reject` — archived to
  `drafts/.rejected/`, never deleted

## The new-VLM ritual: bakeoff before enablement

Before pointing video ingestion at a new vision model, score it against
the annotated session corpus (videos + hand-written golden SKILL.md
files; the SkillForge corpus lives at `oara-4090:~/projects/skillforge`):

```bash
prometheus bakeoff-vlm \
  --corpus ~/projects/skillforge \
  --model gemma-4 --base-url http://localhost:8080 \
  --output ~/bakeoffs/gemma-4.json
```

Each video is run through the full pipeline and its generated skill is
diffed against the golden skill with **hallucination-penalized accuracy**
(`matched / (expected + hallucinated)`) — inventing steps can never buy
score. The command exits nonzero below `--threshold` (default 75%), so
it can gate a config change in CI or a script.

## Config reference

```yaml
learning:
  live_recorder:
    enabled: true        # POST /api/learning/live-upload
    verify_steps: true   # model second-pass review (advisory)
  video_ingest:
    enabled: false       # phase 2 producer; needs a vision-capable model
    vision_model:        # same shape as the top-level `model:` block
      provider: llama_cpp
      base_url: http://localhost:8080
      model: gemma-4
```
