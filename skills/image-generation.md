---
name: image-generation
description: Generate images from text prompts via the `image_generate` tool (Pollinations.ai backend, FLUX-based, no API key). Covers when to generate vs source elsewhere, prompt structure, model picking, dimensions/seed control, and Telegram delivery patterns.
version: 1.0.0
author: Prometheus
license: MIT
---
<!-- Provenance: Prometheus | skills/image-generation.md | MIT -->

# Image Generation

The `image_generate` tool synthesizes images from text prompts using Pollinations.ai's free FLUX-backed endpoint. No API key, no subscription, donation-funded infrastructure. Output saves to `~/.prometheus/cache/images/` (or a custom path) and the tool returns the file path.

## When to Use

✅ **USE `image_generate` when:**

- User asks for "an image of X", "make me a picture", "generate", "illustrate"
- Concept art, mockups, diagrams, illustrations, thumbnails
- The user has no specific image source in mind
- A web search for the image would return copyrighted material

## When NOT to Use

❌ **DON'T use when:**

- The user wants a *specific* existing image — use `web_fetch` to download it
- The user wants a photo of a real person/place/event — generative AI fabricates faces
- The task is *editing* an existing image — Pollinations only generates from text. No img2img here.
- The deliverable is a chart from data — use the data viz skills, not gen AI
- The user wants a diagram of code architecture — use `excalidraw` skill (real shapes)

## Input Schema

```yaml
prompt:        text description (required, 1-2000 chars)
model:         "flux" (default, best) | "turbo" (faster) | "gptimage"
width:         pixels (default 1024, range 256-2048)
height:        pixels (default 1024, range 256-2048)
seed:          int (optional — reproducible output when set)
enhance:       bool (default false — LLM-expand short prompts)
output_path:   absolute path (optional — overrides cache dir)
```

## Prompt Structure

FLUX responds best to **specific, layered** prompts. Build them as:

```
<subject> + <style> + <composition> + <lighting> + <quality modifiers>
```

Good:
> "A red apple on a wooden table, photorealistic, shallow depth of field, soft afternoon window light, 50mm lens, high detail"

Bad:
> "apple"  ← too sparse; output will be generic

For non-photo styles, lead with the style:
> "Watercolor illustration of a mountain village at sunset, loose brushwork, warm palette, paper texture"

## Model Picking Heuristic

| Need | Model | Why |
|---|---|---|
| Default — anything visual | `flux` | Best quality, slowest (~30-60s) |
| Quick iteration / draft | `turbo` | Faster (~10s), lower fidelity |
| Text-heavy in image (signs, UI) | `gptimage` | Better text rendering than FLUX |

If unsure, start with `flux`. The `enhance=true` flag is useful for terse user prompts — Pollinations LLM-expands before generation.

## Dimensions

- **Square** (1024x1024): default, good for most uses
- **Landscape** (1536x1024 or 1920x1080): banners, hero images, wide compositions
- **Portrait** (1024x1536, 768x1024): character art, vertical posters
- **Avoid** non-standard ratios under 512 on either side — quality drops sharply

Larger = slower. 1024² is the sweet spot.

## Seeds

Omit `seed` for variety. Set `seed` (any int) to:
- Reproduce a previous output exactly
- A/B test prompt variations against the same composition
- Generate a series with consistent style

## Delivery Patterns

### Bare path return (default)
The tool returns the file path. Caller decides what to do with it.

### Telegram delivery
When the user invoked via Telegram, follow up with the file:

```python
# After image_generate returns "Saved image to /home/.../img_abc.jpg"
# The gateway can call bot.send_photo(chat_id, open(path, 'rb'))
```

The agent should *announce* the generation ("Generated 1024x1024 image, saving...") so the user knows something happened — the file message arrives separately.

### Inline in a chat reply
Tell the user the path. They can open it directly if running locally.

## Anti-Patterns

- ❌ Generating 10 variants when 1 would do — each call hits the public endpoint
- ❌ Width or height >1536 unless genuinely needed — slow and not always better
- ❌ Prompts with trademarked characters/franchises — Pollinations may serve degraded output for IP-flagged prompts
- ❌ Real-person likeness ("a photo of Elon Musk drinking coffee") — model fabricates faces. Use real photos instead.
- ❌ Calling without telling the user it's happening — image gen takes 30+ seconds; without a status message the chat looks frozen

## Failure Modes

| Symptom | Cause | Recovery |
|---|---|---|
| Timeout after 120s | Service overloaded | Retry with smaller dims or `turbo` model |
| HTTP 4xx | Bad params, blocked prompt | Soften the prompt, drop trigger words |
| HTTP 5xx | Pollinations infra | Wait + retry; or fall back to a different model |
| Got HTML instead of image | Content filter rejection | Rephrase prompt without ambiguous trigger words |

The tool catches all of these and returns an error `ToolResult` with diagnostic text — the agent can read it and either retry or tell the user.

## Privacy

Pollinations is a public endpoint. **Do not include sensitive personal information or proprietary content** in prompts — they may be cached/logged by the service. For sensitive image work, wait for the local-ComfyUI backend (step 4b in the roadmap, gated on the GPU driver fix).

## See Also

- `popular-web-designs.md` — for replicating UI/web aesthetics
- `excalidraw.md` — for diagrams with real geometric primitives
- `canvas-design.md` — for richer composed artwork
- `verify.md` — confirm the image actually got saved and is non-trivial size
