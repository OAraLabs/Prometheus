---
name: local-image-generation
description: Operating notes for the local ComfyUI image-generation backend (FLUX-schnell on the host RTX 3090 Ti). Covers when to use comfyui vs pollinations, model picking, VRAM co-tenancy with Ollama, troubleshooting the systemd unit, and adding new models.
version: 2.0.0
author: Prometheus
license: MIT
---
<!-- Provenance: Prometheus | skills/local-image-generation.md | MIT -->

# Local Image Generation — ComfyUI Backend

`image_generate` has two backends behind one tool:

- **pollinations** — free hosted, FLUX, no API key
- **comfyui** — local server, FLUX-schnell, sovereign

Both share the same tool input. The agent picks via `backend=auto|pollinations|comfyui` or relies on the config default in `image_generation.default_backend`.

## When to Use ComfyUI vs Pollinations

| Scenario | Pick |
|---|---|
| Default — agent doesn't specify, no special need | `auto` (uses comfyui when up, else pollinations) |
| Privacy matters (sensitive prompt, internal content) | `comfyui` |
| Need img2img / inpaint / ControlNet (future) | `comfyui` |
| Bulk batch generation | `comfyui` (no rate limit) |
| Quick one-off, ComfyUI not up | `pollinations` |
| Agent is iterating + needs <5s feedback | `comfyui` on warm GPU |
| Don't care about anything — just generate | `auto` |

## Install State on This Box

- ComfyUI: `~/comfyui/` (cloned from comfyanonymous/ComfyUI main)
- Venv: `~/comfyui/.venv/` (torch 2.11.0+cu128, CUDA 12.8 wheels)
- Model: `~/comfyui/models/checkpoints/flux1-schnell-fp8.safetensors` (~17 GB, all-in-one FLUX-schnell)
- Service: `~/.config/systemd/user/comfyui.service` (manage with `systemctl --user`)

## Operating Commands

```bash
# Status / lifecycle
systemctl --user status comfyui
systemctl --user start comfyui
systemctl --user stop comfyui
systemctl --user restart comfyui
systemctl --user enable comfyui       # auto-start at login
systemctl --user disable comfyui

# Logs (live tail)
journalctl --user -u comfyui -f

# Health check
curl -s http://127.0.0.1:8188/system_stats | jq
```

## VRAM Co-tenancy with Ollama

This box runs both:
- **Ollama** (your primary LLM, currently `qwen3:32b` Q4_K_M → ~24.5 GB VRAM)
- **ComfyUI** (FLUX-schnell-fp8 → ~12 GB VRAM)

Total = ~36 GB on a 24 GB GPU. **They cannot coexist.** One must yield.

Default Ollama config keeps the model resident for 24 hours (`OLLAMA_KEEP_ALIVE=24h`). To free VRAM for ComfyUI:

```bash
# One-shot eviction (Ollama reloads model on next request)
curl -X POST http://localhost:11434/api/generate \
  -d '{"model":"qwen3:32b","keep_alive":0}'

# Permanent (until ollama systemd unit edited)
# Edit ~/.config/systemd/user/ollama.service.d/override.conf:
#   [Service]
#   Environment="OLLAMA_KEEP_ALIVE=5m"
# Then:
systemctl --user daemon-reload && systemctl --user restart ollama
```

**Pattern**: when the agent needs image gen, evict Ollama first. Image gen is human-perceptible (~5-15s on warm FLUX), so the LLM cold-start (~30s when re-requested) is amortized across the user's next several text turns.

A future enhancement could automate this — `image_generate` calls eviction before the ComfyUI prompt, then optionally re-warms Ollama after. Not implemented yet; manual for now.

## Adding New Models

Drop the `.safetensors` into `~/comfyui/models/checkpoints/`, restart ComfyUI (which reloads the checkpoint list), then pass `model=<filename>` to `image_generate`.

Recommended additions:

| Model | Size | Use case | URL |
|---|---|---|---|
| `flux1-dev-fp8.safetensors` | ~12 GB | Higher quality than schnell, 20+ steps | `huggingface.co/Comfy-Org/flux1-dev` |
| `sdxl-1.0.safetensors` | ~7 GB | Smaller footprint fallback | `huggingface.co/stabilityai/stable-diffusion-xl-base-1.0` |
| `flux1-fill-dev-fp8.safetensors` | ~12 GB | Inpainting | `huggingface.co/Comfy-Org/flux1-fill-dev` |

For non-FLUX models, the workflow template in `image_generate.py` needs adjustment (different node graph). Defer until requested.

## Workflow Template Reference

`image_generate.py` embeds a minimal 7-node FLUX-schnell graph:

```
KSampler (steps=4, cfg=1.0, sampler=euler, scheduler=simple)
├── CheckpointLoaderSimple   (loads UNet+CLIP+VAE from .safetensors)
├── EmptyLatentImage          (width × height)
├── CLIPTextEncode positive   (the prompt)
├── CLIPTextEncode negative   (empty — schnell ignores at cfg=1.0)
└── output → VAEDecode → SaveImage  (filename_prefix=prometheus)
```

To use a non-default workflow:
1. Build it in ComfyUI's node editor (browser at http://127.0.0.1:8188)
2. Enable Dev Mode in the gear menu → Save (API Format)
3. Drop the JSON into `config/comfyui_workflows/<name>.json`
4. Reference it from `image_generate.py` (future extension — currently hardcoded template)

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `image_generate: network error at http://127.0.0.1:8188` | ComfyUI not running | `systemctl --user start comfyui` |
| `image_generate: HTTP 400 ... node ... is_changed` | Model file missing | `ls ~/comfyui/models/checkpoints/` — confirm `.safetensors` present |
| `timed out waiting for prompt ... after 180s` | First-load model warmup (12 GB into VRAM) | First call after server restart can take 30-60s. Retry. |
| `CUDA out of memory` in journalctl | Ollama still holding VRAM | Evict per "VRAM Co-tenancy" above |
| `image_generate: /view returned empty bytes` | Workflow ran but saved nothing | Check ComfyUI logs: `journalctl --user -u comfyui -e` |

## Auto-Fallback to Pollinations

`backend=auto` (default) probes ComfyUI's `/system_stats` with a 2s timeout. If it can't reach ComfyUI, falls back to Pollinations transparently. The agent doesn't need to handle service downtime.

To force one backend, pass `backend=comfyui` (errors loudly if unreachable) or `backend=pollinations` (skips probe entirely).

## Anti-Patterns

- ❌ Running ComfyUI on `--listen 0.0.0.0` — no auth, world-reachable. Keep `127.0.0.1`.
- ❌ Setting `backend=comfyui` when you don't care about local — adds a fail path. Use `auto`.
- ❌ Bulk-generating without checking ComfyUI status — if the server is mid-checkpoint-load, you'll see timeouts. Hit `/system_stats` once first.
- ❌ Editing the embedded workflow JSON in `image_generate.py` to add nodes — fragile. Move custom workflows to `config/comfyui_workflows/<name>.json` and reference by name (future extension).
- ❌ Ignoring VRAM co-tenancy — if both Ollama and ComfyUI try to load simultaneously, both fail.

## See Also

- `image-generation.md` — generic usage patterns, prompt structure, model picking
- `docker.md` — for containerizing ComfyUI if you want isolation (`stable-diffusion-webui-docker`)
- `anatomy` tool — verify GPU state and VRAM availability
- ComfyUI repo: https://github.com/comfyanonymous/ComfyUI
- FLUX models: https://huggingface.co/Comfy-Org
