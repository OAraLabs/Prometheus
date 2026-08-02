# Dict-Wrap LoRA — Phase 0 feasibility + provisioning: RESOLVED (feasible)

**Date:** 2026-06-15 · **Spec:** `docs/sprints/SPRINT-DICTWRAP-LORA.md` (PR #42).
The spec's Phase-0 HALT is resolved: every gate passed and the safe provisioning is underway. The
remaining phases (harvest, train, eval) are GPU-bound and need a **daemon-down window** — that is
the next gate, and it is the owner's scheduled outage decision (stopping the daemon takes AutoDream
/ Telegram / Beacon offline).

## Gates (all on the 4090, `oara-4090@<gpu>`)

| gate | result |
|---|---|
| disk | **202 GB free** (need ~100) ✓ |
| HF weights exist + reachable | **`google/gemma-4-26b-a4b-it`, gated=False, ~52 GB safetensors** ✓ |
| HF auth | cached token valid — **whoami = OAraLabs** ✓ |
| training frameworks | installed in `~/lora-train-venv` (`--system-site-packages`, reuses torch 2.11+cu130): **peft 0.19.1, trl 1.6.0, datasets 5.0.0, accelerate 1.13.0** ✓ |
| **bitsandbytes / CUDA 13** (QLoRA make-or-break) | **bitsandbytes 0.49.2, `Params4bit` works on cu130** ✓ — 4-bit QLoRA viable on one 4090 |
| serving path | llama.cpp `--lora` / `--lora-scaled` / `POST /lora-adapters` hot-swap ✓ |
| GGUF conversion | `~/llama.cpp/convert_lora_to_gguf.py` present ✓ |
| weights download | DETACHED + running → `~/models/gemma-4-26b-a4b-it-hf/` (9.7/52 GB at checkpoint); sentinel `~/lora-train/runs/weights-dl.SENTINEL` |

## Architecture (pins the LoRA target modules)

`config.json` → `text_config`: **`gemma4_text`, 30 layers, hidden 2816, 16 heads / 8 KV, 128
experts, intermediate 2112.** Linear modules in the index: `q_proj k_proj v_proj o_proj` (attention,
shared) + `gate_proj up_proj down_proj` under `experts` (per-expert) + router `gate`.

**LoRA target = attention projections `q_proj,k_proj,v_proj,o_proj` (shared, MoE-safe)** — exactly
the spec's plan. The per-expert MLPs (128×) are the deferred risky path.

## The one open risk (validate at train time, Phase 2)

`gemma4` is a **brand-new arch** (transformers 5.5, `model_type=gemma4_text`,
`Gemma4ForConditionalGeneration` multimodal wrapper). peft 0.19 target-module auto-detection and
trl 1.6 DPO support for this specific arch are unvalidated until the first train step runs — a known
unknown, surfaced now, cheap to probe with a 1-step dry run before the full train.

## Next gate (owner): the daemon-down GPU window

Harvest (Phase 1, ~9 h sole-GPU) + train (Phase 2, daemon STOPPED — the 24 GB is fully used by the
live server, so training cannot coexist with serving) + eval (Phase 3) all need the GPU. Provisioning
(download + install) needs no outage and is done/underway; everything from the harvest onward needs
the scheduled window. Held-out eval-shape design (the Goodhart guard) is done coherently as the first
step inside that window. **HALT here for the window.**
