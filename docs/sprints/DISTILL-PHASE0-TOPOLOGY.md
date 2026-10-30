# Phase 0: topology check (2026-10-30, read-only)

**Gate: GPU work BLOCKED.** Both reachable GPUs are fully occupied by live services that, under the runbook's
own rules, can't be swapped without the operator. Running the no-GPU track (Phases 1–3 and scorer code).
The `save business` → `apply flywheel` → `apply business` rehearsal was **not** run.

## GPUs

| Host | GPU | VRAM used / total | What's on it | Health |
|---|---|---|---|---|
| 4090 tower | RTX 4090 | 20,306 / 24,564 MiB | `llama-server.service`: Qwen3.8-27B UD-Q4_K_XL, 32k ctx, 1 slot | running |
| 3090 Ti host | RTX 3090 Ti | 22,372 / 24,564 MiB | `llama-mini.service` Saluki 27B IQ2 (11.5 GiB, daemon primary) · Ollama qwen2.5:7b (5.4 GiB) · TTS (3.0 GiB) · whisper STT (2.1 GiB) · ComfyUI (0.3 GiB idle) | Saluki `/health` ok; daemon active |
| 4060 Ti box | — | — | not reachable (no SSH yet) | skipped |

Saluki is serving on the 3090 Ti and the Prometheus daemon is active.

## Why GPU work is blocked

1. **The 4090's only GPU service is customer-facing with no fallback.** A Cloudflare tunnel for a business
   service on that host points straight at its `llama-server`. The runbook says a customer-facing GPU service
   with no fallback stays running. The fallback it names (the 4060 Ti) is unreachable. With ~4 GiB free there's
   no room for training or a second model beside it.
2. **The 4090's nights are already booked** by another approved overnight job that routes to that same
   `llama-server` (its proxy has been up since 2026-10-25).
3. **The 3090 Ti backs the live daemon.** Saluki is the daemon's primary model (one-week test, judged
   ~2026-11-02), and the runbook's `flywheel` assumes the daemon runs on the Max API instead. Switching it
   means editing the daemon's model config and config pins on the live host: touching live serving.
4. **Disk:** the 4090 has 75 GB free on `/` (runbook asks for ~100 GB). `~/models` holds 186 GB, some of it old
   GGUFs, but deleting anything is out of scope.

All four are in `DISTILL-OPEN-QUESTIONS.md`.

## Other checks

- **HF downloads:** work from the 4090 (206 on a ranged fetch, 0.1 s).
- **Hosts:** 4090 tower 32 CPU / 62 GiB RAM (50 GiB available). 3090 Ti host 16 CPU / 25 GiB RAM, 245 GB disk free.
- **Prior art on the 4090:** `~/lora-train` (an earlier DPO LoRA run with a venv) can be reused for the
  training environment.
- **Setups mechanism:** none exists in the repo or on either host. Phase 0's no-GPU work builds a minimal one
  (`save` / `apply` / `status`) and captures `business` read-only. `apply` is not run.

## Student check

Searched the Qwen org on Hugging Face for an open Qwen3.6–3.8 model of 8–10B, Apache 2.0, ≥256k native context:
**none.** Qwen3.6 ships only 27B and 35B-A3B; Qwen3.8 ships 27B, a ~180B MoE (`Flash-Next`, non-Apache license)
and a 2.4T MoE. The student stays **Qwen3.5-9B** (Apache 2.0, 9.65B params, 262,144 native context).
