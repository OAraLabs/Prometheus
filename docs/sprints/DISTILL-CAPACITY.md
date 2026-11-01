# Distillation capacity: teacher context and 9B QLoRA sequence length

Operator, Sun 2026-11-01 ~01:30 ET: measure the 27B teacher's max stable context on the 4090 (UD-Q4_K_XL and IQ3_XXS, q8_0 KV cache, flash attention) and the 3090 Ti's current limit; use the highest stable context for long-context teacher rows and cap them at what the teacher can read; measure the longest sequence the 9B QLoRA trains at on the 4090 with activation offload to RAM (16K/32K/64K), with tokens/sec.

Code: `scripts/distill/capacity.py` (probes, ladder, block, outputs) and `scripts/distill/tower/seqlen_bench.py` (the QLoRA benchmark).

## Method

- **Needle probe at context N:** one prompt of N - (512 + 1024) tokens of real repo docs, code and wiki text (the eval-longctx builder with its own seed; no eval case), with three invented notes at different depths (A: 10/50/90 %, as read_file results; B: 25/60/85 %, pasted files). Correct = the reply holds all three values. Greedy, thinking off.
- **Each rung:** A and B (two full prefills; prefill tok/s from them), then a 128-token decode benchmark on B's cached prompt (decode tok/s).
- **Stable:** the server loads, both requests answer all three needles, nothing crashes, and VRAM headroom (nvidia-smi free, minimum over the rung) stays >= 300 MiB. The ladder stops at the first unstable rung.
- **4090 server:** the reviewed teacher server (tailnet bind, key file, tools/agent/MCP/web UI off) with only `--parallel 1 -c N` changed; it already runs `-fa on -ctk q8_0 -ctv q8_0` and no mmproj.

## 4090 teacher (dedicated GPU)

Pending: the measurement block is armed on the 3090 Ti host and starts when the post-train pipeline finishes (Sun afternoon EST); this file is re-rendered with its results.

## 3090 Ti teacher (shared GPU, as configured)

Current: IQ3_XXS, 2 slots sharing one unified KV of 56K, beside the voice stack and Ollama. Probes went through the generation runner's probe guard, one at a time, outside the nightly evals.

| Context | Prompt tokens | Prefill tok/s | Decode tok/s | Needles A / B | Min free VRAM MiB | Verdict |
|---|---:|---:|---:|---|---:|---|
| 16K | 14,872 | 1,345 | 43.1 | 3/3 / 3/3 | 471 | stable |
| 32K | 31,267 | 1,264 | 45.0 | 3/3 / 3/3 | 459 | stable |
| 48K | 47,645 | 1,163 | 41.1 | 3/3 / 3/3 | 459 | stable |
| 56K | 55,799 | 1,124 | 40.3 | 3/3 / 3/3 | 459 | stable |

- **Measured:** reads up to **56K** (the configured context (57,344) is the top).
- **Computed (VRAM):** min free 459 MiB over the measurement; with the 300 MiB floor and 64 MiB for buffer growth at 34,816 bytes/token, the most this config could hold is **58K**; with Ollama unloaded (nightly windows, Q9) 217K.
- **Restart with more context:** not advised (2K more tokens fit beside the residents with the 300 MiB floor; under the 16K a restart would need to be worth it, and the voice stack and image jobs allocate on demand).

## Recommendation

- The 3090 Ti teacher (as configured) reads up to 56K: rows up to 52K can be generated there.
