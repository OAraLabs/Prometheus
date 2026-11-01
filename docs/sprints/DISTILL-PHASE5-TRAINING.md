# Phase 5: first full 9B QLoRA run (2026-10-31 → 11-01)

**Status: DONE.** The run completed on the 4090: 2 epochs, 1,120 optimizer steps, 0 non-finite steps. The adapter is private and tagged max-derived; publishing it is refused under Q6b.

## Setup

| | |
|---|---|
| Student | Qwen3.5-9B. A 4-bit copy is used, verified byte-identical to bitsandbytes NF4 of the official weights at commit c2022362. |
| Data | v4b: 4,480 train / 140 val, coding cap 70%, after the local-27B judge pass (see DISTILL-PHASE3-DATA.md) |
| Settings | QLoRA, 2 epochs, sequence length 16,384, batch 1 × grad-accum 8, seed 3407 |
| Data order | Shuffled every epoch (no length grouping), as checked by the shuffle test |
| Activations | Activation offload to CPU RAM (Unsloth checkpointing) |
| Throughput | ~2,540 tok/s, peak 12.7 GiB; 124.1M tokens seen |
| Wall time | 19:54 EDT Sat → 09:23 EST Sun, ~14.5 h |
| Held-out sets | 64-row val subset and a fixed 32-row probe batch (seed 3407), both disjoint from train |

## Held-out curve

| Step | Point | Val loss | Probe loss |
|---:|---|---:|---:|
| 0 | baseline | 1.0000 | 1.1303 |
| 83 | | 0.8722 | 0.9926 |
| 200 | | 0.8351 | 0.9463 |
| 400 | | 0.8137 | 0.9176 |
| 560 | end of epoch 1 | 0.7972 | 0.8925 |
| 600 | | 0.8049 | 0.8954 |
| 682 | | 0.8064 | 0.8895 |
| 800 | | 0.8012 | 0.8892 |
| 1000 | | 0.7966 | 0.8785 |
| 1120 | end of epoch 2 | **0.7960** | **0.8776** |

- **Overall:** val loss fell 20.4% from baseline and the probe batch 22.4%.
- **Epoch 2:** val rose briefly (0.797 → 0.806) while train loss kept falling, then recovered to a new low by the end. The probe batch improved almost the whole way through.
- **Fallback not needed:** the epoch-1 checkpoint was kept separately as a fallback (`9b-r32-v4b-max-derived-keep-epoch1/`). The final adapter beats it on both held-out sets, so the final adapter goes forward.

## Artifacts (4090)

- **Adapter:** `~/distill/phase5/runs/9b-r32-v4b-max-derived/adapter-max-derived` (307 MB). `adapter_model.safetensors` sha256 is `d2fba6bf286e1996f2d496ffcafd34d53de755aaf4c43dfeed5473c533d81ef3`.
- **Run manifest:** status done, max_derived true, private, publish refused (Q6b).
- **Checkpoints:** 1084, 1120, and the preserved epoch-1 checkpoint (560).

## Next

The post-training pipeline started automatically when the run ended. It merges and quantizes to Q4_K_M and Q5_K_M (keeping BF16), then scores:

- Q4 and Q5 on eval-tools and sanity;
- BF16 on the fixed 100-case eval-tools subset;
- Q4 on injection and long context;
- the official base 9B quantized with the same recipe, for a fair comparison.

It then compares against base-9B and Saluki. Results go in DISTILL-PHASE7-SCORES.md.
