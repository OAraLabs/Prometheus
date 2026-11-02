# Phase 7 scorecard: first full 9B student (9b-r32-v4b-max-derived)

**Verdict: NOT PROMOTED.**

- **Fails:** the student misses the tool-call bar (base + 10 pts) against both base recipes at Q4_K_M.
- **Passes:** long-context, injection and sanity pass every comparison.
- **Q5_K_M:** clears the tool bar against the Unsloth base, but not against the like-for-like base. Its long-context and injection sets weren't run.

The adapter is private and max-derived (Q6b): never published.

All numbers come from `score.py` v4 with the same `llama-server` binary and flags for every model (`-c 262144 -np 1 -fa on -ctk/ctv q8_0 --jinja`), concurrency 1. Brackets are 95% Wilson intervals. Paired deltas use Newcombe intervals.

## The models

| Column | What it is |
|---|---|
| Student Q4 / Q5 | the trained adapter merged into official Qwen3.5-9B, our imatrix (train rows only), Q4_K_M / Q5_K_M |
| Student BF16 | the same, unquantized; run on the fixed 100-case eval-tools subset only, as a reference |
| Base (Unsloth) | Unsloth's own Qwen3.5-9B Q4_K_M, the Phase 4 baseline |
| Base (same recipe) | official Qwen3.5-9B quantized with the student's exact recipe (same imatrix text, Q4_K_M). This is the like-for-like base. |
| Saluki | the live test-drive model (27B IQ2 mix); long context measured at 32K only |
| 27B | Qwen3.8-27B Q4_K_M baseline (Phase 4). Its 256K bucket failed to run (0/45), so its pooled long-context number is not comparable. |

## Scores

| Measure | Student Q4 | Student Q5 | Student BF16 | Base (Unsloth) | Base (same recipe) | Saluki | 27B |
|---|---|---|---|---|---|---|---|
| eval-tools task success (329) | **80.5%** [75.9, 84.5] | **83.6%** [79.2, 87.2] | – | 72.3% [67.3, 76.9] | 75.1% [70.1, 79.4] | 70.5% [65.4, 75.2] | 72.9% [67.9, 77.5] |
| eval-tools task success, 100-case subset | 83.0% | 82.0% | 81.0% | 77.0% | 75.0% | – | – |
| tool name accuracy | 81.2% [76.6, 85.0] | 83.6% [79.2, 87.2] | – | 72.3% [67.3, 76.9] | 75.1% [70.1, 79.4] | 70.5% [65.4, 75.2] | 72.9% [67.9, 77.5] |
| arg validity | 99.3% [97.3, 99.8] | 100.0% [98.6, 100.0] | – | 100.0% [98.4, 100.0] | 100.0% [98.5, 100.0] | 100.0% [98.4, 100.0] | 100.0% [98.4, 100.0] |
| **exact arguments** | 9.0% (24/267) [6.1, 13.0] | 8.7% (24/275) [5.9, 12.7] | 4.9% (4/81) [1.9, 12.0] (subset) | 10.1% (24/238) [6.9, 14.6] | 9.7% (24/247) [6.6, 14.1] | 9.5% [6.3, 13.9] | 11.2% (27/240) [7.8, 15.9] |
| sanity (211) | 95.7% [92.1, 97.7] | 95.7% [92.1, 97.7] | – | 96.7% [93.3, 98.4] | 94.3% [90.3, 96.7] | 94.8% [90.9, 97.1] | 97.2% [93.9, 98.7] |
| injection resistance (192) | 99.5% [97.1, 99.9] | – | – | 99.0% [96.3, 99.7] | 99.5% [97.1, 99.9] | 98.4% [95.5, 99.5] | 99.0% [96.3, 99.7] |
| long-ctx recall @32K (45) | 82.2% | – | – | 82.2% | 84.4% | 82.2% | 86.7% |
| long-ctx recall @64K | 82.2% | – | – | 80.0% | 80.0% | – | 84.4% |
| long-ctx recall @128K | 71.1% | – | – | 71.1% | 73.3% | – | 82.2% |
| long-ctx recall @256K | 77.8% | – | – | 77.8% | 82.2% | – | failed to run |
| long-ctx recall, pooled (180) | 78.3% [71.8, 83.7] | – | – | 77.8% [71.2, 83.2] | 80.0% [73.6, 85.2] | (32K only) | (256K missing) |

Exact arguments counts only calls whose tool name was right, so its denominator differs per model. Every model, base and student alike, gets an exact argument match about 1 time in 10. Training did not move this.

## Phase 6 promotion gate

Bars:
- tool-call task success ≥ base + 10 pts;
- long-context recall must not be significantly worse at 95% (Holm across buckets, plus a pooled test) by more than 3 pts;
- injection resistance ≥ base;
- sanity no worse than base by more than 3 pts.

| Comparison | Tool success (Δ, paired CI) | Long-ctx CI gate | Injection | Sanity | Verdict |
|---|---|---|---|---|---|
| Q4 vs base (Unsloth) | +8.2 [+3.1, +13.3]: **FAIL** | PASS (all 5 tests underpowered) | 99.5 ≥ 99.0: PASS | −0.9 [−4.7, +2.7]: PASS | **FAIL** |
| Q4 vs base (same recipe) | +5.5 [+0.5, +10.5]: **FAIL** | PASS (underpowered; @256K −4.4 [−16.5, +7.4]) | 99.5 ≥ 99.5: PASS | +1.4 [−2.3, +5.3]: PASS | **FAIL** |
| Q5 vs base (Unsloth) | +11.2 [+6.3, +16.2]: PASS | not run | not run | −0.9 [−4.3, +2.2]: PASS | **INCONCLUSIVE** |
| Q5 vs base (same recipe) | +8.5 [+3.7, +13.3]: **FAIL** | not run | not run | +1.4 [−1.8, +4.9]: PASS | **FAIL** |

- **Tool success:** the student is clearly better at tool calls than either base. Every paired interval excludes 0. It is not +10 better on the like-for-like base (+5.5 at Q4, +8.5 at Q5).
- **Recipe effect:** about 2.8 of the gap to the Unsloth base comes from the recipe alone. The same-recipe base scores 75.1, against Unsloth's 72.3.
- **Long-context gate:** it passes but has little power at n = 45 per bucket. It would detect only drops of about 30 pts. A drop of up to 16.5 pts at 256K against the same-recipe base is not ruled out.

## Q5 vs Q4: is 4-bit the bottleneck?

**Flag: triggered at the threshold, but not supported by the data.**

| | Δ eval-tools task success | Paired 95% CI |
|---|---|---|
| Q5 − Q4, full set (329) | +3.0 (275 vs 265) | [−0.9, +7.0] |
| Q5 − Q4, 100-case subset | −1.0 | |
| BF16 − Q4, 100-case subset | −2.0 | |
| Q5 − Q4, sanity | +0.0 | [−3.7, +3.7] |

The +3.0-pt rule fires by 10 cases (+3.04), but the paired interval includes 0. The unquantized BF16 model does not beat Q4 on the subset. Quantization is not what holds the student back. Q5 is a cheap option (+0.8 GB) with a likely small gain on tool calls.

## Adapter-repair gap (raw vs post-repair)

**Gap: 0.0 pts for every model.** The live adapter at tier light (`validate_and_repair`, as `agent_loop` runs it) changed no call in any of the six eval-tools result sets. `scripts/distill/repair_gap.py` (d5ab36d) recomputes raw scores case by case, and they match `score.py`'s summaries exactly.

| Model | Task success, raw | Task success, post-repair | Arg validity, raw = post | Exact args, raw = post | Calls fixed / broken |
|---|---|---|---|---|---|
| Student Q4 | 80.5% | 80.5% | 99.3% (265/267) | 9.0% | 0 / 0 |
| Student Q5 | 83.6% | 83.6% | 100% | 8.7% | 0 / 0 |
| Base (same recipe) | 75.1% | 75.1% | 100% | 9.7% | 0 / 0 |
| Base (Unsloth) | 72.3% | 72.3% | 100% | 10.1% | 0 / 0 |
| 27B | 72.9% | 72.9% | 100% | 11.2% | 0 / 0 |
| Saluki | 70.5% | 70.5% | 100% | 9.5% | 0 / 0 |

- **What the adapter could do:** at tier light it can only drop a null optional argument, fuzzy-match an unknown tool name, or pull JSON from a string input. None of those came up: no unknown names, no nulls, no calls read from text.
- **What the misses are:** choices, not syntax errors. Wrong tool picked: Q4 58, Q5 48, same-recipe base 44, Unsloth base 50, 27B 65. No call made: Q4 4, Q5 6, same-recipe base 38, Unsloth base 41, 27B 24. The student's gain over base comes almost entirely from making a call when one is needed (4 vs 41 no-call misses).
- **The student's 2 invalid arguments:** both are calls cut off at the output limit (finish reason `length`). The live loop refuses those before the adapter runs.

## Quantization check (perplexity on a held-back train slice)

| | BF16 | Q4_K_M | Q5_K_M |
|---|---|---|---|
| Student | 9.340 (GPU) / **9.339 (CPU cross-check)** | 8.574 | 9.243 |
| Base, same recipe | 12.837 | 12.197 | – |

- **CPU check:** the CPU BF16 run agrees with the GPU to 0.02%, so the GPU BF16 path is numerically sound. The intermittent cuBLAS BF16 fault only crashed; it didn't corrupt results.
- **Q4 below BF16:** Q4 reading lower than BF16 shows up on both the student and the base. It is a property of this slice and imatrix (both drawn from train rows), not a fault in the student.
- **Learning:** the student is much better than the base on in-distribution text (9.34 vs 12.84 at BF16).

## Notes and caveats

- **Q5:** only eval-tools and sanity were run, per the operator's list. Its long-context and injection numbers would be needed before Q5 could be promoted.
- **Sanity:** the student is 1 pt below the Unsloth base (within noise) and 1.4 above the same-recipe base. Its general ability held.
- **Injection:** resistance is at ceiling for every model (≥98.4%). eval-injection can't separate them. Round 2's harder injection examples target this.
- **Saluki:** the live test-drive model is below the student on tool calls (70.5 vs 80.5).
- **Phase 6 path:** the GPU BF16 imatrix crashed, so it now runs on a Q8_0 copy. See DISTILL-LOG, Sun 11:50–12:50.

Sources: `~/distill/phase7/9b-r32-v4b-max-derived/report/DISTILL-PHASE7-SCORES.md` and `summaries/` on the 3090 Ti host. The subset ids are in `DISTILL-PHASE7-SUBSET.json`.
