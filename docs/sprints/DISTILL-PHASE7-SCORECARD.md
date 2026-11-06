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

## Tool choice and argument equivalence (added 2026-11-01)

Operator ask (Sun ~23:00): make wrong-tool rate a headline metric, and add an argument-equivalence check beside exact match. Same stored eval-tools replies as above (the six full runs, plus Student BF16 on the 100-case subset); no new requests.

**What changed in the scorer.** `score.py` (still v4: the gated metrics are unchanged) now gives every scored eval-tools case exactly one outcome, so the misses add up with the successes to 100%:

- **wrong tool:** the first call names an offered tool other than the expected one;
- **no call:** no call when one was needed;
- **extra call:** a call when none was needed (this set has no such cases, so it is 0 everywhere);
- **other misses:** a tool the case doesn't offer, the right tool but the case still failed, or no readable reply.

`compare` prints these as headline rows with paired deltas. They are information only: the Phase 6 gate is unchanged and none of them is gated.

**Argument equivalence** (`scripts/distill/arg_equiv.py`) sits beside exact match. Two calls to the same tool are equivalent when they would have the same effect. Every case records the tier that decided it:

1. **exact:** equal the way exact match compares.
2. **canonical:** equal after identities that can be proven:
   - JSON normalization;
   - schema defaults filled in, and null optional arguments dropped;
   - paths: `~`, relative paths against the case's working directory, `./`, doubled slashes, and a directory's trailing slash;
   - shell: whitespace, quoting (except where it changes meaning), an unquoted `~`, a trailing `;`, a leading `cd` to the working directory, and the order of boolean flags for a few programs.
3. **execution:** for bash and read_file. Both calls run in the gym's sandbox: the `teacher_gen` grammar, bwrap, and a clone of the gym's repo store (27055dbf). Each call runs in two snapshots, the fixture and a padded copy, and the first snapshot is run twice. A pair is "equivalent" only when both calls exit 0 with output that isn't empty and the output and workspace effects are identical in both snapshots. It is "not equivalent" when both run cleanly and differ.
4. **conservative:** a write tool (edit, write, task or memory) with different arguments counts as not equivalent and is never run.
5. **undecidable:** everything else. It is counted on its own and never guessed. The proven rate is a lower bound. The upper bound adds the undecidable cases.

### Per model

| Model | Task success | Wrong tool | No call | Extra call | Other misses | Exact args | Arg equivalence (proven) | Undecidable | Equivalence upper bound |
|---|---|---|---|---|---|---|---|---|---|
| Student Q4 | 80.5% [75.9, 84.5] | **17.6%** (58) [13.9, 22.1] | 1.2% (4) [0.5, 3.1] | 0.0% | 0.6% (2) | 9.0% (24/267) [6.1, 13.0] | 9.0% (24/267) [6.1, 13.0] | 227 | 94.0% |
| Student Q5 | 83.6% [79.2, 87.2] | **14.6%** (48) [11.2, 18.8] | 1.8% (6) [0.8, 3.9] | 0.0% | 0.0% | 8.7% (24/275) [5.9, 12.7] | 8.7% (24/275) [5.9, 12.7] | 237 | 94.9% |
| Student BF16 (100-case subset) | 81.0% [72.2, 87.5] | **18.0%** (18) [11.7, 26.7] | 1.0% (1) [0.2, 5.4] | 0.0% | 0.0% | 4.9% (4/81) [1.9, 12.0] | 4.9% (4/81) [1.9, 12.0] | 70 | 91.4% |
| Base (same recipe) | 75.1% [70.1, 79.4] | **13.4%** (44) [10.1, 17.5] | 11.6% (38) [8.5, 15.5] | 0.0% | 0.0% | 9.7% (24/247) [6.6, 14.1] | 9.7% (24/247) [6.6, 14.1] | 210 | 94.7% |
| Base (Unsloth) | 72.3% [67.3, 76.9] | **15.2%** (50) [11.7, 19.5] | 12.5% (41) [9.3, 16.5] | 0.0% | 0.0% | 10.1% (24/238) [6.9, 14.6] | 10.1% (24/238) [6.9, 14.6] | 204 | 95.8% |
| 27B | 72.9% [67.9, 77.5] | **19.8%** (65) [15.8, 24.4] | 7.3% (24) [5.0, 10.6] | 0.0% | 0.0% | 11.2% (27/240) [7.8, 15.9] | 11.7% (28/240) [8.2, 16.3] | 201 | 95.4% |
| Saluki | 70.5% [65.4, 75.2] | **24.3%** (80) [20.0, 29.2] | 5.2% (17) [3.3, 8.1] | 0.0% | 0.0% | 9.5% (22/232) [6.3, 13.9] | 9.5% (22/232) [6.3, 13.9] | 196 | 94.0% |

Each row sums to 100%: task success, wrong tool, no call, extra call and other misses. Brackets are Wilson 95% intervals. Both argument columns count only cases where the right tool was named, the same denominator as exact args. Student Q4's 2 other misses are the 2 calls cut off at the output limit.

**Which tier decided each case:**

| Model | Exact | Canonical | Execution (all "differ") | Conservative | Undecidable |
|---|---|---|---|---|---|
| Student Q4 | 24 | 0 | 10 | 6 | 227 |
| Student Q5 | 24 | 0 | 10 | 4 | 237 |
| Student BF16 (subset) | 4 | 0 | 5 | 2 | 70 |
| Base (same recipe) | 24 | 0 | 7 | 6 | 210 |
| Base (Unsloth) | 24 | 0 | 8 | 2 | 204 |
| 27B | 27 | 1 | 7 | 4 | 201 |
| Saluki | 22 | 0 | 8 | 6 | 196 |

### Student vs base, paired

| Comparison | Wrong-tool rate Δ (n = 329) | Paired 95% CI | Arg equivalence Δ (cases where both named the tool) | Paired 95% CI |
|---|---|---|---|---|
| Q4 vs base (same recipe) | +4.3 (13.4 → 17.6) | [−0.2, +8.7] | −0.5 (n = 222) | [−2.4, +1.3] |
| Q4 vs base (Unsloth) | +2.4 (15.2 → 17.6) | [−2.2, +7.1] | +0.0 (n = 216) | [−2.1, +2.1] |
| Q5 vs base (same recipe) | +1.2 (13.4 → 14.6) | [−2.8, +5.3] | −0.9 (n = 228) | [−3.0, +1.0] |
| Q5 vs base (Unsloth) | −0.6 (15.2 → 14.6) | [−5.0, +3.8] | −0.5 (n = 221) | [−2.4, +1.4] |

Δ is the student minus the base. A positive wrong-tool Δ means more wrong tools. Intervals are Newcombe intervals.

### Checker vs hand labels

Labelled sample: **50 inexact pairs.**
- 25 most similar pairs (word overlap, across all six models), plus 15 random ones;
- every distinct pair that execution decided (12 more);
- 2 pairs dropped as unknowable: an edit whose wider context appears nowhere in the case.

Labels were made from the call shapes. The labels and the report are kept on the 3090 Ti host as ids and verdicts only.

Results:
- **Proven "equivalent": precision 1/1, recall 1/2.** The miss is an edit whose wider context is in the file. The checker rightly calls it undecidable, since only the file can tell.
- **Every decisive verdict agrees with the labels (19/19):** 1 canonical, 13 execution "differ", 5 conservative.
- **Undecidable pairs:** 31, of which the labels call 1 equivalent and 30 not.
- **True equivalence is rare even in the similarity-enriched sample:** only 2 of the 50 labelled pairs are equivalent.

### What this says

- **(a) Tool choice:** the student is not better at picking the tool. Its wrong-tool rate is 17.6% at Q4, vs 13.4% for the same-recipe base. That difference is not significant: the interval reaches −0.2. Its whole gain is in making a call at all: no-call misses drop from 11.6% to 1.2%.
  - **Where the student's calls went** (Q4 vs same-recipe base):
    - 25 base no-calls became successes;
    - 10 base no-calls became wrong tools;
    - 24 base successes became wrong tools, and 19 cases went the other way.
  - **The student's wrong tools:** bash ↔ edit_file (16), bash → read_file (5), and the task_* tools ↔ bash (19).
  - Round 2's focus on tool choice targets the right thing.
- **(b) Arguments:** equivalence moves the number by at most one case per model. The only canonical match is the 27B writing `~/…` for an absolute path. Every pair that execution could decide (50 across the six runs) really differs.
  - **Why most pairs are undecidable:**
    - 77% of them have an expected call that names paths the fixture can't have: other repos, worktrees, `/tmp`, and `~/.prometheus`.
    - 20% have an expected call that uses something the sandbox refuses: `gh`, docker, interpreters, the network, `$` expansions, and multi-line scripts.
  - **So** the ~10% exact-match rate is not a scoring artefact. The reference calls are long diagnostic scripts, and the models write different, if often reasonable, ones. With an upper bound of 94–96%, this eval set can't separate "equivalent" from "different but fine".
  - **Next step:** a usefulness judgement on the call ("does it make the same progress?") would be a different metric, not equivalence.

Sources: `~/scratch-arg-equiv/out/` on the 3090 Ti host (`full.json`, `bf16.json`, `validate.json`, `labels.json`, ids and verdicts only).

<!-- acceptable-tools:begin -->
## Acceptable-tool wrong-tool rate (rebuilt 2026-11-06 with Max as the judge)

**Result: with Max as the tool-choice judge, most old "wrong tool" picks are acceptable alternatives, and no case is left undecided.** Read the caveats below before quoting the new rates: the judge passed its gate, but the gate only tests the easy ends.

Operator ask: replace the single-reference wrong-tool metric with an acceptable-tool set per case, built from Max + 27B + judge agreement, plus an outcome check where the gym can run it. Same stored eval-tools replies as above (329 cases; Student BF16 on the 100-case subset), and the same candidate calls, queue and control items as the 2026-11-04 run. No new model requests.

**The rule.** Per case, the acceptable set is Max's expected tool plus every offered tool for which at least one model's first call was ruled acceptable:
- **outcome:** the gym ran both that call and Max's reference (`bash` / `read_file`), both exited 0 with no workspace effect, and the call's output contains the reference's in both snapshots;
- **judge:** Qwen Max (`qwen3.8-max`, Token Plan, temperature 0, thinking off) in the tool-choice view scored right_tool 2 and neither sensible_order nor task_progress 0. The view's prompt and rubric are byte-identical to the 27B run's. Argument scores are recorded but don't decide tool choice.

A tool is rejected when every one of its calls was ruled not acceptable. Nothing here changes task success or the Phase 6 gate.

**The judge's agreement gate** (same 140 control items as the 27B run; bar: ≥90% of Max's own reference calls accepted, ≥95% of obviously wrong tools rejected, on the point estimate):

| Judge | Positive controls accepted | Negative controls rejected | Gate |
|---|---|---|---|
| Max (2026-11-06) | 74/80 = 92.5% [84.6, 96.5] | 60/60 = 100% [94.0, 100] | pass |
| 27B UD-Q4_K_XL (2026-11-04) | 70/79 = 88.6% [79.7, 93.9] | 59/59 = 100% [93.9, 100] | fail |

Max's rulings are applied. The 27B run's are not; its files are unchanged.

**The sets.** 355 candidate calls (198 case-tool pairs in 149 cases): 164 tools acceptable, 34 rejected, 0 undecided. The judge admitted all 164. The outcome check admitted none: 304 calls `not_run` (a tool the gym can't execute), 44 `reference_not_ok`, 7 `differs`.

### Old vs new, per model

| Model | n | Wrong tool, old (single reference) | Wrong tool, new (acceptable set) | Old wrong tools now acceptable | Still wrong | Undecided |
|---|---|---|---|---|---|---|
| Student Q4 | 329 | 17.6% (58) [13.9, 22.1] | 2.1% (7) [1.0, 4.3] | 51 | 7 | 0 |
| Student Q5 | 329 | 14.6% (48) [11.2, 18.8] | 1.8% (6) [0.8, 3.9] | 42 | 6 | 0 |
| Student BF16 (100-case subset) | 100 | 18.0% (18) [11.7, 26.7] | 3.0% (3) [1.0, 8.5] | 15 | 3 | 0 |
| Base (same recipe) | 329 | 13.4% (44) [10.1, 17.5] | 1.5% (5) [0.7, 3.5] | 39 | 5 | 0 |
| Base (Unsloth) | 329 | 15.2% (50) [11.7, 19.5] | 1.5% (5) [0.7, 3.5] | 45 | 5 | 0 |
| 27B | 329 | 19.8% (65) [15.8, 24.4] | 0.6% (2) [0.2, 2.2] | 63 | 2 | 0 |
| Saluki | 329 | 24.3% (80) [20.0, 29.2] | 4.0% (13) [2.3, 6.6] | 67 | 13 | 0 |

Rates are of all scored cases, with Wilson 95% intervals.

| Comparison | Wrong tool, old: Δ | Paired 95% CI | Wrong tool, new: Δ | Paired 95% CI |
|---|---|---|---|---|
| Student Q4 vs Base (same recipe) | +4.3 (13.4 → 17.6) | [-0.2, +8.7] | +0.6 (1.5 → 2.1) | [-1.7, +3.0] |
| Student Q4 vs Base (Unsloth) | +2.4 (15.2 → 17.6) | [-2.2, +7.1] | +0.6 (1.5 → 2.1) | [-1.7, +3.0] |
| Student Q5 vs Base (same recipe) | +1.2 (13.4 → 14.6) | [-2.8, +5.3] | +0.3 (1.5 → 1.8) | [-1.7, +2.4] |
| Student Q5 vs Base (Unsloth) | -0.6 (15.2 → 14.6) | [-5.0, +3.8] | +0.3 (1.5 → 1.8) | [-1.7, +2.4] |

Paired Δ is the student minus the base (Newcombe 95%). On either metric, no student-vs-base difference is significant.

### Caveats

- **The gate tests only the easy ends.** The negative controls are unrelated tools (text-to-speech, image or video generation, a YouTube transcript). Nothing checks whether the judge can tell a plausible-but-wrong tool from an acceptable one, which is the call that decides this metric.
- **The judge is lenient.** It admitted 164 of the 198 candidate tools (83%), including `bash` → `edit_file` (18 case-tool pairs), `web_fetch` (10), `ask_user` (6) and `memory` (6). The 27B judge was about as lenient: on the 350 candidate calls both ruled, they agreed on 299 (85.4%), and each accepted about 78%. So the new rates are best read as "picks a judge calls defensible", not as a verified right-tool rate.
- **The positive gate is narrow.** The 92.5% point estimate clears the 90% bar, but its 95% interval reaches down to 84.6%.
- **Max judges alternatives to its own references.** The expected tool is Max's own choice, so a self-consistency bias would push toward rejecting alternatives. That is the opposite of what happened here, but it was not measured.
- **The outcome check decided nothing.** Every admission rests on the judge alone.
- **Max-derived.** The annotation, agreement and report are tagged `max-derived` and stay private.

### Most common expected → called pairs

| Max's tool | Called | Decision | Case-tool pairs |
|---|---|---|---|
| bash | read_file | acceptable | 37 |
| bash | edit_file | acceptable | 18 |
| bash | web_fetch | acceptable | 10 |
| bash | task_output | acceptable | 7 |
| edit_file | bash | acceptable | 7 |
| bash | ask_user | acceptable | 6 |
| bash | lcm_grep | acceptable | 6 |
| bash | memory | acceptable | 6 |
| bash | task_create | acceptable | 6 |
| bash | task_get | acceptable | 6 |
| task_create | bash | acceptable | 6 |
| bash | write_file | acceptable | 5 |

**Judge cost.** 497 requests to Max on the Token Plan ($0 per call; 495 rulings, plus 2 unparseable replies that were re-asked), 18:08–18:20 UTC. Tokens: 5,354,691 prompt (4,382,592 cached) and 61,630 output, so a mean of 10,774 in and 124 out per item.

Sources are on the 3090 Ti host:
- `~/distill/acceptable-tools-max/` (`report.json`, `agreement.json`, `acceptable-tools.json`: ids, tool names and verdicts only; the judge queue stays there). They are tagged max-derived and private.
- The annotation is also beside the eval set as `phase3/build-v2/eval-tools.acceptable-tools.max-derived.json`. The 27B run's `eval-tools.acceptable-tools.json` is untouched.
- The 27B run is in `~/distill/acceptable-tools/`.

Cases sha256 `234028b0e65f5e15`.
<!-- acceptable-tools:end -->
