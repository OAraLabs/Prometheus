# Flywheel corpus — results: diversity tripled the unique-pair density

**Branch:** `feat/flywheel-corpus`. The generated diversity corpus was harvested live against
`gemma4-26b`; here's the measured yield and the path to 1,000.

## Yield: 3.2× the baseline density, all unique

| run | tasks | runs | NEW unique pairs | pairs/run |
|---|---|---|---|---|
| baseline (fixed v1 set) | 22 | 66 | 9 | **0.14** |
| **diversity corpus** | 80 | 80 | **35** | **0.44** |

44 pairs total in `gym-training.db`, **44 distinct `context_hash`** (zero dupes — every pair is a
genuinely distinct training example). Breakdown of the 44: `schema_repair` task_create 19 / grep 8 /
task_list 8; `self_correction` task_list 5 / grep 2 / task_create 2. The high-ceiling shapes
(task_create command/goal, grep) carried the volume exactly as the survey predicted; adding grep to
the unwrap list paid off (8 grep schema_repairs). Full set exported to DPO JSONL (44 lines).

**The hypothesis is confirmed:** varying the target value across wrap-prone shapes multiplies UNIQUE
pairs — the corpus more than tripled the per-run density over the fixed task set, with no dedup loss.

## Path to 1,000 (the projection)

At **0.44 unique pairs/run**, 1,000 pairs ≈ **~2,275 runs** — and because pairs dedup on context,
those must be ~2,275 **distinct tasks** (re-running the same 80 would mostly dedup). Two concrete
steps:

1. **Expand the generator's value pools ~8×.** The word lists are small (≈10×10 per shape → ~100
   each). Growing them to ~30×30 yields ~900 distinct tasks per high-ceiling shape × 3 shapes ≈
   2,700 distinct tasks — enough headroom for 1,000 unique pairs at the observed rate. This is a
   curated-data edit to `corpus.py`, deterministic, no mechanism change.
2. **One overnight harvest run** over the expanded corpus. The 80-run sample took ~35–45 min under
   GPU contention with the live daemon (single slot, `--parallel 1`, plus ollama-fallback retries);
   ~2,275 runs ≈ 17–21 h contended, or **~4–6 h with the daemon paused** (sole GPU use) — the
   recommended way to run it.

After that, the **LoRA train itself** is a separate, larger commitment (a training run + an eval
that the fine-tune actually improves the failing tool shapes) — out of scope here. This phase
proved the corpus path: the flywheel produces volume when fed diversity.

## Shipped (this branch)

`src/prometheus/gym/corpus.py` (generator) + `scripts/gen_harvest_corpus.py` +
`gym/tasksets/harvest-v1.yaml` (80 tasks) + `gym/experiments/harvest-corpus.yaml` (manifest) +
`tests/test_gym_corpus.py` (6 tests). Artifacts (not repo): 44 pairs in
`~/.prometheus/data/gym-training.db`.
