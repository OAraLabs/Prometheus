# Distillation + flywheel — status

_Updated 2026-10-31 ~10:50 ET. Refreshed on every check-in. Details: `DISTILL-LOG.md`; decisions: `DISTILL-OPEN-QUESTIONS.md`._

## In one paragraph

No model has been trained yet. The dataset has 2,056 training rows against a 5,000-row gate, and the Max quota ran out
Saturday morning (it resets Nov 16). The 3090 Ti is in its flywheel window: the local 27B teacher has Saluki's port and
generates practice tasks, but slowly (about 15 verified rows an hour). Tonight the 4090 runs a first small training
test on the 9B, and the 3090 Ti runs half of the teacher quality test. The plan for speed is to let the 4090 serve the
teacher at night while the sandboxed tools stay on the 3090 Ti host.

## Done

- **Phase 0–1:** hardware mapped; data audited (about 300 distinct tasks, mostly coding).
- **Phase 2 (judge):** Max re-judged 2,874 of 9,895 traces before the quota ran out; it resumes after the reset.
  So far Max fails 20.5% of the traces the old labels passed.
- **Phase 2b (Max teacher):** 787 verified rows from 1,409 tasks built from your real task patterns.
- **Phase 3 (dataset v3, max-derived, private):** 2,056 train rows (Max 2,052, local 27B 4). Every gate passes except
  size. Coding is the limiting category.
- **Phase 4 (baselines):** base-9B, Saluki and Qwen3.8-27B scored on sanity, tool use, injection and long context.
  The 27B leads; Glimmer 30B is partly done.
- **Phase 5–8 code:** trainer, quantize, shadow mode and the weekly flywheel job are committed but not run (off by
  default).
- **/qwen turns:** saving them as training examples was a config switch; it's on now, and the next /qwen turn will
  confirm it.

## Running now

- **3090 Ti (window until Sun 18:00):** local 27B generation, 49 rows so far. The voice probe is healthy (50 tok/s).
- **Tonight 01:00:** teacher A/B arm A (IQ3) on the 3090 Ti. Arm B (Q4) is skipped: it would run under the voice-probe
  floor (OQ #12). Generation resumes afterwards.
- **Tonight 01:00–07:00, 4090:** rehearsal, a 9B training smoke test (200 examples, private, max-derived), the rest of
  the Glimmer baseline, and the business server restored and checked by 06:40.
  Clocks fall back at 02:00.

## Your approvals in effect

- 3090 Ti: the teacher takes Saluki's port in windows (continuous until Sun 18:00, then nightly 01:00–07:00).
  Saluki is restored at every window end.
- 4090: nightly 01:00–07:00 only; business server restored and checked at every window end.
- Max-derived data may train a private model (Q6b). It is tagged and never published.
- Token Plan key allowed for distillation Max work. It is exhausted until Nov 16, 16:00 UTC.

## Waiting on you (nothing blocks)

- **OQ #12:** how to run teacher A/B arm B. Default: on the 4090 at night.
- **OQ #11:** the 4090 can't sandbox tool commands (needs sudo). Default: tools stay on the 3090 Ti host.
- **Pay-per-token key:** `QWEN_PAYG_KEY` still isn't on the 3090 Ti host.
- **4060 Ti box:** still no SSH.

## Next

1. Build cross-host generation: the 4090 serves the teacher on the tailnet, the 3090 Ti runs the sandboxed tools.
   Target: Sunday night.
2. Grow the dataset to 5,000 train rows, then a full 9B QLoRA run, quantize, and score against base and Saluki.
3. After Nov 16: resume the Max re-judge and Max generation.
