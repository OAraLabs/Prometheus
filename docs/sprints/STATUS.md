# Distillation + flywheel — status

_Updated 2026-10-30 19:50 ET. Refreshed on every check-in. Details: `DISTILL-LOG.md`; decisions: `DISTILL-OPEN-QUESTIONS.md`._

## In one paragraph

The groundwork is done and tested, but no model has been trained yet. Nothing has used a GPU so far. Tonight a
read-only baseline of the current local model (Saluki) runs on the 3090 Ti, gently, around its health checks. After
that, the 3090 Ti switches to "flywheel" mode: a local Qwen3.8-27B teacher generates new practice tasks, because the
existing Max traces can't be used for training. Then the 9B student trains on that. The 4090 is left alone.

## Done

- **Phase 0 – hardware check.** Both GPUs mapped. The 4090 is off-limits: it backs a customer tunnel and an overnight job.
- **Phase 1 – data audit.** 11,152 harvested traces, but only ~300 distinct tasks, 86% coding, no chat-mode examples.
- **Phase 3 – dataset builder** (dry run). Works and checks for leaks, but the data fails its own gate: 1,323 training
  rows against a 5,000–10,000 target, and no chat-mode.
- **Eval sets** built on the 3090 Ti host: long-context (32K–256K), prompt-injection and general-sanity tests.
- **Tools, all adversarially reviewed and fixed:** GPU setup save/apply with an automatic restore timer, the scorer,
  the judge, the dataset and eval-set builders.
- **Plan terms checked.** The Qwen Max key is a Token Plan: interactive use only. It is never used for batch work
  here, and 0 Max calls have been made.

## Running now

- **Saluki baseline prep.** The probe-guard fix is in its final rehearsal; timers arm after review. Windows: 01:00–02:45, 03:45–05:45 and 06:15–06:59 ET.
- **Research on the mini's earlier Claude session:** Qwen plan, 4090 tunnel and night job, and the ~22K vs 11,152 trace count.
- **Prep for the 3090 Ti swap:** teacher model download, the daemon Saluki↔Max switch with exact restore, the gym teacher
  pipeline, and keeping Max traces out of training.
- **New since your last message:** a confidence-interval long-context gate (your Q8 approval), and a task generator built
  from the real task patterns. Quantize and flywheel follow-ups start once the current Phase 5–8 code lands.
- **Phase 5–8 code:** trainer, quantize, shadow mode, flywheel cadence. Last fix pass running.

## Blocked / waiting on you

- **Training on the existing Max traces:** on hold until you decide (open question 6).
- **4090:** unchanged per your call. No GPU work there.
- **4060 Ti box:** still no SSH.

## Next

1. Tonight: Saluki baseline (read-only, about 777 test cases).
2. Saturday morning: rehearsal swap on the 3090 Ti, then the flywheel window. The teacher generates tasks until the
   dataset passes its gate.
3. Then: 9B QLoRA training on the 3090 Ti, quantize, and score against base and Saluki.
