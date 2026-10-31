# Distillation + flywheel — status

_Updated 2026-10-30 20:51 ET. Refreshed on every check-in. Details: `DISTILL-LOG.md`; decisions: `DISTILL-OPEN-QUESTIONS.md`._

## In one paragraph

No model has been trained yet, and no GPU has been used so far. Tonight (01:00–07:00) both GPUs start work. The 3090 Ti
runs a gentle read-only baseline of Saluki. The 4090 rehearses its stop/restore round trip, then runs the base-9B
baseline and the 27B feasibility test, and its business model server is restored and checked by 07:00. From Saturday
morning the 3090 Ti switches to flywheel mode: the daemon uses Max, and a local Qwen3.8-27B teacher generates practice
tasks built from your real task patterns. That starts after an IQ3-vs-Q4 teacher quality test.

## Done

- **Phase 5–8 code committed:** trainer, quantize, shadow mode (off by default), weekly flywheel job (off by default), and
  a rental cost packet (H100 about $140 per run; never launched). Full test suite green.

- **Phase 0 – hardware check.** Both GPUs mapped. The 4090 is off-limits: it backs a customer tunnel and an overnight job.
- **Phase 1 – data audit.** 11,152 harvested traces, but only ~300 distinct tasks, 86% coding, no chat-mode examples.
- **Phase 3 – dataset builder** (dry run). Works and checks for leaks, but the data fails its own gate: 1,323 training
  rows against a 5,000–10,000 target, and no chat-mode.
- **Eval sets** built on the 3090 Ti host: long-context (32K–256K), prompt-injection and general-sanity tests.
- **Tools, all adversarially reviewed and fixed:** GPU setup save/apply with an automatic restore timer, the scorer,
  the judge, the dataset and eval-set builders.
- **Plan terms checked.** The Qwen Max key is a Token Plan: interactive use only. It is never used for batch work
  here, and 0 Max calls have been made.

## Answers from the earlier session's notes (done)

- **Qwen key:** a Token Plan (interactive only). Your own Oct 25 ruling was "nothing automated on it", so Max stays out of
  batch judging and generation.
- **4090:** it now serves only the business tunnel: about 5 requests a day, none overnight. The overnight benchmark has
  never actually run. The 4090's nights are free if you want them; until you say so, it stays untouched.
- **~22K traces:** a double count. Each of the 11,152 traces has a twin copy in another database. There are 11,207 distinct
  teacher calls from about 300 tasks.

## Running now

- **Saluki baseline: armed for tonight** (01:00–02:45, 03:45–05:45, 06:15–06:59 ET). One request at a time, only in the
  quiet gap after each voice health check, and it stops and messages you if a check ever looks unhealthy.
- **Prep for the 3090 Ti swap:** teacher model download, the daemon Saluki↔Max switch with exact restore, the gym teacher
  pipeline, and keeping Max traces out of training.
- **Your extra items:** the confidence-interval long-context gate, the task generator built from real task patterns, a
  real tiny-model quantize run, and flywheel end-to-end tests with this week's Monday reviewer file.

## Your approvals in effect

- 3090 Ti: swap windows (continuous until Sun 18:00, then nightly 01:00–07:00); daemon on Max during windows (interactive only).
- 4090: nightly 01:00–07:00 only; notify path installed; restore + tunnel check at every window end.
- Ollama on the 3090 Ti may be stopped in nightly windows only, if the bigger teacher quant wins.
- Long-context gate now uses confidence intervals (Q8).

## Blocked / waiting on you

- **Training on the existing Max traces:** on hold until you decide (open question 6).
- **4060 Ti box:** still no SSH.

## Next

1. Tonight: Saluki baseline (read-only, about 777 test cases).
2. Saturday morning: rehearsal swap on the 3090 Ti, then the flywheel window. The teacher generates tasks until the
   dataset passes its gate.
3. Then: 9B QLoRA training on the 3090 Ti, quantize, and score against base and Saluki.
