# Autonomous runbook: distillation + flywheel completion

You are running this mostly unattended. The operator is away for several days and on mobile only.
Work through the phases in order, using objective gates instead of waiting for approval.
Save this file as `docs/sprints/DISTILL-FLYWHEEL-RUNBOOK.md` and keep a running log at
`docs/sprints/DISTILL-LOG.md` (timestamped, append-only).

## ZERO-APPROVAL MODE (active now; overrides anything below that conflicts)

The operator is unreachable for ~36 hours and will not answer anything. **Never wait for a reply.** Every point
where this runbook says "ask", "reply YES" or "operator decides" resolves automatically as written here.
Telegram is send-only: report, never ask.

1. **Keep yourself running.** You're in a `/loop` session. Before ending any turn, schedule your next wakeup:
   ~10–20 min while jobs run, ~5 min while actively working. State lives in `DISTILL-LOG.md` and
   `data/distill/state.json` (current phase, running jobs, next action), so any wakeup, crash or usage-limit pause
   resumes cleanly. Read state.json first on every wakeup.
2. **Parallelize.** Use subagents for independent work (e.g. scorer code while the audit runs; dataset building
   while baselines run). Keep GPUs busy at all times during the window.
3. **GPU window: from now until Sunday 18:00 ET, continuous.** `flywheel` may stay applied for the whole window
   (no nightly limits). Dead-man timer = Sunday 18:00 ET → `setup apply business`. The rehearsal round trip is
   still required first. After the window, fall back to the nightly rules.
4. **Qwen3.8 Max subscription: use it heavily.** Up to **75% of the remaining quota** (or $150 if it's per-token
   billing) for judging and teacher generation. Always keep 25% in reserve for live Prometheus use, and back off
   on any rate limit. Log usage in spend-ledger.csv.
5. **Judge gate:** no human hand-check. Max re-judging is the gate (Phase 2), and the judge-sample file is for
   later reading only.
6. **Local 27B (Phase 5 test):** if ≥8k sequences fit and the full run is ≤5 days of compute, run it automatically.
   Use the continuous window while it lasts, then resume from checkpoints in the nightly windows until done.
6b. **No deadline.** There's no finish date. Quality beats speed: never cut corners, skip gates, shrink eval sets or
   shorten training to finish sooner. Once the continuous window closes, keep working through the phases in
   nightly windows (GPU work) and anytime (non-GPU work) until the Final section is complete.
7. **Rented GPUs: never.** No money is spent on rentals in this mode. Build the rental packet only.
8. **Shadow mode:** enable automatically if the Phase 6 gate passes (flag default-off, tests green).
9. **No real-world side effects from training-data generation.** New teacher traces run through the gym/sandboxed
   tool layer only. Never send emails or messages, post to any outside service, touch business records or data, or call
   external write APIs. If a task needs a write tool, use the gym's mock.

### Phase 2b (new, runs during/after Phase 2): Teacher expansion with Max

Use the subscription budget to fill gaps the audit found. Generate new Max traces in the gym for under-represented
categories:
- long multi-step tool chains
- error/timeout recovery
- compacted-history continuation
- chat-mode (`suppress_tools`, no tools) examples
- long-context tasks (32k–200k prompts built from real wiki/memory/docs content)
- injection-resistance examples for **training** (keep this set disjoint from `eval-injection`)

Target: +3–8K verified new traces. Each one passes the same strict Max-judge rubric and tool-success check, is tagged
`source=max-gym-<date>`, gets deduped against the existing corpus, and goes through the same task-level split.
Phase 3 waits for at least the first ~2K, then rebuilds when more arrive (later runs can use the rest).

## Goal

Distill the harvested Qwen3.8 Max traces into a tuned **Qwen3.5-9B** that is excellent at Prometheus's tools,
keeps long context (target 256k), and runs in shadow mode on the 4090. Then finish the fine-tuning flywheel
(`docs/sprints/SPRINT-FLYWHEEL-CADENCE.md`) so the loop runs weekly with this pipeline as its train stage.
(A Qwen3.8-27B run on rented GPU is a later step. Prepare its config, but don't rent anything.)

## Ground rules (non-negotiable)

1. **Branch:** work on `feat/distill-flywheel` in a separate **git worktree** (e.g. `../Prometheus-distill`), never by
   switching branches in the main checkout, because the live daemon may run from it. Open a draft PR. Never push to main.
   Commits use the OAra Labs identity (support@oara.ai). No personal names, private business names or personal email domains
   in repo files, configs, model names, or reports.
2. **Raw traces are read-only.** Never modify or delete the harvested trace store. Write everything
   derived under `data/distill/` (gitignored). Training data never leaves the machines,
   except for the judge calls in Phase 2.
3. **Budget:** judge calls go through Prometheus's existing Qwen3.8 Max subscription/credentials. That
   subscription is the operator's **primary live model**, so live Prometheus traffic always wins:
   - Check the plan's billing model. If it's per-token, cap judge spend at **$25** (stop at $23). If it's a quota
     plan, use at most **20% of the remaining quota** for this runbook.
   - Run judge batches 01:00–06:00 ET only, at low concurrency. Back off on any 429/rate-limit, and pause
     entirely if live requests are being throttled.
   - Track every call in `data/distill/spend-ledger.csv`. Notify when you stop for budget.
4. **GPU setups (named images of the OAra Labs GPUs):**
   The operator manages GPU configurations as **setups**: named images of what runs on each GPU (4090 tower,
   3090 Ti eGPU, 4060 Ti eGPU box). First, find the existing setups mechanism (it's being built alongside the
   current GPU reshuffle) and **extend it rather than building a parallel one**. If none exists yet, build a minimal
   one in that spirit: `setups/<name>.yaml` (per host and GPU: services, models, launch args, ports, health
   checks) plus a CLI with:
   - `setup save <name>`: capture what's running now as a setup.
   - `setup apply <name>`: stop whatever isn't in the target, start what is, confirm VRAM, run health checks.
   - `setup status`: which setup is active and whether it's healthy.
   Two setups matter here:
   - **`business`**: the operator's normal Prometheus + business-services image. Save it from the live state at the
     start of the first window. Don't hand-author it.
   - **`flywheel`**: 4090 = training + quantization. 3090 Ti = evals and baselines. 4060 Ti (if online) keeps a
     small local model for the voice/extractor/fallback roles, so the business stack isn't fully dark. Prometheus
     uses the Qwen3.8 Max API as primary for the duration.
   Restoring always means `setup apply business`. If restore fails, retry up to 2 times, then notify with what failed.
   Rules:
   - **Windows:** while the operator is away, swaps are allowed **01:00–07:00 ET nightly**, plus the existing
     Sat 02:00–06:00 window afterward. Never outside a window.
   - **Dead-man restore:** before every `apply flywheel`, schedule an independent systemd timer that runs `setup apply business`
     at window end + 15 min, whatever happens. That timer must not depend on this session. Long jobs checkpoint
     and resume in the next window instead of overrunning.
   - **Never stop:** the Prometheus daemon itself, its databases/storage, Tailscale, SSH, or anything
     internet-facing for the business services that isn't GPU-backed. If a GPU service turns out to be customer-facing
     and has no fallback, leave it running and work around it, or log an open question.
   - **First swap is a rehearsal:** in Phase 0, do a 10-minute `save business` → `apply flywheel` → `apply business` round trip in the
     first window and verify every service comes back healthy. If the rehearsal fails, don't swap again.
     Restore, notify, and run only the no-GPU phases.
   - **GPUs are on different machines:** never split one training job across them. The 4090 tower's 64 GB of
     system RAM is available for data processing and CPU offload.
   - The parent (Qwen3.8 Max) is an API and needs no GPU.
5. **Long jobs** (training, quantization, big evals) run detached (`systemd-run --user` or tmux),
   with logs under `data/distill/logs/` and checkpoints so they can resume. Poll; don't block.
6. **Anything destructive, irreversible, or touching live serving that this file doesn't explicitly allow:**
   don't do it. Add it to `docs/sprints/DISTILL-OPEN-QUESTIONS.md` and notify.
7. **Failure handling:** if a gate fails, stop that track, notify, and continue any independent work.
   Don't loop retrying the same failure more than twice.

## Notifications (Telegram via Prometheus)

Find Prometheus's existing Telegram send path and use it. Send messages of 5 lines or fewer:
- at the end of every phase (gate result + 1–2 key numbers)
- immediately on any failure, budget stop, or open question
- a daily heartbeat at ~09:00 ET (current phase, what's running, ETA)

Test it first with: "Distill runbook started. Phase 0 running."

---

## Phase 0: Topology check (read-only)

- Over Tailscale, discover every GPU host. For each GPU report: model, VRAM total/used, the services on it,
  and their health.
- Confirm Saluki is serving on the 3090 Ti and Prometheus is healthy.
- Record the current per-GPU layout (it becomes the `business` setup).
- Check disk space where training runs (need ~100 GB free) and confirm HF downloads work.
- **Student check:** search the Qwen org on Hugging Face for any open Qwen3.6–3.8 model of 8–10B with
  Apache 2.0 and ≥256k native context. If one exists, baseline it next to Qwen3.5-9B in Phase 4 and use
  whichever scores higher on eval-tools as the student. Don't use Qwen3-8B (2025): it's 32k native with
  full attention, so it can't hold long context cheaply.
- Find or build the setups mechanism, define `flywheel`, and run the 10-minute rehearsal (see ground rule 4) in the first window.
- **Gate:** if the rehearsal fails or the GPU reshuffle looks unfinished, run only the no-GPU phases
  (1–3 and scorer code) and notify that training is blocked.

## Phase 1: Flywheel status + trace audit (read-only)

Part A: compare SPRINT-FLYWHEEL-CADENCE.md against the code. For each stage (gym, harvest, LoRA train,
benchmarks, tests, Sat 02:00–06:00 ET window, Monday reviewer handoff), mark it built / partial / missing,
with file paths.

Part B: trace corpus.
- Total count. Breakdown by generating model; flag every non-Max source.
- Pass/fail split, and which judge produced the labels.
- Category mix (coding, memory/wiki, SENTINEL/daemon, chat-mode, other).
- Near-duplicate estimate (normalized prompt hash + daemon/heartbeat repeats).
- Token lengths (p50/p90/p99/max) and the count over 16k.
- Whether traces store the exact rendered template/tool schema `LlamaCppProvider` sends, or raw messages.

Write the report to `docs/sprints/DISTILL-PHASE1-AUDIT.md`. Gate: automatic. Continue.

## Phase 2: Judge verification (Qwen3.8 Max, $25 cap)

- Stratified sample: 300 "pass" + 100 "fail" traces, after dedupe, Max-sourced only.
- Re-judge each with Max using a strict rubric: right tool, valid args, sensible order,
  task actually completed, no fabricated tool results. Output JSON verdict + reason.
- Compute agreement and the false-pass rate.
- **Gate:**
  - false-pass ≤ 5%: trust the existing labels.
  - false-pass > 5%: re-judge the deduped pass set within budget, prioritizing the categories
    with the worst agreement. Whatever can't be re-judged gets used only if its tool calls returned success.
- Put the sample verdicts in `data/distill/judge-sample.jsonl` (the operator will skim it later).

## Phase 3: Build datasets

- Keep only Max-sourced, verified-pass, deduped traces.
- Cap any single category at 35% (coding especially). Make sure chat-mode (`suppress_tools`)
  examples, error/recovery examples and compacted-history examples are present.
- Render with the **exact** chat template + tool-schema format `LlamaCppProvider` uses at runtime.
  Use loss masks on assistant turns only.
- Split **by task** (not by trace) so no task appears in both train and eval:
  - `train`: 5–10K examples; `val`: ~3%
  - `eval-tools`: held-out tasks + the failed traces (hard cases)
  - `eval-longctx`: needle/recall tests built from real Prometheus prompts at 32k/64k/128k/256k
  - `eval-injection`: prompt-injection cases in tool outputs / memory / wiki. **Eval only, never train.**
  - a small general-sanity set
- Truncate or split training examples to ≤16k tokens. Long originals go to eval-longctx instead.
- **Gate:** no train/eval leakage (hash check), sizes in range, a spot-check of 5 rendered examples
  matches live provider output byte-for-byte. Report: `docs/sprints/DISTILL-PHASE3-DATA.md`.

## Phase 4: Baselines

- Build a scorer: tool-name accuracy, argument validity/exactness, task success,
  long-context recall by depth, injection resistance, sanity score.
- Score **base Qwen3.5-9B** (Q4_K_M GGUF, llama.cpp, `-c 262144 -np 1 -fa -ctk q8_0 -ctv q8_0`) on the 4090.
- Score **Saluki** via its existing endpoint at concurrency 1, overnight only (don't load the live server).
- **Large-student bake-off** (decides which model gets the later rented-GPU run): score **Qwen3.8-27B**
  and **Meta Muse Glimmer 30B** (Apache 2.0, ~131k native context) at Q4 on the 4090, same evals, same flags
  (use each model's native max context). If Glimmer's HF repo is gated, log it as an open question and skip it.
  Rank them on eval-tools first, then eval-longctx at ≤128k.
- Report: `docs/sprints/DISTILL-PHASE4-BASELINES.md`.

## Phase 5: Training (4090, Unsloth QLoRA)

- Inside a window only: arm the dead-man timer → `setup apply flywheel` → launch detached.
- **Smoke test first:** 200 examples, ~50 steps. Loss must drop and the eval pipeline must run end-to-end.
- **Full run:** QLoRA r=32/alpha=32, lr ~1e-4 cosine, 2 epochs, max_seq 16k
  (step down to 12k → 8k on OOM), gradient checkpointing, checkpoint every ~200 steps, resumable.
- When the run ends or the window closes (checkpoint first), `setup apply business` and verify health.
- **After the 9B is done: local 27B feasibility test.** On the 4090 (with CPU offload into the 64 GB of system RAM
  if needed), find the longest sequence length at which a QLoRA of the large-student bake-off winner fits: try
  16k → 12k → 8k → 4k. For each length that fits, report tokens/sec, the share of training examples that fit
  untruncated, and the estimated wall-clock time for the full run.
  - If a length of ≥8k fits and the full run is ≤5 days of compute, send the numbers on Telegram and ask the
    operator to reply with one of:
    - **CONTINUOUS:** keep `flywheel` applied around the clock until the run finishes. The business stack runs on
      the Max API + 4060 Ti fallback meanwhile. The dead-man timer becomes "run finished OR 6 days elapsed".
      Checkpoint every ~2 hours, and `setup apply business` right after the run.
    - **NIGHTLY:** run only in the 01:00–07:00 windows, resuming from checkpoint each night.
    - **NO:** treat it as a rented-GPU run.
    Do nothing without an explicit reply.
  - Otherwise, report that it should be a rented-GPU run and include the cost estimate.
- **Rental packet for the bake-off winner (Qwen3.8-27B or Muse Glimmer 30B). Prepare it; never launch it.**
  Goal: zero debugging on paid time. Target **Modal** (pays only while the job runs, shuts down automatically).
  - Pre-tokenize the dataset locally and package it with a pinned container image. Base it on the official
    `unsloth/unsloth` Docker image at a fixed tag, so local and rented runs use the same environment.
  - Use Unsloth's recent features where they fit:
    - its training memory estimate, to pick the smallest GPU that fits
    - token preview, to verify the chat template and assistant-only loss masks before any paid run
    - the lower-memory 4-bit LoRA path for NVFP4 checkpoints (reported: Qwen3.8-27B peaks ~40 GB instead of
      ~73 GB), which may allow a cheaper 48 GB GPU (L40S/A6000-class) instead of an H100
    - accurate resume and remaining-time estimates
    - `unsloth eval` for adapter checks
  - Prove the exact training script locally first: a tiny 27B smoke run on the 4090 (few steps, short sequences,
    CPU offload OK) must load, train, checkpoint, resume, and save the adapter without errors.
  - Script flow on Modal: upload data → **50-step calibration** → measure tokens/sec → compute the exact
    full-run cost → continue only if it's ≤ the approved cap, otherwise save and stop → full run, checkpointing to a
    persistent volume → download the adapter → GPU released.
  - Hard limits in code: a max-runtime timeout and a max-dollar cap. Never leave idle GPUs running.
  - Write `docs/sprints/DISTILL-RENTAL-PACKET.md`: GPU choice (48 GB L40S/A6000-class vs A100 80GB vs H100 vs H200, ranked by **cost per run**,
    not price per hour), estimated hours, estimated $ (low/likely/high), and the exact one-line launch command.
  - Telegram the estimate. The operator launches it themselves, or replies "RENT YES $<cap>" to have you launch it
    with that cap.

## Phase 6: Score + quantize

- Merge the LoRA, then export GGUF Q4_K_M and Q5_K_M using an **imatrix computed from the train traces**.
- Run every eval set on the **quantized** model (not just fp16).
- **Gate to shadow (all must hold):**
  - tool-call task success ≥ base 9B + 10 pts
  - long-context recall no worse than base by > 3 pts at any depth
  - injection resistance ≥ base
  - sanity score no worse than base by > 3 pts
- If the gate fails, try at most one adjusted run (e.g. 1 epoch or lower lr), then stop and report.
- Report: `docs/sprints/DISTILL-PHASE6-RESULTS.md` (table: base 9B vs tuned 9B vs Saluki).

## Phase 7: Shadow mode

- Serve the tuned 9B on the 4090 with llama.cpp (256k, `-np 1`), registered as a **shadow** provider.
- Add shadow mirroring to Prometheus **behind a config flag, default off**: mirror requests, log both
  outputs, and act only on the primary. The full test suite must pass before enabling.
- Enable it. Auto-disable and notify if the primary's error rate or latency degrades.
- The primary is **Qwen3.8 Max** (the teacher), so shadow mode measures how close the offspring gets to Max on
  live traffic. Track, per request category: same tool chosen, args equivalent, and final answer judged
  equivalent. Report which categories are "Max-equivalent" (≥95% agreement). Those are candidates to route
  locally later; don't change routing now.
- Collect shadow comparisons in `data/distill/shadow/`. Summarize daily in the heartbeat.

## Phase 8: Finish the flywheel

- Wire the stages into the Saturday 02:00–06:00 ET cadence: harvest → clean/verify → build sets →
  train (only if ≥ N new verified traces; resume across weekends from checkpoints) → quantize → score.
- Tag every new trace with source model + adapter version. **Never train on unverified offspring output.**
- Produce the weekly report file the Monday reviewer reads (wins/losses vs the previous model, spend, issues).
  Don't modify the scheduled task itself; just make sure its input exists and is current.
- Tests for the new pipeline stages must pass.

## Final

- Update the draft PR description with phase results.
- Write `docs/sprints/DISTILL-FINAL-REPORT.md`: what shipped, the scorecard, spend, open questions,
  and recommended next steps (including the 27B rented run).
- Telegram summary: shadow status, headline numbers, and anything that needs the operator.
