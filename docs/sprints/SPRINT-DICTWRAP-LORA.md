# SPRINT: Dict-Wrap Specialist LoRA (first real training run)

**Status:** DRAFT spec — design only. The harvest, the train, and the eval are EACH a separate go.
Nothing here is authorized to execute by its mere existence.
**Premise (earned, not assumed):** the breadth sprint (PR #41,
`audits/20260613T060000Z-harvest-breadth-verification.md`) measured the model's live failure
distribution as **~88–98% `dict_wrap_unwrap`** under grammar-constrained tool calling — the other
transitions barely occur (fuzzy-rename is grammar-impossible). So the highest-value fine-tune is a
**dict-wrap specialist**: train on the actual dominant failure, not a synthetically-balanced set
that would be balanced against reality (the Goodhart trap from the other direction).

## The thesis the whole sprint defends

A LoRA that improves the gym number but you can't prove improves PRODUCTION is a Goodhart trophy.
The win condition is **fewer dict-wrap repairs in live tool calling on shapes the corpus did NOT
drill**, with **zero regression on shapes that already work**. The held-out eval (Phase 3) is the
deliverable; the train is plumbing to feed it.

---

## Phase 0 — Feasibility + provisioning (survey, HALT until gates pass)

The serving side is ready; the training side is not. Cited from the 4090 (`oara-4090@<gpu>`):

| gate | state | action |
|---|---|---|
| llama.cpp LoRA serving | ✅ `--lora`, `--lora-scaled`, `POST /lora-adapters` hot-swap | none — enables A/B eval |
| training frameworks | ❌ `peft`/`trl`/`datasets`/`bitsandbytes` MISSING (torch 2.11+cu130, transformers 5.5 present) | install (or `unsloth`) in a dedicated venv |
| base weights for training | ❌ only GGUF on disk — **no HF safetensors** | obtain `google/gemma-4-26b-a4b-it` HF weights (~50 GB); GGUF cannot be trained |
| GPU memory | ⚠ 24 GB total, **22.4 GB used by the live server** | training REQUIRES the daemon STOPPED; even then a 26B model needs **QLoRA 4-bit** (bitsandbytes) to fit one 4090 |
| MoE LoRA | ⚠ gemma-4-26B-**A4B is MoE** | confirm peft/trl target modules — prefer **attention-proj LoRA** (q/k/v/o), avoid expert-MLP LoRA unless verified; flag as the top technical risk |
| GGUF conversion | — | `llama.cpp/convert_lora_to_gguf.py` exists; verify it round-trips a PEFT adapter for this arch |

**HALT after Phase 0** with the provisioning checklist. Do not download 50 GB or install frameworks
without the owner's go — those are real disk/time commitments. The train cannot start until weights
+ frameworks are in place AND a daemon-down training window is scheduled (training takes the GPU
fully; AutoDream/Telegram/Beacon are offline for it).

---

## Phase 1 — Harvest the dict-wrap training set (sized for dict-wrap density)

Recompute from the verified numbers, dict-wrap-ONLY:
- Verify run: **29 dict-wrap pairs / 63 runs = 0.46 dict-wrap pairs/run** (87.9% of the 0.52 overall).
- Target a **DPO set of ~1,000 dict-wrap preference pairs** → **~2,175 distinct dict-wrap tasks** at
  0.46/run (re-derive from a 100-run pilot; don't trust the point estimate).
- The dict-wrap pools must cross-product to ≥ ~2,175 distinct tasks — expand `_CMD_VERBS/_CMD_ARGS/
  _GREP_PATTERNS/_GREP_PATHS` in `gym/corpus.py` ~5× (the overnight runbook already specifies this).
- **Reserve the held-out eval shapes BEFORE harvesting.** The training corpus drills
  `task_create.command`, `grep.root`, `task_list.status`, `task_create.prompt`. The held-out eval
  (Phase 3) MUST use dict-wrap shapes the training set never saw — different wrap-prone params/tools
  from the adapter_unwrap set (`sessions_list`, `download_file`, `browser`, `cron_create`, …) AND a
  disjoint noun pool. Generate the held-out taskset from a SEPARATE seed and never harvest it into
  training. This split is the Goodhart guard; getting it wrong invalidates the whole result.
- Pairs are already DPO-shaped (`{prompt, chosen, rejected}`); export via
  `export_training_pairs.py --db <gym-db> --out dictwrap-dpo.jsonl`. Chosen = unwrapped call,
  rejected = the model's dict-wrapped call. Filter to `dict_wrap_unwrap` via the taxonomy classifier
  so no off-target transitions leak in.

Harvest per `HARVEST-OVERNIGHT-RUNBOOK.md` (daemon down, detached sentinel driver). Post-flight: the
transition histogram must be ~100% dict_wrap after the classifier filter; cite the count.

---

## Phase 2 — Train (QLoRA DPO, daemon down)

- **Method: DPO LoRA** (the pairs are preferences: prefer unwrapped over wrapped — exactly "stop
  emitting the nested form"). Not SFT.
- **Quantization: QLoRA 4-bit** base (bitsandbytes) — the only way a 26B fits one 24 GB 4090 for
  training. Daemon STOPPED for the window.
- **Target modules: attention projections** (q/k/v/o) only, to sidestep MoE-expert-LoRA risk;
  revisit experts only if attention-only underfits.
- Config to pin in the executing session (starting points, sweep if the eval underperforms): rank
  16–32, alpha = 2×rank, lr 1e-4 to 2e-4, 1–3 epochs, DPO β 0.1, effective batch via grad-accum to
  fit memory. Hold out 10% of the DPO pairs as a train-time val split (distinct from the Phase-3
  live eval).
- **Convert** the PEFT adapter → GGUF (`convert_lora_to_gguf.py`), verify it loads:
  `llama-server … --lora dictwrap.gguf` boots and answers a trivial turn.

---

## Phase 3 — Held-out eval (THE deliverable — the whole ballgame)

A/B the SAME server with the adapter applied vs not (hot-swap via `POST /lora-adapters`, or two
boots). Run BOTH against two tasksets via the gym harness:

1. **Held-out dict-wrap taskset** (shapes the corpus never drilled — reserved in Phase 1). Metric:
   **dict-wrap repair RATE = dict-wrap pairs captured / runs** (the model wrapping LESS means the
   adapter has to repair LESS means the fine-tune taught the unwrapped form).
   - **PASS: held-out dict-wrap repair rate drops materially** (propose ≥40% relative reduction vs
     base; set the exact bar from the base-run measurement, pre-registered before seeing the LoRA
     number). A drop on HELD-OUT shapes = genuine generalization, not noun-memorization.
2. **Control taskset** (the v1 currently-passing shapes — bash/read_file/write_file/grep success
   cases). Metric: **task pass rate**.
   - **PASS: no regression** (control pass rate within noise of base; propose ≥ base − 2 pts).

Both gates must pass. Report: base vs LoRA on held-out dict-wrap rate AND control pass rate, with
run counts and both runs (flakiness is signal). A LoRA that drops dict-wrap ONLY on trained nouns,
or that regresses control, is a FAIL even at a great gym number — say so plainly.

Secondary (report, not gate): overall named-tool success on the full v1 set; latency delta from the
adapter; whether `dict_wrap` drop transfers to the OTHER (rare) transitions or is dict-wrap-specific.

---

## Out of scope / DO-NOT
- Synthetic-injection balancing (shelved — the finding argues against it; reach for it only if a
  FUTURE distribution measurement shows a genuinely multi-modal failure profile).
- Training on or peeking at the held-out eval shapes (invalidates the result).
- Expert-MLP LoRA on the MoE unless attention-only is shown to underfit.
- Shipping the LoRA to the live default server without the Phase-3 gates green — the eval decides
  whether it ever serves production traffic.
