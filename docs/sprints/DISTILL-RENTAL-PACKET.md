# Phase 5 rental packet: the large student on Modal (2026-10-30)

> **Status: not launched. Requires the operator. Max-derived data never rents
> (DISTILL-OPEN-QUESTIONS #6, Q6b ruling).** Nothing in this packet has run. No Modal account
> was used and no money was spent.
>
> - **Q6b (operator, 2026-10-30):** Max-derived traces may train a private, internal-only
>   model, on our own machines (`train_qlora.py` under build.py's recorded allow). Renting
>   uploads the training data to Modal, off our machines, so `modal_train.py` refuses
>   Max-derived data at every step: `pretokenize`, `plan`, `rent`, the upload and the GPU
>   call. **There is no override:** no flag, and no approval, lets it rent a Max-derived row.
>   `--i-have-operator-approval-for-max-derived-data` is the local trainer's flag;
>   `modal_train.py` has none. Renting Max-derived data would need a new operator decision
>   and a code change.
> - **A rented run is possible only on clean, gym-only data:** train/val rows from the local
>   27B teacher's real-pattern gym runs (`source = qwen38-27b-gym`, `max_derived: false`),
>   built without `--allow-max-derived`, pre-tokenized into a directory of its own that
>   holds nothing `publish_guard.py` would refuse (not the dataset's Max-derived eval-tools
>   set). Anything that does not show it is clean is refused (fail closed).
> - Launching uploads that data to Modal. Ground rule 2 lets data leave the machines only for
>   judge calls, so the launch decision covers that too.

## Recommendation

- **GPU: H100 (`--gpu H100`).** Likely **~31 h and ~$140** for a 5–10K-example run, in 2
  calls. On paper the B200 is cheapest per run (likely $129), but its utilization is the
  least certain assumption in the table: at 15% instead of 18% it drops to third. The
  H100 is the most proven stack for 4-bit QLoRA, costs within 10% of the B200, and Modal
  may run it on an H200 at the H100 price.
- **Cap: `--max-usd 200`, `--max-hours 48`.** That is ~40% headroom over the likely case.
  The high case (~$382, ~85 h) is caught by the 50-step calibration. If the measured speed
  says the whole run would pass the cap, the run checkpoints and stops after spending only
  the calibration: ~$5–8 on an H100 (50 steps of ~88K tokens, plus the model load). A
  relaunch with the same cap stops again before any GPU is started; to go on, raise
  `--max-usd`.
- **Path: 4-bit QLoRA on `unsloth/Qwen3.8-27B-unsloth-bnb-4bit`**, the checkpoint Unsloth's
  Qwen3.8 guide recommends for QLoRA. Settings match the 9B run: r=32/alpha=32, lr 1e-4
  cosine, 2 epochs, max_seq 16,384 with step-down to 12,288 and then 8,192, gradient
  accumulation 8, and a checkpoint every 200 steps.
- **The model is the bake-off winner.** Phase 4 has not run yet. If Muse Glimmer 30B wins,
  pass `--base-model` with its repo. Unsloth publishes no memory data for it; the in-run
  OOM step-down covers a miss.

## Cost per run, ranked

Ranked by the likely cost of the whole run, not price per hour. These are estimates until
the calibration measures the real speed, and the script re-computes the exact cost from
that measurement before going on.

| Rank | GPU (Modal `gpu=`) | $/h all-in | tok/s low / likely / high | Hours likely | Cost per run low / likely / high | Fits 16k (4-bit) |
|---:|---|---:|---|---:|---|---|
| 1 | B200 180 GB (`B200`) | $6.82 | 1,500 / 2,500 / 3,500 | 18.9 | $48 / **$129** / $355 | yes |
| 2 | H100 80 GB (`H100`) | $4.52 | 916 / 1,527 / 2,138 | 30.9 | $52 / **$140** / $382 | yes |
| 3 | H200 141 GB (`H200`) | $5.11 | 990 / 1,649 / 2,309 | 28.6 | $54 / **$146** / $401 | yes |
| 4 | L40S 48 GB (`L40S`) | $2.52 | 402 / 670 / 939 | 69.5 | $65 / **$176** / $484 | likely |
| 5 | RTX PRO 6000 96 GB (`RTX-PRO-6000`) | $3.60 | 463 / 772 / 1,080 | 60.6 | $80 / **$218** / $600 | yes |
| 6 | A100 80 GB (`A100-80GB`) | $3.07 | 347 / 578 / 809 | 80.8 | $92 / **$248** / $683 | yes |

| GPU | Hours low / likely / high | Calls (likely) | $ per million tokens (likely) |
|---|---|---:|---:|
| B200 | 7.1 / 18.9 / 52.0 | 1 | 0.76 |
| H100 | 11.4 / 30.9 / 84.5 | 2 | 0.82 |
| H200 | 10.6 / 28.6 / 78.4 | 2 | 0.86 |
| L40S | 25.8 / 69.5 / 192.0 | 3 | 1.05 |
| RTX PRO 6000 | 22.2 / 60.6 / 166.4 | 3 | 1.30 |
| A100 80 GB | 29.8 / 80.8 / 222.5 | 4 | 1.48 |

**The corpus we have today can never be rented.** The ~1,900 train examples Phase 1 found
are the hosted Max teacher's traces: Max-derived, so they stay on our machines whatever the
cost (Q6b). (Had it been clean: about 42M tokens at the likely speed, H100 ~8 h / ~$36,
B200 ~5 h / ~$34, L40S ~18 h / ~$45.) A rented run's size is whatever clean, gym-only data
the local 27B teacher generates, and Phase 3's 5K-example gate applies to that data alone.

The tables come from the code. `python3 scripts/distill/modal_train.py estimate` prints
them, and a test keeps this file in step with it. After pre-tokenizing,
`estimate --pretokenized DIR` uses the real token count.

### Assumptions (stated so the calibration can prove them wrong)

- **Run size** comes from the Phase 1 audit and the Phase 3 gate. A run happens only after
  Phase 3 passes its 5–10K train-size gate on clean, gym-only data (the status note above);
  the token sizes below were measured on the Max traces and stand in for gym rows of the
  same shape.

  | Case | Examples | Mean tokens per example | Epochs | Run tokens |
  |---|---:|---:|---:|---:|
  | low | 5,000 | 8,500 | 2 | 85M |
  | likely | 7,500 | 11,000 | 2 | 165M |
  | high | 10,000 | 13,500 | 2 | 270M |

  - The mean adds two parts. The first is the stored context: Phase 1 measured p50 2,738
    and p90 6,152 tokens. The second is the rebuilt request head (system prompt plus tool
    list): the 26-tool catalog alone is 22,020 characters, about 5.6K tokens at Phase 1's
    3.93 characters per token. Many examples are cut at 16,384.
- **Speed** comes from tokens/s = utilization × dense BF16 tensor peak / (6 × 27e9) FLOPs
  per trained token.
  - The 6N per token covers the forward pass, the gradient-checkpoint recompute and the
    input gradients. The frozen 4-bit base gets no weight gradients, and the LoRA's are
    negligible.
  - Attention's extra cost at 16k is ignored. Only one layer in four uses full attention;
    the rest use linear attention.
  - Peaks without sparsity: B200 2,250, H100/H200 989.5, RTX PRO 6000 ~500, L40S 362.05,
    A100 312 TFLOPS. The RTX PRO 6000 figure is half the sparse 1,000 that sources quote;
    they disagree, so treat it as ~500.
  - Likely utilization: 30% for A100/L40S, 27% for H200 (more memory bandwidth for
    dequantization), 25% for H100/RTX PRO 6000, and 18% for B200 (the least-proven stack
    for this model).
  - Low and high speeds are 0.6× and 1.4× likely.
  - The low cost pairs the fewest tokens with the fastest speed; the high cost pairs the
    most tokens with the slowest.
- **Overheads.**
  - GPU seconds per call that are not training (container start, weights from the Volume,
    compile): 600 / 900 / 1,800 s (low / likely / high).
  - 300 s per call of billed startup our code cannot see.
  - 60 s per checkpoint.
  - A new call every ~23.5 h, because Modal caps one call at 24 h.
- **Billing.**
  - Modal list price per second for the GPU, plus 4 CPU cores ($0.0000131/core/s) and
    48 GiB of memory ($0.00000222/GiB/s). That adds $0.57/h to every row.
  - No region is pinned; pinning one costs 1.15–1.75×.
  - Preemptible (the default); non-preemptible costs 3×.
  - Adapter download egress (~1 GB at $0.04/GiB) is negligible.
  - Checkpoints stay in a Volume at $0.09/GiB-month until the operator deletes them.

## Which GPU fits

| Path | Resident weights | Evidence | 48 GB (L40S) | 80 GB (A100, H100) | 96–180 GB |
|---|---|---|---|---|---|
| **4-bit QLoRA (bnb), default** | ~17–19 GB | Unsloth: "QLoRA works with 24GB" (its examples use max_seq 2,048); Unsloth's own 4-bit 27B GGUF is 17.9 GB | likely at 16k (estimate) | yes | yes |
| NVFP4 checkpoint, packed 4-bit LoRA | 22.2 GB after load | Unsloth PR #12185: 40.15 GB peak (was 72.93 GB) at batch 2 × ≤1,024 tokens, r=16, RTX PRO 6000 | uncertain at 16k | likely | yes (tested on RTX PRO 6000 and B200) |
| 16-bit LoRA | ~54 GB | Unsloth's Qwen3.5 guide: 56 GB for the 27B at 2,048 tokens; Qwen3.8 guide: "LoRA needs >36GB" | no | tight at 16k | yes |

- The 16k figures are estimates. Unsloth publishes no 16k numbers for this model. With
  "unsloth" gradient checkpointing, activations are offloaded to system RAM: about 168 MB
  per layer at 16k for a 5,120-wide model (the shape in Unsloth's PR benchmark). That is
  why the container requests 48 GiB.
- The run's preflight (one forward and backward pass on the longest example) and the
  OOM step-down (16k → 12k → 8k) turn a wrong guess into a recorded shorter length, not a
  failed run.
- **The NVFP4 path** turns on automatically when a compressed-tensors NVFP4 checkpoint is
  loaded (`--base-model unsloth/Qwen3.8-27B-NVFP4`). It saves memory only against a
  16-bit load. The bnb 4-bit path is lighter after load (~17–19 GB vs 22.2 GB), so it
  stays the default, and the 48 GB class is open without NVFP4.
  - The PR tested NVFP4 on T4, A100, L4, RTX PRO 6000 and B200, not H100. Its 27B runs
    were on Blackwell.
  - Merging a LoRA trained this way into 16 bits needs unsloth-zoo PR #1480 (merged).
- **For the 9B student (operator decision):** Unsloth's Qwen3.5 guide says "It is not
  recommended to do QLoRA (4-bit) training on the Qwen3.5 models". The runbook asks for
  QLoRA because a 16-bit LoRA of the 9B (22 GB at 2,048 tokens) cannot reach 16k on the
  4090's 24 GB. `train_qlora.py` defaults to 4-bit as the runbook says and offers
  `--load 16bit`. The 27B in this packet is Qwen3.8, where Unsloth recommends 4-bit QLoRA.

## How the run works (`scripts/distill/modal_train.py`)

1. **Locally, offline.**
   - `pretokenize` tokenizes `train.jsonl` with the segment recipe (`render.py`) and checks
     every row's loss spans.
   - It also re-fits each long row to 12k and 8k with the chat template and stores those
     variants, so a step-down needs no template or tokenizer on paid time.
   - `plan` prints the cost for one GPU and cap.
2. **Guards before any spend.** The script refuses when:
   - Any row is Max-derived, with no override (Q6b). `pretokenize` counts a row as
     Max-derived unless something shows otherwise: build.py's `report.json` naming a
     non-Max teacher, or a builder-written `max_derived: false`. A row's `source` alone
     does not clear it, since build.py labels teacher rows `extra` as readily as
     `max-golden`. A path carrying the `max-derived` tag is refused too.
   - The pre-tokenized directory's count of Max-derived rows is missing or not 0, its
     name or manifest says Max-derived, its provenance chain does not show it is clean,
     or it holds anything `publish_guard.py` would refuse (a dataset's Max-derived
     eval-tools rows beside the pre-tokenized files, a stray JSONL, a symbolic link).
     `plan`, `rent`, the upload and the GPU call each check again.
   - `--max-usd` is missing.
   - The optimistic estimate is already over the cap.
   - Without `--launch` it only prints.
3. **Upload** the pre-tokenized directory to the `distill-data` Volume. This is skipped
   when the same content hash is already there.
4. **Weights** download into the `distill-models` Volume on a CPU-only function, pinned to
   the commit `--base-revision` resolves to (`main` at launch if omitted, and recorded in
   the run's ledger). A relaunch with `--run-id` reuses the recorded commit, even if `main`
   has moved since. No GPU is billed for downloads.
5. **One GPU call.**
   - It runs `train_qlora.py`'s trainer (shipped in the image) on the weights in the models
     Volume, and resumes from the newest checkpoint, if any.
   - On the first call it runs the **50-step calibration**: real training steps, kept when
     the run goes on.
   - It times steps 11–50 for tokens/s and computes the **exact remaining cost**. That is
     spent so far + remaining tokens / measured speed + checkpoints + the load of each
     further call + egress.
   - It continues only if the total is ≤ `--max-usd` and the run ends before
     `--max-hours`. Otherwise it checkpoints and returns `over_cap` or `over_time`.
   - Every later call (the next one of a chain, or a relaunch) re-projects the rest of the
     run from the measured speed against its own `--max-usd` and `--max-hours` before it
     trains, so a run that stopped `over_cap` never trains on to the hard cap.
6. **Training.**
   - Checkpoints go to the `distill-runs` Volume every 200 steps, committed each time.
     Each one records a fingerprint of the weights, and every resume checks that the
     restore brought them back.
   - At Modal's 24 h call limit the call checkpoints and returns, and the local side
     starts the next call from that checkpoint.
   - A CUDA OOM records the step-down and returns, and the next call resumes at the
     shorter length. Training containers are single-use, so that call gets a fresh
     container and CUDA context.
7. **Download** the adapter, `ledger.json`, `run_state.json`, `metrics.jsonl` and
   `manifest.json` to `data/distill/rental/<run-id>/`. The GPU was released when its call
   returned.

**Hard limits in code**

- **Dollars.** A ledger on the Volume records every call's billed time at the GPU + CPU +
  memory rate.
  - A call that never closed (preempted, killed) is charged an extra 5 minutes past its
    last heartbeat. Every call is charged 300 s of unseen startup.
  - After every step the run stops if spent + the stop's checkpoint + 1.5× the last step
    would reach `--max-usd`. A step starts only if it can finish under the cap at up to
    1.5× the last step's time.
  - Before a call loads anything, the same check also reserves the load and the first
    step (as calibrated on that GPU, else the worst-case load and a slow-case step), and
    it runs again after the load. Before each call, the local side reads the ledger and
    does not start a container that could not reach its first step under the cap.
- **One call per run.** A call that finds another call's segment still live in the ledger
  (heard from within 15 minutes, or 45 while loading) returns `busy` before loading.
  Two calls on one run would each budget against a stale view of the other's spend and
  overwrite each other's ledger and checkpoints.
- **Time.** The run stops 10 minutes before `--max-hours` from launch, and before each
  call's own timeout. Modal's `timeout` (at most 24 h) is the backstop kill.
- **No idle GPU.**
  - Uploads, downloads and tokenization never run on the GPU.
  - Stops return at once, `scaledown_window=2`, and nothing is kept warm.
  - If the local client goes away, the in-flight call still ends at its own limits;
    `--detach` keeps it from being cancelled half-way.

**Image.** The image is `unsloth/unsloth:core-v0.1.910-beta`: the official image's
training-only variant (no Studio), with PyTorch 2.11 + CUDA 12.8 and installers pinned to
`unsloth>=2026.10.7`. It was released 2026-10-23, a week before this packet. When this
packet was written the tag pointed at digest
`sha256:8703818b64f3541f0b93b947c31702b84d39e23746a64d1c8702e3fda56ed73a`, and Modal
pulls it by that digest, so a re-pushed tag cannot change the environment. The run's
manifest also records the installed package versions. The image's own entrypoint is
cleared. The local 27B smoke should use the same image (`docker pull
unsloth/unsloth@sha256:8703818b64f3541f0b93b947c31702b84d39e23746a64d1c8702e3fda56ed73a`)
so local and rented runs share one environment.

**Unsloth's recent features, and how this packet uses them**

- **Training memory estimates (2026-09-17)** and **token preview (v0.1.909-beta)** are
  Unsloth Studio features with no documented script API. They are covered here by
  `train_qlora.py --dry-run`, which prints two examples with masked and trained spans
  marked and re-checks every row's spans with the tokenizer. The preflight probe and the
  calibration cover memory and speed on the real GPU.
- **Resume with accurate remaining-time estimates (2026-09-22).** The trainer owns exact
  resume (adapter, optimizer, RNG and plan position). The remaining time is the
  calibration's projection, recorded in `ledger.json`.
- **NVFP4 / INT4 / MXFP4 lower-memory LoRA (v0.1.902-beta):** an option (see above), not
  the default.
- **`unsloth eval` (v0.1.902-beta; vLLM backend in v0.1.910-beta)** can sanity-check the
  downloaded adapter. The Phase 6 gate itself is `score.py` on the quantized model.

## Before anyone launches

1. The training data is clean and gym-only (DISTILL-OPEN-QUESTIONS #6, Q6b ruling: Max-derived
   data trains only on our own machines). That means a build.py run without
   `--allow-max-derived`, whose train/val hold only the local 27B teacher's `qwen38-27b-gym`
   rows (report: `data_policy` gate PASS, 0 Max-derived rows in train/val,
   `settings.teacher_model` the local 27B, `max_derived: false` in `dataset-manifest.json`),
   pre-tokenized into a directory of its own (never beside the dataset's eval-tools set,
   which is built from Max traces). Check it first with
   `python3 scripts/distill/publish_guard.py check <pretok dir>`: what it refuses, the
   rental refuses.
2. Phase 3 passes its gates with ≥5K train examples on that gym-only data alone. Today's
   Max corpus (~1.9K examples) does not count toward it: it can never be rented.
3. Phase 4 picks the bake-off winner. The dataset is re-rendered with the winner's own
   tokenizer and template: `build.py build --tokenizer-dir <winner tokenizer>`. The 9B's
   counts are close but not exact.
4. On the 4090, inside a window. Both jobs hold the GPU, so both stop before the window's
   restore (`--stop-at` 06:35 ET with that morning's UTC offset, not the 07:00 window end)
   and run where a `setups.py apply business` restore can stop them:
   - The feasibility probe runs only as the tower job runner's `qwen38-27b-feasibility` step
     (`scripts/distill/tower/job_runner.py`, in the window's detached unit after the
     rehearsal gate and `apply flywheel`): `feasibility_27b.py --execute` once per length,
     with `--out` per length, the staged 4-bit checkpoint as `--base-model`, and
     `HF_HUB_OFFLINE=1`. It starts a length only when its worst case ends before `--stop-at`
     and the GPU is free, and runs each length in a `distill-feas27b-*` systemd scope. If
     ≥8k fits and the full run takes ≤5 days, the operator chooses CONTINUOUS / NIGHTLY / NO,
     and the run may not need renting at all. An `incomplete` verdict means a length ≥8k was
     not measured: probe it again.
   - A tiny 27B smoke, detached in a `distill-train-*` unit: `scripts/distill/launch_detached.sh
     --execute -- --smoke --base-model <staged 4-bit checkpoint dir> --max-seq 8192
     --stop-at <06:35 ET> ...`, with CPU offload. It must load, train, checkpoint,
     restore and save the adapter (manifest status `smoke-train-pass`). A `stopped`
     status is no verdict: launch it again with `--resume`. The eval pipeline (merge,
     quantize, score) then runs on the smoke adapter.
5. `python3 scripts/distill/modal_train.py pretokenize --data <27B train.jsonl>
   --tokenizer-dir <winner tokenizer> --out data/distill/pretok/train-27b`, then `plan`.
6. The operator has a Modal account and a token (`modal token new`). The base repos are
   public, so no Hugging Face token is needed.

## Exact launch command (the operator, or on "RENT YES $<cap>")

```
modal run --detach scripts/distill/modal_train.py::rent --data data/distill/pretok/train-27b --gpu H100 --max-usd <cap> --max-hours 48 --launch
```

- **`<cap>` is the dollar figure from "RENT YES $<cap>"** (the recommendation above is
  200). Never a larger one.
- **This line refuses before anything is uploaded (exit 3) whenever the data holds a
  Max-derived row, or nothing shows that it does not.** Nothing on the line overrides that
  (Q6b): `modal_train.py` takes no approval flag (the trainer's
  `--i-have-operator-approval-for-max-derived-data` is not one of its options). Only clean,
  gym-only data (Before anyone launches, step 1) gets past it.
- "RENT YES" answers the spend question only. It is not an answer to the ground-rule-2
  question of sending the (clean) training data to Modal, and it never makes Max-derived
  data rentable.
- For a different GPU, change `--gpu` and `--max-usd` (the table above).
- Without `--launch`, the same line prints the plan and spends nothing.
- **If the client went away**, the call in flight keeps running (`--detach`) up to its
  own limits. Wait until it has ended (`modal app list` shows the app stopped) before
  relaunching the same line with `--run-id <id>`; while it runs, a relaunch stops with
  `busy` (exit 6). A relaunch reuses the base-model revision the run started on.
- After `over_cap` or `over_time`, relaunching the same line stops again: raise
  `--max-usd` or `--max-hours` to go on.
- Fetch a finished adapter with
  `modal run scripts/distill/modal_train.py::fetch --run-id <id>`.

## Sources

Read 2026-10-30.

- Modal pricing (per-second GPU, CPU, memory, Volume, egress, region and non-preemptible
  multipliers): https://modal.com/pricing
- Modal GPU types and `gpu=` strings (H100 may run on H200 at the same price; L40S 48 GB;
  H200 141 GB): https://modal.com/docs/guide/gpu
- Modal timeouts (1 s to 24 h per call): https://modal.com/docs/guide/timeouts
- Modal `Function.with_options`: https://modal.com/docs/reference/modal.Function
- Modal `App.function` (`single_use_containers`; `max_inputs` is deprecated for it):
  https://modal.com/docs/reference/modal.App
- Modal preemption (a preempted input restarts on the same input):
  https://modal.com/docs/guide/preemption; `modal.current_function_call_id`:
  https://modal.com/docs/reference/modal.current_function_call_id
- Modal `Volume` (batch_upload, read_file, commit): https://modal.com/docs/reference/modal.Volume
- Unsloth Qwen3.8 training guide (QLoRA with 24 GB, LoRA >36 GB, the bnb-4bit checkpoint):
  https://unsloth.ai/docs/models/qwen3.8/train
- Unsloth Qwen3.5 fine-tuning guide (QLoRA not recommended for Qwen3.5; 16-bit LoRA 9B
  22 GB, 27B 56 GB): https://unsloth.ai/docs/models/qwen3.5/fine-tune
- Unsloth changelog (NVFP4 LoRA 40.2 vs 72.9 GB, `unsloth eval`, memory estimates, resume
  estimates, token preview): https://unsloth.ai/docs/new/changelog
- Unsloth PR #12185 (packed NVFP4 LoRA benchmark, GPUs tested, enable/disable switches):
  https://github.com/unslothai/unsloth/pull/12185
- Unsloth NVFP4 page (NVFP4 inference needs Blackwell): https://unsloth.ai/docs/de/grundlagen/nvfp4.md
- Unsloth v0.1.910-beta release notes (pins, eval vLLM backend):
  https://github.com/unslothai/unsloth/releases/tag/v0.1.910-beta
- Unsloth v0.1.800-beta (Qwen3.8-27B support, NVFP4 uploads):
  https://github.com/unslothai/unsloth/releases/tag/v0.1.800-beta
- Unsloth Docker guide and image variants (core vs studio, PyTorch 2.11 + CUDA 12.8):
  https://unsloth.ai/docs/get-started/install/docker, https://hub.docker.com/r/unsloth/unsloth
- `unsloth/unsloth` tags and digests:
  https://hub.docker.com/v2/repositories/unsloth/unsloth/tags?page_size=25&ordering=last_updated
- NVIDIA peaks: L40S https://www.nvidia.com/en-us/data-center/l40s/ (362.05 dense),
  H100 https://www.nvidia.com/en-us/data-center/h100/ (1,979 with sparsity),
  H200 https://www.nvidia.com/en-us/data-center/h200/ (1,979 with sparsity),
  A100 https://www.nvidia.com/en-us/data-center/a100/ (312 dense),
  HGX B200 https://www.nvidia.com/en-us/data-center/hgx/ (36 PFLOPS sparse per 8 GPUs; dense
  is half), RTX PRO 6000 Blackwell
  https://flopper.io/gpu/nvidia-rtx-pro-6000-blackwell-server-edition (1,000 BF16, sparse
  per most sources)
- Third-party Qwen3.8-27B fine-tuning notes (48 GB "much easier" for QLoRA):
  https://www.yottalabs.ai/post/how-to-fine-tune-qwen-3-8-27b-with-unsloth-2026
