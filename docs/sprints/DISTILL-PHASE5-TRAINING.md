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

## Round 2 training across nightly windows

Built 2026-11-02, not run. The operator ruled that the round-2 SFT starts once the round-2 dataset is built (about Nov 9 or later). It is a stepping stone: its adapter is the starting point for GRPO. The 4090 is now ours only in its nightly window, which gives about 5 h of GPU time: window_start runs at 01:20 ET, the job runner stops at 06:35, and window_end runs at 06:40. So the run is trained across several nights.

### How a night works

- **Where it runs.** The job runner has a new step, `train-nights` (`DJR_NIGHTS=1`, `DJR_ONLY=train-nights`). window_start starts the runner each night as `distill-jobs-<stamp>.service`. The job's state machine is `scripts/distill/tower/train_nights.py`, beside `job_runner.py`. State is in `nights.json`, in the step's directory. The phases are train → post-train → done (or failed).
- **Resume.** Every training night runs the same `train_qlora.py` command with `--resume`. The trainer already brought back the plan (one seeded shuffle per epoch, `distill:<seed>:<epoch>`), the position in the plan, the optimizer, the LR schedule and every RNG. New: a held-out measurement now restores the training RNG afterwards. So the eval cadence and the resume points cannot change training.
- **Verified on the Mac.** The run was cut five ways: stop-at, SIGTERM inside an accumulation group, SIGTERM during an eval, a crash that loses the steps after the last checkpoint, and SIGTERM during the epoch-end eval. Concatenated, the cut runs train the same rows in the same order, with the same learning rates, and end on the same weights as an uninterrupted run. A second check runs the real job runner on the real trainer loop with a fake backend, across three nights in three processes. The same order holds.
- **Checkpoints.** A light checkpoint is written every 20 min (`--save-every-minutes 20`), so an unclean end loses at most about 20 min. The held-out val and probe sets are measured only every `--eval-every-minutes` (90) and at every epoch end. Retention keeps the newest 2 plus every epoch end (`--keep-epoch-checkpoints`). The job re-checks this every night.
- **Stopping.**
  - At stop-at, the trainer checkpoints and exits 2 min before the runner's 06:35.
  - At window_end, SIGTERM goes to the runner only (KillMode=mixed). The runner sends SIGTERM to the trainer's group, with `DJR_NIGHTS_GRACE_S` 300, which is under window.conf's `JOB_STOP_TIMEOUT_S` of 420. The trainer drops the group in progress, abandons any held-out measurement (an epoch-end one is measured right after the next resume), checkpoints and exits.
  - So the trainer is gone before the restore. The scope's TimeoutStopSec (900) is the backstop if a restore stops it directly.
- **Gate.** When training is done, `training_verdict` checks the run, using the same rule as post_train.py. It needs status done, the adapter on disk, every epoch recorded, finite losses, and val and fixed-probe loss both lower than at step 0. The done manifest also carries `gate`.
- **Disk.** The shared disk preflight uses the job's own need: the checkpoints the run still has to write (newest 2, one being written, and the epoch ends), plus the adapter and logs.
- **Telegram.** One message per night, at most 5 lines: the steps done out of the total, tonight's rate and why it stopped, the val and probe curve, an ETA in nights from the measured tokens per night, and the checkpoints and disk.

### Post-train across windows

- **Hold.** After the gate passes, each following window is held for the existing post-train pipeline:
  - The runner writes `hold.json` and keeps business down with the GPU free.
  - `train_nights.py mini-night` runs on the 3090 Ti host shortly after 01:20. It arms `post_train.py --window-hold`, with `arm --rerun` from the second night on, and a stop-at 15 min before the hold ends.
  - The pipeline skips wait-train and takes the GPU from the hold, without stopping business itself. When it ends it writes `hold-released-<night>.json` and, once completed, `post-train.done.json`.
  - Business always comes back through the window machinery (the watcher or window_end), never through `train_end.sh` in this mode.
  - A hold that nobody takes within 45 min ends, and the window closes early.
- **Done.** Once the job is done (or parked), the runner's dry run fails when nothing else is queued. window_start then skips the swap and reports it, until the job is disarmed.
- **Restarting mid-step.** Every stage restarts idempotently if a window ends in the middle of it:

| Stage | If a window ends mid-step |
|---|---|
| Merge | `merged.partial`, then rename: restarts from scratch (minutes, CPU, on the 3090 Ti host) |
| Push and fetch | rsync plus sha256 checks: resume |
| Convert, imatrix, quantize | `quantize.py` redoes only a `.partial` output and exits 75 at its stop-at |
| Scoring | `score.py` skips cases already answered |

  A rerun keeps every finished step. It now also resets the start count, so each night gets the full crash-restart budget.

### Estimate

Measured QLoRA speeds on the 4090, with activation offload: 2,549 tok/s at 16K, 2,330 at 32K and 1,990 at 49,152. Round 1's wall overhead was 5.8% (2,401 against 2,549). A night has about 5.1 h of training (`train_nights.py estimate`).

| Case | New rows | Long-context rows (26%) | Other new rows | Replay (40% of the final set) | Tokens over 2 epochs | Nights |
|---|---|---|---|---|---|---|
| Central | 3,750 | 36K | 18K | 2,500 rows at 13.8K (v4b's measured mean) | 239M | **5.7** |
| Low | 3,500 | 30K | 15K | as central | 197M | 4.6 |
| High | 4,000 | 40K | 20K | as central | 275M | 6.6 |

- **Assumptions.** No round-2 dry-run report exists yet, so the row lengths are assumed.
- **If the 3,500–4,000 rows include the replay:** about 3.4 nights.
- **Post-train:** another 1–2 nights (Phase 6, then about 5 h of scoring).
- **Refresh:** when the dataset is built, run `train_nights.py estimate --dataset <train.jsonl>`.
