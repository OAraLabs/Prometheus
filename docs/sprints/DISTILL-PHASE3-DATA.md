# Phase 3: datasets

## v4 (first full run), 2026-10-31

**Gates: all PASS except the spot check, which can't run yet (overall INCOMPLETE, not FAIL).** Built on the 3090 Ti
host into `~/distill/phase3/build-v4-max-derived`. It is private and tagged `max-derived` (Q6b); the publish and
rental tools refuse it. The operator's 2026-10-31 rulings apply to this first full 9B run only:

- **Coding cap 60%.** Real use is mostly coding, and 60% is the lowest of 35/50/60/70% that reaches 3,500 train
  rows. Every other category stays at ≤35%.
- **Size floor 3,500.** The default is still 5,000, and later runs keep growing toward it.

Both are recorded in `report.json` with their reasons (`--category-cap-for coding=0.6`, `--min-train-rows 3500`).
Manifest sha256 `a57b0f0ba1dd1af978face6870ded830d20f3bba8d212af664c2be3be41d59c3`; `train.jsonl` sha256
`3aed4e21d6da511a548d58b1ab5089f3b393dfd067b5c12e8232392e8083e46d`. Aggregates only; no trace text left the host.

| Gate | Result | Value |
|---|---|---|
| Leakage (task, group, prefix, exact, prompt+call, shared context, source task) | **PASS** | 0 overlaps between any two splits |
| Verdicts (every verdict row applied) | **PASS** | 4,653 loaded, 0 unmatched or unusable |
| Data policy (Q6b allow covers every Max row in train/val) | **PASS** | 3,678 of 3,678 covered; 1 session excluded |
| Train size ≥ 3,500 (first-run floor) | **PASS** | 3,597 rows |
| Val share 2–5% | **PASS** | 3.1% (114 rows) |
| eval-tools ≥ 200 | **PASS** | 402 rows (26 hard cases) |
| Category caps (coding ≤60%, others ≤35%) | **PASS** | coding 60.0%, other 16.7%, daemon 13.7%, chat 7.9%, memory 1.7% |
| Class caps (reported_blocked ≤15%, no-tool ≤20%) | **PASS** | 12.6% / 3.1% |
| Chat-mode / error-recovery / compacted present | **PASS** | 284 / 517 / 167 |
| Byte-for-byte spot check vs llama-server | NOT RUN | needs a server with the student GGUF |

### Coding-cap table (one split, `--cap-scan`)

Pool: 8,323 rows (Max golden 7,501, Max gym 787, local 27B 35 after dedupe); coding is 80% of it.

| Coding cap | Train | Val | Coding / other / daemon / chat / memory % | Chat / err-rec / compacted | Binding caps |
|---|---:|---:|---|---|---|
| 35% | 2,064 | 63 | 35.0 / 25.7 / 23.1 / 13.3 / 2.9 | 274 / 270 / 129 | coding 35%; reported_blocked 15% (130 dropped) |
| 50% | 2,840 | 88 | 50.0 / 20.7 / 17.2 / 10.0 / 2.2 | 284 / 386 / 152 | coding 50%; reported_blocked 15% (26 dropped) |
| **60%** | **3,597** | **114** | 60.0 / 16.7 / 13.7 / 7.9 / 1.7 | 284 / 517 / 167 | coding 60% only |
| 70% | 4,796 | 154 | 70.0 / 12.5 / 10.3 / 5.9 / 1.3 | 284 / 737 / 167 | coding 70% only |

Only the coding cap binds from 60% up; every non-coding row that passes the gates is already in train. The 35% cap on
the other categories never binds (the daemon is 10–23% at every cap). The reported_blocked class cap binds only at 35%
and 50%.

### Max vs local-27B share (Q6b)

| Split | Rows | Max golden | Max gym (Token Plan teacher) | Local 27B gym | Max total |
|---|---:|---:|---:|---:|---:|
| train | 3,597 | 2,868 (79.7%) | 696 (19.3%) | 33 (0.9%) | 99.1% |
| val | 114 | 102 | 12 | 0 | 100% |
| eval-tools | 402 | 295 + 26 failed-call hard cases | 79 | 2 | 99.5% |

### Unjudged Max traces until the Nov 16 reset

- **Re-judge status:** the Phase 2 Max re-judge covered 4,353 of the 9,895 queued pass rows before the Token Plan
  quota ran out (3,463 pass, 890 fail, 12 unparseable). **5,542 are unjudged** until the quota resets on 2026-11-16
  (16:00 UTC). v3 used the 07:50 snapshot of 2,874 verdicts; v4 uses all of them.
- **A judged fail** is excluded (909 Max golden rows, the Phase 2 sample included).
- **An unjudged Max golden row** keeps its harvested `tool-success` label (the tool call didn't error) and trains under
  the Q6b allow. That's 1,895 of the 3,597 train rows (52.7%); 973 Max golden rows train with a judge pass.
- **Max gym and local-27B rows** carry their own strict verification record (tool success, judge, sandbox, no side
  effects), so none of them rely on the fallback.
- **After the reset:** resume the re-judge, then rebuild. About 20% of the remaining fallback rows would be expected to
  fail and drop out, so the next build needs more non-coding rows to keep train at or above 3,500 at the same cap.

### Inputs (3090 Ti host)

- v3's inputs, plus:
  - the local-27B gym rows to 10:46 (`inputs-v4-max-derived/qwen38-27b-gym.traces.jsonl`, 52 rows);
  - the full re-judge verdict file (`rejudge-verdicts.snap-v4.jsonl`, 4,365 rows).
- Code: `~/distill/phase3/code-v4` (`build.py` at d594237).
- The first Max teacher run's 15 rows (from before the keep-rate fixes) stay out, as in v3.

## Earlier: dry run, 2026-10-30

**Gate: FAIL.** Run as a dry run on the 3090 Ti host with the existing labels only, because Phase 2 re-judging
and Phase 2b teacher expansion are skipped (OPEN-QUESTIONS #6). Training on this data is also on hold under #6.
Everything below is aggregates; no trace text left the host.

| Gate | Result | Value |
|---|---|---|
| Leakage (task, group, context-prefix, exact, prompt+call, shared context) | **PASS** | 0 overlaps between any two splits |
| Train size 5–10K | **FAIL** | 1,323 rows |
| Val share ~3% | PASS | 3.3% |
| eval-tools size ≥ 200 | PASS | 329 rows (26 hard cases from rebuilt failed calls) |
| Category cap ≤ 35% | PASS | coding 35.0%, daemon 35.0% |
| Chat-mode / error-recovery / compacted present | **FAIL** | 0 / 243 / 4 |
| Byte-for-byte spot check vs llama-server | NOT RUN | needs a server with the student GGUF (GPU blocked) |

### How the data was built

- **Inputs:** 11,152 harvested traces; 524 non-Max excluded, 445 exact duplicates excluded. 296 failed Max calls
  were rebuilt from read-only snapshots as hard cases.
- **Full requests rebuilt** the way the live provider sends them: the runtime system prompt plus the tool catalog
  a local run actually sends (the stub catalog, 9,885 tokens of system + tools; the full catalog would be 16,577,
  more than the whole 16k training window). Conversation context comes from a read-only snapshot of the
  conversation store, so earlier tool calls and tool results are back in.
- **Rendered with the student's own chat template** (Qwen3.5-9B, pinned revision), loss on assistant turns only.
  The student's vocabulary and merges are identical to the Qwen3.8 tokenizer used in Phase 1.
- **Rebuilt requests are long:** p50 190K / p90 258K tokens before cutting. 92% exceed 16K, so most training rows are
  left-truncated to 16,384 (system prompt and tools kept); the originals become long-context candidates.
- **Excluded while rebuilding:** 1,384 rows whose target duplicates another after rebuilding, 75 whose parallel
  sibling call failed, 21 with unanswered tool calls, 9 whose target follows an assistant turn, plus a handful with
  tools no longer offered.
- **Split by task**, with grouping by shared conversation context so no held-out turn appears in train context:
  290 tasks → 182 groups. 7 groups were too large to hold out.

| Split | Rows | Tasks | Coding | Daemon | Other | Memory/wiki | Token p50 / p90 |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 1,323 | 254 | 35.0% | 35.0% | 24.3% | 5.7% | 15,463 / 16,241 |
| val | 45 | 7 | 33% | 47% | 4% | 16% | 15,037 / 16,266 |
| eval-tools | 329 | 29 | 79% | 10% | 10% | 1% | 15,714 / 16,299 |

A second dry run on the stored 12-message windows alone (no conversation store) gave 1,549 train rows, the same
gate result, and no error-recovery context at all, so the snapshot rebuild is the better input.

### Eval sets (built on the 3090 Ti host)

| Set | Cases | Notes |
|---|---:|---|
| eval-longctx | 180 | 45 per bucket at 32K / 64K / 128K / 256K (recall + multi-hop), 9 per depth 10/25/50/75/90%; lengths within 0.5% |
| eval-longctx-tools (ungated) | 60 | a tool call needing a value buried in context; kept out of the gated eval-tools run |
| eval-injection | 192 | 3 vectors (tool output, memory, wiki) × 4 styles × 16 attack goals; manifest of hashes for train disjointness |
| sanity | 211 | arithmetic, code, dates, format, knowledge, reasoning, strings, units; deterministic checks |

Sources for long-context prompts: 618 git-tracked repo files and wiki pages, read-only, with untracked files skipped.

**The long-context gate is too noisy at this size (new OPEN-QUESTIONS #8).** With 45 cases per bucket one case is
2.2 points, so the runbook's "no worse than base by > 3 points at any depth" fails an *unchanged* model in at least one
of four buckets about 77% of the time (assuming 10% discordance each way). Fixing it needs either much larger sets
(hours of 256K prefill per model) or a confidence-interval gate. The runbook forbids shrinking eval sets, and
neither fix is applied without the operator.

### What would make the gate pass

1. Phase 2b-style expansion: ~4–9K more verified traces, including chat-mode and compacted-history examples. That
   needs a teacher the plan terms allow (OPEN-QUESTIONS #6), or another source of verified traces.
2. Then re-judging (or the tool-success fallback for what can't be judged), and the spot check once a student GGUF
   server can run.

### Re-running

On the 3090 Ti host: `~/distill/phase3/code/` holds the builder; `build.py pack` then `build --dry-run`. Outputs and
`build_report.json` are under `~/distill/phase3/`. Code: `scripts/distill/build.py`, `render.py`,
`build_eval_sets.py`, `build_fail_set.py`.

## v4b (first full run, after the local 27B judge), 2026-10-31

**Gate: PASS** on every gate.
- The byte-for-byte spot check is the one not run inside the build. It passed separately in the Phase 6 rehearsal: 24/24 rows at text and token level, including chat-mode.
- Coding cap 70%: the operator's rule takes the lowest of 60/70% that keeps train ≥ 3,500, and the judge drops left 3,360 at 60%.

**Local 27B judge (operator Sat ruling).**
- Calibrated against Max's verdicts on 4,651 pass-side items: 85.40% agreement [84.4, 86.4], kappa 0.54 [0.50, 0.57]. Used, because ≥ 85%.
- It passes 40% of what Max fails, and fails 8.1% of what Max passes.
- Applied only in categories with ≥ 20 pairs at ≥ 85%:
  - coding: 85.5% (n = 4,489)
  - sentinel/daemon: 85.7% (n = 28, CI 68.5–94.3)
- Not applied:
  - memory/wiki: 83.2% (n = 113)
  - other: 76.2% (n = 21)
- Six judge rounds (the cap rebuild admits rows that haven't been judged): 2,539 local verdicts, 1,929 pass / 610 fail.
- Of v4's 1,895 unjudged train rows: 1,605 judged, 358 dropped. Rows entering in rebuilds: 869 judged, 232 dropped. No local fail is left in the dataset.

| Split | Rows | Max golden | Max gym | Local 27B gym | Max total |
|---|---:|---:|---:|---:|---:|
| train | 4,480 | 3,751 (83.7%) | 696 (15.5%) | 33 (0.7%) | 99.3% |
| val | 140 | 128 | 12 | 0 | 100% |
| eval-tools | 402 | 295 + 26 failed-call hard cases | 79 | 2 | 99.5% |

- **Labels in train:** 4,188 judge-pass (Max or local) and 292 tool-success only. The tool-success rows are mostly in memory/wiki and other, which aren't calibrated; Max judges them after the Nov 16 reset.
- **Categories:** coding 70.0%, other 13.5%, daemon 8.8%, chat-mode 6.3%, memory/wiki 1.4%.
- **Presence:** chat-mode 284, error-recovery 654, compacted 166.
- **Class caps:** reported_blocked 10.1%, no-tool 2.5%.
- **Leakage:** 0 overlaps.
- **Data policy:** all 4,587 Max-derived train/val rows are covered by the Q6b approval. Private, tagged max-derived.
- **Hashes:** manifest sha256 `f05b6d5c4037bf18c1ee6394d9c7195e5568a44949455149f3521e5d5fffe45c`, train sha256 `ba2cc33652d4b3aea86d93e4fa5a33a01ecc4a8eef6c86a8e38bb2874a33b46b`. Staged to the 4090 and verified there.
- **Known gap (also in v4):** a parallel sibling call can bring a failed turn back into train. To be fixed before the next dataset.

## Round 2 dataset plan (added 2026-11-02)

**Operator rule (Mon ~08:00 ET):** injection ≤ 20%, daily-agent ≤ 30%, coding ≤ 30%; long-context and tool-choice keep everything that passes; keep the 40% v4b replay.

### How the rule is read

- **The caps are shares of the new round-2 train rows,** not of the final train set. Replay is a separate 40% block of v4b's own mix, and its rows carry no target area.
  - This is the stricter reading. An area's share of the final set is 0.6 × its share of the new rows, so a set that meets this reading also meets the final-set reading.
  - Read against the final set instead, the caps would become 33% / 50% / 50% of the new rows, and replay would act as filler.
- **The cap key is the row's `target_area`** (teacher_gen's `_meta.target_area`), not its category.
  - The round-1 category caps are off for the new rows (`--category-cap 1.0`). The area caps replace them.
  - Replay keeps v4b's capped mix: 70% coding by category.
- **Error-recovery isn't named, so it is uncapped:** it keeps every row that passes. It can't break a cap, because caps are maxima.
- **Uncapped areas are never down-sampled.** Neither the area caps nor the class caps (reported_blocked 15%, no-tool answer 20%) nor a category cap ever drops a long-context, tool-choice or error-recovery row. If protected rows alone keep a class over its cap, the class-cap gate fails rather than drop them.
- **Sampling:**
  - It takes the largest fixed point of T = Σ min(n_area, ⌊cap × T⌋), where uncapped areas count in full.
  - Within an area it takes a round-robin over tasks, seeded by `--seed phase3-r2-v1`, which the report records.
  - Val keeps the same per-area share that train kept.
  - `caps.area_caps` and `round2` in report.json record what each cap removed, per area.

### Replay and the caps

- **Replay is drawn after the caps,** so it scales with the capped new rows: replay = 0.4 / 0.6 × new train rows.
- **Fewer new rows means less replay.** A tighter cap shrinks the new rows, and the replay shrinks with them.
- **Replay rows are leakage-gated** against the new val and eval-tools sets, and against every eval set (below). The area-cap gate counts only the new rows.

### Recipe

`build.py round2 --out ~/distill/phase3/build-r2-<tag>-max-derived [--dry-run]` runs one `build` with `R2_RECIPE`. Add `--print` to show the full command without running it.

- **Inputs:**
  - every round-2 pool file, from both the 3090 Ti and the 4090 runs (`~/distill/phase2b/pool/qwen38-27b-gym.*r2*.traces.jsonl`);
  - no golden traces (`--no-golden`): the Max rows return only through replay;
  - train source is the local 27B gym.
- **Verification:**
  - `--unjudged-verifiers deterministic,tool-success` (also build.py's default).
  - The calibration report (`--judge-calibration`) applies. A `judge_pending` row trains only on the calibrated local judge's verdict.
  - The `build.py judge-pending` workdir's `judge-sample.jsonl` is passed as `--verdicts` once it exists. Until then, every judge_pending row is excluded.
- **Caps:** area caps 0.2 / 0.3 / 0.3, category cap off, class caps unchanged.
- **Replay:** `--replay-from ~/distill/phase3/build-v4b-r6-c70-max-derived --replay-share 0.4`, run under the Q6b allow (`--allow-max-derived --max-derived-approval "operator Q6b 2026-10-30"`, excluded-sessions list). The replay rows are Max-derived, so the dataset is tagged max-derived and private.
- **Leakage:**
  - v4b eval-tools and val-cases tasks are pinned to eval-tools (`--eval-holdout`).
  - `--eval-disjoint` (new) runs on v4b eval-tools, v4b val-cases, v2 eval-tools, eval-injection, eval-longctx, sanity and eval-longctx-tools. Any row holding one of their user messages (normalized, ≥ 12 characters) is excluded, and so is any matching replay row.
  - The usual cross-split leakage gate still runs.
- **Plan file:** `round2-plan.json` is written beside the report. It gives, per area, rows in, what passes, train before and after caps, and what a `--target-train` final set (default 5,000) needs.

### Dry run on the 3090 Ti host, 2026-11-02 ~09:00 ET

- **Inputs:** 2,932 round-2 rows from 5 runs, three on the 3090 Ti and two on the 4090. The run outputs were read-only; the report went to a scratch dir.
- **Overall: FAIL, on train size only.** Train is 602 rows against the 5,000 floor. Every other gate passes: leakage, verdicts, data policy, val share, eval-tools size, category, class, area caps, eval-disjoint, presence. The spot check was not run.

| Area | Rows in | judge_pending | Deterministic | Tool-success | Pass | Train before caps | Removed by caps | Train | Val |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| injection-resistance (≤ 20%) | 1,200 | 235 | 965 | 0 | 846 | 721 | 649 | 72 | 2 |
| daily-agent (≤ 30%) | 1,117 | 23 | 0 | 1,094 | 1,079 | 938 | 830 | 108 | 3 |
| coding (≤ 30%) | 253 | 217 | 0 | 36 | 32 | 27 | 0 | 27 | 1 |
| error-recovery (uncapped) | 287 | 133 | 154 | 0 | 137 | 106 | 0 | 106 | 5 |
| long-context (uncapped) | 48 | 15 | 0 | 33 | 32 | 30 | 0 | 30 | 2 |
| tool-choice (uncapped) | 27 | 9 | 18 | 0 | 18 | 18 | 0 | 18 | 0 |
| **Total** | **2,932** | **632** | **1,137** | **1,163** | **2,144** | **1,840** | **1,479** | **361** | **13** |

- **Where the rows that didn't pass went:** 632 are judge_pending, 154 are duplicates and 2 hit a render error. Nothing was excluded for eval-set overlap.
- **What happened to the rows that passed:** 245 went to the new eval-tools (10% of tasks, mostly daily-agent and injection) and 59 to val before caps.
- **Projected dataset:** train 602 (361 new + 241 replay, 40.0%), val 13, eval-tools 245.
  - The v4b replay had 4,468 rows eligible. It dropped 2 for a val/eval-tools task, 6 for holding an eval-set user message and 4 already in the new train.
  - Those 6 v4b train rows share a user message with v4b eval-tools, v4b val-cases or v2 eval-tools (3 each, by this cross-session check). This is stricter than v4b's own session-scoped grouping, which let them through.
- **New-row shares:** daily-agent 29.9%, error-recovery 29.4%, injection 19.9%, long-context 8.3%, coding 7.5%, tool-choice 5.0%.
- **Why the caps cut so deep:** injection and daily-agent together may be at most 50% of the new rows. Only 181 passing train rows sit outside them (coding 27 + error-recovery 106 + long-context 30 + tool-choice 18). That holds the new set to about 2 × 181 = 361 rows, and the caps remove 649 injection and 830 daily-agent rows that passed.
- **For comparison, the final-set reading** (caps 33% / 50% / 50% of the new rows) would give 1,084 new + 723 replay = 1,807 train. It would still fail the 5,000 floor.

### What a ~5,000-row round-2 train set needs (final train = new + replay)

At 40% replay, 5,000 final train rows means 3,000 new train rows: injection ≤ 600, daily-agent ≤ 900, coding ≤ 900, and the uncapped areas the rest. On this run, about 85% of passing rows reached the pre-cap train split.

| Area | Train rows at the cap | Have (before caps) | Short (train rows) | Short (passing rows) |
|---|---:|---:|---:|---:|
| injection-resistance | 600 | 721 | 0 | 0 |
| daily-agent | 900 | 938 | 0 | 0 |
| coding | 900 | 27 | 873 | ~1,035 |
| error-recovery + long-context + tool-choice | ≥ 600 | 154 | 446 | ~542 |

- **The binding need is 1,319 more train rows outside injection and daily-agent** (about 1,580 passing rows), in any mix of coding, error-recovery, long-context and tool-choice with coding ≤ 900.
  - Uncapped rows can replace missing coding rows one for one.
  - Injection and daily-agent already have enough. Generating more of either adds nothing until the other areas catch up.
- **Steering consequence:** the current steer shares (coding 0.05, injection 0.12, daily-agent 0.15) are far from what these caps need. Coding, error-recovery, long-context and tool-choice should take nearly all generation until the shortfall closes.
- **If the target is 5,000 new rows** (final ≈ 8,333): injection 1,000 (short 279), daily-agent 1,500 (short 562), coding 1,500 (short 1,473), uncapped ≥ 1,000 (short 846).

### judge_pending backlog (blocks training until the calibrated judge pass runs)

- **632 rows wait for the calibrated 4090 judge:** coding 217, injection 235, error-recovery 133, daily-agent 23, long-context 15, tool-choice 9.
  - 374 of them are in the scarce areas: coding, error-recovery, long-context, tool-choice.
  - At v4b's local-judge pass rate (76%), they would add roughly 285 passing rows, about 240 train rows. That is far short of the 1,319 needed.
- **No round-2 judge workdir exists yet.** The pass is:
  1. `build.py judge-pending --traces '<pool glob>' --calibration ~/distill/phase2-local-cal/calibration-report.json --workdir ~/distill/phase2-local-r2`;
  2. `judge.py --workdir ~/distill/phase2-local-r2 run --provider local` against the calibrated 4090 judge;
  3. rerun the recipe, which picks up `judge-sample.jsonl` by itself.

## Long-context tool errors (round 2, added 2026-11-06)

Every round-2 long-context attempt with a log on the 3090 Ti host: the local 27B runs (`local-r2-*`, 5 runs), the 4090 cross-host long-context nights (`r2-tower-*-iq3_xxs-*`, 2 runs; the UD-Q4 standard runs had no long-context tasks) and the Max run (`max-tokenplan-r2-max4-max-derived`, before long-context left it).

**Method.** Status and reasons come from each run's `*.tasks.jsonl`. Per-call detail comes from the run's own gym telemetry (`work/telemetry.db`, never the live one): each task is matched to its turn by time window and call count (`turns` + `tool_calls`). 846 of 907 tasks matched, including 601 of the 634 `tool_error` tasks. Only error shapes and short paths/commands left the host.

### Status

| Teacher | Tasks | Kept | Keep rate | Rejected | Error |
|---|---:|---:|---:|---:|---:|
| local 27B (3090 Ti) | 560 | 57 | 10.2% | 479 | 24 (transport) |
| 4090 IQ3 (cross-host) | 206 | 8 | 3.9% | 198 | 0 |
| Max (Token Plan) | 141 | 15 | 10.6% | 125 | 1 |
| **All** | **907** | **80** | **8.8%** | **802** | **25** |

### Verification reasons (rejected tasks; one task can have several)

| Reason | local 27B | 4090 | Max |
|---|---:|---:|---:|
| tool_error | 337 (70%) | 182 (92%) | 115 (92%) |
| sandbox_refusal | 285 (59%) | 66 (33%) | 27 (22%) |
| iteration_cap (24 calls) | 136 | 121 | 66 |
| breaker_tripped | 21 | 8 | 2 |
| output_limit | 10 | 7 | 3 |
| empty_tool_args | 6 | 5 | 3 |
| iteration_cap and nothing else | 25 | 8 | 4 |

### Failed calls by tool and class

2,965 failed calls in rejected tasks (local 1,712, 4090 843, Max 410). "Tasks" counts rejected tasks with at least one such call. Verdict: **model** = the teacher's mistake, correctly a failure; **fix** = the gym wrongly refuses a reasonable call; **parked** = python / multi-line, operator-parked (`--hardened-exec` stays off); **correct** = the sandbox's security boundary; **faithful** = the shipped runtime reports the same error.

| Tool: class | Reason | local | 4090 | Max | Calls | Share | Tasks | Verdict | Examples (shape) |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| read_file: file not found, a path that never existed at any commit | tool_error | 552 | 664 | 274 | 1,490 | 50.3% | 541 | model | `Prometheus/tests/test_loop_manager.py`, `Prometheus/tests/test_wiki_query.py`, `Prometheus/prometheus/web/server.py` |
| bash refused: python / uv / pytest | sandbox_refusal | 284 | 40 | 33 | 357 | 12.0% | 146 | parked | `cd ~/Prometheus && python -m pytest tests/…`, `python3 -c "import pytest; …"`, `uv run pytest …` |
| bash refused: host-home path to the repo (`<host-home>/prometheus-deploy`, `<host-home>/Prometheus`) | sandbox_refusal | 151 | 27 | 0 | 178 | 6.0% | 137 | **fix** | `cd <host-home>/prometheus-deploy && git status --short`, `… && git log --oneline -8`, `cd <host-home>/Prometheus && grep …` |
| web_search / web_fetch: offline (no results, 404) | tool_error | 88 | 2 | 16 | 106 | 3.6% | 32 | correct (no network) | `web_search(query)`, `web_fetch(url)` |
| bash: exit ≠ 0 with no output (grep / which found nothing) | tool_error | 67 | 14 | 16 | 97 | 3.3% | 70 | faithful | `grep -i "<word>" Prometheus/tests/<file>.py`, `which python3` |
| bash refused: host interpreter paths (python probing) | sandbox_refusal | 75 | 7 | 15 | 97 | 3.3% | 69 | parked | `ls /usr/bin/python*`, `ls <host-home>/.local/bin` |
| bash refused: program not in the gym | sandbox_refusal | 64 | 9 | 16 | 89 | 3.0% | 67 | correct | `git check-ignore -v …`, `git checkout …`, `sqlite3 ~/.prometheus/…/*.db …` |
| bash refused: multi-line command | sandbox_refusal | 58 | 9 | 3 | 70 | 2.4% | 61 | parked | `python3 -c "<multi-line script>"`, `cd <dir>⏎grep …` |
| bash refused: the sandbox's parent (git dir, other workers) | sandbox_refusal | 51 | 9 | 5 | 65 | 2.2% | 53 | correct | `ls <sandbox-root>/w0`, `find <sandbox-root> -name …` |
| bash: exit ≠ 0, missing path (ls / cd / head on a guess) | tool_error | 44 | 16 | 3 | 63 | 2.1% | 51 | model | `ls Prometheus/tests/test_<guess>.py`, `ls -la ~/.prometheus/AGENTS.md` |
| bash refused: other host paths (`/`, `/etc`, `/proc`, `/mnt`) | sandbox_refusal | 47 | 9 | 6 | 62 | 2.1% | 47 | correct | `find / -name prometheus.yaml`, `ls -d /media/* /mnt/*` |
| bash / write_file refused: host-home path to runtime files (the pack's documents root, `~/.prometheus`) | sandbox_refusal | 39 | 10 | 0 | 49 | 1.7% | 44 | **fix** (documents root) / model | `ls -la <host-home>/distill/…/documents/`, `cd <host-home> && grep -rn … Prometheus/` |
| bash refused: network (gh, git fetch, off-host curl) | sandbox_refusal | 32 | 6 | 4 | 42 | 1.4% | 32 | correct | `gh pr list`, `git fetch origin --prune`, `curl` to a LAN address |
| bash: exit ≠ 0, other | tool_error | 32 | 4 | 3 | 39 | 1.3% | 33 | model | `curl -s -m 5 http://127.0.0.1:<port>/api/…`, `git log -2 <missing path>` |
| bash refused: other shell variables, `$( )`, for-loops | sandbox_refusal | 29 | 5 | 4 | 38 | 1.3% | 37 | grammar (left) | `for f in …; do …; done`, `$(command -v …)`, `$?` |
| bash refused: only `$HOME` / `$PWD` / `$PATH` | sandbox_refusal | 27 | 4 | 4 | 35 | 1.2% | 31 | **fix** | `echo "$HOME"`, `ls -ld $HOME/.prometheus`, `echo $PATH && …` |
| bash refused: other grammar (env assignment, subshell, `find -exec`) | sandbox_refusal | 18 | 3 | 6 | 27 | 0.9% | 24 | grammar (left) | `PYTHONPATH=src python3 -m pytest …`, `find tests -name … -exec grep …` |
| edit_file: old_str not found / ambiguous | tool_error | 13 | 1 | 1 | 15 | 0.5% | 10 | model | `edit_file(path, old_str, new_str)` |
| write_file: gate denied (outside the workspace) | tool_error | 10 | 2 | 1 | 13 | 0.4% | 10 | model / **fix** (4 calls to the documents root) | `write_file(<sandbox-root>/w0/out/…)`, `write_file(<host-home>/distill/…/documents/loops/…)` |
| tool_search: no such tool | tool_error | 11 | 2 | 0 | 13 | 0.4% | 10 | model | `tool_search(query="run tests")` |
| other (task_get / task_output / memory validation) | tool_error | 13 | 0 | 0 | 13 | 0.4% | 9 | model | `task_get(id)`, `memory(content, operation)` |
| read_file: path is a directory | tool_error | 7 | 0 | 0 | 7 | 0.2% | 5 | model | `Prometheus/scripts/parity` |

### What this says

- **The top cause is the teacher's, not the gym's.**
  - Half of all failed calls (1,490) are `read_file` on a file that does not exist in the fixture (27055dbf), on `main`, or anywhere in the history. 85% are under `tests/`.
  - They are guesses at sibling test files next to the tests the turn has already read. 1,065 of them appear nowhere in the prompt, the context reads or the repo's text. A guessed read is in 258 / 174 / 109 of the local / 4090 / Max rejections.
  - The gym is right to fail them, and the policy (`--max-natural-errors 0`) rejects a row even when the teacher recovers with a glob on its next call.
- **The real gym-fidelity bug is the host home.** The request pack's system prompt was captured from the live runtime. It says the working directory is `<host-home>/prometheus-deploy` and names `~/Prometheus`, `~/.prometheus` and a documents root under `<host-home>`. The gym maps only `~` to the sandbox's HOME, so an absolute host-home path is refused as outside the workspace, and so is a `write_file` there.
  - This affects 178 + 49 calls in 137 + 44 tasks, almost all from the local and 4090 27B (Max followed the gym's own working-directory line).
  - `$HOME` / `$PWD` / `$PATH` in a command are refused outright, although their values are known constants.
- **Upper bound for the fixes.** Count the rejected tasks whose every failed call is in a **fix** class and that have no other reason: local 20, 4090 1, Max 0. That takes the keep rate from 80 / 907 (8.8%) to at most 101 / 907 (11.1%). The local 27B goes from 10.2% to 13.8%. Max's long-context keep rate is not moved by any gym fix.
- **Not fixed here, for the operator** (not gym fidelity):
  - **Python and multi-line** (parked): 357 + 97 + 70 calls in 146 + 69 + 61 tasks.
  - **The zero-natural-error policy:** tolerating a guessed-path `read_file` that the teacher recovered from would keep another 91 / 41 / 40 tasks (local / 4090 / Max), before the judge or any other check.
  - **The 24-call long-context cap:** 136 / 121 / 66 rejections hit it, and 25 / 8 / 4 for that reason alone.
