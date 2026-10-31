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
