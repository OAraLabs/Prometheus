# Phase 3: datasets (dry run, 2026-10-30)

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

## How the data was built

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

## Eval sets (built on the 3090 Ti host)

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

## What would make the gate pass

1. Phase 2b-style expansion: ~4–9K more verified traces, including chat-mode and compacted-history examples. That
   needs a teacher the plan terms allow (OPEN-QUESTIONS #6), or another source of verified traces.
2. Then re-judging (or the tool-success fallback for what can't be judged), and the spot check once a student GGUF
   server can run.

## Re-running

On the 3090 Ti host: `~/distill/phase3/code/` holds the builder; `build.py pack` then `build --dry-run`. Outputs and
`build_report.json` are under `~/distill/phase3/`. Code: `scripts/distill/build.py`, `render.py`,
`build_eval_sets.py`, `build_fail_set.py`.
