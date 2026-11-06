# Phase 2: judge verification (2026-10-31)

**Gate: RE-JUDGE.** The existing "tool call didn't error" labels let through about 1 in 5 traces that a strict judge
considers wrong. The re-judge of the full deduped pass set (worst categories first) is staged; it starts once the
rubric review from the teacher keep-rate diagnosis is done, so ~10K calls aren't spent on a rubric that may change.

- **Judge:** Qwen3.8 Max on the operator's subscription key, under the operator's informed override of 2026-10-31.
  ~1,500 calls so far with no errors, no throttling and no stop. Every call is in `spend-ledger.csv`, tagged `token-plan`.
- **Sample:** 300 pass + 100 fail traces (stratified, deduped, Max only), rubric v2: right tool, valid arguments,
  sensible order, task advanced, no made-up tool results. Verdicts are in `data/distill/judge-sample.jsonl` on the host.
- **How to read it:** in the outcome view the judge sees the recorded tool result. Blind, without the result, it spots
  only 34% of known failures, so its strength is in reading outcomes, not guessing them.
- **Caveat:** the same rubric rejects a large share of fresh teacher traces too (see the log). Part of the 20%
  may be rubric strictness rather than bad labels; the rubric review addresses that before the full re-judge.

## Judge report (as produced)

### Phase 2 judge report (outcome view, rubric v2)

**Gate: re-judge** (false-pass 20.502% > 5%: re-judge the pass set, worst pass-side agreement first).

- Judged: 300/300 pass, 100/100 fail; 0 without a verdict.
- False-pass (pool-weighted): 20.5% [16.2%, 25.6%], n_eff 285.4; raw 60/300 = 20.0% [15.9%, 24.9%].
- Agreement: 303/400 = 75.8% [71.3%, 79.7%].
- Judge: max-token-plan (400); models qwen3.8-max -> qwen3.8-max; $0.00 by the verdict rows (the spend ledger is authoritative).
- Fail-labelled items the judge also fails: 63/100 = 63.0% [53.2%, 71.8%] (this view shows the judge the recorded error, so this is not a test of the judge).
- Fail side, blind view: 34/99 = 34.3% [25.7%, 44.1%].
- False-pass by recorded result (raw): with 17/119 = 14.3%; without 43/181 = 23.8%.

| Category | Judged | Agreement | 95% CI | False-pass | Fail agree | Pass pool |
|---|---:|---:|---|---:|---:|---:|
| coding | 299 | 74.9% | [69.7%, 79.5%] | 51/238 | 37/61 | 8958 |
| other | 40 | 77.5% | [62.5%, 87.7%] | 1/21 | 11/19 | 429 |
| sentinel/daemon | 45 | 77.8% | [63.7%, 87.5%] | 5/28 | 12/17 | 695 |
| memory/wiki | 16 | 81.2% | [57.0%, 93.4%] | 3/13 | 3/3 | 113 |

Re-judge order: memory/wiki, coding, sentinel/daemon, other.

## Flash calibration and the re-judge's last 5,542 (2026-11-06)

**Routing: Max does the bulk.** Flash agrees with Max on 78.5% of items, below the operator's 90% bar. Everything
here is tagged max-derived and private. The quota is back on an upgraded subscription; the Token Plan key is used
under the operator's standing override (ledger tag `token-plan`).

### Calibration: Flash against Max's existing verdicts (no new Max calls)

- **Set:** 200 pass-side items that Max had already judged, drawn from the re-judge and the Phase 2 sample
  (`judge.py subset --which max-judged --sample 200 --min-per-category 40 --min-per-verdict 10`).
  - Stratified by category, then by Max's verdict: 152 Max passes and 48 Max fails.
  - Sentinel/daemon and other are taken whole (28 and 21 items).
- **Same judge setup as Max:** rubric v2, the same system prompt and user prompt, the outcome view, temperature 0,
  thinking off and max_tokens 1024.
  - The prompt functions are unchanged since the re-judge's code (a269a86).
  - On all 200 items the rendered prompt's length and truncation flag equal Max's recorded `prompt_chars` and
    `truncated`, and the mean input tokens are identical (3,331).
- **Run:** `--provider token-plan --judge-model qwen3.8-flash`, ledger phase `2-judge-flash-cal`, 207 calls, no
  errors. 7 replies (3.5%) were prose instead of JSON; all 7 parsed on the one retry.
- **Bulk-mix figure:** each category's weighted agreement, weighted by its share of the 5,542 items.

| | Value |
|---|---|
| Agreement | 157/200 = **78.5%** [72.3%, 83.6%] |
| Weighted to every Max-judged item | 78.0% |
| Weighted to the bulk's category mix | 78.2% |
| Cohen's kappa | **0.380** [0.215, 0.544] |
| Flash passes what Max fails | 25/48 (52.1%) |
| Flash fails what Max passes | 18/152 (11.8%) |
| Pass rate | Max 76.0%, Flash 79.5% |

| Category | n (Max pass / fail) | Agreement | 95% CI | kappa | Flash pass, Max fail | Flash fail, Max pass |
|---|---:|---:|---|---:|---:|---:|
| coding | 109 (81 / 28) | 76.1% | [67.3%, 83.2%] | 0.375 | 13/28 | 13/81 |
| sentinel/daemon | 28 (23 / 5) | 71.4% | [52.9%, 84.8%] | 0.026 | 4/5 | 4/23 |
| memory/wiki | 42 (28 / 14) | 81.0% | [66.7%, 90.0%] | 0.520 | 7/14 | 1/28 |
| other | 21 (20 / 1) | 95.2% | [77.3%, 99.2%] | n/a (1 fail) | 1/1 | 0/20 |

**Tokens per item (mean):**

| Judge | Input | Output | Cached input | Source |
|---|---:|---:|---:|---|
| Flash | 3,331 | 152 | 965 | Flash's own verdict rows on the 200 items |
| Max | 3,331 | 125 | 1,781 | Max's original verdict rows on the same 200 items |
| Max (whole re-judge) | 3,313 | 124 | 2,253 | the 4,353 `ok` rows of the re-judge |
| Max (ledger) | 3,313 | 124 | n/a | the 4,365 `token-plan ok` ledger rows in `2-judge-all` (12 were unparseable) |

- Input is identical per item: same prompt, same tokenizer. Flash writes about 20% more output.
- **Why not Flash:** it passes half of what Max fails, so as a filter it would let most of Max's rejections back
  in. Only "other" clears 90%, and that rests on a single Max fail.

### The 5,542 never-judged traces

- **What they are:** the Max re-judge queue (`phase2-rejudge`, 9,895 deduped pass-labelled Max golden traces)
  less its 4,353 Max verdicts: coding 4,467, sentinel/daemon 667, other 408.
  - 5,530 never had a Max call.
  - 12 (coding) had one that came back unparseable, so they never got a verdict. That's why the Phase 3 note says
    "4,353 judged, 12 unparseable" and still counts 5,542.
  - memory/wiki was finished before the quota ran out.
- **Excluded for the area caps: 0.** None of these rows carries a round-2 `target_area`; they are harvested
  golden traces, not gym rows. The `pending` subset ran with `--exclude-area injection-resistance --exclude-area
  daily-agent` and removed nothing (queue meta `excluded_by_area: {}`).
- **Run:** the existing re-judge workdir, resumed. Max, the same queue and order, ledger phase `2-judge-all`:

  ```
  judge.py --workdir ~/distill/phase2-rejudge run --provider token-plan --judge-model qwen3.8-max --concurrency 8
  ```

  - It ran as the user unit `distill-max-bulk-20261106` (log `~/distill/flash-judge/max-bulk.log`).
  - The verdicts land in `phase2-rejudge/judge-sample.jsonl`; a backup from before the run sits beside it
    (`.bak-20261106`).
  - `build.py`'s `load_verdicts` reads them as `verdict_rows_tokenplan_max_judge`; the early rows all joined.
- **Rebuild after it finishes:** pass `phase2-rejudge/judge-sample.jsonl` (the full file, not the v4 snapshot) as
  `--verdicts`, after `phase2/judge-sample.jsonl`. The ~1,900 fallback `tool-success` Max golden rows then get a
  real verdict.
