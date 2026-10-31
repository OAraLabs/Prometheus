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
