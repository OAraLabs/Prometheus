# Phase 4: baselines and the long-context gate

**Status (2026-10-31):** the gate math is done and in the scorer (`scripts/distill/score.py`, scorer v4).
Baselines are in for base Qwen3.5-9B, Saluki and Qwen3.8-27B; Muse Glimmer 30B is partial. See
[Baselines](#baselines) at the end.

The operator approved OPEN-QUESTIONS #8 on 2026-10-30. The fixed long-context bar ("no worse than base by
more than 3 pts at any depth bucket") is replaced by a paired gate built on confidence intervals. The other
three Phase 6 criteria keep their bars.

---

## Gate math

### Why the fixed bar had to go

eval-longctx has 180 cases: 45 per length bucket (32K, 64K, 128K, 256K) and 9 per needle depth (10, 25, 50,
75 and 90%) inside each bucket. One case is 2.2 pts of a bucket, so "worse by more than 3 pts" means two net
misses. Two models (or one model run twice) never agree on every case. If an **unchanged** model flips a
share *q* of cases each way, the fixed bar fails it in at least one of the four buckets this often (exact
trinomial sums, `fixed_bar_fail_prob` in `score.py`):

| Flip share each way | Unchanged model fails the fixed bar |
|---|---|
| 5% | 65.5% |
| 10% | 76.9% |
| 20% | 83.4% |

That bar mostly measured noise, so a FAIL from it said almost nothing about the tuned model.

### The idea in plain language

1. **Pair the cases.** Base and tuned models answer the same 180 cases, so we look at each case and compare
   the two answers. A case both models got right, or both got wrong, says nothing about which model is
   better. Only the cases where they **disagree** carry information.
2. **Count the disagreements.** Per bucket, *b* is the number of cases only the base got right, and *c* the
   number only the tuned model got right. The drop is (*b* − *c*) / *n*.
3. **Put an interval around the drop.** A 95% interval shows how big the drop could really be, given how many
   cases there are and how often the models disagree.
4. **FAIL only on a drop we can see.** A bucket FAILs when the tuned model is **significantly** worse (the
   interval, corrected for testing four buckets, sits wholly above zero, and an exact count of the
   disagreements agrees) **and** **materially** worse (the drop is more than 3 pts). A drop spread across
   every bucket is also tested on all 180 cases together, which sees smaller drops.
5. **Otherwise PASS, and say how sure.** A PASS is labelled *non-inferior* when the data rule out a drop over
   3 pts, and *underpowered* when they don't. An underpowered PASS prints the smallest drop the gate would
   catch at that number of cases.

### The table

Per bucket, the *n* cases form a paired 2×2 table:

|  | tuned right | tuned wrong |
|---|---|---|
| **base right** | *a* | *b* |
| **base wrong** | *c* | *d* |

- Base recall *p*₁ = (*a* + *b*) / *n*; tuned recall *p*₂ = (*a* + *c*) / *n*.
- Drop (degradation) θ = *p*₁ − *p*₂ = (*b* − *c*) / *n*. Above zero means the tuned model is worse.
- The report prints Δ = −θ (tuned minus base, so a negative number means worse) in its "Delta" and
  "Paired 95% CI" columns, the same way round as every other criterion. The JSON carries both
  (`diff_pts`/`ci_pts` and `degradation_pts`/`degradation_ci_pts`).

Cases are paired **by case id**. If the two summaries do not hold the same ids (in a bucket or in the set), or
a summary has no per-case outcomes, the criterion is **INCONCLUSIVE**. Summaries now record every case's
outcome (`outcomes`) and, for eval-longctx, each case's bucket and needle depth (summary schema
`distill-score-summary/3`). An older summary is refused, and re-running `summarize` on its rows fixes it
without the GPU.

### The interval: Newcombe's hybrid score interval

The gate uses method 10 of Newcombe (1998), the "square-and-add" interval for paired proportions. First take
the 95% Wilson interval of each recall (*z* = 1.95996):

    Wilson(k, n):  centre = (p + z²/2n) / (1 + z²/n)
                   half   = z·sqrt(p(1−p)/n + z²/4n²) / (1 + z²/n),    p = k/n
    base:  (l₁, u₁) = Wilson(a + b, n)        tuned:  (l₂, u₂) = Wilson(a + c, n)

Then combine the two intervals, using the correlation between the two models' answers:

    φ = (ad − bc − n/2) / sqrt((a+b)(c+d)(a+c)(b+d))   when ad − bc > n/2
      = 0                                            when 0 ≤ ad − bc ≤ n/2, or any margin is 0
      = (ad − bc) / sqrt((a+b)(c+d)(a+c)(b+d))         when ad − bc < 0

    lower = θ − sqrt((p₁ − l₁)² − 2φ(p₁ − l₁)(u₂ − p₂) + (u₂ − p₂)²)
    upper = θ + sqrt((u₁ − p₁)² − 2φ(u₁ − p₁)(p₂ − l₂) + (p₂ − l₂)²)

**Why this interval:**
- It is built for paired data. The two recalls are strongly correlated, because both models find the easy
  needles and miss the hard ones. An interval for two independent rates ignores that correlation and comes
  out much too wide: about twice as wide for the 256K bucket in the worked example below.
- It behaves better than the simple alternatives. The Wald interval (θ ± *z*·SE) collapses to zero width when
  *b* = *c* = 0 and can overshoot ±1. The exact conditional (McNemar-based) interval is conservative, and
  Newcombe showed it can be effectively one-sided. This interval stays inside [−1, 1] and has near-nominal
  coverage once the table holds a few dozen cases. With dozens of disagreements its p-values sit close to
  one-sided mid-P values. For *a* = 30, *b* = 5, *c* = 0, *d* = 10 it gives 0.0172; the one-sided mid-P is
  0.0156 and the one-sided exact McNemar *p* is 0.031. (The report prints the **two-sided** exact McNemar
  *p*, 0.0625 for this table, as `mcnemar_exact_p`.)
- **It is not good with very few disagreements.** When a margin is 0 the correlation term is forced to 0,
  and with a handful of cases the interval comes out too narrow. Three cases, all lost by the tuned model
  (*a* = 0, *b* = 3, *c* = 0, *d* = 0), give an interval *p* of 0.0036, against an exact one-sided McNemar
  *p* of 0.125. On its own, the interval's test rejects an unchanged model far more often than its level
  says when *n* is small. We computed the exact rate by enumerating every table, over base recall 30–90%,
  flip shares 2–40% and the four Holm levels. In the worst case it is 6.4 times the level at *n* = 2,
  10 times at *n* = 3, 5.7 times at *n* = 5, 2.0 times at *n* = 10, 1.4 times at *n* = 20 and 1.12 times at
  *n* = 45. The gate therefore checks it with an exact test (next section).
- It is closed-form, so there is nothing to root-find and no convergence to check. (Tango's score interval is
  the main alternative with similar coverage, but it needs an iterative solution.)

**Checked against published values** (all in `tests/test_distill_score.py`):

| Check | Published | `score.py` |
|---|---|---|
| Paired interval, Armitage & Berry (1994, p. 138): *a* = 20, *b* = 12, *c* = 2, *d* = 16 (StatsDirect's worked example of this method) | 0.056156 to 0.329207 | 0.056156 to 0.329207 |
| Exact two-sided McNemar *P*, same table | 0.0129 | 0.01294 |
| Wilson 81/263, 15/148, 0/20, 1/29 (Newcombe 1998, single proportion) | 0.2553–0.3662, 0.0624–0.1605, 0–0.1611, 0.0061–0.1718 | same to 4 decimals |
| Holm-adjusted *p* for (0.01, 0.04, 0.03, 0.005) | 0.03, 0.06, 0.06, 0.02 | same |

### The p-value: the interval's, checked by an exact test

The gate's one-sided p-value for "the tuned model is worse" is the **larger** of two numbers:

1. **The interval's own p.** It is *p* = 1 − Φ(*z*\*), where *z*\* is the normal quantile at which the
   interval's lower bound reaches 0 (found by bisection). So this *p* < 0.025 exactly when the 95%
   two-sided interval lies wholly above zero. The lower bound falls steadily as *z* rises: we checked every
   one of the 8,372 tables at *n* = 45 with *b* > *c*.
2. **The exact mid-p.** Given the *b* + *c* disagreeing cases, an unchanged model splits them like fair
   coin flips. The one-sided mid-p is *P*(*B* > *b*) + ½ *P*(*B* = *b*), with *B* ~ Bin(*b* + *c*, ½).

So the gate calls a drop significant only when both are below the level. The exact mid-p decides when there
are few disagreements: it takes at least 5 disagreeing cases, all one way, to get below 0.025, and 7 to get
below Holm's first step of 0.025/4. With dozens of disagreements the two numbers are close (0.109 against 0.106 for the
256K bucket below), and the interval's *p* usually decides. When *b* ≤ *c* the tuned model is not worse at
all, and *p* is reported as 1.

With the check, the exact chance that one bucket of an unchanged model FAILs is at most 1.005 times its
level. This comes from the same enumeration as above (*n* = 1–30, 35, 40, 45, 50 and 60, at every Holm
level). The worst case, 1.005 times, is at *n* = 20 with 40% flips; at *n* = 45 the worst case is 0.99
times. These rates are close to nominal but are not a proven bound, because the mid-p is not exact either. The exact one-sided McNemar
*p* would give a strict bound, but it runs at 0.3–0.6 times its level at *n* = 45 (in the cases we
computed), which costs a lot of power.

**The printed interval and the test can disagree.** The "Paired 95% CI" column is the **unadjusted** 95%
interval. A bucket's interval can lie wholly below zero (in Δ terms) and the bucket still PASS, for two
reasons: Holm holds it to a stricter level (next section), or the exact check does not confirm the drop.
The report then says which one applies, and prints the raw and Holm-adjusted *p*, the bucket's Holm rank and
level, and the lower bound of the interval at that level. For example, take three unchanged buckets
(25, 4, 4, 12) and one at 128K of (30, 5, 0, 10). The 128K row prints the interval [−21.6, −0.8] and
"PASS (underpowered: the unadjusted 95% interval excludes 0, but the drop is not significant after Holm's
correction for 4 buckets (p 0.017; Holm-adjusted p 0.069, rank 1 of 4, held to one-sided 0.00625, where
the drop's lower bound is -2.1 pts); …)".

### Four buckets: Holm's correction

Testing four buckets gives four chances to hit an unlucky one. Given valid p-values, Holm's step-down
procedure keeps the chance of **any** false "significantly worse" at or below 2.5% (one-sided) for the whole
family. It works under any dependence between the buckets, which matters here because the same model
answers all of them. Our p-values are near-valid rather than exact (see above), so the family's rate is
**about** 2.5%, not a proven bound. Simulated, it is 0.4–2.0% at 60% base recall (table below) and 2.2% at
50% base recall with 25% flips (40,000 runs).

- Sort the four p-values. Compare the smallest with 0.025/4, the next with 0.025/3, then 0.025/2, then 0.025.
  Stop at the first one that does not clear its level.
- The adjusted p-value is the running maximum of (*m* − *i* + 1)·*p*₍ᵢ₎. A bucket is significant when its
  adjusted *p* < 0.025.
- *m* is the number of buckets the base summary has. A bucket that cannot be read counts as *p* = 1, which
  keeps the readable buckets on the strictest levels they could have.
- Holm rejects everything Bonferroni (every bucket at 0.025/4) rejects, and sometimes more.
- For each bucket, the report's detail text (in the text and Markdown reports, and in `--json-out`) prints
  the raw and Holm-adjusted *p*, the Holm rank and level, and the lower bound of the interval at that level.
  The JSON fields are `p_worse`, `p_worse_adjusted` and `holm`.

### The pooled test

All 180 cases are also tested as one table: one test, one-sided 0.025, unadjusted, with the same 3-pt
margin. It can FAIL on its own. A drop spread evenly over the buckets (say 4 extra misses in each) is
invisible per bucket, but with four times the cases the pooled test sees it.

The pooled test is not in the Holm family. Its own false rate is also about 2.5% (simulated 2.1–2.6% at
40–90% base recall). So the overall false-FAIL rate is about 2.5% from the Holm family plus about 2.5% from
the pooled test: about 5% by the union bound, and the 3-pt margin can only lower it. Because the p-values are near-valid rather than
exact, these are approximate rates, not a proven ceiling. The simulated overall rate is 2.5–3.9% at 60% base
recall (table below) and 4.1% at 50% base recall with 25% flips.

### The decision rule

Per length bucket, using the Holm-adjusted *p*:

| Outcome | Condition | Status |
|---|---|---|
| fail | adjusted *p* < 0.025 **and** drop > 3 pts (*p* is the larger of the interval's and the exact mid-p) | **FAIL** |
| worse within margin | adjusted *p* < 0.025, drop ≤ 3 pts | PASS |
| non-inferior | not significant, 95% upper bound on the drop < 3 pts | PASS |
| underpowered | not significant, upper bound ≥ 3 pts | PASS, with the minimum detectable drop printed |

- **Pooled:** the same rule with the unadjusted *p*.
- **INCONCLUSIVE:** the set is missing or empty, its case content differs between the summaries, its
  outcomes don't pair (an id on one side only, a case filed under another bucket, or no outcomes), or either
  side has errors or missing cases. The whole gate is also INCONCLUSIVE, whatever the criteria say, when the
  two runs are not like for like: different request settings, server context size, scorer version or schema
  validator; a context-exceeded case; or a reply cut short by the context.
- **Per needle depth** (10/25/50/75/90%, 36 cases each across buckets, 9 per bucket): a diagnostic table, never
  gated. Nine cases resolve nothing on their own, but a drop concentrated at one depth shows up there first.
- **Other criteria, unchanged:** task success ≥ base + 10 pts, injection resistance ≥ base, and sanity no worse
  than base by more than 3 pts are still decided on point estimates, as pre-registered. Each now prints its
  paired 95% interval (and the bootstrap and McNemar cross-checks) as information.
- **Verdict, in this order:** INCONCLUSIVE if the two runs are not like for like (the notes above), whatever
  the criteria say, even when one FAILs; else FAIL if any criterion FAILs; else INCONCLUSIVE if any is
  INCONCLUSIVE; else PASS. Exit codes are unchanged: 0 PASS, 1 FAIL, 2 INCONCLUSIVE. `compare` also exits 2,
  and prints no verdict, when it cannot write `--markdown-out` or `--json-out`; it writes both before it
  prints the verdict. Under the verdict, the report says how many long-context tests were underpowered and
  the largest drop not ruled out.

**Why these choices:**
- **95% two-sided (2.5% per side).** This is the level every rate in the summaries already carries (Wilson
  95%). The question asked is one-sided ("is it worse?"), so its error rate is 2.5%.
- **The 3-pt margin stays as the materiality bar.** It was the pre-registered bar, and keeping it means a
  drop that is real but tiny, which bigger sets would be able to detect, never FAILs the model.
- **FAIL needs evidence of a drop; PASS does not need evidence of no drop.** Demanding non-inferiority (upper
  bound < 3 pts) would fail almost every model at this size: at *n* = 45 and 10% flips, the interval
  half-width is about ±13 pts. That is the old bar's flaw in another form. The cost is that a PASS can be
  underpowered, so the gate labels it and prints what it could have caught.
- **Holm rather than Bonferroni:** same guarantee, more power. Per-bucket tests are kept rather than only the
  pooled one because the runbook's intent ("at any depth bucket") is that a drop confined to 256K counts.
- **The minimum detectable drop is quoted at 80% power,** the usual convention.
- **Bootstrap as a cross-check only:** 2,000 seeded paired resamples, percentile interval. It runs too narrow
  when disagreements are few. For *b* = 4, *c* = 0 at *n* = 45 it gives 2.2 to 17.8 pts, which wrongly
  excludes 0, while Newcombe gives −0.6 to 18.6. So it is printed but never decides. The exact two-sided
  McNemar *p* is printed for the same reason: familiar, but conservative. The gate's own exact check uses
  the one-sided mid-p, which is not.

### Power and minimum detectable drop

The model for these tables: an unchanged model flips a share *q* of cases each way (*P*(*b*) = *P*(*c*) =
*q*). A degraded one also loses a further share δ (*P*(*b*) = *q* + δ). The tables are simulated with the
real decision rule, exact check included, by `score.py gate-power --reps 20000` (seed
`phase4-gate-power/1`, base recall 60%, 20,000 runs per cell).

The base recall matters little, within the range where a table exists. A table needs base recall ≥ flip
share and base recall + flip share ≤ 100%, and a bucket can lose at most base recall − flip share. We
reran at 40% and at 90%:
- `--base-rate 0.4`: false-FAIL rates move by 0.1 pt or less, power by 0.4 pt or less, and the pooled MDDs by
  0.04 pt. The one-bucket MDD at 5% flips is 27.4 pts (27.3 at 60%). At 10% and 20% flips the one-bucket MDD
  is "n/a", because a bucket can lose at most 30 and 20 pts, below what the gate needs.
- `--base-rate 0.9 --q 0.05 --q 0.10`: the 20% flip share makes no table at 90%, and `gate-power` exits 2 on
  it. Each MDD moves by 0.3 pt or less (one bucket: 27.4 and 32.7 pts; pooled: 8.9 and 11.4 pts), power by
  up to 2.2 pts, and false FAILs fall slightly, to 2.3% and 3.2%.

**False FAIL on an unchanged model** (four buckets of 45 cases):

| Flip share each way | Fixed 3-pt bar | Paired gate: any | bucket (Holm) | pooled |
|---|---|---|---|---|
| 5% | 65.5% | 2.5% | 0.4% | 2.2% |
| 10% | 76.9% | 3.8% | 1.8% | 2.4% |
| 20% | 83.4% | 3.9% | 2.0% | 2.5% |

**n = 45 per bucket: one bucket degraded, the other three unchanged.** Power to FAIL, counting either test:

| Flip share each way | drop 5 pts | 10 pts | 15 pts | 20 pts | MDD80 simulated | MDD80 formula |
|---|---|---|---|---|---|---|
| 5% | 8% | 21% | 39% | 58% | 27.3 pts | 28.6 pts |
| 10% | 8% | 16% | 28% | 43% | 33.0 pts | 33.7 pts |
| 20% | 7% | 12% | 19% | 29% | over 40 pts* | 41.6 pts |

\*At 60% base recall and 20% flips, no bucket can lose more than 40 pts, and even a 40-pt drop is caught
less than 80% of the time (`gate-power` prints "n/a").

**Pooled n = 180: every bucket degraded by the same amount:**

| Flip share each way | drop 5 pts | 10 pts | 15 pts | 20 pts | MDD80 simulated | MDD80 formula |
|---|---|---|---|---|---|---|
| 5% | 41% | 87% | 99% | 100% | 8.8 pts | 9.0 pts |
| 10% | 30% | 71% | 94% | 99% | 11.4 pts | 11.6 pts |
| 20% | 20% | 50% | 80% | 95% | 15.1 pts | 15.3 pts |

The exact check costs little power at this size. Before it was added, the same runs gave one-bucket MDDs of
26.0 and 32.1 pts and pooled MDDs of 8.7, 11.3 and 14.8 pts.

**The formula column** (`min_detectable_degradation`, the normal approximation printed in every gate report):
the observed drop *D* has mean δ and variance (ψ − δ²)/*n*, with ψ = 2*q* + δ the disagreeing share. The rule
needs *D* above the 3-pt margin and above the point where the lower bound clears 0, which is
*D* = *z*·sqrt(ψ / (*n* + *z*²)). The MDD is the smallest δ with *P*(FAIL) ≥ 80%. A single bucket uses Holm's
first-step level (*z* = 2.498, one-sided 0.025/4); the pooled test uses *z* = 1.960. The formula leaves out
the exact check, which matters only with few disagreements. It runs about 0.2–1.3 pts conservative against
the simulation. In a gate report, *q* is estimated as (*b* + *c*)/2*n* from the table itself.

**What this means.** With 45 cases a bucket, the gate reliably catches only a drop of roughly 27–42 pts
confined to one bucket. A drop across every bucket is caught from about 9–15 pts. Anything smaller can't be
told apart from noise at this size, by any honest test. Bigger sets would close the gap (formula, 10% flips,
drop in one bucket):

| Cases per bucket | 45 | 90 | 180 | 250 | 400 |
|---|---|---|---|---|---|
| MDD80 for a drop in one bucket | 33.7 | 21.9 | 14.3 | 11.7 | 8.9 pts |

Getting a one-bucket drop down to 15 pts takes about 166 cases a bucket, and down to 10 pts about 328. Each
case at 256K is a full 256K prefill per model, so set size is the operator's trade. The gate code runs at
any *n*, and `score.py gate-power --n <cases>` prints these tables for it. With very few cases, though, a
bucket can only PASS. One of four buckets needs at least 7 disagreeing cases, all one way, to FAIL; a bucket
on its own, or the pooled test, needs at least 5.

### Worked example

These are illustrative counts, not results. The numbers are what `score.py compare` prints for them. Base,
Tuned, Δ and the interval are columns of the report. The Holm rank, level and adjusted *p* are in each row's
detail text, and in `--json-out` as `holm` and `p_worse_adjusted`. `tests/test_distill_score.py` checks
these numbers against the code.

| Bucket | *a* | *b* | *c* | *d* | Base | Tuned | Δ | Paired 95% CI | Holm rank and level | adjusted *p* |
|---|---|---|---|---|---|---|---|---|---|---|
| 32K | 38 | 2 | 1 | 4 | 88.9 | 86.7 | −2.2 | [−12.7, +7.8] | 3: 0.0125 | 0.68 |
| 64K | 33 | 3 | 2 | 7 | 80.0 | 77.8 | −2.2 | [−13.5, +8.9] | 4: 0.025 | 0.68 |
| 128K | 24 | 4 | 2 | 15 | 62.2 | 57.8 | −4.4 | [−15.4, +6.8] | 2: 0.00833 | 0.68 |
| 256K | 14 | 4 | 1 | 26 | 40.0 | 33.3 | −6.7 | [−16.8, +3.7] | 1: 0.00625 | 0.44 |
| pooled | 109 | 13 | 6 | 52 | 67.8 | 63.9 | −3.9 | [−8.7, +0.9] | (unadjusted) | 0.058 |

**The 256K bucket, step by step:**
1. The drop is θ = (4 − 1)/45 = 6.7 pts.
2. The Wilson intervals are 27.0–54.5% for the base (18/45) and 21.4–47.9% for the tuned model (15/45).
3. *ad* − *bc* = 364 − 4 = 360, which is above *n*/2 = 22.5, so φ = (360 − 22.5)/sqrt(18·27·15·30) = 0.722.
4. The interval for the drop is therefore −3.7 to +16.8 pts (Δ from −16.8 to +3.7).
5. The one-sided *p* is 0.109: the larger of the interval's own 0.106 and the exact mid-p 0.109. It is the
   smallest of the four, so Holm compares it with 0.025/4, and the adjusted value is 4 × 0.109 = 0.44. Not
   significant.
6. The upper bound, 16.8, is above 3, so this is **PASS (underpowered)**. At this bucket's disagreement rate
   (*q* = 5/90), the gate would catch a drop of 29.2 pts or more with 80% power.

**The rest of the example:**
- The pooled table (−3.9 pts, *p* = 0.058) is also PASS (underpowered). It would catch about 9.1 pts or
  more.
- **The fixed bar would have FAILed this model** at 128K (−4.4) and 256K (−6.7), on 2 and 3 net cases.
- **Change 256K to *a* = 8, *b* = 10, *c* = 1, *d* = 26** (tuned recall 20.0%, a 20-pt drop) and it FAILs.
  The interval for the drop is 5.8 to 33.3 pts. The one-sided *p* is 0.0032 (0.0028 from the interval, 0.0032
  from the exact mid-p), and the adjusted *p* is 0.013, below 0.025. At Holm's first-step level the interval's lower bound is still +2.0 pts, and the drop is far
  over 3 pts. The pooled test FAILs too: 7.2 pts, interval 1.8 to 12.6, *p* = 0.0047.

### Running it

    uv run python scripts/distill/score.py compare --base base-9b.summary.json \
        --candidate tuned-9b.summary.json --markdown-out gate.md --json-out gate.json
    uv run python scripts/distill/score.py gate-power --reps 20000     # the tables above

In `--json-out` (schema `distill-gate/1`), every criterion carries:
- `rule` (`point` or `paired`)
- `paired`: the table, `ci_pts`, `bootstrap_ci_pts` and `mcnemar_exact_p`
- for long-context criteria also `degradation_ci_pts`, `p_worse` (the gate's *p*), `p_worse_interval` and
  `p_worse_midp` (its two parts), `p_worse_adjusted`, `holm`, `outcome` and `mdd_pts`
- JSON numbers are rounded to 2 decimals. The text and Markdown reports compute what they print from the
  table itself, so a bound is never rounded twice.

The file also lists `longctx_underpowered` and the needle-depth diagnostics. The weekly flywheel's score
stage reads only the exit code. The codes are unchanged, and a report file that cannot be written gives 2,
never the verdict's code.

---

## Baselines

Scored in night 1 (2026-10-31). All four summaries were rewritten with HEAD's scorer (`score.py summarize`,
scorer v4, summary schema `distill-score-summary/3`, schema validator jsonschema) from the stored result rows.
No request was re-sent. The new summaries sit beside the originals as `<label>.head-4ef1dfe.summary.json` and
`.md`. Saluki's originals were scorer v3 (schema /2, no per-case outcomes); the 4090 originals were already v4,
and the rewrite changed no number. Every model got the same case files (sha256 checked on both hosts) and the
same request settings: `max_tokens` 4096, thinking suppressed, server-default temperature, no seed,
concurrency 1. No set has errors or invalid rows.

### Models and servers

| Model | Host | Quant (served file) | Context | Sets covered |
|---|---|---|---|---|
| base-9b: Qwen3.5-9B | 4090 | Q4_K_M (`Qwen3.5-9B-Q4_K_M.gguf`) | 262,144, 1 slot | all four; long-ctx all 4 buckets |
| saluki | 3090 Ti | IQ2 mix (`Underdog-Saluki-27B-1.0-IQ2-mix.gguf`) | 65,536 (live server, 2 slots) | all four; long-ctx **32K only** |
| qwen38-27b: Qwen3.8-27B | 4090 | Q4_K_XL (`Qwen3.8-27B-UD-Q4_K_XL.gguf`) | 131,072, 1 slot | all four; long-ctx 256K over context |
| glimmer-30b: Muse Glimmer 30B | 4090 | Q4_K_M (`Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`) | 131,072 (native), 1 slot | **partial**: sanity; eval-tools 259/329 |

- The 4090 servers ran llama.cpp `-np 1 -fa on -ctk q8_0 -ctv q8_0 --jinja -ngl 99`, each at the largest
  context that loaded (the runner tries the native context first).
- **Saluki at 32K only.** It was scored on its live endpoint, read-only, at concurrency 1, behind the probe
  guard. That server's 65,536-token KV is shared with the daemon's slot, so a 64K+ case would have crowded the
  daemon. It ran the 45 cases of the 32K bucket (`eval-longctx-32k.jsonl`, the same case ids as the full
  set's 32K bucket).
- **Qwen3.8-27B at 131,072.** Its native 262,144 context failed to create on the 24 GB card, so the runner
  stepped down to 131,072. The 45 cases of the 256K bucket are `context_exceeded` and count as misses (0/45).
  Its 256K number therefore measures the context limit, not recall.
- **Glimmer is partial.** Night 1 ended at the 06:35 stop with sanity done and eval-tools at 259/329.
  eval-injection (0/192) and eval-longctx (0/180) have not run. Its eval-tools rate is over those 259 cases,
  not the full set. The runner resumes it in a later 4090 window.
- `task_success` has the same count as `tool_name_accuracy` on every model in this run, so the tables below
  show only task success.

### sanity (211 cases)

| Model | sanity_score | 95% CI | n |
|---|---|---|---|
| base-9b | 96.7% | [93.3, 98.4] | 211 |
| saluki | 94.8% | [90.9, 97.1] | 211 |
| qwen38-27b | 97.2% | [93.9, 98.7] | 211 |
| glimmer-30b | 99.5% | [97.4, 99.9] | 211 |

### eval-tools (329 cases)

| Model | task_success [95% CI] | n | arg_validity | arg_exactness [95% CI] | p50 latency |
|---|---|---|---|---|---|
| base-9b | 72.3% [67.3, 76.9] | 329 | 100% (238/238) | 10.1% [6.9, 14.6] | 2.2 s |
| saluki | 70.5% [65.4, 75.2] | 329 | 100% (232/232) | 9.5% [6.3, 13.9] | 8.9 s |
| qwen38-27b | 72.9% [67.9, 77.5] | 329 | 100% (240/240) | 11.2% [7.8, 15.9] | 7.7 s |
| glimmer-30b (partial) | 75.7% [70.1, 80.5] | 259 | 100% (196/196) | **0.5%** [0.1, 2.8] | 4.6 s |

On the same 259 cases Glimmer has answered, task success is: Glimmer 196, Qwen3.8-27B 196, base-9b 195,
Saluki 184 (75.7 / 75.7 / 75.3 / 71.0%). On the other 70 cases Qwen3.8-27B scores 44 (62.9%), so the 70 left
are harder than average, and Glimmer's full-set rate will likely fall. Glimmer's argument exactness is
far below every other model (1 exact call in 196). Its calls are valid but almost never match the
reference arguments.

### eval-injection (192 cases)

| Model | injection_resistance | 95% CI | n | tool_output / memory / wiki |
|---|---|---|---|---|
| base-9b | 99.0% | [96.3, 99.7] | 192 | 63/64, 63/64, 64/64 |
| saluki | 98.4% | [95.5, 99.5] | 192 | 63/64, 62/64, 64/64 |
| qwen38-27b | 99.0% | [96.3, 99.7] | 192 | 64/64, 64/64, 62/64 |
| glimmer-30b | not run | | 0 | |

### eval-longctx by bucket (45 cases per bucket)

| Model | 32K | 64K | 128K | 256K | ≤128K pooled | All buckets |
|---|---|---|---|---|---|---|
| base-9b | 82.2% [68.7, 90.7] | 80.0% [66.2, 89.1] | 71.1% [56.6, 82.3] | 77.8% [63.7, 87.5] | 77.8% [70.1, 84.0], n 135 | 77.8% [71.2, 83.2], n 180 |
| saluki | 82.2% [68.7, 90.7] | not run | not run | not run | 32K only | 82.2%, n 45 |
| qwen38-27b | 86.7% [73.8, 93.7] | 84.4% [71.2, 92.3] | 82.2% [68.7, 90.7] | 0.0% [0.0, 7.9]* | 84.4% [77.4, 89.6], n 135 | 63.3% [56.1, 70.0], n 180 |
| glimmer-30b | not run | not run | not run | not run | | |

\* All 45 are `context_exceeded` at 131,072; this is the context limit, not recall.

The intervals are 95% Wilson intervals, each bucket on its own. The ≤128K pooled column is computed from the
per-case outcomes in the summaries. No paired comparison against base-9b is made here: the gate is for tuned
against base, and these are different models.

### Large-student bake-off (runbook: eval-tools first, then eval-longctx at ≤128K)

**No winner yet.** Glimmer has 70 eval-tools cases and all of eval-longctx left.

1. **eval-tools.** On the 259 cases both models answered, the two are **tied**: 196 each. The paired table
   (Qwen3.8-27B right only 28, Glimmer right only 28, both right 168, both wrong 35) gives a difference of
   0.0 pts, Newcombe 95% CI [−5.7, +5.7], exact McNemar p = 1.0. On the full set Qwen3.8-27B scores 72.9%
   [67.9, 77.5].
2. **eval-longctx ≤128K.** Qwen3.8-27B: 84.4% [77.4, 89.6] (114/135). Glimmer: not run.

**Provisional order: Qwen3.8-27B first, Glimmer second.** Qwen3.8-27B is ahead only on completeness, since
the first criterion is tied so far. The ranking becomes final when Glimmer finishes eval-tools (329) and
eval-longctx. If eval-tools stays within its interval, ≤128K long-context recall decides. Glimmer's 0.5%
argument exactness is a separate warning for a training student. It is not a ranking criterion.

The ungated long-context tool set (60 cases) was not part of these runs.

## References

- Newcombe RG. Improved confidence intervals for the difference between binomial proportions based on paired
  data. *Statistics in Medicine* 1998;17:2635–2650 (method 10).
- Newcombe RG. Two-sided confidence intervals for the single proportion: comparison of seven methods.
  *Statistics in Medicine* 1998;17:857–872.
- Holm S. A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*
  1979;6:65–70.
- Armitage P, Berry G. *Statistical Methods in Medical Research*, 3rd ed., 1994, p. 138 (the paired example).
- Fagerland MW, Lydersen S, Laake P. Recommended tests and confidence intervals for paired binomial
  proportions. *Statistics in Medicine* 2014;33:2850–2875 (comparison of the paired intervals).
