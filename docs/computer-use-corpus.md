# The computer-use candidate-table corpus — harvest spec

A corpus of real candidate tables, each annotated with the answer a human says
is correct, so that a chooser can be *scored* rather than trusted.

This document is the spec. It was written before any harvesting, because the
measurements below change what is worth harvesting.

---

## 1. Why milestone 2's table cannot be the corpus

Milestone 2 executed one real click. It offered **4 candidates**, from a window
containing one label and one button, with a goal that named the button.

A uniformly random chooser scores **25%** on a 4-candidate table. One sample at
25% chance distinguishes nothing at all: every chooser, including a coin flip,
is consistent with the observed result. The table proves the *path* executes. It
cannot score a *chooser*.

## 2. What was measured, and on what

Measured 2026-09-20 on the operator box, reading the live AT-SPI tree directly
and applying this repo's own `_CLICKABLE_ROLES`, `_EDITABLE_ROLES` and
`max_candidates` rules from `candidates.py` — not a model of them.

### 2.1 Real tables are much bigger than four

| app | tree nodes | click | type | **table size** |
|---|---:|---:|---:|---:|
| gnome-shell | 159 | 15 | 1 | 19 |
| gjs (Desktop Icons) | 39 | 0 | 0 | 3 |
| **gnome-calculator** | 93 | 32 | 2 | **37** |
| baobab | 34 | 4 | 0 | 7 |
| gnome-text-editor | 78 | 7 | 0 | 10 |
| gnome-system-monitor | 95 | 15 | 0 | 18 |

**min 3 · median 18 · max 37.**

Two consequences, both load-bearing elsewhere:

- A pocket calculator builds **37** candidates — three short of `max_candidates`
  and *above* a 36-label single-token alphabet (A–Z + 0–9). The alphabet
  overflow ruled on in `chooser.py` is the common case, not an edge case.
- Windows below ~10 candidates exist and are mostly trivial.

### 2.2 Table size is NOT what makes a table discriminating

Self-supervised check: for each candidate in each real table, treat it as the
correct answer and derive the goal from its own description, then ask
`RuleChooser` to recover it.

| app | N | recovered uniquely | tied | missed | discriminating |
|---|---:|---:|---:|---:|---:|
| gnome-shell | 18 | 16 | 1 | 1 | 11% |
| gjs | 3 | 3 | 0 | 0 | 0% |
| gnome-calculator | 35 | 35 | 0 | 0 | **0%** |
| baobab | 7 | 3 | 1 | 3 | **57%** |
| gnome-text-editor | 10 | 10 | 0 | 0 | 0% |
| gnome-system-monitor | 18 | 18 | 0 | 0 | 0% |
| **total** | **91** | **85** | 2 | 4 | **6%** |

**`RuleChooser` recovers its own goal uniquely on 93% of rows.** The
35-candidate calculator is **0%** discriminating; the 7-candidate baobab window
is **57%**.

So size is necessary and nowhere near sufficient. A corpus whose goals are
copied from candidate descriptions is ~94% trivial *regardless of how big the
tables are*, because `RuleChooser` scores by substring overlap and the goal
hands it the substring.

**The goal phrasing is the experiment. The table is just the setting.**

### 2.3 With task-language goals, the deterministic chooser abstains half the time

Nine goals phrased the way a person states a task ("add two numbers together",
"undo what I just did", "find out how much space is left"), against the same
real tables:

| app | N | abstained | picked |
|---|---:|---:|---:|
| gnome-shell | 18 | 4 | 5 |
| gjs | 3 | 8 | 1 |
| gnome-calculator | 35 | 3 | 6 |
| baobab | 7 | 5 | 4 |
| gnome-text-editor | 10 | 3 | 6 |
| gnome-system-monitor | 18 | 4 | 5 |
| **total** | | **27** | **27** |

**`RuleChooser` abstains on 50% of task-language goals.**

This splits the corpus into two populations that answer *different questions*,
and conflating them would produce a single meaningless accuracy number:

- **Abstain rows (~50%)** — `RuleChooser` returns `abstain` with confidence 0.0.
  There is nothing to pair against. The question here is **coverage**: does the
  classifier find the right answer where the rule finds nothing? This is where a
  fast chooser earns its place.
- **Picked rows (~50%)** — both choosers answer. The question here is
  **regression**: do they agree, and when they disagree, who is right?

## 3. Minimum table size

Stated as asked, with the reasoning.

A uniformly random chooser scores `1/N`. For a correct answer to be worth more
than luck, the chance floor has to sit well below the accuracy being claimed.
To detect an accuracy of 0.70 against the chance floor at α=0.05 and 80% power,
by one-sample binomial:

| N | chance floor | tables needed |
|---:|---:|---:|
| 4 | 25% | ~17 |
| 10 | 10% | ~9 |
| 20 | 5% | ~8 |
| 37 | 2.7% | ~7 |

**Minimum: N ≥ 10.** Below it the chance floor eats the signal — at N=4, one row
in four is correct by luck, so a 20-row corpus expects 5 lucky hits and
separating 5 from 8 needs far more rows than the annotation effort justifies.
Above N≈10 the returns flatten: going from 10 to 37 saves two tables.

The measured median real table is **18**, so N ≥ 10 is satisfied by most real
windows without special effort. Windows below it (gjs=3, baobab=7) are recorded
and excluded from headline scoring by a threshold applied **at scoring time**,
derived from the stored `candidate_count` rather than written onto the row —
so raising or lowering the floor never requires rewriting the corpus. They are
kept because baobab is the most *ambiguous* window measured, and ambiguity is
the scarce thing.

**N ≥ 10 is a floor, not a target, and it is the weaker of the two
requirements.** The binding one is §2.2: goals must be written in task language
by a human, not derived from the table.

## 4. Harvest spec

### 4.1 Apps

Installed and confirmed observable on this box:

**Harvest:** `gnome-calculator` (large, dense, repetitive labels),
`gnome-text-editor`, `gnome-system-monitor`, `baobab`, `nautilus`,
`gnome-control-center` (deep, many panes), `evince`, `gnome-disks`.

**Do not harvest:**

- `seahorse` — a key manager. Its accessible tree is key names, fingerprints and
  identities. Never a corpus row.
- `firefox` — `browser_*` is outside this milestone entirely, and browser chrome
  plus page content is where scraped text is most likely to be personal.
- Any window belonging to a running session of the operator's own work — mail,
  chat, terminals, editors with real content open.

Harvest from **clean instances opened for the purpose**, closed afterwards. A
corpus row is a permanent artifact; a window opened on real work is not a safe
thing to make permanent.

### 4.2 Volume

| population | target rows | rationale |
|---|---:|---|
| abstain rows | **50** | at median N=18 the chance floor is 5.6%; 30/50 correct is overwhelming against an expectation of ~3 |
| picked rows | **50** | detects a *large* regression; see the caveat |
| **total tables** | **~100** | at the measured 50/50 split |

**Stated honestly: 100 tables powers the coverage question well and the
regression question poorly.** A paired comparison (McNemar) needs ~25 discordant
pairs for a usable normal approximation. At 50 picked rows and a 30% disagreement
rate that is ~15 discordant pairs — enough to catch a large regression, not
enough to rule out a small one. Ruling out a small regression needs ~170 tables.

100 is the recommended first harvest because the coverage question is the one
that decides whether a classifier is worth having at all. If it passes, extend
the corpus before trusting the regression number. **Do not report a regression
verdict from 100 tables as though it were conclusive.**

### 4.3 Goals

The single most important rule, and the one the measurement in §2.2 exists to
justify:

- **Goals are written by a human, in task language, before seeing the table.**
- **A goal must not be copy-derived from any candidate's description.** If the
  goal contains the button's label verbatim, the row is trivial by construction.
- Aim for 3–5 goals per table, mixing: one the rule should get, one the rule
  should abstain on, one genuinely ambiguous between two candidates.

An automated check should flag any row whose goal shares more than a threshold
of content words with its annotated-correct candidate's description, and the
harvest tool should refuse to store it silently.

## 5. The correct answer is not the executed answer

The two differ exactly when the run was wrong, which is the case the corpus
exists to capture. A schema that can only record what happened cannot score
anything.

Four annotation states, deliberately distinct:

| state | meaning |
|---|---|
| *(unannotated)* | no human has looked at this row yet. **The default.** Never confuse with "no correct answer". |
| a candidate id | that candidate was the right action |
| `none_correct` | the table was usable, but no candidate in it was the right action — `abstain` was correct |
| `table_unusable` | the observation should not have produced a table at all |

`none_correct` and `table_unusable` are separate because they score differently:
on the first, a chooser that abstains is **right**; on the second, the fault is
upstream of the chooser and the row must be excluded from scoring entirely, not
counted as a loss.

A corpus that collapses unannotated into `none_correct` silently scores every
un-reviewed row as "abstain was correct" — which would make a chooser that
always abstains look perfect on an unannotated corpus.

## 6. Table expressiveness — DIAGNOSED 2026-09-20, deliberately NOT fixed here

Verified independently at `424edc1`, by AST over the source rather than by
reading prose.

| | |
|---|---|
| `ACTION_MODELS` declares | `observe, click, scroll, press_key, type_text, invoke_menu, verify` (7) |
| `build_candidates` emits | `computer_click`, `computer_type_text` (caller-supplied text only), `computer_press_key` |
| `press_key` offers | exactly `return`, `tab`, `escape` |
| `ALLOWED_KEYS` declares | **15** keys — so **12 are unreachable** |
| modifier combinations | **none exist at all** — Ctrl+Z, Ctrl+S have no representation |

**`scroll` and `invoke_menu` can never be selected.** All three `Candidate(`
constructions in the entire `src/` tree are inside `build_candidates`, and it
has exactly one caller (`loop.py:118`). There is no other path into a table.
Both verbs nonetheless carry full models, schemas, gate wiring, and
operator-facing consent phrases — `_VERB_PHRASES` offers to grant "scroll
anything" and "use any menu item" for capabilities that cannot be exercised.

### Was the narrowing deliberate? No — the docstring asserts the opposite

`actions.py` carries a section headed **"WHAT IS IN v1, AND WHAT IS
DELIBERATELY NOT"**, and lists `scroll` and `invoke_menu` under **In**.
`build_candidates`' own docstring says "Every bounded action worth offering for
this snapshot", which reads as completeness.

So this is not an undocumented decision. It is a documented claim that the code
contradicts, under the one heading whose entire purpose is to separate
deliberate omission from oversight. That is §1's orphan shape — a capability
fully built and unreachable — inside the subsystem whose premise is that code
and claim must not diverge.

**Not fixed in this PR, on instruction.** Fixing `build_candidates` is
deterministic, adds no dependency, needs no licence and costs no latency — and
it changes the measurement, so it belongs in its own change with its own
before/after.

### Why this reorders the plan

Part of the measured 50% abstain rate is **not chooser weakness — it is a table
that cannot express the goal**. Those rows are `none_correct` by construction
and no chooser, local or hosted, will ever improve them.

Worked example. Goal: *"undo my last change"* in a text editor. The table offers
"Click the button labelled 'Open'", "Click the text area", "Press escape".
`RuleChooser` scores `{undo, last, change}`, gets zero overlap, abstains. A
classifier reads the same table and finds nothing that undoes anything, because
`invoke_menu(['Edit','Undo'])` was never offered and Ctrl+Z is not a candidate.
**Both abstain correctly. The gap was never the chooser.**

### Revised sequence

1. Harvest with `none_correct_reason` recorded (this PR).
2. Split the `none_correct` rows by reason.
3. If `verb_not_offered` / `key_not_offered` are a large share, **fix
   `build_candidates` first and re-measure.**
4. Only then judge a classifier — against the smaller remaining gap, and
   against an **embeddings baseline** rather than against substring matching.

Judging a classifier before step 3 measures the table's limits and reports them
as the chooser's.

## 7. Label provenance — and the harvest order

`label_source` is required whenever `correct_id` is set. There is **no
default**: `human` would be the comfortable one and the wrong one.

| value | meaning |
|---|---|
| `human` | a person decided, having looked at the table. **The only labels a score may be called accuracy against.** |
| `model` | a model proposed it, nobody checked. A score here is **agreement with a model**. |
| `model_confirmed` | a model proposed it and a person confirmed or corrected it. Weaker than `human` — the person saw a suggestion first, and anchoring is real — so it is tracked separately rather than folded into either neighbour. |

**Every summary and score reports its label mix, in words.** A number over
`model` rows prints `AGREEMENT WITH A MODEL — not accuracy`, and a mixed corpus
refuses to be summarised as either. A reader who skims must not be able to pick
up the wrong word, so the rule is in the output rather than in a footnote, and a
test asserts that the word "accuracy" never appears except inside a denial of it.

The reason is the one `tests/fixtures/divergence_traces.py` already states: *a
calibration round that cannot tell the two apart is calibrating against its own
author.* A chooser graded against labels another model produced can only be
measured on how alike they are.

### Harvest order — do not reorder

1. **Will hand-labels ~20 tables COLD**, before any model proposal exists. This
   is the calibration set and it must not see a suggestion first — that is what
   makes it a control rather than a confirmation.
2. **Claude labels all ~100**, recorded as `model`.
3. **Compare against the 20.** Report agreement.
4. **Will decides** whether the remaining ~80 are usable as-is, need confirming
   (`model_confirmed`), or get hand-labelled.

⚠ Step 1 precedes step 2 for a reason that cannot be recovered afterwards: once
a human has seen a model's answer, the label is `model_confirmed` at best and
the calibration set no longer exists.

## 8. Provenance on every row

Four fields exist so a row cannot quietly mean something other than it appears
to. Each one is stored, not inferred at read time.

| field | why it cannot be left out |
|---|---|
| `driver_kind` | **`fixture` vs `cua`.** Every test in this repo drives the loop with `FixtureDriver`. If those rows entered the corpus indistinguishably from real ones, a chooser would be scored against tables the test suite invented. `tests/fixtures/divergence_traces.py` already states this rule: *"a calibration round that cannot tell the two apart is calibrating against its own author."* It is **derived from the driver**, never supplied by a caller, because the row it would be wrong on is exactly a fixture row that reads as real. |
| `goal_source` | `human` \| `derived` \| `unknown`. §2.2 measures that a description-derived goal makes 93% of rows trivial, so goal provenance *is* the experiment. The default is `unknown`, not `human` — a row that merely forgot the flag must not read as one a person vouched for. |
| `harvest_session` | Groups rows captured under the same discipline. Two sessions with different goal-writing rigour mix irreversibly without it. |
| `code_fingerprint` | A hash of `_CLICKABLE_ROLES` and `_EDITABLE_ROLES` — the constants that *define* which elements become candidates at all. Change them and a row harvested beforehand describes a different universe of options; scoring across the change silently blends two experiments. |

`table_fingerprint` is also stored, over the sorted `(id, description)` pairs, so
the same window captured twice is identifiable rather than double-counted.

## 9. Redaction

Candidate descriptions are built from AT-SPI labels scraped off a live desktop:
window titles, button text, field labels, and sometimes document content. A
corpus is a permanent, re-readable artifact, so what goes in it is a privacy
decision, not a formatting one.

- Never store element tokens (snapshot-bound, meaningless later) or raw
  arguments. The chooser's view — `{id, description}` — is the whole record.
- Never store a typed payload. `text_to_type` is the caller's, not the table's.
- Apply the project's existing redaction before writing, not after.
- The app allowlist in §4.1 is the primary control; redaction is the backstop.

## 10. What this spec does not cover

`hf-server`, SimpleJev, any classifier, and any scoring harness. Those come
after a corpus exists and after the licence question is answered. This document
describes the corpus and nothing downstream of it.
