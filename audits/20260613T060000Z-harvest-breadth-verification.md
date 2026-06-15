# Harvest breadth — Phase 1 baseline + Phase 3 verification (the finding)

**Branch:** `feat/harvest-breadth`. Honest result: the corpus STRUCTURE widened (intended 5
transitions, balanced), but the MODEL's behavior did not follow — breadth did **not** meaningfully
widen. This is a real finding about the model's failure distribution, reported with a proposal, not
papered over.

## Phase 1 — baseline breadth (the 44 pairs from #40)

`transition_histogram` over `gym-training.db`:

| transition | n | share |
|---|---|---|
| dict_wrap_unwrap | 43 | **97.7%** |
| json_stuffed_string | 1 | 2.3% |
| (fuzzy_rename, missing_discriminator, type_coercion, other) | 0 | 0% |

**2/6 types present, 97.7% one transition.** Breadth was the real job (confirmed the sprint's worry).

## Phase 3 — verification harvest (the breadth-targeted corpus, 63 runs → verify db)

Density: **33 unique pairs / 63 runs = 0.52/run** (held — even up from #40's 0.44). 33/33 distinct
context_hash (zero dedup loss). Export validates (33 JSONL lines). Achieved histogram:

| transition | n | share |
|---|---|---|
| dict_wrap_unwrap | 29 | **87.9%** |
| json_stuffed_string | 2 | 6.1% |
| other | 2 | 6.1% |
| fuzzy_rename / missing_discriminator / type_coercion | 0 | 0% |

**3/6 present (vs 2/6) — wider, but NOT meaningfully: still ~88% dict_wrap. PASS CONDITION NOT MET.**

## Why the targeted transitions didn't induce (root cause)

- **fuzzy_rename = 0.** Across 14 wrong-tool-name prompts the model emitted ZERO invalid names — it
  called the real tool every time. The local tool-calling path constrains output to the **valid
  tool-name set** (native `--jinja` grammar / GBNF), so a misnamed call is structurally near-
  impossible to emit. Fuzzy-rename can't be harvested from this tier.
- **missing_discriminator = 0.** The agent-intent `task_create` prompts produced **dict-wrap on
  `prompt`** (`{"prompt": {"prompt": …}}`) instead of omitting `type`. The model ADDS structure; it
  does not drop the discriminator. The error it actually makes is (again) dict-wrap.
- **type_coercion = 0.** Word-quoted limits didn't elicit a coerced scalar.
- **json_stuffed_string = 2.** The only targeted bait that worked (args-as-JSON-blob → the model
  stuffed `command: "{\"command\":…}"`). Partial — 2 of 10.

**The reframe:** dict-wrap dominance (88–98%) is the model's TRUE failure distribution under its
grammar-constrained tool calling, not an artifact of a narrow corpus. Training on dict-wrap pairs
is training on the real problem — the "one lesson" is largely the lesson that matters for THIS
model. The other transitions barely occur naturally.

## Proposal (what to add — don't paper over)

1. **Synthetic injection for the rare transitions.** The pair-smoke `--inject` path already
   fabricates a misnamed `ToolUseBlock` through the REAL adapter/repair path (`1367334`). Generalize
   it into a corpus-injection step that deterministically fabricates fuzzy_rename /
   missing_discriminator / type_coercion pairs, so a training set can be breadth-BALANCED without
   depending on model behavior the grammar suppresses. This is the realistic path to multi-
   transition coverage.
2. **Or train a dict-wrap specialist** on the natural distribution (defensible — it IS ~88–98% of
   real failures) and **evaluate on HELD-OUT dict-wrap shapes** (params/tools/nouns the corpus did
   NOT drill) to guard against noun-level Goodhart.
3. The overnight harvest (Phase 4 runbook) will be dict-wrap-dominated regardless; the runbook says
   so and flags injection as the prerequisite for a balanced set.

## Verdict

Density and dedup: green. Breadth: **honest fail** — the gym alone can't widen the transition mix
because the model's grammar-constrained failures are overwhelmingly dict-wrap. The valuable output
is knowing that, and that balance requires synthetic injection, before any LoRA spend.
