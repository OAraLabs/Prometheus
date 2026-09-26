# The model ladder

Which task classes can each local model size handle? The model ladder answers that with
measurements rather than impressions, so Instinct's routing can learn from them. It is a
frozen suite of eight task classes, run through Prometheus's real pipeline against one
served model at a time, with every run ending in a **verdict** and landing in
`telemetry.db`.

This page defines the suite (WP-2.1). The full runs are WP-2.2 (the 4090) and WP-2.3
(Apple Silicon); the **first run** (`first_run` in `rungs.yaml`) is the production 27B, the same
checkpoint in ternary (Bonsai 2), Qwen3.5-9B and Ornith-1.5-9B — see *First run* below. The first cut populates four of the eight classes; the other four are defined —
budgets and success criterion — and deferred (see below).

- Suite: [`gym/ladder/v1/`](../gym/ladder/v1/) — `suite.yaml` plus one file per class.
- Rungs and the judge pin: [`gym/ladder/rungs.yaml`](../gym/ladder/rungs.yaml).
- Harness: [`src/prometheus/gym/ladder/`](../src/prometheus/gym/ladder/).
- Runner: `uv run python scripts/ladder_run.py --help`.
- Reports: `gym/results/ladder/<run-label>.md` (they name no hosts).

## What it is built on

The repo already had two evaluation frameworks. The ladder reuses both rather than adding
a third.

| | `prometheus.evals` (golden dataset) | `prometheus.gym` (tool-calling gym) |
|---|---|---|
| Tasks | 26 built-in: 21 single-tool (tier 1), 5 multi-step (tier 2); 2 need the network | Frozen YAML task sets (`gym/tasksets/v1`, `v2`, dictwrap, harvest) |
| Verdict | ToolUsage (deterministic) + TaskCompletion and NoHallucination, **LLM-judged** | **Deterministic predicates only**, no judge |
| Pipeline | `AgentLoop` with the configured model | Real provider, adapter (tier from config), SecurityGate, agent loop |
| Variables | none — one model, nightly | a manifest changes exactly one variable |
| Records | JSON files + `trends.db`, judge provenance per file | `gym.db` (run rows), KV-cache provenance, emission vs execution |
| Runs | nightly cron on the mini (03:00, 06:00), judge pinned by `evals.judge_model` | by hand, `scripts/gym_run.py` |

What they covered: bash / file / grep / glob / cron-list / todo single calls and a few short
chains (evals); tool-call *shape* — argument wrapping, namespaces, `task_create`, resilience
after a collapse, LCM query syntax (gym). What neither covered: questions without tools,
acceptance-command verdicts, offline web research, memory recall across layers, scheduling
checked against the registry, long coding runs, or per-run records in `telemetry.db`.

The ladder takes from the gym its pipeline construction (`build_pipeline`,
`preflight_endpoint`, the KV-cache probe), its predicates and dual emission/execution
scoring, and its conversation-seed replay; and from the evals framework its judge
(`PrometheusJudge`: constrained JSON decoding, provenance). What it adds is in
`src/prometheus/gym/ladder/`: the class-per-file suite, acceptance commands, the sandbox and
fixture tools, the judge pin, and recording.

## The eight classes

All eight classes the WP proposed are **defined** — each with its budgets and a written success
criterion in `suite.yaml`. The first cut **populates four** of them and **defers four**: a deferred
class has its definition and nothing else, and the loader refuses a task file for it until a later
work package makes it active.

### Populated (suite v1, sha `20974af2f6e0`)

The first run ran at two earlier shas: `45df225d3b36` (the 27B, Bonsai and 9B smokes and all
three sweeps) and `b71c4bc2c390` (Ornith's smoke), which adds two wrong-answer proofs to
`qa-py-mutable-default` and widens its answer shape. `20974af2f6e0` changes only the wording of
the `qa` and `single_tool` success criteria, which said a reply without its answer line is a
fail; no code reads that text, and the reader never did that (*Verdicts*).

**92 tasks.** Every task names its difficulty within its class; the report breaks pass rates out
by it. Three per class are tagged `smoke: true`.

| class | tasks | easy / medium / hard | decided by | proofs: wrong · format miss · right | success criterion (abridged) |
|---|---:|---|---|---|---|
| `qa` — plain Q&A | 24 | 7 / 9 / 8 | answer line 18, judge 6 | 112 · 38 · 130 | the reply's `ANSWER:` line states the right value; explanations scored ≥ 0.7 by the pinned judge against a rubric and reference |
| `single_tool` — one call is enough | 24 | 8 / 9 / 7 | answer line 21, file predicates 2, acceptance 1 | 140 · 31 · 160 | the `ANSWER:` line states the unguessable planted value, or the named file holds exactly the requested content |
| `multi_step` — dependent calls | 22 | 7 / 8 / 7 | end-state predicates 16, answer line 5, acceptance 1 | 60 · 16 · 41 | the end state (output file, or the `ANSWER:` line) is exactly right; reaching it takes 2–5 dependent calls |
| `file_edit` — edit until a command passes | 22 | 7 / 8 / 7 | acceptance 22 | 55 · – · – | the harness's acceptance tests run and pass after the agent stops, against pristine copies of any visible tests |

The proofs are replies and end states each check is proven against (see *Verdicts*): **wrong**
ones it must fail (wrong answers and wrong files), **format misses** it must neither credit nor
fail, and **right** answers in other spellings it must credit. Every mechanically checked task has
at least one wrong proof and does not pass untouched; every answer-line task has at least three right
answers, two wrong ones and one format miss. The six judge-only explanations have none by nature.

### Defined, deferred

| class | budget (rounds / tool calls / time) | success criterion |
|---|---|---|
| `web_research` | 12 / 14 / 6 min | the reply states the facts asked for as given on the authoritative page(s), and none of the distractor values. Offline tasks answer from fixture pages on reserved `.example` hosts through the real `web_fetch` / `web_search` schemas; live tasks are opt-in and declare whether their answer changes over time; syntheses go to the pinned judge only after the fact predicates pass |
| `memory_recall` | 8 / 8 / 3 min | the reply states the planted fact — the latest value where a later session superseded an earlier one — and not the superseded or distractor values; planted in `MEMORY.md` / `USER.md`, earlier sessions (`lcm_grep`), the wiki (`wiki_query`), or early in a seeded conversation; the verdict never reads the seed |
| `scheduling` | 8 / 8 / 3 min | after the run the sandbox's cron registry holds exactly the requested change — the named job, a schedule firing at the requested times in any equivalent spelling, the requested command and enabled state, every other job as it was — checked on the registry, not the reply; inspection tasks are checked on the reply |
| `long_haul` | 40 / 60 / 25 min | a hidden acceptance suite, written into the workspace only after the agent stops, exits 0, within the budget |

The harness already implements what these criteria need — the offline web tools, the memory and
wiki seeding, the cron-registry predicates, hidden acceptance files — and its unit tests cover
them, so populating a deferred class is authoring, not harness work.

### Refinements to the WP's proposal

- **Plain Q&A** is mostly exact answers checked by predicates, with a minority of open-ended
  explanations graded by the judge. A predicate that could pass a wrong answer measures nothing,
  so every text-checked task lists wrong answers the check must reject.
- **Web research** will be offline by default: fixture pages on reserved `.example` hosts are the
  only way a web verdict is reproducible on two machines months apart.
- **Memory recall** covers every layer the model can reach. Passive recall is not a model skill and
  is off in the ladder, as it is in the gym and evals.
- **Scheduling** is checked against what the scheduler would actually run, not the reply.
- **File edits** and **long-haul coding** differ in size and budget: an edit is one to four files
  and a visible or hidden test; a long-haul task is a multi-part program built from a written spec.

Budgets (`suite.yaml`): `max_rounds` is model calls; `max_tool_calls` is the loop's tool-iteration
cap; `timeout_s` is the wall-clock budget. Running out of time or rounds is a **fail** — the model
could not finish within what it was given.

## Verdicts

Every task ends in one of five verdicts (`pass`, `format_miss`, `fail`, `unscored`, `error` — table
below). The checks are applied in this order:

1. **Predicates** — deterministic checks over the finished run: the gym's (`expect_tool`,
   `expect_file`, ...) and the ladder's (`expect_answer`, `expect_text_any/all/regex`,
   `forbid_text`, `forbid_text_regex`, `expect_tool_any`, `expect_tools_all`,
   `expect_file_regex`, `expect_file_absent`, `expect_cron_job`, `forbid_cron_job`).

   **A value is read from the reply's `ANSWER:` line.** Every task whose verdict is a value
   in the reply ends its prompt with *End your reply with a final line `ANSWER: <…>`*, and
   `expect_answer` full-matches one short pattern against that line — the last line that
   starts with `ANSWER` or `Final answer` and a colon (or ends a sentence with one: `…is 48217.
   ANSWER: 48217`), undecorated (bold, code ticks, quotes, a
   trailing full stop, `$…$`, `\boxed{…}`). The reply may discuss anything above it:
   distractors, a derivation, rows quoted from a file. Two rounds of the audit showed why: a
   pattern searched anywhere in free text either passes a reply that mentions the right value
   and commits to a wrong one (`x = -16`, "7." as a list marker in a wrong count of 8) or fails
   a right reply that walks the file in order — and tightening one direction broke the other.
   **Format misses stay separate from wrong answers — in every table and in the routing
   data.** The reader credits a pass only when the committed value is unambiguously right, and
   a fail only when it is unambiguously a wrong value; everything else is a **`format_miss`**
   (`success` NULL, telemetry outcome `partial`) — never a wrong answer.

   | what the reply gives | verdict |
   |---|---|
   | an `ANSWER:` line with exactly the right value (any spelling the task accepts) | pass |
   | an `ANSWER:` line with exactly a value of the answer's kind (`answer_shape`), or one clean token, that is wrong | fail |
   | an `ANSWER:` line holding a sentence with exactly one value of the answer's kind | scored on that value |
   | an `ANSWER:` line that denies there is an answer and names no value of its kind (`No request ID found for HTTP status 503.`) | fail |
   | an `ANSWER:` line with a hedge (`Saturn or Jupiter`), a list, a negation next to a value (`It is not 48217`) or a parenthetical (`12,600 (minutes)`, `3712 (from base.ini)`) | format miss |
   | no line; the last line is exactly the right value, or states it as its only value of that kind (`The configured port is 48217.`) | pass — `answer_format_ok = false` |
   | no line; the last line is exactly a wrong value of the answer's kind (`1071`) | fail — `answer_format_ok = false` |
   | no line; anything else — a procedure, a question back, a sign-off, a quoted file, a list of two or more items | format miss |

   Prose without an answer line is never failed: "an hour has 3,600 seconds, so multiply…" is
   a procedure, not a wrong answer of 3,600. A parenthetical is never stripped into a pass,
   because it may be commentary, a qualifier or a hedge, and no syntax tells them apart.

   Before any of that, the answer line is read generously: a value on the lines below a bare
   `ANSWER:` label (a tight list, a fenced block, display math — each read whole, so a list of two
   candidates is a format miss; a loose list is a known limit below), LaTeX (`$…$`, `\boxed{}`, `\text{}`, `\%`, `1{,}081`), the
   prompt's own `<…>` placeholder brackets copied, a value wholly inside one pair of brackets (a
   tuple: `(1, 2, 1, 4)`), and Unicode hyphens are all read as the value they spell. None of those rules can turn a wrong value into a right one. A list is never
   decided by its last item, with a label or without: that would make the verdict depend on the
   order the model listed its candidates in (a loose list under a bare label still is, by its
   first — see the known limits).
2. **Acceptance tests** — `{unittest} <modules>`, run by the *harness* in the workspace after
   the agent stops, through its own runner (`src/prometheus/gym/ladder/accept.py`). An exit
   code of 0 proves nothing on its own, so the runner does not rely on one:
   - it starts an isolated interpreter (`-I`) and appends the workspace *after* the standard
     library, so a `unittest.py` or `json.py` the agent left behind cannot replace the real one;
   - it loads and runs inside `except BaseException`, so code that calls `sys.exit(0)` at import
     (a module-level `main()` with no `__main__` guard) fails instead of passing;
   - the verdict comes from a result file outside the workspace, and at least one test must
     have **run** with none failing. Code that kills the interpreter leaves no result: a fail.
   Tests the agent could see are restored from `acceptance_files` first — over a symlink, a
   directory, or a package that would shadow the module — so editing them buys nothing.
3. **Judge** — only where no command or predicate can decide, and only after the deterministic
   checks have passed. Only a finite `score` in [0, 1] under that key counts; `PrometheusJudge`
   defaults a missing score to 0.0 and clamps an out-of-range one, and the ladder refuses both.
   A task never carries both an acceptance command and a judge.

| verdict | `success` | when |
|---|---|---|
| `pass` | 1 | every check that applies passed |
| `format_miss` | NULL | the reply had no usable `ANSWER:` line and its answer could not be isolated — the model's answer is unknown, not wrong (outcome `partial`) |
| `fail` | 0 | a check failed; or the run stopped without the model finishing — out of time, rounds or tool calls, stopped by the loop (repeat halt, circuit breaker, empty replies, divergence halt), or its final reply was the provider's reasoning fallback |
| `unscored` | NULL | a check could not run — the judge errored, timed out, returned no 0–1 score, or was not pinned; or the harness could not start the acceptance runner |
| `error` | NULL | the run could not be carried out — the provider raised, the endpoint stopped answering (checked with a one-token request after a timeout; the ladder then stops), or the harness failed while deciding |

What the model did to its workspace is never a harness failure: a directory where a file should
be, a symlinked or read-only test, code that exits at import — each fails the run.

A verdict reads only what the **model** produced in the run:

- seeded conversation turns are context — a model that answers nothing does not inherit the last
  seeded reply, and seeded tool calls do not satisfy tool predicates;
- the messages the agent loop writes itself when it stops a turn ("Tool iteration limit
  reached (4/3)…", "Halted: no progress…", "Circuit breaker tripped…") are not the model's answer.
  The loop gives them no marker, so the ladder recognises their fixed opening words
  (`runner.LOOP_HALTS`); a test pins each one against the engine source, so a reworded message
  fails a test instead of being scored;
- when a thinking model spends the final round's whole output budget reasoning, the llama.cpp
  provider returns the unfinished reasoning as the reply and files a `silent_failures` row. A
  right value mentioned along the way is not an answer: the run fails as `reasoning_fallback`.
  The ladder takes that report from the run's own telemetry handle as it is filed, never from
  the table by time, so another writer's row in a shared database is never this run's.
  The preflight also runs the provider's thinking-suppression probe (as the daemon does at boot),
  records the result, and refuses a rung run whose template ignores the suppression flag.

**Every mechanical verdict is proven to discriminate.** `tests/test_ladder_suite.py` replays each
task through the real predicates and the real acceptance runner, without running a model
(`prometheus.gym.ladder.selfcheck`):

- the untouched setup, with no answer, must **not pass** — an end-state task fails, and an
  answer-line task is a format miss (no answer is not a wrong answer);
- each `reference.wrong_answers` entry (an off-by-one number, a sign slip, a distractor value, a
  distractor stated as the answer) must **fail**;
- each `reference.wrong_files` entry written over the reference (a partial fix, a blind
  replace-all, a subtly buggy implementation) must **fail**;
- each `reference.right_answers` entry (the same answer phrased another way: terse, a sentence,
  bold, `12,600.0`) must **pass** — the guard against a check too strict to credit a right answer;
- each `reference.format_misses` entry (a hedge, a list of candidates, prose with two values and
  no answer line) must be a **format miss** — neither credited nor failed;
- the reference solution must **pass**.

### The answer reader is frozen — known limits

The reader (`verdict.read_answer`) was rewritten twice against invented replies; after the
migration to the committed-value rules it is **frozen** (2026-09-25). Three independent checks
probed it through every answer-line task. The one defect they found that broke the format-miss
rule itself — a list of candidates decided by whichever item came last — was fixed and pinned;
everything else they found is listed here instead of fixed. From here on the reader changes only
where it misreads a **real** reply (the smoke's hand-check), never for an invented one.

Real replies are the only thing that changed it after the freeze. Over the first run, two blind
graders read **333 real replies** (every answer-line reply of every rung's smoke, plus 15 per tier
from each tier sweep) and disagreed with the reader on **two**, both fixed and pinned verbatim:

- a qwen2.5:7b dry run, before the first run, found four misreads: `ANSWER: (1, 2, 1, 4)` and two
  committed "there is none" answers (`ANSWER: No request ID found for HTTP status 503.`) had been
  format misses — all wrong answers, now failed; `…is 48217. ANSWER: 48217` passed but was
  recorded as missing its answer line;
- Bonsai 2 27B: "…is defined in `src/billing/settle.py`. Let me confirm there's no other
  definition elsewhere." — the "no" in the second sentence had made a stated answer a format miss.
  A negation now disqualifies a value only in the sentence that states it;
- Ornith-1.5-9B: "`print` displays the tuple: `(1, 2, 1, 3)`. ANSWER: (1, 2, 1, 3)" — credited,
  though the program prints `1 2 1 3`. Fixed in the TASK, not the reader: for this exact-output
  question a bracketed sequence is a value of the answer's kind, and the tuple and its fixed
  sibling are wrong-answer proofs (suite sha `45df225d3b36` → `b71c4bc2c390`; re-reading every
  stored reply to that task changes only that one row).

Each limit says which way it moves a score. *Down* is the costly direction: a right answer counted
as wrong.

| limit | example | direction |
|---|---|---|
| the hedge words are a closed list (`or`, `either`, `and/or`, `maybe`, `possibly`, `perhaps`, `probably`) | `ANSWER: likely cinder-coast-17` fails; lineless `It might be aurora-basin-42.` passes | both |
| a full-width colon is not an answer line | `ANSWER：aurora-basin-42` passes lineless; the same with a wrong value is a format miss | up |
| a one-word echo of the prompt's placeholder counts as one clean wrong token | `ANSWER: <region>` fails (`<the region>` is a format miss) | down |
| `no remainder` reads as a negation | `ANSWER: 56, no remainder` is a format miss | toward format miss |
| a reply ending in a table is read by its last row | a table of candidates ending on the right row passes; ending on another row it is a format miss | up |
| lineless credit reads the last line only | a closing note naming the right value after a wrong answer passes | up |
| a numbered procedure is a list | `1. …` `2. …` `3. So it is 56` with no answer line is a format miss | toward format miss |
| LaTeX `\frac` is not normalised | fraction answers must carry it in `answer_shape` (qa-algebra-linear does) | — |
| a LOOSE list under a bare `ANSWER:` label (blank lines between items) is read as its first item | right item first passes, wrong item first fails; the tight list is a format miss | both |
| a one-word heading under a bare label is read as a clean wrong token | `ANSWER:` then `Calculation:` then the working fails instead of falling back to the last line | down |
| a comma list wrapped one item per line is read by its first line | `ANSWER: V2_003_add_index.sql,` then the other two names fails | down |
| negation words are ASCII only (`isn't`, not `isn’t`) and a closed list (`couldn't find` is not one) | `The port isn’t 48217.` passes; `There isn't a 503` fails while `I couldn't find a 503` is a format miss | both |

The last four rows came from the pre-PR review, on invented replies. None of the first run's
2,160 stored replies (900 of them on answer-line tasks) has a bare `ANSWER:` label, a wrapped
answer line or a curly-apostrophe negation near its answer, so none of its verdicts rests on
them; they stay limits under the freeze until a real reply hits one.

Task spellings the checks found the reader fails although the value is right (**down**) — each is
kept as written until a real reply uses it: a reversal spelled letter by letter (qa-reverse-word);
a grouped (`M CM XC IV`) or additive (`MDCCCCLXXXXIV`) numeral (qa-roman-numeral); a race order
with arrows and no spaces (qa-logic-race-order); `12h25` (qa-train-arrival); `Thu`
(qa-weekday-of-date); a singular unit (qa-area-units); `0.96`, `64/4`, `fifty-six`
(qa-percent-up-down, qa-algebra-linear, qa-number-sequence); a bare basename where the prompt asks
for the path (st-grep-unique-constant, st-grep-defines-function); a timeout in seconds
(st-read-yaml-staging-timeout); the first name alone (st-read-meeting-owner). A shell command
committed on the answer line fails, not a format miss, when an awk field (`$2`, `$4`) is its only
number (st-count-warn-lines, st-read-config-port, st-run-report-net, st-csv-region-quarter). And
one in the other direction (**up**): a comma list pairing the right value with a look-alike that
is not a value of the answer's kind passes (`heron-staging-3318, heron-prod` in
st-read-manifest-build-tag; the token and its seed in st-run-token-script).

## Judge

The judge is `PrometheusJudge` (OpenAI-compatible, JSON-schema constrained decoding) and it is
**pinned**:

- a run with judged tasks needs `--judge-model` and `--judge-base-url` — or `--no-judge`, which
  records those tasks `unscored`;
- the runner refuses a judge whose model matches the contestant (compared after normalising
  paths, `.gguf` and `:latest`, so a GGUF path and its basename are the same model);
- the runner refuses a pin the judge endpoint does not serve. llama-server ignores the
  request's `model` field and answers with whatever it loaded, so a pin naming one model while
  the endpoint serves another would record a judge that never graded anything;
- the judge's provenance (`model`, `pinned`) is stored with every judged row; the endpoint is
  not, and a judge error is stored with its URL replaced.

For the rungs, the pin is in `rungs.yaml`: **`qwen2.5:14b-instruct` on the mini's ollama**, one
judge for every rung so judged pass rates compare across rungs. It is not any rung's model
(tested) and it is a different generation from all of them. It is weaker than the 27B rung it
grades; that is why judged tasks are a minority (six of 92, all in `qa`), why each rubric states
concrete pass/fail criteria and carries a reference answer, and why a report shows judged and
mechanical verdicts separately (the `decided by` column). A `--rung` run cannot override the pin.

**Where the judge runs is undecided, and until it is, rung runs use `--no-judge`** (the six judged
tasks are recorded `unscored`; `--no-judge` works with `--rung`). The first run serves three rungs
on the mini's 3090 Ti (one at a time), and at Ollama's default context there (32k, chosen from VRAM) this judge
needs about 14.8 GiB — it does not fit beside any rung in the ~11.7 GiB the card has free (an
earlier version of this page said it did; it does not). Options for the full runs: re-pin to the
already-resident `qwen2.5:7b-instruct` (no extra memory, weaker); serve the 14b's existing file
with the mini's llama-server at 8k context in a window with its cover-traffic user paused; or a
judge on another machine.

## Where each field lives in `telemetry.db`

The schema is unchanged. Each run leaves three kinds of rows, all carrying the run's
`session_id` (`ladder:<label>:<task>:<run>:<nonce>`):

- `tool_calls` — one per tool call, written by the real agent loop;
- `subsystem_runs` with `subsystem='agent_loop'`, `operation='loop_round'` — one per model call,
  with its tokens in `input_tokens` / `output_tokens`;
- `subsystem_runs` with `subsystem='model_ladder'` — the run's summary, written by the ladder
  and read back after writing (the underlying `record_run` swallows write errors).

| field (WP-2.1) | where | column? |
|---|---|---|
| task class | `subsystem_runs.operation` on the summary row | yes |
| model | `subsystem_runs.model` (requested); `summary_json.served_model` (what the endpoint reports) | yes |
| success | `subsystem_runs.outcome` (`success` / `failed` / `partial` = a format miss / `skipped` = unscored or error) and `summary_json.success`, `verdict` | yes |
| time | `subsystem_runs.duration_ms` on the summary row | yes |
| tokens | per round: `input_tokens` / `output_tokens` on the run's `loop_round` rows; per run: `summary_json.input_tokens` / `output_tokens` | per round only |
| quantization | `summary_json.quantization` (+ `quantization_source`) | **no column** |
| adapter strictness tier | `summary_json.adapter_tier` + `adapter_strictness` | **no column** |
| tool-call success | `summary_json.tool_call_success` (+ the counts); derivable from `tool_calls` | **no column** |
| repairs needed | `summary_json.repairs` | **no column** |
| rounds | `summary_json.rounds`; derivable from `loop_round` rows | **no column** |

The run's token totals stay off the summary row's columns on purpose: `usage_rollup`
(`/api/usage`) sums every `subsystem_runs` row with tokens, so a copy there would count each run
twice.

`summary_json` also carries the suite sha, the harness commit (`-dirty` when the ladder code
differs from it), run label, rung, KV-cache provenance, the thinking-suppression probe result,
the bash write-floor state, verdict source and reasons, the acceptance result (tests run /
failed), the judge's score and provenance (model and `pinned`; the endpoint is not stored),
`stopped_by` (done / round_cap / tool_call_cap / timeout / repeat_halt / circuit_breaker /
empty_response / divergence_halt / parse_disagreement / boundary_escape / context_overflow /
reasoning_fallback / error — the loop's own halts are `runner.LOOP_HALTS`), and a bounded trace of
the calls.
Errors and reasons have URLs and IP addresses replaced (`<url>`, `<ip>`): provider errors quote
the full request URL, and telemetry.db is backed up and copied between machines.

Definitions:

- **tool-call success** = ok calls ÷ (calls − policy denials − non-zero exits), telemetry's own
  denominator rule (`NON_CALL_FAILURE_TYPES`). Calls the loop's repeat guard answered `BLOCKED`
  get no `tool_calls` row; the ladder counts them from the transcript as failed calls
  (`tool_calls_blocked`). NULL when the run made no calls: undefined, not 0.
- **repairs needed** = tool calls whose executed form differed from what the model emitted — the
  adapter's repair or unwrap, observed per call. (The loop's own per-call repair count, kept as
  `repair_ops_telemetry`, is written only on some paths.) At the light tier the validator is
  `NONE`, so this is 0 unless unwrapping applies.
- **tokens**: a provider that reports no usage leaves 0 on every round. A real round always has
  input tokens, so an all-zero run is stored as NULL with `tokens_source='unreported'` — never as
  a measured 0.
- **rounds** counts `operation='loop_round'` only; the loop files other per-turn rows under
  `agent_loop` (tool advertisement, breaker trips).
- **node**: every row names the machine (`node_id`) when the machine has a node identity; the
  sandbox keeps the real node directory for that and never creates one.
- **answer format**: `summary_json.answer_format_ok` — for tasks checked on an `ANSWER:` line,
  whether the reply had one. NULL for other tasks and for runs that never finished.

The default database is **`~/.prometheus/ladder/telemetry.db`**, not the live one: ladder runs
include deliberately hard tasks and small models failing them, and mixing those into the live
`tool_calls` table would distort the dashboards and SENTINEL's inputs — the reason the gym keeps
`gym.db` separate. `--telemetry-db ~/.prometheus/telemetry.db` writes the live file; the ladder
then switches off the time-window attribution below, because the daemon writes there too.

### Known gaps the ladder works around, not fixes

All of these are in daemon code (engine, adapter or providers), which this work package does not
change.

- Early-exit tool calls (`permission_denied`, `validation_failed`, `unknown_tool`,
  `tool_exception`, `hook_blocked`, `tool_timeout`, ...) and `_loop_transition` rows are written
  to `tool_calls` **without a session_id**; only the final execution path sets it. The ladder
  attributes session-less rows inside a run's time window to that run (one ladder writer per DB
  is enforced by a lock) and counts them in `tool_calls_unattributed`.
- Until #577 `OllamaProvider` never asked for streamed usage (`stream_options.include_usage`),
  so every ollama round recorded 0 tokens and a ladder run through ollama recorded tokens as
  unreported. #577 (on main since the first run's code base) asks for it and sends `max_tokens`;
  no first-run rung was served through Ollama.
- The loop's own halt messages carry no marker (recognised by their wording, pinned by a test),
  and its repeat guard writes no `tool_calls` row for a blocked call (counted from the
  transcript).
- The loop writes a repaired call's repair count only when the call executes (or times out); a
  repaired call that then fails is recorded with 0. The ladder counts repairs from the per-call
  observer instead.
- **Tier `full` could not read XML tool calls — until #582, after the first run.** At `full` the
  tools are withheld from the request, so llama-server's own parser has nothing to parse against,
  and until WP-X.28 PR 1 (#582) Prometheus's extractor read JSON only. A model trained on
  Qwen3-Coder XML calls (Qwen3.5 / 3.8, Bonsai 2) keeps writing them: a reply that was only the XML
  was stripped and retried (`stripped_to_empty`), and XML after prose was deleted silently — the
  prose became the answer. Seen in real Bonsai 2 output (the smoke's `parse_disagreement` halts and
  "Let me use Python…" format misses). #582 reads the XML at `full`; the first run's smokes and
  sweeps were measured before it, and the ladder has not re-measured `full` since. The ladder
  counts XML-markup turns per run either way, so a re-sweep shows whether the calls are now read
  (`calls from text` rises, parse-disagreement halts fall).
- At `full` the model never sees parameter schemas (the tool list is name: description; the schema
  exists only inside the grammar and a retry prompt), and at `light` it gets two call formats at
  once (the template's XML block and the formatter's JSON instruction).
- A circuit-breaker tier bump works on a copy of the adapter that keeps the old validator and
  retry budget — a hybrid of two tiers. Rows record the tier at start and end, and the tier-sweep
  report leaves bumped runs out of each tier's figures.

## Running it

```bash
uv run python scripts/ladder_run.py --provider llama_cpp --base-url "$LADDER_BASE_URL" \
  --rung r27b --judge-base-url "$LADDER_JUDGE_BASE_URL"
```

- **First run** (`first_run` in `rungs.yaml`, decided 2026-09-25): `r27b` (Qwen3.8-27B
  UD-Q4_K_XL — the 4090's production model, run against the production server untouched and only
  inside 06:30–10:00), `r27b-pq2` (PrismML's ternary Bonsai 2 27B — the same checkpoint, what
  compression costs), `r08b` (Qwen3.5-9B UD-Q4_K_XL — what size costs) and `r09b-ornith`
  (Ornith-1.5-9B Q4_K_M — the same size and base as `r08b`, agentic training). Each pins its file
  by revision and SHA-256. `r04b` (Qwen3.5-4B) comes later.
- **Serving.** Every rung runs on llama.cpp — never Ollama, which turns thinking on, drops the
  grammar-constrained tool decoding, swaps the prompt renderer and sampler, and reports the file
  type as the quant. The mini rungs mirror production's flags (`-c 32768 --parallel 1
  --flash-attn on --jinja --reasoning-budget 2048`), set sampling explicitly to production's
  effective values (`--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.05` — Prometheus sends none, and
  each GGUF carries its own defaults), bind to localhost, and differ only in port, no vision
  projector (text-only suite) and, for Bonsai, `-ub 512` (memory). `r08b` runs on llama.cpp built on
  the mini at production's commit; `r27b-pq2` on PrismML's fork, release `prism-b10735` or later
  (earlier builds crash loading PQ2_0 on the mini's AVX-512 CPU). Record each server's `/props`
  (build, context, sampler) with the run. Thinking stays suppressed on every rung, and every rung
  runs at least 3 runs per task.
- **Tier sweep.** `--force-adapter-tier off|light|full` (with `--rung`) runs the rung's model with
  the daemon's own adapter for that tier — only the tier decision is replaced. The rung's checks
  still run; rows are filed under no rung (`tier_sweep` says what was forced and what the daemon
  picks), `--compare` refuses them, and `--tier-report L1,L2,…` writes the per-tier table: task
  success, tool-call success, repairs, adapter retries and aborts, calls recovered from text, text
  calls tier off missed, XML-markup turns, breaker halts — plus paired per-task differences. Run
  each tier as its own arm, rotate the tier order across repetitions, and use a separate
  `--telemetry-db`.

- Endpoints come from flags or `LADDER_BASE_URL` / `LADDER_JUDGE_BASE_URL`; none are committed.
- `--rung` refuses to run unless the endpoint serves a model matching the rung, the adapter picks
  the rung's tier, and the probed quantization equals the rung's.
- `--smoke` runs the tasks tagged `smoke: true` (three per populated class); `--classes` and
  `--tasks` narrow further (a deferred class is refused by name); `--runs-per-task N` repeats.
- Each run gets a fresh sandbox: a workspace, plus a private stand-in for `~/.prometheus`
  selected with the `PROMETHEUS_*` path variables, so cron jobs, wiki, memory files and LCM
  history are throwaway and nothing reaches the real stores.
- The config is built from the flags, not read from the machine's `prometheus.yaml`, so two boxes
  running the same rung run the same pipeline. The adapter tier is the one the daemon would pick
  (`config/model_registry.yaml`).
- One ladder run per sandbox root and per telemetry DB at a time (file locks): a second run
  is refused — give it its own `--workdir` and `--telemetry-db`. A run label that already has
  rows is refused too, so a report never mixes two runs; unknown class names or task ids are
  refused rather than dropped.
- Every report puts **accuracy** (pass ÷ (pass + fail): format misses, unscored and errors are
  not wrong answers) **next to the answer-line-missing rate**, per class and overall.
  `--report-only --compare L1,L2,…` writes the cross-model ladder table the same way — one row
  per model, accuracy and format misses side by side, overall and per class — so a small model
  that answers right but skips the answer line is never read as failing those tasks.
- After the run, the report is written and the **empty-field check** runs: every required field
  must be populated in at least one row, or the command exits 1. `--report-only --run-label L`
  re-runs both from the database and never overwrites a report when there are no rows.
- A run whose endpoint stops answering, or whose sandbox cannot be cleaned, is **aborted**
  (exit 3): the rows recorded until then stand, and the report covers them.

## What the ladder does not control

- **Sampling.** Prometheus sends no sampler settings, so each server's defaults decide — and
  llama-server takes them from each GGUF's `general.sampling.*`, which differ by file. The pipeline
  has no per-run override (the gym reserves `sampling` for the same reason), so the servers are
  pinned instead: the first run started every mini server with production's effective values
  (`--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.05`) and checked each server's `/props`;
  production itself gets the same values from its GGUF. Every later rung must do the same.
- **Thinking.** `suppress_thinking` stays at the daemon default (on). The effective flag per round
  is recorded.
- **Bash confinement.** Bash's write floor is enforced where bubblewrap works (Linux) and absent on
  macOS. Correct solutions write only inside the workspace, so verdicts do not depend on it; each
  row records which regime it ran under. Bash also runs with the machine's real `HOME`, as it does
  in the daemon, behind the same SecurityGate: the sandbox relocates Prometheus's own stores, not
  the shell's home. So a model *can* read `~/.prometheus` through bash, and on macOS write it. The
  ladder is not a stronger sandbox than Prometheus itself; for the Apple Silicon run, use a
  dedicated macOS user account.
- **The tool surface.** Every task sees the same fourteen tools (`LADDER_TOOLS`), not the daemon's
  full registry, so a class's pass rate reflects the task and not a different menu.
- **Relative write paths.** The SecurityGate treats a relative path on a write tool as unknown and
  asks for approval; a ladder run has no one to approve, so the call is denied. Tasks therefore
  name every file to write as `{workspace}/...` (the gym's prompts use absolute paths for the
  same reason), and denials are counted per run (`tool_calls_denied`).
- **`lcm_grep` semantics** (for memory tasks, when that class is populated). FTS5 with an implicit AND over every query word: a query with one
  word absent from the history returns nothing. Memory tasks plant facts in natural wording, but
  this is the real tool and small models do trip on it.
- **`wiki_query` semantics** (likewise). It scores `index.md` lines by whitespace-split word overlap with
  the page name and summary, so `length,` does not match `length`. Wiki fixtures keep index
  summaries plain; again, this is the real tool.

## First run (2026-09-25 / 26)

WP-2.1's end-to-end proof, on the rungs decided on 2026-09-25 plus Ornith-1.5-9B. Every rung is
pinned by revision and SHA-256 (`rungs.yaml`); every run was thinking-suppressed, `--no-judge`
(the one judged smoke task is `unscored`), 3 runs per task, from one harness host (this Mac).

**The daemon code measured is main at `27399d4` (#572)**, the base this branch had when the runs
were made; the `harness` commit ids in the reports are this branch's commits from before it was
rebased. Of what main has merged since, one change sits on the path these runs took and changes
behaviour: **#582 (WP-X.28 PR 1), which makes tier `full` read Qwen's XML tool calls**. Everything
below about tier `full` describes the code before it. (#570 moved the routing step without
changing it; #580's identity line is written by the model router, which ladder runs do not use;
#577 changed the Ollama provider, which no rung used.) Nothing here was re-measured after the
rebase.

| rung | model (quant) | served on | server | tier |
|---|---|---|---|---|
| `r27b` | Qwen3.8-27B (Unsloth UD-Q4_K_XL, the 08-14 upload) | the 4090, **production, untouched** — run 06:31–06:36, llama-server PID, uptime and VRAM identical before and after | llama.cpp `9d57ce456` (production's own) | light |
| `r27b-pq2` | Ternary Bonsai 2 27B (PrismML PQ2_0), the same Qwen3.8-27B checkpoint | the mini's 3090 Ti | PrismML fork `prism-b10735` (the Zen 4 load fix) | full (daemon's pick) |
| `r08b` | Qwen3.5-9B (Unsloth UD-Q4_K_XL, -MTP build) | the mini's 3090 Ti | llama.cpp `9d57ce456`, built on the mini for sm_86 | light |
| `r09b-ornith` | Ornith-1.5-9B (official Q4_K_M), Qwen3.5-9B base, agentic RL | the mini's 3090 Ti | llama.cpp `9d57ce456`, built on the mini | full (daemon's pick) |

Each mini server ran with production's flags, `--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.05`
(production's effective sampler, confirmed from every server's `/props`), no vision projector,
bound to localhost, started and stopped by its recorded PID; the card was back to its baseline
(12,159 MiB used by its four residents) after each.

### Smokes (18 tasks × 3: the 12 smoke tasks and 6 more answer-line tasks)

Reports: [`ladder-first-run-smokes.md`](../gym/results/ladder/ladder-first-run-smokes.md) and one
per rung in `gym/results/ladder/`.

| rung | pass | fail | format miss | unscored | accuracy (95% CI) | answer line missing | tool-call success |
|---|---:|---:|---:|---:|---|---:|---:|
| `r27b` | 49 | 2 | 0 | 3 | 49/51 (0.87–0.99) | 0/31 | 115/115 |
| `r27b-pq2` (at full) | 40 | 8 | 3 | 3 | 40/48 (0.70–0.91) | 3/32 | 90/92 |
| `r08b` | 49 | 2 | 0 | 3 | 49/51 (0.87–0.99) | 0/32 | 117/120 |
| `r09b-ornith` (at full) | 38 | 10 | 3 | 3 | 38/48 (0.66–0.88) | 3/33 | 82/85 |

Empty-field check: passed for every rung. The 27B, Bonsai and 9B smokes ran at suite sha
`45df225d3b36`, Ornith's at `b71c4bc2c390`; the two differ only in `qa-py-mutable-default` (its
shape and two wrong-answer proofs), and re-reading the other rungs' replies to that task under the
new shape changes none of their verdicts. The smokes are 54 runs each: they show the pipeline and the reader on real output, not
a ranking — the intervals overlap, and Bonsai and Ornith ran at tier full (see below).

**Hand-check.** Two graders, blind to the reader, graded 333 real replies (all 198 answer-line
replies of the smokes, both Bonsai runs and both Ornith runs included, plus 15 per tier from
each of the three sweeps): the graders agreed with each other on every reply, and with the reader on all but two —
both fixed (*The answer reader is frozen*).

### Adapter-tier sweeps: what the adapter layer adds

The same model, server, sampler and 68 tool-using tasks (single_tool, multi_step, file_edit) at
tier off, light and full, 3 repetitions each with the tier order rotated
(`--force-adapter-tier`; reports `tier-sweep-*.md`). *XML-markup turns* are replies with no
structured tool call that carry `<tool_call>` / `<function=` markup — the replies the adapter is
asked to read; a reply carrying markup beside a structured call is not counted, so 0.00 at off and
light means no markup-only reply, not no markup. *Task success* is pass ÷ all runs; non-passes
split into wrong answers, format misses, parse-disagreement halts (the loop stopped on a tool call
the adapter could not read) and other halts (round/tool caps, repeat, breaker, empty replies).
Runs the circuit breaker bumped to another tier are left out (0 to 3 per sweep), as are runs with
no verdict (none in these sweeps).

**Corrected before merge.** The pre-PR review found that a run's end tier was read from the
caller's copy of the loop context, which never sees the breaker's bump, so a bumped run was
caught only if its adapter was asked something after the bump. Two were not: Bonsai
`fe-interval-set-merge` and Ornith `st-read-csv-sku-price`, one run each at tier off, both
round-cap fails. Both came to light when the breaker's own `circuit_breaker_diagnostics` rows
were cross-checked against the run rows (the report now does that), and both are now left out.
Each model's tier off loses one other halt; full − light is unchanged.

| model | tier | task success (95% CI) | wrong | format miss | parse-disagreement halts | other halts | XML-markup turns / run |
|---|---|---|---:|---:|---:|---:|---:|
| Bonsai 2 27B | off | 188/202 (0.89–0.96) | 3 | 0 | 0 | 11 | 0.00 |
| | light | 186/204 (0.86–0.94) | 7 | 0 | 0 | 11 | 0.00 |
| | full | 135/204 (0.59–0.72) | 31 | 5 | 21 | 12 | 1.90 |
| Qwen3.5-9B | off | 189/204 (0.88–0.95) | 9 | 0 | 0 | 6 | 0.00 |
| | light | 187/204 (0.87–0.95) | 11 | 0 | 0 | 6 | 0.00 |
| | full | 155/204 (0.70–0.81) | 31 | 2 | 6 | 10 | 1.34 |
| Ornith-1.5-9B | off | 194/202 (0.92–0.98) | 2 | 0 | 0 | 6 | 0.00 |
| | light | 191/203 (0.90–0.97) | 4 | 0 | 0 | 8 | 0.00 |
| | full | 130/204 (0.57–0.70) | 46 | 15 | 2 | 11 | 2.51 |

Paired by task (mean per-task difference in task success, 95% bootstrap interval over tasks):

| model | light − off | full − light |
|---|---|---|
| Bonsai 2 27B | −0.017 (−0.064 – +0.029) | **−0.250 (−0.338 – −0.172)**, 19 tasks flip |
| Qwen3.5-9B | −0.010 (−0.039 – +0.020) | **−0.157 (−0.221 – −0.098)**, 10 tasks flip |
| Ornith-1.5-9B | −0.025 (−0.064 – +0.015) | **−0.299 (−0.377 – −0.221)**, 23 tasks flip |

- **Off ≈ light for all three.** With native tools in the request, llama-server's own parser
  reads the Qwen3-Coder XML calls; at light the adapter recovered almost nothing from text (at most
  0.04 calls per run) and repaired nothing. **Full cost 16–30 points** for every model, before
  #582: full withholds the native tools, so the model's trained XML calls reach Prometheus as
  text, which the extractor then read as JSON only — a reply that was only XML was stripped and
  retried (parse disagreements), and XML after prose was deleted silently, leaving the prose as the
  answer (the "Let me search the codebase." format misses, and many of the wrong answers). Full
  did not help the weaker model either: the 9B lost less than the others but still 16 points. How
  much of the gap #582 closes is not measured here.
- **Ornith vs Qwen3.5-9B** (same size and base, so mostly the agentic training): at light they are
  level within the intervals (191/203 vs 187/204). At full Ornith loses the most of any model
  (−0.30): it writes the most XML-markup turns (2.51 per run vs 1.34) and the most format misses
  (15 vs 2). Its agentic training leans harder on the trained call format, which tier full broke
  before #582.
  Confounds, labelled: the quantizer (official Q4_K_M vs Unsloth UD-Q4_K_XL), and the tier — the
  daemon gives Ornith `full` by default because its file name matches no registry entry.
- **Evidence for WP-X.28, not changed here.** Qwen3.8 derivatives (Bonsai 2 27B included) and
  Qwen3.5 derivatives (Ornith) behave like tier-light models — native XML tool calls, off ≈ light,
  full much worse — but the daemon puts both at `full` because their file names do not contain
  `qwen3`. `config/model_registry.yaml` and the adapter are untouched in this work package.
  #582 (WP-X.28 PR 1) has since made `full` read the XML; it does not change which tier these
  models get. A re-sweep on main would measure what #582 recovers — not run here.
- **The 4-rung comparison therefore reads at light**: the 27B and the 9B ran their rung smokes at
  light; Bonsai's and Ornith's rung smokes ran at the daemon's pick (full), and their sweeps give
  the light numbers for the three tool-using classes.

### Not controlled, and not measured

- Time: the 27B ran on the 4090 sharing production's one slot with the daemon; the others on the
  3090 Ti beside four resident services. Wall times do not compare across boxes.
- Judge: undecided; the one judged task in the smokes (`qa-explain-sky-blue`) is `unscored` in
  every rung, and the other five judged tasks were not run.
- Thinking-on: every rung ran thinking-suppressed. Ornith's fairness check (a thinking-on smoke if
  it landed more than ~10 points below the 9B at light) did not trigger.
- The first run is a smoke per rung plus the sweeps, not the full 92-task × 3 rung runs (WP-2.2).
