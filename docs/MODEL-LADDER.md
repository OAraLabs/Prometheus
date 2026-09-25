# The model ladder

Which task classes can each local model size handle? The model ladder answers that with
measurements rather than impressions, so Instinct's routing can learn from them. It is a
frozen suite of eight task classes, run through Prometheus's real pipeline against one
served model at a time, with every run ending in a **verdict** and landing in
`telemetry.db`.

This page defines the suite (WP-2.1). The full runs are WP-2.2 (the 4090) and WP-2.3
(Apple Silicon); the **first run targets about 4B, about 14B and the 27B** (`first_run` in
`rungs.yaml`). The first cut populates four of the eight classes; the other four are defined —
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

### Populated (suite v1, sha `45df225d3b36`)

**92 tasks.** Every task names its difficulty within its class; the report breaks pass rates out
by it. Three per class are tagged `smoke: true`.

| class | tasks | easy / medium / hard | decided by | proofs: wrong · format miss · right | success criterion (abridged) |
|---|---:|---|---|---|---|
| `qa` — plain Q&A | 24 | 7 / 9 / 8 | answer line 18, judge 6 | 110 · 38 · 130 | the reply's `ANSWER:` line states the right value; explanations scored ≥ 0.7 by the pinned judge against a rubric and reference |
| `single_tool` — one call is enough | 24 | 8 / 9 / 7 | answer line 21, file predicates 2, acceptance 1 | 140 · 31 · 160 | the `ANSWER:` line states the unguessable planted value, or the named file holds exactly the requested content |
| `multi_step` — dependent calls | 22 | 7 / 8 / 7 | end-state predicates 16, answer line 5, acceptance 1 | 60 · 16 · 41 | the end state (output file, or the `ANSWER:` line) is exactly right; reaching it takes 2–5 dependent calls |
| `file_edit` — edit until a command passes | 22 | 7 / 8 / 7 | acceptance 22 | 55 · – · – | the harness's acceptance tests run and pass after the agent stops, against pristine copies of any visible tests |

The proofs are replies and end states each check is proven against (see *Verdicts*): **wrong**
ones it must fail (wrong answers and wrong files), **format misses** it must neither credit nor
fail, and **right** answers in other spellings it must credit. Every mechanically checked task has
at least one wrong proof and fails untouched; every answer-line task has at least three right
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

Every task ends in one of four verdicts, decided in this order:

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
   `ANSWER:` label (a list, a fenced block, display math — each read whole, so a list of two
   candidates is a format miss), LaTeX (`$…$`, `\boxed{}`, `\text{}`, `\%`, `1{,}081`), the
   prompt's own `<…>` placeholder brackets copied, a value wholly inside one pair of brackets (a
   tuple: `(1, 2, 1, 4)`), and Unicode hyphens are all read as the value they spell. None of those rules can turn a wrong value into a right one. A list is never
   decided by its last item, with a label or without: that would make the verdict depend on the
   order the model listed its candidates in.
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
  The preflight also runs the provider's thinking-suppression probe (as the daemon does at boot),
  records the result, and refuses a rung run whose template ignores the suppression flag.

**Every mechanical verdict is proven to discriminate.** `tests/test_ladder_suite.py` replays each
task through the real predicates and the real acceptance runner, without running a model
(`prometheus.gym.ladder.selfcheck`):

- the untouched setup, with no answer, must **fail**;
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

Real replies have changed it once so far. A dry run of the smoke on `qwen2.5:7b-instruct` gave
five answer-line replies; four were misread, and each is now pinned verbatim in
`tests/test_ladder.py`. `ANSWER: (1, 2, 1, 4)` and two committed "there is none" answers
(`ANSWER: No request ID found for HTTP status 503.`) had been format misses, but all three are
wrong answers and now fail. `…is 48217. ANSWER: 48217` had passed but was recorded as missing
its answer line.

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
- the judge's provenance (`model`, `pinned`, `base_url`) is stored with every judged row.

For the rungs, the pin is in `rungs.yaml`: **`qwen2.5:14b-instruct` on the mini's ollama**, one
judge for every rung so judged pass rates compare across rungs. It is not any rung's model
(tested), it is a different generation from all of them, and it fits on the mini's card during a
ladder window. It is weaker than the 27B rung it grades; that is why judged tasks are a
minority (six of 92, all in `qa`), why each rubric states concrete pass/fail criteria and carries a reference answer, and
why a report shows judged and mechanical verdicts separately (the `decided by` column). A `--rung`
run cannot override the pin.

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
| success | `subsystem_runs.outcome` (`success` / `failed` / `skipped` = undecided) and `summary_json.success`, `verdict` | yes |
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
empty_response / divergence_halt / reasoning_fallback / error), and a bounded trace of the calls.
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

All four are in daemon code (engine or providers), which this work package does not change.

- Early-exit tool calls (`permission_denied`, `validation_failed`, `unknown_tool`,
  `tool_exception`, `hook_blocked`, `tool_timeout`, ...) and `_loop_transition` rows are written
  to `tool_calls` **without a session_id**; only the final execution path sets it. The ladder
  attributes session-less rows inside a run's time window to that run (one ladder writer per DB
  is enforced by a lock) and counts them in `tool_calls_unattributed`.
- `OllamaProvider` never asks for streamed usage (`stream_options.include_usage`), so every
  ollama round records 0 tokens; the llama.cpp and OpenAI-compatible providers do ask. A run
  through ollama records tokens as unreported, and the empty-field check says so.
- The loop's own halt messages carry no marker (recognised by their wording, pinned by a test),
  and its repeat guard writes no `tool_calls` row for a blocked call (counted from the
  transcript).
- The loop writes a repaired call's repair count only when the call executes (or times out); a
  repaired call that then fails is recorded with 0. The ladder counts repairs from the per-call
  observer instead.

## Running it

```bash
uv run python scripts/ladder_run.py --provider llama_cpp --base-url "$LADDER_BASE_URL" \
  --rung r27b --judge-base-url "$LADDER_JUDGE_BASE_URL"
```

- **First run** (`first_run` in `rungs.yaml`): `r04b` (Qwen3.5-4B, Q8_0), `r14b` (Qwen3-14B,
  Q4_K_M) and `r27b` (Qwen3.8-27B, UD-Q4_K_XL — the 4090's production model). The 4B and 14B
  files were not on either box on 2026-09-24 and must be downloaded first. `r02b`, `r08b` and the
  26B-A4B MoE stay proposed for a later run.

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

- **Sampling.** Temperature and friends are the serving backend's defaults; the pipeline has no
  per-run override (the gym reserves `sampling` for the same reason). WP-2.2 should launch every
  rung's server with the same sampling flags.
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

## Smoke run

WP-2.1's end-to-end proof, 2026-09-24: the 24 smoke tasks (three per class, suite sha
`0b18eb23f710`), one run each, on one model. Reports:
[`smoke-qwen2.5-7b-ollama.md`](../gym/results/ladder/smoke-qwen2.5-7b-ollama.md) and
[`smoke-qwen2.5-7b-openai-compat.md`](../gym/results/ladder/smoke-qwen2.5-7b-openai-compat.md).

- **Model:** `qwen2.5:7b-instruct`, Q4_K_M (8B class), adapter tier `light` / strictness `NONE` —
  the daemon's pick. It is not a proposed rung: it was the one model already resident on a GPU
  that could be used without loading or unloading anything on a shared box.
- **Judge:** `Qwen3.8-27B-UD-Q4_K_XL`, pinned by flag, on a different machine from the contestant.
  The suite's rung pin (`qwen2.5:14b-instruct`) would have meant loading a second model beside the
  resident one on the same shared card. One smoke task is judged; it was graded 1.0.
- **Two transports, same tasks:** Prometheus's `OllamaProvider` (the daemon's path to ollama), and
  the local OpenAI-compatible provider against the same endpoint, which asks for streamed usage.

| class | pass (ollama) | pass (OpenAI-compat) | tool calls ok / counted | rounds (mean) | tokens in / out (mean, compat) | time s (mean) |
|---|---:|---:|---:|---:|---|---:|
| qa | 2/3 | 2/3 | 0/2 | 1.5 | 2646 / 45 | 1.2 |
| single_tool | 1/3 | 1/3 | 10/10 | 2.7 | 9143 / 236 | 1.5 |
| multi_step | 0/3 | 0/3 | 20/29 | 4.5 | 18022 / 593 | 3.8 |
| file_edit | 0/3 | 0/3 | 14/16 | 3.3 | 10012 / 380 | 9.4 |
| web_research | 2/3 | 3/3 | 14/14 | 3.3 | 11471 / 184 | 1.9 |
| memory_recall | 2/3 | 2/3 | 9/12 | 3.0 | 6446 / 92 | 1.5 |
| scheduling | 1/3 | 1/3 | 4/7 | 2.3 | 6346 / 97 | 1.1 |
| long_haul | 0/3 | 0/3 | 5/5 | 2.3 | 7733 / 874 | 7.3 |
| **all** | **8/24** | **9/24** | | | | |

Three runs per class says nothing about what a 7B model can do; the Wilson intervals in the reports
span most of [0, 1]. What the smoke shows is the pipeline: 48 runs, every one decided (no `error`,
no `unscored`), zero permission denials, the judged run graded with provenance, 12
session-less `tool_calls` rows attributed by time window.

**Empty-field check:** the OpenAI-compatible run **passes** — all twelve required fields are
populated. The ollama run **fails on `input_tokens` and `output_tokens`**, empty in all 24 rows:
66 model rounds, every one recording 0 input tokens (the OpenAI-compatible run: 72 rounds, none zero), because `OllamaProvider` does not request
streamed usage (see *Known gaps*). Everything else is populated in both runs;
`tool_call_success` is empty only where a run made no tool calls.
