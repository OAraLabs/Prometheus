# Phase 1: flywheel status + trace audit (2026-10-30, read-only)

**Gate: automatic (continue).** No trace content left the 3090 Ti host; everything below is aggregates.

## Summary

- **Corpus:** 11,152 harvested teacher traces, 10,628 (95%) from Qwen3.8 Max; 524 from other cloud models
  (excluded). After dedupe: **10,195 Max traces**.
- **They come from only 299 distinct Max user requests in 22 conversations** (16 calls per request at p50).
  The effective sample for a task-level split is ~300 tasks, not ~10K.
- **Labels are weak.** No model has judged anything. "Pass" means the tool call ran without error. Of the
  Max traces that link to a turn outcome, at least 41% sit in turns that were abandoned or force-stopped.
- **Mix is skewed:** coding 86%, SENTINEL/daemon 8%, other 5%, memory/wiki 1%, **chat-mode 0**,
  **compacted-history 0**, error/recovery 254. `bash` is 79% of all calls.
- **Traces store raw, lossy messages,** not what `LlamaCppProvider` sends: no system prompt, no tool results
  or earlier tool calls, one tool schema instead of the full list, context capped at 12 messages (79% hit
  the cap). Phase 3 must rebuild the full request (system prompt, full tool list, structured context) and
  render it with the student's own template.
- **Lengths as stored** (exact Qwen3.8 tokenizer): p50 2,738 / p90 6,152 / p99 10,962 / max 18,376 tokens;
  26 over 16k. These are floors; rebuilt requests will be much longer.
- **Flywheel code:** gym, golden-trace capture and pair mining are built; harvest, clean/verify, LoRA train,
  benchmarks and tests are partial; quantize, the Saturday cadence and the Monday report are missing.

## What this means for the next phases

1. **Phase 3 can't reach its size gate from this corpus alone.** Capping coding at 35% leaves about
   1,240 non-coding Max traces plus ~670 coding ones: **~1.9K train examples**, against a 5–10K target.
   Chat-mode and compacted-history examples don't exist at all.
2. **So Phase 2b (teacher expansion in the gym) is required, not optional,** and Phase 2 re-judging matters
   more than usual. Both use the Max API and are held on OPEN-QUESTIONS #6 (plan terms) until resolved.
3. **The fail side for Phase 2's 100-fail sample** exists only as 422 failed Max calls in telemetry, with no
   exported context. Building it means rebuilding context from the conversation store with the exporter's
   own resolver, read-only, on the host.

## Part A: flywheel spec vs code

Read-only audit of `docs/sprints/SPRINT-FLYWHEEL-CADENCE.md` (plus the stages added by
`DISTILL-FLYWHEEL-RUNBOOK.md` Phase 8) against `feat/distill-flywheel` @ 2d33659. Paths are repo-relative
unless they start with `~`.

**Short version:** every *science* stage the spec wraps exists as a manual script. None of the *cadence*
the spec adds exists: no `flywheel_*` jobs, no timers, no `flywheel:` config, no report, no
`tests/test_flywheel_jobs.py`. Quantize and the distillation-style SFT train were never in the code.

### Stage table

| Stage | Status | Paths (key entry points) |
|---|---|---|
| Gym (runner, frozen tasksets, dual score) | **built** (index pieces missing) | `scripts/gym_run.py:27` `main`; `src/prometheus/gym/runner.py:480` `run_experiment`, `:82` `preflight_endpoint`; `gym/scoring.py:164` `score`; `gym/report.py:117` `render_report` (`MIN_RUNS_PER_ARM=30`); `gym/store.py` (`gym_runs` in `~/.prometheus/data/gym.db`); `gym/tasksets/{v1,v2,harvest-v1,dictwrap-train,dictwrap-heldout}.yaml` |
| Harvest (gym `--harvest`, corpus gen) | **partial** | `scripts/gym_run.py:41` `--harvest`; `src/prometheus/gym/harvest.py:30` `configure_harvest`; `gym/corpus.py:189` `generate_harvest_corpus`; `scripts/gen_harvest_corpus.py`; `HARVEST-OVERNIGHT-RUNBOOK.md` ("STAGED, not executed") |
| Teacher-trace capture (cloud "golden" traces) | **built, off by default** | `src/prometheus/telemetry/tracker.py:1195` (`is_golden`), `:2252` `export_golden_traces`, `:2308` `_build_example`, `:2446` `export_new_golden_traces`; `src/prometheus/sentinel/golden_trace_exporter.py:76` `lcm_context_resolver`, `:151` `GoldenTraceExporter`; wired at `src/prometheus/daemon.py:2483`; CLI `export-traces` (`src/prometheus/__main__.py:1766`) |
| Mine + export pairs | **built** | `scripts/mine_training_pairs.py:186`; `scripts/export_training_pairs.py:62`; `scripts/transition_histogram.py:21`; `src/prometheus/learning/transition_taxonomy.py:34` |
| Clean / verify | **partial** | `src/prometheus/learning/pair_capture.py:182` (markup guard → `CorruptPairRejected`), `:192` capture-time redaction, `context_hash` UNIQUE dedupe; `scripts/flag_corrupt_pairs.py:88`; `src/prometheus/security/scrub_capture_stores.py` (`oara scrub`); exporter skip rules in `golden_trace_exporter.py:91-145` |
| LoRA train | **partial** | `scripts/lora/train_dictwrap_dpo.py:129` (TRL `DPOTrainer` + PEFT + bnb 4-bit, `--dry-run` = 1 step); `src/prometheus/gym/dictwrap_specialist.py`; `docs/sprints/SPRINT-DICTWRAP-LORA.md` |
| Benchmarks / scoring | **partial** | `scripts/lora/eval_dictwrap.py:58` (pre-registered bars at `:28-30`); `scripts/run_nightly_evals.py:25`; `src/prometheus/evals/runner.py:65` `EvalRunner`; `src/prometheus/evals/judge.py:207` `PrometheusJudge`; `src/prometheus/benchmarks/suite.py`; `src/prometheus/gym/ladder/` (`verdict.py`, `runner.py`); `scripts/run_model_ab_eval.py:621` |
| Quantize (merge, GGUF, imatrix) | **missing** | none: no `convert_hf_to_gguf`, `convert_lora_to_gguf`, `llama-quantize`, `llama-imatrix` or merge step anywhere in `src/`, `scripts/`, `tests/` |
| Tests | **partial** | existing: `tests/test_gym.py`, `test_gym_harvest.py`, `test_gym_corpus.py`, `test_training_pairs.py`, `test_pair_quality_gate.py`, `test_repair_capture_t2.py`, `test_transition_taxonomy.py`, `test_dictwrap_specialist.py`, `test_golden_trace_*.py` (3), `test_evals.py`, `test_eval_judge_*.py`, `test_ladder*.py`, `test_run_model_ab_eval.py`; spec's `tests/test_flywheel_jobs.py` **missing** |
| Sat 02:00–06:00 ET window / scheduling | **missing** | no `src/prometheus/jobs/flywheel_{nightly,weekly,monthly}.py` (only `daily_briefing.py`, `db_snapshot.py`); no `deploy/systemd/` dir, no `.timer` or `prometheus-flywheel-recover.service`; no `flywheel:` block in `config/prometheus.yaml.default`; no setups/`setup apply` mechanism; `run_nightly_evals.py` has only a cron *example* in its docstring |
| Weekly report + Monday reviewer handoff | **missing** | no `flywheel/reports/`, no `LATEST.md` writer, no `GET /api/flywheel/latest`, no `prometheus flywheel` CLI, no `/flywheel` command. The reviewer is an external scheduled task, not in the repo; its input file does not exist |

#### Per-stage notes

- **Gym.** Real `run_loop` turns through the real adapter, telemetry into `gym.db`, raw-emission vs
  post-repair dual score, thin-sample refusal, KV-cache type probed and stamped per row
  (`runner.py:495`). Missing for the spec: the frozen index taskset/manifest it names
  (`gym/tasksets/index-v1.yaml`, `gym/experiments/index-weekly.yaml`, `heldout-v1.yaml`,
  `harvest-overnight.yaml` do not exist). `gym_runs` records model and KV types but not quant, GPU or
  llama.cpp build.
- **Harvest.** Wires pair capture into the gym process against `gym-training.db`, stamped
  `pair_source=gym_harvest`. Missing: the time-budget loop (no deadline check per task in `runner.py`),
  the automated 20-task pilot gate, and partial-result flush on deadline. Note
  `gen_harvest_corpus.py`'s docstring says `--per-shape`; the flag is `--per-transition`.
- **Clean/verify.** Everything is deterministic: template-markup guard on the `chosen` side, secret
  redaction at capture and again at export, exact-hash dedupe, flag-not-delete for corrupt rows, and
  for teacher traces: skip shared-surface session ids, skip group-member conversations, keep only
  messages strictly before the call. Missing for the runbook: LLM re-judge of pass labels,
  near-duplicate detection, task-level train/eval split, leakage hash check.
- **LoRA train.** One script, DPO only, written for a 26B-A4B MoE; records it does not fit one 24 GB
  card (the train phase of the dict-wrap sprint is marked BLOCKED). Saves a PEFT adapter only
  (`save_strategy="no"`, no checkpoints, no resume). No SFT/QLoRA distillation trainer, no Unsloth,
  no remote dispatch, no adapter manifest.
- **Benchmarks/scoring.** Deterministic gym scoring, an LLM-judged eval suite, a tool-call benchmark
  suite (tier 1/2), the model ladder, and an A/B orchestrator exist. Missing for the runbook's gate:
  long-context recall by depth, injection resistance, general-sanity scoring.

### (a) Where traces are written, and what they contain

There are four stores. None is a JSONL "harvest" of full conversations from the local model.

| Store | Writer | Contents |
|---|---|---|
| `~/.prometheus/telemetry.db` → `tool_calls` (rows with `is_golden=1`) | `ToolCallTelemetry` record path, `tracker.py:1195` | one row per tool call: `model`, `served_model`, `raw_model_output` (prose before the call), `parsed_tool_call` (`{"name","input"}`), `session_id`, `tool_schema` (that one tool), `turn_id`, `round_index`, `result_summary`, `node_id`. Golden = cloud provider AND success AND `retries==0` AND raw output present |
| `~/.prometheus/trajectories/golden_traces_<unixts>_<lastrowid>.jsonl` (+ `.export_state.json` watermark); CLI writes `~/.prometheus/golden_traces_<unixts>.jsonl` | `GoldenTraceExporter` (`trajectory_export.enabled: false` by default, 24 h interval) and `export-traces` CLI | OpenAI fine-tune shape: `messages` (up to 12 prior messages rebuilt from LCM) + assistant turn with `tool_calls`, `tools` (only the called tool's schema), `_meta` (model, tool, ts, session) |
| `~/.prometheus/data/lcm.db` → `lcm_messages` | LCM engine (`memory/lcm_engine.py:273,319`) | the conversation text (`content`) plus structured blocks (`content_json`); joined to golden rows by `session_id` |
| `~/.prometheus/data/training.db` → `training_pairs` (live); `~/.prometheus/data/gym-training.db` (gym harvest; also `gym-training-verify.db`, `dictwrap-train.db` per `flag_corrupt_pairs.py`) | `pair_capture.capture_pair` from `engine/agent_loop.py:4339,4354,5454,5706`; `mine_training_pairs.py --commit` | DPO pairs: `rejected`/`chosen` calls as `{"name","input"}` JSON; `context` = `{"kind":"lcm_ref", session_id, ts, model, tool_schema:{name, property names, required}}` |

Also: teacher escalations write `telemetry.db` → `signal_events` (`signal_type="teacher_escalation"`,
`escalation/teacher.py`), off unless `escalation.teacher_model` is set. Exports from
`export_training_pairs.py` go wherever `--out` points (the spec's `flywheel/pairs/` does not exist yet).

**Rendered template vs raw messages: raw, and lossy.** `LlamaCppProvider._build_request_payload`
(`src/prometheus/providers/llama_cpp.py:623`) sends OpenAI-shape `messages` + the full `tools` list
(+ `tool_choice`, `chat_template_kwargs` for thinking suppression, or a GBNF `grammar` when no tools) to
`/v1/chat/completions`; llama-server renders the chat template server-side, so Prometheus never holds the
rendered prompt string (it can fetch the template itself via `/props`, `detect_tool_template` at `:458`).
No store keeps that payload. Compared with what was sent, a golden JSONL example:
- has no system prompt (LCM does not persist it) and only flat `content` text, so earlier tool calls
  and tool results in the window are dropped (the `content_json` blocks are not read);
- carries one tool's schema, not the full advertised `tools` list, with the description cut to 1,000 chars;
- caps context at 12 messages, so long-context examples cannot be rebuilt from it.

Pair rows are worse: `context` holds only an LCM reference and schema property names. For gym pairs
the LCM is a temp DB (`gym/runner.py:226`) that is discarded, so the request text is gone;
`train_dictwrap_dpo.py` rebuilds a synthetic request from the chosen args for that reason. Phase 3's
"render with the exact template" must therefore re-render from `lcm.db` + the tool registry + the
server's template, not read it from a store.

### (b) Judges, verifiers, and who labels pass/fail

- **Teacher traces:** "pass" = the `is_golden` rule (cloud provider, tool succeeded, zero retries). No
  model judges them; success is the tool's own result.
- **Pairs:** no judge. `chosen` = the repaired or retried call that then validated/succeeded;
  `outcome` (turn outcome) is copied at mine time. Only the markup guard filters content.
- **Gym:** deterministic predicates, no LLM (`gym/scoring.py` header says so).
- **Model ladder:** predicates first, then harness-run acceptance tests, then a local judge pinned to
  a different model from the contestant (`gym/ladder/verdict.py`, enforced in `ladder/runner.py`).
- **Evals:** `PrometheusJudge` (`evals/judge.py:207`), local OpenAI-compatible endpoint with JSON-schema
  constrained decoding, one strict parser (`parse_judge_reply`, `:132`), provenance recorded (`:227`).
  Default config pins it to a 7B instruct model on a local Ollama endpoint
  (`config/prometheus.yaml.default:1603-1604`); without a pin it grades with whatever the endpoint loaded.
- **Teacher escalation:** `escalation/detector.py`, regex-only Tier-1 failure detector.
- **Skill recordings:** `learning/live_recorder/step_verifier.py`, advisory model review; the
  deterministic quality gate decides.
- No code calls the cloud teacher to re-judge traces (the runbook's Phase 2 judge is not built).

### (c) Reusable training and eval code

- `scripts/lora/train_dictwrap_dpo.py`: TRL/PEFT/bnb QLoRA skeleton, attention-only LoRA targets,
  `--dry-run` one-step validator, chat-template-first rendering with a raw fallback. Reusable as a
  pattern; DPO-only and model-specific. A prior venv for it exists on the GPU host (`~/lora-train`).
- `scripts/lora/eval_dictwrap.py`: pre-registered-bar PASS/FAIL/INCONCLUSIVE judge; the pattern to copy
  for the distillation gate.
- `scripts/export_training_pairs.py`, `mine_training_pairs.py`, `transition_histogram.py`,
  `flag_corrupt_pairs.py`; `tracker.export_golden_traces` / `_build_example` for OpenAI-shape SFT rows.
- `gym/` + `gym_run.py`, `gym/ladder/`, `benchmarks/`, `evals/`, `run_model_ab_eval.py` for scoring.
- `jobs/daily_briefing.py`: the fail-loud job pattern and the Telegram token bridge the spec reuses.
- **Not present:** Unsloth (only as a GGUF repo name in `gym/ladder/rungs.yaml`), SFT trainer, LoRA
  merge, GGUF export, imatrix, quantization, checkpoint/resume, rental/remote dispatch.

## Part B: trace corpus audit

Audited 2026-10-30, read-only. All numbers are aggregates. No trace content left the 3090 Ti host.

### Where the traces live

- **Corpus:** `~/.prometheus/trajectories/golden_traces_<unix>_<last_rowid>.jsonl` on the 3090 Ti host. There are 21 files (≈140 MB, 11,152 lines), exported 2026-09-28 to 2026-10-15. The watermark is in `.export_state.json` (`last_rowid` 44929).
- **Writer:** `ToolCallTelemetry.export_new_golden_traces()` (`src/prometheus/telemetry/tracker.py`), driven by the SENTINEL `golden_trace_exporter` (`src/prometheus/sentinel/golden_trace_exporter.py`).
  - The target call comes from `telemetry.db` → `tool_calls WHERE is_golden = 1`.
  - The input half is rebuilt from the LCM conversation store at export time (`lcm_context_resolver`: the last ≤12 user/assistant messages strictly before the call).
  - The exported files hold everything the audit needs, so lcm.db was not opened.
- **Not in the corpus:**
  - `~/.prometheus/trajectories-quarantine/` holds 5,519 lines. These are older exports of the same rowid range from before the context-resolver fixes. The 2026-09-28 full re-export replaced them, so they were excluded.
  - `training.db` has `pair_source = cloud_golden` rows: 10,682 Max, 500 Flash, 24 Grok and 1 Sonnet. These are a chosen-only copy of the same cloud calls with no context. They are not a separate corpus.
  - No teacher-escalation traces exist (0 events).
- **Freshness:** the newest golden row is rowid 44929 (2026-10-15), which equals the watermark, so there is no export backlog. Production traffic moved to local models after that date, and local calls never become golden.

### Total and generating model

| Source model | Traces | Sessions | Share | Flag |
|---|---:|---:|---:|---|
| qwen3.8-max | 10,628 | 22 | 95.3% | teacher |
| qwen3.8-flash | 500 | 3 | 4.5% | **non-Max: exclude** |
| grok-4.5 | 24 | 1 | 0.2% | **non-Max: exclude** |
| **Total** | **11,152** | **25** | | |

- Max traces span 2026-08-15 to 2026-10-15. All the material comes from 25 conversations: 23 web-app sessions, 1 Telegram chat and 1 probe session (2 traces).
- **Concentration:** the largest session holds 18.9% of all traces, and the top 5 hold 54%.
- **Coverage:** the telemetry store holds 13,599 golden rows. 11,859 of them are Max, and 10,628 of those were exported. The other 1,231 Max rows had no session id (1,586 golden rows overall) or no context that could be recovered, so they were skipped as untrainable.

### Pass/fail labels and the judge

- **No LLM judge has labelled this corpus.** The only label is the rule-based `is_golden` flag, set at write time in `tracker.py` `_record()`. It requires all four of these:
  - cloud provider
  - tool execution reported success
  - `retries == 0`
  - raw model output captured
- `retries` is 0 on every Max row, so in practice the label means "the tool call executed without error". It says nothing about whether the call was the right one or whether the task got done.
- **Every exported trace is a "pass" by construction.** The fail side exists only in `telemetry.db` (failed tool calls, with no exported context):

| Model | Cloud tool calls | Success (= golden) | Failed |
|---|---:|---:|---:|
| qwen3.8-max | 12,281 | 11,859 | 422 (3.4%) |
| qwen3.8-flash | 1,415 | 1,358 | 57 |
| grok-4.5 | 322 | 292 | 30 |

- **Secondary turn-level signal:** 4,561 Max traces could be joined to a telemetry v2 `turns` row (2026-10-03 onward). They break down as follows:
  - 1,069 accepted by the user
  - 1,425 abandoned
  - 465 forced stop
  - 8 user-corrected
  - 1,594 with no outcome

  So at least 41% of the turn-labelled Max traces are "pass" calls inside turns that ended badly. Phase 2 re-judging is needed. The tool-success label alone is too weak.

### Category mix

**Classifier** (the first rule that matches wins; the code is in `classify()` in `audit_traces.py`):

1. **chat-mode:** the trace offers no tools or has no tool call.
2. **SENTINEL/daemon:** any one of these:
   - the session is not a human surface (`beacon:`/`telegram:`/`slack:`/`discord:`)
   - the last user message is a system injection: worker re-engagement ("A worker you started…"), restart notice, or SENTINEL/cron/scheduled/heartbeat marker
   - the tool is `cron_*`/`sentinel*`/`symbiote_*`
   - a `bash` command touches the daemon or host services (`systemctl`, `journalctl`, `crontab`, `~/.prometheus/` paths, `prometheus.yaml`, `oara|prometheus <daemon|status|doctor|setup|pair|logs>`, `nvidia-smi`, `llama-server`, `ollama`, `docker`)
3. **memory/wiki:** a `memory`, `wiki_*`, `lcm_*` or `vault_search` tool, or a file or bash operation on `wiki/`, `MEMORY.md`, `USER.md`, `memory.db` or `lcm.db`.
4. **coding:** a file or code tool (`edit_file`, `write_file`, `read_file`, `grep`, `glob`, `lsp`, `todo_write`), a `coding:` session, or a `bash` command that matches a dev tool, a source-file extension or a common shell utility.
5. **other:** everything else: worker/task orchestration, web, `ask_user` and unmatched bash.

| Category | All | Max only | Max after dedupe | Distinct Max prompts |
|---|---:|---:|---:|---:|
| coding | 9,550 (85.6%) | 9,239 | 8,958 | 275 |
| SENTINEL/daemon | 894 (8.0%) | 700 | 695 | 164 |
| other | 585 (5.2%) | 576 | 429 | 134 |
| memory/wiki | 123 (1.1%) | 113 | 113 | 60 |
| chat-mode | **0** | 0 | 0 | 0 |

(Distinct prompts overlap across categories, because one task's call chain can span several of them.)

- **Tools:** the corpus is dominated by one tool. `bash` is 8,802 of 11,152 (79%), then `edit_file` 1,185, `read_file` 327, `write_file` 237, `task_create` 199 and `task_output` 160, with 24 more tools making up the rest.
- **Bash breakdown:**

  | Kind of bash command | Traces |
  |---|---:|
  | dev tool (git/python/test/build) | 4,044 |
  | shell utility on a source file | 2,724 |
  | generic shell utility only | 1,101 |
  | daemon/services pattern | 773 |
  | unmatched | 160 |

  The 1,101 generic-utility calls are the softest part of "coding". Even without them, coding is well above the 35% cap.
- **SENTINEL/daemon is mostly operator shell work.** 763 traces are `bash` on the daemon or services, and 126 are worker re-engagement prompts. **There are zero cron, heartbeat or SENTINEL-alert traces.** Those runs never produced golden rows.
- **Chat-mode (`suppress_tools`) has 0 traces by construction.** A golden trace is always a tool call. Chat-mode examples must come from Phase 2b.
- **Gaps for Phase 2b:**
  - compacted-history context: 0 traces
  - error/recovery (the previous call in the same session failed within 10 minutes): 274 traces, 254 of them Max
  - memory/wiki: 1.1%

### Near-duplicates

| Measure | Distinct | Redundant rows |
|---|---:|---:|
| Exact duplicate (context + call + prose) | 11,141 | 11 |
| Normalized prompt + normalized call | 10,703 | 449 (4.0%) |
| Normalized call only | 10,556 | 596 |
| **Normalized prompt only (task proxy)** | **324** | — |

- **Normalization:** the prompt is the last user message in the context, lowercased, with hex runs of 6 or more characters and all digit runs masked and whitespace collapsed. The call is the tool name plus its sorted JSON arguments, normalized the same way.
- **The real issue is correlation, not duplication.** 11,152 traces come from only **324 distinct user requests** (299 for Max), at 16 calls per request at p50, 86 at p90 and 355 at most. Each trace is one step of a long agentic chain. Consecutive steps share up to 11 of their 12 context messages. The effective sample size for a task-level split is about 300 tasks from 22 conversations, not about 10K.
- **Daemon and heartbeat repeats:** there are no heartbeat or cron traces to repeat. Within SENTINEL/daemon, prompt+call dedupe removes 16 rows (largest group 5). 6 normalized prompts recur across sessions, covering 1,191 rows. These are short generic follow-ups, not daemon prompts.
- **Max after prompt+call dedupe: 10,195 traces.**

### Token lengths

- **Tokenizer:** the exact Qwen3.8 BPE (248,320 tokens, `qwen35` pre-tokenizer). It was rebuilt from the vocab, merges and chat template embedded in the local Qwen3.8-27B-based GGUF on the 3090 Ti host. No `tokenizer.json` or `transformers` for this family is installed there.
  - It was checked by round-tripping synthetic strings only. There was no llama.cpp tokenizer binary to compare against byte for byte.
  - Phase 3 must confirm that the Qwen3.5-9B student uses the same vocabulary.
- **What was measured:** each trace was rendered with that GGUF's own chat template (0 render errors) and counted as stored: its context, its one tool schema and the assistant call in the template's XML `<tool_call>` format.

| Set | p50 | p90 | p99 | max | > 16,384 |
|---|---:|---:|---:|---:|---:|
| All 11,152 | 2,738 | 6,152 | 10,962 | 18,376 | 26 |
| Max only | 2,715 | 5,531 | 10,336 | 18,376 | 26 |
| Max, deduped | 2,729 | 5,522 | 10,620 | 18,376 | 26 |
| Target turn only (prose + call) | 150 | 632 | 2,358 | 4,215 | 0 |
| Stored tool schema | 242 | 242 | 687 | 809 | 0 |

- **Measured density:** 3.93 chars per token. The chars/3.5 estimate would overstate lengths: it gives p50 3,029 and 81 traces over 16k.
- By category, the p50/p90 is 2,711/5,571 for coding, 3,736/9,796 for SENTINEL/daemon, 2,787/6,486 for other and 2,924/6,882 for memory/wiki.
- **These lengths are a floor, not the runtime length.** Each stored trace lacks the system prompt and the full tool list, and its context is capped at 12 messages: 8,845 traces (79%) hit that cap. Once Phase 3 adds the system prompt and full tool set back, rendered lengths will be far higher, and the over-16k count will grow.

### Stored format vs what `LlamaCppProvider` sends

**The traces store raw messages, not the rendered template.** Each line has these keys:

- `messages`: a list of `{role, content}`. Context roles are user and assistant only. There are 0 system messages and 0 tool-result messages, and earlier assistant turns are prose only, with no record of their own tool calls.
- The final assistant turn: an OpenAI-shape `tool_calls` entry with arguments as a JSON string.
- `tools`: exactly **one** schema, the called tool's, in OpenAI function format.
- `_meta`: `{model, tool_name, timestamp, session_id}`.

`LlamaCppProvider` (`src/prometheus/providers/llama_cpp.py`) sends `/v1/chat/completions` with `messages`, the **full** `tools` list, `tool_choice` and `chat_template_kwargs`. llama-server then renders all of that with the GGUF's Jinja template (`--jinja`), so the rendered prompt is never stored on our side.

The Max traces were also produced by a cloud API through the Qwen provider, so no llama-rendered form of them ever existed. For a byte-exact Phase 3, the following must be rebuilt:

1. Render with the student GGUF's own template, which uses the Qwen3.5/3.8 XML `<tool_call><function=…><parameter=…>` format.
2. Re-supply the system prompt and the full tool list. `telemetry.db` `tool_sets` stores tool *names* only (4 sets, from 2026-10-03 onward), not schemas, so the schemas must come from the current registry and may have drifted since.
3. Decide whether to restore tool results and earlier tool calls from LCM, because the exported context drops them.

### Re-running

The scripts are on the 3090 Ti host in `~/distill/phase1/`. Run these with the system `python3`:

1. `snapshot.py`: backup-API copies of telemetry.db and training.db, taken over read-only connections, into `~/distill/snapshot/`.
2. `build_tokenizer.py`: needs `PYTHONPATH=<llama.cpp>/gguf-py`. Writes `qwen38_tokenizer.json` and `qwen38_chat_template.jinja`.
3. `audit_traces.py`: writes a per-trace index (hashes, labels and token counts, no text) to `trace_index.jsonl`, and the aggregates to `audit_summary.json`.
4. `audit_supplement.py`: classifier sensitivity, task counts and the telemetry pass/fail table, written to `audit_supplement.json`.
