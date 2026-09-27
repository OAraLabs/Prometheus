# Telemetry gaps on the agent loop's turn path (WP-X.21)

**Date:** 2026-09-26 · **Code audited:** origin/main `5fb0000` (the mini's daemon runs v0.9.3 `b2603ad`;
every writer below is the same in both unless a line says otherwise) · **Data:** SQLite backup-API copies of
the mini's live `telemetry.db` (and `data/lcm.db`, for two counts), taken 2026-09-26 23:59 UTC ·
**Status:** audit only. Nothing in `src/` changes; `engine/agent_loop.py` is untouched. The scripts in
`docs/audits/telemetry-gaps/` hold no data.

Main moved to `d481b9e` while this was written:

- #585 gives coding runs the served model's name and re-records `coding_run`.
- #601 changes the memory store's snapshot.

Neither touches a telemetry writer, so every line reference and row count stands. The golden sets
below are stated at `d481b9e`, the base the fixes will start from. The two that #585 changed were
replayed there again (§3).

## Answer

An earlier count of nine gaps was not saved, so the list below is re-derived from the code. It has
**15 gaps**, and the fix plan (§4) is 12 PRs:

- **6 change no parity golden,** so they may go now. Five were prototyped and replay 11/11 on the
  mini. The sixth changes only an HTTP reader that no scenario calls.
- **6 change goldens** and queue for the trace-changing slot after WP-4.2. They change what is
  recorded, never a model request, so they can share one re-record.

The headline numbers, in the 14 and 30 days before the copy:

- **No row has ever recorded a prompt-cache count.** `LLMCallEnvelope.stream()` passes the cache
  counts on its failure path only. Its success path (`learning/llm_envelope.py:409`) has dropped
  them since PR #119 added them (2026-07-31). The llama.cpp provider never parses them either,
  although the 4090 sends them: 81% of the prompt tokens in the recorded parity exchanges were
  cached. **3,816 / 8,262 successful rounds** have no cache count. Only 2 of them, both Ollama,
  are a correct NULL.
- **Session ids.**
  - 637 of the 10,814 tool calls since the column appeared (2026-08-15) have no session id. That
    splits exactly into 296 failure-path rows, 28 `lucky_guess` markers and 313 rows from runs with
    no session.
  - **309 of those 313 were written by an older build on 08-16/17.** They have no `tool_schema`,
    which today's main path always fills. So they come from no current code path.
  - Separately, **3,593 / 7,151 `_loop_transition` rows** have no session.
- **The 1,586 golden rows with no session id all predate the column.** Every golden row written
  since has one.
- **The Anthropic parser under-reports the prompt.** In the recorded `hosted_route` exchange,
  `input_tokens` was 348 of a 15,083-token prompt. The rest was a cache write that the row does not
  count. That under-bills metered cost and under-reads the context meter.
- **Microcompact rows are filed under the literal `web`:** at least **214 / 468**. That is the
  routing namespace, not the conversation.
- **`served_model` is never parsed for qwen, xai, Anthropic or Ollama.** That leaves it empty on
  3,100 / 4,967 tool calls. All four send it on the wire.

## The gaps

Rows are "last 14 days / last 30 days" before 2026-09-26 23:59 UTC. The daemon had no conversations
after 2026-09-24 20:02 UTC, so both windows end two days early in practice. **Golden** is the answer
from replaying a prototype of the fix on the mini (§3), not a prediction.

| # | Gap | Rows 14d / 30d | Who reads the field | Fix | Golden-changing |
|---|---|---:|---|---|---|
| T1 | Ten failure-path writers in `engine/agent_loop.py` write `tool_calls` with no session id | 48 / 201 | skill audit; Instinct corpus; model ladder | pass the turn's session like the main path | **yes**: gate_blocked |
| T2 | `lucky_guess` writes a second, "successful" `tool_calls` row per call, with no session id | 2 / 4 | every `tool_calls` reader: `/api/telemetry`, `/health`, Beacon's tool feed, model ladder | move the marker to `subsystem_runs`; readers skip old markers | no |
| T3 | `_loop_transition` rows carry no session id | 3,593 / 7,151 | per-session turn outcomes: skill audit, Instinct corpus | `_log_iteration` records the run's session | **yes**: 7 goldens |
| T4 | Two entry points run with no session at all: POST `/api/chat` and subagents | 19 / 21 loop rows, 4 / 4 calls | golden-trace exporter; `/api/lcm` context meter; every per-session reader | a descriptive-only session id; origin and routing untouched | no |
| T5 | Microcompact rows are filed under the routing namespace `web` | ≥ 214 / ≥ 468 | per-session compaction history | record the turn's session | no |
| T6 | The envelope's success path drops cache counts | 3,816 / 8,262 | `/api/usage` and Beacon's usage view (show 0 cached); cost per round | pass them on the success row | **yes**: hosted_route |
| T7 | llama.cpp never parses the cache count the 4090 sends | 999 / 3,821 | as T6, plus KV-cache reuse on the 4090's one slot | parse `prompt_tokens_details.cached_tokens` | no on its own; **yes** with T6: the 9 llama.cpp goldens |
| T8 | Anthropic `input_tokens` leaves out cache reads and writes | 0 / 2 (38 all time) | `/api/usage` metered cost; `/api/lcm` context meter | `input_tokens` = the whole prompt | **yes**: hosted_route |
| T9 | Local rounds are billed `unknown` (blank model names; Ollama names) | 231 / 256 | `/api/usage` coverage; Beacon's usage view | classify by provider class first | **yes**: model_switch, repaired_tool_call |
| T10 | The context compactor's model call records no tokens and no session column | 123 / 175 | `/api/usage`; 4090 slot accounting | `call()` keeps the usage; takes a session | **yes**: compaction |
| T11 | Turn-path model calls that write no row: LCM summaries, session titles | 832 / 1,503 summaries; ≤ 24 / ≤ 92 titles | `/api/usage`; 4090 slot accounting | route them through the envelope | **yes**: 10 goldens |
| T12 | `served_model` is never parsed for qwen, xai, Anthropic or Ollama | 3,100 / 4,967 | fine-tuning corpus provenance; model ladder | parse the model the stream names | a) qwen/xai/Anthropic: no; b) Ollama: **yes**, repaired_tool_call |
| T13 | `loop_round` rows name only the requested model | 231 / 1,776 rounds with a blank name | per-model rounds and tokens; audits pairing rounds to models | the served model in the round's summary | **yes**: 9 goldens (the llama.cpp ones); all 11 once T12 lands |
| T14 | A round served by the provider fallback is recorded under the model that failed, and can be marked golden | 0 / 0 (latent) | golden-trace corpus; per-model success rates | the round's serving model and provider on its rows | no |
| T15 | Six failure writers drop the call's input (`parsed_tool_call`) | 12 / 54 | Beacon's tool feed (shows no inputs); forensics | record it as the three validation writers do | **yes**: gate_blocked |

`/api/status` reads telemetry only for the skill-load rows (`skills`/`load`), which carry their session
correctly. No gap reaches it.

---

## Data, method, constraints

- **Sources.** Both copies were taken on the mini with `backup_copy.py`: the SQLite backup API from a
  read-only source connection, one step, `integrity_check` ok, 0600 inside a private `mktemp -d`
  directory.
  - `telemetry.db`: 59.6 MB, 31,433 `tool_calls` and 49,208 `subsystem_runs` rows.
  - `lcm.db`: 185.7 MB. It was used only for the T11 counts and the `web` check in T5.
  - The live files were never opened for writing. Both copies were deleted when the work ended.
- **What left the mini.** Counts, shares and sums, plus the parity replay reports, whose content is the
  public fixtures' synthetic data. No session id, input, output or error text from the live stores.
- **Windows.** Anchored at the copy (2026-09-26 23:59 UTC). The figure of 4,626 successful rounds in 14
  days quoted when this audit was commissioned is the same query anchored at the 2026-09-25 nightly
  snapshot. `gaps.py --also-anchor 1790319300` prints it.
- **Surface of a session-less row.** A row with no session takes the class of its nearest neighbour,
  by the skill-usage audit's rule: the same model with a session within 5 minutes, else a `loop_round`
  within 10 minutes.
- **Which NULL is correct.** Decided from what the providers actually send. `wire_shapes.py` reads the
  usage chunk of every recorded completion in the parity fixtures:
  - the 4090's llama.cpp (build `9d57ce456`);
  - the mini's Ollama 0.23.0;
  - the Anthropic API.

  For qwen and xai, which have no recorded exchange, it relies on the providers' documented shape.
  Alibaba documents `usage.prompt_tokens_details.cached_tokens` for its OpenAI-compatible mode
  ([context cache](https://www.alibabacloud.com/help/en/model-studio/context-cache)), and that is the
  shape our parser reads. Whether the flat-plan host fills it for `qwen3.8-max` is unverified: no row
  has ever kept one.
- **Golden impact.** Each fix was prototyped on a scratch tree at `5fb0000`, which was never pushed.
  Each was then replayed with the parity harness on the mini: from a throwaway clone, one run at a
  time, waiting on the shared lock.
- **Rules kept.** The daemon was not restarted or reconfigured, and the 4090 was not used. The only
  processes started were short SSH scans and the replay scripts. Each ran to completion; none had
  to be stopped.

---

## 1. Every writer on the turn path

### `tool_calls` (all in `engine/agent_loop.py` unless named)

Every row also gets `id`, `timestamp` and `node_id` from `ToolCallTelemetry.record`. `is_golden` is
computed there: a cloud provider, success, no retries, and raw output present.

| Writer | Line | Fills | Leaves empty |
|---|---:|---|---|
| main execution path | 4471 | model, tool, success, retries, latency, error_type (`None` / `nonzero_exit` / `tool_error`), error_detail on errors, raw output, parsed call, provider → is_golden, repairs, served_model, **session_id**, tool_schema | nothing. An ephemeral turn nulls the content columns and the session on purpose. |
| `hook_blocked` | 3818 | model, tool, success=0, error_type, error_detail, served_model | **session_id**, **parsed call**, tool_schema, repairs |
| `no_registry` | 3834 | same as `hook_blocked` | same |
| `validation_failed` | 3921 | + retries=1, latency NULL (never ran), parsed call | **session_id**, tool_schema, repairs |
| `unknown_tool` | 3961 | model, tool, error_type, error_detail, served_model | **session_id**, **parsed call** |
| `lucky_guess` | 3991 | success=**1**, error_type, error_detail, served_model: a **second row** for a call that gets its own row later | **session_id** |
| `template_markup` | 4033 | + parsed call | **session_id** |
| `input_validation` | 4104 | + parsed call | **session_id**, repairs from an unwrap attempt |
| `permission_denied` (user declined) | 4319 | model, tool, error_type, error_detail, served_model | **session_id**, **parsed call** |
| `permission_denied` (gate) | 4334 | same | same |
| `tool_timeout` | 4395 | + retries, latency, repairs | **session_id**, **parsed call**, tool_schema |
| `tool_exception` (`_safe_execute`) | 3310 | model, tool, error_type, error_detail (non-ephemeral), served_model | **session_id**, **parsed call**, latency |
| `_loop_transition` (`_log_iteration`) | 2901 | model, success, error_type = the loop's reason, error_detail | **session_id**, served_model. The round index is in hand and not kept. |
| `_malformed` (`providers/stub.py`, the shared parser) | 216 | model, error_type `malformed_empty`, error_detail, raw output | **session_id**: no session is in scope at the provider. 0 rows since 2026-08-15. |

### `subsystem_runs`

| Writer | Line | Fills | Leaves empty |
|---|---:|---|---|
| `agent_loop`/`tool_advertisement` | agent_loop 1198 | the turn's session, model, summary (deferred, source, advertised, registered, profile) | tool names (counts only) |
| `agent_loop`/`context_preflight_refusal` | agent_loop 1651 | the turn's session, model, summary | — |
| `agent_loop`/`loop_round`, success | llm_envelope 409 | duration, summary (stop_reason, dropped_malformed, empty_content), input/output tokens, round, the turn's session, **requested** model, thinking, billing mode + marker | **cached_input_tokens, cache_write_tokens** (dropped), served model |
| `agent_loop`/`loop_round`, failed / empty stream | llm_envelope 366 / 385 | the same, plus the classifier's summary; the failure path does pass cache counts | — |
| `agent_loop`/`microcompact` | agent_loop 3255 | model, round, summary | the session is `context.session_id` (line 3259): **`web` on every web turn** |
| `repeat_detector`/`trip`, `divergence`/`halt` | agent_loop 2389, 2492 | the turn's session, model, summary | — |
| `skills`/`load` | tools/builtin/skill.py 79 | the turn's session (ephemeral-nulled), summary | — |
| `context_compactor`/`summarize_span` (`call()` → `_record_success`) | llm_envelope 619; compactor 721 | model, billing, duration, summary with the session inside `context` | **tokens**, **the session column** |
| `llama_cpp_provider`/`reasoning_budget_exhausted` | llama_cpp 715 | model, summary | session (none at the provider) |

### Other tables

- `silent_failures`: a stream failure writes its row with the session in `context`
  (`llm_envelope.py:353`). The providers' own rows (`llama_cpp.py:799`, `ollama.py:525`) carry none.
- `circuit_breaker_diagnostics` (`agent_loop.py:529`) has no session column at all. It had 3 rows in
  14 days.

### Providers' usage parsing

| Provider | Parses | What the wire also carries (from `wire_shapes.py`) |
|---|---|---|
| llama.cpp (`llama_cpp.py:883`, `:965`) | input, output, served_model (`:881`) | `prompt_tokens_details.cached_tokens` and `timings.cache_n`: **not parsed** |
| OpenAI-compatible: qwen, xai, … (`openai_compat.py:346`, `:392`) | input, output, cached, cache write | `model` on every chunk: **not parsed** |
| Anthropic (`anthropic.py:300`, `:427`) | input (**excluding** cache reads and writes), output, cached, cache write | `message.model` in `message_start`: **not parsed** |
| Ollama (`ollama.py:431`, `:396`) | input, output | `model` on every chunk: **not parsed**. No cache fields at all, so NULL is correct. |
| `StubProvider` (`stub.py:378`, `:425`) | input, output | the same OpenAI shape; not used by the live daemon |

### The adapter, subagents, and calls that write nothing

- **The adapter** (`adapter/`) writes no rows. Its work reaches telemetry only as `repairs`, which
  only the main and timeout rows carry, and `retries`, which is 1 on `validation_failed` and 0 on
  every executed call. The tier and the parse path it chose are recorded nowhere.
- **Subagents** (`coordinator/subagent.py:116-138`) build an `AgentLoop` with the parent's
  telemetry handle, no tool loader and no session. Every row one writes is session-less. The `agent`
  tool and `_try_escalate_tool_call` spawn them. There were 0 subagent runs in the window.
- **POST `/api/chat`** (`web/server.py:4539`) calls `run_async` with no session.
- **Model calls with no row at all:**
  - the LCM summarizer (`memory/lcm_summarize.py:182`, run on every turn from
    `engine/session.py:478`);
  - session titles (`engine/session_titles.py:94`);
  - the `vision` tool (`tools/builtin/vision.py:131`);
  - prompt hooks (`hooks/executor.py:304`).

  All four take the daemon's primary provider, the 4090. None is metered today, but each one takes
  the 4090's single slot.

---

## 2. The gaps in detail

### T1. Failure-path rows carry no session id

- **Where.** The ten writers marked in §1. The main path passes
  `session_id = effective_session_id` (or NULL for an ephemeral turn). None of the others passes
  anything, and every failure row since the column appeared is NULL: 296 of 296.
- **Rows.** 48 / 201.
  - By error type, over 30 days: `validation_failed` 109, `permission_denied` 51,
    `input_validation` 38, `tool_timeout` 3.
  - By surface, over 30 days: evals 125, beacon 37, telegram 26, ios 9, coding 2, unattributed 2.
  - By model: blank-name llama.cpp rounds (evals and coding runs) 119, `qwen3.8-max` 48, the 4090's
    GGUF 22, `qwen3.8-27b` 8, `qwen3.8-flash` 3, `grok-4.5` 1.
- **Who reads it.**
  - The skill-usage audit can't count failures per session.
  - The Instinct corpus pairs each LCM tool call with its telemetry row by session, name and outcome.
    Its query takes only rows with a session, so every failed call falls back to LCM's bare
    `is_error` and loses its `error_type`.
  - The model ladder (`gym/ladder/record.py:73`) already works around this: it joins session-less
    rows to a run by time window. It turns the join off on the live daemon's database, where other
    writers share the window, so there it misses every failure row.
- **Fix.** Every failure writer passes
  `None if ephemeral else (effective_session_id or context.session_id)`, the main path's own rule.
  `_safe_execute` already has `effective_session_id` in scope. The provider's `_malformed` row has
  no session in scope. A run-scoped context variable, like `_RUN_PATHS`, would give it one. It has
  had 0 rows since 2026-08-15, so it can wait.
- **Golden:** yes, `gate_blocked` (two `permission_denied` rows). Same PR as T3 and T15. At `5fb0000`
  it also changed `coding_run`'s `validation_failed` row. #585's re-recorded `coding_run` makes no
  failed call.

### T2. `lucky_guess` is a marker written as a call

- **Where.** `agent_loop.py:3991`. A deferred tool called by name gets a `success=1` row with
  `error_type='lucky_guess'`, and then its real row.
- **Rows.** 2 / 4, and 28 since 2026-08-15. In the windows, all come from the 4090's GGUF.
  Surfaces over 30 days: test harness 2, beacon 1, desktop 1.
- **Who reads it.** Every `tool_calls` reader counts the marker as a successful call:
  - `report()` → `/api/telemetry`;
  - `health_summary()` → `/health`;
  - `recent_tool_calls()` → Beacon's tool feed, where it shows as a phantom success;
  - the model ladder's success rate.

  It is small (4 of 7,445 successful calls in 30 days), but it is a wrong count, not a missing field.
- **A trap in the obvious fix.** Adding a session id to the marker would pull it into the Instinct
  corpus, whose pairing matches calls by name and outcome within a small window. There a marker can
  take the real call's match.
- **Fix.** Write the marker to `subsystem_runs` as `agent_loop`/`lucky_guess`, with the session,
  model and tool name in the summary. That is the pattern the skill-load counter uses. The
  `tool_calls` readers then skip historical `error_type='lucky_guess'` rows through a named constant,
  as they already skip `_loop_transition`.
- **Golden:** no (replayed 11/11). No golden has a lucky guess.

### T3. `_loop_transition` rows carry no session id

- **Where.** `_log_iteration`, `agent_loop.py:2901`, called from 14 sites in `_run_loop`.
- **Rows.** 3,593 / 7,151. There are 9,898 since 2026-08-15, almost as many as the 10,814 real
  calls.
  - By reason, over 30 days: tool success 6,492, tool error retry 456, stripped to empty 175,
    breaker trip 19, iteration cap 3, unproductive repeat 3, divergence halt 2, empty response 1.
  - By surface, over 30 days: beacon 3,777, evals 1,339, telegram 1,242, desktop 496.
- **Who reads it.** No reader can say which session's turn ended on a breaker trip, a parse
  disagreement or the iteration cap. That is the one question these rows exist to answer. Every
  per-call reader, the skill audit and the Instinct corpus among them, excludes them as not calls,
  and none can use them per session.
- **Fix.** `_log_iteration` records the run's session under the ephemeral rule. The prototype sets a
  run-scoped context variable in `run_loop`; threading the id through the 14 calls works too. Moving
  the rows to `subsystem_runs` with their round index is the cleaner end state, but it is a larger
  change for every reader.
- **Golden:** yes. The 7 goldens with transitions: checkpoint_undo, coding_run, gate_blocked,
  linked_workspace, memory_write, repaired_tool_call, tool_calls.

### T4. Runs with no session at all

- **Where.**
  - POST `/api/chat` (`web/server.py:4539`) calls `run_async` with no session, although the
    conversation is `web:<id>` in LCM.
  - `SubagentSpawner.spawn` (`coordinator/subagent.py:135`) likewise.
- **Rows.**
  - In the windows: 19 / 21 agent-loop rows (12 / 13 rounds, 7 / 8 advertisements), and 4 / 4 tool
    calls from current code, all on 2026-09-21.
  - Every advertisement is the daemon's own loop (deferred, 15 of 55 tools), which is POST
    `/api/chat`'s shape. A subagent would advertise "registry direct".
  - The skill-usage audit's 313 session-less calls are almost all something else. 309 were written
    on 2026-08-16/17 with no `tool_schema`, which the main path has filled since #209 (2026-08-15):
    an older build writing to the same file.
- **Who reads it.**
  - The golden-trace exporter can't export a session-less golden row (none since the column
    appeared).
  - `/api/lcm`'s context meter finds no round for the conversation.
  - Every per-session reader loses the run.
- **Fix.** A descriptive-only id: a `run_async`/`run_loop` keyword that reaches the telemetry writers
  and nothing else.
  - POST `/api/chat` records `web:<id>`.
  - A subagent records its parent's session (naming is Will's call).
  - `LoopContext.session_id`, which decides the permission origin, stays as it is, and so does the
    router's override lookup.
  - Whether POST `/api/chat` should also *route and permission* as its session is a separate
    behaviour question. Today it runs with origin `system` and no per-session override, unlike a
    WebSocket turn on the same conversation.
- **Golden:** no (replayed 11/11). No scenario uses either path.

### T5. Microcompact rows are filed under the routing namespace

- **Where.** `agent_loop.py:3259` passes `context.session_id`. On the web path that is the literal
  `web` that `daemon.py` pins on the shared context, not the turn's conversation. #258 and #458 fixed
  the same substitution in the other writers; this one was missed.
- **Rows.** ≥ 214 / ≥ 468. A microcompact row runs at the top of a round, so each one was paired with
  its own round's `loop_round` row. Those counts are the rows under `web` whose round ran under a real
  session: 791 all time.
  - Another 29 / 66 are ambiguous. Their round is under `web` too, which fits either a client's
    conversation really named `web` (LCM holds 369 messages in one) or a round from before #458.
- **Who reads it.** Anyone asking which conversation had its history rewritten, and so lost its
  prompt cache. That trail exists for nothing else.
- **Fix.** Pass the turn's session, ephemeral-nulled.
- **Golden:** no (replayed 11/11). No golden has a microcompact row.

### T6, T7, T8. Prompt-cache counts, and Anthropic's input tokens

- **T6.** `stream()` reads `cached_input_tokens` and `cache_write_tokens` off every completion.
  - Only the failure path (`llm_envelope.py:366`) passes them to `_record_usage_row`. The success
    path (`:409`) never did: PR #119's diff added them to the failure call only, and its tests
    exercise `record_run` directly, not the envelope.
  - Result: **no row in the table has ever held a cache count.** Since #119 merged on 2026-07-31,
    14,764 successful rounds have carried their token counts and none has carried a cache count.
- **T7.** The llama.cpp provider builds its `UsageSnapshot` from input and output tokens only
  (`llama_cpp.py:967`). The 4090's server sends `usage.prompt_tokens_details.cached_tokens` (and
  `timings.cache_n`) on every streamed completion. In the recorded completions, 81% of prompt tokens
  were cached: 109,592 of 135,507 across 39 at `5fb0000`, and 110,001 of 136,431 across 37 after
  #585 re-recorded `coding_run`.
- **T8.** Anthropic reports `input_tokens` without its two cache counters. The provider stores it as
  is, while the usage contract is that `input_tokens` is the whole prompt
  (`ToolCallTelemetry.last_request_tokens`). In the recorded `hosted_route` exchange that is 348 of
  15,083 tokens (2.3%): the rest was a 14,735-token cache write.
  - Metered cost is computed on 348, and cache writes are billed above the base rate.
  - Once T6 lands, a cache-hit ratio of cached ÷ input could exceed 1.

**Rows** (successful rounds with no cache count):

| why the count is NULL | 14d | 30d |
|---|---:|---:|
| on the wire, parsed, then dropped by the envelope (qwen 2,813 / 4,411; xai 4 / 26) | 2,817 | 4,437 |
| on the wire, never parsed, and dropped anyway (llama.cpp: T7) | 999 | 3,821 |
| parsed, then dropped (Anthropic) | 0 | 2 |
| **correct**: the provider sends nothing (Ollama) | 0 | 2 |
| **total** | **3,816** | **8,262** |

**Who reads it.**

- `/api/usage` returns `cached_input_tokens` per model as `COALESCE(SUM(…), 0)`, so it reports **0
  cached** where the truth is "not recorded". Beacon's usage view shows that 0.
- #119 added the columns to attribute cost per round, and that has never worked.
- The model ladder's rung metrics have no cache figure.

**Fix.**

- **T6:** pass the two counts on the success row, and add a test that goes through `stream()`.
- **T7:** call the shared `_parse_cache_usage` in the llama.cpp stream parser.
- **T8:** set `input_tokens` to input + cache read + cache creation in the Anthropic parser.
- **Reader half (not golden-changing):** make `/api/usage` say how many runs reported a cache count,
  instead of turning "none" into 0.

**Golden:**

- **T6:** yes, `hosted_route`, the Anthropic stand-in.
- **T7 alone:** no, because the envelope still drops the value.
- **T6 and T7 together:** 10 goldens: `hosted_route` and the 9 whose rounds run on llama.cpp.
  `repaired_tool_call` is unchanged: its rounds run on Ollama, which sends no cache count.
- **T8:** yes, `hosted_route`.

### T9. Local rounds billed `unknown`

- **Where.** `billing_for` (`telemetry/cost.py:185`), called at write time from the envelope
  (`llm_envelope.py:453`), decides from the model name alone: a path is local, a flat-plan host marker
  is subscription, a priced name is metered. A blank name and an Ollama `name:tag` are neither, so
  they come out `unknown`.
- **Rows.** 231 / 256. All are blank-name llama.cpp rounds: evals 189 / 214, coding runs 42 / 42.
  #585 (merged, not deployed) gives coding runs the served model's path, which bills `local`. The
  evals keep the config's blank name. No Ollama round has been stamped since the stamp began on
  2026-09-11, but `model_switch` and `repaired_tool_call` show that every Ollama round would be.
- **Who reads it.** `/api/usage` treats `unknown` as its one real gap in coverage, and Beacon's usage
  view shows it.
- **Fix.** `billing_stamp_full` classifies by provider class before the name. A llama.cpp or Ollama
  provider is `local`.
- **Golden:** yes, model_switch and repaired_tool_call. At `5fb0000` `coding_run` too; since #585 its
  rounds already bill `local`.

### T10. The compactor's model call has no tokens and no session column

- **Where.** `LLMCallEnvelope.call()` streams the completion but keeps only its text.
  `_record_success` (`llm_envelope.py:619`) then writes the model and billing with no tokens. The
  code says so itself: "NOT fixed here, deliberately, and worth its own change". The compactor's
  session goes into the summary, not the column.
- **Rows.** 123 / 175 `summarize_span` calls, all on the 4090 (424 all time). By surface, over 30
  days: telegram 99, beacon 40, desktop 25.
- **Who reads it.** `/api/usage`, and anyone accounting for the 4090's single slot.
- **Fix.** `call()` keeps the completion's usage and takes a `session_id`. That widens the rows of
  every `call()` user (memory extractor, knowledge synth, skill creator and refiner, curator), which
  is the point.
- **Golden:** yes. `compaction`.

### T11. Turn-path model calls that write nothing

- **Rows.** No row exists, so the counts come from what the calls leave behind:
  - LCM summaries written: 832 / 1,503;
  - session titles last written: ≤ 24 / ≤ 92, an upper bound, since a manual rename writes the same
    row;
  - `vision` tool calls: 0;
  - prompt hooks: none configured on the mini.
- **Who reads it.** As T10.
- **Fix.** Route them through the envelope's `call()` (after T10), each under its own subsystem name.
- **Golden:** yes. Titles alone: every scenario that names its session, which is all but `coding_run`
  (10).

### T12, T13. Which model served

- **T12.** `ApiMessageCompleteEvent.served_model` is set only by llama.cpp. The OpenAI-compatible,
  Anthropic and Ollama parsers ignore the `model` each stream names. In the recorded exchanges all
  three upstreams name it.
  - Rows: 3,100 / 4,967 tool calls, from `qwen3.8-max` (2,902 / 4,746), `qwen3.8-flash` (196 / 196)
    and `grok-4.5` (2 / 25).
  - Who reads it: the fine-tuning corpus cannot say which snapshot of a cloud alias produced a golden
    example. The model ladder lists served models.
- **T13.** `loop_round` rows keep `request.model`, the name asked for. The envelope sees
  `event.served_model` and drops it.
  - Rows: 231 / 1,776 rounds have a blank name and so no model at all (evals 189 / 1,734, coding runs
    42 / 42).
  - #585 (merged, not yet deployed) gives coding runs the served name. The evals' blank name comes
    from the config's empty `model.model` and stays.
- **Fix.**
  - T12a: the OpenAI-compatible and Anthropic parsers read the echoed model.
  - T12b: the Ollama parser does the same.
  - T13: the envelope adds `served_model` to the round's summary. That needs no schema change.
- **Golden:**
  - T12a: no (replayed 11/11). No golden has a tool call from those providers.
  - T12b: yes, `repaired_tool_call`, whose tool call is served by Ollama.
  - T13: yes. Alone it changes the 9 goldens with llama.cpp rounds, the only provider that reports a
    served model today. Once T12 lands, it changes `hosted_route` and `repaired_tool_call` too: all
    11.

### T14. A fallback-served round is recorded under the model that failed

- **Where.** On a terminal provider failure, `stream_round_with_fallback` serves the round from
  `context.fallback`. `_on_degrade` (`agent_loop.py:1567`) rewrites the identity line and nothing
  else.
  - The round's tool rows are written with `context.model` and
    `_provider_name_for_telemetry(context.provider)`: the cloud model that failed, and its cloud
    provider.
  - A successful zero-retry call from the local fallback therefore gets `is_golden = 1`. That is
    the direction `telemetry/tracker.py` calls the worse one: student output filed as a teacher
    example.
  - The round also keeps the cloud adapter (tier `off`), so no repairs run on the local model's
    output.
- **Rows.** 0 / 0. No tool row has ever named a cloud model with a local `served_model`, and no
  fallback-served round was found.
- **Fix.** The round's serving model and provider go to its tool rows and transitions.
- **Golden:** no (replayed 11/11). No scenario degrades.

### T15. Six failure writers drop the call's input

- **Where.** `hook_blocked`, `no_registry`, `unknown_tool`, `permission_denied` (both sites),
  `tool_timeout` and `tool_exception` write no `parsed_tool_call`. The three validation writers were
  given one for forensics. Their comment notes that `input_validation` rows once had 0 of 21 covered.
- **Rows.** 12 / 54 (`permission_denied` 10 / 51, `tool_timeout` 2 / 3), and 346 all time. For
  contrast, the validation writers kept the input on 190 rows.
- **Who reads it.** Beacon's tool feed shows the inputs from this column, so a denied or timed-out call
  shows none. Forensics has to dig the conversation store instead.
- **Fix.** Record it, null for an ephemeral turn, in the same PR as T1.
- **Golden:** yes. `gate_blocked` (its two `permission_denied` rows).

### Missing signals, not scheduled

These are gaps in what is recorded at all, not fields a writer leaves empty. Each would move every
golden it touched.

- `tool_advertisement` records counts, not names. The skill audit had to infer the advertised set from
  its size.
- `tool_calls` has no `tool_use_id` or round index. So the Instinct corpus and the golden exporter join
  to LCM by time, name and outcome.
- No row records the adapter tier or the parse path per round. The X.28 finding that coding runs sit at
  tier `full` came from reading code.
- `retries` is 0 on every executed call, so it measures the adapter's retry and nothing else.
- `thinking` is NULL on Ollama rounds, by design since #592, because thinking is per model there.

---

## 3. Golden impact, replayed

All replays ran on the mini from a throwaway clone, uv + Python 3.11.15. Each patch was applied to its
own detached worktree and replayed with `parity_harness.py replay`. The runs below are at `5fb0000`.
The two whose golden sets #585 changed (T1 + T15, T9) were run again at `d481b9e`, together with a
control and the non-golden bundle; see the end of this section.

| Run | Result | What differs |
|---|---|---|
| control (`5fb0000`) | 11/11 PARITY | — |
| T2 + T4 + T5 + T12a + T14 together | **11/11 PARITY** | — |
| T1 + T15 | 2 DIFF | coding_run's `validation_failed` row gains its session; gate_blocked's two `permission_denied` rows gain session and input |
| T3 | 7 DIFF | the `_loop_transition` rows of checkpoint_undo, coding_run, gate_blocked, linked_workspace, memory_write, repaired_tool_call, tool_calls gain their session |
| T6 | 1 DIFF | hosted_route: `cache_write_tokens` NULL → 14,735, `cached_input_tokens` NULL → 0 |
| T7 | **11/11 PARITY** | nothing on its own: the envelope still drops the value |
| T6 + T7 | 10 DIFF | hosted_route, and 28 llama.cpp rounds in the other 9 gain a cache count |
| T8 | 1 DIFF | hosted_route: `input_tokens` 348 → 15,083 |
| T9 | 3 DIFF | coding_run, model_switch, repaired_tool_call: `billing_mode` `unknown` → `local` |
| T10 | 1 DIFF | compaction: both `summarize_span` rows gain tokens and their session |
| T11 (titles only) | 10 DIFF | each title call gains a `subsystem_runs` row: every scenario but coding_run, which titles nothing |
| T12b | 1 DIFF | repaired_tool_call's tool call gains `served_model`. The first run also had a daemon boot failure in memory_write (a harness error); a re-run of the same tree was clean apart from repaired_tool_call. |
| T13 | 9 DIFF | 27 llama.cpp rounds gain `served_model` in their summary; hosted_route and repaired_tool_call wait for T12 |

In every DIFF run, the only categories were `tool_calls` and `telemetry`. **No model request, reply or
step changed.** The golden-changing fixes change what is recorded, never what is sent.

`golden_impact.py` predicts, from the committed dumps alone, which goldens each of these runs
changes. It matched every run.

**Again at `d481b9e`** (after #585 re-recorded `coding_run`):

| Run | Result | What differs |
|---|---|---|
| control (`d481b9e`) | 11/11 PARITY | — |
| T2 + T4 + T5 + T12a + T14 together | **11/11 PARITY** | — |
| T1 + T15 | 1 DIFF | gate_blocked only: the new `coding_run` makes no failed call |
| T9 | 2 DIFF | model_switch, repaired_tool_call: the new `coding_run` already bills `local` |

---

## 4. Fix plan

PRs in order. Each is its own PR, opened from `origin/main` and never from the audit's prototype.
Every one that touches `engine/` also runs the speed gate on the mini.

**Not golden-changing. These may get a go now.**

1. **Lucky guess out of `tool_calls` (T2).** The marker moves to `subsystem_runs` with its session.
   `report`, `health_summary`, `recent_tool_calls` and the ladder's harvest skip historical markers.
   Tests: the marker row, and each reader ignoring an old one. *Not golden-changing.*
2. **Microcompact under the turn's conversation (T5).** Pass the turn's session, ephemeral-nulled.
   Test: a web turn's row carries `web:<id>`. *Not golden-changing.*
3. **The served model from qwen, xai and Anthropic (T12a).** Both parsers read the echoed model.
   Tests over recorded chunk shapes. *Not golden-changing.*
4. **A session for session-less runs (T4).** A descriptive-only keyword from `run_async`/`run_loop`
   to the telemetry writers, used by POST `/api/chat` and the subagent spawner. Origin and routing are
   untouched. Two decisions for Will: the subagent's id, and whether POST `/api/chat` should also run
   as its session (a behaviour change, not telemetry). *Not golden-changing.*
5. **A degraded round belongs to its fallback (T14).** The round's serving model and provider reach
   its tool rows and transitions. The test asserts no golden flag on a local fallback.
   *Not golden-changing.*
6. **`/api/usage` says when cache counts were not recorded (reader half of T6).** It returns NULL
   rather than 0 when no row in a group recorded a count, plus how many runs did. It stays correct
   after PR 7 lands, when history is still NULL. *Not golden-changing:* no scenario step calls an
   HTTP reader, and readers write nothing.

**Golden-changing. These queue for the slot after WP-4.2.**

7. **Cache counts reach the row (T6 + T7 + T8).** The envelope's success row, the llama.cpp parser,
   and Anthropic's whole-prompt `input_tokens`. It restores the data #119 was written to collect, on
   every round. *Golden-changing: 10, all but `repaired_tool_call`* (its Ollama rounds report no
   cache).
8. **A session on every row the loop writes to `tool_calls` (T1 + T3 + T15).** The failure writers
   (with the call's input where missing) and `_log_iteration`. *Golden-changing: 7* (checkpoint_undo,
   coding_run, gate_blocked, linked_workspace, memory_write, repaired_tool_call, tool_calls).
9. **Which model served (T13 + T12b).** `served_model` in the round's summary, and Ollama's echoed
   model. *Golden-changing: all 11, once PR 3 (T12a) has landed;* 9 without it.
10. **Billing by provider class (T9).** *Golden-changing: model_switch, repaired_tool_call.*
11. **`call()` keeps usage and session (T10).** *Golden-changing: compaction.*
12. **Turn-path model calls through the envelope (T11).** LCM summaries and titles (and `vision` and
    prompt hooks, unused on the mini today), after PR 11. *Golden-changing: 10.*

**One slot, one re-record.** PRs 7–12 change only observables (§3), so they can share one re-record
of the affected goldens instead of six. No request changes, so a recording with every upstream
answered by a strict stand-in from the committed exchanges would reproduce the same requests and
replies. That is the technique already used for the alt and hosted sides. It would need no 4090 time.
The exception is `model_switch`: its two session-title requests race unless turn 2 takes real time,
so its alt must stay live, as in earlier re-records. Whether a stand-in recording counts as a
recording is Will's decision.

---

## 5. Other findings (not telemetry gaps)

- **A cloud run's tier bump switches on local-only machinery.** On 2026-09-23 a circuit-breaker trip
  on `qwen3.8-max` "recovered" by bumping the adapter from tier `off` to `light`
  (`circuit_breaker_diagnostics`). The rest of that run microcompacted 65 times, invalidating the
  provider's prompt cache each time, although the gate means to skip cloud tiers. T6 is why nobody
  could have seen the cost. `_TIER_BUMP_LADDER` maps `off` → `light`, and whether a cloud adapter
  should be bumped at all is a question for the X.28 series.
- **A client uses the literal id `web` for a conversation.** LCM holds 369 messages in it. In the last
  30 days it wrote 136 tool calls under `web`. The golden-trace exporter skips `web` as a routing
  namespace (`SHARED_SURFACE_IDS`), so those rows can never be exported. The permission layer treats
  `web` as a user surface.
- **POST `/api/chat` runs with origin `system` and without its session's routing override.** A
  WebSocket turn on the same conversation gets both. Noted under T4; not decided here.

---

## Reproduce

The scripts are in `docs/audits/telemetry-gaps/`. They print aggregates only. On the mini:

```
D=$(mktemp -d); cd docs/audits/telemetry-gaps
python3 backup_copy.py ~/.prometheus/telemetry.db "$D/telemetry.db"
python3 backup_copy.py ~/.prometheus/data/lcm.db "$D/lcm.db"
python3 gaps.py "$D/telemetry.db" --lcm "$D/lcm.db" --anchor <epoch of the copy> \
    --also-anchor 1790319300                                   # §2 and the table; 2026-09-25 06:55 UTC
rm -rf "$D"
```

On any checkout (the parity fixtures only):

```
python3 docs/audits/telemetry-gaps/wire_shapes.py      # what each provider sends (T6–T8, T12)
python3 docs/audits/telemetry-gaps/golden_impact.py    # which goldens hold the rows each fix touches
```

`golden_impact.py` is the static half of §3. The replays are the evidence.
