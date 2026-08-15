# SPRINT-CONTEXT-COMPACTOR — Checkpoint 1 (Phase 0 survey)

**Verdict: PROCEED.** One spec assumption was corrected by the survey (item 1: live history is NOT assembled from LCM) — but the spec's Phase-0 framing explicitly treats this as the question that "shapes everything," and the design (assembly-time substitution, storage untouched) lands unchanged, in fact *stronger*: the live path never reads lcm.db at all. One spec detail adjusted in consequence (cache keys). Both flagged below for review; judged anticipated-adjustment, not contradicted-premise. Everything else confirmed.

**Branch:** `feat/context-compactor` @ `c59fe34` (fresh off `main` == `origin/main`, fetched this session). Tree: zero tracked modifications (untracked owner artifacts only). Independent of Sprint A (branched from main; the only overlap will be a trivial `Provenance` enum merge conflict — noted for the PR).

All paths `src/prometheus/`-relative.

## Survey findings

### 1. Prompt assembly + current overflow behavior (the load-bearing answer)
- The final request is rendered **per model call from the in-memory message list**: `engine/agent_loop.py:803-814` — `ApiMessageRequest(model=…, messages=render_messages_for_model(messages), system_prompt=per_call_system_prompt, tools=…)`. History enters from `session.get_messages()` at the gateways (`gateway/telegram.py:1761-1772`); `AgentLoop.run_async` shallow-copies (`:2124-2131`); `run_loop` mutates its own copy (appends, microcompaction).
- **`LCMAssembler` is NOT in the live path** (`memory/lcm_assembler.py:57 assemble()` — no callers outside the memory package). LCM is write-mirror only in the live turn: `session._persist_to_lcm` (`engine/session.py:236-270`).
- **Current overflow behavior — explicit statement:** NOTHING measures total assembled tokens before the call. Existing bounds are indirect: per-result truncation (`tool_result_max` 4000, `:1938-1946`), per-turn results budget (8000, `:1306-1359`), in-run microcompaction (`:1361-1428`), and `session.trim(50)` (message COUNT, not tokens). A prompt exceeding the model's real context reaches the llama.cpp server as-is; the provider raises / the server errors (or server-side-truncates per its own config), the gateway catches the exception and replies `"Error: …"` with session rollback (`gateway/telegram.py:1799-1802`). **There is no in-repo overflow handler — failure is provider-side.** This is the documented fallback behavior the compactor falls back TO on its own failure.
- Two **built-but-unwired** context controls exist (prior audit's tested-but-unwired class, still unwired at c59fe34): `context/budget.py` (`TokenBudget`, model-aware limits, `is_approaching_limit(threshold=0.75)`) and `context/compression.py` (`ContextCompressor`: Tier-1 prune + Tier-2 batch summarization, `_SUMMARY_BATCH_SIZE=8`) — zero imports from engine/daemon (verified). Live config carries their intended keys (`compression_trigger: 0.75`, `fresh_tail_count: 32`, `config/prometheus.yaml:18,22`), all currently inert.

### 2. Token estimation — EXISTS
`context/token_estimation.py:10` `estimate_tokens` (chars/4 heuristic); already used by truncation, budget, and the loop's per-turn budget (`agent_loop.py:1316`).

### 3. n_ctx awareness — EXISTS, per-model
`context.effective_limit` (72000 live) + `context.model_overrides.<model>.effective_limit` (`config/prometheus.yaml:17,25-29`); resolution precedent in `ContextBudget.from_config` (`context/budget.py:62-88`, file-based). The compactor resolves the same keys from the already-loaded config dict (no file I/O).

### 4. LCM read path — CONFIRMED (stronger than assumed)
Prompt assembly performs zero lcm.db reads, so assembly-time substitution structurally cannot write (or even touch) the DAG. **Adjustment:** the spec's idempotence cache "keyed on the span's LCM node IDs" cannot be implemented at this layer — in-memory messages carry no LCM row ids (the known M7 mapping gap). Cache keys are `sha256(session_id + span messages' (role, provenance, content_json))` — same mechanism, same layer, content-addressed instead of row-addressed. Flagged for owner review.

### 5. Existing summarization to match — LCMSummarizer
`memory/lcm_summarize.py:38-48` `_MESSAGE_SUMMARY_PROMPT` — "preserve decisions, code changes/file paths, action items/open questions, names/entities" — matches the spec's required preservation list almost verbatim; `max_tokens=1024` (`:171`); circuit breaker on consecutive failures. The compactor's prompt follows this style. NOTE: lcm_summarize also has `_NODE_SUMMARY_PROMPT` (summaries-of-summaries) — that is LCM's own pre-existing durable hierarchy, untouched by this sprint. The new compactor is strictly single-layer BY CONSTRUCTION: the synthetic summary message carries `provenance="compactor"` and non-"user"-provenance messages are span barriers, so a summary can never be ingested into a later span. Multiple sibling summaries can accumulate in a very long session — siblings, never layers.

### 6. LLMCallEnvelope — CONFIRMED
`learning/llm_envelope.py:104` (verified importable + used this session by Sprint A); compactor uses `subsystem="context_compactor"`, `on_failure="return_none"` — failures land in `silent_failures` + failed `subsystem_runs` for free.

## Design decisions (for review at checkpoint)
- **New module per spec** (`context/compactor.py`); the dead `ContextCompressor` is left untouched (different semantics: multi-call batches, no cache/tagging/telemetry). PR follow-up: delete-or-absorb it (prior audit already recommended wire-or-delete).
- **Synthetic message:** `role="user"` (injected-context convention: results/steers/nudges are user-role), `provenance="compactor"` (new closed-enum value), **`is_trusted=True`** — machinery-authored from trusted source material; untrusted injected turns can never be IN a span (they're barriers), and `True` avoids wrapping the user's own history summary in the untrusted-input banner. Flagged for review.
- **Trigger estimate** covers system prompt + messages + tool-schema chars (honest total), against `effective_limit − compaction.reserve_tokens` (default 4096) × `threshold_pct` (default 0.75).
- **Protected tail** counts user-role messages (`compaction.protect_recent_turns`, default 8) — same convention as microcompaction's fresh-window walk and the compressor's fresh_tail.
- **Pinned content:** no pin mechanism exists in the repo (grepped) — the spec's pinned-exclusion is vacuously satisfied; recorded rather than invented.
- **Fail loud:** sanity check (non-empty AND summary tokens < span tokens) → on any failure: ERROR log + `context_compaction_failed` signal event + telemetry, and fall back to the pre-existing behavior (send unmodified, provider-side failure mode) — "system-visible note" interpreted as the /events-visible signal row + ERROR log, not a chat injection from inside the loop.
- **Wiring:** `LoopContext.compactor` (default None) + `AgentLoop(compactor=…)` + daemon builds from config when `compaction.enabled` (default false → `from_config` returns None → zero behavior change); SignalBus late-wired in the SENTINEL block. CLI path intentionally not wired this sprint (flag is off by default; noted in PR).
- Substitution happens on a **render view** passed to `render_messages_for_model` — the loop's `messages` list, the session, and lcm.db are never mutated. The load-bearing test compares full lcm.db table dumps before/after (dump-identical; raw file bytes are WAL-unstable even without writes — documented).
