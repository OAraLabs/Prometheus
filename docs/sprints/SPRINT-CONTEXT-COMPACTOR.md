# SPRINT: Context Compactor (single-layer)

**Branch:** `feat/context-compactor`
**Status:** EXECUTED 2026-06-11 (implemented behind `compaction.enabled`, default OFF; PR-ready). This file is the corrected spec.
**Hard constraint:** This is SINGLE-LAYER compaction only. Three-layer compression is on the DO-NOT-BUILD list. If during design you find yourself proposing tiers of summaries-of-summaries, HALT and report instead.

> **Errata 2026-06-11** (branch `chore/spec-errata`): the Phase-0 premise that the live prompt is "assembled from LCM" is FALSE and was corrected during execution — live assembly is the in-memory session message list; `LCMAssembler` is not in the live path; LCM is a write-mirror. Consequence folded in below: substitution happens on the per-call RENDER VIEW (the list handed to `render_messages_for_model`), and the spec's "cache keyed on LCM node IDs" became content-hash keys because in-memory messages carry no LCM row ids. The single-layer guarantee is structural (synthetic summaries are span barriers). Full rationale: session summary `audits/20260611T055841Z-session-summary.md`; design detail: `audits/20260611T050051Z-context-compactor-checkpoint1.md`.

## Concept

When the assembled prompt approaches the context window, summarize the oldest conversational span via the same local model and substitute the summary into the per-call RENDER VIEW only — the in-memory message list handed to the provider at assembly time. **Erratum:** the live loop assembles from the in-memory session list, NOT from LCM (LCM is a durable write-mirror, not the live read path); so "storage untouched" is structural — the live path never reads lcm.db at all, and the synthetic summary is substituted into the rendered request, never written back. The LCM DAG remains the untouched source of truth. Inspired by the context-compactor design in the Odysseus project (MIT); clean-room, design knowledge only.

## Phase 0 — Survey (read-only, mandatory halts)

```
git fetch origin && git rev-parse HEAD
```

Cite SHA. HALT if dirty tree / wrong branch / behind origin.

Cite file:line for:
1. **Prompt assembly**: where the system prompt + history is assembled into the final request; what currently happens when assembled tokens exceed n_ctx (truncation? error? silent overflow?). This answer shapes everything — report it explicitly. **Erratum (as found):** assembly is from the IN-MEMORY session message list at the model-call site (`engine/agent_loop.py`, the `render_messages_for_model(messages)` call), NOT from LCM — `LCMAssembler` has no live callers. Current overflow behavior: nothing measures total assembled tokens before the call; an oversized prompt reaches the llama.cpp server and fails provider-side, the gateway catches the exception and replies `"Error: …"` with session rollback. There is no in-repo overflow handler — this provider-side failure is exactly the documented fallback the compactor falls back TO on its own failure.
2. **Token estimation**: any existing token-count utility; if none, note what the provider layer exposes.
3. **n_ctx awareness**: where the configured context length lives and whether the daemon knows it per-model.
4. **LCM read path**: confirm reads for prompt assembly are separable from the stored DAG (i.e., we can substitute at assembly time without writing to lcm.db). **Erratum (as found):** stronger than assumed — live assembly performs ZERO lcm.db reads, so substitution structurally cannot touch the DAG. Consequence: the spec's idempotence cache "keyed on the span's LCM node IDs" is unimplementable at this layer (in-memory messages carry no LCM row ids — the M7 mapping gap); the implementation keys the cache on `sha256(session_id + span messages' role/provenance/content_json)` instead. Same mechanism, same layer, content-addressed rather than row-addressed.
5. **Existing summarization**: any prior summarizer (MemoryExtractor or similar) whose call pattern/prompt style should be matched.
6. **LLMCallEnvelope**: confirm wrapper for the summarization call.

**HALT CHECKPOINT 1**: findings + citations + an explicit statement of current overflow behavior. Wait for approval.

## Phase 1 — Implementation

`context/compactor.py` (path per repo conventions):

- Trigger: estimated assembled tokens > `compaction.threshold_pct` (default 0.75) of effective context (n_ctx minus reserved output budget, `compaction.reserve_tokens` default 4096).
- Selection: oldest contiguous span of turns beyond a protected tail (`compaction.protect_recent_turns`, default 8 turns) and excluding the system prompt, pinned content, and any turn carrying unresolved task state (managed-task notes, trust-tagged injections — identified via the LCM fields from Phase 0).
- Summarization: ONE call through the provider layer (LLMCallEnvelope), same model as the session, prompt instructing factual compression preserving: user goals, decisions made, file paths, identifiers, unresolved questions. Hard output cap.
- Substitution: at assembly time, the span is replaced by a single clearly-marked synthetic message ("[Compacted summary of turns N–M] ..."). The synthetic message is tagged through provenance fields as compactor-generated, not user/model authored.
- Idempotence: a compacted summary is cached keyed on the span's LCM node IDs so repeated assemblies don't re-pay the summarization call; cache invalidates only if the span changes (it shouldn't — LCM is append-oriented).
- **Fail loud**: if the summarization call fails or the result fails a sanity check (empty, or longer than the span it replaces), do NOT silently fall back to overflow. Log at error level, emit a SignalBus event, and fall back to the pre-existing behavior found in Phase 0 item 1 — explicitly, with a system-visible note. Telemetry records every compaction: spans, tokens before/after, duration, failures.

## Tests (side effects, not call-counting)

- Synthetic long conversation fixture exceeding threshold → assert assembled prompt token estimate drops below threshold AND the summary message is present with correct provenance tag AND lcm.db rows for the original span are byte-identical before/after (storage untouched — this is the load-bearing assertion).
- Protected-tail test: recent N turns never compacted.
- Pinned/trust-tagged exclusion test.
- Failure-path test: summarizer fixture returns garbage → loud failure, fallback engaged, telemetry row exists.
- Idempotence test: second assembly performs zero additional model calls (assert via telemetry count, which IS a side effect record).
- `python3 -m pytest` green; recorded fixtures only, no live model in CI.

**HALT CHECKPOINT 2**: implementation map + test results before any wiring into the default assembly path. Feature ships behind `compaction.enabled` (default false) — flipping the default is a separate decision for the PR review.

## Acceptance

1. Disabled by default; suite green with zero behavior change.
2. Enabled: fixture conversation compacts correctly, lcm.db untouched, telemetry complete.
3. Pre-commit passes without `--no-verify`.
4. PR follow-ups recorded: tuning thresholds against real sessions, surfacing compaction events in Beacon, interaction with mid-session model switching (future work — different n_ctx per model).

## Out of scope

- Multi-tier / hierarchical summarization (DO-NOT-BUILD).
- Compacting tool schemas or the system prompt.
- Any write to lcm.db.
