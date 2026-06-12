# SPRINT: Agent Loop Inside the Envelope (F1)

**Branch:** `feat/loop-envelope`
**Status:** Ready. Small, surgical. PRECURSOR to SPRINT-coding-mode v2 — merge this first.
**Origin:** Bakeoff finding F1 (BAKEOFF-REPORT-20260611.md): the agent loop's model calls
bypass LLMCallEnvelope entirely; UsageSnapshot is produced by the loop but never
persisted. The main execution path sits outside the silent-failure capture layer, in
direct violation of the standing principle that ALL _call_model patterns use the
envelope. This is the produced-but-unwired orphan pattern on the most important call
site in the system.

## Phase 0 — Survey (read-only, mandatory halts)

git fetch origin && git rev-parse HEAD — cite SHA; HALT on dirty tree / behind origin /
wrong branch.

Cite file:line:
1. Every model-call site in the agent loop (the path the bakeoff exercised) and exactly
   why each bypasses the envelope today (different call shape? predates envelope?).
2. LLMCallEnvelope's current interface and what the autonomous-subsystem call sites pass
   it; identify any interface gap that blocked loop adoption (e.g., streaming, tool
   schemas, per-round metadata).
3. Where UsageSnapshot is produced in the loop and every place it currently flows
   (suspected answer: nowhere — confirm).
4. telemetry.db schema: where a per-call usage row belongs (new table vs extending the
   envelope's existing table). Prefer extending the envelope's table so all call sites
   share one query surface.

**HALT CHECKPOINT 1**: findings + a one-paragraph wrap plan. If the envelope interface
cannot host the loop's call shape without redesign, HALT — that's a planning
conversation, not an improvisation.

## Phase 1 — Implementation

- Wrap every loop model call in LLMCallEnvelope. Behavior-preserving: identical request
  payloads (assert via recorded fixture diff), identical streaming behavior, identical
  error propagation — the envelope OBSERVES and RECORDS; it must not change semantics.
- Persist per-call usage: input/output tokens (from UsageSnapshot), duration, round
  index, session id, model id, and the effective thinking flag for the call (needed by
  coding-mode telemetry; the suppress_thinking override shipped at 1c1ba6c — record what
  was actually sent).
- Silent-failure capture: empty-content, exception, and budget-exhausted outcomes
  recorded with the same fail-loud envelope conventions as other subsystems.

## Tests (side effects)

- Fixture-driven loop turn → assert telemetry.db usage row EXISTS with correct tokens,
  round index, thinking flag.
- Payload-equivalence test: request bytes identical pre/post wrap (recorded fixture).
- Failure-path test: provider fixture raises mid-round → envelope row records it AND the
  loop's existing error behavior is unchanged.
- Full suite green: python3 -m pytest. No --no-verify.

## Acceptance

1. Zero behavior change to any agent turn (payload-equivalence + suite green).
2. Every loop model call produces a usage row; the bakeoff's "prometheus tokens = null"
   condition is structurally impossible going forward.
3. PR follow-ups: (a) backfill note — pre-merge traces have no usage rows, re-baseline
   economics dashboards; (b) F4 investigation (9.2k/round tool-schema tax,
   deferred-loading currently OFF) — separate experiment, one variable at a time, now
   measurable for free via the new usage rows.

## Out of scope
- Any change to prompt assembly, schemas, or sampling. Observation only.
- F4 itself (measure later with this sprint's instrumentation).
