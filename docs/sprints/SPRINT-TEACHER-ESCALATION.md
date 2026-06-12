# SPRINT: Teacher Escalation + Skill Flywheel

**Branch:** `feat/teacher-escalation`
**Status:** EXECUTED 2026-06-11 (Phases 0–3 complete on the branch; PR-ready). This file is the corrected spec.

> **Errata 2026-06-11** (branch `chore/spec-errata`): folds the corrections from the bakeoff/sprint session into the spec text so it matches what was actually built. Changes here: (1) the Phase-3 gate no longer references the non-existent `SPRINT-TOOL-CALLING-MIDDLE-LAYER.md` — it points at the surviving six-feature enumeration audit; (2) the Phase-1 detector signature is annotated with the dict shape the implementation actually consumes; (3) the Phase-3 `/traces` reference is corrected to `/escalations` (no `/traces` command exists in any gateway). Rationale and the full errata list live in the session summary `audits/20260611T055841Z-session-summary.md`.

**Prerequisite:** None for Phases 0–2. Phase 3 (loop integration) touches the agent loop and is gated on a **same-cycle spec-vs-implementation audit of the middle-layer tool-calling subsystem**. NOTE (erratum): `SPRINT-TOOL-CALLING-MIDDLE-LAYER.md` does not exist and never existed in git history — the audit therefore runs against the surviving six-feature enumeration in `AUDIT-2026-06-8e5adf0.md §4.4.6`. The cycle audit satisfying this gate for the executed run is `audits/20260611T050051Z-middle-layer-audit.md`. If no such cycle audit exists, HALT at the Phase 3 gate and report.

## Concept

When the local model fails an agent turn, escalate the full failed context to a configured cloud teacher model. The teacher produces (a) a corrective reply delivered to the user and (b) a SKILL.md procedure so the local model can handle that task class next time. Every escalation also records a golden trace into telemetry.db for the LoRA flywheel. Two flywheels, one trigger.

Pattern credit: clean-room reimplementation inspired by the teacher-escalation design in the Odysseus project (MIT). No source code is to be copied. Design knowledge only: tiered failure detection, skill-persisted-only-if-teacher-passes-detector.

## Phase 0 — Survey (read-only, mandatory halts)

```
git fetch origin && git rev-parse HEAD
```

Cite the SHA in your report. HALT immediately if: working tree dirty, not on a fresh branch off current main, or local main behind origin.

Survey and cite file:line for each. Do NOT write code in this phase.

1. **Skills system**: where skills are stored, how SKILL.md files are created/loaded/injected into the system prompt, whether a programmatic write path exists.
2. **Cloud override path**: how `/claude` routes a turn to the Anthropic provider; whether a single turn can be sent to a different provider mid-session without switching the session's primary model.
3. **Golden traces**: telemetry.db schema for traces; how a trace is currently recorded; what tagging fields exist.
4. **Trust tagging**: the provenance/is_trusted fields through the LCM schema (from feat/honest-async-promises); confirm injected content can be tagged.
5. **Tool result surface**: where in the agent loop tool outputs and the final assistant reply are available post-turn (the post-turn validator from feat/honest-async-promises is a likely anchor point).
6. **Provider telemetry**: confirm LLMCallEnvelope is importable and is the required wrapper for any new model call.
7. **Endpoint classification**: is there an existing way to determine "current model is local/self-hosted vs cloud"? Cite it or report absence.

**HALT CHECKPOINT 1**: Report findings with citations. List any assumption in this spec that the survey contradicts. Wait for approval.

## Phase 1 — Failure Detector (Tier 1, deterministic)

New module: `escalation/detector.py` (adjust path to repo conventions found in Phase 0).

Pure function: `detect_failure(tool_results: list, final_reply: str) -> FailureVerdict` where `FailureVerdict` carries `failed: bool`, `reasons: list[str]`, `matched_patterns: list[str]`.

> **Erratum (signature detail):** `tool_results` is a list of dicts in the repo's trace shape `{"tool_name": str, "arguments": dict|str, "result": str, "is_error": bool}` (the SkillCreator trace shape plus an error flag). The bare `list` in the signature cannot express the repetition signal ("same tool called with identical args ≥3×") without the per-call `arguments`, so the items carry them. `is_error` is optional; absent, an anchored error-text fallback is used so informative results (e.g. grep's "(no output)") are not miscounted as errors.

Tier-1 signals (regex/deterministic only — no LLM judging in this sprint):
- Tool errors the model did not recover from: last tool result in the turn is an error/timeout AND the reply does not acknowledge a retry plan.
- Capability denial in reply: patterns like "I don't have a tool", "I am unable to", "Unknown action".
- Clarification stall: reply is only a question back to the user when the turn's instruction was concrete (heuristic: reply < N chars AND ends with "?" AND contains no tool call).
- Repetition: the same tool called with identical args ≥3 times in one turn.
- Empty/whitespace reply after tool activity.

Patterns live in a module-level list with a comment per pattern explaining what real failure it catches. This detector is also consumed by `BAKEOFF-harness.md` — keep it dependency-light (stdlib only).

Tests: fixture transcripts (passing and failing) checked into tests/fixtures; unit tests for every pattern; at least 3 negative fixtures that look failure-adjacent but must NOT trigger (e.g., model legitimately asking a clarifying question on an ambiguous request).

## Phase 2 — Teacher Call + Skill Writer

New module: `escalation/teacher.py`.

Trigger conditions (ALL must hold, checked in this order, each logged when it blocks):
1. Agent mode (tool-capable turn), not plain chat.
2. Current primary model classified as local/self-hosted (use the Phase 0 finding; if absent, add a minimal classifier keyed on the configured endpoint host — config-driven, no hardcoded hostnames or IPs).
3. `escalation.teacher_model` is set in config (default: unset → feature inert).
4. Detector returned `failed=True`.
5. Per-session escalation budget not exhausted (config `escalation.max_per_session`, default 3 — prevents loops and cost runaway).

Teacher call:
- Goes through the existing provider layer, wrapped in LLMCallEnvelope. No new HTTP client.
- Prompt contains: the user's request, the failed transcript (tool calls + results + final reply), the detector's reasons, and instructions to produce TWO fenced sections: `CORRECTIVE_REPLY` and `SKILL_DRAFT` (SKILL.md format matching the repo's existing skill format found in Phase 0).
- Parse both sections deterministically. Missing section = teacher failure → log, count, do not persist anything, fall through to the local model's original reply plus a visible system note that escalation failed. Fail loud in telemetry, never silently.

Skill persistence gate:
- Run the SAME Tier-1 detector against the teacher's CORRECTIVE_REPLY. Persist SKILL_DRAFT only if the teacher passes. A teacher that also stalled or denied capability writes nothing.
- Skill writes go through the existing skill-creation path (not raw file writes) so existing validation applies. SecurityGate constraints apply; the skill writer must not be able to touch `config/prometheus.yaml` or any denied path.

Golden trace:
- Record the full escalation exchange in telemetry.db tagged `source=teacher_escalation`, including detector reasons and whether a skill was persisted. This is LoRA flywheel input.

Trust tagging:
- The corrective reply injected into the session is tagged through the LCM provenance fields as teacher-sourced, is_trusted per the conventions found in Phase 0 item 4.

Tests (side effects, not call-counting):
- Wiring test: simulated failed turn with a recorded teacher fixture → assert the SKILL.md file EXISTS on disk with expected content, assert the telemetry.db row EXISTS with correct tags. (Recorded fixtures only — no live network in CI.)
- Gate test: teacher fixture that itself fails the detector → assert NO skill file written, telemetry row records the rejection.
- Budget test: 4th escalation in a session is refused and logged.
- If any new Tool class is introduced: registration test AND side-effect output test, per standing rule.

**HALT CHECKPOINT 2**: Report Phase 1–2 implementation with file:line map and test results (`python3 -m pytest`). Wait for approval before Phase 3.

## Phase 3 — Agent Loop Integration

GATE: confirm the middle-layer audit has been run this cycle (see Prerequisite). HALT if not.

- Hook the detector at the post-turn point identified in Phase 0 item 5. Prefer extending the existing post-turn validator over adding a second post-turn pass.
- Escalation runs after the local turn completes; the user sees the local reply replaced/augmented by the corrective reply with a brief visible note ("escalated to teacher"). No silent substitution.
- Telegram/Beacon notification path unchanged; escalation events emit on SignalBus consistent with managed-task events.
- New `/escalations` command showing count fired / skills written / teacher failures / budget state. (Erratum: the spec originally said "extend the existing `/traces` command" — no `/traces` command exists in any gateway, so the spec's own fallback was taken: a new `/escalations` command. As built it lives in the Telegram gateway alongside the honesty validator.)

## Acceptance

1. With `teacher_model` unset, zero behavior change (assert via existing test suite green).
2. Fixture-driven end-to-end: failed local turn → corrective reply delivered → SKILL.md on disk → telemetry row tagged → LCM provenance tagged.
3. Teacher-also-fails fixture → no skill, loud telemetry.
4. `python3 -m pytest` green; pre-commit hook passes without `--no-verify`.
5. PR description carries the follow-up list, including: Tier-2 LLM self-eval detector (explicitly out of scope this sprint), surfacing escalation stats in Beacon.

## Out of scope (do not build)

- Tier-2 LLM-based failure evaluation.
- Any automatic retry loop where the LOCAL model re-attempts using the new skill within the same turn.
- Programmatic orchestration of any kind (DO-NOT-BUILD list).
