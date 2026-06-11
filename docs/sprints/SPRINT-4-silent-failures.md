# SPRINT 4 — Silent Failure Eradication

**Type:** Two-phase audit + implementation
**Branch:** `audit/silent-failures` (Phase 1) → `feat/silent-failure-fixes` (Phase 2)
**Estimated time:** 2–3 days total (Phase 1: 4–6 hours, Phase 2: 1–2 days)
**Prerequisites:** PR #1 merged. Sprint 1 stable on main.
**Trust level:** Phase 1 is read-only. Phase 2 modifies internal code paths (test infrastructure, telemetry schema, exception handling) — no user-facing behavior changes.

---

## Why this sprint exists

PR #1 / commit `ed8f1a6` fixed a three-line `ValidationError` that had been silently disabling SkillCreator, SkillRefiner, and MemoryExtractor for weeks. The wiring tests passed the entire time. The failure mode:

1. Pydantic schema for `ConversationMessage.content` went strict (became `list[ContentBlock]`)
2. Three callers continued passing `content=prompt` (a string)
3. Each call raised `ValidationError` inside `_call_model`
4. Each `_call_model` had a broad `except` block that caught and silently returned
5. Wiring tests confirmed the call graph reached `_call_model` — they did not confirm `_call_model` produced output
6. Telemetry recorded the invocation but not the silent exception
7. Three core learning subsystems were structurally wired and functionally inert

Without the soak revealing this, the pattern would have continued indefinitely. This sprint exists to:

- Find every other silent-failure pattern in the codebase
- Close the test-category gap (wiring tests must verify function, not just structure)
- Surface future silent failures immediately via telemetry

Sprint 2 (`/steer` + agent loop modifications) sits behind this sprint deliberately. The agent loop is the worst possible place for an unsurfaced silent failure, and Sprint 2's changes are exactly where one could land. This sprint clears the ground first.

---

## PHASE 1 — AUDIT

**Branch:** `audit/silent-failures`
**Output:** `docs/audits/SILENT-FAILURE-AUDIT.md`
**Estimated time:** 4–6 hours

This phase is read-only. No code changes. After Phase 1 lands, halt and report to Will for review before Phase 2.

### Step 0 — Fix the checkout (mandatory)

```bash
git fetch origin
git checkout origin/main
git rev-parse HEAD                          # cite this in the report
git log -1 --format='%H %s' origin/main     # confirm PR #1 has merged
```

If PR #1 has not merged, halt and tell Will. The audit baseline is post-PR-#1 main.

### Step 1 — Find every exception-swallowing pattern

For each file under `src/prometheus/`:

```bash
grep -rn "except.*:" src/prometheus/ \
  --include="*.py" \
  -A 5 | less
```

For each `except` block, categorize:

- **KEEP** — intentional swallow with documented reason. Examples:
  - Optional dependency import (`except ImportError: # optional`)
  - Best-effort cleanup (`except: # best-effort, ignored`)
  - Tested fallback path with logged warning
- **CONVERT** — swallows silently; should log at WARN+ and/or write to telemetry
- **RE-RAISE** — caught generically when it should propagate (e.g., bare `except:` or `except Exception:` with no logging)
- **HIGH-RISK** — swallows in a load-bearing code path (agent loop, tool execution, model adapter, hook executor, gateway dispatch). Flag for immediate fix.

Pay specific attention to these patterns that have a high risk of being silent failures:

- `_call_model` methods (Curator already correct; verify Hermes-attribution divergence didn't introduce new variants)
- Tool execution wrappers (`tools/base.py`, `tools/registry.py`)
- Hook executor (`hooks/executor.py`)
- Provider adapters (`providers/*.py`)
- Gateway message dispatch (`gateway/telegram.py`, `gateway/slack.py`, `gateway/commands.py`)
- SignalBus emission paths (Sprint 1 added emissions to existing wired code — verify the emissions themselves don't swallow)
- Memory write paths (`memory/extractor.py`, `memory/MemoryTool`)
- SENTINEL phases (`sentinel/*.py`)
- GRAFT-SYMBIOTE pipeline (`symbiote/*.py`)
- Adapter Layer (`adapter/*.py` — this is the moat, audit carefully)

### Step 2 — Find wiring tests that don't verify function

For each test in `tests/test_wiring.py` (and any tests/wiring/ subdirectory):

- Read the test
- Identify what it asserts
- Categorize:
  - **STRUCTURAL-ONLY** — asserts only that a class exists, a hook is registered, an attribute is set
  - **FUNCTIONAL** — asserts that calling the subsystem produces an observable side effect (file written, row inserted, message emitted, state changed)
  - **MIXED** — asserts both

For STRUCTURAL-ONLY tests, draft what a functional assertion would look like. Don't write the test in Phase 1; draft it.

### Step 3 — Find unsurfaced telemetry gaps

For each subsystem that runs autonomously (Curator, SkillCreator, SkillRefiner, MemoryExtractor, PeriodicNudge, SENTINEL phases, GEPA engines, BackupVault, MorphEngine, GRAFT-SYMBIOTE pipeline stages):

- Identify what it writes to `telemetry.db` on success
- Identify what it writes to `telemetry.db` on failure
- If failure isn't explicitly recorded (no row written, no `is_error=True` flag, no failure-category column), flag the subsystem

The check is "could you tell, by querying telemetry.db alone, whether this subsystem has been silently failing for weeks?" If the answer is no, that's a gap.

### Step 4 — Re-baseline metrics (read-only observation)

Record the current state of the project against post-`ed8f1a6` reality:

- Count of files in `~/.prometheus/skills/auto/` (skill creation has been broken; expect mostly empty)
- Current `telemetry.db` golden trace count (`SELECT COUNT(*) FROM ... WHERE is_golden=true`)
- Current MEMORY.md and USER.md character counts
- Any other metrics quoted in `userMemories`, `README.md`, or recent project notes

These are the post-fix baseline. Will compare against these over the next 30 days to see whether the now-functional subsystems actually produce output.

### Phase 1 output format

`docs/audits/SILENT-FAILURE-AUDIT.md`:

```
# Silent Failure Audit

**Verified against:** origin/main at <SHA>
**Date:** <today>
**Triggered by:** PR #1 / ed8f1a6 — three-line ValidationError fix

## Summary

| Category | Count |
|---|---|
| Exception swallows — KEEP | x |
| Exception swallows — CONVERT | x |
| Exception swallows — RE-RAISE | x |
| Exception swallows — HIGH-RISK | x |
| Wiring tests — STRUCTURAL-ONLY | x |
| Wiring tests — FUNCTIONAL | x |
| Subsystems missing failure telemetry | x |

## High-risk silent failures

(Anything categorized HIGH-RISK. Each entry: file:line, current code, why it's
high-risk, recommended fix.)

## Exception swallows — categorized

(Table of every swallow with file:line, current code one-liner, category, and
one-sentence recommendation.)

## Wiring tests — categorized

(Table of every wiring test with name, current assertion shape, category, and
one-sentence draft of what a functional assertion would add.)

## Subsystems missing failure telemetry

(List of subsystems that don't record their own failures, with one-sentence
recommendation for what telemetry row should land on failure.)

## Re-baseline metrics

(Post-ed8f1a6 snapshot of skill count, golden trace count, MEMORY/USER sizes,
plus any other quoted metrics.)

## Phase 2 recommendations

Ranked list of fixes to implement in Phase 2:
1. HIGH-RISK swallows (immediate)
2. CONVERT swallows in core paths (agent loop, tool execution, adapter)
3. Shared _call_model telemetry envelope
4. Functional assertions added to existing wiring tests
5. New /health command surfacing silent failures
6. (whatever else surfaced)

Each item: estimated effort (S/M/L), dependencies, risk.
```

### Phase 1 halt and checkpoint

After writing `docs/audits/SILENT-FAILURE-AUDIT.md`:

1. Commit the audit doc to `audit/silent-failures` branch
2. Push the branch
3. Open a draft PR (NOT marked ready for review) so Will can read the audit
4. Stop. Do not start Phase 2 until Will reviews and approves the Phase 2 recommendations.

The checkpoint exists because the audit will surface things we didn't predict, and Will needs to decide which of the recommendations to implement and in what order. Some HIGH-RISK items may need immediate hotfixes outside the sprint; some CONVERTs may not be worth the effort. The audit is the input to that decision, not the decision itself.

---

## PHASE 2 — IMPLEMENTATION

**Branch:** `feat/silent-failure-fixes` (off main, after Phase 1 audit reviewed)
**Estimated time:** 1–2 days
**Prerequisites:** Phase 1 complete, Will has approved the Phase 2 recommendations list

### Read These Files First

Before any implementation:

- `docs/audits/SILENT-FAILURE-AUDIT.md` (your own Phase 1 output, now with Will's prioritization annotations)
- `src/prometheus/learning/curator.py` — the `_call_model` shape that worked (use this as the reference for the shared envelope)
- `src/prometheus/learning/skill_creator.py`, `skill_refiner.py` — the fixed shape from ed8f1a6
- `src/prometheus/memory/extractor.py` — the fixed shape from ed8f1a6
- `src/prometheus/telemetry/` (wherever telemetry writes live)
- `tests/test_wiring.py` — every test classified STRUCTURAL-ONLY in Phase 1

### Hermes source (read for patterns; do NOT clone)

For the telemetry envelope and failure surfacing patterns, check whether Hermes has equivalents:

- `https://github.com/NousResearch/hermes-agent` — search for telemetry, health check, status command, silent_failure handling
- `hermes_agent/` tree — look for any `_call_model` or LLM-invocation envelope pattern they may have already solved

If a clean equivalent exists, attribute per the standing Hermes addendum. If not, build native.

### Work Stream 1 — Shared `_call_model` telemetry envelope

**Goal**

Every subsystem that calls the LLM does it through a shared envelope that:

1. Logs the invocation start at INFO
2. Catches every exception
3. Writes a `telemetry.silent_failures` row (or equivalent) on every exception
4. Re-raises by default, unless the caller explicitly opts in to swallow
5. Records success metadata (tokens, latency, output size) on completion

**Implementation**

Create `src/prometheus/learning/llm_envelope.py`:

```python
class LLMCallEnvelope:
    """Shared envelope for LLM invocations in autonomous subsystems.

    Replaces the per-subsystem `_call_model` pattern that silently swallowed
    ValidationError for weeks (see PR #1).

    Usage:

        envelope = LLMCallEnvelope(
            subsystem="SkillCreator",
            telemetry=telemetry_store,
            on_failure="raise",   # "raise" | "log_only" | "return_none"
        )
        result = envelope.call(provider, messages, ...)

    Always writes a telemetry row:
      - On success: subsystem, latency_ms, tokens, output_size
      - On failure: subsystem, exception_type, exception_msg, traceback,
        is_silent_failure=True
    """
```

Migrate Curator, SkillCreator, SkillRefiner, MemoryExtractor to use the envelope. Their `_call_model` methods become thin wrappers around `envelope.call(...)`.

For new subsystems (anything Sprint 2 or Sprint 3 adds), the envelope is the default pattern.

**Tests**

- `test_envelope_records_success` — successful call writes a telemetry row
- `test_envelope_records_failure` — exception writes a `silent_failures` row
- `test_envelope_re_raises_by_default` — `on_failure="raise"` propagates
- `test_envelope_swallows_when_explicit` — `on_failure="log_only"` doesn't propagate
- `test_envelope_validates_message_shape` — if `messages[*].content` is a string instead of list[ContentBlock], envelope catches it explicitly (cite ed8f1a6)

### Work Stream 2 — Functional wiring tests

**Goal**

Every wiring test classified STRUCTURAL-ONLY in Phase 1 gains a "and the side effect happened" assertion.

**Pattern**

Before (structural-only):

```python
def test_skill_creator_wired_to_post_task_hook(self):
    daemon = create_daemon(...)
    assert daemon.skill_creator is not None
    assert daemon.skill_creator in daemon.hooks.post_task
```

After (functional):

```python
def test_skill_creator_produces_skill_on_post_task(self, tmp_skills_dir):
    daemon = create_daemon(skills_dir=tmp_skills_dir)
    # Fire a fake completed task with multi-tool trajectory
    daemon.signal_bus.emit(TaskCompleted(trajectory=multi_tool_trajectory))
    # Assert the side effect actually happened
    skill_files = list(tmp_skills_dir.glob("auto/*/SKILL.md"))
    assert len(skill_files) >= 1, "SkillCreator wired but produced nothing"
```

**Coverage requirement**

For each STRUCTURAL-ONLY test from Phase 1:

- Either upgrade it in place to include a functional assertion
- Or add a paired functional test alongside it (if the structural assertion has standalone value as a sanity check)

Aim for full coverage of the Phase 1 STRUCTURAL-ONLY list. Acceptable to defer 3–5 tests with explicit reasoning (e.g., "this would require mocking too much of the LLM provider; tracked as separate work").

**Fixtures**

Add `tests/fixtures/llm_responses.py` with canned multi-tool trajectories and canned LLM responses, so functional tests can run without hitting a real provider. Hermes likely has similar fixtures — check upstream first.

### Work Stream 3 — `/health` command

**Goal**

A Telegram (and Slack, if wired post-Sprint 3) command that surfaces silent failures from telemetry in real time, so the next instance of this pattern gets caught in days, not weeks.

**Implementation**

Add `/health` to `gateway/commands.py`:

```
🩺 Prometheus Health — last 24h

✅ Tool calls:        1,247 (3 failures, 0 silent)
✅ Curator runs:      3 / 3 successful
⚠️  SkillCreator:     21 invocations, 19 successful, 2 SILENT FAILURES
✅ SkillRefiner:      8 / 8 successful
✅ MemoryExtractor:   47 / 47 successful
✅ SENTINEL phases:   84 / 84 successful

Silent failures detected (most recent 5):
  - 2026-05-21 14:32  SkillCreator  ValidationError: ...
  - 2026-05-21 09:18  SkillCreator  TimeoutError: ...
  ...

Run /health verbose for full breakdown.
```

The query is a single SELECT against `telemetry.silent_failures` joined with the subsystem invocation table. Easy implementation; the value is the visibility.

**Tests**

- `test_health_command_surfaces_zero_failures` — clean state returns ✅ across the board
- `test_health_command_flags_silent_failures` — inject a `silent_failures` row, confirm `/health` flags it
- `test_health_command_groups_by_subsystem` — multiple failures from same subsystem aggregate correctly

### Work Stream 4 — Implement Phase 1 HIGH-RISK items

These are the recommendations from Phase 1's audit that Will marked for immediate fix. The exact list is unknowable until Phase 1 completes; this work stream is a placeholder for "do whatever Phase 1 + Will's review identified as high-priority."

Each item gets:

- A commit on this branch
- A test covering the fix
- A note in the PR description linking back to the audit entry it addresses

---

## Acceptance criteria for the whole sprint

After Phase 2 merges:

1. The exception-swallowing audit doc lives in `docs/audits/` and is the project's source of truth for "what could fail silently"
2. Every wiring test that was STRUCTURAL-ONLY in Phase 1 either has a functional assertion or has explicit reasoning for why it doesn't
3. `LLMCallEnvelope` is in use by Curator, SkillCreator, SkillRefiner, and MemoryExtractor; future subsystems are documented to use it by default
4. `telemetry.silent_failures` table exists and is being populated
5. `/health` command works in Telegram, surfaces real telemetry, flags any silent failures from the last 24h
6. The pattern that produced PR #1's bug can no longer recur silently — if a `_call_model` throws ValidationError tomorrow, `/health` shows it within minutes, not weeks

---

## Constraints

- Phase 1 is audit-only. No code changes. Halt for Will's review before Phase 2.
- Phase 2 modifies internal infrastructure, not user-facing behavior. Telegram users should see no change except the addition of `/health`.
- Do NOT touch Adapter Layer behavior. If the audit flags swallows in the Adapter Layer (which it might — the moat does have try/except blocks for provider failures), do not modify them in this sprint. Flag for separate decision.
- Do NOT touch LCM. Same reason.
- Provenance discipline. If any pattern is adapted from Hermes, attribute.
- Branch hygiene. `audit/silent-failures` for Phase 1, `feat/silent-failure-fixes` for Phase 2. No commits to main. Will squash-merges both.
- Phase 1 outputs a draft PR for review. Phase 2 outputs a normal PR.

---

## Reporting back

### After Phase 1

1. Branch `audit/silent-failures`, audit doc path
2. Counts: HIGH-RISK / CONVERT / RE-RAISE / KEEP swallow counts; STRUCTURAL-ONLY / FUNCTIONAL test counts; subsystems missing failure telemetry
3. Top 5 HIGH-RISK swallows by risk
4. Top 5 wiring-test functional-assertion additions ranked by importance
5. Re-baseline numbers (skill count, golden trace count, MEMORY/USER sizes)
6. Draft PR URL

Halt for Will's review.

### After Phase 2

1. Branch `feat/silent-failure-fixes`, commit SHAs
2. PR URL
3. Confirmation of acceptance criteria 1–6
4. List of Phase 1 items that were implemented vs deferred, with reasoning
5. Sample `/health` output from a real telemetry query
6. Any drive-by findings — note, don't fix
