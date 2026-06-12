# SPRINT: Coding Mode v2 (FINAL — un-drafted, evidence-ordered)

**Branch:** `feat/coding-mode`
**Status:** READY. Gates satisfied: BAKEOFF-REPORT-20260611.md, BAKEOFF-ADDENDUM-
thinking-on-20260611.md, middle-layer audit (20260611T050051Z). Supersedes the v1
skeleton in full.
**Prerequisite:** `feat/loop-envelope` (F1) merged first — coding turns must be
envelope-recorded golden traces from day one.
**Provenance:** White-room from OpenHands (MIT) design knowledge. Standing exception,
approved scope: recovery/iteration PROMPT TEXT may be adapted with attribution (prompts
are not code); all code is native.

## What the evidence settled (drives every scope decision below)

- Addendum Q1 decomposition: thinking-on closed T1 (7/10 vs OH 5/10) and T2 (4/10 vs
  OH 3/10) with the existing loop. Edit-task capability is NOT a gap. Thinking explains
  ~64% of the original 47-pt deficit.
- The residual is T3 only: 1/10 vs 9/10 WITH thinking on. OH's T3 wins are
  edit → run tests → read failures → re-edit, persisting 10–25 rounds. Prometheus
  thinking-on runs tests sometimes (14/30) but never iterates to green. THE SPRINT IS
  THE ITERATE-TO-GREEN LOOP POLICY. Everything else is supporting cast.
- Q2 settled: 0 malformed calls in all three arms; server-side --jinja grammar covers
  native calls. Client-side GBNF is NOT a reliability requirement for the new tools —
  strict schemas suffice; do not build an extra grammar layer for them.
- silent_wrong_answer was 21/27 of thinking-off failures and persists in spirit at T3:
  the model's self-report of success has zero evidentiary value. Ground truth = test
  execution, always.
- De-prioritized by evidence: task_tracker/planning scaffolding (redundant where
  thinking-on already matches OH), heavy condensation (reuse Sprint B compactor;
  coding-session settings are config, not code).

## Scope

### 1. Per-task thinking control (config, do first — addendum priority #1)
The global override shipped at 1c1ba6c. Coding mode needs PER-CALL scope: a coding turn
sets suppress_thinking=false on its own provider calls without flipping the global
default. Plumb a per-call override through the provider path (the envelope from F1
records the effective flag per call — use it to assert in tests). Gated assumption:
server runs a bounded --reasoning-budget; Phase 0 verifies and HALTS if unbounded.

### 2. Editor + execution tools (native Tool classes, strict schemas, no extra GBNF layer)
- code_view(path, range?) — numbered lines
- code_str_replace(path, old, new) — exactly-once match; multi-match and no-match are
  distinct, loud errors instructing re-view
- code_create(path, content) — fails if exists
- code_grep(pattern, glob) / code_glob(pattern)
- code_run(cmd) — sandboxed execution (tests, builds); is_read_only=false, truthfully
Every tool: registration test AND side-effect output test (standing rule; run the
orphan-tool grep as part of this sprint's checklist).

### 3. ProcessSandbox
One interface; ProcessSandbox backend now (subprocess, cwd jail = dedicated FULL CLONE
of the target repo — not a worktree; hard isolation over disk savings, decided), env
scrub (no tokens, no provider keys beyond what the loop injects), wall/round resource
limits. SecurityGate denied_paths enforced at the sandbox boundary in addition to the
gate. DockerSandbox: interface-shaped, not built.

### 4. THE CORE: iterate-to-green loop policy (addendum priority #2, the irreducible port)
Coding turns run under a policy layer the generic loop does not have:
- DONE IS A VERDICT, NOT A CLAIM. The model cannot finish a coding task by assertion.
  Finish requires the task's acceptance command (or repo test invocation) executed via
  code_run in the current round or the one prior, with exit 0. A finish attempt without
  it is rejected back to the model with the policy stated.
- On test failure: the failure output (tail-truncated, structured) is injected and the
  model must edit before re-running; two consecutive identical failure signatures
  trigger a step-back instruction (re-view the failing site before further edits);
  three trigger the Sprint A escalation path if configured (failed coding turns are
  premium golden traces).
- Patch-failure recovery: str_replace no-match/multi-match → mandatory re-view of the
  target span before retrying the edit. No blind retries.
- Caps: default 30 rounds / 20 min per task (evidence: OH T3 wins ran 10–25 rounds at
  the bakeoff's 25-round cap; give one tier of headroom). Cap exhaustion = honest
  abandonment with the artifact in whatever state, clearly labeled — never a success
  claim.
- Condensation: Sprint B compactor with coding-profile config (tighter
  protect_recent_turns); zero new compression code.

### 5. Managed-task lifecycle + terminal artifact (unchanged from v1)
A coding run is a tasks.db managed task: durable, SignalBus events, Telegram/Beacon
completion notification, honest-async validation applies. Terminal artifact: feature
branch in the sandbox clone + diff summary + final test output, surfaced for review.
Never merges, never pushes; repo-local identity. Reachability v1: Beacon/API only
(artifact review needs a screen); Telegram trigger is a follow-up.

### 6. E2E fixture
Permanent fixture: the bakeoff's marshmallow clone @ 27bfa77 with the frozen T3 tasks —
acceptance for this sprint reuses bakeoff tasks t11–t15 directly: coding mode must
convert at least 3/5 of them (current loop: 0–1/5; OH: ~4.5/5). That number is the
sprint's success metric, measured with the bakeoff runner.

## Phase plan
- Phase 0: survey + halts (standard git preamble). Key citations: managed-task
  registration path; per-call provider kwarg path for thinking; CAN THE LOOP HOST A
  30-ROUND BOUNDED SESSION — the v1 skeleton's biggest unknown, still open. HALT with a
  written report if the loop needs structural surgery; that returns to planning.
  Verify bounded --reasoning-budget on the server; HALT if absent.
- Phase 1: tools + sandbox (scope items 2–3).
- Phase 2: per-call thinking (item 1) — small, lands with tests against F1's envelope
  rows.
- Phase 3: iterate-to-green policy (item 4). CHECKPOINT before adapting any OpenHands
  prompt text: list the exact passages and their sources for approval.
- Phase 4: managed-task wiring + the t11–t15 acceptance run via the bakeoff runner;
  report the conversion rate honestly, both runs per task.

## Out of scope (evidence-backed cuts)
- task_tracker / planning scaffolding (addendum: redundant on T1/T2 with thinking-on).
- Client-side GBNF layer for the new tools (Q2: zero marginal validity here).
- DockerSandbox implementation; Telegram trigger; any merge/push autonomy.
- New compression code of any kind.
- Orchestrator/relay automation (separate horizon item; DO-NOT-BUILD applies).
