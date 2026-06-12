# SPRINT: Coding Mode (white-room) — SKELETON / NOT READY FOR EXECUTION

**Branch:** `feat/coding-mode` (do not create yet)
**Status:** DRAFT. Gated on TWO inputs: (1) the BAKEOFF report at `~/bakeoff-harness/BAKEOFF-REPORT-20260611.md` (executed 2026-06-11), which sets the port-priority ranking in Phase 2; (2) a same-cycle spec-vs-implementation audit of the middle-layer tool-calling subsystem. Fable: if handed this spec without both attached, HALT immediately and request them.
**Provenance discipline:** White-room. Design knowledge from OpenHands (MIT) — tool semantics, sandbox abstraction shape, recovery behaviors — reimplemented natively. No OpenHands source copied. Exception, pending Will's per-case approval at Checkpoint 2: recovery/condensation PROMPT TEXT may be adapted with attribution, since prompts are not code.

> **Errata 2026-06-11** (branch `chore/spec-errata`): two gate corrections. (1) Gate input #2 referenced `SPRINT-TOOL-CALLING-MIDDLE-LAYER.md`, which **does not exist and never existed in git history** — the audit instead runs against the surviving six-feature enumeration in `AUDIT-2026-06-8e5adf0.md §4.4.6`; the cycle audit satisfying this gate is `audits/20260611T050051Z-middle-layer-audit.md`. (2) Gate input #1 now names the actual executed bakeoff report path. See also the openhands version-skew warning below before any Phase-1 work that builds an openhands comparison harness.

> **⚠ openhands version-skew trap (erratum, observed 2026-06-11):** if any phase here stands up an openhands-sdk comparison/reference harness, pin BOTH `openhands-sdk==1.17.0` and `openhands-tools==1.17.0` (they are separate packages; a naive install floats them to incompatible versions, and SDK 1.28.0 is blocked by an `lmnr`/`opentelemetry-instrumentation` resolver conflict). In 1.17.0 the terminal tool is `TerminalTool` (no `BashTool`) and `get_default_agent(llm, cli_mode=True)` is the preset. Full detail in `BAKEOFF-harness.md` step 1.

## Concept

A native Prometheus coding capability: a managed task that drives an agent loop over a target repo with coding-specific tools, inside an execution sandbox, terminating at a reviewable artifact (branch + diff). Never merges. Never pushes to main. Human gate is structural, not policy.

## Scope (what ships)

1. **Editor tool family** (native Tool classes, GBNF-constrained):
   - `code_view(path, range?)` — numbered-line file/dir view
   - `code_str_replace(path, old, new)` — old must match exactly once; mismatch and multi-match are loud, distinct errors
   - `code_create(path, content)` — fails if path exists
   - `code_grep(pattern, path_glob)` / `code_glob(pattern)`
   Semantics deliberately mirror the str_replace-editor convention (view-before-edit, unique-match) because the local model has seen that convention in training data — familiarity is reliability.
2. **Sandbox abstraction**: one interface, `ProcessSandbox` backend first (subprocess + cwd jail + env scrub + resource/time limits). `DockerSandbox` is a follow-up, interface-shaped now, not built now. All coding tools and bash execution route through the sandbox; SecurityGate denied_paths enforced at the sandbox boundary IN ADDITION to the gate (defense in depth). The sandbox root is the task's target repo checkout — a dedicated clone/worktree, never the live Prometheus tree.
3. **Coding loop policies** (port order set by bake-off Q3 ranking):
   - patch-failed recovery (re-view then re-replace, never blind retry)
   - repeated-identical-failure breaker (3 strikes → step back and re-plan, ties into Sprint A detector signals)
   - test-run discipline (run acceptance/tests before declaring done)
   - context condensation for long sessions (REUSES Sprint B compactor — do not build a second compactor; if coding sessions need different protect/threshold settings, that's config, not code)
4. **Managed-task lifecycle**: a coding run is a managed task in tasks.db — durable, SignalBus events, Telegram/Beacon notification on completion, honest-async-promise validation applies.
5. **Terminal artifact**: feature branch in the target repo + diff summary + test results, surfaced for review. Repo-local git identity. No push without explicit instruction; never to main.
6. **Teacher integration**: Sprint A escalation active inside coding mode — failed coding turns are exactly the high-value golden traces and skills the flywheel wants.

## Phase plan (to be expanded post-gate)

- **Phase 0**: standard survey with git preamble + halts. Key citations needed: managed-task registration path; GBNF schema definition path; how a long-running multi-turn loop differs from the single-turn loop today (this may be the largest unknown — report honestly if the current loop architecture can't host a 25-round coding session without changes, and HALT there rather than improvising loop surgery).
- **Phase 1**: editor tools + tests. Standing rules: every Tool gets a registration test AND a side-effect output test (the MemoryTool orphan must not recur); orphan-tool grep added to this sprint's checklist.
- **Phase 2**: ProcessSandbox + routing.
- **Phase 3**: loop policies per bake-off ranking. HALT CHECKPOINT before prompt adaptation (provenance exception approval).
- **Phase 4**: managed-task wiring + end-to-end fixture: a planted-bug fixture repo → coding task → branch + passing test as the acceptance artifact.

## Open decisions (resolve before un-drafting)

1. Does the current agent loop support a bounded multi-round session of this length, or does coding mode need its own loop runner? (Phase 0 answers; affects everything.)
2. Round/time caps per coding task — defaults?
3. Should coding mode be reachable from Telegram, or Beacon/API only at first?
4. Worktree vs full clone for the sandbox root?
5. Which fixture repo becomes the permanent e2e test target?

## Out of scope

- Merging, pushing to main, or any autonomy past the reviewable artifact.
- DockerSandbox implementation (interface only).
- Prometheus-as-orchestrator / `ask_orchestrator` relay automation (separate future work; and no programmatic orchestration per DO-NOT-BUILD).
- A second context-compression mechanism.
