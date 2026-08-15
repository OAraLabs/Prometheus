# Autonomous Sprint Session Summary — 2026-06-11

**Sequence directive:** (1) middle-layer audit → (2) SPRINT-teacher-escalation → (3) SPRINT-context-compactor → (4) stretch: BAKEOFF Phase-0 setup only → session summary. SPRINT-coding-mode untouched per directive.
**Result: steps 1–3 COMPLETE and green; step 4 complete except the smoke test itself, which is blocked on one input only you can provide (below). No sprint hard-stopped.**

## What completed

### 1. Middle-layer spec-vs-implementation audit (gates Sprint A Phase 3)
`audits/20260611T050051Z-middle-layer-audit.md` — read-only, at `main c59fe34` (== origin/main, fetched). Headline: the spec file `SPRINT-TOOL-CALLING-MIDDLE-LAYER.md` **does not exist and never existed in git history** (independently re-verified; matches AUDIT-2026-06-8e5adf0.md:68,427), so the audit ran against the surviving six-feature enumeration (§4.4.6) + the 2026-06-10 closeout. All six features present, none regressed; 13 commits landed since the prior audit (C1, C2, H1–H5, M1/M2/M4/M5/M6 fixes verified in place); three known divergences persist (dashboard `adapter_repairs` mislabel, M8 wrong-tool example_call + 12/52 coverage, L11 no-decay) — all already on the closeout follow-up list, zero new findings. Gate: SATISFIED.

### 2. Sprint A — `feat/teacher-escalation` (PR-READY)
`main c59fe34` + 3 commits: `b601b3c` (detector + teacher engine), `62e1df6` (post-turn integration + /escalations + daemon wiring), `a8a48ea` (PR-DESCRIPTION.md).
- Phases 0–3 executed; both checkpoints CONFIRMED-proceed (reports: `…teacher-escalation-checkpoint{1,2}.md`).
- Suite: **2520 passed, 28.2s** (main baseline 2467; +53 tests). Pre-commit hook clean on every commit.
- All five acceptance criteria met, including the full fixture e2e (corrective reply delivered with visible note → SKILL.md on disk → `signal_events` golden-trace row → `lcm_messages` row tagged `("teacher_escalation", is_trusted=0)`), the teacher-also-fails gate, and the inert-by-default guarantee.
- `PR-DESCRIPTION.md` in the branch root carries the config example, the decisions made where the spec was silent, and the follow-up list (Tier-2 detector out of scope, Beacon surfacing, gateway-parity gap, post-merge live smoke).

### 3. Sprint B — `feat/context-compactor` (PR-READY)
`main c59fe34` + 2 commits: `d2ab007` (compactor + wiring), `5156ede` (PR-DESCRIPTION.md).
- Both checkpoints written (`…context-compactor-checkpoint{1,2}.md`). Checkpoint 1 verdict PROCEED with one survey-corrected spec assumption (below) — judged anticipated-adjustment, not contradicted-premise; flagged prominently for your review.
- Suite: **2478 passed, 28.4s** (+11). Default OFF (`compaction.enabled` absent) → zero behavior change; flipping the default is the PR review's decision per spec.
- The load-bearing test holds: lcm.db full-table dumps identical before/after compaction (dump-identity, not raw file bytes — SQLite WAL makes file bytes unstable; documented). Single-layer is structural (summaries are span barriers). Idempotence asserted via `subsystem_runs` count (one summarize call across repeat assemblies).

### 4. Stretch — BAKEOFF Phase 0 setup (`~/bakeoff-harness/`, outside the repo)
Report: `~/bakeoff-harness/READINESS-20260611T055724Z.md`.
- venv + **openhands-sdk 1.17.0 + openhands-tools 1.17.0** (SDK-only, no agent-server) — import-verified agent stack.
- Fixture: **marshmallow @ `27bfa77`** (MIT, 4,972 src LOC, 12 modules; baseline 1178 tests green at the pin).
- **15 tasks (T1/T2/T3 ×5) with deterministic acceptance commands, frozen before any run** (`tasks/tasks.json` + exact-anchor plant scripts). `verify_readiness.py` run clean: every acceptance gate fails pre-work, plants apply, fixture resets clean.
- Detector imported standalone from a local clone of `feat/teacher-escalation` — works.
- Daemon REST alive (HTTP 200, `llama_cpp`, model file `google_gemma-4-26B-A4B-it-Q4_K_M.gguf`); token read from its file, never inline.
- **Smoke test PENDING — the one open item: `BAKEOFF_LLM_BASE_URL` is unset and your directive says ask, not guess.** To run: `cd ~/bakeoff-harness && BAKEOFF_LLM_BASE_URL=<llama.cpp server>/v1 ./venv/bin/python smoke_openhands.py` (exit 2 = the spec's halt condition, reported with traceback — that result would conclude the bake-off early).

## What stopped / unverified (honest accounting)

- **Nothing hard-stopped.** Every checkpoint reached a CONFIRMED/PROCEED verdict; the two judgment calls (middle-layer audit target substitution; Sprint B's corrected assembly assumption) are documented in their reports for you to overrule.
- **Unverified:** (a) the bakeoff smoke (env var); (b) whether the RUNNING daemon's code == main (process uptime ~9.1h; REST responds, code provenance not provable from outside); (c) the live smoke-test safeguard (`scripts/smoke_test_tool_calling.py`) was NOT run — both branches are unmerged, the live daemon is untouched, so per the standing rule it belongs before+after the merge/restart that picks these up (listed in both PR descriptions); (d) Sprint B wired daemon-only (CLI path untouched; flag off by default).
- Working tree left on `feat/context-compactor`… then returned to `main` at session end; all session artifacts (`audits/`, this file) untracked as directed. `main` was never committed to or merged.

## Branch / test state

| Branch | Tip | vs main | Suite |
|---|---|---|---|
| `main` | `c59fe34` | — (untouched, == origin/main) | 2467 passed (baseline) |
| `feat/teacher-escalation` | `a8a48ea` | +3 commits | 2520 passed |
| `feat/context-compactor` | `5156ede` | +2 commits | 2478 passed |

Both branches local-only (nothing pushed), each with `PR-DESCRIPTION.md` at its root. Merge note: both extend the `Provenance` Literal — whichever lands second has a one-line conflict; keep both values.

## Checkpoint / report artifacts (all untracked)

- `audits/20260611T050051Z-middle-layer-audit.md`
- `audits/20260611T050051Z-teacher-escalation-checkpoint1.md` / `…checkpoint2.md`
- `audits/20260611T050051Z-context-compactor-checkpoint1.md` / `…checkpoint2.md`
- `~/bakeoff-harness/READINESS-20260611T055724Z.md`
- `audits/20260611T055841Z-session-summary.md` (this file)

## Ranked: what the specs got wrong (or reality contradicted)

1. **The middle-layer audit's target document doesn't exist** (`SPRINT-TOOL-CALLING-MIDDLE-LAYER.md`, referenced as a hard prerequisite by both the teacher and coding-mode specs). The audit ran against the §4.4.6 enumeration — strongest possible form — but if a real spec file exists somewhere off-box, nothing has ever been checked against its actual language. Worth resolving before SPRINT-CODING-MODE, which cites it again.
2. **Compactor spec item 1's premise** ("history is assembled from LCM into the final request") is false — live assembly is the in-memory session list; LCM is a write-mirror. Consequence: the spec'd "cache keyed on LCM node IDs" is unimplementable at that layer (content-hash keys used instead). The spec's own Phase-0 framing anticipated this; flagged at both checkpoints.
3. **Teacher spec Phase 3** says "extend the existing traces command" — no `/traces` command exists in any gateway. The spec's own fallback (`/escalations`) was used.
4. **Teacher spec detector signature** (`detect_failure(tool_results: list, final_reply: str)`) cannot express the repetition signal ("same tool called with identical args") without call arguments — `tool_results` items had to be the richer trace-dict shape (kept the signature, documented the contract).
5. **Bakeoff spec setup step 1** ("pip install openhands-sdk") is insufficient on its own: tools live in a separate `openhands-tools` package, and naive co-installation produces an incompatible version skew (sdk 1.17.0 ↔ tools 1.28.0; sdk 1.28 is blocked by a real `lmnr`/opentelemetry resolver conflict). Pin both to 1.17.0.
6. **"SecurityGate constraints apply" to the skill writer** (teacher spec) is a category mismatch — SecurityGate gates tool execution, not subsystem file writes. The intent (cannot touch config or denied paths) is satisfied structurally by SkillCreator's slug confinement, and now tested (hostile `name: ../../config/prometheus` stays inside `skills/auto/`).
7. Minor environment note: the session directive said "uv is not installed" — it exists at `~/.local/bin/uv`. All test runs used `python3 -m pytest` as directed; no functional impact.

## Suggested next actions (yours)

1. Provide `BAKEOFF_LLM_BASE_URL` → run the smoke (one command, above).
2. Review/squash-merge the two branches (order doesn't matter; trivial Provenance conflict either way), then daemon restart + the standing before/after live smoke.
3. Decide the two flagged judgment calls: compactor cache-keys + `is_trusted=True` on synthetic summaries; teacher-reply `is_trusted=False` + non-delivery when the teacher fails the detector.

---

## BAKEOFF EXECUTED (later session, 2026-06-11) — outcome

The bakeoff (BAKEOFF-harness.md) was executed to completion. Report:
`~/bakeoff-harness/BAKEOFF-REPORT-20260611.md` (outside the repo). 60/60 runs.

**Result:** openhands-sdk 17/30 (57%) vs Prometheus 3/30 (10%); openhands swept
T3 9/10 and T2 3/10, Prometheus 0 on both. **Both arms 0 malformed calls** —
the GBNF-death-spiral premise did NOT reproduce (the server runs `--jinja` →
server-side tool grammar, so both arms are grammar-enforced; Q2 reframed to
"client GBNF vs server grammar", and grammar is not the differentiator).

**Two first-order caveats** the report leads with: (1) thinking-mode ASYMMETRY
— Prometheus suppresses Gemma's thinking every turn, openhands runs thinking-on;
the headline gap is openhands(loop+thinking) vs Prometheus(loop−thinking),
entangled. (2) Prometheus's dominant failure = silent_wrong_answer (21/27): it
declares done in ~3 rounds, ran pytest in only 3/30 runs. openhands runs tests
2–11× and iterates (13.8 rounds avg).

**Q3 port priority for coding-mode:** (1 cheapest) config experiment — enable
bounded thinking on Prometheus coding turns, gated on `--reasoning-budget`,
to isolate the confound; (2) test-run discipline (run tests before done); (3)
persistence + task_tracker planning; (4) editor semantics (already the spec
design, confirmed sound).

**Daemon was RESTARTED to current main (c59fe34) at 09:50** before the matrix —
the running process was stale (started 16:37 the prior day, predated 511ed2f).
`systemctl --user restart prometheus.service` (no sudo, --user unit). It remains
on current main (do not revert). All 60 rows verified post-restart on c59fe34.

**Spec errata:** committed on `chore/spec-errata` (`daf125a`, PR-ready). Five
report follow-ups (F1–F5): agent-loop-bypasses-LLMCallEnvelope (wiring), lcm
batch-persist durability gap, false-confident-invisible-to-Tier1, 9.2k schema
tax / deferred-loading-off, revive --reasoning-budget backstop.
