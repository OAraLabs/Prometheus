# BAKEOFF: Prometheus Loop vs openhands-sdk (same local model)

**Location:** scratch directory OUTSIDE the Prometheus working tree (e.g. `~/bakeoff-harness/`). Nothing in this document touches the Prometheus repo. No branch. No PR. Output is a report only.
**Prerequisite:** `feat/teacher-escalation` Phase 1 (the detector module) exists on its branch — imported directly from the branch checkout, not merged.
**Decision this informs:** which OpenHands behaviors the white-room `SPRINT-coding-mode.md` ports first, and whether GBNF-side reliability already beats litellm native function calling for the current local model.

> **Errata 2026-06-11** (branch `chore/spec-errata`): Phase 0 setup was executed 2026-06-11; this note records the install trap found then so a future runner doesn't lose hours to it. See the version-skew warning on step 1 and the timestamped bakeoff report in `~/bakeoff-harness/` for the executed-run results.

## Setup (Phase 0 — halts)

1. Fresh venv in the scratch dir. `pip install openhands-sdk` (and `openhands-agent-server` only if the SDK alone cannot run an agent locally — prefer SDK-only).
   > **⚠ Version-skew trap (erratum, observed 2026-06-11):** `pip install openhands-sdk` alone is insufficient — the tool-using agent needs the SEPARATE `openhands-tools` package, and a naive install floats the two to incompatible versions (SDK resolved to 1.17.0 while `openhands-tools` floated to 1.28.0, whose module layout expects `openhands.sdk.utils.path`, absent in 1.17.0 → `ModuleNotFoundError`). Upgrading the SDK to 1.28.0 is itself blocked by a real resolver conflict (`lmnr` pins `opentelemetry-instrumentation==0.63b1`). **Working combination: pin BOTH to the same version (`openhands-sdk==1.17.0` + `openhands-tools==1.17.0`).** Note also that in 1.17.0 there is no `BashTool` — the terminal tool is `TerminalTool`, and `get_default_agent(llm, cli_mode=True)` is the supported preset (cli_mode disables browser tooling). Native function calling is via litellm's `openai/<served-model>` route against the llama.cpp `/v1` endpoint.
2. Inference endpoint comes from env var `BAKEOFF_LLM_BASE_URL` (the existing llama.cpp server). NO hardcoded IPs or hostnames anywhere in scratch code, in case files are later shared.
3. Smoke test: drive ONE trivial task ("create hello.txt containing 'hi'") through openhands-sdk against the endpoint.
   - **HALT CONDITION**: if openhands-sdk cannot complete native function calling against llama.cpp at all (malformed-call death spiral, provider incompatibility), STOP. Do not write adapters, do not patch the SDK. Report the exact failure — that result alone answers the main question (GBNF wins by forfeit) and the bake-off concludes early.
4. Prometheus side: drive turns through the existing daemon REST API as a normal client (bearer token read from its file on the host per standing rule — never inline in code or transcripts). Prometheus daemon must be on current main; record `git rev-parse HEAD` of the daemon in the report.
5. Record exact model file name, n_ctx, and sampler settings in the report header. Both harnesses MUST hit the same server config.

## Task set

15 tasks against a dedicated fixture repo (clone a small, permissively-licensed Python project ~2–5k LOC into the scratch dir; pin the commit in the report). Three tiers, five tasks each:

- **T1 single-file**: rename a function and its call sites in one file; fix a planted off-by-one; add a docstring matching repo style; add a parameter with default; delete dead code block.
- **T2 multi-file**: rename across 3+ files; move a function between modules and fix imports; add a new module wired into an existing entry point; change a return type and all consumers; extract duplicated logic into a shared helper.
- **T3 test-driven**: given a failing test, make it pass; write a test for an untested function then make repo tests green; fix a planted bug only discoverable by running tests; refactor under a green test suite; resolve a planted circular import.

Each task has a deterministic acceptance command (a pytest invocation or a grep/diff assertion) defined BEFORE any run. Reset the fixture repo to the pinned commit between every task (`git checkout -- . && git clean -fd`).

## Metrics per task per harness

- success: acceptance command exit 0 (binary)
- turns/rounds to completion or abandonment
- wall time
- total tokens (Prometheus: telemetry.db; openhands-sdk: its event/usage records)
- malformed tool-call count (Prometheus: GBNF makes this structurally ~0, confirm from telemetry; openhands: count function-call parse/validation errors in its event log)
- failure classification via the Sprint A detector (import `escalation/detector.py` from the feature branch checkout) applied to each harness's transcript
- cap: 25 rounds or 15 minutes per task, whichever first → recorded as abandonment

Run each task 2x per harness (local model variance is real); report both runs, not an average that hides flakiness — flaky results in this context are signal.

## Report

Timestamped markdown in the scratch dir (mirrors the audit-output convention): per-tier success table, malformed-call rates, token economics, and a final section answering exactly three questions:

1. Does openhands-sdk's loop beat the Prometheus loop on T2/T3 when tool calls land? (scaffolding-maturity question)
2. What fraction of openhands failures are malformed-call deaths vs genuine reasoning failures? (GBNF-value question)
3. Ranked list: which specific openhands behaviors (recovery prompts, condensation, editor semantics, retry policy) visibly drove its wins — these become the port priority for SPRINT-coding-mode.

## Out of scope

- Any modification to the Prometheus repo.
- Any persistent integration of openhands-sdk.
- Tuning either harness mid-bake-off. Run as-is; tuning is a confound.
