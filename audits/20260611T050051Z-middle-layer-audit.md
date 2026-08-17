# Middle-Layer Spec-vs-Implementation Audit — cycle 2026-06-11

**Audited commit:** `main @ c59fe34985568e618cce247ac85589a556c455ee` (== `origin/main` after `git fetch origin`; checked out clean, working tree carried only pre-existing untracked files, zero tracked modifications)
**Mode:** read-only — no code modified by this audit.
**Suite at this SHA:** `python3 -m pytest` → **2467 passed, 7 warnings, 27.61s** (prior audit baseline was 2,386; +81 tests came with the post-audit fix commits).
**Purpose:** this is the spec-vs-implementation audit of `SPRINT-TOOL-CALLING-MIDDLE-LAYER.md` required *this cycle* as the Phase-3 gate of `docs/sprints/SPRINT-TEACHER-ESCALATION.md` (and as a future gate input of `docs/sprints/SPRINT-CODING-MODE.md`).
**Output convention:** untracked timestamped markdown, per the convention of `AUDIT-2026-06-8e5adf0.md` and `docs/audits/`.

---

## Finding 0 — the spec document does not exist (re-confirmed this cycle)

`SPRINT-TOOL-CALLING-MIDDLE-LAYER.md` is absent from the working tree and from **all git history** (re-verified this cycle: `find` across the tree, `git log --all --format= --name-only | sort -u | grep -i middle` → empty, content grep matches only documents that *reference* the name). The previous full repo audit reached the identical conclusion (`AUDIT-2026-06-8e5adf0.md:68` and `:427`: "never existed in this repository's history").

Consequence: spec-vs-implementation divergence can only be judged against the six-feature enumeration preserved in `AUDIT-2026-06-8e5adf0.md §4.4.6` (the only surviving normative source) plus the sprint's own closeout (`SPRINT-CLOSEOUT-TOOLCALLING-2026-06-10.md`, untracked). That is what this audit does. If a real spec file exists outside this machine, none of the per-feature requirement language below has been checked against it.

## What changed since the prior audit (main `8e5adf0` → `c59fe34`, 13 commits)

`8e5adf0` is an ancestor of `c59fe34` (verified). In order:

| Commit | Addressed |
|---|---|
| `e84ad40` | **C1** — LCM summarizer wraps prompt as TextBlock (compaction was silently inert) |
| `128307e` | **C2** — image_generate writes confined to `~/.prometheus`, honest `is_read_only` |
| `610c3b3` | **H4** — tool failures isolated in dispatch (no scramble, no turn-kill) |
| `20bf04f` | **H5** — per-tool retry budget resets on success (was per-daemon-lifetime) |
| `b81e46f` | **M1/M2/M4/M5** — per-tool timeout, repair telemetry, route-latest, honest health |
| `776ecbc` | **M6** — per-session turn serialization (re-engagement can't interleave) |
| `e1de1ee` | **H1/H2/H3** — per-result truncation wired, GBNF fixed+applied, hooks loader wired |
| `2b8ad08` | deferred-loading viability validator (Tier-3 explicitly **NOT** flipped) |
| `152c5fe` | stream fix — `<tool_call>` grammar markup no longer streamed to users |
| `ce333fb` | `PROMETHEUS_FILES_ROOT` (file-browser root; not middle-layer) |
| `5eceac7` | denominator honesty — policy denials out of success rates; completed M1 exclusion |
| `511ed2f` | **invariants-vs-policy split** + `malformed_empty` provider guard (the D1 incident fix) |
| `c59fe34` | FTS5 sanitizer (lcm_expand_query + memories store; not middle-layer proper) |

Note: the *running* daemon may predate some of these (memory/handoff notes say a restart was still pending after the last three). This audit assesses the code at the SHA, not the live process.

## Six-feature status at `c59fe34`

All paths below are `src/prometheus/`-relative.

### 1. Deferred loading / ToolSearchTool — **Implemented, validated, deliberately disabled**
- Mechanism: `context/dynamic_tools.py:88-91` reads `tools.deferred_loading`; `:109-156` gates prompt schemas to `always_loaded` when enabled; ToolSearchTool delivers schemas in-conversation; loop executes registered-but-unprompted tools via the lucky-guess path (`engine/agent_loop.py:1735-1750`, with `lucky_guess` telemetry `:1745`).
- Live config: `config/prometheus.yaml:107-118` — `enabled: false`, 8 `always_loaded` tools.
- Change since prior audit: `2b8ad08` added a viability validator and recorded the explicit decision **not** to flip the default (Tier-3). Status quo is now a documented decision rather than dormancy.
- Prior-audit dead-code caveats (`on_demand()` unused; keyword-mode `active_schemas(task_description)` unreachable from the live loop) — still present (`context/dynamic_tools.py:93-156`; L1 class). Unchanged.

### 2. Cross-result token budget — **Implemented, on**
- `engine/agent_loop.py:1074-1075` (trigger), `:1306-1359` (proportional truncation across results). Config `context.tool_results_turn_budget: 8000` (`config/prometheus.yaml:20`).
- **Improvement since prior audit:** H1 per-result truncation is now wired *upstream* of the budget — `engine/agent_loop.py:1938-1946` applies `ToolResultTruncator(context.tool_result_max)` (config `:18`, 4000) before injection, with the in-code note that error detail is captured untruncated for diagnostics first.
- Persisting caveat (M7 family): the truncation hint "[truncated — use lcm_expand or re-read for full content]" (`:1349`) is only half-fulfillable — `lcm_expand` per-result recall depends on `tool_use_id ↔ LCM` mapping that does not exist (see feature 3).

### 3. Microcompaction — **Implemented, on; in-run trigger scope persists**
- Call site `engine/agent_loop.py:744-745`; implementation `:1361-1428`.
- The trigger still requires the **in-run** iteration counter to reach `microcompact_after_turns` (`:1371`), so typical 1–3-iteration chat turns never microcompact — the prior audit's scope caveat stands. The fresh-window walk (`:1377-1385`) does count user messages across the whole history, so once a long run triggers, prior-run results are eligible.
- `is_ingested` is now *documented in-code* as returning False for every tool_use_id until a tool_use_id↔message_id mapping exists (`:1403-1415`) — the LCM-aware keep-length branch remains inert, but honestly so. Divergence acknowledged, not fixed.

### 4. Telemetry dashboard — **Substantially repaired; one mislabel persists**
- Fixed since prior audit: `repairs` column exists and is recorded (`telemetry/tracker.py:63,168,253,282-298` — M2); policy denials separated from failures with honest denominators (`:42`, `:718-763` — D3); synthetic `_loop_transition` rows excluded from health and per-tool stats (`:754`, `:953-958` — M1, completed by `5eceac7`).
- **Persisting divergence:** `telemetry/dashboard.py:57,71-80,174` — `adapter_repairs` still counts records where `retries > 0`, not the `repairs` column it now has. Both gateways render this mislabeled number (`gateway/telegram.py:753-754`, `gateway/slack.py:1118-1119`). Already on the closeout follow-up list (#2); confirmed still open at `c59fe34`.

### 5. Adaptive strictness — **Implemented, live**
- `adapter/__init__.py:81-111` (construction; tier interaction), `:150-166` (per-tool effective strictness applied at validate time), `:247-294` (recording + NONE→MEDIUM→STRICT bumps, light→full tier escalation note, manual override API). Live config `adapter.adaptive_strictness: true`, window 100, threshold 0.8 (`config/prometheus.yaml:121-123`).
- The invariants-vs-policy split (`511ed2f`) means structural invariants now run at **every** strictness (`adapter/validator.py:4-6` docstring; `validate()` `:162-205`) — the D1 dead-defense class is closed; the reachability audit it spawned is recorded in `docs/audits/RECURRING.md` §2.
- Persisting caveats (L11): strictness never decays (only `_bump_tool_strictness` exists; no decay path — re-verified `:262-283`); plain-valid calls record success at `:177` (`record_tool_call(tool_name, success=True)`). The prior audit's "repaired calls count as successes" reading is consistent with the unchanged code structure but was not re-traced line-by-line this cycle — **unverified this cycle**.

### 6. Structured errors + `example_call` — **Partial; M8 persists**
- `_build_structured_error` is wired into the validator's failure paths (`adapter/validator.py:24-46`, used at `:191,205,297,369`).
- **Persisting bug (M8):** the example shown comes from "the first tool that has example_call set" (`:42-44`) — not the failing tool. A `bash` failure can still show a `read_file` example.
- Coverage: **12 of 52** builtin tool modules define `example_call` (grep count, unchanged from prior audit).
- Gym evidence (branch `feat/tool-gym`, not main): s1 exp1 shows `example_call` worth +17pp on `task_create` — and the closeout records that live system-prompt injection of examples is **not wired** (gym variable only, flip gated on owner review). Spec-vs-implementation: improvement validated, deliberately unlanded.

### Adjacent (not in the six, but middle-layer-relevant): GBNF
H2 closed by `e1de1ee`: grammar is generated AND applied at the provider (`providers/llama_cpp.py:61` builds from `enforcer.generate_grammar(tool_schemas)`, `set_grammar` `:171`; `providers/ollama.py:64`), gated by `model.grammar_enforcement: true` (`config/prometheus.yaml:11`). The prior audit's "fix it or delete it" decision was resolved in the *fix* direction.

## Verdict

| Feature | Prior audit (8e5adf0) | This cycle (c59fe34) |
|---|---|---|
| Deferred loading | Implemented, disabled | Implemented, **validated**, deliberately disabled |
| Cross-result budget | Implemented, on | Implemented, on + H1 per-result truncation wired |
| Microcompaction | Implemented, on (in-run only) | Same; inert LCM branch now documented in-code |
| Telemetry dashboard | Partial | Mostly repaired; `dashboard.py` adapter_repairs mislabel persists |
| Adaptive strictness | Implemented, live | Same + invariants split (D1 closed); no decay (L11) persists |
| Structured errors/example_call | Partial | Partial; M8 + 12/52 coverage persist; gym-validated improvement unlanded |

**No feature regressed; six of six are present in the enumerated form; three carry known, documented divergences (dashboard mislabel, M8, L11/M7 caveats) — all already on the closeout follow-up list except none newly discovered.** New findings this cycle: **zero** beyond re-confirmation; the only audit-level finding remains Finding 0 (the spec document itself does not exist, so this enumeration-based audit is the strongest possible form of the check).

## Gate statement

For `SPRINT-TEACHER-ESCALATION.md` Phase 3 ("the spec-vs-implementation audit of SPRINT-TOOL-CALLING-MIDDLE-LAYER.md must have been run this cycle"): **this audit satisfies the gate as of 2026-06-11 at main `c59fe34`**, with the explicit caveat that it audits the surviving enumeration, not the (nonexistent) document. The agent-loop surface relevant to Phase 3 integration (post-turn region, validator paths, telemetry envelope) was read directly at this SHA in the course of the audit.
