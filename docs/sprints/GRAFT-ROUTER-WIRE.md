# GRAFT: Router Wire-Up (v3, review-corrected)
## Codename: ROUTER-WIRE

**Date:** April 20, 2026 (v1); April 20, 2026 (v2 after survey pass); April 21, 2026 (v3 after review pass)
**Purpose:** Consolidate the two parallel routers into one canonical `router/model_router.py`, then wire the full routing stack end-to-end — `RetryAction.ESCALATE`, user-facing provider override commands (direct mode), and a config-driven subagent preset system (delegated mode) with `/spawn`. After this sprint, multi-model juggling works both as a UI toggle (you talk to Claude instead of Gemma) and as delegation (Gemma dispatches a specialist subagent that reports back).
**Estimated Time:** 6–8 hours across seven phases. v1 estimated 4–6 / five phases. v2 added two integration phases (1.5 and 3.5), taking it to 5–7. v3 adds ~45 min for Phase 2 (concrete import-site enumeration) and Phase 3.5 (LoopContext plumbing through `AgentLoop.run_async()`).
**Dependencies:**
- CLOUD-API-PROVIDERS (complete) — ProviderRegistry, OpenAICompatProvider, AnthropicProvider
- Sprint 3 (complete) — Model Adapter Layer
- Sprint 8 (complete) — SubagentSpawner
- Sprint 10 / GRAFT-ROUTER (partial) — `adapter/router.py` is live, `router/model_router.py` is dormant

**This is a CONSOLIDATION + WIRING sprint. Phase 2 deletes `adapter/router.py`. All other phases are additive.**

---

## v2 revision notes (what changed since v1)

A survey pass against the two live router files and every `AgentLoop` / `router.route()` call site across the codebase surfaced three structural issues with v1. All three are addressed in v2.

| # | v1 claim | Reality | Resolution |
|---|---|---|---|
| **A** | "Phase 2 preserves method signatures identically" | The two routers have different constructor arity, different `route()` return types (`ProviderConfig` strings vs `RouteDecision` with pre-instantiated objects), different `get_fallback()` parameter names, and read different config section keys (`model_router:` vs `router:`) | **Phase 2 rewritten** to enumerate every file and line that needs updating. Framed as a call-site rewrite, not a flip. |
| **B** | Phase 1 copies `TaskClassifier` into dormant router and task-type routing "just works" | Dormant router's `route()` does not call any classifier. Copying the class in is necessary but not sufficient — without integration, Phase 2 silently drops task-type routing. | **New Phase 1.5** wires `TaskClassifier` into dormant `route()` decision tree. |
| **C** | `set_override()` / `clear_override()` are per-session ready | `_override_config` at `router/model_router.py:129` is a **single global slot** on the router instance. One `/claude` floods every chat, every eval, every benchmark. | **New Phase 3.5** refactors to `dict[session_id, ProviderOverride]` before user-facing commands land. |

Minor corrections (all folded into the relevant phase sections):

- `providers/anthropic_provider.py` → `providers/anthropic.py` (actual filename)
- Phase 4 safety guarantee re-targeted: SENTINEL does **not** go through `ModelRouter` (grep confirmed); the real leak-prone sites are eval/benchmark/smoke-test runners and any cron-dispatched `LoopContext` construction
- Phase 2 test consolidation enumerates **all 11** existing call sites across `tests/test_router.py` (8), `tests/test_wiring.py` (2), `tests/test_model_router.py` (1)
- `_build_adapter_for` gets renamed to `build_adapter_for` in Phase 5 (decision: remove the leading underscore; it's used across package boundaries in the SubagentRunner, so the Python `_` convention is misleading)
- Phase 1 adds a small edit to `config/prometheus.yaml.default` so fresh installs inherit the `config/prometheus.yaml` denied_path protection that the local-only config already has

---

## v3 revision notes (what changed since v2)

A review pass against v2 — cross-checking every import site, `LoopContext` construction point, and `AgentLoop.run_async()` caller — surfaced three blockers (B1, B2, B3), four important-but-not-blocking items (I1, I2, I3, C1, C2), and two fresh-eyes observations. All addressed in v3. v3 is a detail-tightening pass, not a structural change — phase ordering and behavioral guarantees from v2 are preserved.

| # | v2 gap | v3 resolution |
|---|---|---|
| **B1** | Phase 2's "cross-codebase import sweep" was generic; missed that `adapter/__init__.py` explicitly re-exports `ModelRouter`, `TaskClassifier`, `TaskType`, `ProviderConfig` via `__all__` | Phase 2 gains an **"Import sites to update"** subsection enumerating all 5 concrete sites (adapter/__init__.py:27 + :30-47, engine/agent_loop.py:44, __main__.py:379, tests/test_wiring.py:555+568, tests/test_router.py:5) |
| **B2** | Phase 3.5 said "LoopContext gains a session_id field" but handwaved through the fact that `LoopContext` is constructed inside `AgentLoop.run_async()` at agent_loop.py:927, plus two more direct-construction sites | Phase 3.5's **"Files to modify"** rewritten to enumerate the 3 LoopContext construction sites (agent_loop.py:927, daemon.py:613, __main__.py:869) and spell out the `AgentLoop.run_async(session_id=...)` parameter addition |
| **B3** | Phase 4 listed system-session AgentLoop callers (eval/benchmark/smoke) but missed the user-facing `run_async()` callers in the gateways | Phase 4 gains a **"User-facing run_async() callers"** table: telegram.py:648+836 pass `str(chat_id)`, slack.py:400+492 pass `str(channel_id)` |
| **I1** | Phase 1.5 test stub had `RouteReason.PRIMARY  # or whatever we name this branch` as a literal TODO | Phase 1.5 explicitly adds `TASK_RULE = "task_rule"` to the `RouteReason` enum; test asserts `RouteReason.TASK_RULE` concretely |
| **I2** | Smart-routing-before-task-rules ordering was defensible but undocumented | Phase 1.5 now has a **"Design note: ordering rationale"** subsection explaining why cost optimization (smart routing at #3) wins over capability matching (task rules at #4) |
| **I3** | Phase 2's migration story for existing `model_router:` config keys was "optional but recommended" — silent degradation risk | Phase 2 commits to: on daemon startup, if `config["model_router"]` key is present, log a **WARNING** and run with primary-only routing. No auto-migration. No refusal to start. User migrates manually. |
| **C1** | "Cron audit during Phase 4 prep" was an open bullet | Resolved: grep confirmed zero `AgentLoop`/`LoopContext`/`run_async` references in `gateway/cron_scheduler.py` or `gateway/cron_service.py`. Cron dispatches shell commands only. Phase 4 updated to reflect the confirmation. |
| **C2** | Phase 1 didn't acknowledge that COPY (not move) creates a Phase 1→2 window where two TaskClassifier classes coexist | Phase 1 adds a one-line acknowledgment that the dual-class window is intentional for the zero-behavior-change guarantee |
| **FE1** | Unclear whether Phase 3's escalation uses Phase 5's new SubagentRunner (would create a reverse dependency) | Phase 3 now states explicitly that it uses the existing `SubagentSpawner` from Sprint 8, not Phase 5's `SubagentRunner`. Phase 3 does not depend on Phase 5. |
| **FE2** | Phase 5's `_build_adapter_for` → `build_adapter_for` rename could break out-of-tree consumers | Phase 5 adds a one-release compatibility alias: `_build_adapter_for = build_adapter_for` |

---

## Current State (Pre-Sprint)

Two routers exist in parallel:

| Router | Location | Status | Responsibilities |
|--------|----------|--------|------------------|
| Adapter router | `src/prometheus/adapter/router.py` | **Live** | `TaskClassifier` + rule-based routing (5 TaskTypes) + fallback (tool-format errors only) |
| Model router | `src/prometheus/router/model_router.py` | **Dormant** | Smart routing (binary simple/complex) + `/claude`-style presets (global scope) + `_build_adapter_for()` per-provider adapter adjustment + escalation scaffolding |

The dormant router has the novel piece — `_build_adapter_for()` — which flips formatter, strictness, GBNF, and context limit based on provider. The adapter router does not. This sprint promotes the dormant router to canonical, absorbs and integrates the task classifier, and deletes the adapter router.

Additional dead wiring:
- `RetryAction.ESCALATE` exists in `src/prometheus/adapter/retry.py:74-86` but `RetryEngine` never receives the router at construction (`src/prometheus/adapter/__init__.py:96`)
- `/model` Telegram command only reports the active provider — no switching
- Dormant router's preset system (`/claude`, `/gpt`, `/gemini`, `/xai`) is never instantiated, and even if it were, the override slot is global

---

## Design Sources — The Blend

| Source | What It Contributes | How It Appears in Prometheus |
|--------|---------------------|------------------------------|
| **Hermes** `smart_model_routing` | Task classification scoring | `TaskClassifier` — moved from `adapter/router.py` into the new router and *integrated* into its decision tree (Phase 1 + 1.5) |
| **OpenClaw** degraded mode | Fallback chain with graceful degradation | `_fallback_chain()` — ordered provider list with availability caching |
| **Claude Code** s04 coordinator | Isolated-context subagent with curated toolset | `SubagentSpawner` + `/spawn` preset system |
| **Prometheus** (novel) | Adapter auto-adjustment per provider | `build_adapter_for()` — the glue that makes Gemma↔Claude switching not blow up the adapter layer |

---

## The Two Modes (Canonical Terminology)

Claude Code executing this sprint must keep these distinct at every phase boundary.

| | **Direct Mode** (Phase 4) | **Delegated Mode** (Phase 5) |
|---|---|---|
| **Command shape** | `/claude`, `/gpt`, `/gemini`, `/xai`, `/grok`, `/local`, `/route` | `/spawn`, `/spawn <preset> <prompt>` |
| **Who the user is talking to in Telegram** | The override provider directly | Still the primary (Gemma) |
| **Context** | Shared with main thread | Isolated subagent context |
| **Main thread pollution** | Every turn is in the main context | Only the final result lands back |
| **Tools** | Full main-agent toolset | Curated per-preset whitelist |
| **Persistence** | Sticky per-chat until `/local` (**per-session, Phase 3.5**) | One-shot, dies on completion |
| **Memory writes** | Normal (flows into Oara Memory Extractor) | Isolated — transcript saved to disk, not fed to main memory |

**The two modes must not share a command namespace.** Provider names (`/claude`, `/gpt`, etc.) are reserved for direct mode. Preset names under `/spawn` are reserved for delegated mode. If a preset is ever named the same as a provider, `/spawn <name>` rejects with a configuration error at daemon startup, not at invocation time.

---

## Read These Files First

```
# Routers (consolidation target):
src/prometheus/adapter/router.py           # LIVE — TaskClassifier + rule routing + fallback
src/prometheus/router/model_router.py      # DORMANT — preset system + _build_adapter_for() + smart routing
src/prometheus/router/__init__.py          # May be empty/minimal

# Adapter layer:
src/prometheus/adapter/__init__.py         # RetryEngine construction at L96 (router must thread in)
src/prometheus/adapter/retry.py            # RetryAction.ESCALATE at L74-86
src/prometheus/adapter/formatter.py        # QwenFormatter, GemmaFormatter, PassthroughFormatter, AnthropicFormatter
src/prometheus/adapter/validator.py        # Strictness levels

# Provider layer (already wired):
src/prometheus/providers/base.py
src/prometheus/providers/registry.py
src/prometheus/providers/openai_compat.py
src/prometheus/providers/anthropic.py       # (NOT anthropic_provider.py — v1 spec had the filename wrong)
src/prometheus/providers/llama_cpp.py

# Agent loop + subagent:
src/prometheus/engine/agent_loop.py        # L173-185 (route call), L278-294 (fallback trigger)
src/prometheus/coordinator/subagent.py     # SubagentSpawner — isolated context execution (L117)

# Gateway:
src/prometheus/gateway/telegram.py         # L404-416 (/model command — will be extended/replaced)
src/prometheus/gateway/slack.py            # Secondary — also constructs LoopContext

# Config and daemon wiring:
config/prometheus.yaml                     # Routing rules live here; SecurityGate denied_paths includes this
config/prometheus.yaml.default             # Fresh-install defaults — needs denied_paths update in Phase 1
scripts/daemon.py                          # L38 (create_model_router import), L216 (factory call), L283-296 (AgentLoop wiring)
src/prometheus/__main__.py                 # create_model_router factory at L377-380

# System-invocation sites (affected by Phase 4 safety guarantee):
scripts/run_nightly_evals.py               # L95 — AgentLoop construction (system session)
src/prometheus/benchmarks/runner.py        # L151 — AgentLoop construction (system session)
scripts/smoke_test_tool_calling.py         # L593 — AgentLoop construction (system session)
src/prometheus/gateway/cron_scheduler.py   # Audit during Phase 4 prep — may dispatch agent tasks

# Tests:
tests/test_wiring.py                       # Mandatory wiring tests — every phase adds to this; 2 existing ModelRouter(config) sites
tests/test_router.py                       # 8 existing ModelRouter(config) sites (adapter router constructor)
tests/test_model_router.py                 # 1 existing ModelRouter(config, primary, adapter) site (dormant router)
tests/test_engine.py                       # AgentLoop test sites — regression guard for Phase 2

# Security reference:
# config/prometheus.yaml is in SecurityGate denied_paths (line 53 of the local config).
# Phase 1 adds the same protection to prometheus.yaml.default so fresh installs inherit it.
# Nothing in this sprint should give the agent a path to edit its own config.
```

---

## Architecture (Post-Sprint)

```
User message arrives via Telegram
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Telegram gateway                                                │
│                                                                  │
│   Is it a direct-mode command? (/claude, /gpt, etc.)            │
│     Yes → router.set_override(session_id=chat_id, provider=X)   │
│           → reply "Switched to X"                               │
│     No  → continue                                              │
│                                                                  │
│   Is it a delegated-mode command? (/spawn ...)                  │
│     Yes → dispatch to SubagentRunner with preset                │
│           → wait for result → reply with framed result          │
│     No  → continue                                              │
│                                                                  │
│   Otherwise → dispatch to main AgentLoop                        │
│              (LoopContext gets session_id=chat_id)              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │   ModelRouter     │
                    │                  │
                    │  1. Per-session  │  ← direct-mode override
                    │     override?    │    (Phase 3.5 + 4)
                    │                  │
                    │  2. Classify     │  ← Task classifier
                    │     task_type    │    (Phase 1 + 1.5)
                    │                  │
                    │  3. Rule match?  │  ← 5 TaskType rules
                    │                  │    (Phase 1.5)
                    │                  │
                    │  4. Smart-route  │  ← Hermes pattern
                    │     fallback     │    (binary simple/complex)
                    │                  │
                    │  5. Fallback     │  ← OpenClaw pattern
                    │     chain        │    (availability cache)
                    │                  │
                    │  6. Build        │  ← NOVEL
                    │     adapter      │    (per-provider config)
                    └─────────┬────────┘
                              │
                              ▼
                         AgentLoop

Delegated path (separate; Phase 5):
/spawn research <prompt>
         │
         ▼
┌─────────────────────────────────────┐
│   SubagentRunner                    │
│                                     │
│   1. Resolve preset from config     │
│   2. Build provider (preset's own)  │
│   3. Build adapter via              │
│      router.build_adapter_for()     │
│   4. Build curated tool whitelist   │
│   5. Spawn SubagentSpawner with     │
│      isolated context + preset's    │
│      system_prompt_extension        │
│      (session_id=None so parent     │
│      overrides don't leak in)       │
│   6. Run to completion (max_turns,  │
│      timeout caps)                  │
│   7. Persist transcript to disk     │
│   8. Return final result to caller  │
└─────────────────────────────────────┘
```

---

## Phase 1 — Consolidation (No Behavior Change)

**Goal:** Move `TaskClassifier` + `TaskType` + `TaskClassification` from `adapter/router.py` into `router/model_router.py`. Both files remain, but the new router now has everything it needs to replace the old one in Phase 2. Also add `config/prometheus.yaml` to the default denied_paths so fresh installs are protected.

**Files to modify:**
- `src/prometheus/router/model_router.py` — add `TaskClassifier`, `TaskType`, `TaskClassification` classes (copy from `adapter/router.py:24-160`). Keep existing `_build_adapter_for()`, `OVERRIDE_PRESETS`, `ModelRouter`, `RouterConfig` untouched — this phase is additive only.
- `src/prometheus/router/__init__.py` — export `ModelRouter`, `TaskClassifier`, `TaskType`, `TaskClassification`, `ProviderConfig`-equivalent (consider adding a new `RouteDecision` export if not already present), `RouteReason`, `load_router_config`.
- `config/prometheus.yaml.default` — add `config/prometheus.yaml` to the `denied_paths:` list around L54-56. Keeps fresh installs consistent with the author's local config.

**Files NOT to modify (yet):**
- `src/prometheus/adapter/router.py` — still live, still imported, still doing the routing
- `src/prometheus/engine/agent_loop.py` — unchanged
- `scripts/daemon.py` — unchanged
- `src/prometheus/__main__.py` — unchanged

**Wiring tests (add to `tests/test_wiring.py`):**
```python
def test_model_router_has_task_classifier_class():
    from prometheus.router import TaskClassifier, TaskType
    tc = TaskClassifier()
    result = tc.classify("write a python function to parse json")
    assert result.task_type == TaskType.CODE_GENERATION

def test_model_router_still_instantiable_with_primary_provider():
    # Regression guard — the existing dormant constructor still works
    router = build_model_router_from_config(test_config, primary_provider=stub_provider, primary_adapter=stub_adapter)
    assert router is not None
```

**Acceptance:**
```bash
uv run pytest tests/ -v --tb=short
# All existing tests pass (no behavior change)
# Two new wiring tests pass
# Fresh daemon still uses adapter/router.py for routing decisions
```

**Regression: what must not break.** Every existing routing decision must happen exactly the same way. The daemon still uses `adapter/router.py`. The new router has TaskClassifier available but isn't using it yet (Phase 1.5's job). This phase is pure class relocation + one small default-config safety improvement.

**Intentional duplicate-class window (C2):** Between Phase 1 completion and Phase 2 completion, **two copies** of `TaskClassifier`, `TaskType`, and `TaskClassification` exist in the codebase — one in `adapter/router.py` (the live path), one in `router/model_router.py` (the new home). `adapter/__init__.py` still re-exports the adapter-router copy, so existing code that does `from prometheus.adapter import TaskClassifier` keeps working throughout the window. The new path `from prometheus.router import TaskClassifier` gets the new copy. Phase 2 deletes the adapter-router copy and redirects imports. This duplication is intentional — it's what makes Phase 1's zero-behavior-change guarantee work.

---

## Phase 1.5 — Integrate TaskClassifier into dormant router's route() (NEW in v2)

**Goal:** The dormant `ModelRouter.route()` currently has five priority levels: override, retry escalation, smart simple routing, primary. Add a task-type rule-matching level that uses the classifier. Without this, Phase 2 silently downgrades behavior from 5-way rule-based routing to binary simple/complex.

**Files to modify:**
- `src/prometheus/router/model_router.py`:
  - Add `TaskClassifier` instance to `ModelRouter.__init__` (lazy-constructed or eager — eager is simpler)
  - Add `task_rules: list[RoutingRule]` field to `RouterConfig`, parsed from `config["router"]["rules"]` (same schema as adapter router's rules: `task_type`, `provider`, `model`, optional `base_url`, optional `min_confidence`)
  - **Add `TASK_RULE = "task_rule"` to the `RouteReason` enum** (I1). Existing values: PRIMARY, USER_OVERRIDE, SMART_SIMPLE, SMART_COMPLEX, ESCALATION, FALLBACK, QUEUE, AUXILIARY. The new branch gets its own reason so logs and the `/route` command can distinguish "routed to Claude because task_type=CODE_GENERATION" from "routed to Claude because primary."
  - Add `_route_by_task_type(message)` method that classifies the message and searches `task_rules` for a matching rule; returns `RouteDecision | None` with `reason=RouteReason.TASK_RULE`
  - In `route()`, insert the task-type branch between smart routing (#3) and primary (#4):
    ```
    1. Override (per-session in Phase 3.5)
    2. Retry escalation
    3. Smart routing (simple → cheap)
    4. Task-type rules ← NEW in Phase 1.5, reason=TASK_RULE
    5. Primary
    ```
  - Task-type provider instances cached on the router (same lazy pattern as `_escalation_provider`, `_simple_provider`)
- `src/prometheus/router/__init__.py` — export `RoutingRule` if not already exported; ensure `RouteReason` export includes the new `TASK_RULE` value

**Design note: ordering rationale (I2)**

Smart routing runs at position #3, task-type rules at #4. This prioritizes **cost optimization over capability matching**. Example:

- User sends `"what is json?"` (short, ambiguous, classifies weakly as QUICK_ANSWER)
- **Smart routing fires first:** message is under 160 chars with no complex indicators → routed to `simple_provider` (e.g., a cheap local model)
- **Task-type rule never fires**, even if config has `QUICK_ANSWER → gpt-4o-mini`

The reverse ordering (task rules first, smart as fallback) would prioritize capability matching — "every QUICK_ANSWER goes to gpt-4o-mini regardless of length." We pick cost-first because:

1. Smart routing is opt-in via `smart_routing.enabled = true`; users who don't want this behavior simply leave it off and task rules become the primary decision layer
2. Short/simple messages by definition don't benefit much from capability matching (the task *is* trivial)
3. Cost surprises from bypassed smart routing are worse UX than occasional capability mismatches on trivial queries

If a user wants capability-first for a specific task type, they can set `smart_routing.enabled = false` globally or add a long-message indicator to the task rule's trigger conditions (a future extension, not v1).

**Config schema (read but not yet authoritative — daemon still uses adapter router):**
```yaml
router:
  rules:
    - task_type: code_generation
      provider: anthropic
      model: claude-sonnet-4-6
      min_confidence: 0.4
    - task_type: tool_heavy
      provider: llama_cpp
      model: ""          # auto-detect from server
      base_url: http://100.110.140.39:8080
    - task_type: reasoning
      provider: anthropic
      model: claude-opus-4-7
      min_confidence: 0.5
```

**Files NOT to modify (yet):**
- `scripts/daemon.py` — unchanged
- `src/prometheus/engine/agent_loop.py` — unchanged
- `src/prometheus/adapter/router.py` — still live

**Wiring tests (add to `tests/test_wiring.py`):**
```python
def test_dormant_router_classifies_message():
    router = build_dormant_router_with_rules([
        ("code_generation", "anthropic", "claude-sonnet-4-6"),
    ])
    decision = router.route("write a python function to parse json")
    assert decision.provider_name == "anthropic"
    assert decision.reason == RouteReason.TASK_RULE

def test_task_rule_yields_correct_route_reason():
    # I1 regression guard: ensure TASK_RULE is distinct from PRIMARY/SMART_SIMPLE/etc.
    # So /route and logs can distinguish "routed because rule matched" from "routed to default primary"
    router = build_dormant_router_with_rules([("reasoning", "anthropic", "claude-opus-4-7")])
    decision = router.route("explain the tradeoffs of immutable data structures in depth")
    assert decision.reason == RouteReason.TASK_RULE
    assert decision.reason != RouteReason.PRIMARY

def test_dormant_router_falls_through_to_primary_on_no_rule_match():
    router = build_dormant_router_with_rules([])  # no rules
    decision = router.route("hi")
    assert decision.reason == RouteReason.PRIMARY

def test_dormant_router_respects_min_confidence():
    # Ambiguous message ("what about code?") classifies weakly;
    # rule with min_confidence=0.9 should NOT match → falls through to primary
```

**Acceptance:**
```bash
uv run pytest tests/ -v --tb=short
# All existing tests pass
# Three new task-type routing tests pass
# Dormant router now has feature parity with adapter router (plus the extras: override, escalation, smart routing, fallback, _build_adapter_for)
# Daemon is STILL using adapter/router.py — no production behavior change
```

**Regression: what must not break.**
- Every existing routing decision in production is identical (adapter router still live)
- Dormant router instantiation without a `rules:` config key still works (empty rule list → falls through to primary as before)
- Existing smart routing (`_classify_complexity`) is preserved and still runs for messages that don't hit a task-type rule

**Why 1.5 and not 2:** This is pure dormant-router work. No production call sites change. Zero risk. Merging it independently gives us a clean rollback point if Phase 2 turns out harder than expected.

---

## Phase 2 — Flip the Switch (call-site rewrite, not a drop-in swap)

**Goal:** Daemon instantiates `router/model_router.py` instead of `adapter/router.py`. `AgentLoop` consumes the new router's richer `RouteDecision` return type. Delete `adapter/router.py`. **This is the risky commit — gets its own PR with full regression run.**

**⚠️ v2 correction:** v1 claimed "method signatures preserve identically." They do not. The two routers have incompatible interfaces. The agent loop call sites must be rewritten, not merely rebound.

### Interface delta table

| | Adapter router (current) | Dormant router (target) |
|---|---|---|
| **Constructor** | `ModelRouter(config: dict)` — 1 arg | `ModelRouter(config: RouterConfig, primary_provider, primary_adapter, primary_model="local")` — 4 args, reversed dependency order |
| **Config key** | `config["model_router"]` | `config["router"]` |
| **`route()` signature** | `route(message, tool_mentions=None, force_provider=None)` | `route(message, context: dict \| None = None)` |
| **`route()` return** | `ProviderConfig` — string fields (`provider`, `model`, `base_url`, `reason`) | `RouteDecision` — instantiated objects (`provider: ModelProvider`, `adapter: ModelAdapter`, `reason: RouteReason`, `use_subagent: bool`, `model_name: str`, `provider_name: str`, `cost_warning: str \| None`) |
| **`get_fallback()` signature** | `get_fallback(failed_provider: str)` → `ProviderConfig \| None` | `get_fallback(failed_provider_name: str = "")` → `RouteDecision \| None` |

### Files to modify — every line range that needs touching

**`scripts/daemon.py`** (reorder startup):
- L38 — change `from prometheus.__main__ import create_model_router` if factory relocates; may stay as-is if `__main__.py`'s factory is updated instead
- L216 — factory call `create_model_router(config)` now needs `(config, primary_provider, primary_adapter, primary_model)` — **build primary provider and adapter first**, then pass in. Reorder daemon startup lines accordingly.
- L283-296 — `AgentLoop(model_router=model_router, ...)` — keyword name unchanged; the object it holds has a different return-type contract, but this line's syntax is identical.

**`src/prometheus/__main__.py`** (factory):
- L377-380 — `create_model_router(config)` updated to accept + forward primary provider + adapter, OR inline the construction into `daemon.py` and delete the factory. Pick one and document in PR description.
- L848 — `create_model_router(config)` call site — same update as L377.

**`src/prometheus/engine/agent_loop.py`** (the call-site rewrite):
- L173-185 — the main route call:
  - Current: `route = context.model_router.route(first_user)` then logs `route.provider, route.model, route.reason`
  - New: `decision = context.model_router.route(first_user, context=loop_meta)` then reads `decision.provider_name`, `decision.model_name`, `decision.reason.value`. The `decision.provider` and `decision.adapter` are now **pre-instantiated objects** — swap `context.provider = decision.provider` and `context.adapter = decision.adapter` directly, no need to call `ProviderRegistry.create()`.
- L278-294 — the fallback path:
  - Current: `_try_model_fallback()` wraps `router.get_fallback()`, returns a tuple `(fallback_provider, fallback_model)`
  - New: `_try_model_fallback()` returns a `RouteDecision | None`; unpacks via `decision.provider`, `decision.model_name`. Update tuple unpacking at L281-286.
  - The `context.adapter` re-format block at L288-292 is also affected — the new decision carries its own adapter, so prefer `context.adapter = decision.adapter` over rebuilding.

**`src/prometheus/adapter/__init__.py`**:
- Remove any `from prometheus.adapter.router import ...` re-exports (if present)
- Any direct calls to `adapter.router.ModelRouter` — switch to `prometheus.router.ModelRouter`

**`src/prometheus/adapter/router.py`** — **DELETE** after all imports are moved.

### Import sites to update (B1)

All 5 concrete sites that must be updated from `prometheus.adapter.router` → `prometheus.router` **before** `adapter/router.py` is deleted:

| # | File:line | Current | After Phase 2 |
|---|---|---|---|
| 1 | `src/prometheus/adapter/__init__.py:27` | `from prometheus.adapter.router import ModelRouter, TaskClassifier, TaskType, ProviderConfig` | Delete this line. The adapter package no longer re-exports router types. |
| 2 | `src/prometheus/adapter/__init__.py:30-47` | `__all__` list includes `"ModelRouter"`, `"TaskClassifier"`, `"TaskType"`, `"ProviderConfig"` | Remove all four entries from `__all__` |
| 3 | `src/prometheus/engine/agent_loop.py:44` | `if TYPE_CHECKING: from prometheus.adapter.router import ModelRouter, ProviderConfig` | `if TYPE_CHECKING: from prometheus.router import ModelRouter, RouteDecision` (note: `ProviderConfig` type hint becomes `RouteDecision`) |
| 4 | `src/prometheus/__main__.py:379` | `from prometheus.adapter.router import ModelRouter` (inside `create_model_router()` factory) | `from prometheus.router import ModelRouter` |
| 5a | `tests/test_wiring.py:555` | `from prometheus.adapter.router import ModelRouter` | `from prometheus.router import ModelRouter` + constructor call-site update (already tracked in test consolidation table) |
| 5b | `tests/test_wiring.py:568` | same | same |
| 5c | `tests/test_router.py:5` | `from prometheus.adapter.router import TaskClassifier, TaskType, ModelRouter` | `from prometheus.router import TaskClassifier, TaskType, ModelRouter` |

**Downstream consumers:** any code doing `from prometheus.adapter import ModelRouter` / `TaskClassifier` / `TaskType` / `ProviderConfig` also breaks when the `adapter/__init__.py` `__all__` entries are removed. Grep-and-replace: `prometheus.adapter` → `prometheus.router` for all four symbols. `ProviderConfig` specifically becomes `RouteDecision` — not a rename, a type change.

**Order of operations inside Phase 2's single PR:**
1. Update all 5 import sites (+ downstream `from prometheus.adapter import ...` consumers) to import from `prometheus.router`
2. Update `adapter/__init__.py:27` and `:30-47` to stop re-exporting
3. Update call sites in `engine/agent_loop.py` (L173-185, L278-294) to consume `RouteDecision`
4. Update `scripts/daemon.py` L216 + startup order
5. **Only then** delete `src/prometheus/adapter/router.py`
6. Run full test suite — any residual import error is a site step #1 missed

### Migration story for existing `model_router:` config keys (I3)

The adapter router reads `config["model_router"]`; the new router reads `config["router"]`. Users with an existing `model_router:` section in their local `config/prometheus.yaml` would silently lose their routing rules without a warning. Pick the non-silent option:

**On daemon startup, if `config["model_router"]` key is present:**
1. Log a **WARNING** via `prometheus.daemon` logger: `"model_router: config key is deprecated (renamed to router: in GRAFT-ROUTER-WIRE v3). Your existing rules are not being applied. Migrate the block manually: 'model_router:' → 'router:' and rename 'rules:' schema if it differs. Running with primary-only routing until migrated."`
2. Run the daemon with **primary-only routing** (no rules consulted)
3. Do **NOT** auto-migrate the file (would require writing to a gitignored user-owned file)
4. Do **NOT** refuse to start (would break users on upgrade)
5. The warning fires once per daemon startup, not per routing decision

This gives users an explicit signal to act without forcing a breaking change on upgrade.

**`config/prometheus.yaml.default`**:
- Update the example config block at L119+ (the old `model_router:` schema) to demonstrate the new `router:` schema with task rules, fallback chain, smart routing, and commented-out escalation.
- This does not break existing local configs (absence of `router:` falls through to primary-only, same as before) but guides new users toward the canonical schema.

### Test consolidation — all 11 existing call sites

| File | Count | Action |
|---|---|---|
| `tests/test_router.py` | 8 | Each `ModelRouter(config)` → update to new 4-arg constructor with stub primary provider + adapter. Consolidate any adapter-router-specific tests (TaskClassifier unit tests) into `test_model_router.py`. |
| `tests/test_wiring.py` | 2 | Same update — new constructor signature |
| `tests/test_model_router.py` | 1 | Already uses new-style constructor — verify and enhance |
| **Total** | **11** | |

After consolidation, `tests/test_router.py` either:
- (a) is deleted (its tests absorbed into `test_model_router.py`), OR
- (b) remains as a thin alias file that imports from `test_model_router.py` (discouraged — pick (a))

### Wiring tests (add to `tests/test_wiring.py`)
```python
def test_daemon_builds_primary_provider_before_router():
    # Regression guard on the startup order inversion
    # ModelRouter construction receives primary_provider and primary_adapter as real objects

def test_agent_loop_consumes_route_decision_not_provider_config():
    # Agent loop's model_router.route() returns an object with .provider (instance), .adapter (instance), .model_name
    # Agent loop swaps context.provider and context.adapter from the decision, not from strings

def test_fallback_returns_route_decision_not_tuple():
    # _try_model_fallback(context) returns a RouteDecision or None
    # Main loop's unpacking path updated
```

**Acceptance:**
```bash
# Full test suite
uv run pytest tests/ -v --tb=short

# Daemon starts cleanly
python -m prometheus daemon --debug

# Manual verification on mini:
# 1. Send Telegram message → logs show RouteDecision with provider_name=llama_cpp, reason=primary
# 2. Force a tool-format error (e.g., prompt-injected bad JSON tool call) → fallback triggers, swaps provider live
# 3. SENTINEL, AutoDream, cron jobs still run (they bypass the router, see Phase 4 safety section)
# 4. Config drift guard still blocks agent edits to config/prometheus.yaml
```

**Regression: what must not break.**
- Every Telegram message still gets a response (routing decisions are behaviorally equivalent for the default config)
- Tool-format fallback still kicks in on the same trigger conditions
- No raised exceptions from the call-site rewrite
- Provider and adapter lifecycle: the new router *owns* its primary provider and primary adapter instances. Confirm no double-free, no garbage-collection surprise when swapping `context.provider` from primary to fallback and back.

**Backup discipline:**
```bash
cp src/prometheus/adapter/router.py /tmp/router.py.pre-wire.backup
# Keep this until Phase 5 lands and the full sprint is proven in production for a week
```

**Why this is the risky PR:** Not because of line count — because of behavioral-semantic change even though the outward routing decisions remain equivalent. The agent loop is now receiving pre-built provider + adapter instances rather than strings. Any subtle assumption in the loop about "the router hands me strings and I call ProviderRegistry.create()" is now wrong. Review the agent_loop.py diff twice.

---

## Phase 3 — Wire ESCALATE

**Goal:** `RetryAction.ESCALATE` becomes live. When tool retries exceed max and the router has an escalation provider configured, the agent loop spawns a subagent with the escalation provider and returns its result as a tool_result to the main loop.

**Dependency note (FE1):** Phase 3's escalation path uses the existing **`SubagentSpawner` from Sprint 8** (in `src/prometheus/coordinator/subagent.py`), **NOT** Phase 5's new `SubagentRunner`. Phase 3 does not depend on Phase 5. `SubagentSpawner` already knows how to run an isolated agent loop with a given provider + adapter + tool set; that's exactly what escalation needs. `SubagentRunner` (Phase 5) is a higher-level orchestration layer on top of `SubagentSpawner` with preset resolution and transcript persistence — escalation doesn't need those features.

**Files to modify:**
- `src/prometheus/adapter/__init__.py` — line 96: `RetryEngine(...)` constructor now accepts and stores `router=...`
- `src/prometheus/adapter/retry.py` — `RetryEngine.decide_action()` checks `self.router` at the max-retries boundary. If router has escalation configured for this task type, return `RetryAction.ESCALATE`. Otherwise, fall through to existing behavior.
- `src/prometheus/engine/agent_loop.py` — handle `RetryAction.ESCALATE`:
  - Look up escalation provider from router (`router.config.escalation_enabled`, `router.config.escalation_provider`)
  - Use `SubagentSpawner` (from `coordinator/subagent.py`) to run the failing operation with the escalation provider
  - Adapter for escalation subagent built via `router.build_adapter_for(escalation_provider_name)` (renamed from `_build_adapter_for` in Phase 5 — if Phase 5 hasn't landed yet when this goes in, continue using `_build_adapter_for` with explicit underscore import and update in Phase 5)
  - Result comes back as a `tool_result` message in the main loop
  - **Do not swap the main agent's provider.** Escalation is delegated, not a takeover.
- `config/prometheus.yaml` — add `router.escalation` section (commented by default):
  ```yaml
  router:
    escalation:
      enabled: false          # off by default; user opts in
      provider: anthropic
      model: claude-sonnet-4-6
      trigger_after_retries: 3      # matches current hardcoded threshold at model_router.py:150
  ```

**Wiring tests (add to `tests/test_wiring.py`):**
```python
def test_retry_engine_receives_router():
    # Construct RetryEngine with a real router. .router attribute is non-None.

def test_escalate_action_spawns_subagent():
    # Force max retries exhausted on a tool call
    # Router has escalation configured
    # Agent loop spawns subagent, receives result as tool_result
    # Main loop's provider/model unchanged after escalation

def test_escalate_disabled_falls_through():
    # router.escalation.enabled = false
    # Max retries exhausted → existing behavior (error up, no escalation)

def test_escalate_budget_gate():
    # If router.escalation.budget_usd is exceeded, escalation refuses and falls through
    # Guards against runaway cloud cost from repeated escalations
```

**Acceptance:**
```bash
uv run pytest tests/ -v --tb=short

# Manual verification:
# 1. Enable escalation in config with anthropic provider + key in env
# 2. Force a tool-calling failure that exhausts retries (can use a tool with a bogus schema mismatch)
# 3. Confirm in logs: "ESCALATE triggered, spawning subagent with anthropic/claude-sonnet-4-6"
# 4. Result returns to main loop as tool_result
# 5. Main agent's provider unchanged (confirm via /route after escalation)
```

**Regression: what must not break.**
- Escalation disabled (the default) → existing retry behavior is byte-identical
- Router instantiation without escalation config still works
- Subagent spawn failure doesn't crash the main loop (graceful fallback to existing error path)

---

## Phase 3.5 — Per-session override refactor (NEW in v2)

**Goal:** Refactor `_override_config` from a single global slot to a per-session dict. This is the behavioral prerequisite for Phase 4's Telegram commands. Without this, `/claude` in one chat instantly applies to every other chat, every eval, every benchmark, every cron-dispatched task.

**⚠️ v2 correction:** v1 assumed the existing dormant scaffolding was per-session ready. It isn't. See survey Finding C.

**Files to modify (B2):**

### Router side

- `src/prometheus/router/model_router.py`:
  - Line 129: replace `self._override_config: dict | None = None` with `self._overrides: dict[str, ProviderOverride] = {}`. Define `ProviderOverride` as a small dataclass:
    ```python
    @dataclass
    class ProviderOverride:
        provider: Any              # ModelProvider instance (lazy-built)
        adapter: Any               # ModelAdapter instance (lazy-built)
        provider_config: dict      # source config for diagnostics
    ```
  - Rename fields `self._override_provider` and `self._override_adapter` — these become per-session, stored inside each `ProviderOverride`
  - `set_override(session_id: str, provider_config: dict)` — stores under `self._overrides[session_id]`, lazy-builds provider + adapter on first access
  - `clear_override(session_id: str)` — `del self._overrides[session_id]` (or pop with default)
  - `get_override_for_session(session_id: str | None) -> ProviderOverride | None` — returns None for `session_id in (None, "system")` (reserved system session)
  - `has_override` property — reflect any override for any session, used by diagnostic / status commands
  - `_route_override()` → `_route_override(session_id: str)` — looks up by session, builds on first access (same lazy pattern as before), returns `RouteDecision`
  - `route()` — accept `session_id` from `context` dict (`context.get("session_id")`); the first priority check now uses `get_override_for_session(session_id)`

### Agent loop plumbing

- `src/prometheus/engine/agent_loop.py`:
  - Add `session_id: str | None = None` to `LoopContext` dataclass (new field)
  - **Add `session_id: str | None = None` parameter to `AgentLoop.run_async()` signature.** This is the key plumbing change — callers pass their session identifier in here, and it flows through to the LoopContext constructed at L927.
  - At the LoopContext construction at **L927**, forward `session_id=session_id` into the constructor
  - At L173-185 (the route call), pass `context={"session_id": context.session_id}` so the router's override lookup sees it
  - `_try_model_fallback(context)` also threads session_id for consistency (overrides are already decided at primary routing time, but API uniformity matters)

### Three LoopContext construction sites — enumerated

| # | Site | Purpose | session_id value after Phase 3.5 |
|---|---|---|---|
| 1 | `src/prometheus/engine/agent_loop.py:927` | **Primary production path** — every Telegram/Slack message, every `agent_loop.run_async()` call flows through here | Comes from `run_async(session_id=...)` parameter (new in Phase 3.5) |
| 2 | `scripts/daemon.py:613` | Web/Beacon bridge LoopContext (separate from Telegram; used by the web bridge server) | Hard-code `session_id="web"` — all web-bridge-initiated requests share this session identifier |
| 3 | `src/prometheus/__main__.py:869` | CLI REPL / one-shot mode (`prometheus --once "..."` or interactive REPL) | Hard-code `session_id="cli"` — distinct from `"web"` and `"system"` so a user setting `/claude` in CLI doesn't contaminate Telegram (and vice versa) |

**Reserved session IDs:**
- `None` — "don't apply overrides, ever" (unit tests, ad-hoc constructions)
- `"system"` — evals, benchmarks, smoke tests, any cron-dispatched agent path if one emerges (Phase 4 enumerates the system call sites)
- `"web"` — web/Beacon bridge (Phase 3.5 wires this at daemon.py:613)
- `"cli"` — command-line REPL / one-shot (Phase 3.5 wires this at __main__.py:869)
- `str(chat_id)` for Telegram, `str(channel_id)` for Slack — set by Phase 4's user-facing callers table

Phase 3.5 itself does NOT modify the gateway `run_async()` callers (Telegram, Slack). Those are the subject of Phase 4's B3 table. After Phase 3.5, the gateway callers invoke `run_async()` without `session_id` → defaults to `None` → no overrides apply → existing behavior preserved (zero-behavior-change guarantee).

**Reserved session IDs:**
- `None` — "don't apply overrides, ever"
- `"system"` — same semantic as None, but explicit in logs; preferred for eval/benchmark/cron paths

**Wiring tests (add to `tests/test_wiring.py`):**
```python
def test_override_sets_and_clears_per_session():
    router.set_override("chat_123", {"provider": "anthropic", "model": "claude-sonnet-4-6"})
    assert router.get_override_for_session("chat_123").provider_config["provider"] == "anthropic"
    router.clear_override("chat_123")
    assert router.get_override_for_session("chat_123") is None

def test_override_is_isolated_across_sessions():
    router.set_override("chat_123", {"provider": "anthropic"})
    assert router.get_override_for_session("chat_456") is None

def test_system_session_never_inherits_override():
    router.set_override("chat_123", {"provider": "anthropic"})
    router.set_override("chat_456", {"provider": "openai"})
    # Every system-invocation path uses session_id=None or "system"
    assert router.get_override_for_session(None) is None
    assert router.get_override_for_session("system") is None

def test_clearing_one_session_preserves_others():
    router.set_override("chat_a", {"provider": "anthropic"})
    router.set_override("chat_b", {"provider": "openai"})
    router.clear_override("chat_a")
    assert router.get_override_for_session("chat_a") is None
    assert router.get_override_for_session("chat_b") is not None

def test_route_consults_session_override_first():
    router.set_override("chat_123", {"provider": "anthropic", "model": "claude-sonnet-4-6"})
    decision = router.route("write a python function", context={"session_id": "chat_123"})
    assert decision.provider_name == "anthropic"
    assert decision.reason == RouteReason.USER_OVERRIDE

def test_loop_context_sites_have_expected_session_id():
    # B2 regression guard: each direct LoopContext construction site gets its reserved session_id
    # Note: these are spot checks that read the source — the actual wiring is validated at runtime by
    # tests that actually spin up the web bridge and CLI paths.

    # Site 1: agent_loop.py:927 — session_id comes from run_async() param, tested via run_async() callers
    loop = make_test_agent_loop()
    # default (no session_id arg) → None
    # (pseudo — wrap in a context-capturing shim)
    assert loop._last_context.session_id is None  # after a run with no session_id

    # Site 2: daemon.py:613 — session_id="web" hard-coded
    # Site 3: __main__.py:869 — session_id="cli" hard-coded
    # Both verified via source inspection in this test (import daemon module, read module source)
    import inspect
    from scripts import daemon
    daemon_src = inspect.getsource(daemon)
    assert 'session_id="web"' in daemon_src

    from prometheus import __main__ as main_mod
    main_src = inspect.getsource(main_mod)
    assert 'session_id="cli"' in main_src
```

**Acceptance:**
```bash
uv run pytest tests/ -v --tb=short

# Manual verification (requires Phase 4's commands to be useful, so this phase's
# manual test is mostly just: daemon still starts, no behavior change visible to
# users because no commands exist yet to SET overrides).
python -m prometheus daemon --debug
# Send Telegram message from two different chat IDs → both route to primary
# (no overrides set yet — Phase 4 will wire the UI)
```

**Regression: what must not break.**
- Dormant router instantiation without any overrides still works (empty dict is fine)
- Any code path that previously accessed `router._override_config` directly (there should be none outside the class, but audit) is updated
- `has_override` property still works as a diagnostic signal

**Why 3.5 and not inside 4:** This refactor is pure router-internal plumbing. It's the prerequisite for Phase 4's commands to behave correctly. Isolating it in its own PR makes the Phase 4 review focus on gateway handlers, not router state machines.

---

## Phase 4 — Direct Mode (user-facing provider overrides)

**Goal:** `/claude`, `/gpt`, `/gemini`, `/xai`, `/grok`, `/local`, `/route` as real Telegram handlers. Switching is sticky per-chat (see Phase 3.5). Main agent responds to user as the override provider until `/local` clears it. System paths (evals, benchmarks, smoke tests, cron-dispatched agent calls) never inherit overrides because they use `session_id="system"` or `session_id=None`.

**⚠️ v2 correction:** v1 targeted SENTINEL as the path that needs `session_id="system"`. SENTINEL doesn't use ModelRouter at all — it calls providers directly. Survey Finding D documents this. Real targets are the system-invocation AgentLoop call sites enumerated below.

**Files to modify:**
- `src/prometheus/gateway/telegram.py` — add command handlers:
  - `/claude [message]` — set override to `anthropic`; if `message` is present, dispatch immediately. Uses `OVERRIDE_PRESETS` from `router/model_router.py`.
  - `/gpt [message]` — set override to `openai`
  - `/gemini [message]` — set override to `gemini`
  - `/xai [message]` — set override to `xai`
  - `/grok [message]` — alias for `/xai` (same preset)
  - `/local [message]` — clear override, back to primary; if message present, dispatch immediately with primary
  - `/route` — show current session's effective provider (override if set, else primary) + list available override commands
  - **Remove or replace** the read-only `/model` at L404-416 — its behavior is absorbed by `/route`
- `src/prometheus/router/model_router.py` — already has `set_override` / `clear_override` / `get_override_for_session` from Phase 3.5; no changes needed here
- `src/prometheus/engine/agent_loop.py` — already consults `router.get_override_for_session(context.session_id)` from Phase 3.5

### User-facing `run_async()` callers (B3)

These are the gateway call sites where the user's session identifier must be threaded into `AgentLoop.run_async(session_id=...)`. Without these, Phase 4's `/claude` command has nowhere to look up the session override.

| # | File:line | Caller | session_id value |
|---|---|---|---|
| 1 | `src/prometheus/gateway/telegram.py:648` | Telegram `/benchmark` handler invocation | `session_id=str(chat_id)` |
| 2 | `src/prometheus/gateway/telegram.py:836` | Telegram main message dispatch — every user Telegram message routes through here | `session_id=str(chat_id)` |
| 3 | `src/prometheus/gateway/slack.py:400` | Slack message dispatch | `session_id=str(channel_id)` |
| 4 | `src/prometheus/gateway/slack.py:492` | Slack alternate dispatch path | `session_id=str(channel_id)` |

**Why string, not int:** Telegram `chat_id` is an int; Slack `channel_id` is a string. The router's `_overrides: dict[str, ProviderOverride]` is keyed by string. Forcing `str(...)` at the call site keeps the key space uniform and avoids `1234567` vs `"1234567"` mismatches.

**Group chat vs private chat:** Telegram `chat_id` is chat-level, not user-level. In a group chat, every member shares the same override. This is intentional — `/claude` in a group chat means "this conversation is now happening via Claude for everyone," not "each user has their own override." If per-user overrides in groups become a need later, key on `f"{chat_id}:{user_id}"` instead. Not in scope for this sprint.

### System-invocation sites that must use `session_id="system"`

| # | File:line | Purpose |
|---|---|---|
| 1 | `scripts/run_nightly_evals.py:95` | Eval runner — explicit `session_id="system"` |
| 2 | `src/prometheus/benchmarks/runner.py:151` | Benchmark runner — explicit `session_id="system"` |
| 3 | `scripts/smoke_test_tool_calling.py:593` | Smoke test — explicit `session_id="system"` |

**Cron (C1 — resolved):** Confirmed via grep on `src/prometheus/gateway/cron_scheduler.py` and `src/prometheus/gateway/cron_service.py` that neither module constructs `AgentLoop`, `LoopContext`, or calls `run_async()`. Cron dispatches shell commands only — no session plumbing needed. This bullet from v2's "audit during Phase 4 prep" is resolved and removed.

**Explicitly NOT in scope** (these were v1 targets that were misdirected):
- SENTINEL (`sentinel/autodream.py`, `sentinel/observer.py`, `sentinel/memory_consolidator.py`, `sentinel/knowledge_synth.py`, `sentinel/wiki_lint.py`, `sentinel/telemetry_digest.py`) — grep confirmed these bypass ModelRouter entirely. No session_id plumbing needed.
- Subagents (`coordinator/subagent.py:117`) — they construct their own isolated LoopContext with `session_id=None` as part of Phase 5 preset isolation. Explicitly not affected by parent session overrides.

**Config addition:**
```yaml
router:
  overrides:
    enabled: true          # direct-mode commands are active
    sticky: true           # overrides persist until /local (false = one-shot per message)
```

**Wiring tests (add to `tests/test_wiring.py`):**
```python
def test_telegram_claude_command_sets_session_override():
    # Simulate /claude via the gateway with chat_id="chat_123"
    # Router override for that session is now "anthropic"
    # Next message from chat_123 routed through anthropic

def test_telegram_local_clears_session_override():
    # After /claude in chat_123, then /local
    # Next message from chat_123 routed through primary
    # chat_456's override (if any) unaffected

def test_eval_runner_constructs_with_system_session():
    # scripts/run_nightly_evals.py creates LoopContext with session_id="system"
    # Global /claude override from a user chat does NOT affect the eval

def test_benchmark_runner_constructs_with_system_session():
    # Same guarantee for benchmarks

def test_smoke_test_constructs_with_system_session():
    # Same guarantee for smoke tests

def test_route_command_reports_effective_provider():
    # /route on a chat with override → "anthropic/claude-sonnet-4-6 (override)"
    # /route on a chat without override → "llama_cpp/gemma (primary)"
```

**Acceptance:**
```bash
uv run pytest tests/ -v --tb=short

# Manual on Mini:
# Send: "hi" → uses Gemma (primary)
# Send: /claude → bot replies "Switched to Claude (anthropic/claude-sonnet-4-6)"
# Send: "what is 2+2?" → bot replies as Claude (check logs for provider=anthropic)
# Send: /route → shows override: anthropic/claude-sonnet-4-6
# Send: /local → "Back to primary (gemma-3-27b)"
# Send: "what is 2+2?" → bot replies as Gemma
# Trigger SENTINEL → confirm logs show provider=llama_cpp (primary) — SENTINEL bypasses router anyway
# Run eval nightly → confirm logs show provider=llama_cpp (primary), not override
```

**Regression: what must not break.**
- Users not using any override command get exactly the current behavior
- `router.overrides.enabled=false` in config → commands are no-ops (log warning)
- System-session paths (evals, benchmarks, smoke tests) never pick up any user's override
- SENTINEL and AutoDream are unaffected because they don't use the router (same as before)

---

## Phase 5 — Delegated Mode (subagent preset system)

**Goal:** `/spawn <preset> <prompt>` dispatches a subagent with the preset's provider, curated toolset, and system prompt extension. Subagent runs isolated, main agent reports back with framed result. `/spawn` with no args lists configured presets.

**Files to modify:**

**New file:** `src/prometheus/router/subagent_runner.py` (~200 lines)
```python
"""
SubagentRunner — orchestrates delegated-mode subagent spawning.

Reads presets from config, resolves provider/adapter/tools per preset,
invokes SubagentSpawner with isolated context, persists transcript,
returns final result framed for parent agent consumption.
"""

@dataclass
class SubagentPreset:
    name: str
    provider: str                    # Provider name (openai, anthropic, xai, gemini, ...)
    model: str | None                # Specific model; defaults to provider default
    tools_allowed: list[str]         # Tool whitelist (default-deny)
    system_prompt_extension: str     # Layered on top of base SOUL
    max_turns: int = 20              # Safety cap
    timeout_seconds: int = 600       # Wall-clock cap
    result_format_hint: str = "freeform"  # "freeform" | "structured"
    passthrough: bool = False        # If true, parent returns result verbatim.
                                     # If false, parent adds brief framing commentary.

class SubagentRunner:
    def __init__(self, router: ModelRouter, spawner: SubagentSpawner,
                 presets: dict[str, SubagentPreset], transcript_dir: Path):
        ...

    def list_presets(self) -> list[SubagentPreset]:
        """For /spawn with no args."""

    async def run(self, preset_name: str, prompt: str,
                  parent_session_id: str) -> SubagentResult:
        """Resolve preset, build provider+adapter+tools, spawn, persist, return.

        Subagent LoopContext is constructed with session_id=None so parent
        session overrides (Phase 3.5 + 4) do NOT leak into the subagent.
        """
```

**Modify:** `src/prometheus/router/model_router.py`
- **Rename** `_build_adapter_for` → `build_adapter_for` (remove leading underscore). This is a package-boundary function used by `SubagentRunner`; the `_` convention was misleading.
- Update all call sites (internal `_route_auxiliary`, `_route_simple`, `_route_escalation`, and any others) to the new name.
- **Compatibility alias (FE2):** after the rename, add `_build_adapter_for = build_adapter_for` as a one-release alias. Any out-of-tree consumer (forks, downstream projects embedding Prometheus, or internal tooling that hasn't been audited) that does `from prometheus.router.model_router import _build_adapter_for` keeps working. Delete the alias in a future release once the ecosystem has caught up.

**Modify:** `src/prometheus/gateway/telegram.py`
- `/spawn` (no args) — list presets with brief description table
- `/spawn <preset> <prompt>` — dispatch via `SubagentRunner.run()`; while running, send a status message ("Spawning `research` subagent via xAI..."); on completion, send the framed result
- **Error handling:**
  - Unknown preset → list available presets
  - Preset name collides with a direct-mode provider name → startup validation catches this (see below); at runtime this case cannot occur
  - Subagent timeout / max_turns → parent reports failure with transcript path for forensics

**Modify:** `scripts/daemon.py`
- Load subagent presets from config at startup
- **Startup validation:** for each preset, assert `preset.name not in {"claude", "gpt", "gemini", "xai", "grok", "local", "route", "model"}`. Fail hard with a clear error if collision detected. Validation happens once at daemon startup, not at invocation time.
- Construct `SubagentRunner` with router, spawner, presets, and transcript directory
- Pass `SubagentRunner` into the Telegram gateway

**Config addition:**
```yaml
router:
  subagents:
    transcript_dir: "~/.prometheus/subagent_runs"

    presets:
      research:
        provider: xai
        model: grok-4.20
        tools_allowed:
          - web_search
          - web_fetch
          - file_read
          - grep
          - glob
        system_prompt_extension: |
          You are a research subagent. Your job is to investigate the user's
          question using web search and file reading tools, and report back
          with a concise, factual summary. Do not modify files.
        max_turns: 20
        timeout_seconds: 600
        result_format_hint: freeform
        passthrough: false

      video:
        provider: gemini
        model: gemini-3-flash-preview
        tools_allowed:
          - file_read
          - file_write
          - web_fetch
        system_prompt_extension: |
          You are a video/multimodal generation subagent.
        max_turns: 15
        timeout_seconds: 900
        passthrough: false

      deepthink:
        provider: anthropic
        model: claude-opus-4-7
        tools_allowed:
          - file_read
          - grep
          - glob
          - web_search
        system_prompt_extension: |
          You are a deep-reasoning subagent. Take your time. Show your work.
        max_turns: 30
        timeout_seconds: 900
        passthrough: true   # Return Opus output verbatim

      code-review:
        provider: openai
        model: gpt-5
        tools_allowed:
          - file_read
          - grep
          - glob
        system_prompt_extension: |
          You are a code review subagent. Read-only filesystem access.
          Produce findings with file:line references.
        max_turns: 15
        timeout_seconds: 600
        passthrough: false
```

**Transcript persistence:**
```
~/.prometheus/subagent_runs/
  2026-04-20T14:30:22Z-research/
    preset.json          # Preset used
    prompt.txt           # Original user prompt
    messages.jsonl       # Full message history
    result.txt           # Final result returned to parent
    metadata.json        # provider, model, tokens, cost, turns, duration
```

**Composition rules (phase-5 tests enforce these):**
1. Subagents cannot spawn other subagents (enforce by omitting `/spawn` equivalent from the subagent's tool access — there is no spawn tool, only a user-facing command)
2. Direct-mode overrides on the parent do not affect subagents (subagent uses its preset's provider unconditionally; LoopContext has `session_id=None`)
3. A subagent's cost attributes to the preset's provider, not to the parent's current override
4. Subagents run against the same `SubagentSpawner` from Sprint 8 — do not fork a second spawning system

**Wiring tests (add to `tests/test_wiring.py`):**
```python
def test_subagent_presets_load_from_config():
    runner = build_subagent_runner_from_config(test_config)
    assert "research" in [p.name for p in runner.list_presets()]

def test_preset_name_collision_fails_startup():
    config_with_collision = {"router": {"subagents": {"presets": {"claude": {...}}}}}
    with pytest.raises(ConfigValidationError):
        build_subagent_runner_from_config(config_with_collision)

def test_spawn_research_runs_in_isolated_context():
    # Run /spawn research "test prompt"
    # Subagent's message list does NOT include parent's conversation
    # Subagent uses xai provider, not primary

def test_spawn_returns_result_to_parent_session():
    result = await runner.run("research", "what is X?", parent_session_id="chat_123")
    assert result.final_text is not None
    # Parent session's provider is unchanged after subagent completes

def test_subagent_transcript_persisted():
    await runner.run("research", "test", "chat_123")
    # A directory exists under transcript_dir
    # Contains preset.json, prompt.txt, messages.jsonl, result.txt, metadata.json

def test_subagent_respects_tool_whitelist():
    # Research preset forbids bash
    # Subagent with malicious prompt "run rm -rf" should have no bash tool available
    # Spawner refuses the tool call

def test_spawn_list_reads_from_config_not_hardcoded():
    # Add a custom preset to config
    # /spawn (no args) output includes it

def test_direct_override_does_not_leak_into_subagent():
    router.set_override("chat_123", {"provider": "anthropic"})
    result = await runner.run("research", "test", "chat_123")
    # Subagent used xai (preset provider), not anthropic (override)
    # Because subagent LoopContext has session_id=None

def test_build_adapter_for_is_public_api():
    # Import from prometheus.router.model_router (no underscore)
    from prometheus.router.model_router import build_adapter_for
    adapter = build_adapter_for("anthropic")
    assert adapter is not None
```

**Acceptance:**
```bash
uv run pytest tests/ -v --tb=short

# Manual on Mini:
# Send: /spawn → bot lists available presets in a table
# Send: /spawn research "current state of MLX vs llama.cpp on M4 Pro"
#   → bot replies "Spawning research subagent (xAI / grok-4.20)..."
#   → some time later, bot replies with framed result
# Send: /route → still shows primary (subagent didn't change override)
# Check filesystem: ~/.prometheus/subagent_runs/<timestamp>-research/ has all five files
# Check logs: subagent had access only to research preset's tool whitelist
# Send: /spawn claude "hi" → bot rejects with "claude is not a preset. Did you mean /claude ?"
```

**Regression: what must not break.**
- Users not using `/spawn` get exactly the current behavior
- No subagent preset config → `/spawn` reports "no presets configured"
- Direct-mode commands continue to work identically to Phase 4
- Primary agent's memory extractor still runs on parent-thread messages; subagent messages do not pollute the wiki
- `build_adapter_for` rename: any code still calling `_build_adapter_for` is either updated or aliased (`_build_adapter_for = build_adapter_for` for one-release compatibility)

---

## Phase Ordering and PR Strategy

| Phase | Behavior change | Risk | Depends on | Can merge independently? |
|-------|-----------------|------|------------|--------------------------|
| 1 — Consolidation | None | Low | — | Yes |
| 1.5 — Classifier integration | None (dormant) | Low | 1 | Yes |
| 2 — Flip the Switch | None (routing equivalence preserved) | **High** | 1.5 | Yes — but merge alone, observe in production for 24h before Phase 3 |
| 3 — Wire ESCALATE | Escalation opt-in | Medium | 2 | Yes |
| 3.5 — Per-session override refactor | None (no commands yet) | Low-Medium | 3 | Yes — but must land before Phase 4 |
| 4 — Direct Mode | New commands | Low | 3.5 | Yes |
| 5 — Delegated Mode | New commands | Low | 4 | Yes |

**Strong recommendation:** land Phase 2 as its own PR. Observe production for at least a day. Then land Phases 3, 3.5, 4, 5 either individually or in two batches — they're all additive from that point on.

**Dependency chain:**
```
1 ─► 1.5 ─► 2 ─► 3 ─► 3.5 ─► 4 ─► 5
                           │
                           └── 3.5 is the behavioral prerequisite for 4;
                               do not attempt 4 before 3.5 is merged
```

---

## Summary

| Component | Net lines (est.) | Complexity |
|-----------|------------------|-----------|
| Phase 1 — move TaskClassifier + .default denied_paths | +160 / −0 | Low |
| Phase 1.5 — integrate classifier + `TASK_RULE` enum + ordering note | +130 / −0 | Low |
| Phase 2 — flip switch + call-site rewrite + 5-site import update + migration warning + delete old | +100 / −500 | **High (risk, not volume)** |
| Phase 3 — ESCALATE wiring (uses existing SubagentSpawner) | +80 / −5 | Medium |
| Phase 3.5 — per-session override refactor + `run_async(session_id=...)` plumbing + 3 LoopContext sites | +135 / −30 | Low-Medium |
| Phase 4 — direct mode commands + 4 user-facing callers + 3 system-session sites | +185 / −40 | Low |
| Phase 5 — subagent preset system + build_adapter_for rename + compat alias | +400 / −0 | Medium |
| Tests (across all phases, including new B2/B3 site regression tests) | +420 | — |
| **Total net** | **~+1110** | **Medium overall** |

Compared to v1 (+570 net): v2 added ~+300 from the two new integration phases (1.5 and 3.5) plus ~+180 from the more complete test coverage across the test-consolidation set. v3 adds another ~+60 on top of v2 from the concrete B1/B2/B3 enumerations — each blocker resolution surfaces small but real work (import updates, `run_async()` parameter plumbing, migration warning logic) that v2 handwaved. The extra scope is real work that v1/v2 silently assumed was free; surfacing it is the point of the survey + review revisions.

After this sprint:
- **Direct mode:** `/claude`, `/gpt`, `/gemini`, `/xai`, `/grok`, `/local`, `/route` — you talk to a different provider in the same thread, per-session (Phase 3.5)
- **Delegated mode:** `/spawn research <prompt>`, `/spawn video`, `/spawn deepthink`, `/spawn code-review` — main agent dispatches a specialist, result comes back as a message
- **Both composable:** you can `/spawn research` to get facts via Grok, then `/claude` to think through them with Opus, all in one Telegram thread, without overrides leaking between them
- **Escalation wired:** `RetryAction.ESCALATE` stops being dead code — hard tool-calling failures spawn a cloud-provider subagent as a rescue path
- **Consolidated:** one router, not two
- **Per-provider adapter auto-adjustment preserved:** `build_adapter_for()` is the glue. Switching Gemma → Claude flips formatter, strictness, GBNF, and context limit automatically.
- **Fresh-install safety parity:** `prometheus.yaml.default` now lists `config/prometheus.yaml` in denied_paths, so new users inherit the same agent-can't-edit-its-own-config protection the author has locally.

---

## What NOT to Build in This Sprint

| Temptation | Why not |
|-----------|---------|
| Subagent-to-subagent spawning | Out of scope. Bounded blast radius is a feature. |
| Cost caps per subagent run | `CostTracker` logs it; hard caps are a separate sprint. |
| Parent-initiated subagent cancellation / pause | Spawn-and-wait is the v1 contract. Interactive control is a later sprint. |
| ML-driven provider auto-routing | Rule-based is fine for v1. Collect telemetry first. |
| Beacon dashboard subagent live view | Data model lands here; Beacon integration is a follow-up. |
| Per-chat persistent memory of which preset the user likes | Can add later via userMemories pattern; not core to wiring. |
| Ad-hoc `/spawn-raw <provider>` (no preset) | Footgun. Add only after we have real usage data showing the presets aren't enough. |
| Route SENTINEL through the router | Out of scope. SENTINEL's direct-provider-access pattern is fine; routing it would add latency and complexity for no immediate gain. Revisit only if SENTINEL starts needing provider switching. |
