# SPRINT-TEACHER-ESCALATION — Checkpoint 1 (Phase 0 survey)

**Verdict: ALL SPEC ASSUMPTIONS CONFIRMED — proceeding to Phase 1/2 per the session checkpoint protocol.**
Two non-blocking notes (§ Divergences) where the spec anticipated its own fallback.

**Branch:** `feat/teacher-escalation` @ `c59fe34985568e618cce247ac85589a556c455ee` (fresh off `main` == `origin/main`, fetched this session)
**Tree at Phase 0:** zero tracked modifications; pre-existing untracked owner artifacts only (CSVs, scratch scripts, audit/spec docs) — treated as clean per the same convention the 2026-06-10 sprint recorded ("the spirit of Hard Rule 1").
**Suite baseline on main:** 2467 passed, 27.61s (`python3 -m pytest`).
**Gating audit for Phase 3:** run this cycle — `audits/20260611T050051Z-middle-layer-audit.md`.

All paths `src/prometheus/`-relative unless noted.

## Survey findings (spec items 1–7)

### 1. Skills system — CONFIRMED, better than assumed
- Storage: builtin `skills/builtin/*.md` (flat .md); user `~/.prometheus/skills/*.md` **plus `~/.prometheus/skills/auto/*.md` for machine-generated skills** (`skills/loader.py:14-18,41-67` — sources tagged `"builtin"|"user"|"auto"`).
- Format: markdown with YAML frontmatter `name:` + `description:`, fallback first-heading/first-paragraph (`skills/loader.py:81-130`).
- Prompt injection: `context/prompt_assembler.py:108-150` — section "5. Available skills" (name+description) + "9. Loaded skill content"; registry loaded at startup (`__main__.py:211-212,528-531`; daemon holds it via ToolSearchTool, `daemon.py:467-471`).
- **Programmatic write path exists:** `learning/skill_creator.py` (SkillCreator) — writes to `skills/auto/`, validates via frontmatter-`name:` extraction with NO fallback (missing name = hard skip + `silent_failures` row, `:222-241`), slugify confinement `[a-z0-9-]` ≤64 chars (`:67-79`), no-overwrite policy (`:246-247`), emits `skill_created` ActivitySignal (`:261-285`), model calls through LLMCallEnvelope (`:127-131,362-375`). Hot reload without daemon restart: `SkillRegistry.reload_user_skills()` (`skills/registry.py:30-66`).

### 2. Cloud override path — CONFIRMED
- `/claude` → `_cmd_claude` (`gateway/telegram.py:343,1491-1495`) → `_apply_override` (`:1384`) → `resolve_slash_command_target("claude", config)` merging `slash_commands.claude` config over `OVERRIDE_PRESETS` (`router/model_router.py:269-355`) → `router.set_override(session_id, provider_config)` (`:499`). **Session-sticky**, cleared by `/local` (`telegram.py:1521-1558`).
- Single-turn-to-different-provider without touching the session's primary: not a router feature, but the established repo pattern for one-off subsystem calls is **direct provider construction + envelope** (SkillCreator/MemoryExtractor pattern): `ProviderRegistry.create({provider, model, api_key_env})` (`providers/registry.py:87-156`) + `LLMCallEnvelope.call(...)`. The teacher call will use exactly this — no router involvement, session primary untouched. Spec's "existing provider layer, wrapped in LLMCallEnvelope, no new HTTP client" — satisfied.

### 3. Golden traces in telemetry.db — CONFIRMED
- `tool_calls` table: `raw_model_output`, `parsed_tool_call`, `is_golden` (computed: cloud provider + success + zero retries, `telemetry/tracker.py:49-64,264-271,297`; `_CLOUD_PROVIDERS` `:30-37`); reader `get_golden_traces` (`:820`).
- Generic durable event store: **`signal_events`** (`:124-140`) — `signal_type`, JSON `payload`, `source_subsystem`; written via `record_signal_event` (`:476`); feeds `/events` + Beacon. **Plan:** escalation golden traces land as `signal_events` rows `signal_type="teacher_escalation"`, `source_subsystem="teacher_escalation"`, payload = full exchange + detector reasons + skill-persistence outcome. (Rationale: the per-tool-call `tool_calls` table is the wrong shape for a multi-message exchange; `signal_events` is the repo's existing typed-JSON trace surface. The envelope additionally writes `subsystem_runs` liveness rows for every teacher call for free.)

### 4. Trust tagging through LCM — CONFIRMED
- `lcm_engine.ingest*(provenance: str = "user", is_trusted: bool = True)` (`memory/lcm_engine.py:187-215,228-257`).
- `ConversationMessage` carries `provenance`/`is_trusted`; `session.add_result_messages` persists them per-message (`engine/session.py:255-256`); `add_user_message(provenance=, is_trusted=)` (`:145-177`).
- Existing conventions: user turns `("user", True)`; managed-task re-engagement `("task_supervisor", False)` (`gateway/telegram.py:1804-1810`). **Teacher-injected corrective replies will use `("teacher_escalation", False)`** — the conservative machine-injected convention.

### 5. Tool result surface post-turn — CONFIRMED (the expected anchor)
- `gateway/telegram.py:1744-1798` `_run_agent_turn_locked` (shared by user turns AND `inject_turn`, under the M6 per-session lock): after `agent_loop.run_async(...)` → `result: RunResult` (`engine/agent_loop.py:64-70`: `.text`, `.messages`, `.usage`, `.turns`), `pre_len` marks the turn boundary, **this turn's new messages = `result.messages[pre_len:]`**, and the honest-async-promise validator is invoked right there (`:1786-1797`, `engine/honesty.py:86-120` `evaluate_and_record`). The detector hook extends this exact point. Note: the honesty validator is invoked **only by the Telegram gateway** today (no web/slack call sites) — escalation will match that scope and record the parity gap as a PR follow-up.

### 6. LLMCallEnvelope — CONFIRMED
- `learning/llm_envelope.py:104` — importable; modes `raise|log_only|return_none`; on failure writes `silent_failures` + failed `subsystem_runs`; on success writes `subsystem_runs` (`:249-305`). Used by skill_creator/refiner/curator/extractor. The teacher call uses `on_failure="return_none"` (failure already loud in telemetry; caller falls back per spec).

### 7. Endpoint classification — EXISTS (spec's fallback unnecessary)
- `ProviderRegistry.is_cloud(provider_name)` (`providers/registry.py:158-161`): cloud = openai-compat set ∪ {anthropic}; local = `llama_cpp|ollama|stub`. Mirror constant `_CLOUD_PROVIDERS` in `telemetry/tracker.py:30-37` (kept in sync deliberately). Trigger condition 2 will use `not ProviderRegistry.is_cloud(<provider that served the turn>)`, where the serving provider = session override's provider if set (`router.get_override_for_session`) else the daemon primary (`telegram.py:116,125` `self.model_provider`). Config-driven; no hostnames involved.

## Real-failure grounding for detector patterns
From `FINDINGS-TOOLCALLING-2026-06-10.md` + the loop source:
- The dominant captured failure is the collapse arc ending in the loop's own terminal reply: **"Circuit breaker tripped: … The model cannot produce valid tool calls for this request."** (`engine/agent_loop.py:902-903,1062-1063`) — a deterministic, high-precision Tier-1 signal worth a dedicated pattern.
- "(no output)" tool results are usually *informative* (grep no-match, exit-1) — a documented false-positive trap for the unrecovered-error signal; covered by a negative fixture.
- "I was unable to find…" is an honest negative search outcome — capability-denial patterns must be verb-constrained (unable to *access/execute/run/perform*), not bare "I am unable to".

## Divergences from spec (none blocking)
1. **"extend the existing traces command"** — no `/traces` command exists in any gateway (grepped). The spec's own alternative applies: Phase 3 adds `/escalations`. Not a contradiction; the spec anticipated it.
2. **Detector signature** — spec's `detect_failure(tool_results: list, final_reply: str)` is kept verbatim, but the repetition signal ("same tool called with identical args ≥3×") requires call *arguments*, so `tool_results` items are dicts `{tool_name, arguments, result, is_error}` (the existing SkillCreator trace shape, `learning/skill_creator.py:377-386`, plus `is_error`). The Phase-3 hook builds these from the turn's `ToolUseBlock`/`ToolResultBlock` pairs. Documented here per "adjust to repo conventions".
3. (Environment note, not spec) Session instruction says uv is not installed; `uv` exists at `~/.local/bin/uv` but per instruction all test runs use `python3 -m pytest`. The standing-rules identity `dev/dev@prometheus.local` replaced the repo-local `OAraLabs` identity for this session's commits.

## Phase 1–2 design committed to (summary)
- `src/prometheus/escalation/detector.py` — **zero prometheus imports** (stdlib only: `re`, `dataclasses`, `json`) so BAKEOFF-harness.md can import it from the branch checkout standalone. `FailureVerdict(failed, reasons, matched_patterns)`. Module-level pattern table, one comment per pattern naming the real failure it catches. Signals: unrecovered tool error, capability denial (verb-constrained), clarification stall (< 80 chars, ends "?", no tool calls), repetition (≥3 identical name+args), empty reply after tool activity, breaker terminal reply.
- `src/prometheus/escalation/teacher.py` — `TeacherEscalation.from_config()` returns `None` when `escalation.teacher_model` unset (feature inert). Trigger order: agent-mode → local-primary (`is_cloud` false) → teacher-configured → detector-failed → budget (`escalation.max_per_session`, default 3); each block logged. Teacher call: `ProviderRegistry.create` + `LLMCallEnvelope(subsystem="teacher_escalation", on_failure="return_none")`. Two fenced sections parsed deterministically (```CORRECTIVE_REPLY / ```SKILL_DRAFT labels); missing section = loud failure + fall through to local reply + visible note. Skill gate: same detector against CORRECTIVE_REPLY; persistence through a `persist_skill_content()` method **extracted from SkillCreator's existing write path** (validation + slug confinement + no-overwrite + signal, reused not duplicated). Golden trace: `signal_events` row per escalation (fired/refused/teacher-failed all recorded). Decision: if the teacher itself fails the detector, the whole escalation is treated as failed — local reply + note delivered, nothing persisted (spec leaves delivery unspecified for this case; chosen to avoid replacing one failure with another).
- Config keys: `escalation.teacher_model` (gate, default unset), `escalation.teacher_provider` (default `anthropic`), `escalation.api_key_env` (default per provider), `escalation.max_per_session` (default 3). No live-config edits this sprint; a commented example only in PR description.
