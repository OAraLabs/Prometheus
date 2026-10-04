# Self-Improving Loop (SUNRISE) — detail

Moved out of [PROMETHEUS.md](../../PROMETHEUS.md) on 2026-10-03 so that file fits the
12,000-character cap on a project instruction file. The text below is the original
section, word for word. Its rules are also kept in PROMETHEUS.md, which is
authoritative: where this copy disagrees, this copy is stale.

## Self-Improving Loop (SUNRISE)

Closed loop wired during the SUNRISE sprint. Disabled-by-default keys live
in `config/prometheus.yaml.default` under `learning:` and `trajectory_export:`.

### Post-task hook chain (engine/agent_loop.py)
- `AgentLoop.add_post_task_hook(hook)` — append; multiple hooks fire in order.
- `AgentLoop.set_post_task_hook(hook)` — back-compat alias (replaces the list).
- A failing hook does not block subsequent hooks.

### Wired in scripts/daemon.py (in order)
1. **SkillCreator** — `add_post_task_hook(skill_creator.maybe_create)`
   creates SKILL.md from successful traces (>=3 tool calls).
2. **SkillRefiner** — `learning.skill_refinement_enabled` gates wiring of
   `add_post_task_hook(skill_refiner.maybe_refine_recent)`. Refines the most
   recently modified auto-skill against the trace; archives `.bak-{ts}.md`.
3. **PeriodicNudge** — `learning.nudge_enabled` + `nudge_interval` gates
   construction; passed to `AgentLoop(nudge=...)`. Injects a
   `[system-internal]` user message every N completed turns.
4. **MemoryExtractor** — `extractor.run_forever()` spawned as
   `asyncio.create_task(..., name="memory_extractor")`.
5. **GoldenTraceExporter** — `trajectory_export.enabled` gates an interval
   loop (default 24h) that calls `telemetry.export_golden_traces()` and
   emits `golden_traces_exported` on the SignalBus.
6. **GEPAEngine** — `learning.gepa_enabled` gates an idle-driven cycle that
   subscribes to `idle_start`/`idle_end` on the bus and runs at most once
   per `gepa_max_frequency_hours` after `gepa_min_idle_minutes` of idle.

### GEPA — Generalized Evolutionary Prompt Architecture
- `prometheus.learning.gepa.GEPAOptimizer` — reads JSONL traces from
  `~/.prometheus/trajectories/`, finds candidate auto-skills referenced by
  `Skill`-tool invocations, generates N variants, judges via
  `evals.judge.PrometheusJudge` (constrained-decode JSON output), promotes
  winners that beat both the current score and `gepa_judge_threshold`.
  Archives the prior version to `~/.prometheus/skills/auto/archive/`.
- `prometheus.sentinel.gepa_engine.GEPAEngine` — bus-subscribed wrapper.
  `run_now()` bypasses the idle gate (used by `/gepa run`).
- Operates ONLY on `~/.prometheus/skills/auto/`. Never touches manual skills.
- Input requirement, measured 2026-09-01: golden traces are cloud-only
  (`telemetry/tracker.py` `_CLOUD_PROVIDERS`) and only `Skill`-tool rows are
  candidates; below `gepa_min_traces_required` (10) a cycle is a no-op. The
  reference deployment has never promoted a skill. It does not consume the
  `evals/golden_dataset.py` tasks and does not measure task outcomes.

### CLI / Telegram surface
- `prometheus export-traces [--limit N --output PATH --tool NAME]`
  manually triggers a JSONL export.
- `/gepa status | run | history` — Telegram. `/gepa run` enqueues an
  approval via the existing `ApprovalQueue` (Trust Level 1).

### Tests
- `tests/test_gepa.py` (18 tests) — GEPAOptimizer unit coverage.
- `tests/test_wiring.py::TestSunrise*` (18 tests) — hook list, SkillRefiner
  gating, PeriodicNudge injection, GoldenTraceExporter, GEPAEngine bus
  subscription. Real instances; LLM calls stubbed.
