# Prometheus

## Project Rules
- Python 3.11+, package managed with uv
- All imports use `from prometheus.` prefix
- Config lives at config/prometheus.yaml, loaded via prometheus.config
- Run tests: uv run pytest tests/ -v
- All donor code has provenance headers (Source, License, Modified)
- Do not modify files in reference/

## Key Paths
- Tools: src/prometheus/tools/builtin/
- Adapter: src/prometheus/adapter/
- Engine: src/prometheus/engine/agent_loop.py
- Providers: src/prometheus/providers/
- Memory/LCM: src/prometheus/memory/
- Gateway: src/prometheus/gateway/telegram.py
- Config: config/prometheus.yaml
- Skills: skills/

## Conventions
- New tools extend BaseTool in tools/base.py
- Security checks go through SecurityGate (permissions/)
- Tool results truncated by tool_result_max in config
- ADDITIVE ONLY: extend existing files, don't replace them

## Security
Shared security utilities live in `src/prometheus/security/`.

- `SecurityGate` (`permissions/checker.py`) — Trust-level evaluator wired
  into `AgentLoop` as `permission_checker`.
- `ExfiltrationDetector` (`permissions/exfiltration.py`) — bash-command
  pattern detector for credential-leak attempts.
- **`DangerousCodeScanner`** (`security/code_scanner.py`) — AST-based
  static-analysis pass on Python source. Flags `exec/eval/compile/__import__`
  and `os.system/popen/exec*/pty.spawn/ctypes.CDLL` at any scope, plus
  `subprocess/socket/httpx/requests/urllib` at module scope (suspicious,
  not blocking). Returns `ScanResult(verdict: clean|suspicious|dangerous)`.
  First introduced in GRAFT-SYMBIOTE for harvest-time gating; promoted to
  shared so any subsystem (hooks, audit pipelines, future eval gates) can
  reuse it. The old import path
  `prometheus.symbiote.code_scanner` still works via a re-export shim.

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

## SYMBIOTE — GitHub research → assimilation pipeline (GRAFT-SYMBIOTE Session A)

Closed Scout → Harvest → Graft loop wired during the GRAFT-SYMBIOTE sprint.
Disabled by default in `config/prometheus.yaml.default` under `symbiote:`.
MORPH (blue-green deploy) and BackupVault are Session B scope and are NOT
in this build.

### Package layout (`src/prometheus/symbiote/`)
- `license_gate.py` — `LicenseGate` / `LicenseCheck` / `LicenseVerdict`. Hard
  blocks GPL/AGPL/SSPL/BUSL and unknown licenses. Detects via GitHub API
  metadata, LICENSE/COPYING file content, or `SPDX-License-Identifier`
  header comments.
- `code_scanner.py` — backward-compat shim re-exporting
  `DangerousCodeScanner` from `prometheus.security.code_scanner`
  (promoted on 2026-04-25). New code should import from
  `prometheus.security.code_scanner` directly. See the **Security**
  section above for the scanner's behavior.
- `github_search.py` — `GitHubClient` (`search/repositories`, `repos`,
  `readme`, `contents`) + `GitHubSearchTool` (BaseTool wrapper). Token
  bucket: 10/min unauthenticated, 30/min authenticated. Token from
  `symbiote.github_token` config or `PROMETHEUS_GITHUB_TOKEN` env;
  never logged.
- `scout.py` — `ScoutEngine` / `ScoutReport` / `ScoutCandidate`. LLM
  generates 2-3 search queries from a problem statement, deduplicates
  results, filters BLOCKED licenses, scores remaining candidates with
  GBNF-enforced JSON output. Classification: recommended | viable |
  risky | blocked.
- `harvest.py` — `HarvestEngine` / `HarvestReport` / `ExtractedModule` /
  `AdaptationStep`. Shallow `git clone` (60s timeout), repo-size guard,
  LLM file picker, 15-file/50KB read budget, scanner pass on every file
  (any DANGEROUS aborts), LLM-produced adaptation plan. Persists to
  `~/.prometheus/symbiote/harvests/<repo>_<ts>/` and deletes the sandbox.
- `graft.py` — `GraftEngine` / `GraftReport` / `GraftedFile`. Provenance
  header on every adapted file (matches existing donor-file format),
  donor-import rewriting, allowed-roots guard (`src/prometheus/`,
  `tests/`), wiring tests appended to `tests/test_wiring.py`, full
  pytest run, PROMETHEUS.md update.
- `coordinator.py` — `SymbioteCoordinator` / `SymbioteSession` /
  `SymbiotePhase`. State machine, single-session mutex, SQLite
  persistence at `~/.prometheus/symbiote/sessions.db`.

### Wired in `scripts/daemon.py`
1. `create_tool_registry()` registers `github_search`,
   `symbiote_scout`, `symbiote_harvest`, `symbiote_graft`, `symbiote_status`.
2. The daemon constructs Scout/Harvest/Graft engines plus the coordinator
   if `symbiote.enabled`, then calls `prometheus.symbiote.set_coordinator()`
   so the agent-facing tools and `/symbiote` Telegram command can find it.

### CLI / Telegram surface
- `/symbiote <problem>` — start Scout (read-only, no approval).
- `/symbiote approve <full_name>` — request Harvest approval via the
  existing `ApprovalQueue` (Trust Level 1, mirrors `/gepa run`).
- `/symbiote graft` — request Graft approval via the same queue.
- `/symbiote status [session_id]`, `/symbiote abort`,
  `/symbiote history [N]` — read-only.

### Profile
A new `symbiote` builtin profile exposes the SYMBIOTE tools plus
`bash` / `file_read` / `file_write` / `file_edit` / `grep` / `glob` /
`github_search`. Other subsystems (sentinel, wiki, cron, learning) are
disabled in this profile.

### Tests (105 new)
- `tests/test_license_gate.py` (24)
- `tests/test_code_scanner.py` (15)
- `tests/test_github_search.py` (12)
- `tests/test_scout.py` (16)
- `tests/test_harvest.py` (10)
- `tests/test_graft.py` (11)
- `tests/test_symbiote_coordinator.py` (10)
- `tests/test_wiring.py::TestSymbioteWiring` (6)

Real instances throughout; only the GitHub API HTTP calls and the
provider's `stream_message` are stubbed.
