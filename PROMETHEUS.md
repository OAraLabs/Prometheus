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

## Security Philosophy

Prometheus is designed for sovereign single-operator deployment on
dedicated hardware. The security model protects the operator from
autonomous agent actions, not from co-tenants. User-initiated commands
via Telegram have full trust. Background and self-improvement tasks run
under restricted trust with scanner verification.

### Trust Model
- User says it in Telegram → full trust, no blocks
- Background tasks (SENTINEL, AutoDream, cron) → SecurityGate applies
- External code from SYMBIOTE harvest → DangerousCodeScanner applies
- Self-improvement output (GEPA, SkillRefiner) → scanner applies
- Credentials loaded from local config files → always allowed
- Network commands (pip, curl) initiated by user → always allowed

This mirrors Hermes Agent's single-tenant assumption but without Docker
isolation, as Prometheus runs on dedicated hardware where the machine
itself is the security boundary.

### Origin classification
The trust origin is derived from `LoopContext.session_id`:
- `telegram:<chat_id>`, `slack:<channel>`, `cli`, `web` → **user**
- `system`, `None`, SYMBIOTE/GEPA/SENTINEL UUIDs → **system**

Helper: `prometheus.permissions.checker.origin_from_session_id()`.
Default for unrecognized values is `system` (the safer classification).

## Security
Shared security utilities live in `src/prometheus/security/`.

- `SecurityGate` (`permissions/checker.py`) — Trust-level evaluator wired
  into `AgentLoop` as `permission_checker`. Takes an `origin` parameter:
  `user` skips ExfiltrationDetector and the network/install
  approve-patterns; `system` applies the full restriction set. Always-
  blocked patterns (`rm -rf /`, `mkfs`, fork bomb), `denied_commands`,
  `denied_paths`, and the write_file workspace gate fire in BOTH origins.
- `ExfiltrationDetector` (`permissions/exfiltration.py`) — bash-command
  pattern detector. Flags only when an actual sensitive *file* on disk
  shows up in a network command (cat ~/.ssh/, < ~/.aws/, $(cat ~/...),
  pipes/redirects from sensitive paths, base64+sensitive_path+network).
  Bare `$VAR`-style env-var references are no longer flagged — that
  pattern is too coarse to distinguish exfil from legitimate auth.
- **`DangerousCodeScanner`** (`security/code_scanner.py`) — AST-based
  static-analysis pass on Python source. Flags `exec/eval/compile/__import__`
  and `os.system/popen/exec*/pty.spawn/ctypes.CDLL` at any scope, plus
  `subprocess/socket/httpx/requests/urllib` at module scope (suspicious,
  not blocking). Returns `ScanResult(verdict: clean|suspicious|dangerous)`.
  First introduced in GRAFT-SYMBIOTE for harvest-time gating; promoted to
  shared so any subsystem (hooks, audit pipelines, future eval gates) can
  reuse it. The old import path
  `prometheus.symbiote.code_scanner` still works via a re-export shim.
  - `scan_markdown_content()` — extracts Python from `​```python` /
    `​```py` fenced code blocks in markdown and runs the AST scan on each.
    Used by GEPA's promotion gate (`learning/gepa.py::_promote_winner`)
    and SkillRefiner (`learning/skill_refiner.py::maybe_refine`) before
    AI-generated skill variants are written to disk. A `dangerous`
    verdict refuses the write silently and continues.
- **`assert_path_under_roots`** (`security/path_guard.py`) — write-boundary
  helper. Resolves a candidate path BEFORE checking against an allow-list
  of roots, so `../` traversals that escape the allow-list are rejected
  even when the literal input starts with an allowed prefix. Used by
  `MemoryExtractor`'s `ObsidianWriter` to confine its write surface to
  `~/.prometheus/` (covering `MEMORY.md` and `wiki/`).

## Security Conventions

### Path Traversal Defense
Always resolve paths before prefix-checking. Never check prefix on the raw input string.

WRONG:

    if not str(user_path).startswith(str(allowed_root)):  # bypassable with ../

RIGHT:

    resolved = Path(allowed_root / user_path).resolve()
    if not str(resolved).startswith(str(allowed_root.resolve())):
        raise SecurityError(f"Path traversal attempt: {user_path}")

First caught in GRAFT-SYMBIOTE Step 5 (`GraftEngine._resolve_target`).
Test: `tests/test_graft.py::TestAllowedRoots::test_rejects_traversal`.

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

## SYMBIOTE Session B — BackupVault + MorphEngine + blue-green hot swap

Closed Phase-4 loop wired during GRAFT-SYMBIOTE Session B (commit on
2026-04-26). Disabled-by-default in `config/prometheus.yaml.default`
under `symbiote.morph.enabled` and `symbiote.backup.enabled`.

### `BackupVault` — versioned snapshot store (`symbiote/backup_vault.py`)
- Tarball-based snapshots written to `~/.prometheus/symbiote/backups/`.
  Includes `src/prometheus/`, `tests/`, `config/prometheus.yaml`,
  `PROMETHEUS.md`, `scripts/daemon.py`; optionally
  `~/.prometheus/{SOUL,AGENTS,ANATOMY}.md`.
  Excludes `.git/`, `__pycache__/`, `.venv/`, `node_modules/`,
  `~/.prometheus/wiki/`, `~/.prometheus/memory/`.
- Manifest at `~/.prometheus/symbiote/backups/manifest.db` (SQLite)
  tracks every snapshot plus a `restore_log` table.
- Retention: keeps the newest `max_backups` non-exempt snapshots.
  Exempt sources (never auto-deleted): `manual`, `symbiote_morph`,
  `pre_restore`. Today's snapshots are also retained regardless.
- `create_snapshot(description, source, capture_test_status=True)`
  records `tests_passing|failing|unknown` from `python3 -m pytest`
  with a 60s timeout (don't-block-on-failure).
- `restore_snapshot(backup_id, dry_run=False)` ALWAYS creates a
  `pre_restore` safety backup first. Path-traversal-safe extraction.

### `MorphEngine` — blue-green self-deployment (`symbiote/morph.py`)
- `prepare_candidate()` snapshots the live tree, copies it to
  `~/.prometheus/symbiote/candidate/`, runs the full pytest suite
  against the candidate, and produces a `MorphReport`.
- `execute_swap()` performs the hot swap with auto-rollback. Atomic
  shape: `mv live live.pre_swap; mv candidate live; start daemon;
  health-check; rollback if unhealthy`.
- `_detect_daemon_manager()` chooses between three strategies in this
  exact order, with a 3s timeout on systemctl and zero hangs:
  1. `systemctl is-active --user prometheus` → `"systemd"`
  2. `~/.prometheus/daemon.lock` exists AND PID alive → `"pidfile"`
  3. Otherwise → `"pkill"` (matches `python.*prometheus daemon`)
  Result is cached so stop and start use the same strategy.
- Health-check watchdog: 60s timeout, 5s interval, 3 consecutive passes
  required. Optional HTTP `/health` ping if `daemon_health_url` set.
- Auto-rollback is the ONE autonomous (Trust Level 3) action — a
  broken daemon can't ask permission to fix itself. Failed candidate
  is preserved in `~/.prometheus/symbiote/post_mortem/failed_<ts>/`
  with a `REASON.txt`.

### State machine extension (`SymbioteCoordinator`)
New phases on top of Session A's: `MORPHING`,
`AWAITING_SWAP_APPROVAL`, `SWAPPING`, `HEALTH_CHECK`, `ROLLED_BACK`.
New methods: `start_morph(session_id, morph_engine)` and
`approve_swap(session_id, morph_engine)`. The MORPH path is opt-in —
a graft session can also exit straight to `COMPLETE` via the existing
`approve_graft()` if the user doesn't want to hot-swap.

### Telegram surface (5 new `/symbiote` subcommands)
| Subcommand | Trust | Behaviour |
|---|---|---|
| `/symbiote backup [desc]` | 2 | Create a manual snapshot via BackupVault |
| `/symbiote backups [N]` | 2 | List most recent N snapshots |
| `/symbiote restore [id\|dry]` | 1 | Approval-gated restore; dry-run flag prints diff only |
| `/symbiote morph` | 1 | Stage candidate, run tests, produce MorphReport |
| `/symbiote swap` | 1 | Approval-gated hot swap with auto-rollback |
The five Session-A subcommands (`<problem>`, `approve`, `graft`,
`status`, `abort`, `history`) are unchanged. All approval gates use
the same `ApprovalQueue.request_approval(...)` background-task pattern
as `/gepa run`.

### Daemon wiring (`scripts/daemon.py`)
After `SymbioteCoordinator` is instantiated:
1. If `symbiote.backup.enabled`, build `BackupVault` and attach as
   `telegram._backup_vault`.
2. If `symbiote.morph.enabled`, build `MorphEngine` and attach as
   `telegram._morph_engine`. The `daemon_manager` config key (default
   `auto`, set to `pidfile` on OAra-Mini) maps directly to the
   override constructor arg.

### Tests (49 new — total 1,727 passing)
- `tests/test_backup_vault.py` (16) — tarball creation, manifest,
  retention, dry-run/full restore, pre-restore safety net.
- `tests/test_morph.py` (16) — daemon-manager detection (incl. a
  "must not hang within 5s when systemctl missing" guard), candidate
  staging, swap-aborts-when-stop-fails, auto-rollback preserves the
  failed candidate, path-traversal guards.
- `tests/test_symbiote_coordinator.py::TestMorphTransitions` (7)
- `tests/test_wiring.py::TestSymbioteSessionBWiring` (5)
- `tests/test_morph.py::TestStageCandidate` (2) — pycache exclusion
- `tests/test_morph.py::TestPrepareCandidate` (2) — backup created,
  blocked-when-tests-fail
- `tests/test_morph.py::TestSwap` (4)
- `tests/test_morph.py::TestPathTraversalGuard` (3)

## WEAVE — Core Web Tools + Capability Audit

Closed loop wired during the WEAVE sprint (commit on 2026-04-26).
Additive on top of the pre-existing `web_fetch` and `web_search` tools.

### New tools (`src/prometheus/tools/builtin/`)
- **`youtube_transcript`** (`youtube_transcript.py`) — wraps `yt-dlp`
  to fetch subtitle tracks without downloading video. Accepts full
  YouTube URLs, `youtu.be` short URLs, embed/shorts URLs, and bare
  11-character video IDs. Stdlib-only VTT and SRV3 parsers (no
  beautifulsoup). Read-only unless `save_to` is set; all errors
  return `ToolResult(is_error=True)` — never raises.
- **`download_file`** (`download_file.py`) — streams a URL to disk
  with `httpx.AsyncClient.stream`. Default destination
  `~/.prometheus/downloads/`. SSRF protection mirrors `web_fetch`.
  Path-traversal guard rejects `/etc`, `/sys`, `/boot`, `/proc`,
  `/dev`, `~/.ssh` by resolving the path before checking the prefix
  (matches the GraftEngine pattern). Size cap 100MB by default —
  trips on either the server's `Content-Length` header or
  bytes-written during streaming.

### Wiring
- Both tools are exported from `prometheus.tools.builtin` and
  registered in `__main__.create_tool_registry()` in the
  "Web + messaging" group, alongside `WebSearchTool` and
  `WebFetchTool`. The pre-existing tools were not modified.
- `config/prometheus.yaml.default` adds a `web_tools:` section with
  `fetch_timeout_seconds`, `fetch_max_chars`, `search_max_results`,
  `download_dir`, `download_max_mb`, and
  `youtube_transcript_language`. The corresponding `web_tools`
  block in `config/prometheus.yaml` is gitignored per project policy.

### `scripts/web_capability_audit.py`
Subclasses `SmokeTestRunner` from `scripts/smoke_test_tool_calling.py`
into `WebAuditRunner`. Adds:
- `expect_in_output_any` (any-of-list matching) and
  `expect_no_circuit_breaker` assertions.
- Failure classification: `circuit_breaker`, `wrong_tool`,
  `execution_failure`, `wrong_answer`, `timeout`.
- Hard time-budget gating; tasks past the cap are skipped.
- ~36 audit tasks across 7 categories (search, fetch, youtube,
  download, research, graceful, railway).
- JSON + Markdown reports written to `~/.prometheus/audits/`.

### Telegram surface — `/audit`
Three forms, all additive:
| Form | Behaviour |
|---|---|
| `/audit` | Show summary of the most recent audit JSON |
| `/audit run` | Spawn the full audit as a background subprocess |
| `/audit <category>` | Single-category run (search, fetch, etc.) |

The implementation lives at the bottom of `gateway/telegram.py`
(`_cmd_audit`, `_audit_show_last`, `_audit_kick_off`). It does not
touch any existing `/symbiote`, `/gepa`, `/sentinel`, or core
commands.

### Pytest marker
A new `network` marker is registered in `pyproject.toml` so
network-dependent tests can be skipped with
`pytest -m 'not network'`. The existing `integration` marker is
unchanged.

### Tests
- `tests/test_web_tools.py` (52 unit tests) — URL normalization,
  VTT/SRV3 parsing, error classification, path-traversal guard,
  SSRF guard, file-size formatting, network smoke tests behind the
  `network` marker.
- `tests/test_web_audit.py` (18 tests) — category aggregation,
  failure breakdown, circuit-breaker detection, time-budget gating,
  JSON/Markdown report generation, `/audit` Telegram wiring.
- `tests/test_wiring.py::TestWeaveWebToolsWiring` (6) — verifies
  both new tools are registered and the pre-existing `web_fetch` and
  `web_search` are still present.
