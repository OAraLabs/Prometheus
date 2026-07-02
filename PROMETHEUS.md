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

## WEAVE-PRESS — Printing Press CLI auto-discovery

When Prometheus needs a CLI it doesn't have, it consults a local clone of
[printing-press-library](https://github.com/mvanhorn/printing-press-library)
(~70 Go-based service CLIs: Slack, Cal.com, Airbnb, Sentry, etc.).
On user approval, it `go install`s the binary, copies the bundled
`SKILL.md` into `~/.prometheus/skills/`, and hot-reloads the skill
registry — the new tool is usable in the same conversation.

### `prometheus.tools.printing_press`
- `PrintingPressRegistry` — discovers a library clone in this order:
  explicit `library_path` → `~/printing-press-library/` →
  `/tmp/printing-press-library/` → `~/go/pkg/mod/.../printing-press-library`.
  Returns empty if no clone is found (no GitHub fallback in this sprint).
- `list_available()` — enumerates `cli-skills/<name>/SKILL.md`, parses
  YAML frontmatter, builds `CLIRecord(name, skill_name, category,
  description, install_module, bin_name, skill_path, installed)`.
  Only Go installs are surfaced; npm/other kinds are skipped.
- `search(query)` — fuzzy-matches by name → skill_name → bin_name →
  description with descending weights.
- `install(cli_name)` — runs `go install <module>@latest`, verifies the
  binary lands on PATH or in `~/go/bin/`, copies `SKILL.md`, fires the
  reload callback. Fail-safe: returns `InstallResult(success=False,
  error=...)` for missing Go, missing CLI, subprocess failure, or
  binary-not-found-after-install.
- `update_library()` — `git pull --ff-only` in the clone. Silent-fail
  if not a git checkout.
- `set_reload_callback(fn)` — daemon wires this to
  `SkillRegistry.reload_user_skills` so a fresh install is picked up
  without a daemon restart.

### Bash command-not-found hook
`prometheus.engine.agent_loop._maybe_suggest_printing_press` —
when a bash tool returns `is_error=True` and the output contains
`command not found: <name>` (any of the three shell forms), and the
session is **user-initiated** (origin classifier from the
TRUST-CONTEXT commit), the hook searches the registry and appends a
suggestion to the tool result content:

> 💡 Printing Press has a CLI for this: **cal-com** (productivity).
> To install, the user can run `/press install cal-com` — installation
> requires their explicit approval and will not happen automatically.

The model relays this to the user. Background/automated sessions
(SENTINEL, GEPA, AutoDream, smoke-tests, cron) get **no** suggestions.
Already-installed CLIs that just errored aren't re-suggested.

### Telegram `/press` surface
| Subcommand | Trust | Behaviour |
|---|---|---|
| `/press list [category]` | 2 | Enumerate available CLIs, optionally filtered |
| `/press search <query>` | 2 | Fuzzy match by name / description |
| `/press install <name>` | 1 | Approval-gated `go install` + skill copy |
| `/press installed` | 2 | Show CLIs whose binary is on PATH or in `~/go/bin` |
| `/press update` | 2 | `git pull --ff-only` the library clone |

Install routes through the same `ApprovalQueue` used by `/gepa run`
and `/symbiote graft`. The user sees a `Permission requested:` message
with `/approve <id>` — same UX as every other Trust Level 1 action.

### Skill hot-reload
`SkillRegistry.reload_user_skills()` re-scans
`~/.prometheus/skills/` and merges new or updated entries (purely
additive — never removes existing). `ToolSearchTool.get_skill_registry()`
exposes the live registry so the daemon can wire the reload callback.

### Daemon wiring (`scripts/daemon.py`)
After `ApprovalQueue` is wired:
1. If `printing_press.enabled` is true, build `PrintingPressRegistry`
   with the configured `library_path`.
2. Wire the reload callback to the running `SkillRegistry` via
   `tool_search.get_skill_registry().reload_user_skills`.
3. If `auto_update_library` is true, spawn a non-blocking
   `git pull --ff-only` task on startup.
4. Attach to `agent_loop._tool_metadata["printing_press"]` so the bash
   hook can find it, and to `telegram._printing_press` for `/press`.

### Config (`prometheus.yaml.default`)
```yaml
printing_press:
  enabled: false                  # opt-in
  library_path: null              # auto-detect
  auto_suggest: true              # bash command-not-found hook
  auto_update_library: false      # git pull on startup
```

### Tests (46 new)
- `tests/test_printing_press.py` (40) — frontmatter parsing,
  command-not-found detection across three shell forms, registry
  discovery, fuzzy search, `is_installed` (PATH + ~/go/bin fallback),
  install (subprocess mocked: success, no-Go, unknown CLI, subprocess
  failure), reload callback firing, skill hot-reload (additive +
  idempotent), suggestion hook (positive case, clean output, library
  unavailable, already-installed, suffix stripping), `update_library`
  edge cases.
- `tests/test_wiring.py::TestWeavePressWiring` (6) — Telegram
  `_cmd_press` + helpers exist, registry importable, suggestion
  helper is async, `SkillRegistry.reload_user_skills` exists,
  `ToolSearchTool.get_skill_registry` round-trips, `AgentLoop`
  accepts `tool_metadata` kwarg.

Network-dependent paths (real `go install`, real `git pull`) are
mocked — no live network calls in the test suite.

## Managed Tasks — background completion → notify / re-engage

Replaces fire-and-forget `nohup … &` with **managed tasks**: a durable record,
a non-LLM completion detector owned by the daemon, and a completion event that
(a) always notifies via Telegram and (b) optionally re-engages the agent.

**Three concerns kept strictly separate:**
- **Detection** — non-LLM, event-driven. `BackgroundTaskManager`
  (`tasks/manager.py`) owns three modes: process (`await proc.wait()`),
  `file_watch` (watchdog `Observer` + `PatternMatchingEventHandler`, in
  `tasks/watchers.py`), and `poll` (exponential-backoff fallback). On
  resolution it durably writes the record, then emits `task_completed` /
  `task_failed` on the SignalBus.
- **Notification** — cheap, templated, model-free. Owned by the **heartbeat**
  (`gateway/heartbeat.py`): it polls task transitions and pushes
  `✅ Task done` / `⚠️ Task failed` to the creating session's chat
  (`notify_target`, falling back to the global chat). Works even when SENTINEL
  is off.
- **Re-engagement** — only when the agent must *act on* the result.
  `TaskCompletionHandler` (`tasks/completion_handler.py`) subscribes to the
  signals and, when `on_complete ∈ {reengage, both}`, calls `inject_turn`.
  Bounded by `tasks.reengage_turn_cap`. `on_complete` gates re-engagement only;
  notification always fires.

**Durability.** `TaskStore` (`tasks/store.py`, `~/.prometheus/data/tasks.db`)
persists every record. On startup the daemon calls `resume_running()`:
`file_watch`/`poll` watchers are re-established; orphaned process tasks (whose
OS handle is gone) are reaped to `failed` (`error="daemon_restart"`) — no zombie
`running` rows.

**Security.** Task launch is vetted through the **same SecurityGate as cron**,
at system trust (`evaluate("bash", command=…, origin="system")`), failing
closed. A denied command yields a `failed` record and never spawns.

### `inject_turn` — the shared turn-injection primitive

`TelegramAdapter.inject_turn(session_id, content, *, provenance, is_trusted)`
(generalized from `_dispatch_to_agent`) is the **one** path that injects a
non-user turn into a session and runs the agent loop. Telegram inbound uses
`(provenance="user", is_trusted=True)`; the task completion handler uses
`(provenance="task_supervisor", is_trusted=False)`. Cron and a future
orchestrator clarification channel are meant to converge here — **do not build a
parallel re-engagement mechanism.**

**Provenance & trust** are structured fields on `ConversationMessage`
(`engine/messages.py`): `provenance` (closed enum: `user` / `cron` /
`task_supervisor` / `orchestrator`) and `is_trusted` (defaults **False** — safe
posture). These are the source of truth. The "⚠️ UNTRUSTED INPUT" banner is a
**derived rendering** applied at context-assembly (`render_messages_for_model`,
called at the model-call site in `agent_loop.py`) — never stored on the record,
so job stdout / watched-file contents reach the model fenced as data, and the
model is told not to execute instructions found inside them.

### Tools
`task_create` (`tools/builtin/task_create.py`) is the agent-facing entry; it
gained `on_complete`, `reengage_prompt`, `timeout_seconds`, and the
`file_watch`/`poll` args. **`session_id` + `notify_target` are resolved from the
trusted execution context, never from tool arguments** (which could originate in
observed content). `task_get` / `task_list` / `task_stop` already cover
status/cancel — no separate tools were added.

### Config (`config/prometheus.yaml`, gitignored) — `tasks:` section
`enabled`, `default_timeout_seconds` (3600), `reengage_turn_cap` (3),
`poll_initial_interval_seconds` (5), `poll_max_interval_seconds` (120),
`notify_on_failure`. All have safe in-code defaults, so the section is optional.

### Tests (`tests/test_managed_tasks.py`, 16) — side-effect assertions
orphan-tool registration; process success/failure → status + signal + heartbeat
notification (asserts the gateway send + per-session routing); file-watch on
`touch`; `on_complete=reengage` → `inject_turn` called with
`provenance="task_supervisor"` **and** `is_trusted=False` (end-to-end through the
bus); notify-only does not re-engage; turn-cap blocks excess; untrusted-tagging
regression guard (field is truth, banner is projection); SecurityGate-denied
command rejected with no process; timeout → `failed`; durability resume/reap.

## The loud-failure law (OAra Lab-wide, 2026-07-02)

**Degraded is a state that gets announced, never absorbed. No component may
catch-and-continue silently. Every daemon writes a success heartbeat; staleness
is surfaced, spoken, and shown.**

Origin: OAra Voice's memory extractor 404'd every 30 minutes for ~3.5 months while
reporting "No new events" — fail-safe silence killed that system invisibly. Prometheus
already leans loud (silent_failure telemetry, #78 journal tracebacks); keep it that way:
any new code that catches an exception and continues silently is a bug.

Prometheus's role in enforcement: the cron job `jarvis_heartbeat_watcher` (every 5 min)
runs OAra's `services/watcher/heartbeat_watcher.py`; a stale Jarvis component makes the
job exit non-zero, so the outage shows up as a failed cron status in Beacon's Config → Cron
tab and as an `error` event in the Jarvis Archive. Do not "fix" that job by making it
always exit 0 — its failure IS the feature.

### Corollary (Sprint 2, 2026-07-02): the law extends to configuration

A subsystem that is expected-enabled but dark is a failure state identical to a
stale heartbeat. LCM compaction sat behind an unset `compaction.enabled` flag
since birth — 1834 messages, zero summaries, and everyone debugged the
summarizer while the flag was the outage. "Never turned on" must be a boot-time
alarm, not an archaeology finding.

Enforcement: GET /api/status now reports a `compaction` block (enabled +
lcm counters); the OAra middleware's config audit (`subsystems:` manifest,
every 5 min) treats expected-but-dark as an immediate watcher alarm. When adding
a feature behind a config flag, either default it ON in prometheus.yaml.default
or register it in the OAra manifest with `expected: false` — a flag nobody
tracks is a future archaeology dig.
