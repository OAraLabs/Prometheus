# GRAFT: GitHub Research & Code Assimilation
## Codename: SYMBIOTE

**Date:** April 21, 2026
**Purpose:** Prometheus can create skills from its own experience (SkillExtractor) and evolve them (SkillRefiner), but it can't go *outward* — it can't research GitHub for solutions to capability gaps, evaluate candidates, extract relevant code, and graft it into itself. SYMBIOTE gives Prometheus the ability to scout, harvest, and assimilate open-source code from GitHub repos, with full provenance tracking, license enforcement, and SecurityGate integration at every phase.

**Estimated Time:** 3–4 hours
**Dependencies:** Sprint 4 (SecurityGate), Sprint 11/14 (Security Audit + DangerousCodeScanner), GRAFT-BOOTSTRAP (SOUL.md, provenance conventions), GRAFT-PROFILES (profile system for symbiote profile)

**This is ADDITIVE ONLY. Do not remove or replace any existing functionality.**

> **Session A status (2026-04-25):** Steps 0–11 landed at commit `fb9efe2`
> (Scout → Harvest → Graft pipeline, 4 agent tools, `/symbiote` Telegram
> command, `symbiote` profile, 105 new tests, 1,678 total passing).
> `DangerousCodeScanner` was subsequently promoted to
> `prometheus.security.code_scanner` (commit `0635508`) and a
> backward-compat shim left at `prometheus.symbiote.code_scanner`.
> Path-traversal guard pattern documented in PROMETHEUS.md
> (commit `6153881`).
>
> **Session B starts at Step 12** (BackupVault) and finishes at Step 17.

---

## The Problem

```
TODAY:
  Will: "Prometheus, I need you to handle YAML config validation better"
  Prometheus: "I can try with what I have"
  Will: <manually searches GitHub, finds pydantic-settings or strictyaml>
  Will: <manually reads source, extracts patterns>
  Will: <writes GRAFT spec, feeds to Claude Code>
  Will: <waits for sprint to execute>

AFTER SYMBIOTE:
  Will: "/symbiote I need robust YAML config validation with schema enforcement"
  Prometheus: <scouts GitHub API, finds candidates, ranks by relevance/license/quality>
  Prometheus: "Found 3 candidates. Top pick: strictyaml (MIT, 1.2K stars, pure Python,
               no deps). Runner-up: pydantic-settings. Want me to harvest strictyaml?"
  Will: "yes"
  Prometheus: <clones to sandbox, reads source, extracts validation patterns>
  Prometheus: "Harvest complete. Extracted SchemaValidator pattern (142 lines).
               License: MIT. Modifications needed: adapt to prometheus.yaml structure,
               wire into ConfigLoader. Ready to graft?"
  Will: "do it"
  Prometheus: <adapts interfaces, adds provenance headers, writes tests, updates PROMETHEUS.md>
```

---

## Architecture: Four-Phase Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│  SYMBIOTE PIPELINE                                                   │
│                                                                       │
│  Phase 1: SCOUT (Trust Level 2 — auto, read-only)                   │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Problem Statement (from user or SENTINEL)                    │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  GitHubSearchTool.search(query, filters)                     │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  For each candidate:                                          │    │
│  │    ├── Read README.md (via GitHub API raw content)            │    │
│  │    ├── Check license (SPDX identifier)                       │    │
│  │    ├── Check stars, last commit, language                     │    │
│  │    └── Score relevance against problem statement              │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  ScoutReport: ranked candidates with rationale                │    │
│  │  → Present to user for approval                               │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  ── USER APPROVAL GATE (Trust Level 1) ──                            │
│                                                                       │
│  Phase 2: HARVEST (Trust Level 1 — requires approval)               │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Approved candidate repo                                      │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  git clone --depth 1 → ~/.prometheus/symbiote/sandbox/       │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  Directory scan → identify relevant source files              │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  Targeted file reads (budget: max 15 files, 50KB total)      │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  DangerousCodeScanner.scan() on all read files               │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  LLM analysis: extract relevant modules/patterns/functions    │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  HarvestReport:                                               │    │
│  │    ├── harvest.md — what was extracted, interfaces, deps      │    │
│  │    ├── license.md — full license text + SPDX + obligations    │    │
│  │    ├── extracted/ — raw extracted source files (read-only)    │    │
│  │    └── adaptation_plan.md — what needs to change for Prometheus│   │
│  │  → Present to user for approval                               │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  ── USER APPROVAL GATE (Trust Level 1) ──                            │
│                                                                       │
│  Phase 3: GRAFT (Trust Level 1 — requires approval)                 │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Approved adaptation plan                                     │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  Create new files with provenance headers                     │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  Adapt interfaces to Prometheus conventions                   │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  Write integration tests → tests/test_wiring.py              │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  Run tests → all must pass (existing + new)                  │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  Update PROMETHEUS.md with new interfaces                     │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  GraftReport: what was added, where, how it wires in         │    │
│  │  → Cleanup sandbox                                            │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  ── USER APPROVAL GATE (Trust Level 1) ──                            │
│                                                                       │
│  Phase 4: MORPH (Trust Level 1 — requires approval)                 │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Approved GraftReport                                         │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  Snapshot current Prometheus → backup vault                   │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  Clone live instance → candidate staging area                │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  Apply graft to candidate (not live)                         │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  Run FULL test suite against candidate                        │    │
│  │       │                                                       │    │
│  │       ├── Tests fail? → Abort, discard candidate, report     │    │
│  │       │                                                       │    │
│  │       └── Tests pass? → Present swap confirmation to user    │    │
│  │                │                                              │    │
│  │                ▼                                              │    │
│  │  ── USER SWAP APPROVAL ──                                    │    │
│  │                │                                              │    │
│  │                ▼                                              │    │
│  │  Stop daemon → swap candidate into live → restart daemon     │    │
│  │       │                                                       │    │
│  │       ▼                                                       │    │
│  │  Health check (60s watchdog)                                  │    │
│  │       │                                                       │    │
│  │       ├── Healthy? → COMPLETE. Backup retained.              │    │
│  │       │                                                       │    │
│  │       └── Unhealthy? → AUTO-ROLLBACK (no approval needed)   │    │
│  │                         Restore from backup, restart daemon   │    │
│  │                         Notify user: "Rolled back to v{N}"   │    │
│  └──────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Read These Files First

```
# Security — SYMBIOTE must integrate with all of these:
src/prometheus/permissions/checker.py         # SecurityGate (Trust Levels)
src/prometheus/security/code_scanner.py       # DangerousCodeScanner (canonical)
src/prometheus/symbiote/code_scanner.py       # backward-compat shim
src/prometheus/permissions/exfiltration.py    # ExfiltrationDetector

# Tool registration — understand how to add new tools:
src/prometheus/tools/base.py                  # BaseTool, ToolRegistry
src/prometheus/tools/builtin/__init__.py      # tool exports
src/prometheus/tools/builtin/bash.py          # BashTool

# Existing learning system — SYMBIOTE extends, doesn't replace:
src/prometheus/skills/loader.py               # SkillLoader
src/prometheus/learning/skill_creator.py      # SkillCreator (post-task hook)
src/prometheus/learning/skill_refiner.py      # SkillRefiner (post-task hook)

# Wiki — harvested knowledge gets filed here:
src/prometheus/memory/wiki_compiler.py        # WikiCompiler

# Profiles — the symbiote profile already exists from Session A:
src/prometheus/config/profiles.py             # ProfileStore, AgentProfile

# Approval flow — Session B reuses the existing /gepa-style pattern:
src/prometheus/permissions/approval_queue.py  # ApprovalQueue (Trust Level 1)

# Already wired SYMBIOTE Session A modules — DO NOT recreate:
src/prometheus/symbiote/license_gate.py       # LicenseGate
src/prometheus/symbiote/github_search.py      # GitHubSearchTool
src/prometheus/symbiote/scout.py              # ScoutEngine
src/prometheus/symbiote/harvest.py            # HarvestEngine
src/prometheus/symbiote/graft.py              # GraftEngine
src/prometheus/symbiote/coordinator.py        # SymbioteCoordinator (state machine)
src/prometheus/tools/builtin/symbiote_*.py    # 4 agent-facing tools

# Daemon + service management — for MORPH phase:
scripts/daemon.py
src/prometheus/gateway/heartbeat.py           # Health check patterns

# Config:
config/prometheus.yaml
config/prometheus.yaml.default

# Provenance convention (from CONTRIBUTING.md):
# Every file extracted from a donor project must include:
# Source: <project> (<github_url>)
# Original: <original_path>
# License: <SPDX>
# Modified: <what changed>

# Gateway — for /symbiote command (already has Session A subcommands):
src/prometheus/gateway/telegram.py
```

---

## Prompt for Claude Code

```
This builds the SYMBIOTE system for Prometheus — a four-phase pipeline
(Scout → Harvest → Graft → Morph) that researches GitHub repos, extracts
relevant code, integrates it into Prometheus with full provenance and
security, and (Session B) deploys it via blue-green hot swap with auto-
rollback.

ADDITIVE ONLY. Do not remove or replace any existing functionality.

⚠️ EXISTING TELEGRAM COMMANDS — DO NOT TOUCH ANY OF THESE:
/start, /status, /help, /reset, /model, /wiki, /sentinel,
/benchmark, /context, /skills, /approve, /deny, /pending, /profile,
/anatomy, /doctor, /route, /gepa, /claude, /gpt, /gemini, /xai,
/grok, /local, /tools, /beacon, /clear, /symbiote (existing
subcommands: <problem>, approve, graft, status, abort, history)

These all work and must continue working exactly as they are.
Session B ADDS the morph/swap/backup/backups/restore subcommands to
/symbiote — do not modify the existing five subcommands.

STEP 0 — SURVEY CURRENT STATE: (Session A landed; re-read these.)

* src/prometheus/permissions/checker.py
* src/prometheus/security/code_scanner.py
* src/prometheus/permissions/approval_queue.py
* src/prometheus/symbiote/coordinator.py     # state machine to extend
* src/prometheus/tools/base.py
* src/prometheus/tools/builtin/__init__.py
* src/prometheus/config/profiles.py          # symbiote profile exists
* src/prometheus/gateway/telegram.py         # /symbiote partly wired
* config/prometheus.yaml
* scripts/daemon.py                          # SymbioteCoordinator wired

Answer these before proceeding:
1. How does SecurityGate classify trust levels for tool calls?
2. How does DangerousCodeScanner work — what does it scan for?
3. How are tools registered and exported?
4. What is the provenance header format from existing donor files?
5. How does the profile system control which tools are available?
6. How does /gepa run enqueue through ApprovalQueue?
   (Session B's MORPH approval gate must use the SAME pattern.)

Steps 1–11 are SESSION A and are already complete (commit fb9efe2).
DO NOT re-implement them. Continue at Step 12 below.
```

---

## STEPS 1–11 — SESSION A (already landed, do not redo)

| Step | Component | Status |
|---|---|---|
| 1 | LicenseGate | ✅ `prometheus.symbiote.license_gate` |
| 2 | GitHubSearchTool + GitHubClient | ✅ `prometheus.symbiote.github_search` |
| 3 | ScoutEngine | ✅ `prometheus.symbiote.scout` |
| 4 | HarvestEngine | ✅ `prometheus.symbiote.harvest` |
| 5 | GraftEngine | ✅ `prometheus.symbiote.graft` |
| 6 | SymbioteCoordinator | ✅ `prometheus.symbiote.coordinator` (state machine has placeholders for Session B phases) |
| 7 | 4 agent-facing tools | ✅ `prometheus.tools.builtin.symbiote_{scout,harvest,graft,status}` |
| 8 | /symbiote Telegram | ✅ subcommands: `<problem>`, `approve`, `graft`, `status`, `abort`, `history` |
| 9 | symbiote profile | ✅ `prometheus.config.profiles._BUILTINS["symbiote"]` |
| 10 | Config keys | ✅ `symbiote:` section in `config/prometheus.yaml.default` |
| 11 | Tests | ✅ 105 new across 7 files; full suite 1,678 passing |

**Session A deviations to be aware of:**

1. The spec assumed `src/prometheus/security/code_scanner.py` already
   existed from "Sprint 11/14"; it didn't. Session A built
   `DangerousCodeScanner` and a follow-up promoted it into the canonical
   `prometheus.security` package. The shim at
   `prometheus.symbiote.code_scanner` still works.
2. The spec's path-traversal guard (`startswith` on raw input string) was
   bypassable via `..`. Session A's `GraftEngine._resolve_target` resolves
   the path *before* checking the prefix. See PROMETHEUS.md §Security
   Conventions.

---

## STEP 12 — CREATE BACKUP VAULT:

Create: `src/prometheus/symbiote/backup_vault.py` (~250 lines)

The backup vault is the safety net that makes MORPH viable. It creates
timestamped, versioned snapshots of Prometheus's source code, config,
and state files. Backups are compressed tarballs stored in a dedicated
directory with metadata.

This is also independently useful outside of SYMBIOTE — it's a general
"backup before doing something risky" system.

```python
@dataclass
class BackupSnapshot:
    backup_id: str               # Format: "v{N}_{timestamp}" e.g. "v3_20260421_143022"
    version_number: int          # Auto-incrementing
    timestamp: str               # ISO 8601
    description: str             # What triggered this backup
    source: str                  # "symbiote_morph" | "manual" | "pre_graft" | "scheduled"
    tarball_path: Path           # Path to .tar.gz
    size_bytes: int
    file_count: int
    manifest: list[str]          # List of files included
    prometheus_md_hash: str      # SHA256 of PROMETHEUS.md at backup time
    test_status: str             # "passing" | "failing" | "unknown"
    metadata: dict               # Arbitrary metadata (session_id, etc.)

class BackupVault:
    """Versioned backup system for Prometheus source and state.

    Storage: ~/.prometheus/symbiote/backups/
    ├── manifest.db              — SQLite index of all backups
    ├── v1_20260420_100000.tar.gz
    ├── v2_20260421_120000.tar.gz
    └── v3_20260421_143022.tar.gz

    What gets backed up:
    - src/prometheus/            — all production source code
    - tests/                     — all test files
    - config/prometheus.yaml     — config (secrets included — vault is local-only)
    - PROMETHEUS.md              — project documentation
    - ~/.prometheus/SOUL.md      — identity (optional, configurable)
    - ~/.prometheus/AGENTS.md    — agent registry (optional)
    - ~/.prometheus/ANATOMY.md   — hardware awareness (optional)

    What does NOT get backed up:
    - .git/                      — use git for git history
    - ~/.prometheus/wiki/        — wiki has its own persistence
    - ~/.prometheus/memory/      — memory has its own persistence
    - ~/.prometheus/symbiote/    — don't backup the backup system
    - __pycache__/, .venv/, node_modules/

    Retention policy:
    - Keep last N backups (default: 10, configurable)
    - Never auto-delete backups from the current day
    - Manual backups are exempt from auto-cleanup
    """

    VAULT_ROOT = Path.home() / ".prometheus" / "symbiote" / "backups"

    def __init__(self, project_root: Path,
                 max_backups: int = 10,
                 include_identity: bool = True):
        ...

    async def create_snapshot(self, description: str,
                               source: str = "manual",
                               metadata: dict = None) -> BackupSnapshot:
        """Create a new backup snapshot.

        Steps:
        1. Determine next version number from manifest.db
        2. Collect files to backup (respecting include/exclude rules)
        3. Create tarball: tar -czf {vault_root}/{backup_id}.tar.gz {files}
        4. Compute SHA256 of PROMETHEUS.md
        5. Run pytest to capture current test status (timeout 60s, don't block on failure)
        6. Write manifest entry to SQLite
        7. Enforce retention policy (delete oldest if over max_backups)

        Returns BackupSnapshot with all metadata.
        """
        ...

    async def restore_snapshot(self, backup_id: str,
                                dry_run: bool = False) -> RestoreResult:
        """Restore Prometheus from a backup snapshot.

        Steps:
        1. Verify backup_id exists in manifest
        2. Create a PRE-RESTORE backup automatically (safety net for the safety net)
        3. If dry_run: list what would change, don't actually restore
        4. Extract tarball to a staging area
        5. Diff staging vs live: show what files changed/added/removed
        6. Replace live files with staged files
        7. Run pytest to verify restored state
        8. Update manifest with restore event

        The pre-restore backup means you can always undo a restore.
        Worst case: /symbiote restore → breaks things → /symbiote restore pre_restore_v3
        """
        ...

    def list_snapshots(self, limit: int = 20) -> list[BackupSnapshot]: ...
    def get_snapshot(self, backup_id: str) -> BackupSnapshot | None: ...
    def get_latest(self) -> BackupSnapshot | None: ...

    def _collect_files(self) -> list[Path]:
        """Collect files for backup, respecting include/exclude rules."""
        ...

    def _enforce_retention(self):
        """Delete oldest backups beyond max_backups limit.
        Skip: today's backups, manual backups, pre-restore backups.
        """
        ...

    def _get_next_version(self) -> int:
        """Get next version number from manifest.db."""
        ...

@dataclass
class RestoreResult:
    backup_id: str
    files_restored: int
    files_added: list[str]       # Files in backup but not in live
    files_removed: list[str]     # Files in live but not in backup
    files_changed: list[str]     # Files that differ
    pre_restore_backup_id: str   # The safety backup created before restoring
    tests_passed: bool
    tests_output: str
```

SQLite schema (at `~/.prometheus/symbiote/backups/manifest.db`):

```sql
CREATE TABLE IF NOT EXISTS backup_manifest (
    backup_id TEXT PRIMARY KEY,
    version_number INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    description TEXT NOT NULL,
    source TEXT NOT NULL,
    tarball_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    file_count INTEGER NOT NULL,
    manifest_json TEXT NOT NULL,
    prometheus_md_hash TEXT,
    test_status TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS restore_log (
    restore_id TEXT PRIMARY KEY,
    backup_id TEXT NOT NULL,
    pre_restore_backup_id TEXT NOT NULL,
    restored_at TEXT NOT NULL,
    files_restored INTEGER NOT NULL,
    tests_passed INTEGER,
    FOREIGN KEY (backup_id) REFERENCES backup_manifest(backup_id)
);
```

---

## STEP 13 — CREATE MORPH ENGINE:

Create: `src/prometheus/symbiote/morph.py` (~350 lines)

The morph engine orchestrates Phase 4. It takes an approved GraftReport,
creates a backup, stages a candidate instance with the graft applied,
validates it, and (on approval) performs the hot swap with automatic
rollback on health check failure.

```python
@dataclass
class MorphReport:
    backup_snapshot: BackupSnapshot    # The backup created before morphing
    candidate_path: Path               # Where the candidate is staged
    graft_applied: bool                # Was the graft cleanly applied?
    tests_passed: bool                 # Did the full test suite pass?
    tests_output: str                  # Full pytest output
    files_changed: list[str]           # Files that differ from live
    new_dependencies: list[str]        # pip packages the graft needs
    ready_to_swap: bool                # All checks passed?
    reason_if_not_ready: str | None    # Why swap is blocked

@dataclass
class SwapResult:
    success: bool
    previous_version: str              # backup_id of what was live
    new_version: str                   # What's live now
    health_check_passed: bool
    rolled_back: bool                  # True if auto-rollback happened
    rollback_reason: str | None
    downtime_seconds: float            # How long daemon was stopped
    notification_sent: bool            # Did user get notified?

CANDIDATE_ROOT = Path.home() / ".prometheus" / "symbiote" / "candidate"

class MorphEngine:
    """Phase 4: Blue-green self-deployment with automatic rollback.

    On Will's hardware specifically:
    - Prometheus runs on OAra-Mini, currently launched via
      `python -m prometheus daemon` (NOT a systemd unit). The daemon
      may be moved under systemd later. _detect_daemon_manager() must
      handle BOTH today's reality and tomorrow's setup without code
      changes.
    - llama.cpp runs on the 4090 in a separate process and is unaffected.
    - Total downtime: ~2-5 seconds.

    Safety invariants:
    - A backup ALWAYS exists before any swap.
    - Rollback is ALWAYS automatic on health check failure.
    - The pre-swap backup is NEVER auto-deleted (exempt from retention).
    - If the swap process itself crashes, the backup is still on disk
      for manual recovery.
    """

    def __init__(self, backup_vault: BackupVault,
                 project_root: Path,
                 health_check_timeout: int = 60,
                 health_check_interval: int = 5):
        ...

    async def prepare_candidate(self, graft_report: GraftReport) -> MorphReport:
        """Stage a candidate instance with the graft applied.

        Steps:
        1. Create backup snapshot (source="symbiote_morph")
        2. Copy live src/prometheus/ → candidate staging area
        3. Apply graft changes to candidate (from GraftReport)
        4. Copy tests/ → candidate staging area
        5. Install any new dependencies to a candidate venv
        6. Run FULL test suite against candidate
        7. Diff candidate vs live
        8. Produce MorphReport

        The candidate is a COMPLETE copy — it could run independently.
        This lets us test it in isolation before touching the live instance.
        """
        ...

    async def execute_swap(self, morph_report: MorphReport) -> SwapResult:
        """Execute the hot swap. REQUIRES prior user approval.

        Steps:
        1. Verify morph_report.ready_to_swap is True.
        2. Verify backup exists and is valid.
        3. _detect_daemon_manager() to pick stop/start strategy.
        4. Notify user: "Swapping in 5 seconds. Auto-rollback active."
        5. Stop daemon (using detected strategy).
        6. Atomic swap:
           a. mv src/prometheus src/prometheus.pre_swap
           b. mv candidate/prometheus src/prometheus
           c. mv tests tests.pre_swap (if changed)
           d. mv candidate/tests tests (if changed)
        7. Start daemon (using detected strategy).
        8. Start health check watchdog.
        9. If healthy within timeout → COMPLETE
           - Clean up .pre_swap dirs.
           - Notify user: "Swap complete. Now running v{N}."
        10. If unhealthy → AUTO-ROLLBACK
           - Stop daemon (best effort).
           - mv src/prometheus src/prometheus.failed
           - mv src/prometheus.pre_swap src/prometheus
           - Start daemon.
           - Notify user: "Rolled back to v{N-1}. Reason: {reason}"
           - Save failed candidate for post-mortem.
        """
        ...

    # ----- DAEMON-MANAGER DETECTION (CRITICAL) -----------------------
    #
    # The spec originally said `daemon_stop_method: systemd` as if systemd
    # were guaranteed. In reality the daemon may be running under any of
    # three strategies. _detect_daemon_manager() MUST check in this exact
    # order and never assume systemd. NEVER hang waiting for a unit that
    # doesn't exist.

    async def _detect_daemon_manager(self) -> str:
        """Return one of "systemd" | "pidfile" | "pkill".

        Detection order (run sequentially, return on first hit):

        1. ``systemctl is-active --user prometheus`` — return "systemd"
           IFF stdout is "active" AND exit code is 0.
           Use a 3s timeout; if systemctl is missing or hangs, fall through.

        2. ``~/.prometheus/daemon.pid`` exists AND points to a live PID
           (verified via ``os.kill(pid, 0)``) — return "pidfile".

        3. Otherwise — return "pkill".
           This is the fallback for "we lost track of the daemon" cases.
           The pkill pattern is ``python.*prometheus daemon`` (matches
           both ``python -m prometheus daemon`` and ``python scripts/daemon.py``).

        Cache the result on the instance so the second call (start, after
        stop) returns the same strategy without re-detecting.
        """
        ...

    async def _stop_daemon(self) -> bool:
        """Stop the Prometheus daemon. Returns True if stopped, False otherwise.

        Strategy is dictated by _detect_daemon_manager():
          systemd  → ``systemctl --user stop prometheus``
          pidfile  → ``kill -TERM <pid>``, wait up to 10s, ``kill -KILL`` if needed
          pkill    → ``pkill -TERM -f "python.*prometheus daemon"``,
                     wait 10s, ``pkill -KILL -f ...`` if anything still matches

        Timeout: 10 seconds total. If daemon won't stop, ABORT the swap and
        log loudly — DO NOT proceed to the directory swap when an old
        daemon is still alive on the original tree.
        """
        ...

    async def _start_daemon(self) -> bool:
        """Start the Prometheus daemon. Strategy must match _stop_daemon's.

          systemd  → ``systemctl --user start prometheus``
          pidfile  → ``nohup python -m prometheus daemon >/dev/null 2>&1 &``,
                     write the new pid into ~/.prometheus/daemon.pid
          pkill    → same launch as pidfile; write pid file too so future
                     stops can use the cleaner pidfile path
        """
        ...

    async def _health_check_watchdog(self, timeout: int = 60,
                                      interval: int = 5) -> tuple[bool, str]:
        """Monitor the restarted daemon for health.

        Checks (every `interval` seconds for `timeout` seconds):
        1. Process is running (PID exists / systemctl is-active "active").
        2. Telegram gateway is connected (check heartbeat).
        3. llama.cpp is reachable (provider health check).
        4. Can process a simple prompt ("health check ping").

        Returns (healthy: bool, reason: str).

        First check is at interval/2 (give daemon time to start).
        Must pass 3 consecutive checks to be considered healthy.
        Single failure after 3 successes → still healthy (transient).
        """
        ...

    async def _atomic_swap(self, candidate_path: Path) -> bool:
        """Perform the directory swap.

        This is as atomic as filesystem operations get:
        - mv is atomic on same filesystem (which it will be).
        - If mv fails mid-operation, either old or new is live, never partial.
        - .pre_swap dirs are the rollback path.

        Guard: verify candidate_path is under CANDIDATE_ROOT before swapping.
        Guard: verify project_root/src/prometheus exists before swapping.
        Guard: resolve before prefix-check — see PROMETHEUS.md §Security
               Conventions / Path Traversal Defense.
        """
        ...

    async def _auto_rollback(self, reason: str) -> bool:
        """Automatic rollback — NO user approval needed.

        This is the one autonomous action in the entire SYMBIOTE pipeline.
        Rationale: if the daemon is broken, it can't ask for permission
        to fix itself. The backup is the contract — user approved the swap
        knowing rollback is automatic.

        Steps:
        1. Stop daemon (if running) — best effort.
        2. mv src/prometheus src/prometheus.failed (preserve for post-mortem).
        3. mv src/prometheus.pre_swap src/prometheus.
        4. Start daemon.
        5. Verify health.
        6. Notify user via Telegram.
        7. Log to restore_log in BackupVault.
        """
        ...

    def _cleanup_pre_swap(self):
        """Remove .pre_swap directories after successful swap.
        Only called after health check confirms new version is stable.
        """
        ...

    def _preserve_failed_candidate(self, candidate_path: Path, reason: str):
        """Move failed candidate to a post-mortem directory for debugging.
        ~/.prometheus/symbiote/post_mortem/{timestamp}/
        Includes the failure reason and health check logs.
        """
        ...
```

---

## STEP 14 — ADD MORPH CONFIG:

Modify: `config/prometheus.yaml` and `config/prometheus.yaml.default`
(MERGE into the symbiote section that Session A added)

```yaml
symbiote:
  # ... existing config from Session A ...
  morph:
    enabled: true
    health_check_timeout_seconds: 60    # How long to watch after swap
    health_check_interval_seconds: 5    # Check frequency
    auto_rollback: true                 # Always true in V1 (not configurable)
    daemon_manager: auto                # auto | systemd | pidfile | pkill
                                         # auto = run _detect_daemon_manager()
                                         # The other values FORCE that strategy.
                                         # Use "pidfile" today on OAra-Mini.
    pre_swap_notification: true         # Notify user before swap
  backup:
    vault_root: "~/.prometheus/symbiote/backups"
    max_backups: 10                     # Retention limit
    include_identity: true              # Backup SOUL.md, AGENTS.md, ANATOMY.md
    include_config: true                # Backup prometheus.yaml
    pre_graft_backup: true              # Auto-backup before any graft (not just morph)
    exempt_from_retention:              # These backup sources are never auto-deleted
      - "symbiote_morph"
      - "manual"
      - "pre_restore"
```

---

## STEP 15 — UPDATE /symbiote TELEGRAM COMMAND FOR MORPH + BACKUP:

The handler from Session A registered subcommands for `<problem>`,
`approve`, `graft`, `status`, `abort`, `history`. Session B ADDS:

```python
# Inside _cmd_symbiote, extend known_subcommands and dispatch:
known_subcommands = {
    "approve":  self._symbiote_approve,     # existing
    "graft":    self._symbiote_graft,       # existing
    "morph":    self._symbiote_morph,       # NEW: stage candidate
    "swap":     self._symbiote_swap,        # NEW: execute hot swap
    "status":   self._symbiote_status,      # existing
    "abort":    self._symbiote_abort,       # existing
    "history":  self._symbiote_history,     # existing
    "backups":  self._symbiote_backups,     # NEW: list backups
    "restore":  self._symbiote_restore,     # NEW: restore from backup
    "backup":   self._symbiote_manual_backup, # NEW: create manual backup
}

async def _symbiote_morph(self, update, context):
    """Stage a blue-green candidate with the current graft applied.

    Responds with MorphReport summary:
    - Backup created: v{N} ({size}MB)
    - Candidate staged at: {path}
    - Tests: {passed}/{total} passing
    - Ready to swap: Yes/No
    - "Reply /symbiote swap to execute the hot swap."
    """
    ...

async def _symbiote_swap(self, update, context):
    """Execute the hot swap after user confirms.

    Approval flow MUST mirror /gepa run / /symbiote approve:
    asyncio.create_task spawns a background task; the task awaits
    queue.request_approval(...) and on APPROVED runs MorphEngine.execute_swap.

    Pre-swap message:
    "⚠️ This will:
     1. Stop the daemon (~2-5s downtime)
     2. Replace live code with the candidate
     3. Restart the daemon
     4. Auto-rollback if health check fails within 60s

     Backup v{N} is ready if anything goes wrong.

     /approve {request_id} to proceed."
    """
    ...

async def _symbiote_backups(self, update, context):
    """List available backup snapshots.

    📦 Backup Vault (7 snapshots)
    v7 — 2026-04-21 14:30 — symbiote_morph — 2.1MB — tests: passing
    v6 — 2026-04-20 09:15 — manual — 1.9MB — tests: passing
    v5 — 2026-04-19 16:00 — pre_restore — 1.8MB — tests: passing
    ...
    """
    ...

async def _symbiote_restore(self, update, context):
    """Restore from a backup snapshot.

    /symbiote restore        → restore most recent
    /symbiote restore v5     → restore specific version
    /symbiote restore dry    → show what would change without doing it

    Always creates a pre-restore backup first.
    Trust Level 1 — must request approval via ApprovalQueue.
    """
    ...

async def _symbiote_manual_backup(self, update, context):
    """Create a manual backup snapshot.

    /symbiote backup                    → backup with auto description
    /symbiote backup "before experiment" → backup with custom description

    Manual backups are exempt from retention cleanup.
    Trust Level 2 — backups are read-only on the source tree, no
    approval needed.
    """
    ...
```

⚠️ **DO NOT** modify the existing five Session-A subcommands or any
unrelated `/start`, `/status`, `/help`, `/gepa`, etc. handler.

---

## STEP 16 — TESTS:

Create: `tests/test_backup_vault.py` (~150 lines)
- Create snapshot produces valid tarball.
- Snapshot contains expected files (`src/`, `tests/`, `config/`).
- Snapshot excludes `.git/`, `__pycache__/`, `~/.prometheus/wiki/`, `~/.prometheus/symbiote/`.
- Restore from snapshot replaces live files correctly.
- Pre-restore backup created automatically.
- Dry-run restore shows diff without modifying files.
- Retention policy deletes oldest, keeps `manual` + `symbiote_morph`.
- Version numbers auto-increment.
- Manifest.db tracks all snapshots.
- Corrupted tarball detected and reported.
- Restore to non-existent backup_id fails gracefully.
- Multiple backups on same day all retained.

Create: `tests/test_morph.py` (~150 lines)
- Candidate staged in correct directory.
- Candidate is a complete copy (can import prometheus from it).
- Graft applied to candidate, not live.
- Full test suite runs against candidate.
- `MorphReport.ready_to_swap` reflects test results.
- Backup created before candidate staging.
- Swap moves candidate to live.
- Health check watchdog runs after swap.
- Auto-rollback triggers on health check failure.
- `.pre_swap` dirs created and cleaned up correctly.
- Failed candidate preserved in `post_mortem`.
- **`_detect_daemon_manager` returns "systemd" when systemctl reports active**
  (use monkeypatch + a fake `subprocess.run`).
- **`_detect_daemon_manager` returns "pidfile" when systemctl is missing
  but `~/.prometheus/daemon.pid` is live.**
- **`_detect_daemon_manager` returns "pkill" when neither is available.**
- **`_detect_daemon_manager` does NOT hang** when `systemctl` is missing
  (timeout-bound, return within 5s).
- Path traversal guards on swap directories (resolve before prefix check).

Update: `tests/test_symbiote_coordinator.py` (~50 added lines)
- Coordinator state machine extends with MORPH phases:
  AWAITING_GRAFT_APPROVAL → MORPHING → AWAITING_SWAP_APPROVAL →
  SWAPPING → HEALTH_CHECK → COMPLETE | ROLLED_BACK.
- Restore via coordinator works end-to-end.
- Backup listing returns correct snapshots.

Add to: `tests/test_wiring.py`
- `MorphEngine` initializes with real `BackupVault`.
- `BackupVault` initializes with correct paths.
- `/symbiote morph`, `swap`, `backup`, `backups`, `restore` handlers
  registered.
- Coordinator's `SymbiotePhase` enum includes the new MORPH/SWAP values.

Total new test lines: ~400.

---

## STEP 17 — REGRESSION:

After all changes:
- All existing tests pass: `python3 -m pytest tests/ -v` (1,678 + ~30 new
  expected — was 1,678 at end of Session A).
- All existing `/` commands still work — see EXISTING TELEGRAM COMMANDS
  list at top of this prompt.
- Text messages still get responses.
- Media handling still works.
- SENTINEL still works.
- Profiles system still works.
- No new files in `SecurityGate.denied_paths`.
- `ExfiltrationDetector` doesn't flag local backup tarballs as exfil.
- Backup vault directory created on first use.
- Daemon restart doesn't lose SYMBIOTE session state (already verified
  in Session A's coordinator persistence test — re-verify after the
  MORPH-phase additions).

---

## File Plan (Session B only)

New files:
- `src/prometheus/symbiote/backup_vault.py` (250 lines) — `BackupVault`, `BackupSnapshot`, `RestoreResult`
- `src/prometheus/symbiote/morph.py` (350 lines) — `MorphEngine`, `MorphReport`, `SwapResult`, `_detect_daemon_manager`
- `tests/test_backup_vault.py` (150 lines)
- `tests/test_morph.py` (150 lines)

Modified:
- `src/prometheus/symbiote/coordinator.py` — extend `SymbiotePhase` enum + state-machine methods
- `src/prometheus/gateway/telegram.py` — add 5 new `/symbiote` subcommands
- `config/prometheus.yaml` and `.default` — add `symbiote.morph` and `symbiote.backup` sections
- `tests/test_wiring.py` — `MorphEngine` + `BackupVault` wiring checks
- `tests/test_symbiote_coordinator.py` — extend state-machine tests
- `PROMETHEUS.md` — add MORPH/BACKUP documentation

Total: **~900 new lines + ~150 modified**.

---

## SecurityGate Integration Summary

| Phase | Action | Trust Level | Rationale |
|-------|--------|-------------|-----------|
| Scout | GitHub API search | 2 (auto) | Read-only, no side effects |
| Scout | README fetch | 2 (auto) | Read-only |
| Scout | LLM query generation | 2 (auto) | Internal, no external effect |
| Harvest | git clone | 1 (approve) | Downloads external code to disk |
| Harvest | File reads in sandbox | 2 (auto) | Reading already-cloned files |
| Harvest | DangerousCodeScanner | 2 (auto) | Security scan is always auto |
| Graft | File writes to src/ | 1 (approve) | Modifying Prometheus source |
| Graft | Test execution | 2 (auto) | Running pytest is safe |
| Graft | PROMETHEUS.md update | 1 (approve) | Modifying project documentation |
| Morph | Create backup snapshot | 2 (auto) | Backup is always safe |
| Morph | Stage candidate | 1 (approve) | Creates full copy of source |
| Morph | Run tests on candidate | 2 (auto) | Testing candidate is safe |
| Morph | Execute hot swap | 1 (approve) | Replaces live running code |
| Morph | Auto-rollback | 3 (autonomous) | Broken daemon can't ask permission |
| Backup | Create manual backup | 2 (auto) | Backup is always safe |
| Backup | Restore from backup | 1 (approve) | Replaces live running code |
| Backup | List/view backups | 2 (auto) | Read-only |

---

## Verification

```bash
# All tests pass
python3 -m pytest tests/ -v --tb=short

# Symbiote-specific tests
python3 -m pytest tests/test_license_gate.py tests/test_github_search.py \
    tests/test_scout.py tests/test_harvest.py tests/test_graft.py \
    tests/test_backup_vault.py tests/test_morph.py \
    tests/test_symbiote_coordinator.py -v

# Integration on Mini:
python3 -m prometheus daemon --debug

# BACKUP VAULT TESTS (via Telegram):
# 1. /symbiote backup "manual safety checkpoint"
#    → Should create manual backup, report version and size
# 2. /symbiote backups
#    → Should list all backups with version, date, source, size, test status
# 3. /symbiote restore dry
#    → Should show what would change without restoring
# 4. /symbiote restore v{N}
#    → Should create pre-restore backup, then restore, then run tests

# MORPH TESTS (via Telegram, only AFTER a Session-A graft is complete):
# 5. /symbiote morph
#    → Should create backup, stage candidate, run tests, report ready_to_swap
# 6. /symbiote swap
#    → Should ask for approval via the queue, then execute hot swap
#    → Should report health check status

# ROLLBACK TEST (critical — test this carefully):
# 7. Manually break something in the candidate before swap:
#    → Edit a file in the candidate staging area to cause an import error
#    → /symbiote swap → /approve {id}
#    → Health check should fail → auto-rollback should trigger
#    → Prometheus should be back on the previous version
#    → Telegram should report: "Rolled back to v{N}. Reason: ..."
#    → /symbiote backups → failed candidate preserved in post_mortem

# DAEMON-MANAGER DETECTION:
# 8. Start MorphEngine with no systemd unit, no PID file:
#    → _detect_daemon_manager() returns "pkill" within 5s, no hang
# 9. Touch ~/.prometheus/daemon.pid with the live daemon PID:
#    → _detect_daemon_manager() returns "pidfile"

# SECURITY:
# 10. Verify backup vault exists (ls ~/.prometheus/symbiote/backups/ → has tarballs)
# 11. Verify .pre_swap dirs cleaned up after successful swap
# 12. Verify CANDIDATE_ROOT path-traversal guard rejects ../.. inputs

git add -A && git commit -m "GRAFT-SYMBIOTE Session B: BackupVault + MorphEngine + blue-green hot swap"
```

---

## Estimated Time

| Step | What | Est. Time |
|------|------|-----------|
| 12 | BackupVault | 30 min |
| 13 | MorphEngine (incl. _detect_daemon_manager) | 50 min |
| 14 | Morph + backup config | 5 min |
| 15 | Telegram subcommands | 25 min |
| 16 | Tests | 40 min |
| 17 | Regression | 10 min |
| **Total** | | **~2.5 hours** |

---

## What This Sprint Does NOT Do (Future)

- **Autonomous swap without approval.** The swap always requires user
  confirmation in V1. A config flag (`morph.auto_swap: true`) is reserved
  for future use once the pipeline is proven. The ONLY autonomous action
  is rollback on health check failure.
- **Multi-language harvesting.** V1 is Python-only. Rust/JS/Go support is
  a future extension.
- **Dependency auto-installation.** If harvested code needs pip packages,
  SYMBIOTE flags them in the report but does NOT auto-install. User must
  approve.
- **Cross-repo composition.** V1 harvests from one repo at a time.
  Composing solutions from multiple repos is a future extension.
- **Fine-tuning data generation.** Successful SYMBIOTE sessions could
  feed the fine-tuning flywheel. Not wired in V1.
- **Reverse symbiosis.** Prometheus contributing code back to upstream
  repos (PRs). Out of scope.
- **Distributed swap.** V1 assumes single-machine deployment. Coordinating
  swaps across multiple machines is future work.
- **Incremental backup.** V1 creates full snapshots every time. Delta /
  incremental backups would save disk but add complexity.
- **Backup encryption.** V1 stores tarballs unencrypted. Encryption is a
  future hardening step.
- **Scheduled backups.** V1 only creates backups on-demand. Wiring into
  SENTINEL for nightly backups is a natural extension.

---

## Donor Lineage

| Concept | Source | What Was Adapted |
|---------|--------|-----------------|
| Provenance headers | CONTRIBUTING.md convention | Header format extended with "Harvested via SYMBIOTE" |
| GRAFT additive-only pattern | PROMETHEUS-GRAFT.md | Phases 3 & 4 follow same rules |
| DangerousCodeScanner | `prometheus.security.code_scanner` | Reused for harvest scanning |
| SecurityGate trust levels | Sprint 4 | Phase-specific trust classification |
| Self-evolution concept | Hermes Self-Evolution (NousResearch) | Inspiration for the scout-to-graft pipeline (clean-room build) |
| Blue-green deployment | Industry standard (AWS, k8s) | Adapted for single-machine Python daemon swap |
| Health check watchdog | Prometheus `heartbeat.py` | Pattern reused for post-swap monitoring |
| Backup before destructive ops | Key learning from config drift incident | Formalized as BackupVault with retention policy |
| Approval gate pattern | `/gepa run` (SUNRISE Session B) | `/symbiote swap` mirrors the same `asyncio.create_task` + `queue.request_approval` shape |

---

"Extract modules, not monoliths. Adapt interfaces, not implementations.
Test at integration boundaries. Attribute everything. Always have a backup."
— Prometheus Extraction Principles
