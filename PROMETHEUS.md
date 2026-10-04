# Prometheus

## Project Rules
- **NEVER work in the deploy clone (`~/prometheus-deploy`).**
  See "Two checkouts" below — this one has cost real work twice.
- **Commit from a per-session WORKTREE, not from `~/Prometheus` itself.**
  Sessions run concurrently and share that tree; it has been switched to
  another branch mid-session, leaving a second session's edits on the wrong
  one. The pre-commit hook refuses commits from the shared checkout —
  worktrees are exempt. See "Three trees" below.

  ```bash
  git worktree add .claude/worktrees/<name> -b <branch> origin/main
  ```
- Python 3.11+, package managed with uv
- All imports use `from prometheus.` prefix
- Config lives at config/prometheus.yaml, loaded via prometheus.config
- Run tests: uv run pytest tests/ -v
- All donor code has provenance headers (Source, License, Modified)
- Do not modify files in reference/
- Stage files BY NAME. `git add tests/` once swept three unrelated untracked
  files into a provider PR.

Every rule is here; runbooks, history, examples and reference are in [docs/project/](docs/project/README.md).

## Three trees — which one you are in matters

| Path | What it is | May you commit from it? |
|---|---|---|
| `.claude/worktrees/<name>` | per-session worktree — branch, commit, PR from here | **yes** |
| `~/Prometheus` | shared dev checkout — read, review, `git worktree add` | **only with the override below** |
| `~/prometheus-deploy` | ff-only mirror of `origin/main`; **the tree the daemon runs** | **no** |

The shared checkout is a *speed bump*, not a wall — solo work, a rebase, a
hotfix are all legitimate there. Say so explicitly and it is recorded rather
than hidden:

```bash
PROMETHEUS_ALLOW_DEV_COMMIT=1 git commit ...
```

The deploy clone is updated one way only:

```bash
git -C ~/prometheus-deploy fetch origin && git -C ~/prometheus-deploy merge --ff-only origin/main
```

Once the daemon runs from a **managed venv** (its unit carries the drop-in `scripts/deploy.sh` writes), deploy with the script instead.

**Merging a PR and deferring the deploy is two steps, not one.**

Before restarting the daemon, always check the clone is both clean and equal
to `origin/main` — `git -C ~/prometheus-deploy status --short` and
`git -C ~/prometheus-deploy log origin/main..HEAD`. A clone that is *diverged*
looks the same as one that is merely *behind* until the ff fails.

Detail: [checkouts-and-deploy.md](docs/project/checkouts-and-deploy.md)

## Conventions
- FILES FOR THE USER go in ~/.prometheus/files/ (the OUTBOX). Anything saved there is published:
  indexed by GET /api/artifacts and downloadable in the user's apps (Beacon shows a download chip
  when your reply mentions the filename). Do NOT invent sibling dirs (~/.prometheus/downloads etc.)
  for deliverables — outside the outbox nothing is delivered. Working files stay in the workspace.
- New tools extend BaseTool in tools/base.py
- Security checks go through SecurityGate (permissions/)
- Tool results truncated by tool_result_max in config
- ADDITIVE ONLY: extend existing files, don't replace them

### No real infrastructure identifiers in persisted content
The test is **"does it persist?"**, not "is it committed?".

Never write real tailnet/LAN host addresses, Telegram chat ids, tokens, device
ids or account ids into: PR bodies or PR comments, commit messages, docs,
scheduled-task prompts (they are stored as `SKILL.md` on disk), audit reports,
`OAra-Brain/wiki/log.md`, memory files, or scripts under `~/.local/`.

Refer to hosts by NAME (`OAra-mini`, `oara-4090`) and sessions by NAMESPACE
(`telegram:<operator-chat-id>`). Resolve addresses at runtime from
`tailscale status` rather than writing one down. When you author a prompt for
an agent or scheduled task that will WRITE a report, put this rule in that
prompt too — it persists what it writes.

EXEMPT: range constants that are part of the logic, e.g. the CGNAT block
`100.64.0.0/10` in `security/url_guard.py` and the `100.64.0.x` literals in
`tests/test_tailnet_hop_asymmetry.py`. Those are the specification, not an
address.

Get it right the first time.

Detail: [conventions.md](docs/project/conventions.md)

## Security Philosophy

User-initiated commands via Telegram have full trust. Background and self-improvement tasks run under restricted trust with scanner verification.

### Trust Model
- User says it in Telegram → full trust, no blocks
- Background tasks (SENTINEL, AutoDream, cron) → SecurityGate applies
- External code from SYMBIOTE harvest → DangerousCodeScanner applies
- Self-improvement output (GEPA, SkillRefiner) → scanner applies
- Credentials loaded from local config files → always allowed
- Network commands (pip, curl) initiated by user → always allowed

### Origin classification
The trust origin is derived from `LoopContext.session_id`:
- `telegram:<chat_id>`, `slack:<channel>`, `cli`, `web` → **user**
- `system`, `None`, SYMBIOTE/GEPA/SENTINEL UUIDs → **system**


Default for unrecognized values is `system` (the safer classification).

## Security
Shared security utilities live in `src/prometheus/security/`.

- `SecurityGate` (`permissions/checker.py`): Takes an `origin` parameter: `user` skips ExfiltrationDetector and the network/install approve-patterns; `system` applies the full restriction set. Always- blocked patterns (`rm -rf /`, `mkfs`, fork bomb), `denied_commands`, `denied_paths`, and the write_file workspace gate fire in BOTH origins.

Detail: [security.md](docs/project/security.md)

## Security Conventions

### Path Traversal Defense
Always resolve paths before prefix-checking. Never check prefix on the raw input string.

## Self-Improving Loop (SUNRISE)

- A failing hook does not block subsequent hooks.
- GEPA: Operates ONLY on `~/.prometheus/skills/auto/`. Never touches manual skills.

Detail: [sunrise.md](docs/project/sunrise.md)

## SYMBIOTE (Sessions A and B)

- `license_gate.py` — `LicenseGate` / `LicenseCheck` / `LicenseVerdict`. Hard blocks GPL/AGPL/SSPL/BUSL and unknown licenses.
- New code should import from `prometheus.security.code_scanner` directly.
- `github_search.py`: Token from `symbiote.github_token` config or `PROMETHEUS_GITHUB_TOKEN` env; never logged.
- BackupVault: Exempt sources (never auto-deleted): `manual`, `symbiote_morph`, `pre_restore`.
- `restore_snapshot(backup_id, dry_run=False)` ALWAYS creates a `pre_restore` safety backup first.
- Auto-rollback is the ONE autonomous (Trust Level 3) action — a broken daemon can't ask permission to fix itself.

Detail: [symbiote.md](docs/project/symbiote.md)

## WEAVE and WEAVE-PRESS

- `youtube_transcript`: Read-only unless `save_to` is set; all errors return `ToolResult(is_error=True)` — never raises.
- Background/automated sessions (SENTINEL, GEPA, AutoDream, smoke-tests, cron) get **no** suggestions.
- Install routes through the same `ApprovalQueue` used by `/gepa run` and `/symbiote graft`.
- `SkillRegistry.reload_user_skills()` re-scans `~/.prometheus/skills/` and merges new or updated entries (purely additive — never removes existing).

Detail: [weave.md](docs/project/weave.md)

## Managed Tasks

**Three concerns kept strictly separate:**
- **Detection** — non-LLM, event-driven.
- **Notification** — cheap, templated, model-free.
- **Re-engagement** — only when the agent must *act on* the result. `on_complete` gates re-engagement only; notification always fires.

**Durability.** On startup the daemon calls `resume_running()`: `file_watch`/`poll` watchers are re-established; orphaned process tasks (whose OS handle is gone) are reaped to `failed` (`error="daemon_restart"`) — no zombie `running` rows.

**Security.** Task launch is vetted through the **same SecurityGate as cron**,
at system trust (`evaluate("bash", command=…, origin="system")`), failing
closed. A denied command yields a `failed` record and never spawns.

`TelegramAdapter.inject_turn(session_id, content, *, provenance, is_trusted)` (generalized from `_dispatch_to_agent`) is the **one** path that injects a non-user turn into a session and runs the agent loop. Cron and a future orchestrator clarification channel are meant to converge here — **do not build a parallel re-engagement mechanism.**

**Provenance & trust** are structured fields on `ConversationMessage`
(`engine/messages.py`): `provenance` (closed enum: `user` / `cron` /
`task_supervisor` / `orchestrator`) and `is_trusted` (defaults **False** — safe
posture). These are the source of truth. The "⚠️ UNTRUSTED INPUT" banner is a
**derived rendering** applied at context-assembly (`render_messages_for_model`,
called at the model-call site in `agent_loop.py`) — never stored on the record,
so job stdout / watched-file contents reach the model fenced as data, and the
model is told not to execute instructions found inside them.

### Tools
**`session_id` + `notify_target` are resolved from the trusted execution context, never from tool arguments** (which could originate in observed content).

Detail: [managed-tasks.md](docs/project/managed-tasks.md)

## The loud-failure law (OAra Lab-wide, 2026-07-02)

**Degraded is a state that gets announced, never absorbed. No component may
catch-and-continue silently. Every daemon writes a success heartbeat; staleness
is surfaced, spoken, and shown.**

Prometheus already leans loud (silent_failure telemetry, #78 journal tracebacks); keep it that way: any new code that catches an exception and continues silently is a bug.

- The `oara_heartbeat_watcher` cron job: Do not "fix" that job by making it always exit 0 — its failure IS the feature.

### Corollary (Sprint 2, 2026-07-02): the law extends to configuration

A subsystem that is expected-enabled but dark is a failure state identical to a stale heartbeat. "Never turned on" must be a boot-time alarm, not an archaeology finding.

When adding a feature behind a config flag, either default it ON in prometheus.yaml.default or register it in the OAra manifest with `expected: false` — a flag nobody tracks is a future archaeology dig.

### Directory ownership — one writer per vault directory (Sprint 2, 2026-07-02)

Prometheus owns the wiki root — `wiki.root`, default `~/.prometheus/wiki`
(tools/patterns/projects et al.) — and NEVER
writes `~/OAra-Brain`; the Jarvis extractor (via the OAra middleware) owns the
OAra-Brain life-note directories and never writes the wiki. Vault notes carry
`writer:` frontmatter; the hourly `oara_vault_lint` cron (detection-only)
flags any file whose claimed writer doesn't own its tree — a Prometheus wiki
file claiming `writer: oara_extractor` is a violation, and vice versa.

Detail: [loud-failure-law.md](docs/project/loud-failure-law.md)
