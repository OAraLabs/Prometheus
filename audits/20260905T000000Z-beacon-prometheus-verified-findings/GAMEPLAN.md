# PROMETHEUS + BEACON AUDIT — GAME PLAN
Generated 2026-09-07 · source: read-only audit of Prometheus `3de47f5` / Beacon `a86f62a`
Evidence: `prom-verified.json` (98 daemon), `beacon-verified.json` (23 client) in this dir.
All 121 findings below are ADVERSARIALLY VERIFIED. Severity = verified, not as-filed.

## THREAT MODEL (governs every security decision)
Single-operator sovereign harness. Operator = principal, agent acts AS him.
Threat = STRANGERS ON THE NETWORK, not the agent going rogue on its own box.
- Constrain WHO CAN REACH the agent  -> FIX
- Constrain WHAT THE AGENT MAY DO for its operator -> DO NOT (unless named)
- Docstring promises confinement code doesn't give -> FIX THE DOCSTRING, don't build confinement.

## CURRENT STATE OF MAIN (drift since audit)
- Prometheus main = `6c808a7`. Audit = `3de47f5`, 2 commits behind. `#320` token-rotate-no-op ALREADY FIXED (6c808a7).
- Beacon origin/main = `365265a`. Audit = `a86f62a`, 6 commits behind. `25d429b` already shipped an asar include-list + artifact audit.
- Line numbers below are re-verified against CURRENT main where I checked them. Trust file+symbol over raw line.
- Local beacon-desktop checkout is 49 behind origin/main and 1 ahead (`f21feb2`, on branch `fix/context-meter-legible-segments`). MUST fetch+worktree off origin/main before any Beacon work.

## WORKTREE / RW STATUS (critical for execution)
- `/` and `/home/will` are READ-ONLY right now.
- Bind-mounts that ARE writable: `~/Prometheus`, `~/.prometheus`, `~/prometheus-deploy`, `~/.cache`, `~/projects`.
- `~/.claude` is READ-ONLY -> cannot create `.claude/worktrees/`. Use `~/Prometheus/.claude/worktrees/` (that path IS writable) or a worktree dir under `~/projects`.
- VERIFIED this session: `git worktree add` works from `~/Prometheus`. Committing works (pre-commit hook allows worktrees).
- write_file tool was misbehaving this turn; files created via bash heredoc.

---
# PRIORITY ORDER
Sequenced by risk-retired-per-unit-work. Audit Section Five order, re-mapped against
Will's brief classification. Each phase = one PR-able chunk. Bugs (capability-restoring)
can go ANY time and in parallel — they have no agentic cost.

## PHASE 0 — Bugs that RESTORE capability (do first, freely, parallelizable)
These are NOT constraints. They fix things that are simply broken. No threat-model tension.
- [x] P0.1 lcm_expand.py + lcm_expand_query.py call store methods that don't exist (`get`, `get_by_id`). Add them to the store or repoint to `get_by_id` that exists. Tools are DEAD today. [by:agent — Prometheus #332, merged 2026-09-08]
- [x] P0.2 lcm_engine.py:42 config path resolves one dir ABOVE repo -> operator's compaction config NEVER read. Fix the parents[] index. (Today's values == defaults, so it's silent.) [by:agent — Prometheus #340, merged 2026-09-09]
- [x] P0.3 daemon.py:74 daemon ignores every env override the CLI honours (PROMETHEUS_MODEL, PERMISSION_MODE, TRUST_LEVEL, *_FILE secrets). Unify the two config loaders. Security knob dark on the surface that matters. [by:agent — Prometheus #343, merged 2026-09-09]
- [x] P0.4 agent_loop.py:1937 (+2147, +2153) four turn-exit paths commit tool_use with no tool_result -> hard 400 bricks the next cloud message on that session. Add a pairing invariant/helper. [by:agent — Prometheus #345, merged 2026-09-09]
- [x] P0.5 agent_loop.py:1117 router mutates LoopContext SHARED across web sessions -> concurrent turns overwrite provider (mis-billing) + iteration cap flips mid-turn. Make per-run/immutable. [by:agent — Prometheus #347, merged 2026-09-09]
- [x] P0.6 evals/runner.py:213 judge outage/exception -> every task records PASS, exit 0. A failed judge must FAIL, not pass. [by:agent — Prometheus #344, merged 2026-09-09]
- [x] P0.7 daemon.py:432 config pins never correct a falsy/missing value while doctor reports pin active. [by:agent — Prometheus #342, merged 2026-09-09]
- [x] P0.8 session.py:653 trim() reuses durable turn_index -> post-restart compaction interleaves old+new. Advance the offset. [by:agent — Prometheus #336, merged 2026-09-09]
- [ ] P0.9 session.py:747 /reset undone by rehydrate_if_cold (clear() == the 'cold' shape restore() accepts).
- [ ] P0.10 image_generate.py:916 output allow-list = whole state dir -> can overwrite lcm.db/tasks.db. Narrow to images subdir.
- [ ] P0.11 registry.py:367 override/backend local providers ignore probe `vision:True` -> approved image kills turn w/ UnsupportedContentBlock.
- [ ] P0.12 model_router.py:855 keyless cloud override raises every turn; loop silently serves primary while status reports override. Make it loud.
- [ ] P0.13 model_router.py:1111 override/fallback/rule providers get tier-FULL QwenFormatter, bypassing tier selection.

## PHASE 1 — Close the two open front doors (smallest change, largest exposure)
Both open by DEFAULT on one install path each. Perimeter, not agentic cost.
- [ ] P1.1 config/prometheus.yaml.default:799 — `api_token:` (null) reads as "deliberately open" -> template copy boots UNAUTHENTICATED REST+WS+OpenAI-compat (bash reachable) on 0.0.0.0. COMMENT OUT the key so unset != deliberate. STILL LIVE on main (verified).
- [ ] P1.2 daemon.py:~1992 — refuses to serve when token persist FAILS, instead of serving OPEN. Today: one ERROR line then `web auth: OPEN`. (api_token.py was touched by #320 but THIS path unchanged.)
- [ ] P1.3 server.py:160 — CORS `allow_origins=["*"]` on 0.0.0.0 turns token-less mode into browser-reachable RCE. Tighten.
- [ ] P1.4 BEACON index.ts:474 — import the browser pane's `isAllowedNavigation` allowlist in front of shell.openExternal. Model-authored relative href -> file:// -> OS launch in packaged builds. The allowlist + a smoke asserting file:///x refused ALREADY EXIST (browser-pane.ts, browser-smoke.ts); shell window just doesn't import it.
- [ ] P1.5 BEACON MessageRenderer — add urlTransform dropping relative + non-web hrefs (the other half of P1.4).

## PHASE 2 — Make authorisation STRUCTURAL in all three gateways
The #202 Telegram fix, carried to Slack+Discord by construction, not per-handler.
- [ ] P2.1 slack.py:1307 — Bolt GLOBAL middleware so all ~49 slash handlers pass the channel allowlist. Slash cmds are workspace-global (any member incl. Slack Connect externals). STILL LIVE.
- [ ] P2.2 discord.py:863 — interaction check INSIDE the registration callback (covers all ~43 command families at once).
- [ ] P2.3 discord.py:1446 — CRIT: app commands bypass every allowlist; `ops approve always` from any reachable user = persisted SecurityGate grant = remote RCE. (Same registration callback as P2.2.)
- [ ] P2.4 gateway/config.py:125 — Discord DMs unconditionally allowed, no user allowlist (justified by an inverted Telegram premise). ADD DM user allowlist.
- [ ] P2.5 gateway/config.py:99 — Slack empty allowed_channels = allow-all, no boot refusal (Telegram got the refusal; Slack didn't).
- [ ] P2.6 TESTS — per-adapter test walking every registered command from a DISALLOWED origin, asserting refusal. Make the property enforced, not remembered.

## PHASE 3 — Give CI the four checks it lacks + un-skip security floors
- [ ] P3.1 ruff (no config exists) — incl. ASYNC ruleset (makes the event-loop-blocker class visible).
- [ ] P3.2 mypy on a GROWING allowlist (445 errors today; don't gate all at once).
- [ ] P3.3 coverage floor (75.6% today; daemon.py is 16.3%).
- [ ] P3.4 pip-audit / osv (21 daemon advisories; Electron 33 EOL w/ 33).
- [ ] P3.5 tests/test_bash_write_floor.py:51 + docker/bwrap sandbox — UN-SKIP on Linux runner. Confinement regressions currently merge green.
- [ ] P3.6 Fix the 1 real undefined-name (full-text search escaper).
- [ ] P3.7 BEACON — no PR gate at all. Land a workflow turning 74 smokes into a real gate. (release.yml:88 uploads NOTHING — literal cross-platform manifest paths under `set -e`.)

## PHASE 4 — Perimeter + tool bugs (GO list, no agentic cost)
- [ ] P4.1 denied_prune.py:48/52 — globs use pathlib semantics, gate uses fnmatch -> `/*/.ssh` matches NOTHING. The #214 floor is silently absent. Fix the MATCHING. DO NOT expand the deny list.
- [ ] P4.2 web_fetch.py:96 — check POST-redirect host (httpx follows unchecked) + 100.64.0.0/10 tailnet is not treated private + DNS resolved twice (rebinding TOCTOU). NO domain allowlist.
- [ ] P4.3 download_file.py:202 — normalise percent-encoded traversal (`unquote` then re-check). Do NOT restrict which URLs.
- [ ] P4.4 bash.py:143/206 — strip key/token/secret-shaped vars from tool env. PATH/HOME/LANG stay. README already claims this; make code match the standing claim.
- [ ] P4.5 hooks/executor.py:213 — model-controlled JSON spliced unquoted into `/bin/bash -lc`. Shell injection from tool input via the DOCUMENTED hook example.
- [ ] P4.6 tasks/manager.py:180 — ANTHROPIC_API_KEY embedded in persisted task `command`, echoed by REST + task_get. Don't persist the secret in the command string.
- [ ] P4.7 web/server.py:2062 — POST /api/mcp/servers spawns arbitrary cmd, no gate/scanner/approval/master-switch. One POST naming /bin/sh = RCE as daemon user. GATE IT.
- [ ] P4.8 dashboard.py:95 — binds 0.0.0.0 unauth, model-chosen port+HTML, server_close() never called (port bound for daemon life), shutdown() blocks loop. Bind localhost, add config gate, call server_close().
- [ ] P4.9 audit.py:147 — raw command persisted unredacted, fed back to model by audit_query. Redact secret-shaped tokens at the store boundary.
- [ ] P4.10 web/server.py:226 — enrolled device tokens == global token on every route except /api/devices. Scope device tokens.
- [ ] P4.11 cli/token.py — ALREADY FIXED by #320 (verify, then strike).

## PHASE 5 — First-run path end to end (every new operator walks this)
- [ ] P5.1 pyproject.toml:121 + generate_identity.py:16 — templates/ NOT in wheel, resolved 4 parents above __file__ -> `oara setup` CRASHES on every non-editable pip install (sdist has it, wheel doesn't). STILL LIVE (verified).
- [ ] P5.2 generate_identity.py:81 — _detect_gpu catches only FileNotFoundError -> crashes on multi-GPU/MIG/[N/A]/hung nvidia-smi. Catch broadly.
- [ ] P5.3 cli/init.py:211 — fast wizard writes `provider: lm_studio`/`vllm` the factory rejects -> "Setup complete" then unbootable daemon. Map to accepted names.
- [ ] P5.4 setup_wizard.py:626 + BEACON setup-wizard.ts:240 — wizard says "leave blank to allow all users" then daemon REFUSES to start; Beacon sends no chat-id allowlist at all; only one chat id enterable (Slack/Discord allow many).
- [ ] P5.5 cli/doctor.py:119 — cloud-key check reads only process env, not the env file the daemon loads -> false ERROR + exit 1 on working cloud installs.
- [ ] P5.6 cli/migrate.py:229 — migration writes config/skills/daily-notes/cron to locations NOTHING reads, reports "migrated", then wizard overwrites config.

## PHASE 6 — One config loader; pins correct falsy/missing
- [ ] P6.1 daemon.py:74 + __main__.py:75 — resolve the TWO loaders into one (daemon applies same env overrides + secret files as CLI). Closes P0.3 class.
- [ ] P6.2 daemon.py:432 — pins correct falsy AND missing (closes P0.7).
- [ ] P6.3 lcm_engine.py:42 — point loader at real config path (closes P0.2).

## PHASE 7 — Turn-end as ONE enforced invariant + immutable run plan
- [ ] P7.1 One helper pairs every committed tool_use with a tool_result before any assistant text + render-time assertion that fails LOUD on an unanswered call. Replaces the 7 hand-rolled end-of-turn sequences (closes P0.4).
- [ ] P7.2 Replace shared mutable LoopContext with immutable per-run plan -> closes router race (P0.5), tier-bump leak (agent_loop.py:481), identity-line leak in one move.

## PHASE 8 — Take blocking work off the single event loop (the ten blockers)
- [ ] P8.1 video_ingest/pipeline.py:122 — ffmpeg/Whisper/SSIM sync up to 10 min.
- [ ] P8.2 grep.py:84 (+glob, read_file) — sync tree walks/whole-file reads, timeout can't preempt.
- [ ] P8.3 symbiote/github_search.py:189 — time.sleep up to 60s in handler.
- [ ] P8.4 web/server.py:3251 — artifact list/download SHA-256 hashes up to 1GiB sync.
- [ ] P8.5 web/server.py:4628 — blocking subprocess.run(git diff) in async route, up to 20s.
- [ ] P8.6 ws_server.py:1602 — broadcast iterates LIVE client set across await -> join/leave mid-fanout raises into turn (rollback+error frame). Snapshot first.
- [ ] P8.7 ws_server.py:239 — handler swallows every non-1009 exception -> handler bug = unexplained disconnect.
- [ ] P8.8 lsp/client.py:159 — stderr piped, never drained -> 64KiB fills, server deadlocks, every later LSP call waits full 30s timeout.
- [ ] P8.9 lsp/client.py:454 — reader loop dies at DEBUG, leaves is_alive True, CancelledError escapes except-Exception, aborts turn.
- [ ] P8.10 web/launcher.py:196 — bridge launched unsupervised, logged "started" before it runs -> bind failure invisible until shutdown.
- [ ] P8.11 ENFORCEMENT: one async rule + ASYNC ruff ruleset (ties to P3.1) keeps the class closed.

## PHASE 9 — Beacon drift against a daemon that moved on
- [x] P9.1 ChatShell.tsx:2109 — NO Stop/interrupt. Daemon has POST /api/chat/interrupt + chat_done{interrupted} for weeks; iOS already ships it. Add the button. [by:agent — Beacon #141, merged 2026-09-09]
- [x] P9.2 slash-commands.ts:76 — hand-mirrored catalog drifted into a DESTRUCTIVE LIE: says /reset is Telegram-only but it clears daemon context on web; 8 web commands missing. Derive from daemon, don't mirror. [by:agent — Beacon #140, merged 2026-09-09]
- [x] P9.3 rest.ts:23 + :238 — provenance/is_trusted dropped at wire -> machinery turns render as operator's own bubbles, untrusted 3rd-party content indistinguishable. Carry through. [by:agent — Beacon #139, merged 2026-09-09]
- [ ] P9.4 MainView.tsx:763 — phantom unread: `viewed:` written only at open, daemon watermark advances on your OWN message -> every conversation you touch shows a dot + "Reply in…" toast after you leave.
- [ ] P9.5 cache/repo.ts:79 — reconcile stamps last_active=now -> opening ANY old session promotes it to Today/top of sidebar permanently. Recency model corrupted by reading it.
- [ ] P9.6 cache/service.ts:125 — optimistic row whose WS echo is missed never retired -> permanent duplicate pending row.
- [ ] P9.7 ipc.ts:104/105 — background reconcile never reaches renderer; restored session paints stale until manual switch-away-and-back. (Daemon fans task frames as `sentinel_signal`; main tests `task_completed` -> index.ts:772 dead wire.)
- [x] P9.8 ws.ts:328 — chat_done.row_id dropped -> streamed assistant row gets no rowId -> edit-into-branch forks BEFORE last live reply. [by:agent — Beacon #142, merged 2026-09-09]
- [x] P9.9 coding.ts:221 — live rounds dedup by round_index but daemon restarts it each episode (key on seq) -> episodes 2+ silently dropped, frozen Live view. [by:agent — Beacon #143, merged 2026-09-09]
- [ ] P9.10 coding.ts:62 — terminal report parsed from 4000-char output_tail the report routinely exceeds -> successful runs land as "No report".
- [ ] P9.11 release.yml:88 — fix the upload (ties to P3.7).

## PHASE 10 — Standing credential + dependency items
- [ ] P10.1 Rotate the GLOBAL API token (issue #320: token reached a test transcript 28 Aug, not rotated). The no-op fix (#320) is LANDED, so rotation now works — DO IT.
- [x] P10.2 Revoke the live GitHub token in `~/projects/SkillForgeRecorder/.git/config` remote URL. DONE by Will 2026-09-07 (outside both repos; no plan action needed).
- [ ] P10.3 Upgrade Electron off the EOL line (33.4.11 -> 44.x; 33 advisories touch exposed surfaces).
- [ ] P10.4 Lift the 7 vulnerable Python pins (starlette×5, cryptography×6, python-multipart×3, mcp×3, idna, pydantic-settings, pytest).

## PHASE 11 — Reconcile docs with code once, then keep it mechanical
- [ ] P11.1 daemon.py:174 — docstring claims template ENABLES SENTINEL; template disables it (dark on every fresh install, no announcement). Fix prose.
- [ ] P11.2 "honest status" section wrong on 3 of 9 bullets; Beacon README drifted in the SAME commit that changed code.
- [ ] P11.3 Generate route reference + command list + config-key table FROM SOURCE. Largest drift class stops recurring.

---
# WILL'S EXPLICIT GUARDRAILS — READ BEFORE ANY SECURITY EDIT
These OVERRIDE the audit's severity labels. Where they conflict, this section wins.

## STAND DOWN — do NOT change behaviour
- permissions/checker.py:111 — workspace write boundary covers TWO tool names ON PURPOSE. README: "a speed bump, not confinement." DO NOT extend to notebook_edit/download_file/tts/youtube_transcript. FIX THE DOCSTRING to say what it does. Confinement lives in --sandbox + coding-mode cwd jail.
- daemon.py:1344 — approval queue OFF by default + only built under a Telegram branch. DO NOT make approvals mandatory or add a 2nd surface. DO make the dark state LOUD: extend the boot-time registry (covers 3 keys: web, trajectory-export, compaction) to warn when the gate can emit APPROVE with no human route. VISIBILITY, not friction.
- symbiote/* — OFF by default. Leave approval semantics alone. No new gates on a subsystem the operator hasn't enabled. (symbiote_graft.py:66, graft.py:262, morph.py:294 -> STAND DOWN.)
- message.py:140 — do NOT add a destination allowlist for OPERATOR-origin sends. If narrowing anything, narrow SYSTEM-origin only (cron/Sentinel/managed tasks) — and ASK first.

## CAREFUL — do the fix, NOT the adjacent constraint
- denied_prune.py:48 — fix the MATCHING (pathlib->fnmatch). DO NOT expand the deny list. (= P4.1)
- web_fetch.py:96 — check POST-redirect host. NO domain allowlist. (= P4.2)
- download_file.py:202 — normalise percent-encoded traversal. Do NOT restrict which URLs. (= P4.3)
- bash.py:206 — strip ONLY key/token/secret-shaped vars. PATH/HOME/LANG stay. README already claims this; match the standing claim. (= P4.4)

## GO — perimeter + bugs, no agentic cost
P1, P2, P3, P4 (minus STAND DOWN items), P5, all of PHASE 0 bugs. These are the free wins.

## BUGS — RESTORE capability, do freely
All of PHASE 0. Plus P0.4/P0.5 feed PHASE 7.

---
# EXECUTION RULES (per session)
1. NEVER commit from ~/Prometheus shared checkout or ~/prometheus-deploy. Use a worktree.
   - `~/.claude` is RO. Create worktrees under `~/Prometheus/.claude/worktrees/<name>` (RW, verified).
   - `git -C ~/Prometheus worktree add .claude/worktrees/<name> -b <branch> origin/main`
2. Branch from origin/main, NEVER main directly (push-protected; PR required).
3. Stage BY NAME. Never `git add -A`/`.`/`<dir>`.
4. Tests in a worktree pick up DEPLOY-tree code via PYTHONPATH. Use `env -u PYTHONPATH uv run pytest ...`.
5. Beacon work: `cd ~/projects/beacon-desktop && git fetch && git worktree add ../beacon-wt-<name> origin/main`. Local HEAD is 49 behind.
6. Run tests before "done"; quote the count. 6778 daemon tests, ~4m31s.
7. Commit messages say WHY + what you verified / didn't.
8. main is push-protected -> open PR, then STOP. Don't self-merge without a go.
9. After merge: ff the deploy clone, then daemon restart is a SEPARATE step (or boot-refusal arms).
   `git -C ~/prometheus-deploy fetch origin && git -C ~/prometheus-deploy merge --ff-only origin/main`
10. Each phase = one PR (or a tight cluster). Keep PRs reviewable.

# RESUME STATE
- Last updated: 2026-09-17
- Phases complete:
  - PHASE 0 — DONE, all shipped and merged.
    - P0.1 dead LCM tools — Prometheus #332 (merged 2026-09-08)
    - P0.2 config path — Prometheus #340 (merged 2026-09-09)
    - P0.3 secret env file — Prometheus #343 (merged 2026-09-09)
    - P0.4 turn-end pairing — Prometheus #345 (merged 2026-09-09)
    - P0.5 immutable run plan — Prometheus #347 (merged 2026-09-09)
    - P0.6 session titles — Prometheus #344 (merged 2026-09-09)
    - P0.7 config pinning — Prometheus #342 (merged 2026-09-09)
    - P0.8 boot-time warning — Prometheus #336 (merged 2026-09-09)
  - Beacon P9.1 Stop button — Beacon #141 (merged 2026-09-09)
  - Beacon P9.2 slash catalog — Beacon #140 (merged 2026-09-09)
  - Beacon P9.3 provenance/trust — Beacon #139 (merged 2026-09-09)
  - Beacon P9.8 row_id — Beacon #142 (merged 2026-09-09)
  - Beacon P9.9 live-rounds key on seq — Beacon #143 (merged 2026-09-09)
  - Beacon #145 step4-smoke coverage — Beacon #145 (merged 2026-09-17)
  - Prometheus #492 lsp stderr drain — Prometheus #492 (merged 2026-09-17)
  - Prometheus #494 task frame promotion — Prometheus #494 (merged 2026-09-17)
  - Beacon #131 audit evidence — closed 2026-09-17
  - Beacon #132 umbrella — closed 2026-09-17 (all 3 verified)
  - Beacon #128 task_completed wire — closed 2026-09-17 (fixed by #494)
  - Beacon #87 smoke coverage — closed 2026-09-17 (fixed by #145)
- In flight (awaiting Will merge):
  - Beacon #146 coding-report field — CI green, merge clean
  - Beacon #147 retire-optimistic — CI green, merge clean
  - Prometheus #495 coding-report parse — CI green, merge clean
  - Prometheus #496 client_msg_id persistence — CI green, merge clean
  - oara-voice #14 token drift — CI green (fixes Prometheus #488)
  - beacon-ios #4 — needs Mac verify.sh (Can't Run on Linux)
- Next action: merge the 4 in-flight PRs (2 Prometheus, 2 Beacon), then
  ff the deploy clone + restart daemon. After that: PHASE 1 (hardening) —
  P1.1 token-in-URL, P1.2 env-file read, P1.3 path traversal, P1.4 git clone
  --config, P1.5 device token. P1.4 is paired with Prometheus #344 (session
  titles) — see that PR for the daemon half; Beacon half not yet started.
