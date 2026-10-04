# Managed Tasks — detail

Moved out of [PROMETHEUS.md](../../PROMETHEUS.md) on 2026-10-03 so that file fits the
12,000-character cap on a project instruction file. The text below is the original
section, word for word. Its rules are also kept in PROMETHEUS.md, which is
authoritative: where this copy disagrees, this copy is stale.

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
