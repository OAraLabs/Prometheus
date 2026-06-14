# Coding-task restart survival — Phase 0 survey (HARD-STOP CHECKPOINT)

**Branch:** `chore/coding-task-restart-survival` (off `main @ 0b6381f`, clean tracked tree, 0/0 vs
origin). **Date:** 2026-06-13.

## Verdict: WORLD 1 — completed coding tasks ARE persisted durably in tasks.db.

The fix is the SMALL one (the durable row exists; the lookup doesn't consult it). NOT report-on-disk
rehydration-from-scratch — though the on-disk report stays the source of truth for the diff content,
exactly as #37 established.

## Citations

**Q1 — `resume_running` rehydrates only RUNNING tasks; completed are dropped.**
`manager.resume_running` calls `self.store.list(status="running")` and loops only those: file_watch/
poll get re-watched, other (process) tasks get reaped to `failed`/`daemon_restart`. There is **no
branch for `completed`/`killed`/`failed`** — they are never read back into `_tasks`.

**Q2 — the report on disk + what a rehydrated task needs.**
A coding run's report is the subprocess stdout captured to `TaskRecord.output_file`; the `/diff`
route (#37) already parses `sandbox_root` from it via `_load_coding_report(output_file)`. The fields
a rehydrated completed task needs — `id`, `status` (terminal), `output_file` (→ the report →
`sandbox_root`), `ended_at`/`created_at`, `return_code`, `error` — are ALL columns on the durable
`tasks` row (`store.py` schema: id/type/status/description/cwd/output_file/command/… + timestamps).
So the durable row supplies the record; the on-disk report supplies `sandbox_root` for the diff. No
new schema needed.

**Q3 — tasks.db has durable rows for completed tasks (THE decider).**
`_watch_process` on subprocess exit sets `task.status = "completed" if rc==0 else "failed"`,
`task.ended_at`, then **`self._persist(task)`** → `store.upsert(record)` → `INSERT OR REPLACE INTO
tasks` (`store.py:72`). Rows are never deleted on completion (`delete` is a separate explicit method).
`TaskStore.get(task_id)` (`store.py:83`) returns any task by id regardless of status. **⇒ Completed
coding tasks are durably persisted and retrievable; they are simply absent from the in-memory
`_tasks` map after a restart.**

**Q4 — every `/api/code` read route that 404s post-restart** (all funnel through `get_task`):
- `GET /api/code/{id}` (server.py:1029): `get_task(id)` → None → 404.
- `POST /api/code/{id}/stop` (server.py:1058): `get_task(id) is None → 404`, then `stop_task` →
  `_require_task` (also reads only `_tasks`).
- `GET /api/code/{id}/diff` (#37): `get_task(id)` → None → 404 (never reaches the report parse).

## Selected fix (World 1): lazy store-fallback in the task lookup

Add one private helper `_load_task(id)`: in-memory first, then `self.store.get(id)`, caching the
store hit into `_tasks`. Route `get_task` and `_require_task` through it. This rehydrates a
completed/killed task into `_tasks` **on demand**, so all three routes resolve, and `stop_task` on
a now-terminal, process-less task hits its existing `process is None and status in
TERMINAL_STATUSES → return task` branch (correct terminal state, not 404).

**Why lazy `get_task` fallback, not eager `resume_running` of all terminal states** (a refinement of
the literal instruction, same observable outcome): eager rehydration loads EVERY historical
completed/failed/killed row into `_tasks` on every startup — unbounded growth as runs accumulate.
The lazy fallback loads only the tasks actually requested, is bounded, and changes no startup
behavior. It still "rehydrates completed/killed states into `_tasks`" — just on first access. If the
owner prefers the eager form, it's a one-line swap; flagging the choice, not assuming silently.

**Defense-in-depth preserved:** I touch only the manager's lookup. The `/diff` route's path
validation (#37 — `sandbox_root` resolves under `~/.prometheus/coding`, is a git repo, git via list
form) is untouched.

## Out of scope (flag, don't fix) — for the PR follow-ups
- Teacher-escalation traces may share the in-memory pattern → survey-flag for restart survival.
- `CostTracker.record()` orphan at daemon.py:228 still open.
- F4 schema-tax now measurable via F1 usage rows.

Proceeding to Phase 1 (no contradiction with the anticipated World 1).
