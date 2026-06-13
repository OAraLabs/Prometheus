# Coding-API follow-ups — Survey + Plan (stop + diff routes)

**Branch:** `feat/coding-api-stop-diff` (off `main @ 625c4cc`).
**Origin:** the gaps Beacon Phase C flagged (`docs/CODING-SURFACE-PHASE-C.md`) — the Beacon coding
surface has a disabled Stop button and a diff-stat-only artifact view because the daemon exposes
neither a stop route nor a full-diff route. Both are thin wrappers over machinery that already
exists; this closes them.

## Findings (cited)

- **`BackgroundTaskManager.stop_task(task_id)`** ([manager.py:291](src/prometheus/tasks/manager.py))
  already does the right thing: SIGTERM → 3 s → SIGKILL on the subprocess (a coding run is a
  `local_agent` shell task), marks `killed`, persists, emits completion. Raises `ValueError` when
  the task isn't running. **No REST route exposes it.**
- **Artifact diff range:** `session._commit_artifact` makes exactly ONE `--allow-empty` commit on
  branch `coding/<id>` and reports `git diff --stat HEAD~1..HEAD`
  ([session.py:156-163](src/prometheus/coding/session.py)). So the FULL diff is
  `git diff HEAD~1..HEAD` in the sandbox root — identical range, guaranteed consistent with the
  reported `diff_stat`.
- **Sandbox location:** API-launched runs use the default parent `get_data_dir().parent / "coding"`
  = `~/.prometheus/coding/` ([__main__.py:733](src/prometheus/__main__.py)); the API path does not
  pass `--sandbox-parent`. The run's `sandbox_root` is in the subprocess's final JSON report
  (parsed from the task's output file). It is the daemon's OWN trusted output, but the diff route
  will still **validate** `sandbox_root` resolves under `~/.prometheus/coding/` and is a git dir
  (defense in depth) and run git via the list form (`git -C <path> …`, never a shell string).

## Plan (additive; two routes + tests)

**`POST /api/code/{task_id}/stop`** — `get_task_manager().stop_task(task_id)`. 404 unknown task;
409 when not running (`ValueError`); else `{task_id, status}` (status `killed`). Bearer-gated like
all `/api/*`.

**`GET /api/code/{task_id}/diff`** — parse the run report from the task's output file; if none yet
(still running) → `{ready: false, diff: ""}`. Else validate `sandbox_root` is under
`~/.prometheus/coding/` + a git repo, run `git -C <root> --no-pager diff HEAD~1..HEAD`, return
`{ready: true, branch, diff, truncated}` (cap ~256 KB, matching the files-preview cap). 404 unknown
task; 422 if the report has no usable sandbox_root.

A small `_load_coding_report(output_file)` helper (last balanced JSON object in the output) is
shared by the existing GET status route's tail and the new diff route.

## Tests (`python3 -m pytest`)

Extend `tests/test_api_code.py`: stop calls `stop_task` and returns killed / 404 / 409; diff
against a REAL tiny git repo (one base commit + one artifact commit) returns the diff text, rejects
a `sandbox_root` outside the coding dir, and returns `ready:false` when no report exists yet.

## Merge gate

Phase tests + full suite green (`python3 -m pytest`), squash-merge, daemon restart, smoke
(REST 6/6 + a live stop and a live diff against a real coding run), then the Beacon wiring
(separate branch/PR).
