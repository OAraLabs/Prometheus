# Handoff — the plan panel arc (Prometheus side), 2026-09-18

Named for the arc rather than the date alone: an untracked `docs/HANDOFF-2026-09-18.md` from an
earlier session already exists in some working copies, and this is a different document.

Everything below is verified against the live daemon, not recalled.

## What shipped here

| PR | merge | |
|---|---|---|
| 507 | `86d6473` | the documents instruction named `file_write`; the registered tool is `write_file` |
| 508 | `d1282ab` | plans are asked for as GFM task-list items, not a numbered list |
| 509 | `460ec0e` | the plan file is named after the **Task ID** the model is actually given |
| 510 | `b29bac0` | anyio 4.13.0 → 4.15.1, two CVEs; `main` was red on it, not a feature branch |

Deployed: `~/prometheus-deploy` on the mini at `460ec0e`, restarted 2026-09-18 16:46 EDT.

## The one idea worth carrying

**Prose in a system prompt is an interface reference with nothing type-checking it.** #507 is the
proof: the instruction said `file_write` for weeks, the tool is `write_file`, so the tree the
instruction points at was unreachable the entire time — and the only observable was an empty
directory, which is indistinguishable from disuse. `tests/test_system_prompt_tool_names.py` now
pins prompt tokens against the real registry, flagging any snake_case token that is unregistered
but whose **word set** matches a registered tool (that is the mistake that actually happens when a
module and its tool differ in order).

**And prose is a lever, whose only proof is re-running the request.** #508 pinned its checkbox
example against Beacon's parser — a good test that proves *the example parses*, not that the model
complies. The check that counts is re-issuing the original user-level request against the restarted
daemon and reading what lands on disk. Before: four numbered steps, 0 checkbox lines. After: 4.
Same for #509: a dispatched story, a request naming neither file nor convention, and
`BC-7-checklist-panel-rollout.md` appeared, keyed and tickable.

Keep the before-artifact. The #509 run **overwrote** the file #508's run had produced, so the
pre-change bytes survive only because they had been captured as a fixture first.

## Wording is load-bearing

The schema, the kanban card and Beacon's resolver all say `story_id`. **The model is never shown
that word.** The only place a story's key reaches it is the dispatch message, stamped at
`web/server.py` as `_Task ID: BC-7_`. The convention sentence therefore uses the wire's spelling,
and `_TASK_ID_LABEL` plus a test that reads that f-string **out of server.py** pins the two so
neither drifts alone. The fixture fails loud if the line is reworded or moved.

## Two things fixed that nobody had noticed

- **`documents.root` was a promise the code did not keep.** `get_documents_dir()` reads only
  `PROMETHEUS_DOCUMENTS_DIR`; `/api/documents` honours the config key `documents.root`; **both
  docstrings claimed the key worked.** They agreed only because no deployment set it — setting it
  would have told the model to write to a directory the Board cannot read, with no error anywhere.
  The prompt builders now take a resolved `documents_root` and the runtime assembler passes the
  config's value.
- **The `# Current Task State` block has no caller.** `prompt_assembler`'s docstring advertises it
  as section 8 of the prompt; nothing in `src/` ever passes `task_state=`. If board state should
  ever ride the system prompt, that is the pre-cut seam — but wiring it means reviving an
  aspirational parameter, which this codebase has been bitten by before.

## Open, deliberately

- **Plain chat is unkeyed.** A plan asked for in ordinary chat carries no task id, so the
  convention sentence is inert there. Closing it needs a mechanism that does not exist:
  `session_key` is written only at dispatch, is not unique, and the agent is never told its own
  session key in the prompt or in any model-facing tool schema. Waiting on a real need.
- **`story_id` has no UNIQUE constraint** and stays guarded rather than constrained. `PUT
  /api/stories/{pk}` will rename one onto a colliding value with no check, while `POST` rejects
  only an empty value. There is **no migration machinery for `kanban.db`** — `_apply_schema` is
  `CREATE TABLE IF NOT EXISTS` with no ALTER, no version gate and no snapshot, so editing the DDL
  is a silent no-op on an existing database.
- **`/api/documents` truncates a listing at 2000 entries and returns 200 with no flag**, and caps a
  read at 256 KiB with `truncated: true`. Beacon now surfaces both; the daemon still does not
  announce the first one.

## The operational trap

`scripts/deploy_guard.sh` refuses to start the service when `~/prometheus-deploy` is on anything
but `main`, and because it is an `ExecStartPre`, a refusal leaves the daemon **down**. It is right
to do this, and its header already records that the checkout had been left on a feature branch
three times in eleven days. Read the guard before deploying rather than discovering its rule by
tripping it: an outcome check that needs deployed code cannot run before the merge, by design.
