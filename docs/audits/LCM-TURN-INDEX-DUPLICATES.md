# LCM `turn_index` duplicates: investigation report

**Date:** 2026-09-25 (investigation), 2026-09-26 (fix) · **Code investigated:** origin/main `caa24db`
(the mini's daemon runs v0.9.3 `b2603ad`; every producer below except trim is present in both) ·
**Status:** the §5 design is implemented by the PR that adds this file. §5.7 records what was built,
where it differs from the plan, and the gates it passed.

## Data, method, constraints

- **Source.** The mini's nightly snapshot `~/.prometheus/db-snapshots/20260925T065501Z/data/lcm.db`,
  opened `mode=ro&immutable=1`. All queries ran on the mini; only aggregate counts came back. No
  message content, session ids or DB files left the mini. The live DB was never opened.
- **Deviation from the brief.** That snapshot was made by `prometheus.jobs.db_snapshot` with
  `VACUUM INTO`, not with SQLite's backup API. Making a backup-API copy would mean holding a read
  transaction on the live DB for the whole copy, which the brief rules out, so I used the nightly copy.
  I checked that it preserved rowids, which is the only way I use them: the gaps are intact
  (18,879 rows, max rowid 22,586), all 18,879 FTS rowids join to a message, and all 23
  `message_client_ids.row_id` join to a message in the same session. The backup API was used once,
  from the snapshot file into `:memory:`, for the migration dry run in §5.
- **Daemon lifetimes** come from the mini's user journal (292 `Started server process` lines since
  2026-06-27, which is where the journal starts) plus 35 `rehydrate: … restored` lines.
- **Synthetic reproductions** first ran as failing tests on `caa24db` (10 failed, the live-prompt
  control passed; see the Appendix). They are now the regression tests in
  `tests/test_lcm_turn_index_unique.py`. The scan scripts are in `docs/audits/lcm-turn-index/`; they
  contain no data.
- The investigation edited nothing in `engine/agent_loop.py` or `adapter/`, did not touch the mini's
  daemon, did not use the production model server, and started no long-running processes.

## Summary

- **Scale.** 16 of 112 sessions are affected. There are 1,210 duplicated `(session_id, turn_index)`
  pairs, covering 8,678 rows. **7,468 rows reuse an index** that an earlier row in the same session
  already had. This is the number earlier notes called "7,468 duplicate pairs"; it actually counts
  colliding rows, spread over 1,210 pairs. Duplicates run from the first day of data (05-27) to the
  last (09-24). The worst week was mid-August (88% of writes). Since the trim fix was deployed on
  09-11, it has been 10.5% of writes (832 of 7,905), and that is not falling.
- **Shape.** The duplicates are **different messages that were given the same index**
  (7,390 of 7,468). Only 78 rows are byte-identical copies.
- **Cause.** Every duplicate comes from a numbering restart: a `ChatSession` begins numbering again
  below the session's durable maximum. Six live paths do this:
  - a restart where rehydrate declines;
  - a restart through a path that never rehydrates;
  - rehydrate numbering from its 40-row window instead of from the session maximum;
  - `/reset` and `/clear`;
  - rollback and retry;
  - a failed WS turn discarding a durable mid-turn row.

  Trim was the largest producer until #446 (deployed 09-11 14:11; last collision 09-08). Only
  the restart paths have fired since: restart-from-0 as late as 09-24, window-max as late as 09-21.
- **Impact.** The **live prompt is not affected.** It comes from the in-memory list, and rehydrate
  reads by rowid (the control test passes). What *is* affected:
  - **LCM compaction.** The summarizer model is sent two conversations zipped together
    (A0, B0, A1, B1, …): 442 of 2,472 leaf summaries (18%) were built from out-of-order batches. The
    compactor also summarizes new rows while keeping old ones as the "fresh tail" (220 rows). Those
    summaries reach the model through the `lcm_*` tools (103 calls).
  - **Golden-trace training exports.** Their contexts are resolved in `turn_index` order.
- **Fix.** Keep `turn_index` as the durable *prompt position*. Anchor its base to the store's
  `MAX(turn_index)+1` whenever numbering restarts, add a UNIQUE index with a non-destructive insert, and
  run a one-time renumbering migration after taking a backup. The dry run on the snapshot kept every
  row and left 0 duplicates in 0.15 s. The added cost is 3.4 µs, once per numbering restart. **No
  parity golden changes**, provided the fix adds no table and no column.

---

## 1. Scale

| | |
|---|---|
| Rows / sessions in `lcm_messages` | 18,879 / 112 |
| Affected sessions | **16** (14%) |
| Duplicated `(session, turn_index)` pairs | **1,210** (849 are pairs of 2; the largest holds 189 rows) |
| Rows inside duplicated pairs | 8,678 (55% of the 15,902 rows in affected sessions) |
| Colliding rows (reuse an index already written in the session) | **7,468** |
| First / last colliding row | 2026-05-27 / 2026-09-24 (first and last day of data) |

**By surface** (sessions, affected, rows, colliding rows): telegram 1/1/8,345/**6,388** (86%) ·
beacon 27/5/8,457/712 · no prefix 24/5/814/343 · voice 9/3/93/14 · desktop 15/1/1,016/10 ·
web 8/1/36/1 · ios, smoke, probe, verify, beacon-verify: 0 collisions.

**Is it getting worse?** Not compared with August, but it is not dying out either.

| ISO week | rows | colliding | rate |
|---|---:|---:|---:|
| W33 (08-10…16) | 5,518 | 4,847 | 88% |
| W34 | 591 | 246 | 42% |
| W35 | 1,513 | 618 | 41% |
| W36 | 379 | 258 | 68% |
| W37 (trim fix deployed 09-11) | 3,944 | 184 | 4.7% |
| W38 | 3,305 | 456 | 13.8% |
| W39 (to 09-24) | 2,617 | 310 | 11.8% |

Two factors keep it going:

- The remaining producers fire on ordinary events. There have been 292 daemon starts since 06-27
  (10 on 09-19 alone), plus `/reset` and failed turns.
- **Collision debt.** 7 of the 16 affected sessions are still numbering below their own maximum. Their
  next 78, 81, 124, 347, 460, 767 and 1,359 rows will collide even if no new producer fires.

## 2. Shape

Each colliding row, compared with the earlier row that has the same index:

| | rows |
|---|---:|
| **Byte-identical** (role, content, content_json) | **78** (1%): gaps 14 under 1 s, 37 under 60 s, 15 under 1 h, 12 of 1 h or more |
| Different message, same role, text present | 2,575 |
| Different message, different role, text present | 910 |
| Different tool call or tool result (both rows have empty flat text) | 3,983 |

By pair: 1,172 of the 1,210 pairs hold only distinct messages, 29 are mixed, and 9 are all-identical.
164 colliding rows are exact copies of *some* earlier row in their session. The identical copies fit
rollback-and-retry, where the user resends the same text (§3, P5). The collisions are two
conversations that share numbers; they are not doubled writes.

## 3. Cause

`turn_index` is stamped at persist time as *list position + `_turn_index_offset`*
(`engine/session.py:469`). The offset starts at 0 and is set in only two places: `restore()` and
`trim()`. Any code path that starts numbering again without an offset that clears the session's
durable maximum writes duplicates.

### Every path that assigns `turn_index`

| # | Path | What it does | Duplicates? | Regression test (`tests/test_lcm_turn_index_unique.py`) |
|---|---|---|---|---|
| P1 | **Cold restart, rehydrate declines.** `rehydrate_if_cold` (`session.py:826`) restores nothing when its window (40 rows, newest-first budget of 8,000 tokens, `session.py:58-59`) contains no clean human turn. One large tool result is enough. | Session starts at offset 0. | **Yes.** All 9 from-0 restarts after 08-31 had this window shape when replayed against the snapshot. | `test_restart_where_rehydrate_declines_continues_above_history` |
| P2 | **Cold restart through a path that never rehydrates.** `inject_turn` (task completions, `telegram.py:2212`), `POST /api/chat` (`web/server.py:4466`), Slack (`slack.py:707`), Discord (`discord.py:600`). After that first write the session is warm, so a later human message's rehydrate does nothing. | Offset 0. | **Yes.** 5 of the 9 post-08-31 restarts were first written by a `task_supervisor` injected turn. | `test_restart_through_a_path_that_never_rehydrates` |
| P3 | **Rehydrate numbers from the window, not the session.** `next_turn_index=max(p.turn_index for p in kept)+1` (`session.py:838`). Once one lifetime has restarted from 0 and grown past 40 rows, the window no longer sees the older, higher indices. | Carries the collision forward. | **Yes. This is the largest live producer:** 596 of 832 colliding rows since 09-11. | `test_rehydrate_continues_from_the_session_max_not_the_window` |
| P4 | **`/reset`, `/clear`.** Telegram `_cmd_reset`/`_cmd_clear` (`telegram.py:942/877`), WS/slash `_sc_reset` (`commands.py:1983`), Slack, Discord → `ChatSession.clear()` (`session.py:604`) resets positions and the watermark but **not** the offset. | Numbers from 0 mid-lifetime. | **Yes.** 745 rows; last seen 08-26. | `test_reset_continues_above_the_cleared_rows` |
| P5 | **Rollback, then retry.** The Telegram/Slack/Discord user row is durable before the turn runs. A failed turn calls `rollback_last()` (`telegram.py:2088`, `session.py:537`), the watermark retreats, and the next message is persisted at the same position. | Same index. With the same text, a byte-identical copy. | **Yes.** 14 rows land on "the same index as the row just before"; last seen 08-18. | `test_rollback_then_retry_takes_a_new_index` |
| P6 | **WS failure with a durable mid-turn row.** A message sent mid-turn is persisted at once, above the tail. `rollback_to(original_len)` (`ws_server.py:1407`, `session.py:552`) removes it from memory but the row stays durable, and the next turn numbers over it. | Same index. | **Yes** (code path). Not separable in the data. | `test_ws_failure_numbers_the_next_turn_above_a_dropped_durable_row` |
| P7 | **Trim.** `trim()` shifted positions without moving the offset, so it plateaued at index 50 (+offset). | — | **Fixed** by #446 (deployed 09-11 14:11). **Historically the largest producer:** 3,387 rows; the last one on 09-08. | covered by `tests/test_trim_advances_turn_offset.py` |
| P8 | **CLI REPL** (`__main__.py:753/788`). The user row and the assistant row share one `turn_index` per turn, and numbering restarts at 0 per run. Session ids are unique per run (`cli-<hex>`). | Pairs by design. | Not on the mini (0 `cli-` sessions). It violates the uniqueness invariant, so the fix must change it. | `test_cli_rows_are_appended_by_the_store` |
| P9 | **Fork** (`lcm_conversation_store.py:974`). Copies `turn_index` verbatim. | Inherits the origin's duplicates. | Only by inheritance. | — |
| P10 | **Normal appends; concurrent turns; several clients.** Every surface shares one `ChatSession` per id, turns serialize on `turn_lock_for`, and the persistence watermark makes overlapping persists idempotent. A mid-turn row persisted early (the "ahead-set") gets a *higher* unused index, so it is out of rowid order but does not collide. | — | **No.** 16 such descents into unused indices in unaffected sessions. No evidence of a second writer process. | — |
| P11 | **Purge / undo / checkpoints.** Purge deletes rows (it creates gaps, not repeats). Checkpoint undo writes other tables. | — | **No.** | — |

### Attribution of all 7,468 colliding rows

Each colliding row is attributed to the most recent numbering event before it in its session. A
restart is a daemon start that falls between the two rows.

| Producer | 05-27…06-26 (no journal) | 06-27…08-30 (no rehydrate) | 08-31…09-11 15:55 (rehydrate on) | 09-11 15:55…09-24 (trim fixed) | Total |
|---|---:|---:|---:|---:|---:|
| P7 trim plateau (fixed) | 95 | 3,157 | 135 | 0 | 3,387 |
| P1/P2 restart numbered from 0 | — | 2,237 | 37 | **233** | 2,507 |
| P4 `/reset`, `/clear` (from 0 mid-lifetime)¹ | 291 | 454 | 0 | 0 | 745 |
| P3 rehydrate from the window max | — | — | 75 | **596** | 671 |
| Restart to another value below the max² | — | 2 | 129 | 1 | 132 |
| P5 same index as the row just before | 3 | 11 | 0 | 0 | 14 |
| Unattributed | 2 | 8 | 0 | 2 | 12 |
| **Rows written / colliding** | 490 / 391 | 8,144 / 5,869 | 2,340 / 376 | 7,905 / **832** | 18,879 / 7,468 |

¹ Before 06-27 the journal cannot separate a restart from a reset, so column 1 may include restarts.
In the journal era, no `DELETE /api/sessions/…` (forget) falls inside any of these events.
² Rehydrate-window variants and trim-with-offset in the rehydrate era.

Replaying today's rehydrate over the 62 pre-08-31 restarts that numbered from 0 shows that it would
not have prevented 33 of them (1,650 rows), because of P3.

## 4. Impact: what the model actually sees

**SQLite's real tie order.** All three readers plan as
`SEARCH … USING INDEX idx_lcm_messages_session (session_id=?)`. Every one of the 1,210 duplicated pairs
comes back in rowid order. So `ORDER BY turn_index` is deterministic in practice, but it **interleaves
lifetimes**: A0, B0, A1, B1, …. The reproductions showed exactly this (Appendix).

| Consumer | Order it uses | Affected? | Evidence |
|---|---|---|---|
| **Live prompt** (the model's per-turn context) | in-memory `ChatSession.messages`; rehydrate reads by rowid (`messages_page`) | **No** | `TestControl::test_live_prompt_after_restart_is_chronological` passes. #548 already established that `/api/lcm`'s assembly is not the prompt. |
| **LCM compactor → summarizer model** (`lcm_compaction.py:78`) | `get_uncompacted_messages`, `ORDER BY turn_index` | **Yes: misorders, and picks the wrong tail** | Repro: the real summarizer prompt reads `A0, B0, A1, B1, …`. The fresh tail kept is `A10, A11, B10, B11`, not `B8…B11`. **Production:** 442 of 2,472 leaf summaries (18%) in 12 sessions came from out-of-order batches, created from 07-02 through 09-23. Their median source span is 32.5 h, against 0.0 h for in-order batches. 248 of them alternate between eras more than once. 220 uncompacted rows are older than a compacted row, and every newer compacted row has a *lower* index: the compactor summarized new messages and kept old ones raw. Right now, in 5 sessions the 32-row fresh tail by `turn_index` is not the newest 32 rows. |
| Summaries → the model | `lcm_grep`, `lcm_expand`, `lcm_expand_query`, `lcm_describe` | **Yes, when called** | 103 calls in telemetry through 09-24 (68 of them `lcm_grep`). Depth ≥1 nodes are built from these leaves. |
| **Duplicated content to the model** | — | Only in summarizer input | The 78 identical rows (P5) would each appear twice in a compaction batch. Nothing is doubled in the live prompt. |
| **Golden-trace export** (training data; `golden_trace_exporter.py:102`) | `get_messages(limit=500)`, `ORDER BY turn_index ASC` | **Yes** | Of 4,416 golden calls in affected sessions, 3,243 resolve a context different from the rowid-ordered one: 128 purely because of ordering, 642 because of ordering and the limit together, and **2,473 purely because of `LIMIT 500`**. That limit returns the 500 *lowest* indices, a separate defect that a `turn_index` fix does not cure (see §6). 26 export files (5,519 lines, 08-15 → 09-25) were written through this path. |
| `/api/lcm` context meter (`web/server.py:2883`) | assembler fresh tail | Display only | Repro on `caa24db`: the fresh tail was `A10, B10, A11, B11`. |
| Title backfill (`session_titles.py:167`) | `get_all_messages` → first exchange | Minor | It can title from an interleaved "first" exchange. |
| REST history / `?since=`, rehydrate, MemoryExtractor | rowid or timestamp | No | — |

**Why not simply `ORDER BY rowid`?** Rowid is *persist* order. During a long agentic turn, a Beacon
message sent mid-turn is persisted immediately at its prompt position (e.g. 650), and the turn's
326-row tail (324…649) is persisted afterwards. The model saw the message *after* those rows. On the
mini, 660 rows sit in a different place in prompt order than in rowid order. `turn_index` order is
right for them and rowid order is wrong. So the fix must keep prompt-position semantics.

## 5. Fix design

**Invariant.** `turn_index` is the durable prompt position: unique per session and monotonic across
lifetimes. Readers order by `(turn_index, rowid)`.

### 5.1 Allocation: anchor the numbering to the store

- Add `LCMConversationStore.next_turn_index(sid)`, defined as
  `SELECT COALESCE(MAX(turn_index), -1) + 1 FROM lcm_messages WHERE session_id = ?`. It is an
  index-only seek: **3.2–3.4 µs p50** on an 8,345-row session.
- In `ChatSession._persist_to_lcm`, which is the single choke point every surface goes through, anchor
  once before the first write after the numbering (re)starts:
  `offset = max(offset, store.next_turn_index(sid) - first_position)`, where `first_position` is the
  lowest position not yet durable (§5.7).
  Mark numbering as unanchored in `__init__` (fixes P1 and P2 on every surface at once), in `clear()`
  (P4), and after `rollback_last` / `rollback_to` discards a durable row (P5, P6). Do nothing when the
  engine has no store, so the fake engines in `test_session_persist_exact_once.py` keep working.
- `restore()` / `rehydrate_if_cold`: take `next_turn_index` from `store.next_turn_index(sid)` rather
  than the window max (P3).
- `add_user_message` should return the **durable** ordinal (position + offset). Today it returns the
  list position, which is wrong after rehydrate or trim, even though its docstring promises the
  durable value.
- CLI (P8): as built, the REPL passes no index and the store appends each row atomically
  (`ingest(turn_index=None)`). For the REPL, arrival order is prompt order, so the store-side
  numbering rejected below for `ChatSession` is exactly right there (§5.7).
- **Rejected alternatives:**
  - A store-side `MAX+1` inside `insert_message` is atomic, but it switches the semantics to arrival
    order and misplaces the 660 mid-turn rows. Use it only as the conflict fallback below.
  - `ORDER BY rowid` in the readers fails for the same reason.

### 5.2 Guard: a UNIQUE index, but only with a non-destructive insert

- Replace `idx_lcm_messages_session` (a non-unique index on `(session_id, turn_index)`) with a UNIQUE
  index on the same key. That adds no index-maintenance cost, and the planner still uses it: I checked
  `EXPLAIN` for both `ORDER BY turn_index` and `ORDER BY turn_index, rowid`; neither needs a temp
  B-tree.
- ⚠ **`insert_message` uses `INSERT OR REPLACE` (`lcm_conversation_store.py:294`).** With a UNIQUE
  index, a colliding insert would **silently delete the older message** and leave its FTS row
  orphaned. Switch to plain `INSERT` (`ON CONFLICT(id) DO NOTHING` if id idempotence is wanted). On an
  IntegrityError on the turn key, reassign to `next_turn_index` and retry once, then call
  `record_silent_failure(subsystem="lcm", operation="turn_index_collision")`. That way a new producer
  shows up loudly and never costs a row. A bare IntegrityError is not an option: `_persist_to_lcm`
  swallows exceptions, so the row would simply vanish from LCM.

### 5.3 Migration of existing rows

It runs once, at daemon start, before gateways write. It is gated by `PRAGMA user_version` 0 → 1 on
`lcm.db`: the value is 0 today, and `memory/store.py` already uses this pattern.

1. **Precheck.** If there are no duplicates, create the unique index, set `user_version = 1`, and
   stop. No backup and no file is written.
2. **Backup first.** Use the sqlite3 backup API to write `<lcm.db>.pre-turn-index-<ts>.bak` next to
   the DB, then check `PRAGMA integrity_check` and the row count on the backup before touching
   anything.
3. **Renumber per affected session, in one `BEGIN IMMEDIATE` transaction:**
   - Walk the rows in rowid order and split them into **runs**. A run ends when an index repeats
     inside it, or when an index drops below everything the run has used.
   - A drop *into an unused gap* of the run is the ahead-set shape and stays in the run.
   - Keep run 0's values. Shift each later run up to start above every value so far, keeping its
     internal gaps. Within a run, order by `(turn_index, rowid)`.
   - Apply through a TEMP map table (a portable correlated `UPDATE`, not `UPDATE … FROM`, which needs
     SQLite 3.33), then drop the old index and create the UNIQUE one.
4. **Verify before COMMIT:**
   - row count unchanged;
   - a digest of every other column, by rowid, unchanged;
   - zero duplicate pairs;
   - `ORDER BY turn_index` equals the computed order for every session.

   If any check fails, ROLLBACK, log loudly and keep running without the unique index. The migration
   must never brick the daemon.
5. **Nothing else keys on `turn_index`.** ids, rowids, FTS rowids, summaries (which reference ids) and
   `message_client_ids` (which reference rowids) are untouched. The wire `ordinal` changes value for
   renumbered rows. It was documented as non-unique and display-only ("Do not key on it"); it is now
   documented as the unique prompt position, a display order and not an identity (`web/server.py`).

**Dry run of the prototype on the 09-25 snapshot** (on the mini: backup API from the snapshot into
`:memory:`, migrated, verified, discarded, nothing written; §5.7 has the run with the final code):

- 16 sessions, 302 runs, **13,791 rows renumbered** (largest shift +7,855).
- Rows 18,879 → 18,879. Digest of all other columns unchanged. 0 duplicate pairs after. Intended order
  holds in 16 of 16 sessions. FTS `integrity-check` passes. A second pass finds nothing (idempotent).
- 0.09 s to compute plus 0.05 s to apply. Copying the 185 MB file with the backup API took 0.12–0.2 s
  (into memory).
- 660 rows end up in a different position than rowid order; all are prompt-order restorations of
  mid-turn rows.
- There are 19 in-run descents in the data, and none of them has a daemon start between its two rows.
  So the run rule never merged two lifetimes here.

**Not repaired by renumbering:**

- **The 442 leaf summaries built from zipped batches, and their ancestors, are left as they are, by
  decision (2026-09-26).** The migration renumbers rows and touches no summary. Rebuilding them costs
  model calls and is separate follow-up work.
- The 26 golden-trace exports already written. Regenerating them is also separate follow-up work.

### 5.4 Parity goldens (`tests/fixtures/parity/`)

**None change** with the design above:

- None of the 11 traces contains `turn_index`.
- No scenario runs LCM compaction: `lcm_summaries` has 0 rows in all 11 `expected.json`.
- Every scenario writes one session numbered 0…n in rowid order, so the precheck is a no-op and the
  values are identical.
- The store dump lists **tables and rows only** (`scripts/parity/observe.py` reads
  `sqlite_master WHERE type='table'`). Indexes and `user_version` are not in it.

**Traps that would change all 11 `expected.json`:**

- adding a column to `lcm_messages`;
- adding **any** table to `lcm.db`, even an empty one (for example a migration audit table);
- writing the backup file under the parity HOME when there is nothing to migrate.

The fix touches `engine/session.py`, so the seam rule applies: parity `replay` and
`bench --baseline` on the mini.

### 5.5 Speed

Synthetic benchmark on the Mac (a DB shaped like production: 18,879 rows, one session of 8,345;
400 inserts × 2 rounds):

- Insert p50/p95 is **70–82 / 109–136 µs** today and **70–73 / 111–117 µs** with the UNIQUE index.
  That is the same within noise; the per-commit fsync dominates.
- The anchor query costs **3.4 µs p50**, once per numbering restart, not per row.
- `ORDER BY turn_index, rowid` uses the same plan.
- The migration runs once, in about 0.15 s, plus the backup copy (0.12–0.2 s from file into memory on
  the mini; I did not measure a copy to disk).
- I expect no measurable change per agent round. The parity bench on the fix PR confirms it (§5.7).

### 5.6 Tests and docs that pin the old contract and must change with the fix

- `tests/test_durable_message_id.py:27` inserts two rows with `turn_index=0` and asserts
  `ordinals == [0, 0]`.
- The "non-unique display position" comments in `tests/test_wire_contract.py:147,257`, plus the REST
  docstring at `web/server.py:1311-1312`.
- `lcm_conversation_store.py:569` ("repeats across restart/trim").
- `session.py:835-837` ("turn_index restarts per daemon lifetime").

### 5.7 As built

What the PR that adds this file does, where it differs from or adds to the plan above, and the gates it
passed.

- **The backup comes just before the write lock, and `PRAGMA data_version` proves nothing moved in
  between.** SQLite cannot back up from a connection that holds a write transaction: the step reports
  BUSY for as long as the transaction lasts, and Python's `backup()` retries BUSY forever. So the
  migration copies first, then takes `BEGIN IMMEDIATE`. If another connection committed in the gap, it
  aborts without changing anything, and the next start retries. The copy is created owner-only (0600),
  never over an existing file, and a partial or failed copy is deleted rather than left looking like a
  backup.
- **Only the daemon installs the UNIQUE index.** `migrate_turn_index` runs in `run_daemon` right after
  `LCMEngine` is built, before the session manager, the agent loop, any gateway, the cron scheduler, the
  extractor, the golden-trace exporter or the web server is wired. `TestDaemonOrdering` pins that order.
  The store keeps the legacy index until then and never re-creates it afterwards. This is deliberate:
  another process on new code must not install a guard that a pre-fix writer would then trip over by
  deleting rows (below).
- **The anchor sits at the lowest position not yet durable,** not at the first row being written, so a
  turn's tail waiting under a mid-turn message also clears the store's maximum. If the store ever has to
  move a row, the session moves its numbering with it: one loud collision, not one per row.
- **The CLI REPL (P8) lets the store number its rows:** `ingest(turn_index=None)`, an atomic
  `INSERT … SELECT MAX + 1`.
  - Two indices per turn would still restart at 0 on every run, and would collide when a failed turn's
    user row is retried.
  - Routing the REPL through `ChatSession` would change what the CLI persists.
  - For the REPL, arrival order is prompt order.

  The `ingest()` default is now `None` (append); the old default of 0 guaranteed a collision for any
  caller that omitted it.

**⚠ Pre-fix code must not write a migrated `lcm.db`.** It inserts with `INSERT OR REPLACE`, which under
the UNIQUE index resolves a collision by deleting the older row. A pre-fix daemon collides on the first
write of every restarted session. The pre-fix CLI REPL collides on every turn, because its user and
assistant rows share an index. Before running a pre-fix build, stop the daemon and undo the index:

```sql
DROP INDEX idx_lcm_messages_session_turn;
CREATE INDEX idx_lcm_messages_session ON lcm_messages (session_id, turn_index);
PRAGMA user_version = 0;
```

Alternatively, restore the `.bak`, which loses what was written since.

**Gates**

- **Tests.** `tests/test_lcm_turn_index_unique.py` has 41 tests. The ten reproductions failed on
  `caa24db` (10 failed, the control passed) and pass now. Full suite on the Mac (Python 3.14.3): 8,694
  passed, 513 skipped, 0 failed. ruff and the mypy gate are clean (250 modules clean, 122 on the debt
  list).
- **Parity on the mini** (Python 3.11.15): replay 11/11 PARITY (0 diffs, all model requests matched);
  stability IDENTICAL across 2 runs. No golden changed: the PR touches no `expected.json`.
- **Speed gate on the mini:** branch median p50 17.1 ms (Δ +0.7 ms against the baseline's 16.4; mean
  Δ +1.3 ms; peak RSS Δ +1.7 MB), within the noise band. The main control, run right after, also came
  in at 17.1 ms (Δ +0.6 ms; mean Δ +1.3 ms; peak RSS Δ +1.5 MB).
- **Insert path** (`bench_insert.py`, synthetic, on the Mac, 400 inserts × 3 rounds): p50 75–80 µs
  against 71–79 µs before the fix. The extra 1–4 µs is the read-back of the stored index. The anchor
  query costs 3.4–3.5 µs, once per numbering restart.
- **Final dry run of this code on the 2026-09-26 nightly snapshot** (`lcm_migration_dryrun.py`, in
  memory, nothing written). It gave the same result under the daemon's own interpreter, Python 3.12.3
  with SQLite 3.45.1:
  - before: 7,468 colliding rows in 16 sessions;
  - `migrated`: 13,791 rows renumbered in 302 runs (largest shift +7,855), with 660 rows placed in
    prompt order ahead of rowid order;
  - after: 0 duplicates, 18,879 rows before and after, tables and columns unchanged, no TEMP table
    left;
  - `integrity_check` and FTS integrity ok; a second run is a no-op;
  - 0.54 s.
- **Beacon clients** (desktop and iOS) key and page on `message_id`. Both cache `ordinal`, and one
  display sort in the desktop client orders by it; unique values make that sort correct. No client
  change is needed.

## 6. Found in passing (not part of this defect)

1. **Golden-trace exporter `LIMIT 500`** (`golden_trace_exporter.py:102`). `get_messages(limit=500)`
   is ascending, so for any call after a session's 500th message the resolver sees context from around
   row 500. 2,473 golden calls differ for this reason alone, and 642 more in combination with the
   ordering problem. It needs a "rows before `ts`, newest first, then reversed" read.
2. **A failed WS turn drops a durable mid-turn user message from the conversation** (P6). The row
   stays in LCM, but the model never sees it again.
3. **Rehydrate declines entirely** whenever the newest 8,000 tokens hold no clean human turn, for
   example after one large tool result. That is a context-loss problem ("the model starts blind") in
   its own right, separate from the numbering: 9 of the restarts since 08-31 in affected sessions.
4. `add_user_message`'s return value is not the durable ordinal after rehydrate or trim (see §5.1).

## Appendix: reproduction

Before the fix, on `caa24db`, the reproductions failed one per claim (10 failed, 1 passed):

- `reset` → `[(0, 2)]`
- `rehydrate declines` → `[(0, 2)]`
- `never rehydrates` → `[(0, 2)]`
- `window max` → "lifetime C numbered 46: inside lifetime A's 0..99"
- `rollback retry` → `[(0, 2)]`
- `ws failure` → `[(2, 2)]`
- compactor prompt → `['A0', 'B0', 'A1', 'B1', …]`
- compactor tail → `['A10', 'A11', 'B10', 'B11']`
- assembler → `['A10', 'B10', 'A11', 'B11']`
- golden context → `['A3', 'B3', 'A4', 'B4', …]`

The control, `test_live_prompt_after_restart_is_chronological`, passed. With the fix, all of them are
regression tests in `tests/test_lcm_turn_index_unique.py` and pass:

```
PYTHONPATH=src python -m pytest tests/test_lcm_turn_index_unique.py -q
```

Scan scripts: `docs/audits/lcm-turn-index/`. The `lcm_dup_scan*.py` and `lcm_run_rule_check.py`
scripts run on the mini against a nightly snapshot (`python3 - <snapshot> < script.py`) and print
aggregates only. `lcm_migration_dryrun.py` runs the real migration on an in-memory copy of a
snapshot. `bench_insert.py` is the synthetic insert benchmark.
