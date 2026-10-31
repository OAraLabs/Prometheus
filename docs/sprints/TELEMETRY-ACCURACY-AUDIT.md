# Telemetry and trace accuracy audit (2026-10-31)

Read-only audit of every place that counts traces or reads "pass"/success, done for
`fix/telemetry-trace-accuracy`. Code at `origin/main` ea7a3b2. No live database was opened. The
numbers quoted from the corpus come from the distillation audit (`DISTILL-PHASE1-AUDIT.md`,
`DISTILL-PHASE2-JUDGE.md` and `DISTILL-LOG.md` on `feat/distill-flywheel`), which measured the
3090 Ti host's stores on 2026-10-30.

## Summary

- **"Pass" on a trace has only ever meant `tool_calls.success`: the call ran without error.**
  `is_golden` = cloud provider + `success` + zero retries + raw output captured
  (`telemetry/tracker.py` `_record`). Nothing records whether the turn got anywhere, and no
  label says so. A strong-model re-judge failed 20.5% [16.2, 25.6] of a 300-trace pass sample.
- **No code adds the two trace stores together.** The ~22K was a manual sum of 11,152 JSONL
  lines and 11,207 `training.db` `cloud_golden` rows: the same cloud calls, copied twice.
  Three surfaces invite that sum (below), and one causes a real duplicate:
  `oara export-traces` writes its dumps into the folder, and under the file-name glob, that
  the incremental exporter and the distillation harvester read.
- **The "41% in abandoned or force-stopped turns" mixes two signals.** `turns.outcome =
  'abandoned'` is a USER signal (no message within 30 minutes of a finished turn), and
  `telemetry/outcomes.py` says it is neutral and "must never be used as a negative label": it
  usually means the user got what they needed. Of the 4,561 linkable Max traces, 465 (10.2%)
  sat in force-stopped turns and 8 in user-corrected ones; 1,425 (31.2%) were "abandoned" in
  that neutral sense. A defensible "ended badly" share from that data is about 10%, not 41%.
  The judge's 20% false-pass rate is the stronger evidence that tool success is not task
  success, and it stands.
- **No router, model picker, ladder verdict or A/B result ranks on tool success.** The ladder
  keeps "tool-call success" in its own column; evals and the A/B script use judge metrics.
  What is wrong is labelling, two places that double count or invite it, three script bugs in
  pass checks, and one success rate computed three different ways.
- **`training_pairs.outcome` is declared "copied at mine time" and is never written.** It is
  NULL on every row, so nothing downstream can tell a pair's turn outcome.

## The stores

| Store | What a row is | "Pass" | Turn outcome |
|---|---|---|---|
| telemetry.db `tool_calls` | one tool call (plus `_loop_transition` loop markers) | `success` (ran without error); `is_golden` derived from it at write time | none on the row; `turn_id` (telemetry v2, 2026-10-03 on) joins `turns` |
| telemetry.db `turns` | one `run_loop` turn | — | `ended_at`, `forced_stop_reason`, `terminal_kind` (end state); `outcome` (user signal: `accepted_user`, `user_corrected`, `abandoned`, ...; daemon: `forced_stop`, `error_terminal`; coding: `accepted_verified`, `rejected_verified`) |
| `trajectories/golden_traces_<ts>_<rowid>.jsonl` | one golden call, context rebuilt from LCM at export | implicit: every line is golden | none |
| training.db `training_pairs` | a repair/retry pair, or (`cloud_golden`) a chosen-only copy of a cloud call | implicit: `chosen` ran (except `levenshtein_repair`/`schema_repair`, banked before the call runs) | `outcome` column exists, never written |
| gym.db `gym_runs` | one gym run | `execution_pass` (all predicates) | n/a |

**Canonical store per count.** Teacher/golden traces: telemetry.db `tool_calls WHERE is_golden = 1`.
JSONL exports are a derived, filtered subset (rows with no recoverable context are skipped);
`cloud_golden` rows are a second, independently filtered copy (no zero-retry rule; refused for
group-member turns, which `is_golden` is not). Neither may be added to the other or to telemetry.

## Consumers

Legend: **TS** = treats tool success as task/turn success (or presents it unqualified),
**DC** = double counts or invites it, **OK** = correct as is, **BUG** = wrong for another reason.

### telemetry.db `tool_calls` / `is_golden`

| Consumer | Who sees it | Finding |
|---|---|---|
| `ToolCallTelemetry.report()` `telemetry/tracker.py` | `GET /api/telemetry` (Beacon telemetry card), `TelemetryDigest` | **TS (label).** Per model × tool `success_rate` and `overall_success_rate` are tool-call rates under an unqualified name; the per-model breakdown reads like model quality. No turn-level number exists to show beside it. |
| `health_summary()` → `/health` (`gateway/commands.py` `cmd_health`; Telegram, Slack, Discord, web) | users | **TS (label).** "Tool calls: N (F failures, X% success)" with a ✅/⚠/❌ verdict. No turn outcome anywhere in the health view. |
| `TelemetryDigest` `sentinel/telemetry_digest.py` (AutoDream phase) | activity feed `dream_insight` | **TS (label) + BUG.** Summary says "X% overall success rate" (tool calls). Its 7-day baseline includes the 24 h window it is compared against, which dilutes every drop it looks for. |
| `ToolDashboard` `telemetry/dashboard.py` → `/tools` | users | **BUG + TS (label).** The rate keeps `nonzero_exit` (a command that ran and exited non-zero) in the denominator; `report()` and `/health` exclude it, so `/tools` shows a lower number for the same calls. "Success rate" unqualified. |
| `recent_tool_calls()` → `GET /api/tools/recent`; `coding/livestream.py` | Beacon tool feed, coding view | **OK.** Per-call `success`, shown per call. |
| `get_golden_traces()` → circuit-breaker `golden_reference` (`engine/agent_loop.py`) | diagnosis text | **OK.** Picks an example call, no count. |
| `export_new_golden_traces` / `GoldenTraceExporter` | `trajectories/` corpus, `golden_traces_exported` event | **TS.** Every line is a "pass" by construction and carries no label saying what kind. Lossy shape (below). |
| `export_golden_traces` → `oara export-traces` (`__main__.py`) | CLI | **DC.** Default output is `~/.prometheus/trajectories/`, file `golden_traces_<ts>.jsonl`: the incremental exporter's folder and the glob the distillation harvester (`scripts/distill/build.py` `DEFAULT_TRACES`) and GEPA read. Each dump re-states rows already exported. |
| `check_trajectory_export` `cli/doctor.py` | `oara doctor` | **BUG + DC.** Treats a missing `trajectory_export.enabled` as on (the daemon treats it as off), reads a hard-coded `~/.prometheus/telemetry.db`, and counts every `*.jsonl` file, CLI dumps included. |
| `is_golden` itself (`_record`) | every golden count | **TS + overcount.** Counts a group member's cloud calls as golden; the exporter later drops their context, and `cloud_golden` refuses them. |
| GEPA evidence `learning/gepa_evidence.py` | judge prompt | **OK.** Labels each call ok/failed/denied/nonzero_exit, per call; does not read the JSONL. |
| Model ladder `gym/ladder/record.py` | `scripts/ladder_run.py` report | **OK.** "tool-call success" is its own column; accuracy/task success come from the verdict (predicates, acceptance test, pinned judge). |
| `scripts/mine_training_pairs.py` | training.db | **OK for "pass"** (`chosen` = a `success = 1` call). Out of scope: it may pair a failure with a success from another session; mined pairs carry no `turn_id`. |
| `scripts/web_capability_audit.py` | `/audit` report | **BUG.** Reads `report()["by_tool"]`; the key is `tools`, so every test with `expect_tools` fails. |
| `scripts/smoke_test_tool_calling.py` `telemetry_recording` | smoke exit code | **BUG.** Passes on `bool(report())`, which is always true. |
| `scripts/vibe_check.py`, `scripts/parity/scenarios.py`, `gym/runner.py` deltas | reports, parity checks | **OK.** Counts or per-scenario checks, not success rates. |

### telemetry.db `turns`

`turn_end_outcome`, `sweep_abandoned`, `note_user_message` (`telemetry/outcomes.py`) write it;
`scripts/local_gate/run_gate.py` checks one turn's `forced_stop_reason`. **Nothing computes a
rate from turns**, so no surface can show turn-level success today.

### training.db

| Consumer | Who sees it | Finding |
|---|---|---|
| `PairStore.stats()` → `/pairs`, `GET /api/pairs` | users, Beacon | **DC.** "Training Pairs — Total" counts chosen-only `cloud_golden` copies as pairs; added to a trace count it double counts the same calls (the ~22K). |
| `scripts/export_training_pairs.py` | DPO JSONL | **OK** by default (skips `cloud_golden`). With `--include-golden` the output overlaps the golden JSONL. |
| `scripts/transition_histogram.py` | report | **OK.** `cloud_golden` lands in "other". |
| `training_pairs.outcome` | nobody | **BUG.** Never written (see Summary). |

### Gym, evals, benchmarks

| Consumer | Finding |
|---|---|
| Gym `gym/scoring.py` + `gym/report.py` | **TS** for tool-only tasks. A task whose only predicates are `expect_tool` (plus arg-shape checks) passes when the call ran: harvest-v1 (63 tasks), dictwrap-train (519), dictwrap-heldout (40), and 3 tasks each in v1/v2. The report calls execution pass "the 'did it work' axis" without saying which tasks it cannot see past the call. |
| Evals `evals/runner.py`, `evals/classifier.py`; `scripts/run_model_ab_eval.py` | **OK.** Pass needs judge metric scores; the two scripts use different thresholds (0.5 everywhere vs per-metric), noted, not changed. |
| Benchmarks `benchmarks/runner.py` | **OK by design.** A tool-selection benchmark: "pass" is the expected tool being chosen, not task success, and nothing presents it as such. |

## Stored trace format (what a golden line is missing)

From the distillation audit, confirmed against `_build_example` and `lcm_context_resolver`:
no system prompt; context is flat LCM text, so earlier tool calls and tool results are
dropped; `tools` holds only the called tool's schema; context is capped at 12 messages (79%
of the corpus hit the cap); the route (primary, per-session override, fallback) is not
recorded. Phase 3 had to rebuild every request from `lcm.db` and today's tool registry, whose
schemas may have drifted.

## Fix plan (this branch)

1. **Labels:** `tool_ok`, `turn_outcome` (end state: `completed` / `stopped` / `errored` /
   `abandoned` = never ended / `unknown` = unlinkable), `judged_ok` (+ `judged_by`,
   `judged_at`) on `tool_calls` and `training_pairs`; additive columns; stamped at turn end;
   `oara trace-labels` backfills (dry run by default). The user signal stays in
   `turns.outcome` and rides in v2 exports as `turn_signal`.
2. **Callers:** turn-outcome rates beside every tool-call rate (`/api/telemetry`, `/health`,
   digest), tool-call rates named as such; `/tools` denominator fixed; digest baseline no
   longer overlaps; `/pairs` splits pairs from exemplars; `export-traces` dumps leave the
   canonical glob; doctor counts only canonical exports from the configured DB; gym runs
   record whether their pass is tool-only; the two script bugs fixed; group-member calls
   are no longer golden.
3. **Capture v2:** for new golden calls, the request as sent (system prompt, full tool list,
   every prior message with tool calls and results, no cap) plus served model, provider and
   route, stored content-addressed; exports write `_meta.trace_format = 2`, a superset of v1.

## Not changed (follow-ups)

- `mine_training_pairs.py` cross-session pairing; mined pairs have no `turn_id`, so their
  `turn_outcome` is `unknown`.
- Evals vs A/B pass thresholds differ (0.5 everywhere vs per metric).
- `turns.outcome = 'abandoned'` keeps its name; renaming a user-signal value is a separate call.
