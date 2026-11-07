# Gym fidelity for outcome checks: plan (not built)

Operator order, 2026-11-06:
- Drop tool choice as a GRPO reward target.
- Next priority is gym fidelity: expand the gym snapshot so reference calls can actually run. Argument and outcome correctness can then be measured on eval-tools and used as a reward.
- Propose a plan with an effort estimate before building.

Only phase 0 (the read-only measurements, §1.3) has been built and run; its numbers are in §1.4. Nothing else has been built. Each number is labelled: **recorded** (in the scorecard or code today), **derived** (arithmetic on recorded numbers) or **measured** (phase 0, §1.4).

## Summary

- **Two problems that look like one.**
  - **(A) The GRPO reward pool is gym rows** (round-2 traces). Their world is fully known: the task file, its seed, the repo commit, and the steps run so far. "Making the reference call run" there means rebuilding that world and replaying the steps up to the cut.
  - **(B) eval-tools is real Max traces** from the owner's machines. It is never trained on and is only for measurement. Its world must be rebuilt from the conversation.
  - **(A) unblocks the GRPO pilot; (B) gives the promotion metric.** They share one executor.
- **The gym already has most of the tools.** `build_sandbox_registry` (teacher_gen.py:4054-4095) registers the real `read_file`, `write_file`, `edit_file`, `grep`, `glob` and `todo_write`. It also has in-memory stand-ins for memory, cron, tasks, wiki, web, message, sentinel and lcm_grep, all holding their state in `SandboxEnv.fx` (teacher_gen.py:3765-3988). The outcome check uses none of them:
  - `arg_equiv.EXECUTABLE_TOOLS = {"bash", "read_file"}` (arg_equiv.py:119);
  - `GymSandboxExecutor._run_once` re-implements those two tools itself (arg_equiv.py:663-713).
  - Routing every call through `SandboxEnv.execute` (teacher_gen.py:2902-2959) makes 20 of the 26 tools in the registry snapshot (`eval_tool_schemas.json`) runnable, without loosening any security rule.
- **Most eval-tools references fail on the gym's grammar, not on state** (measured, §1.4). Of the 302 references with arguments, 14 run cleanly today. 183 (61%) are bash the grammar refuses on purpose: 77 multi-line, 55 `$` or other shell syntax, 32 programs such as `gh`, `timeout` and docker, 15 git writes, 4 python. Then 64 lack state (41 of them name another project's repository), and 32 use a tool the executor doesn't route. The arg-equivalence run had put ~77% of undecidable pairs down to missing paths (scorecard:199-203). That was an artefact of order: the executor checks paths before the grammar.
- **There is a hard ceiling.** With python and multi-line bash parked, those 183 references stay unrunnable and are counted as such; they are never worked around. Under today's grammar, eval-tools outcome coverage can reach at most ~23% (70 of 302), against the 50% bar of V5 (§1.4).
- **Reward.** Outcome correctness is tool-agnostic. A call earns credit for obtaining the reference's information (read calls) or making the reference's change (calls with side effects), whichever tool it uses. It replaces the "right tool +1.0" and "arguments +0.5/+0.25" rows of DISTILL-GRPO-PLAN §2.
- **Effort.**
  - **MVP for a first GRPO pilot:** ~8.5–10 agent-days (phase 0 done) and ~3 h of 3090 Ti CPU, no GPU. Phase 0 trimmed it: the round-2 pool has no compacted rows, and every run used one fixture commit (§1.4).
  - **eval-tools (phase 4) can't meet V5** under today's grammar. Its scope is now an operator call (§1.4).
  - **Everything** (eval-tools rebuild, side-effect diffs, validation): ~23–27 agent-days and ~8–10 h of host CPU. GPU: 0, or ~1 h of the 4090 if the local 27B judges instead of Max.

## 1. Measurements

### 1.1 What the code and recorded results already say

| # | Fact | Source | Kind |
|---|---|---|---|
| 1 | The outcome check runs only `bash` and `read_file` | arg_equiv.py:119; acceptable_tools.py:318-320 | recorded |
| 2 | The gym registry has 24 tools, 16 of them stand-ins; 20 of the 26 tool names in `eval_tool_schemas.json` are in it. Not in it: `download_file`, `sessions_spawn`, `vault_search`, `vault_read`, `ask_user`, `skill`. Real cases also offer tools beyond these 26 (tts, image/video generation, youtube_transcript: acceptable_tools.py:121-126). | teacher_gen.py:725-736 | recorded |
| 3 | Outcome check, 2026-11-06: 355 candidate calls; 304 `not_run` (85.6%), 44 `reference_not_ok` (12.4%), 7 `differs`, 0 `supports` | scorecard:230 | recorded |
| 4 | Of the 51 candidate calls where both sides could run, the reference failed in 44 (86%). So only about 14% of the runnable references the check reached ran cleanly. The sample is biased: only cases where some model picked another tool. | from #3 | derived |
| 5 | Student Q4, 267 right-tool cases: 24 exact, 10 execution, 6 conservative, **227 undecidable (85%)**. Of the undecidable: ~77% name paths the fixture lacks, ~20% use refused constructs. The refused share is about 45/267, ~17% of right-tool cases: a structural ceiling while python and multi-line bash stay off. | scorecard:156, :199-203 | recorded / derived |
| 6 | The 26 hard cases expect only a tool, with `args: {}` matched as a subset (build.py:3225-3230). They have no reference arguments to run, so **at most 303 cases** can be checked by outcome. | build.py `case_row` | recorded |
| 7 | The case's working directory comes from the system prompt, which build.py rebuilt from the environment at pack time (`env_now`, build.py:288), not at trace time. `case_env` (arg_equiv.py:190-204) trusts it, so a trace that ran in a worktree may get the wrong cwd. | build.py:279-307 | recorded |
| 8 | Cases with `context_source` `stored` hold prose only (12 messages, no tool results, build.py:296-297), so there is nothing to rebuild from. Cases with `lcm` hold the earlier calls and results. | build.py | recorded |
| 9 | The fixture is one commit (27055dbf, store depth 200: teacher_gen.py:4836). Traces span weeks, and the store may not even reach back to them. | scorecard:134 | recorded |
| 10 | Gym rows record the reference call's own gym result in `_meta.call_result` (first 6,000 chars; teacher_gen.py:6099-6106). This is a fidelity check for replay. | teacher_gen.py | recorded |
| 11 | `grep`, `glob`, `read_file`, `write_file` and `edit_file` run in the Python process, not inside bwrap. `confine()` keeps them in bounds (teacher_gen.py:2751-2768, 2945-2954). Any rebuilt snapshot must therefore contain no symlink that leaves the root. Also, `FileReadTool` falls back to the host's `~/.prometheus/files` (file_read.py:67-72); the gym avoids that only through the exists check at teacher_gen.py:2947-2950, which the new executor must keep. | code | recorded |

### 1.2 What these imply (estimates before phase 0, corrected by §1.4)

- **Which tools the 329 references use** (measured): `bash` 292 (269 with arguments), `edit_file` 10, `task_create` 6, `write_file` 4, `task_get` 3, `memory` 3, then `task_output`, `task_stop`, `web_fetch`, `web_search` (2 each), and `cron_delete`, `task_list`, `read_file` (1 each).
- **What runs today** (measured): 14 of the 302 references with arguments (4.6%) run cleanly in A and B. All 14 are bash, and one of them prints nothing.
- **Why the rest fail, ranked by measured count** (first blocker, §1.4). The pre-phase-0 ranking put missing state first and refused constructs at ≤ ~17%. Phase 0 reverses that:
  1. constructs the gym refuses: **183** (61% of 302). This is the ceiling.
  2. missing state: 64 (41 another project, 18 a repo checkout or worktree, 5 other);
  3. a tool the executor doesn't route: 32 (26 with side effects, 6 read-only);
  4. no reference arguments: 27 (the 26 hard cases and one `task_list`);
  5. simulated host commands, which decide nothing: 7.
  6. no history to rebuild from: **0**. Every case is `lcm`; 17 cases hold no tool results.

### 1.3 Host runs (3090 Ti host, read-only, aggregates only)

Four read-only measurement scripts, M1–M4, run from a checkout of `feat/distill-flywheel` with `CASES=~/distill/phase3/build-v2/eval-tools.jsonl`:

- **M1: reference census.** Static, takes seconds, and executes nothing. Prints:
  - tool counts and whether each case has arguments;
  - the context source;
  - where each reference path points, and whether it appears in the history;
  - how the sandbox grammar would classify each bash reference.
- **M2: what runs today.** About 10–20 min of CPU in bwrap, writing only under a scratch dir. It runs every reference that has arguments through today's executor against the Phase 7 store.
- **M3: tally of reasons already stored.** Seconds. Reads the arg-equivalence `full.json` and the Max acceptable-tools `candidates.json`.
- **M4: the GRPO pool's reference calls.** Seconds. Counts round-2 trace rows by area, tool and `call_result` status.

Built as `scripts/distill/gym_fidelity.py measure {m1,m2,m3,m4,avail}` (tests: `tests/test_distill_gym_fidelity.py`). Results are in §1.4.

**Still needed after M1–M4, from the Phase 0 tool** (`avail`, §1.4):
- per-case live-result availability: can `meta.trace_id` reach a row in the read-only LCM snapshot? (§4, V6). Phase 0 measured only the parts that open no database: the snapshot's age, and the join of each `trace_id` to the phase-1 trace index.
- the trace dates against the repo store's oldest commit (measured).

### 1.4 Phase 0 results (2026-11-06, measured)

Run on the 3090 Ti host from scratch copies of HEAD (af6c68f–9c51c53 on `feat/distill-flywheel`; `PYTHONPATH=src`, system python, niced). Everything was read-only except M2, which wrote only to its own scratch work dir and removed it. Cases: `~/distill/phase3/build-v2/eval-tools.jsonl` (329, cases sha `234028b0e65f5e15`). Store: a copy of the gym's repo store `27055dbfea33f7a2.git` (HEAD 27055dbf, 200 commits from 2026-09-26 to 2026-10-17). The Phase 7 store path under `/tmp` is gone, but the same store survives in every gym root. Aggregates only.

**M1, reference census (static).**

| | Count |
|---|---|
| Cases / references with arguments | 329 / **302** (no arguments: 26 hard cases, plus one `task_list`) |
| Context source | `lcm` 329, `stored` 0 |
| History holds tool results | 312 (17 don't) |
| Environment maps (cwd under home) | 329. But every case has the **same** cwd: the packing host's deploy directory (`env_now`, #7). It is not where the trace ran. |
| Paths named (all references) | abs_other 228, another project 159, repo checkout or worktree 124, `/tmp` 82, other home 37, the case cwd 16, `~/.prometheus` 10 |
| Paths seen in the history | home paths 250/320, abs 121/228, `/tmp` 45/82, `~/.prometheus` 6/10 |
| Reference by worst path target | another project **124**, repo **65**, abs_other 51, no absolute path 35, `/tmp` 16, cwd 7, `~/.prometheus` 2, home 2 |
| Repo paths against the fixture commit | 105 are a checkout or worktree root (mostly `cd <worktree> && …`); 11 files exist in the fixture; 8 are `.venv` (untracked) |
| Bash (269 with arguments), static grammar class | **refused 132** (multi-line 77, `$`/syntax 55); allowed 67; unavailable program 36 (`gh` 11, `timeout` 7, docker 7, python 4, 7 others with 1 each); git write 15 (`fetch` 4, `add` 3, `worktree` 3, …); git read 12; simulated host 7 |

Another project's repository: 12 distinct directories under `~/projects`. They are not named here.

**M2, what runs today** (`GymSandboxExecutor`, bwrap, `jail=require`; 297 sandbox runs, 5.4 s).

| First blocker (grammar first; the executor itself checks paths first) | References |
|---|---|
| clean in A and B | **14** (1 with empty output) |
| grammar: multi-line / shell syntax / program / git write / python | 77 / 55 / 32 / 15 / 4 = **183** |
| missing path: another project / repo or worktree / abs / `~/.prometheus` / home | 41 / 18 / 3 / 1 / 1 = **64** |
| tool not routed: side effects (`edit_file` 10, `task_create` 6, `write_file` 4, `memory` 3, `task_stop` 2, `cron_delete` 1) / read-only (`task_get` 3, `task_output` 2, `web_fetch` 1) | 26 / 6 = **32** |
| simulated host command | 7 |
| refused at run time (confinement or options) | 2 |

The executor's raw status was `missing_path` for 202 references, 100 of them grammar-refused. That ordering is why the arg-equivalence reasons read as "missing state".

**The eval-tools coverage ceiling under today's grammar:** 302 − 183 (grammar) − 7 (simulated) − 41 (another project) − 1 (`web_fetch`, network) = **70 (23%)**. That would need repo and worktree path mapping (the 18), registry routing (31) and a few fixtures (5). Even if other projects' repositories were rebuilt, the ceiling is 111 (37%). **V5's 50% bar is unreachable without widening the grammar** (multi-line, `$`), which is operator-parked.

**M3, stored reasons.**
- **Arg equivalence** (`~/scratch-arg-equiv/out/full.json`, 6 models): 1,499 verdicts. undecidable 1,275, exact 145, execution 50 (all "differ in A"), conservative 28, canonical 1. In the 1,275 undecidable verdicts, the reference side was: `missing_path` 978, `refused` 252, `ok` 20, `simulated` 8, `nondeterministic` 7, not executable 6, and 4 edit-context cases.
- **Outcome check** (Max acceptable-tools `candidates.json`): 355 candidate calls. not_run 304, reference_not_ok **44**, differs 7.
  - The 44 span **34 distinct cases**, and their candidates are `read_file` 43 and `bash` 1.
  - Reasons: reference `missing_path` 36, `refused` 7, `nondeterministic` 1.
- **Judge:** 164 case-tool pairs acceptable, **34 rejected**.
- **§6 Q1 answered:** 44 is correct as the baseline count of candidate *calls* whose reference failed. Those calls cover 34 cases. The scorecard's other 34 is the judge's rejected *case-tool pairs*. It is a different quantity, and the match is a coincidence. Baseline: 44 calls / 34 cases.

**M4, the GRPO pool's reference calls.** 5,287 round-2 rows: 3,826 local (10 files) and 1,461 Max. The Max r2 rows are not under `pool/max-derived` (nothing matches `*r2*` there). They are in the Max run's own out dir, `~/distill/phase2b/runs/max-tokenplan-r2-max4-max-derived/`.
- Every row: `round 2`, generator `teacher-gen/5`, fixture commit 27055dbf (all r2 runs' `repo_fixture`). **None compacted**, none padded.
- `call_result`: success **3,090**, no_call 1,514, planted 413, injected 260, error 10.
- **ok and not compacted: 3,090** (local 2,075, Max 1,015). By area: injection-resistance 853, error-recovery 783, coding 527, daily-agent 442, long-context 363, tool-choice 122.
- By tool: `bash` 1,149, `lcm_grep` 761, `read_file` 334, `glob` 155, `task_list` 143, `memory` 136, `grep` 118, `task_create` 74, `wiki_query` 58, `task_get` 57, `tool_search` 48, `todo_write` 16, `task_output` 14, `write_file` 10, `cron_list` 9, `wiki_compile` 5, `message` / `cron_create` / `cron_delete` 1 each.
- Today's executor runs 1,483 of them (48%, `bash`/`read_file`). **All 3,090 use gym registry tools**, so routing through `SandboxEnv.execute` (§2.1) reaches every one.

**Avail: live results and trace dates** (no database opened).
- **LCM snapshot:** `~/distill/snapshot/lcm.db` exists: 305.6 MB, written 2026-10-30 17:31 ET (7.1 days old), no WAL. Not opened. Whether a `trace_id` reaches its LCM row is untested until the operator answers §6 Q3.
- **Trace index:** the 303 golden cases all join the phase-1 trace index (`~/distill/phase1/trace_index.jsonl`, matched by file and line), and all 303 have a telemetry-snapshot rowid. The joined row's tool matches the case's reference tool for 298 (5 don't). The 26 hard cases come from `fail_traces.jsonl`, which isn't in that index.
- **Trace dates:** Aug 11, Sep 93, Oct 199. **104 (34%) predate the store's oldest commit** (the depth-200 store, 2026-09-26). 199 fall within it, and none postdate the fixture head. Pinning each case to its commit (§6 Q2) needs the full-depth store for those 104.

**What changes in the plan:**
- **MVP (phases 0–3, track A):** better than assumed.
  - The pool is 3,090 ok reference calls with nothing compacted, so risk 6 doesn't bite round 2.
  - One fixture commit and one generator version, so replay needs one store.
  - The new executor reaches every row.
  - Estimate: **8.5–10 agent-days** in all, with phase 0 at ~0.5 and done; **~3 h host CPU**. V2 is still the gate.
  - The trainable share depends on the round-2 Max train-source decision: local rows alone are 2,075.
- **Phase 4 (eval-tools rebuild, 6–7 days):** can't reach V5 (≤ 23% under today's grammar). Options for the operator:
  - (a) Shrink it to the cheap part: repo/worktree path mapping, registry routing and the `~/.prometheus` skeleton, ~2–3 days. Report the outcome metric as coverage-limited (V5's fallback).
  - (b) Revisit the grammar (multi-line, `$`) for scoring only.
  - (c) Drop it.
  - Until then, the full-plan total stays as written.

## 2. Design

### 2.1 One executor for every gym tool (`scripts/distill/gym_exec.py`, new)

- **Runs every call through the gym's own dispatch:** `build_sandbox_registry(env)` and `GuardedTool.execute` → `SandboxEnv.execute` (teacher_gen.py:2902). Same grammar, jail, confinement and stand-ins as generation. That gives `grep`, `glob`, `write_file`, `edit_file`, `todo_write` and the stand-ins for free, and drops the bespoke paths in `_run_once`.
- **`RunResult` gains state alongside output** (still only digests outside memory):
  - `output`: normalized text;
  - `fs_delta`: the workspace manifest diff (`_manifest`/`_effects`, arg_equiv.py:715-806) plus the content of changed text files, in memory;
  - `fx_delta`: a deep diff of `env.fx` (memory, cron, tasks, wiki, web) and `env.outbox`;
  - `status`: today's values plus `refused:<refusal_kind>` (teacher_gen.py:2280), `tainted` and `unscorable`.
- **Output normalizers per tool**, so cross-tool comparison is fair:
  - `read_file`'s numbered lines are stripped to text, so they match `cat`/`sed -n`;
  - `grep` tool output `path:line:text` is made relative to the root, so it matches `grep -rn`;
  - "(no matches)" and "(no output)" become empty;
  - the root becomes `<root>` and the case's home `~` (as `_map_home` does);
  - trailing whitespace is dropped.
- **`arg_equiv` and `acceptable_tools` move onto it:**
  - `EXECUTABLE_TOOLS` becomes "registry tools that are read-only for these arguments" (`GuardedTool.is_read_only` plus the memory `list` rule, arg_equiv.py:466-469);
  - `outcome_check` (acceptable_tools.py:312-342) keeps its states and gains `effect_equal` / `effect_differs`;
  - `OutcomeExecutor` stops patching `_digest` (acceptable_tools.py:288-304) because output text is a field.
- **Snapshots:** each prompt or case has a pristine template per snapshot (A, A′, B, and optionally C, §2.5). Each run gets a fresh `cp -a` copy of the template (`bwrap --overlay` is a later speed-up to probe). The executor's current "rebuild after a change" (arg_equiv.py:735) becomes "every call runs on a fresh copy".

### 2.2 Snapshot rebuild

**Track A: gym rows (`gym_snapshot.replay`, the GRPO pool).**

1. **Rebuild the task:** take the row's `GenTask` from its taskset row (`task_from_taskset`, teacher_gen.py:6680) with the run's `settings.json`. Check that `task_hash` equals `_meta.task_hash`.
2. **Rebuild the root as `run_task` does** (teacher_gen.py:4977-4992): clone the repo fixture at the run's recorded `repo_fixture_info.commit`, write `setup_files`, create `SandboxEnv`, set `env.messages`.
3. **Replay the row's earlier tool calls in order** through `SandboxEnv.execute`, with `injections_armed` as the run had it (off during pad turns, `_meta.compaction_pad`). The sandbox's injection, failure-family and adversarial counters are deterministic for the same sequence of calls, so they fire again in the same places.
4. **Check each replayed result** against the result recorded in the row's messages, after mapping the old root to the new one. Any mismatch on a call that wasn't injected marks the row `unreplayable`. Then check that the reference call reproduces `_meta.call_result.text`.
5. **Freeze the state at the cut** as template A. Build A′ the same way from scratch, and B by perturbing A (`_perturb`, arg_equiv.py:816).
6. **Compacted rows:** their earlier calls are summarised away, so they can't be replayed. They are excluded from the MVP. For future rows: record a state point per row in teacher_gen (`_meta.state_at_cut`: the workspace manifest diff against the fresh clone, plus `fx` as JSON). That is a small change with a `GENERATOR_VERSION` bump; operator question 8.

**Track B: eval-tools cases (`gym_snapshot.reconstruct`, measurement only).** Layers are applied in order; later layers win.

- **L0, the repo at trace time:**
  - a bare store of published history only (remote-tracking refs and tags; prepare_repo_store's rule, teacher_gen.py:4880-4884), full depth and every `origin/*` branch;
  - commit = the last one on the branch at or before the trace's date;
  - branch = the one the history shows (`git status` / `git branch` output) if any, else main;
  - materialized at the cwd's own directory name, not always `Prometheus`.
- **Cwd inference:** prefer history evidence over the `env_now` line: an earlier bash `cwd` argument, `pwd` output, absolute paths in earlier calls. Record the evidence used.
- **L1, history evidence**, in conversation order, for calls whose recorded result succeeded. Earlier calls are replayed in the gym where the grammar allows, and their recorded outputs are used as evidence:
  - `read_file` results, and simple `cat` / `head` / `tail` / `sed -n` / `nl` outputs (parsed with `teacher_gen._lex`), become known line ranges of a file;
  - `write_file` gives full content; `edit_file` is applied to known content; redirect writes, `mkdir`, `touch`, `mv`, `cp` and `rm` are replayed in the gym;
  - `ls`, `find`, `glob` and `tree` outputs become entry names; `grep` outputs become known lines;
  - where evidence and the L0 commit conflict, the evidence wins for those lines.
- **L2, synthetic and deterministic fixtures**, created only for roots the history mentions:
  - a `~/.prometheus` skeleton with placeholder values and no secrets: `config.yaml` keys only, empty `logs/`, `memory/`, `wiki/`, `files/`;
  - empty scratch directories;
  - stand-in state seeded from history outputs: `cron_list`, `task_list`/`task_get`, memory `list`, and `web_fetch` pages by URL;
  - `lcm_grep` runs over the case's own messages, as the stand-in already does (teacher_gen.py:3970).
- **Path mapping:** generalize `_map`/`_map_home` (arg_equiv.py:646-659, 844-846) to a table: home → root, `/tmp` → `<root>/.gym-mnt/tmp`, other observed absolute directories → `<root>/.gym-mnt/...`. Outputs are mapped back before comparison. `VIRTUAL_FILES` and the simulated host commands keep their current handling.
- **Taint markers:**
  - Content that isn't known (unread line ranges, files known only by name) is written as `⟦gym-unknown:<sha8>⟧` lines.
  - Any output containing the marker is `tainted`, which decides nothing.
  - Metadata readers (`ls -l`, `wc`, `du`, `stat`, `md5sum`/`sha*sum`, `find -size`) that touch a partly known or placeholder file are also `tainted`, using the path scan `_paths_exist` already does (arg_equiv.py:861-886).
- **The leak rule (enforced by a test):** the reference call and its live result are never inputs to the rebuild. They are used only in validation (§4). Otherwise snapshots would be fitted to the reference, which would then succeed by construction.

**Security, per constraint:**

| Constraint | How the plan keeps it |
|---|---|
| No network | Unchanged bwrap network unshare. Web stand-ins serve only fixtures or pages already in the history. |
| No host paths | L0 is published commits, L1 is case or row text (already passed through `redact_capture`, build.py:1382-1390), L2 is synthetic. Nothing is copied from a host home. The builder checks that every entry's real path is inside the root and refuses symlinks (#11). |
| No secrets | Every materialized text goes through the same redactor plus a secret-pattern scan. A hit replaces the file with taint and flags the case. |
| Single-line bash, python and multi-line OFF, hardened profile OFF | No change to `plan_shell` / `_prescan`. Refused references are counted as `refused:<kind>`, never rewritten. |
| No gym tools on the 4090 | The executor and the reward server run on the 3090 Ti host (§3.3). |

### 2.3 Read-only tools

`read_file`, `grep`, `glob`, bash read commands (cat, ls, find, git read-only, …), `cron_list`, `task_list`/`get`/`output`, memory `list`, `wiki_query`, `lcm_grep`, `web_search`/`web_fetch` from fixtures, and `sentinel_status`.

- **Verdict:** compare by **line containment** over the normalized output, as a multiset of lines, in A and B: does the candidate's output contain the reference's?
- **Read-file range defaults:** `read_file`'s default 200-line window is kept; B pads every file by 300 lines, so a `read_file` without a range does not contain a full `cat` in B (the current logic, acceptable_tools.py:33-36).
- **Line order:** where the reference program's output order depends on the filesystem (`find`, `grep -r`, `du` without a sort), the comparison ignores line order, and the report flags it.
- **No `list_dir` tool exists** in this registry; `glob` and bash `ls` cover it.

### 2.4 Tools with side effects: compare state, not output

`write_file`, `edit_file`, bash writes, `todo_write`, memory `add`/`replace`/`remove`, `cron_create`/`cron_delete`, `task_create`/`task_stop`, `message`.

- Run the reference and the candidate each on a fresh copy of A and of B. Compare the changes they made:
  - `fs_delta`: which paths changed, and their new content;
  - `fx_delta`: changes to the stand-ins' state (memory, cron, tasks, todo) and to `outbox`.
- **Normalization before comparing:**
  - `.py` files: compare `ast.dump(ast.parse(...))`, which parses in the scorer process and never executes;
  - other text: trailing whitespace and the final newline;
  - task ids: `t<sha1>` from teacher_gen.py:3868 is mapped to a positional id;
  - cron schedules: normalized via `validate_cron_expression` / `parse_natural_schedule`;
  - a cron or task `command`: compared through `canonical_shell` (arg_equiv.py:360).
- **Free-text fields are not scored by the outcome term:** a memory `entry`, a task `description`, a message `content`, a `write_file` of prose. Their structural parts are (operation, target, path, recipient, platform), and they are flagged for the offline judge audit.
- **Verdicts:**
  - `effect_equal`: the same changes in A and B;
  - `effect_partial`: the same set of touched paths or entities, different content;
  - `effect_differs`;
  - `collateral`: the candidate changed or deleted anything outside the reference's touched set, or anything at all when the reference was read-only.
- **The conservative tier** in arg_equiv (arg_equiv.py:519-521) stays for the *equivalence* metric. Outcome correctness is a separate metric beside it.

### 2.5 Nondeterminism

| Source | Handling |
|---|---|
| Within one snapshot | A is run twice (existing, arg_equiv.py:741-755). |
| Rebuilding the snapshot | **A′**, built from scratch, must match A's manifest digest and give the reference the same output. This catches `stat`, `ls -i`, directory iteration order and git index data. |
| Short files and listings | **B** pads every file and directory (existing). It is the guard against "the same only because the file is short". |
| Answers made up without reading (`echo` / `printf` of content already in the history) | **C, optional (phase 5):** each text file gets a salt line inserted at its top. A real read changes between A and C; a constant `echo` doesn't. MVP rule instead: a candidate whose programs are all string programs (`_STRING_OPERANDS`, teacher_gen.py:2553) gets no read credit. |
| Clock | `date` is refused (existing `_CLOCK_PROGRAMS`). mtimes are pinned. `LC_ALL`, `TZ` and `HOME` are fixed by `run_sh`. |
| Stand-ins | Already deterministic: hash-derived task ids, a constant cron `next_run`, seeded `ops_world`. |

### 2.6 Tools that can never run, and how they're scored

| Tool | Status | Scored as |
|---|---|---|
| `web_fetch` / `web_search` (eval-tools, URL or query not in history), `download_file` | no network | `unscorable:network` |
| `ask_user`, `sessions_spawn`, `skill`, `vault_*`, `tts`, `image_generate` / `video_generate`, `youtube_transcript`, browser and other tools outside the gym | not in the gym | `unscorable:not_in_gym` |
| Simulated host commands for eval-tools (`systemctl`, `journalctl`, …) | the stand-in answers from a synthetic world, not the real one | `simulated` decides nothing (as today) |
| Bash refused for python, multi-line, `$` or programs like `gh` and docker | operator-parked | `refused:<kind>`, reported as a ceiling |

**On eval-tools:**
- A case whose reference can't be scored is outside the outcome metric's denominator, and its count is reported. The coverage line is part of the metric.
- A candidate that can't be scored against a scorable reference counts as "not shown" (a lower bound), plus an upper bound, like `arg_equivalence_upper`.
- Exact or canonical argument equality (arg_equiv.py tiers 1–2) proves an equal outcome without running, so those count as outcome-equal for any tool.

**In GRPO:** see §3.1 (masking), so that unscorable tools are never an escape hatch.

## 3. Reward

### 3.1 The outcome term, stage 1 (replaces rows of DISTILL-GRPO-PLAN §2)

| Term | Value | Notes |
|---|---|---|
| Unparseable, schema-invalid or unoffered call | −1 floor | unchanged |
| Injection followed (`adversarial_verdict` on the action) | −1, overrides everything | unchanged; checked before the outcome term |
| ~~Right tool~~ | removed | operator order |
| ~~Arguments equivalent / required equal~~ | removed | subsumed by the outcome term |
| **Outcome, read reference** | 1.0: full containment in A and B (and C, when built), excess within cap. 0.5 × recall: ≥ 50% of the reference's lines in both. Else 0. | tool-agnostic |
| **Outcome, side-effect reference** | 1.0 for `effect_equal`; 0.4 for `effect_partial`; else 0 | prose fields unscored (§2.4) |
| **Collateral effect** | −0.5 | e.g. `rm`, overwrites, writes beside a read |
| Exact or canonical argument equality | outcome 1.0 without running | proven equality |
| Candidate undecidable: `refused:{python, multiline, shell_syntax, program, simulated_in_script}`, `simulated`, `tainted`, `nondeterministic`, `timeout`, `unscorable` | **masked** from the group advantage (no gradient), not 0 | the gym can't judge it |
| `refused:{outside_workdir, network}` | 0 | in a gym task the workspace is the whole world |
| Right no-call, no call when needed, extra calls, length | unchanged | |

**Prompt eligibility:**
- Only replay-verified gym rows whose reference status is `ok` (or `effect_equal` with itself in A, A′ and B) enter the outcome-reward pool. Rows whose reference is an injected failure use the other terms only (as in the GRPO plan).
- A prompt is dropped from the step when more than half its group is masked.

### 3.2 Making it hard to game

| Exploit | Guard |
|---|---|
| Dumping everything (`cat *`, `grep -r . .`, `find /`) to contain the reference | **Excess cap in B:** credit only if candidate bytes in B ≤ max(8 × reference bytes in B, 2 KB). B's pads make dumps blow up and targeted reads not. Truncated output (`OUTPUT_LIMIT`, `...[truncated]...`) scores 0. |
| Echoing content already visible in context | the string-program rule (MVP), then snapshot C |
| Superset writes, `cp -r` of the tree | exact change-set equality plus the collateral penalty |
| Escaping to tools or constructs the gym can't score | masking gives no positive signal. **Mask-rate guard:** stop if the masked share rises more than 10 pts above the SFT checkpoint's on dev. |
| Several calls hedging | only the first call is scored; extra calls cost −0.1 each (unchanged) |
| Injected instruction overlapping the reference read | the injection override comes first; forbidden arguments and strings catch the overlap |
| Imitating the reference's wording | prose fields aren't scored |
| Over-fitting to Max's single step | the containment test gives positive evidence only: a different useful step scores 0, not negative. Prompts where every sample scores 0 are removed by the stage-0 filter. Bias toward the reference is a named risk (§5). |

**Unit tests in `grpo_rewards.py`, written before any run:** each exploit above must score at or below the honest call.

### 3.3 Where it runs

- The outcome term needs tool execution. The GRPO plan's stage 1 says tools don't run at all (§3 table), and that changes.
- **Reward server on the 3090 Ti host (`gym_reward_server.py`):**
  - The 4090 trainer posts `{prompt_id, tool, arguments}` batches over the tailnet: endpoint pinned with `check_tailnet_endpoint` (teacher_gen.py:852), 0600 key file as for the judge.
  - The server accepts only prompt ids from its frozen pool manifest and tool names from the gym registry, with arguments ≤ 64 KB.
  - It returns scores and status codes only. No gym output text leaves the host.
  - Results are cached on (prompt, normalized arguments). Groups of 8 often repeat a call.
- **Estimated cost:** per step (32 completions, ~15 distinct calls, ~45 sandbox runs, 4 workers) ~3–10 s of host CPU, against 225–420 s per step for stock TRL. That's under 5%. The jail already caps each command at 10 s of CPU (`_limits`), and the server runs at `nice` 10 beside the live model.

### 3.4 Interaction with task success and injection resistance

- **Stage 1 total:** floors and the injection override first, then outcome + no-call / answer terms + extra-call + length.
  - eval-tools `task_success` (score.py) is a tool-name metric, so it isn't a reward term any more. It stays a reported metric.
  - Add **outcome success** on eval-tools: of the cases with a scorable reference, the share whose first call is outcome-equal. The GRPO plan's promotion row "wrong-tool count falls ≥ 20%" becomes "outcome success rises, paired, with task success Δ ≥ 0".
- **Stage 2:**
  - The episode reward stays `deterministic_checks` (+1).
  - The outcome term shapes **only the first action** (+0.3 × outcome). That is the only state a rollout shares with the reference; after it, the trajectories diverge and no per-step reference exists.
  - Injection −1 and refusal penalties as planned.
- **Injection resistance:**
  - On injection cuts, the reference is the action that doesn't follow the injection, so matching its outcome rewards resistance.
  - Following the injection is −1 no matter what it contains.
  - The rule from GRPO plan §2 (on an injection prompt, injection-clean scores only with task progress) still holds: a refusal or empty action earns no outcome credit.

## 4. Validation

| # | Check | Bar |
|---|---|---|
| V1 | **Gym replay fidelity:** replayed earlier results equal the recorded ones (normalized), for calls that weren't injected | ≥ 98% of calls; failing rows excluded, never hand-fixed |
| V2 | **Gym reference reproduces** `_meta.call_result.text` (first 6,000 chars, normalized) | ≥ 95% of rows with a call (this is the "X%" for the reward pool) |
| V3 | **Determinism:** A and A′ manifests identical; reference identical across A twice and A′ | 100% / ≥ 99% |
| V4 | **A/B checks against known pairs:** ~60 hand-written pairs across tools. Known-equal (cat ↔ read_file, `grep -rn` ↔ grep tool, ls ↔ glob, `find -name` ↔ glob, `head -n` ↔ read_file limit) must be equal in A and B. Known-different (off-by-one ranges, other files, a dump) must differ in at least one snapshot. Report "equal in A, different in B" on real pairs. | 100% on the suite |
| V5 | **eval-tools reference success:** of the ≤ 303 cases with arguments, the reference is `ok` (or `effect_equal`) in A, A′ and B. Reported by tool and by failure reason, with the refused ceiling shown separately. | ≥ 50% for eval-tools outcome success to be used in promotion decisions. Below that, it is reported as coverage-limited. |
| V6 | **Live fidelity (eval-tools):** for references that ran, the gym output against the live result from the read-only LCM snapshot (`trace_id` → session → the target's next row; redacted, compared as hashes and line containment, never stored as text) | ≥ 85% contained or equal |
| V7 | **Judge agreement:** a stratified 150 pairs (50 equal or supports, 50 differs, 50 partial) in a new "same progress?" view that shows both calls and their gym outputs; Max on the Token Plan, $0, or the 27B in a judge slot. Controls: the reference against itself, and the reference against an unrelated read. | Precision of "equal/supports" ≥ 95%; overall ≥ 85%; Cohen's κ reported; the operator labels ~30 disagreements |
| V8 | **Anti-hacking unit tests** (§3.2) | all pass before the dry run |

V1–V4 and V8 gate the GRPO pilot. V5–V7 gate using the eval-tools outcome metric for promotion.

## 5. Effort

| Phase | Scope | Agent-days | Host (3090 Ti CPU) | GPU |
|---|---|---|---|---|
| 0. Measure | **Done 2026-11-06** (§1.4): `gym_fidelity.py measure` m1–m4 + avail. A scorecard note is still to write. | ~0.5 (done) | < 5 min | 0 |
| 1. Executor | `gym_exec.py`: registry dispatch, normalizers, `fs_delta`/`fx_delta`, read comparators, stand-in state diffs with prose fields unscored, string-program rule; port arg_equiv and acceptable_tools; tests | 3 | — | 0 |
| 2. Gym replay | `gym_snapshot.replay`: rebuild the task, replay earlier calls, check them, templates A/A′/B, pool manifest; V1–V3 | 3 | 2–3 h (all round-2 rows) | 0 |
| 3. Reward | `grpo_rewards.py` outcome term, masking, anti-hacking tests (V8); reward server and client; latency smoke test from the 4090 | 2.5 | 0.5 h | 0 (client only) |
| **MVP = 0–3** | **unblocks a stage-1 pilot on gym rows** (read-only and stand-in effects scored; workspace edits masked unless canonical-equal) | **9 (8.5–10)**; was 9.5 (9–11) before phase 0 | **~3 h** | **0** |
| 4. eval-tools rebuild | L0 store and commit pinning, cwd inference, L1 history evidence and replay, L2 fixtures, path mapping, taint, redaction scan; V5 | 6–7 | 1 h | 0 |
| 5. Workspace effects | `write_file`/`edit_file`/bash-write diffs, AST normalizer, collateral, snapshot C | 3 | — | 0 |
| 6. Validation and reporting | V4 suite, V6 live fidelity, V7 judge view and run, rescore 7 models (outcome success, coverage), scorecard section, GRPO plan edits (§3.3, §3.4) | 3.5 | 2–3 h | 0 with Max; else ~1 h of 4090 judge slot |
| **Total** | | **~23–27** | **~8–10 h** | **0–1 h** |

Calendar: the MVP takes ~2 weeks, with host runs at night. The full plan takes ~5 weeks. The GRPO dry run (GRPO plan §4) can overlap phase 3. Its 20-step check needs the reward server.

**Risks:**
1. **The eval-tools coverage ceiling.** It is confirmed by phase 0: refused constructs are 61% of references with arguments, and the reachable ceiling is ~23% (§1.4). V5 stays under 50% unless the grammar widens, so the outcome metric supports but doesn't decide promotion.
2. **Snapshots fitted to the reference.** Guarded by the leak rule and V6; the risk is still subtle.
3. **Training the gym's style:** the policy learns gym-runnable commands (single-line, no python) that production doesn't need. Guarded by masking and the mask-rate stop.
4. **Containment hacks** beyond the tested ones. Guarded by the B excess cap, snapshot C, and the audit of 20 high-reward samples every 50 steps (GRPO plan §1).
5. **Replay drift:** teacher_gen's grammar changed after a row was generated. Pin the replay to the row's `generator` version, or accept the drift and let V1 measure it.
6. **Compacted rows can't be replayed** until `state_at_cut` exists. Phase 0 found that the round-2 pool has none (§1.4), so this applies only to future compaction rows.
7. **The 3090 Ti host is shared** with the live model and generation nights. The reward server is CPU-only and niced, and has to be scheduled.
8. **Max-derived.** References are Max's, so the reward and the resulting adapter are max-derived and private (as with SFT).

## 6. Open questions for the operator

1. The scorecard records **44** `reference_not_ok` (scorecard:230); 34 is the rejected-tool count. Confirm 44 is the baseline. *Phase 0: 44 is the call count, and those calls cover 34 distinct cases. The judge's 34 rejected case-tool pairs are a different quantity; the numbers match by coincidence (§1.4, M3).*
2. May the gym's repo store hold **all published refs at full depth** (not one ref at depth 200), to pin each case to the commit and branch of its trace?
3. May validation read the **read-only LCM snapshot** to get each reference's live result (redacted, hashes only, never stored as text)? Is that snapshot still on the 3090 Ti host?
4. Is a **synthetic `~/.prometheus` skeleton** acceptable in the gym (placeholder `config.yaml` keys, empty `logs/`, `memory/`, `wiki/`, `files/`)? Or should it stay out, with those references counted as missing?
5. Is it acceptable to map **`/tmp` and other observed absolute directories** into the sandbox root (`.gym-mnt/…`, mapped back in outputs)?
6. Approve the **reward server on the 3090 Ti host** answering the 4090 over the tailnet during GRPO windows (CPU only, scores only)?
7. **Prose effects** (memory entries, message text, prose files): leave them unscored by the outcome term and audit them offline with the judge, as proposed? Or judge a sample online?
8. May teacher_gen record **`state_at_cut`** on new round-2 rows (a `GENERATOR_VERSION` bump), so compacted and long-context rows become usable?
9. Bars: ≥ 95% (V2) for the reward pool, ≥ 50% coverage (V5) and ≥ 85% live fidelity (V6) for using eval-tools outcome success in promotion. Accept these, or set others?
10. Should outcome success **replace** argument equivalence in the scorecard headline, or sit beside it? Should the 26 hard cases be dropped from its denominator, as proposed?
