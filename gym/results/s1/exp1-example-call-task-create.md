# Gym report — series `s1`, experiment `exp1-example-call-task-create`

> Hypothesis (D2): task_create failures are discriminator-omission plus JSON-stuffing, not nested schemas (all 49 builtin schemas are flat). A single concrete example showing `type: local_agent` with a plain-string prompt should cut both shapes. Control categories watch for regressions.


## Totals

- Runs: **66** — passed 57/66 (86%)
- Adapter repairs: 0 · malformed drops: 0 · breaker trips: 0 · loop feedback retries: 0
- ⚠️ harness-level errors (timeouts/crashes): 3

## Verdict

Baseline 82% → experiment 86% (**+5%** over 66+66 runs). Within noise for samples this size — treat as a wash unless per-category effects below are large and consistent.

## By category

| category | baseline | experiment | Δ |
|---|---|---|---|
| argshape | 8/9 (89%) | 8/9 (89%) | +0% |
| control | 21/24 (88%) | 22/24 (92%) | +4% |
| lcm | 3/3 (100%) ⚠️thin | 3/3 (100%) ⚠️thin | +0% |
| namespace | 6/6 (100%) | 6/6 (100%) | +0% |
| resilience | 12/12 (100%) | 12/12 (100%) | +0% |
| task_create | 4/12 (33%) | 6/12 (50%) | +17% |

## By task

| task | baseline | experiment |
|---|---|---|
| argshape_grep_two_params | 3/3 (100%) | 3/3 (100%) |
| argshape_task_list_status | 2/3 (67%) | 3/3 (100%) |
| argshape_write_json_content | 3/3 (100%) | 2/3 (67%) |
| control_bash_echo | 1/3 (33%) | 2/3 (67%) |
| control_edit_file | 3/3 (100%) | 3/3 (100%) |
| control_glob | 3/3 (100%) | 3/3 (100%) |
| control_grep | 2/3 (67%) | 3/3 (100%) |
| control_multi_step | 3/3 (100%) | 3/3 (100%) |
| control_no_arg_tool | 3/3 (100%) | 3/3 (100%) |
| control_read_file | 3/3 (100%) | 3/3 (100%) |
| control_write_file | 3/3 (100%) | 2/3 (67%) |
| lcm_query_punctuation | 3/3 (100%) | 3/3 (100%) |
| namespace_task_list_not_bash | 3/3 (100%) | 3/3 (100%) |
| namespace_tool_search | 3/3 (100%) | 3/3 (100%) |
| resilience_collapse_arc_replay | 3/3 (100%) | 3/3 (100%) |
| resilience_misnamed_tool | 3/3 (100%) | 3/3 (100%) |
| resilience_missing_file | 3/3 (100%) | 3/3 (100%) |
| resilience_nonexistent_tool | 3/3 (100%) | 3/3 (100%) |
| task_create_agent_explicit | 0/3 (0%) | 0/3 (0%) |
| task_create_agent_implicit | 0/3 (0%) | 0/3 (0%) |
| task_create_bash_explicit | 1/3 (33%) | 3/3 (100%) |
| task_create_watch | 3/3 (100%) | 3/3 (100%) |

_Sample bar: ≥30 runs/arm for a verdict; cells under 6 runs flagged thin. Deterministic predicate scoring; no LLM judging._