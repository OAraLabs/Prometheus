# Gym report — series `s1`, experiment `exp3-unwrap`

> Evidence-reshaped Phase 4: all 49 builtin schemas are flat; the model INVENTS nesting. Instead of lowering our schemas, absorb the phantom layer adapter-side — losslessly and only when validation proves the transform right. Every unwrap is telemetry (repairs) + a training pair. Code differs from exp0/exp1 only by inert-when-off gates (verified by tests).


## Totals

- Runs: **66** — passed 49/66 (74%)
- Adapter repairs: 10 · malformed drops: 0 · breaker trips: 0 · loop feedback retries: 0

## Verdict

Baseline 82% → experiment 74% (**-8%** over 66+66 runs). Within noise for samples this size — treat as a wash unless per-category effects below are large and consistent.

## By category

| category | baseline | experiment | Δ |
|---|---|---|---|
| argshape | 8/9 (89%) | 6/9 (67%) | -22% |
| control | 21/24 (88%) | 22/24 (92%) | +4% |
| lcm | 3/3 (100%) ⚠️thin | 3/3 (100%) ⚠️thin | +0% |
| namespace | 6/6 (100%) | 6/6 (100%) | +0% |
| resilience | 12/12 (100%) | 12/12 (100%) | +0% |
| task_create | 4/12 (33%) | 0/12 (0%) | -33% |

## By task

| task | baseline | experiment |
|---|---|---|
| argshape_grep_two_params | 3/3 (100%) | 3/3 (100%) |
| argshape_task_list_status | 2/3 (67%) | 0/3 (0%) |
| argshape_write_json_content | 3/3 (100%) | 3/3 (100%) |
| control_bash_echo | 1/3 (33%) | 1/3 (33%) |
| control_edit_file | 3/3 (100%) | 3/3 (100%) |
| control_glob | 3/3 (100%) | 3/3 (100%) |
| control_grep | 2/3 (67%) | 3/3 (100%) |
| control_multi_step | 3/3 (100%) | 3/3 (100%) |
| control_no_arg_tool | 3/3 (100%) | 3/3 (100%) |
| control_read_file | 3/3 (100%) | 3/3 (100%) |
| control_write_file | 3/3 (100%) | 3/3 (100%) |
| lcm_query_punctuation | 3/3 (100%) | 3/3 (100%) |
| namespace_task_list_not_bash | 3/3 (100%) | 3/3 (100%) |
| namespace_tool_search | 3/3 (100%) | 3/3 (100%) |
| resilience_collapse_arc_replay | 3/3 (100%) | 3/3 (100%) |
| resilience_misnamed_tool | 3/3 (100%) | 3/3 (100%) |
| resilience_missing_file | 3/3 (100%) | 3/3 (100%) |
| resilience_nonexistent_tool | 3/3 (100%) | 3/3 (100%) |
| task_create_agent_explicit | 0/3 (0%) | 0/3 (0%) |
| task_create_agent_implicit | 0/3 (0%) | 0/3 (0%) |
| task_create_bash_explicit | 1/3 (33%) | 0/3 (0%) |
| task_create_watch | 3/3 (100%) | 0/3 (0%) |

_Sample bar: ≥30 runs/arm for a verdict; cells under 6 runs flagged thin. Deterministic predicate scoring; no LLM judging._