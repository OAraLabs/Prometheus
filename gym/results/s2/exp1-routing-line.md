# Gym report — series `s2`, experiment `exp1-routing-line`

> Arm 1. The ONLY change vs s2-exp0 is the routing sentence appended to the system prompt (the rest is the v2 baseline prompt verbatim). Dual scoring shows whether it lifts EMISSION on task_create agent-mode (the model emits type=local_agent itself) vs merely shifting adapter load. Harvest OFF to keep exact condition-parity with the s2-exp0 baseline.


## Totals

- Runs: **72**
- Emission pass (raw model call): 51/72 (71%)
- Execution pass (post-adapter): 51/72 (71%)
- Adapter value (execution − emission): **+0%** (0 run(s) saved by repair/unwrap)
- Adapter repairs: 0 · malformed drops: 0 · breaker trips: 0 · loop feedback retries: 0

## Verdict (execution pass — the 'did it work' axis)

Baseline 79% → experiment 71% (**-8%** over 72+72 runs). Within noise for samples this size — treat as a wash unless per-category effects below are large and consistent.

## By category (emission → execution, Δ = adapter value)

| category | base emit | base exec | exp emit | exp exec | exp Δ |
|---|---|---|---|---|---|
| argshape | 11/15 (73%) | 11/15 (73%) | 7/15 (47%) | 7/15 (47%) | +0% |
| control | 21/24 (88%) | 21/24 (88%) | 22/24 (92%) | 22/24 (92%) | +0% |
| lcm | 3/3 (100%) ⚠️thin | 3/3 (100%) | 3/3 (100%) ⚠️thin | 3/3 (100%) | +0% |
| namespace | 6/6 (100%) | 6/6 (100%) | 5/6 (83%) | 5/6 (83%) | +0% |
| resilience | 12/12 (100%) | 12/12 (100%) | 12/12 (100%) | 12/12 (100%) | +0% |
| task_create | 4/12 (33%) | 4/12 (33%) | 2/12 (17%) | 2/12 (17%) | +0% |

## By task (emission → execution)

| task | base exec | exp emit | exp exec |
|---|---|---|---|
| argshape_browser_action | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| argshape_download_destination | 0/3 (0%) | 0/3 (0%) | 0/3 (0%) |
| argshape_grep_two_params | 2/3 (67%) | 0/3 (0%) | 0/3 (0%) |
| argshape_task_list_status | 3/3 (100%) | 1/3 (33%) | 1/3 (33%) |
| argshape_write_json_content | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| control_bash_echo | 1/3 (33%) | 2/3 (67%) | 2/3 (67%) |
| control_edit_file | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| control_glob | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| control_grep | 3/3 (100%) | 2/3 (67%) | 2/3 (67%) |
| control_multi_step | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| control_no_arg_tool | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| control_read_file | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| control_write_file | 2/3 (67%) | 3/3 (100%) | 3/3 (100%) |
| lcm_query_punctuation | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| namespace_task_list_not_bash | 3/3 (100%) | 2/3 (67%) | 2/3 (67%) |
| namespace_tool_search | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| resilience_collapse_arc_replay | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| resilience_misnamed_tool | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| resilience_missing_file | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| resilience_nonexistent_tool | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) |
| task_create_agent_explicit | 1/3 (33%) | 1/3 (33%) | 1/3 (33%) |
| task_create_agent_implicit | 0/3 (0%) | 0/3 (0%) | 0/3 (0%) |
| task_create_bash_explicit | 1/3 (33%) | 0/3 (0%) | 0/3 (0%) |
| task_create_watch | 2/3 (67%) | 1/3 (33%) | 1/3 (33%) |

_Sample bar: ≥30 runs/arm for a verdict; cells under 6 runs flagged thin. Dual deterministic scoring (emission vs execution); no LLM judging._